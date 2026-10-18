// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"
#include "rpb_test_support.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#ifndef RPB_SOURCE_ID
#define RPB_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace fs = std::filesystem;
using namespace rpb_test;

void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &label) {
  check(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() && torch::equal(a, b), label);
}
void same_parameters(rpb::Model &a, rpb::Model &b, const std::string &label) {
  const auto x = a->named_parameters(), y = b->named_parameters();
  check(x.size() == y.size(), label + " named parameter count");
  int64_t count = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    check(x[i].key() == y[i].key() && x[i].value().is_cuda() && y[i].value().is_cuda(), label + " ordered CUDA names");
    exact(x[i].value(), y[i].value(), label + "/" + x[i].key()); count += x[i].value().numel();
  }
  check(count == 225805, label + " exact production parameter count");
  const auto xb = a->named_buffers(), yb = b->named_buffers(); check(xb.size() == yb.size(), label + " buffers");
  for (size_t i = 0; i < xb.size(); ++i) {
    check(xb[i].key() == yb[i].key(), label + " ordered buffer names");
    exact(xb[i].value(), yb[i].value(), label + " buffer/" + xb[i].key());
  }
}
template<class Module> std::shared_ptr<Module> child(rpb::Model &model, const std::string &name) {
  auto result = std::dynamic_pointer_cast<Module>(model->named_children()[name]);
  check(bool(result), "literal reference module " + name); return result;
}

// Literal historical late ordering, using exactly the registered model weights.
// This path does not read the new placement selector, so default equality tests
// check the preserved computation rather than two calls to the same new branch.
rpb::EncodeOutput literal_late(rpb::Model &model, const rpb::Input &input) {
  const auto &c = model->config(); const auto p = rpb::tokenize_visible(input, c);
  const auto B = input.data.size(0), C = c.channel_count, K = c.history_length/c.patch_length, W = c.encoder_width;
  check(p.row_indices.numel() > 0, "literal fixture has active rows");
  auto positions = p.positions.clamp_min(0);
  auto h = child<torch::nn::LinearImpl>(model,"patch_projection")->forward(torch::cat({p.values,p.visibility.to(c.dtype)},-1)) +
      child<torch::nn::EmbeddingImpl>(model,"positions")->forward(positions) +
      child<torch::nn::EmbeddingImpl>(model,"channels")->forward(p.channel_indices).unsqueeze(1);
  h = torch::where(p.valid.unsqueeze(-1),h,torch::zeros_like(h));
  for (int64_t i=0;i<c.num_layers;++i) h=child<rpb::PreNormBlockImpl>(model,"block_"+std::to_string(i))->forward(h,p.valid);
  h=torch::where(p.valid.unsqueeze(-1),child<torch::nn::LayerNormImpl>(model,"final_norm")->forward(h),torch::zeros_like(h));
  auto pool=[&](const torch::Tensor &states) {
    auto scores=child<torch::nn::LinearImpl>(model,"pool_score")->forward(torch::tanh(states)).squeeze(-1)+
        child<torch::nn::EmbeddingImpl>(model,"pool_positions")->forward(positions).squeeze(-1);
    scores=scores.masked_fill(p.valid.logical_not(),-std::numeric_limits<float>::infinity());
    return child<torch::nn::LinearImpl>(model,"export_projection")->forward((states*torch::softmax(scores,1).unsqueeze(-1)).sum(1));
  };
  auto local=torch::zeros({B*C,c.export_width},h.options()).index_copy(0,p.row_indices,pool(h));
  const auto packed=torch::nonzero(p.valid.reshape({-1})).reshape({-1});
  const auto lookup=(p.row_indices.unsqueeze(1)*K+p.positions.clamp_min(0)).reshape({-1});
  const auto indices=lookup.index_select(0,packed);
  auto grid=torch::zeros({B*C*K,W},h.options()).index_copy(0,indices,h.reshape({-1,W}).index_select(0,packed));
  auto support=torch::zeros({B*C*K},p.valid.options()).index_fill(0,indices,true);
  auto groups=grid.reshape({B,C,K,W}).permute({0,2,1,3}).reshape({B*K,C,W});
  auto valid=support.reshape({B,C,K}).permute({0,2,1}).reshape({B*K,C});
  const auto active_groups=torch::nonzero(valid.any(1)).reshape({-1});
  auto active=groups.index_select(0,active_groups); const auto active_valid=valid.index_select(0,active_groups);
  active=child<rpb::PreNormBlockImpl>(model,"channel_mixer_block_0")->forward(active,active_valid);
  auto mixed=torch::zeros_like(groups).index_copy(0,active_groups,active).reshape({B,K,C,W}).permute({0,2,1,3}).reshape({B*C*K,W})
      .index_select(0,lookup).reshape_as(h);
  mixed=torch::where(p.valid.unsqueeze(-1),mixed,torch::zeros_like(h));
  auto contextual=torch::zeros_like(local).index_copy(0,p.row_indices,pool(mixed));
  rpb::EncodeOutput out; out.z_local=local.reshape({B,C,c.export_width});out.z_contextual=contextual.reshape({B,C,c.export_width});
  out.channel_valid_mask=p.patch_counts.gt(0);out.sample_valid_mask=out.channel_valid_mask.any(1);
  out.visible_patch_counts=p.patch_counts;out.visible_observation_counts=p.observation_counts;
  out.channel_ids=(input.channel_ids.dim()==1?input.channel_ids.unsqueeze(0).expand({B,C}):input.channel_ids).to(c.device);
  auto global=[&](const torch::Tensor &z) {
    const auto order=rpb::channel_indices(input.channel_ids,c,B,c.device).argsort(int64_t{1});
    const auto canonical=z.gather(1,order.unsqueeze(-1).expand({B,C,c.export_width}));
    const auto v=out.channel_valid_mask.gather(1,order);
    const auto values=torch::where(v.unsqueeze(-1),canonical,torch::zeros_like(canonical));
    const auto joined=torch::cat({values.flatten(1),v.to(c.dtype)},1);
    return torch::where(out.sample_valid_mask.unsqueeze(-1),child<torch::nn::LinearImpl>(model,"global_pool_second")->forward(
        torch::gelu(child<torch::nn::LinearImpl>(model,"global_pool_first")->forward(joined))),torch::zeros({B,c.export_width},h.options()));
  };
  out.z_global=global(out.z_local);out.z_contextual_global=global(out.z_contextual);return out;
}

rpb::Settings settings(int64_t placement) {
  auto s=rpb::default_settings();s.model.device=torch::kCUDA;s.model.global_bottleneck_mode=2;
  s.model.channel_mixer_layers=1;s.model.channel_mixer_placement=placement;
  s.model.channel_ids={101,202,303};s.steps=4;s.batch_size=3;s.seed=891;s.threads=1;return s;
}
void configuration_and_default() {
  const auto late=settings(0),early=settings(1);const auto old_text=rpb::settings_text(late);
  check(old_text.find("channel_mixer_placement=")==std::string::npos,"default placement omitted from canonical settings");
  check(rpb::settings_text(rpb::parse_settings(old_text+"channel_mixer_placement=0\n"))==old_text,"explicit zero unchanged canonical identity");
  check(rpb::parse_settings(rpb::settings_text(early)).model.channel_mixer_placement==1,"early setting roundtrip");
  for (int64_t bad:{-1,2}) { auto c=early.model;c.channel_mixer_placement=bad;rejects([&]{rpb::validate_config(c);},"invalid placement"); }
  auto c=early.model;c.dropout=.1;rejects([&]{rpb::validate_config(c);},"early dropout");
  c=early.model;c.channel_mixer_layers=0;rejects([&]{rpb::validate_config(c);},"early without mixer");
  c=early.model;c.global_bottleneck_mode=3;rejects([&]{rpb::validate_config(c);},"early alternate bottleneck");
  auto raw=input(late.model,3);const auto scale=rpb::fit_scaler(raw,late.model);auto x=scale.transform(raw,late.model);
  torch::manual_seed(4201);rpb::Model actual(late.model);const auto actual_rng=torch::rand({8});
  torch::manual_seed(4201);rpb::Model reference(late.model);exact(torch::rand({8}),actual_rng,"default initialization consumes identical RNG");
  actual->eval();reference->eval();same_parameters(actual,reference,"late initialized reference");
  {
    torch::NoGradGuard guard;const auto a=actual->encode(x),b=literal_late(reference,x);
    exact(a.z_local,b.z_local,"literal default local export");exact(a.z_contextual,b.z_contextual,"literal default contextual export");
    exact(a.z_contextual_global,b.z_contextual_global,"literal default learned global export");
  }
  torch::optim::AdamW ao(actual->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  torch::optim::AdamW bo(reference->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for (int64_t step=0;step<2;++step) {
    const auto mask=rpb::make_training_mask(x.observed,late.model,77+step);
    ao.zero_grad();bo.zero_grad();auto a=actual->forward(x,mask.hidden);
    auto visible=x;visible.observed=mask.visible;auto b=literal_late(reference,visible);
    auto prediction=reference->decode(b.z_contextual_global,x.channel_ids);
    auto loss=rpb::hierarchical_huber(prediction,x.data.detach(),mask.target,mask.eligible_channels,late.model.huber_delta).loss;
    exact(a.loss,loss,"literal default CUDA loss");a.loss.backward();loss.backward();
    torch::nn::utils::clip_grad_norm_(actual->parameters(),1.,2.,true);
    torch::nn::utils::clip_grad_norm_(reference->parameters(),1.,2.,true);ao.step();bo.step();
    same_parameters(actual,reference,"literal default CUDA AdamW update");
  }
}

void early_semantics() {
  const auto late=settings(0),early=settings(1);auto raw=input(early.model,3);
  auto x=rpb::fit_scaler(raw,early.model).transform(raw,early.model);
  torch::manual_seed(8821);rpb::Model a(late.model);const auto ar=torch::rand({8});
  torch::manual_seed(8821);rpb::Model b(early.model);exact(torch::rand({8}),ar,"placement independent initialization RNG");
  same_parameters(a,b,"all common initialized weights");a->eval();b->eval();
  {
    torch::NoGradGuard guard;const auto initial=b->encode(x),old=a->encode(x);exact(initial.z_local,old.z_local,"unmixed local preserved");
    auto changed=x;changed.data=x.data.clone();changed.data.select(1,1).add_(17.);
    const auto shifted=b->encode(changed);exact(shifted.z_local.select(1,0),initial.z_local.select(1,0),"local other channel isolation");
    check((shifted.z_contextual.select(1,0)-initial.z_contextual.select(1,0)).abs().max().item<double>()>1e-6,"early cross channel interaction");
    const auto order=torch::tensor({2,0,1},torch::kInt64).to(torch::kCUDA);auto perm=x;
    perm.data=x.data.index_select(1,order);perm.observed=x.observed.index_select(1,order);perm.channel_ids=x.channel_ids.index_select(0,order);
    const auto shuffled=b->encode(perm);close(shuffled.z_contextual,initial.z_contextual.index_select(1,order),"early semantic channel permutation",2e-5,2e-6);
    close(shuffled.z_contextual_global,initial.z_contextual_global,"early semantic global permutation",2e-5,2e-6);
    // Zero residual attention/MLP produces an exact identity mixer.
    for (auto *model:{&a,&b}) for (auto &parameter:(*model)->named_parameters())
      if(parameter.key().rfind("channel_mixer_block_",0)==0)parameter.value().zero_();
    const auto late_identity=a->encode(x),early_identity=b->encode(x);
    exact(late_identity.z_contextual,early_identity.z_contextual,"identity mixer contextual equivalence");
    exact(late_identity.z_contextual_global,early_identity.z_contextual_global,"identity mixer global equivalence");
  }
  // A fresh nonidentity model isolates the original patch grid from packed rank.
  torch::manual_seed(81);rpb::Model aligned(early.model);aligned->eval();
  {
    torch::NoGradGuard guard;for(auto &p:aligned->named_parameters())if(p.key().rfind("block_",0)==0)p.value().zero_();
    auto sparse=x;sparse.observed=torch::zeros_like(x.observed);
    sparse.observed.select(1,0).narrow(1,24,8).fill_(true);
    sparse.observed.select(1,1).narrow(1,0,8).fill_(true);sparse.observed.select(1,1).narrow(1,16,8).fill_(true);
    const auto before=aligned->encode(sparse);auto off=sparse;off.data=sparse.data.clone();off.data.select(1,1).narrow(1,0,24).add_(19.);
    exact(aligned->encode(off).z_contextual.select(1,0),before.z_contextual.select(1,0),"early original patch alignment with unequal packing");
    sparse.observed.select(1,1).narrow(1,24,8).fill_(true);const auto overlap=aligned->encode(sparse);
    auto on=sparse;on.data=sparse.data.clone();on.data.select(1,1).narrow(1,24,8).add_(19.);
    check((aligned->encode(on).z_contextual.select(1,0)-overlap.z_contextual.select(1,0)).abs().max().item<double>()>1e-6,"early aligned patch communicates");
    auto poisoned=sparse;poisoned.data=sparse.data.masked_fill(sparse.observed.logical_not(),std::numeric_limits<float>::quiet_NaN());
    exact(aligned->encode(poisoned).z_contextual,overlap.z_contextual,"early natural absence ignores poisoned storage");
    auto absent=poisoned;absent.observed=torch::zeros_like(sparse.observed);const auto empty=aligned->encode(absent);
    check(!empty.sample_valid_mask.any().item<bool>(),"early all missing support");exact(empty.z_contextual_global,torch::zeros_like(empty.z_contextual_global),"early all missing exactzero");
  }
  torch::manual_seed(9901);rpb::Model trained(early.model);trained->eval();const auto mask=rpb::make_training_mask(x.observed,early.model,19);
  auto out=trained->forward(x,mask.hidden);check(out.loss.is_cuda() && out.eligible_example_count==3,"actual CUDA early objective");
  exact(out.reconstruction,trained->decode(out.encoding.z_contextual_global,x.channel_ids),"decoder exact served global32");
  auto hidden_change=x;hidden_change.data=torch::where(mask.target,x.data+1000.,x.data);
  exact(trained->forward(hidden_change,mask.hidden).encoding.z_contextual_global,out.encoding.z_contextual_global,"early target signal cannot reach encoder");
  out.loss.backward();
  for(const std::string prefix:{"patch_projection","channel_mixer_block_0","block_0","pool_score","export_projection","global_pool_first","decoder_first"}) {
    double total=0;for(const auto &p:trained->named_parameters())if(p.key().rfind(prefix,0)==0 && p.value().grad().defined()) {
      check(p.value().grad().is_cuda(),"early gradients CUDA");finite(p.value().grad(),prefix);total+=p.value().grad().abs().sum().item<double>();
    } check(total>0,"early reconstruction reaches "+prefix);
  }
  trained->zero_grad();auto differentiable=x;differentiable.data=x.data.detach().clone().set_requires_grad(true);
  trained->forward(differentiable,mask.hidden).encoding.z_contextual_global.sum().backward();
  check(differentiable.data.grad().masked_select(mask.target).abs().sum().item<double>()==0,"early hidden input gradient exactzero");
  check(differentiable.data.grad().masked_select(mask.visible).abs().sum().item<double>()>0,"early visible input gradient");
  trained->zero_grad();differentiable.data=x.data.detach().clone().set_requires_grad(true);
  trained->encode(differentiable).z_local.select(1,0).sum().backward();
  check(differentiable.data.grad().select(1,1).abs().sum().item<double>()==0 &&
        differentiable.data.grad().select(1,0).abs().sum().item<double>()>0,"local autograd independently preserved");
}

struct TemporaryFiles {
  fs::path directory=fs::temp_directory_path()/("rpb-early-model-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  TemporaryFiles(){check(fs::create_directory(directory),"exclusive engineering fixture directory");}
  ~TemporaryFiles(){std::error_code ec;fs::remove_all(directory,ec);}
};
std::string config_identity(const std::string &text) {
  uint64_t hash=14695981039346656037ULL;for(unsigned char byte:text){hash^=byte;hash*=1099511628211ULL;}hash^=0;hash*=1099511628211ULL;
  std::ostringstream out;out<<"rpb-config-fnv1a-v1-"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}
void malformed_prefix(const fs::path &path,const rpb::Settings &s,int placement,const std::string &architecture,
                      bool wrong_dtype=false,bool wrong_shape=false,bool wrong_output=false,bool wrong_reconstruction=false) {
  torch::serialize::OutputArchive a;auto text=[&](const std::string &key,const std::string &value){a.write(key,embedding::archive::text_tensor(value),true);};
  text("encoder_id",rpb::kEncoderId);a.write("format_version",torch::tensor(int64_t(1)),true);text("artifact_kind","rpb_training_checkpoint_v1");
  text("rng_policy","splitmix64-counter-rows-masks-torch-attempt-v1");const auto settings_text=rpb::settings_text(s);
  text("settings",settings_text);text("configuration_id",config_identity(settings_text));
  if(placement>=0) {
    auto tag=wrong_dtype?torch::tensor(double(placement),torch::kFloat64):torch::tensor(int64_t(placement));
    if(wrong_shape) tag=tag.reshape({1});
    a.write("channel_mixer_placement",tag,true);
  }
  if(!architecture.empty())text("architecture_id",architecture);
  text("output_semantics",wrong_output?"forged-late-output":rpb::output_semantics(s.model));
  a.write("global_bottleneck_mode",torch::tensor(int64_t(2)),true);
  text("reconstruction_export_semantics",wrong_reconstruction?"forged-late-reconstruction":rpb::reconstruction_output_semantics(s.model));
  a.save_to(path.string());
}
void rejects_metadata(const fs::path &path,const std::string &message) {
  bool expected=false;try{rpb::load_checkpoint(path.string(),torch::kCUDA);}catch(const std::exception &error){expected=std::string(error.what()).find(message)!=std::string::npos;}
  check(expected,"actual metadata rejected before missing model/weights: "+message);
}
void archive_contract() {
  TemporaryFiles files;const auto early=settings(1),late=settings(0);auto raw=input(early.model,3);
  const auto configuration=files.directory/"early.conf", ordinary_output=files.directory/"ordinary-early.pt";
  { std::ofstream config(configuration);check(bool(config),"artificial early config");config<<rpb::settings_text(early); }
  auto ordinary_reject=[&](std::vector<std::string> arguments) {
    std::vector<char*> argv;for(auto &arg:arguments)argv.push_back(arg.data());
    bool rejected=false;try{rpb::run_cli(int(argv.size()),argv.data());}catch(const std::exception &error) {
      rejected=std::string(error.what()).find("ordinary train/resume rejects early mixer placement")!=std::string::npos;
    }
    check(rejected && !fs::exists(ordinary_output),"ordinary early training/resume rejected before output generation");
  };
  ordinary_reject({"rpb-mae","train","--config",configuration.string(),"--checkpoint",ordinary_output.string()});
  for(int placement:{0,1}) {
    auto s=placement?early:late;auto dataset=rpb::describe_dataset(raw,s.model,"unitless,unitless,unitless");
    rpb::Checkpoint cp;cp.settings=s;cp.model=rpb::Model(s.model);cp.scaler=rpb::fit_scaler(raw,s.model);
    cp.dataset_id=dataset.dataset_id;cp.schema_id=dataset.schema_id;cp.scaler_fit_dataset_id=dataset.dataset_id;
    cp.training_policy_id="rpb-training-context-deletion-015-v1";cp.model->eval();
    torch::optim::AdamW optimizer(cp.model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
    const auto path=files.directory/("placement-"+std::to_string(placement)+".pt");rpb::save_checkpoint(path.string(),cp,optimizer);
    torch::serialize::InputArchive archive;archive.load_from(path.string(),torch::kCPU);torch::Tensor p,id;
    check(archive.try_read("channel_mixer_placement",p,true)==bool(placement) && archive.try_read("architecture_id",id,true)==bool(placement),"only early CP fields are present");
    if(placement)check(p.scalar_type()==torch::kInt64 && p.dim()==0 && p.item<int64_t>()==1 && embedding::archive::tensor_text(id)==rpb::kEarlyMixerArchitectureId,"typed early CP architecture");
    auto restored=rpb::load_checkpoint(path.string(),torch::kCUDA);restored.model->eval();same_parameters(cp.model,restored.model,"CUDA checkpoint roundtrip");
    if(placement)ordinary_reject({"rpb-mae","train","--resume",path.string(),"--device","cuda","--checkpoint",ordinary_output.string()});
    auto x=cp.scaler.transform(raw,s.model);torch::NoGradGuard guard;
    exact(cp.model->encode(x).z_contextual_global,restored.model->encode(x).z_contextual_global,"exact restored CUDA global export");
    const auto data_path=files.directory/("raw-"+std::to_string(placement)+".pt"),export_path=files.directory/("export-"+std::to_string(placement)+".pt");
    rpb::save_dataset(data_path.string(),dataset);
    std::vector<std::string> arguments{"rpb-mae","embed","--input",data_path.string(),"--checkpoint",path.string(),"--output",export_path.string(),"--device","cuda"};
    std::vector<char*>argv;for(auto &arg:arguments)argv.push_back(arg.data());check(rpb::run_cli(int(argv.size()),argv.data())==0,"CUDA embedding export");
    torch::serialize::InputArchive exported;exported.load_from(export_path.string(),torch::kCPU);
    check(exported.try_read("channel_mixer_placement",p,true)==bool(placement) && exported.try_read("architecture_id",id,true)==bool(placement),"only early feature export fields present");
    if(placement)check(p.item<int64_t>()==1 && embedding::archive::tensor_text(id)==rpb::kEarlyMixerArchitectureId,"early feature export typed architecture");
  }
  int index=0;auto reject=[&](const rpb::Settings &s,int p,const std::string &id,bool dtype=false,bool shape=false,bool output=false,bool reconstruction=false) {
    const auto path=files.directory/("malformed-"+std::to_string(index++)+".pt");malformed_prefix(path,s,p,id,dtype,shape,output,reconstruction);
    rejects_metadata(path,output?"output semantics":reconstruction?"global bottleneck metadata":"placement/architecture metadata");
  };
  reject(early,-1,rpb::kEarlyMixerArchitectureId);reject(early,1,"");reject(early,0,rpb::kEarlyMixerArchitectureId);
  reject(early,1,rpb::kLateMixerArchitectureId);reject(early,1,rpb::kEarlyMixerArchitectureId,true);
  reject(early,1,rpb::kEarlyMixerArchitectureId,false,true);reject(late,0,"");reject(late,-1,rpb::kEarlyMixerArchitectureId);
  reject(early,1,rpb::kEarlyMixerArchitectureId,false,false,true);reject(early,1,rpb::kEarlyMixerArchitectureId,false,false,false,true);
}
} // namespace

int main() try {
  check(torch::cuda::is_available(),"early mixer admission requires actual CUDA, no CPU fallback");torch::set_num_threads(1);
  configuration_and_default();early_semantics();archive_contract();
  std::cout<<"Early mixer model CUDA admission passed\n"<<RPB_SOURCE_ID<<'\n';return 0;
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
