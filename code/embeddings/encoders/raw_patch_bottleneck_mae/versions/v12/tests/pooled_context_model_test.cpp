// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"
#include "pooled_context_old_model_reference.h"
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

void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &why) {
  check(a.scalar_type()==b.scalar_type() && a.sizes()==b.sizes() && torch::equal(a,b),why);
}
rpb::Settings settings(int source) {
  auto s=rpb::default_settings();s.model.device=torch::kCUDA;s.model.global_bottleneck_mode=2;
  s.model.channel_mixer_layers=1;s.model.channel_mixer_placement=1;s.model.global_pool_input_source=source;
  s.model.channel_ids={101,202,303};s.batch_size=3;s.steps=4;s.seed=531;s.threads=1;return s;
}
template<class A,class B> void same_model(A &a,B &b,const std::string &why) {
  const auto ap=a->named_parameters(),bp=b->named_parameters();check(ap.size()==bp.size(),why+" names");
  int64_t count=0;
  for(size_t i=0;i<ap.size();++i) {
    check(ap[i].key()==bp[i].key() && ap[i].value().is_cuda() && bp[i].value().is_cuda(),why+" CUDA names");
    exact(ap[i].value(),bp[i].value(),why+"/"+ap[i].key());count+=ap[i].value().numel();
  }
  check(count==225805,why+" parameter count");const auto ab=a->named_buffers(),bb=b->named_buffers();
  check(ab.size()==bb.size(),why+" buffers");
  for(size_t i=0;i<ab.size();++i) {check(ab[i].key()==bb[i].key(),why+" buffer name");exact(ab[i].value(),bb[i].value(),why+" buffer");}
}
template<class A,class B> void same_adam(A &a,B &b,torch::optim::AdamW &ao,torch::optim::AdamW &bo) {
  check(ao.state().size()==bo.state().size(),"literal old AdamW state count");
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  for(const auto &p:ap) {
    const auto x=ao.state().find(p.value().unsafeGetTensorImpl());
    const auto y=bo.state().find(bp[p.key()].unsafeGetTensorImpl());
    check((x==ao.state().end())==(y==bo.state().end()),"literal old AdamW presence");
    if(x==ao.state().end())continue;
    const auto *xs=dynamic_cast<const torch::optim::AdamWParamState *>(x->second.get());
    const auto *ys=dynamic_cast<const torch::optim::AdamWParamState *>(y->second.get());
    check(xs && ys && xs->step()==ys->step(),"literal old AdamW steps");
    exact(xs->exp_avg(),ys->exp_avg(),"literal old AdamW first moment");
    exact(xs->exp_avg_sq(),ys->exp_avg_sq(),"literal old AdamW second moment");
  }
}
void default_parity() {
  const auto s=settings(0);const auto text=rpb::settings_text(s);
  check(text.find("global_pool_input_source=")==std::string::npos,"default field omitted");
  check(rpb::settings_text(rpb::parse_settings(text+"global_pool_input_source=0\n"))==text,"explicit zero same settings identity");
  auto raw=input(s.model,3);auto x=rpb::fit_scaler(raw,s.model).transform(raw,s.model);
  torch::manual_seed(9201);rpb::Model actual(s.model);auto rng=torch::rand({8});
  torch::manual_seed(9201);rpb::PrePooledModel old(s.model);exact(torch::rand({8}),rng,"default literal constructor RNG");
  actual->eval();old->eval();same_model(actual,old,"literal pre-pooled initialization");
  {torch::NoGradGuard guard;const auto a=actual->encode(x),b=old->encode(x);
    exact(a.z_local,b.z_local,"literal old local");exact(a.z_contextual,b.z_contextual,"literal old contextual");
    exact(a.z_global,b.z_global,"literal old global");exact(a.z_contextual_global,b.z_contextual_global,"literal old served export");}
  torch::optim::AdamW ao(actual->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  torch::optim::AdamW bo(old->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for(int64_t step=0;step<3;++step) {
    const auto mask=rpb::make_training_mask(x.observed,s.model,531+step);ao.zero_grad();bo.zero_grad();
    auto a=actual->forward(x,mask.hidden),b=old->forward(x,mask.hidden);
    exact(a.encoding.z_contextual_global,b.encoding.z_contextual_global,"literal old masked export");
    exact(a.reconstruction,b.reconstruction,"literal old reconstruction");exact(a.loss,b.loss,"literal old objective");
    a.loss.backward();b.loss.backward();
    torch::nn::utils::clip_grad_norm_(actual->parameters(),1.,2.,true);
    torch::nn::utils::clip_grad_norm_(old->parameters(),1.,2.,true);ao.step();bo.step();
    same_model(actual,old,"literal old CUDA update");same_adam(actual,old,ao,bo);
  }
}
void pooled_semantics() {
  const auto s=settings(1);const auto compact=settings(0);auto raw=input(s.model,3);
  auto x=rpb::fit_scaler(raw,s.model).transform(raw,s.model);
  check(rpb::parse_settings(rpb::settings_text(s)).model.global_pool_input_source==1,"pooled settings roundtrip");
  for(int bad:{-1,2}) {auto c=s.model;c.global_pool_input_source=bad;rejects([&]{rpb::validate_config(c);},"closed pooled discriminator");}
  for(int change=0;change<4;++change) {auto c=s.model;
    if(change==0)c.global_bottleneck_mode=0;
    if(change==1)c.channel_mixer_layers=0;
    if(change==2)c.channel_mixer_placement=0;
    if(change==3)c.dropout=.1;
    rejects([&]{rpb::validate_config(c);},"unsupported pooled graph");}
  torch::manual_seed(9331);rpb::Model a(compact.model);
  torch::manual_seed(9331);rpb::Model b(s.model);a->eval();b->eval();
  int64_t common=0,registered=0;const auto ap=a->named_parameters(),bp=b->named_parameters();
  {torch::NoGradGuard guard;check(ap.size()==bp.size(),"same registered name set");
    for(const auto &p:bp) {registered+=p.value().numel();check(ap.contains(p.key()),"common parameter name");
      if(p.key()=="global_pool_first.weight") {check(p.value().sizes()==torch::IntArrayRef({64,195}) && ap[p.key()].sizes()==torch::IntArrayRef({64,99}),"only wider first weight");continue;}
      check(p.value().sizes()==ap[p.key()].sizes(),"shared shape");p.value().copy_(ap[p.key()]);common+=p.value().numel();}
    const auto ab=a->named_buffers(),bb=b->named_buffers();check(ab.size()==bb.size(),"common buffers");
    for(const auto &p:bb) {check(ab.contains(p.key()),"common buffer name");p.value().copy_(ab[p.key()]);}
    check(common==219469 && registered==231949,"declared shared and registered counts");
    const auto ca=a->encode(x),cb=b->encode(x);exact(ca.z_local,cb.z_local,"preserved local diagnostics after shared copy");
    exact(ca.z_contextual,cb.z_contextual,"preserved contextual diagnostics after shared copy");
    check(cb.z_contextual_global.sizes()==torch::IntArrayRef({3,32}),"native BD32 stays compact");
    const auto order=torch::tensor({2,0,1},torch::kInt64).to(torch::kCUDA);auto perm=x;
    perm.data=x.data.index_select(1,order);perm.observed=x.observed.index_select(1,order);perm.channel_ids=x.channel_ids.index_select(0,order);
    const auto shuffled=b->encode(perm);close(shuffled.z_contextual_global,cb.z_contextual_global,"configured-semantic pooled permutation",2e-5,2e-6);
    close(shuffled.z_contextual,cb.z_contextual.index_select(1,order),"configured-semantic local permutation",2e-5,2e-6);
    auto sparse=x;sparse.observed=x.observed.clone();sparse.observed.select(1,1).fill_(false);
    auto poisoned=sparse;poisoned.data=x.data.clone();poisoned.data.select(1,1).fill_(std::numeric_limits<float>::quiet_NaN());
    const auto legal=b->encode(sparse),hidden=b->encode(poisoned);exact(legal.z_contextual_global,hidden.z_contextual_global,"unobserved NaN isolation");
    check(!hidden.channel_valid_mask.select(1,1).any().item<bool>(),"unsupported channel remains unsupported");
    auto absent=poisoned;absent.observed=torch::zeros_like(x.observed);const auto empty=b->encode(absent);
    check(!empty.sample_valid_mask.any().item<bool>(),"all absent invalid");exact(empty.z_contextual_global,torch::zeros_like(empty.z_contextual_global),"biased global all-absent zero");
    auto zeros=empty.z_contextual_global.to(torch::kCPU).contiguous();const auto *z=zeros.data_ptr<float>();
    for(int64_t i=0;i<zeros.numel();++i)check(z[i]==0 && !std::signbit(z[i]),"positive absent zero");
    auto changed=x;changed.data=x.data.clone();changed.data.select(1,1).add_(9.);
    exact(b->encode(changed).z_local.select(1,0),cb.z_local.select(1,0),"independent local diagnostic");
  }
  const auto before_projection=bp["export_projection.weight"].detach().clone();
  const auto before_bias=bp["export_projection.bias"].detach().clone();
  const auto before_global=bp["global_pool_first.weight"].detach().clone();
  torch::optim::AdamW optimizer(b->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for(int64_t step=0;step<4;++step) {
    auto mask=rpb::make_training_mask(x.observed,s.model,663+step);optimizer.zero_grad();auto out=b->forward(x,mask.hidden);
    check(out.loss.is_cuda() && out.eligible_example_count==3,"actual pooled CUDA loss");finite(out.loss,"pooled loss");
    exact(out.reconstruction,b->decode(out.encoding.z_contextual_global,x.channel_ids),"decoder only exact served BD32");
    auto changed=x;changed.data=torch::where(mask.target,x.data+1000.,x.data);
    exact(b->forward(changed,mask.hidden).encoding.z_contextual_global,out.encoding.z_contextual_global,"Q target isolated from encoder");
    out.loss.backward();
    for(const std::string prefix:{"patch_projection","channel_mixer_block_0","block_0","pool_score","pool_positions","global_pool_first","global_pool_second","decoder_first","decoder_second"}) {
      double total=0;for(const auto &p:b->named_parameters())if(p.key().rfind(prefix,0)==0 && p.value().grad().defined()) {
        check(p.value().grad().is_cuda(),"CUDA pooled gradients");finite(p.value().grad(),prefix);total+=p.value().grad().abs().sum().item<double>();}
      check(total>0,"reconstruction reaches "+prefix);
    }
    check(!bp["export_projection.weight"].grad().defined() && !bp["export_projection.bias"].grad().defined(),"diagnostic projection absent gradients");
    torch::nn::utils::clip_grad_norm_(b->parameters(),1.,2.,true);optimizer.step();
    exact(bp["export_projection.weight"],before_projection,"inactive diagnostic weight unchanged");
    exact(bp["export_projection.bias"],before_bias,"inactive diagnostic bias unchanged");
    check(optimizer.state().count(bp["export_projection.weight"].unsafeGetTensorImpl())==0 &&
          optimizer.state().count(bp["export_projection.bias"].unsafeGetTensorImpl())==0,"no fabricated inactive AdamW states");
  }
  check(!torch::equal(bp["global_pool_first.weight"],before_global),"actual wider CUDA weight updates");
  b->zero_grad();auto differentiable=x;differentiable.data=x.data.detach().clone().set_requires_grad(true);
  auto mask=rpb::make_training_mask(x.observed,s.model,761);
  b->forward(differentiable,mask.hidden).encoding.z_contextual_global.sum().backward();
  check(differentiable.data.grad().masked_select(mask.target).abs().sum().item<double>()==0,"hidden input gradient exactzero");
  check(differentiable.data.grad().masked_select(mask.visible).abs().sum().item<double>()>0,"visible W-route gradient");
}
std::string config_identity(const std::string &text) {
  uint64_t h=14695981039346656037ULL;for(unsigned char c:text){h^=c;h*=1099511628211ULL;}h^=0;h*=1099511628211ULL;
  std::ostringstream out;out<<"rpb-config-fnv1a-v1-"<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return out.str();
}
void malformed(const fs::path &path,const rpb::Settings &s,int source,bool dtype=false,bool shape=false,bool semantics=true) {
  torch::serialize::OutputArchive a;auto text=[&](const std::string &key,const std::string &value){a.write(key,embedding::archive::text_tensor(value),true);};
  text("encoder_id",rpb::kEncoderId);a.write("format_version",torch::tensor(int64_t(1)),true);text("artifact_kind","rpb_training_checkpoint_v1");
  text("rng_policy","splitmix64-counter-rows-masks-torch-attempt-v1");const auto cfg=rpb::settings_text(s);
  text("settings",cfg);text("configuration_id",config_identity(cfg));
  if(source>=0) {auto tag=dtype?torch::tensor(double(source),torch::kFloat64):torch::tensor(int64_t(source));if(shape)tag=tag.reshape({1});a.write("global_pool_input_source",tag,true);}
  if(semantics)text("global_pool_input_semantics",rpb::global_pool_input_semantics(s.model));a.save_to(path.string());
}
void archive_contract() {
  const auto dir=fs::temp_directory_path()/("rpb-pooled-model-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(dir),"exclusive preserved artificial archive leaf");
  for(int source:{0,1}) {
    auto s=settings(source);auto raw=input(s.model,3);auto dataset=rpb::describe_dataset(raw,s.model,"unitless,unitless,unitless");
    rpb::Checkpoint cp;cp.settings=s;cp.model=rpb::Model(s.model);cp.scaler=rpb::fit_scaler(raw,s.model);
    cp.dataset_id=dataset.dataset_id;cp.schema_id=dataset.schema_id;cp.scaler_fit_dataset_id=dataset.dataset_id;
    cp.training_policy_id="rpb-training-context-deletion-015-v1";cp.model->eval();
    torch::optim::AdamW optimizer(cp.model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
    const auto path=dir/("source-"+std::to_string(source)+".pt");rpb::save_checkpoint(path.string(),cp,optimizer);
    torch::serialize::InputArchive a;a.load_from(path.string(),torch::kCPU);torch::Tensor tag,sem;
    check(a.try_read("global_pool_input_source",tag,true)==bool(source) && a.try_read("global_pool_input_semantics",sem,true)==bool(source),"conditional checkpoint fields");
    if(source)check(tag.scalar_type()==torch::kInt64 && tag.dim()==0 && tag.item<int64_t>()==1 && embedding::archive::tensor_text(sem)==rpb::global_pool_input_semantics(s.model),"typed source1 semantics");
    auto restored=rpb::load_checkpoint(path.string(),torch::kCUDA);restored.model->eval();
    {torch::NoGradGuard guard;auto x=cp.scaler.transform(raw,s.model);exact(cp.model->encode(x).z_contextual_global,restored.model->encode(x).z_contextual_global,"CUDA typed checkpoint roundtrip");}
    const auto input_path=dir/("raw-"+std::to_string(source)+".pt"),export_path=dir/("export-"+std::to_string(source)+".pt");
    rpb::save_dataset(input_path.string(),dataset);
    std::vector<std::string> args{"rpb-mae","embed","--input",input_path.string(),"--checkpoint",path.string(),"--output",export_path.string(),"--device","cuda"};
    std::vector<char*>argv;for(auto &arg:args)argv.push_back(arg.data());check(rpb::run_cli(int(argv.size()),argv.data())==0,"CUDA pooled export");
    torch::serialize::InputArchive exported;exported.load_from(export_path.string(),torch::kCPU);
    check(exported.try_read("global_pool_input_source",tag,true)==bool(source) && exported.try_read("global_pool_input_semantics",sem,true)==bool(source),"conditional export fields");
    const auto config_path=dir/("source-"+std::to_string(source)+".conf");{std::ofstream f(config_path);f<<rpb::settings_text(s);}
    const auto rejected=dir/("rejected-"+std::to_string(source)+".pt");
    args={"rpb-mae","train","--config",config_path.string(),"--checkpoint",rejected.string()};argv.clear();for(auto &arg:args)argv.push_back(arg.data());
    rejects([&]{rpb::run_cli(int(argv.size()),argv.data());},"ordinary early/pooled train");check(!fs::exists(rejected),"ordinary training creates no output");
    args={"rpb-mae","train","--resume",path.string(),"--device","cuda","--checkpoint",rejected.string()};argv.clear();for(auto &arg:args)argv.push_back(arg.data());
    rejects([&]{rpb::run_cli(int(argv.size()),argv.data());},"ordinary tagged pooled resume");check(!fs::exists(rejected),"ordinary resume creates no output");
  }
  int index=0;auto reject=[&](const rpb::Settings &s,int source,bool dtype=false,bool shape=false,bool semantics=true) {
    const auto path=dir/("malformed-"+std::to_string(index++)+".pt");malformed(path,s,source,dtype,shape,semantics);
    bool expected=false;try {rpb::load_checkpoint(path.string(),torch::kCUDA);}catch(const std::exception &e) {expected=std::string(e.what()).find("global pool input source/semantics")!=std::string::npos;}
    check(expected,"typed pool metadata rejects before absent model/weights");
  };
  reject(settings(1),-1);reject(settings(1),0);reject(settings(1),2);reject(settings(1),1,true);
  reject(settings(1),1,false,true);reject(settings(1),1,false,false,false);reject(settings(0),0);
  std::cout<<"Preserved pooled model engineering archives: "<<dir.string()<<'\n';
}
}
int main() try {
  check(torch::cuda::is_available(),"pooled model admission requires actual CUDA");torch::set_num_threads(1);
  default_parity();pooled_semantics();archive_contract();
  std::cout<<"Pooled context model CUDA admission passed\n"<<RPB_SOURCE_ID<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
