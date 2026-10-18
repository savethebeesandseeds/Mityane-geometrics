// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include "embedding/shared/data.h"
#include "pooled_context_old_model_reference.h"
#include "rpb_test_support.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#ifndef RPB_SOURCE_ID
#define RPB_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace fs = std::filesystem;
using namespace rpb_test;

void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &why) {
  check(a.scalar_type()==b.scalar_type() && a.sizes()==b.sizes() && torch::equal(a,b),why);
}
rpb::Settings settings(int mode) {
  auto s=rpb::default_settings();s.model.device=torch::kCUDA;s.model.channel_mixer_layers=1;
  s.model.channel_mixer_placement=1;s.model.global_bottleneck_mode=2;s.model.temporal_difference_input=mode;
  s.model.channel_ids={101,202,303};s.batch_size=3;s.steps=4;s.seed=631;s.threads=1;return s;
}
template<class A,class B> void common(A &a,B &b,bool appended=false) {
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  check(bp.size()==ap.size()+size_t(appended),"exact common parameter prefix length");int64_t count=0;
  for(size_t i=0;i<ap.size();++i) {
    check(ap[i].key()==bp[i].key() && ap[i].value().is_cuda() && bp[i].value().is_cuda(),"common CUDA registration order");
    exact(ap[i].value(),bp[i].value(),"common parameter "+ap[i].key());count+=ap[i].value().numel();
  }
  check(count==225805,"common225805 parameter values");
  if(appended)check(bp[ap.size()].key()=="visible_difference_projection.weight" && bp[ap.size()].value().numel()==3072,
      "last child adds only3072 bias-free weights");
  const auto ab=a->named_buffers(),bb=b->named_buffers();check(ab.size()==bb.size(),"common buffer count");
  for(size_t i=0;i<ab.size();++i) {check(ab[i].key()==bb[i].key(),"common buffer order");exact(ab[i].value(),bb[i].value(),"common buffer");}
}
template<class A,class B> void same_adam(A &a,B &b,torch::optim::AdamW &ao,torch::optim::AdamW &bo) {
  check(ao.state().size()==bo.state().size(),"literal AdamW state count");
  const auto ap=a->named_parameters(),bp=b->named_parameters();
  for(const auto &p:ap) {
    const auto x=ao.state().find(p.value().unsafeGetTensorImpl()),y=bo.state().find(bp[p.key()].unsafeGetTensorImpl());
    check((x==ao.state().end())==(y==bo.state().end()),"literal AdamW presence");if(x==ao.state().end())continue;
    const auto *xs=dynamic_cast<const torch::optim::AdamWParamState *>(x->second.get());
    const auto *ys=dynamic_cast<const torch::optim::AdamWParamState *>(y->second.get());
    check(xs && ys && xs->step()==ys->step(),"literal AdamW step");
    exact(xs->exp_avg(),ys->exp_avg(),"literal AdamW first moment");exact(xs->exp_avg_sq(),ys->exp_avg_sq(),"literal AdamW second moment");
  }
}
void default_parity() {
  const auto s=settings(0);const auto text=rpb::settings_text(s);
  check(text.find("temporal_difference_input=")==std::string::npos,"legacy canonical flag omitted");
  check(rpb::settings_text(rpb::parse_settings(text+"temporal_difference_input=0\n"))==text,"explicit zero canonical identity");
  auto raw=input(s.model,3);auto x=rpb::fit_scaler(raw,s.model).transform(raw,s.model);
  torch::manual_seed(9301);rpb::Model actual(s.model);auto cpu_rng=torch::rand({8});auto cuda_rng=torch::rand({8},torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(9301);rpb::PrePooledModel old(s.model);
  exact(torch::rand({8}),cpu_rng,"default literal constructor CPU RNG");exact(torch::rand({8},torch::TensorOptions().device(torch::kCUDA)),cuda_rng,"default literal constructor CUDA RNG");
  actual->eval();old->eval();common(actual,old);
  check(!actual->named_children().contains("visible_difference_projection"),"disabled child absent");
  torch::optim::AdamW ao(actual->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  torch::optim::AdamW bo(old->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for(int64_t step=0;step<3;++step) {
    auto mask=rpb::make_training_mask(x.observed,s.model,631+step);ao.zero_grad();bo.zero_grad();
    auto a=actual->forward(x,mask.hidden),b=old->forward(x,mask.hidden);
    exact(a.encoding.z_contextual_global,b.encoding.z_contextual_global,"literal disabled native32");
    exact(a.reconstruction,b.reconstruction,"literal disabled reconstruction");exact(a.loss,b.loss,"literal disabled Huber");
    a.loss.backward();b.loss.backward();torch::nn::utils::clip_grad_norm_(actual->parameters(),1.,2.,true);
    torch::nn::utils::clip_grad_norm_(old->parameters(),1.,2.,true);ao.step();bo.step();common(actual,old);same_adam(actual,old,ao,bo);
  }
}
void manual_differences() {
  auto c=settings(1).model;auto x=input(c,1);x.data=torch::arange(288,torch::kFloat32).reshape({1,3,32,3}).to(torch::kCUDA);
  x.observed=torch::ones_like(x.data,torch::kBool);x.observed[0][0][7][1]=false;
  x.observed[0][0].narrow(0,8,8).fill_(false);x.observed[0][1].narrow(0,0,8).fill_(false);
  x.observed[0][1].narrow(0,16,16).fill_(false);x.observed.select(1,2).fill_(false);
  auto poisoned=x;poisoned.data=torch::where(x.observed,x.data,torch::full_like(x.data,std::numeric_limits<float>::quiet_NaN()));
  const auto d=rpb::visible_first_differences(poisoned,c);
  const auto p=rpb::tokenize_visible(poisoned,c);
  const auto packed=rpb::pack_visible_differences(d,p,c);auto dv=d.values.to(torch::kCPU),dm=d.visibility.to(torch::kCPU);
  auto mask=x.observed.to(torch::kCPU);auto m=mask.accessor<bool,4>();auto v=dv.accessor<float,4>();auto ok=dm.accessor<bool,4>();
  for(int64_t ch=0;ch<3;++ch)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f) {
    const bool legal=t>0 && m[0][ch][t][f] && m[0][ch][t-1][f];
    check(ok[0][ch][t][f]==legal && v[0][ch][t][f]==(legal?3.f:0.f),"manual original adjacency/support/hidden NaN isolation");
  }
  check(!dm[0][0][16].any().item<bool>(),"packed ranks do not bridge missing original patch");
  check(dm[0][0][24].all().item<bool>(),"visible adjacent original patch boundary allowed");
  auto pv=packed.values.to(torch::kCPU),pm=packed.visibility.to(torch::kCPU),positions=p.positions.to(torch::kCPU),rows=p.row_indices.to(torch::kCPU);
  for(int64_t r=0;r<rows.numel();++r)for(int64_t k=0;k<positions.size(1);++k)for(int64_t j=0;j<24;++j) {
    const auto pos=positions[r][k].item<int64_t>(),ch=rows[r].item<int64_t>();
    const bool legal=pos>=0 && ok[0][ch][pos*8+j/3][j%3];
    check(pm[r][k][j].item<bool>()==legal && pv[r][k][j].item<float>()==(legal?3.f:0.f),"exact same packed positions/padding");
  }
  auto all_absent=poisoned;all_absent.observed=torch::zeros_like(x.observed);
  auto empty=rpb::pack_visible_differences(rpb::visible_first_differences(all_absent,c),rpb::tokenize_visible(all_absent,c),c);
  check(empty.values.numel()==0 && empty.visibility.numel()==0,"all-absent packed difference empty");
  auto overflow=x;overflow.data=x.data.clone();overflow.data[0][0][0][0]=-std::numeric_limits<float>::max();
  overflow.data[0][0][1][0]=std::numeric_limits<float>::max();
  rejects([&]{rpb::visible_first_differences(overflow,c);},"supported difference overflow");
}
void candidate_graph() {
  const auto control=settings(0),candidate=settings(1);auto raw=input(candidate.model,3);
  const auto scaler=rpb::fit_scaler(raw,control.model);check(scaler.identity()==rpb::fit_scaler(raw,candidate.model).identity(),"same original TRAIN scaler");
  auto x=scaler.transform(raw,candidate.model);
  torch::manual_seed(9331);rpb::Model a(control.model);auto cpu_rng=torch::rand({8});auto cuda_rng=torch::rand({8},torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(9331);rpb::Model b(candidate.model);
  exact(torch::rand({8}),cpu_rng,"zero child preserves CPU RNG");exact(torch::rand({8},torch::TensorOptions().device(torch::kCUDA)),cuda_rng,"zero child preserves CUDA RNG");
  a->eval();b->eval();common(a,b,true);auto weight=b->named_parameters()["visible_difference_projection.weight"];
  exact(weight,torch::zeros_like(weight),"new last child exact zero initialization");
  {torch::NoGradGuard guard;const auto ac=a->encode(x),bc=b->encode(x);
    exact(ac.z_local,bc.z_local,"initial local parity");exact(ac.z_contextual,bc.z_contextual,"initial contextual parity");
    exact(ac.z_contextual_global,bc.z_contextual_global,"initial served32 parity");
    exact(ac.channel_valid_mask,bc.channel_valid_mask,"initial support parity");}
  const auto mask=rpb::make_training_mask(x.observed,candidate.model,881);
  a->zero_grad();b->zero_grad();auto ao=a->forward(x,mask.hidden),bo=b->forward(x,mask.hidden);
  exact(ao.loss,bo.loss,"initial original-query Huber parity");exact(ao.reconstruction,bo.reconstruction,"initial original-query output parity");
  exact(ao.target_counts,bo.target_counts,"initial query denominators parity");ao.loss.backward();bo.loss.backward();
  for(const auto &p:a->named_parameters())if(p.value().grad().defined())exact(p.value().grad(),b->named_parameters()[p.key()].grad(),"common preclip gradient parity");
  check(weight.grad().is_cuda() && weight.grad().abs().sum().item<double>()>0,"zero branch receives native32 reconstruction gradient");finite(weight.grad(),"difference weight gradient");
  torch::optim::AdamW optimizer(b->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for(int64_t step=0;step<4;++step) {
    const auto q=rpb::make_training_mask(x.observed,candidate.model,883+step);optimizer.zero_grad();auto out=b->forward(x,q.hidden);
    check(out.loss.is_cuda() && out.encoding.z_contextual_global.sizes()==torch::IntArrayRef({3,32}),"CUDA exact native32 training path");
    exact(out.reconstruction,b->decode(out.encoding.z_contextual_global,x.channel_ids),"decoder takes exact served32 only");
    auto changed=x;changed.data=torch::where(q.target,x.data+1000.,x.data);
    exact(b->forward(changed,q.hidden).encoding.z_contextual_global,out.encoding.z_contextual_global,"original-Q values cannot enter delta branch");
    out.loss.backward();finite(weight.grad(),"four-step CUDA difference gradient");torch::nn::utils::clip_grad_norm_(b->parameters(),1.,2.,true);optimizer.step();
  }
  check(weight.abs().sum().item<double>()>0,"actual CUDA difference weight updates");
  const auto state=optimizer.state().find(weight.unsafeGetTensorImpl());
  check(state!=optimizer.state().end(),"new branch actual AdamW state present");
  const auto *adam=dynamic_cast<const torch::optim::AdamWParamState *>(state->second.get());
  check(adam && adam->step()==4,"new branch actual AdamW step");
  {torch::NoGradGuard guard;auto sparse=x;sparse.observed=x.observed.clone();sparse.observed[0][0][7][1]=false;
    auto poisoned=sparse;poisoned.data=torch::where(sparse.observed,x.data,torch::full_like(x.data,std::numeric_limits<float>::quiet_NaN()));
    exact(b->encode(poisoned).z_contextual_global,b->encode(sparse).z_contextual_global,"trained delta hidden storage isolation");
    const auto order=torch::tensor({2,0,1},torch::kInt64).to(torch::kCUDA);auto perm=x;
    perm.data=x.data.index_select(1,order);perm.observed=x.observed.index_select(1,order);perm.channel_ids=x.channel_ids.index_select(0,order);
    close(b->encode(perm).z_contextual_global,b->encode(x).z_contextual_global,"configured semantic channel permutation",2e-5,2e-6);
    auto absent=x;absent.observed=torch::zeros_like(x.observed);auto out=b->encode(absent);
    exact(out.z_contextual_global,torch::zeros_like(out.z_contextual_global),"all absent native zero");check(!out.sample_valid_mask.any().item<bool>(),"all absent invalid");}
  b->zero_grad();auto leaf=x;leaf.data=x.data.detach().clone().set_requires_grad(true);
  b->forward(leaf,mask.hidden).encoding.z_contextual_global.sum().backward();
  check(leaf.data.grad().masked_select(mask.target).abs().sum().item<double>()==0,"Q-hidden input gradient exact zero");
  check(leaf.data.grad().masked_select(mask.visible).abs().sum().item<double>()>0,"visible input gradient remains");
}
std::string configuration_id(const std::string &text) {
  uint64_t h=14695981039346656037ULL;for(unsigned char c:text){h^=c;h*=1099511628211ULL;}h^=0;h*=1099511628211ULL;
  std::ostringstream out;out<<"rpb-config-fnv1a-v1-"<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return out.str();
}
void malformed(const fs::path &path,const rpb::Settings &s,int mode,bool dtype=false,bool shape=false,bool semantics=true) {
  torch::serialize::OutputArchive a;auto text=[&](const std::string &k,const std::string &v){a.write(k,embedding::archive::text_tensor(v),true);};
  text("encoder_id",rpb::kEncoderId);a.write("format_version",torch::tensor(int64_t(1)),true);text("artifact_kind","rpb_training_checkpoint_v1");
  text("rng_policy","splitmix64-counter-rows-masks-torch-attempt-v1");const auto cfg=rpb::settings_text(s);
  text("settings",cfg);text("configuration_id",configuration_id(cfg));
  if(mode>=0) {auto tag=dtype?torch::tensor(double(mode),torch::kFloat64):torch::tensor(int64_t(mode));if(shape)tag=tag.reshape({1});a.write("temporal_difference_input",tag,true);}
  if(semantics)text("temporal_difference_input_semantics",rpb::visible_difference_input_semantics(s.model));
  a.save_to(path.string());
}
void archive_contract() {
  const auto dir=fs::temp_directory_path()/("rpb-visible-difference-model-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(dir),"exclusive preserved synthetic archive leaf");
  for(int mode:{0,1}) {
    auto s=settings(mode);auto raw=input(s.model,3);auto data=rpb::describe_dataset(raw,s.model,"unitless,unitless,unitless");
    rpb::Checkpoint cp;cp.settings=s;cp.model=rpb::Model(s.model);cp.model->eval();cp.scaler=rpb::fit_scaler(raw,s.model);
    cp.schema_id=data.schema_id;cp.dataset_id=data.dataset_id;cp.scaler_fit_dataset_id=data.dataset_id;cp.training_policy_id="rpb-training-context-deletion-015-v1";
    torch::optim::AdamW optimizer(cp.model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
    const auto path=dir/("mode-"+std::to_string(mode)+".pt");rpb::save_checkpoint(path.string(),cp,optimizer);
    torch::serialize::InputArchive saved;saved.load_from(path.string(),torch::kCPU);torch::Tensor tag,sem;
    check(saved.try_read("temporal_difference_input",tag,true)==bool(mode) && saved.try_read("temporal_difference_input_semantics",sem,true)==bool(mode),"conditional checkpoint difference fields");
    auto restored=rpb::load_checkpoint(path.string(),torch::kCUDA);restored.model->eval();
    {torch::NoGradGuard guard;auto x=cp.scaler.transform(raw,s.model);exact(cp.model->encode(x).z_contextual_global,restored.model->encode(x).z_contextual_global,"typed CUDA checkpoint roundtrip");}
    const auto rawpath=dir/("raw-"+std::to_string(mode)+".pt"),exportpath=dir/("export-"+std::to_string(mode)+".pt");rpb::save_dataset(rawpath.string(),data);
    std::vector<std::string> args{"rpb","embed","--input",rawpath.string(),"--checkpoint",path.string(),"--output",exportpath.string(),"--device","cuda"};
    std::vector<char*> argv;for(auto &a:args)argv.push_back(a.data());check(rpb::run_cli(int(argv.size()),argv.data())==0,"CUDA embed conditional schema");
    torch::serialize::InputArchive exported;exported.load_from(exportpath.string(),torch::kCPU);
    check(exported.try_read("temporal_difference_input",tag,true)==bool(mode) && exported.try_read("temporal_difference_input_semantics",sem,true)==bool(mode),"conditional exported difference fields");
    if(mode)check(tag.scalar_type()==torch::kInt64 && tag.dim()==0 && tag.item<int64_t>()==1 && embedding::archive::tensor_text(sem)==rpb::kVisibleDifferenceInputSemantics,"exact typed difference export");
    const auto config=dir/("mode-"+std::to_string(mode)+".conf");{std::ofstream f(config);f<<rpb::settings_text(s);}
    args={"rpb","train","--config",config.string(),"--checkpoint",(dir/"rejected.pt").string()};argv.clear();for(auto &a:args)argv.push_back(a.data());
    rejects([&]{rpb::run_cli(int(argv.size()),argv.data());},"ordinary fresh rejects early/difference architecture");check(!fs::exists(dir/"rejected.pt"),"rejection before output");
  }
  int i=0;auto reject=[&](const rpb::Settings &s,int mode,bool dtype=false,bool shape=false,bool semantics=true) {
    const auto path=dir/("bad-"+std::to_string(i++)+".pt");malformed(path,s,mode,dtype,shape,semantics);
    bool expected=false;try{rpb::load_checkpoint(path.string(),torch::kCUDA);}catch(const std::exception &e){expected=std::string(e.what()).find("temporal difference input/semantics")!=std::string::npos;}
    check(expected,"malformed discriminator rejects before absent model/weights");
  };
  reject(settings(1),-1);reject(settings(1),0);reject(settings(1),2);reject(settings(1),1,true);reject(settings(1),1,false,true);reject(settings(1),1,false,false,false);reject(settings(0),0);
  std::cout<<"Preserved visible-difference model fixture archives: "<<dir.string()<<'\n';
}
void configuration_contract() {
  auto s=settings(1);check(rpb::parse_settings(rpb::settings_text(s)).model.temporal_difference_input==1,"difference settings roundtrip");
  for(int bad:{-1,2}){auto c=s.model;c.temporal_difference_input=bad;rejects([&]{rpb::validate_config(c);},"closed difference discriminator");}
  for(int change=0;change<11;++change){auto c=s.model;
    if(change==0)c.channel_count=2;
    if(change==1)c.history_length=24;
    if(change==2)c.input_width=2;
    if(change==3)c.patch_length=4;
    if(change==4)c.encoder_width=32;
    if(change==5)c.export_width=16;
    if(change==6)c.channel_mixer_layers=0;
    if(change==7)c.channel_mixer_placement=0;
    if(change==8)c.global_bottleneck_mode=0;
    if(change==9)c.global_pool_input_source=1;
    if(change==10)c.dropout=.1;
    rejects([&]{rpb::validate_config(c);},"closed difference configuration");}
}
} // namespace
int main() try {
  check(torch::cuda::is_available(),"visible-difference admission requires actual CUDA");torch::set_num_threads(1);
  configuration_contract();default_parity();manual_differences();candidate_graph();archive_contract();
  std::cout<<"Visible difference model CUDA admission passed\n"<<RPB_SOURCE_ID<<'\n';return 0;
} catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
