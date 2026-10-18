// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#include "cuwacunu/evaluation/fixed_regression.h"
#include <ATen/CPUGeneratorImpl.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace cuwacunu::evaluation {
namespace {
namespace fs = std::filesystem;
constexpr std::array<uint64_t,3> repetitions{2701,2802,2903};
void require(bool value, const std::string &why) {
  if (!value) throw std::runtime_error("[fixed regression] " + why);
}
bool safe_name(const std::string &s) {
  return !s.empty() && s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos;
}
std::string quote(const std::string &s) {
  std::ostringstream o; o << '"';
  for (unsigned char c:s) {
    if(c=='"'||c=='\\')o<<'\\'<<c;
    else if(c=='\n')o<<"\\n"; else if(c=='\r')o<<"\\r"; else if(c=='\t')o<<"\\t";
    else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else o<<c;
  }
  return o.str()+'"';
}
void matrix(const torch::Tensor &x, const char *name) {
  require(x.defined() && x.device().is_cpu() && x.dim()==2 && x.size(0)>0 && x.size(1)>0 &&
      (x.scalar_type()==torch::kFloat32 || x.scalar_type()==torch::kFloat64),std::string(name)+" requires CPU floating [N,D]");
}
void bool_shape(const torch::Tensor &x, at::IntArrayRef shape) {
  require(x.defined()&&x.device().is_cpu()&&x.scalar_type()==torch::kBool&&x.sizes()==shape,"validity shape/type mismatch");
}
void ids(const torch::Tensor &x, int64_t n, bool unique) {
  require(x.defined()&&x.device().is_cpu()&&x.scalar_type()==torch::kInt64&&x.dim()==1&&x.size(0)==n,"row/group IDs require CPU int64[N]");
  if(unique){std::set<int64_t> seen;for(int64_t i=0;i<n;++i)require(seen.insert(x[i].item<int64_t>()).second,"duplicate row ID");}
}
void validate(const FrozenFeatures &f) {
  matrix(f.values,"features");bool_shape(f.valid,{f.values.size(0)});ids(f.row_ids,f.values.size(0),true);
  require(torch::isfinite(f.values.index({f.valid})).all().item<bool>(),"nonfinite valid feature row");
}
void validate(const TargetRows &t) {
  matrix(t.targets.values,"targets");require(t.targets.values.scalar_type()==torch::kFloat64,"target precision must be float64");
  bool_shape(t.targets.valid,t.targets.values.sizes());ids(t.row_ids,t.targets.values.size(0),true);ids(t.group_ids,t.targets.values.size(0),false);
  require(torch::isfinite(t.targets.values.masked_select(t.targets.valid)).all().item<bool>(),"nonfinite valid target");
}
torch::Tensor selected(const torch::Tensor &valid) { return valid.nonzero().reshape({-1}); }
void archive_save(const std::string &path, torch::serialize::OutputArchive &out) {
  const auto target=fs::path(path);
  auto pattern=(target.parent_path()/(target.filename().string()+".tmp-XXXXXX")).string();
  std::vector<char> temporary(pattern.begin(),pattern.end());temporary.push_back('\0');
  const auto fd=::mkstemp(temporary.data());require(fd>=0,"cannot create archive staging file");::close(fd);
  try{out.save_to(temporary.data());require(::link(temporary.data(),path.c_str())==0,"archive already exists or cannot publish: "+path);::unlink(temporary.data());}
  catch(...){::unlink(temporary.data());throw;}
}
uint64_t stream_seed(uint64_t seed,uint64_t stream){
  uint64_t v=seed+0x9e3779b97f4a7c15ULL*stream;
  v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);
}
struct Normalizer {
  torch::Tensor mean,scale;
  bool supported{false};
  explicit Normalizer(const FrozenFeatures &f) {
    mean=torch::zeros({f.values.size(1)},torch::kFloat64);scale=torch::ones_like(mean);
    auto i=selected(f.valid);supported=i.numel()>=2;if(!supported)return;
    auto x=f.values.to(torch::kFloat64).index_select(0,i);mean=x.mean(0);scale=(x-mean).square().mean(0).sqrt().clamp_min(1e-8);
    require(torch::isfinite(mean).all().item<bool>()&&torch::isfinite(scale).all().item<bool>(),"feature normalizer overflow");
  }
  torch::Tensor transform(const FrozenFeatures &f) const {
    auto x=torch::zeros_like(f.values,torch::kFloat64);auto i=selected(f.valid);
    x.index_copy_(0,i,(f.values.to(torch::kFloat64).index_select(0,i)-mean)/scale);
    require(torch::isfinite(x).all().item<bool>(),"normalized feature overflow");return x;
  }
};
struct TargetNormalizer {
  torch::Tensor mean,scale,counts,supported;
  explicit TargetNormalizer(const TargetRows &t) {
    mean=torch::zeros({t.targets.values.size(1)},torch::kFloat64);scale=torch::ones_like(mean);counts=t.targets.valid.sum(0);supported=counts.ge(2);
    for(int64_t j=0;j<mean.numel();++j){auto i=selected(t.targets.valid.select(1,j));if(i.numel()<2)continue;
      auto y=t.targets.values.select(1,j).index_select(0,i);mean.select(0,j).copy_(y.mean());scale.select(0,j).copy_((y-y.mean()).square().mean().sqrt().clamp_min(1e-8));}
    require(torch::isfinite(mean).all().item<bool>()&&torch::isfinite(scale).all().item<bool>(),"target scale overflow");
  }
};
struct Fit {
  Normalizer features;
  torch::Tensor target_mean,target_scale,target_counts,target_supported,ridge_w,ridge_b,supported,w1,b1,w2,b2;
  int64_t training_rows;
  Fit(const FrozenFeatures &f,const TargetRows &t,const TargetNormalizer &labels):features(f),target_mean(labels.mean),target_scale(labels.scale),target_counts(labels.counts),target_supported(labels.supported){
    training_rows=f.valid.sum().item<int64_t>();auto x=features.transform(f);const auto width=x.size(1),outputs=t.targets.values.size(1);
    ridge_w=torch::zeros({width,outputs},torch::kFloat64);ridge_b=torch::zeros({outputs},torch::kFloat64);supported=torch::zeros({outputs},torch::kBool);
    for(int64_t j=0;j<outputs;++j){auto i=selected(f.valid.logical_and(t.targets.valid.select(1,j)));if(i.numel()<2||!target_supported[j].item<bool>())continue;
      auto a=x.index_select(0,i);auto xm=a.mean(0);a=a-xm;auto y=(t.targets.values.select(1,j).index_select(0,i)-target_mean[j])/target_scale[j];
      auto ym=y.mean();auto target=(y-ym).unsqueeze(1);torch::Tensor weights;
      if(width>i.numel())weights=a.transpose(0,1).matmul(at::linalg_solve(a.matmul(a.transpose(0,1))+torch::eye(i.numel(),a.options()),target));
      else weights=at::linalg_solve(a.transpose(0,1).matmul(a)+torch::eye(width,a.options()),a.transpose(0,1).matmul(target));
      ridge_w.select(1,j).copy_(weights.reshape({width}));ridge_b.select(0,j).copy_(ym-xm.matmul(weights).squeeze());supported.select(0,j).fill_(true);
    }
    require(torch::isfinite(ridge_w).all().item<bool>()&&torch::isfinite(ridge_b).all().item<bool>(),"ridge fit nonfinite");
  }
  void fit_neural(const FrozenFeatures &f,const TargetRows &t,uint64_t seed){
    const auto rows=selected(f.valid);auto x=features.transform(f).index_select(0,rows);
    auto eligible=t.targets.valid.index_select(0,rows).logical_and(supported.unsqueeze(0));
    if(!eligible.any().item<bool>()){
      w1=torch::zeros({x.size(1),16},x.options());b1=torch::zeros({16},x.options());w2=torch::zeros({16,t.targets.values.size(1)},x.options());b2=torch::zeros({t.targets.values.size(1)},x.options());return;
    }
    auto y=torch::where(eligible,(t.targets.values.index_select(0,rows)-target_mean)/target_scale,torch::zeros_like(eligible,torch::kFloat64));
    auto generator=at::make_generator<at::CPUGeneratorImpl>(seed);
    w1=(at::randn({x.size(1),16},generator,x.options())*std::sqrt(2.0/x.size(1))).set_requires_grad(true);
    b1=torch::zeros({16},x.options()).set_requires_grad(true);
    w2=(at::randn({16,y.size(1)},generator,x.options())/4.0).set_requires_grad(true);
    b2=torch::zeros({y.size(1)},x.options()).set_requires_grad(true);
    torch::optim::Adam optimizer({w1,b1,w2,b2},torch::optim::AdamOptions(.01));
    for(int64_t step=0;step<100;++step){optimizer.zero_grad();auto p=torch::tanh(x.matmul(w1)+b1).matmul(w2)+b2;
      auto loss=(p-y).square().masked_select(eligible).mean();require(torch::isfinite(loss).item<bool>(),"neural MSE nonfinite");loss.backward();optimizer.step();}
    w1=w1.detach();b1=b1.detach();w2=w2.detach();b2=b2.detach();
    require(torch::isfinite(w1).all().item<bool>()&&torch::isfinite(b1).all().item<bool>()&&torch::isfinite(w2).all().item<bool>()&&torch::isfinite(b2).all().item<bool>(),"neural fit nonfinite");
  }
  std::pair<torch::Tensor,torch::Tensor> predict(const FrozenFeatures &f)const{
    torch::NoGradGuard guard;auto x=features.transform(f);
    auto ridge=(x.matmul(ridge_w)+ridge_b)*target_scale+target_mean;
    auto neural=(torch::tanh(x.matmul(w1)+b1).matmul(w2)+b2)*target_scale+target_mean;
    require(torch::isfinite(ridge).all().item<bool>()&&torch::isfinite(neural).all().item<bool>(),"prediction nonfinite");return {ridge,neural};
  }
  void save(const std::string &path,uint64_t seed)const{
    torch::serialize::OutputArchive a;
    a.write("feature_mean",features.mean,true);a.write("feature_scale",features.scale,true);
    a.write("feature_training_supported",torch::tensor(features.supported,torch::kBool),true);
    a.write("target_mean",target_mean,true);a.write("target_scale",target_scale,true);
    a.write("target_training_counts",target_counts,true);a.write("target_training_supported",target_supported,true);
    a.write("ridge_weights",ridge_w,true);a.write("ridge_intercept",ridge_b,true);a.write("supported_targets",supported,true);
    a.write("neural_w1",w1,true);a.write("neural_b1",b1,true);a.write("neural_w2",w2,true);a.write("neural_b2",b2,true);
    a.write("actual_seed",torch::tensor(static_cast<int64_t>(seed),torch::kInt64),true);
    a.write("training_feature_rows",torch::tensor(training_rows,torch::kInt64),true);archive_save(path,a);
  }
};
std::string score_json(const RegressionScore &s){
  std::ostringstream o;o<<std::setprecision(17)<<"{\"total\":"<<s.total<<",\"target_eligible\":"<<s.target_eligible<<",\"valid\":"<<s.valid<<",\"coverage\":"<<s.coverage<<",\"mae\":";
  if(s.supported)o<<s.mae;else o<<"null";o<<",\"rmse\":";if(s.supported)o<<s.rmse;else o<<"null";
  o<<",\"r2\":";if(s.r2_supported)o<<s.r2;else o<<"null";return o.str()+"}";
}
void save_predictions(const fs::path &path,const TargetRows &t,const FrozenFeatures &f,const Fit &fit,const std::pair<torch::Tensor,torch::Tensor> &p){
  torch::serialize::OutputArchive a;a.write("ridge_predictions",p.first,true);a.write("neural_predictions",p.second,true);
  a.write("valid",f.valid.unsqueeze(1).logical_and(t.targets.valid).logical_and(fit.supported.unsqueeze(0)),true);
  a.write("targets",t.targets.values,true);a.write("target_valid",t.targets.valid,true);a.write("row_ids",t.row_ids,true);a.write("group_ids",t.group_ids,true);archive_save(path.string(),a);
}
} // namespace

std::vector<TargetSpec> past_window_target_specs(){return {{"past_volatility","logreturn"},{"past_trend","logreturn/tick"},{"past_activity","log1p(base_volume)"},{"past_relation","correlation"}};}
TargetValues past_window_targets(const torch::Tensor &values,const torch::Tensor &observed){
  require(values.defined()&&values.device().is_cpu()&&values.is_floating_point()&&values.dim()==4&&values.size(0)>0&&values.size(1)>0&&values.size(2)>=9&&values.size(3)==9,"raw target input needs CPU [N,C,H>=9,9]");
  bool_shape(observed,values.sizes());require(torch::isfinite(values.masked_select(observed)).all().item<bool>(),"observed raw target values nonfinite");
  auto raw=torch::where(observed,values.to(torch::kFloat64),torch::zeros_like(values,torch::kFloat64));
  auto close=raw.select(3,3);auto close_valid=observed.select(3,3).logical_and(close.gt(0));
  auto logclose=torch::where(close_valid,close,torch::ones_like(close)).log();const auto h=close.size(2);
  auto returns=logclose.slice(2,1)-logclose.slice(2,0,h-1);
  auto valid=close_valid.slice(2,1).logical_and(close_valid.slice(2,0,h-1));returns=torch::where(valid,returns,torch::zeros_like(returns));
  auto count=valid.sum(2);auto mean=returns.sum(2)/count.clamp_min(1);auto centered=torch::where(valid,returns-mean.unsqueeze(2),torch::zeros_like(returns));
  auto volatility=(centered.square().sum(2)/count.clamp_min(1)).sqrt();auto movement_valid=count.ge(8).all(1);
  auto volume=raw.select(3,4);auto av=observed.select(3,4).logical_and(volume.ge(0));
  auto activity=torch::where(av,volume,torch::zeros_like(volume)).log1p();auto ac=av.sum(2);
  auto activity_mean=activity.sum(2)/ac.clamp_min(1);auto activity_valid=ac.ge(8).all(1);
  auto relation=torch::zeros({values.size(0)},torch::kFloat64);auto pairs=torch::zeros({values.size(0)},torch::kInt64);
  for(int64_t c=0;c<values.size(1);++c)for(int64_t d=c+1;d<values.size(1);++d){
    auto common=valid.select(1,c).logical_and(valid.select(1,d));auto n=common.sum(1);
    auto x=torch::where(common,returns.select(1,c),torch::zeros_like(returns.select(1,c)));
    auto y=torch::where(common,returns.select(1,d),torch::zeros_like(returns.select(1,d)));
    auto xc=torch::where(common,x-x.sum(1).div(n.clamp_min(1)).unsqueeze(1),torch::zeros_like(x));
    auto yc=torch::where(common,y-y.sum(1).div(n.clamp_min(1)).unsqueeze(1),torch::zeros_like(y));
    auto vx=xc.square().sum(1)/n.clamp_min(1),vy=yc.square().sum(1)/n.clamp_min(1);
    auto eligible=n.ge(8).logical_and(vx.ge(1e-12)).logical_and(vy.ge(1e-12));
    auto correlation=(xc*yc).sum(1).div(n.clamp_min(1)).div((vx*vy).sqrt().clamp_min(1e-12)).clamp(-1,1);
    relation+=torch::where(eligible,correlation,torch::zeros_like(correlation));pairs+=eligible.to(torch::kInt64);
  }
  auto output=torch::stack({volatility.mean(1),mean.mean(1),activity_mean.mean(1),relation/pairs.clamp_min(1)},1);
  auto eligibility=torch::stack({movement_valid,movement_valid,activity_valid,pairs.ge(2)},1);
  output=torch::where(eligibility,output,torch::zeros_like(output));require(torch::isfinite(output).all().item<bool>(),"target arithmetic overflow");return {output,eligibility};
}
RegressionScore regression_score(const torch::Tensor &prediction,const torch::Tensor &target,const torch::Tensor &target_valid,const torch::Tensor &prediction_valid){
  require(prediction.defined()&&target.defined()&&prediction.device().is_cpu()&&target.device().is_cpu()&&prediction.is_floating_point()&&target.is_floating_point()&&prediction.dim()==1&&prediction.sizes()==target.sizes()&&target.numel()>0,"score needs matching CPU vectors");
  bool_shape(target_valid,target.sizes());bool_shape(prediction_valid,target.sizes());auto valid=target_valid.logical_and(prediction_valid);auto i=selected(valid);
  RegressionScore s;s.total=target.numel();s.target_eligible=target_valid.sum().item<int64_t>();s.valid=i.numel();s.coverage=double(s.valid)/s.total;s.supported=s.valid>0;if(!s.supported)return s;
  auto y=target.to(torch::kFloat64).index_select(0,i),p=prediction.to(torch::kFloat64).index_select(0,i);
  require(torch::isfinite(y).all().item<bool>()&&torch::isfinite(p).all().item<bool>(),"nonfinite scored target/prediction");auto error=p-y;
  s.mae=error.abs().mean().item<double>();s.rmse=error.square().mean().sqrt().item<double>();double denominator=(y-y.mean()).square().sum().item<double>();
  s.r2_supported=denominator>0&&s.valid>=2;if(s.r2_supported)s.r2=1-error.square().sum().item<double>()/denominator;
  require(std::isfinite(s.mae)&&std::isfinite(s.rmse)&&(!s.r2_supported||std::isfinite(s.r2)),"score arithmetic overflow");return s;
}
void save_features(const std::string &path,const FrozenFeatures &f){validate(f);torch::serialize::OutputArchive a;a.write("values",torch::where(f.valid.unsqueeze(1),f.values,torch::zeros_like(f.values)),true);a.write("valid",f.valid,true);a.write("row_ids",f.row_ids,true);archive_save(path,a);}
FrozenFeatures load_features(const std::string &path){torch::serialize::InputArchive a;a.load_from(path,torch::kCPU);FrozenFeatures f;a.read("values",f.values);a.read("valid",f.valid);a.read("row_ids",f.row_ids);validate(f);return f;}
void save_targets(const std::string &path,const TargetRows &t){validate(t);torch::serialize::OutputArchive a;a.write("targets",torch::where(t.targets.valid,t.targets.values,torch::zeros_like(t.targets.values)),true);a.write("valid",t.targets.valid,true);a.write("row_ids",t.row_ids,true);a.write("group_ids",t.group_ids,true);archive_save(path,a);}
TargetRows load_targets(const std::string &path){torch::serialize::InputArchive a;a.load_from(path,torch::kCPU);TargetRows t;a.read("targets",t.targets.values);a.read("valid",t.targets.valid);a.read("row_ids",t.row_ids);a.read("group_ids",t.group_ids);validate(t);return t;}

std::string run_fixed_regression(const FixedRegressionRun &run){
  validate(run.training);validate(run.validation);require(!run.methods.empty(),"no methods");
  const auto hash=[](const std::string &s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;};
  require(hash(run.card_sha256)&&hash(run.input_manifest_sha256),"frozen card/input manifest SHA256 required");
  require(!run.dataset_id.empty()&&run.designed_complexity_level>=1&&run.designed_complexity_level<=5,"dataset identity/design level required");
  const auto t=run.training.targets.values.size(1);require(run.validation.targets.values.size(1)==t&&int64_t(run.target_specs.size())==t,"target schema mismatch");
  std::set<std::string> target_names;for(const auto &s:run.target_specs)require(!s.name.empty()&&!s.unit.empty()&&target_names.insert(s.name).second,"target names/units require distinct declared identities");
  std::set<int64_t> train_ids,train_groups;for(int64_t i=0;i<run.training.row_ids.numel();++i){train_ids.insert(run.training.row_ids[i].item<int64_t>());train_groups.insert(run.training.group_ids[i].item<int64_t>());}
  for(int64_t i=0;i<run.validation.row_ids.numel();++i)require(!train_ids.contains(run.validation.row_ids[i].item<int64_t>())&&!train_groups.contains(run.validation.group_ids[i].item<int64_t>()),"TRAIN/VALIDATION row or group overlap");
  std::set<std::string> names;for(const auto &m:run.methods){require(safe_name(m.name)&&names.insert(m.name).second,"unsafe/duplicate method");validate(m.training);validate(m.validation);
    require(torch::equal(m.training.row_ids,run.training.row_ids)&&torch::equal(m.validation.row_ids,run.validation.row_ids),"feature/target row association differs");require(m.training.values.size(1)==m.validation.values.size(1),"method width mismatch");}
  require(!fs::exists(run.output_directory),"output directory exists");TargetNormalizer labels(run.training);
  const auto output_path=fs::path(run.output_directory);if(!output_path.parent_path().empty())fs::create_directories(output_path.parent_path());require(fs::create_directory(output_path),"exclusive output leaf already exists");std::ostringstream report;report<<std::setprecision(17)<<"{\"kind\":\"fixed_past_window_regression_v1\",\"dataset_id\":"<<quote(run.dataset_id)<<",\"designed_complexity_level\":"<<run.designed_complexity_level<<",\"complexity_scale_max\":5,\"evidence_stage\":\"development\",\"view\":\"intact chronological VALIDATION\",\"card_sha256\":"<<quote(run.card_sha256)<<",\"input_manifest_sha256\":"<<quote(run.input_manifest_sha256)<<",\"training_rows\":"<<run.training.row_ids.numel()<<",\"validation_rows\":"<<run.validation.row_ids.numel()<<",\"heads\":{\"ridge_penalty\":1,\"neural_hidden\":16,\"neural_learning_rate\":0.01,\"neural_updates\":100,\"outputs\":"<<t<<",\"loss\":\"masked TRAIN-target-standardized MSE\",\"device\":\"CPU (frozen readouts only)\",\"uncertainty\":\"not estimated; one chronological development split\"},\"methods\":[";
  size_t method_index=0;
  for(const auto &m:run.methods){if(method_index++)report<<',';Fit fit(m.training,run.training,labels);const auto d=m.training.values.size(1);
    const bool has_supported=fit.supported.any().item<bool>();report<<"{\"name\":"<<quote(m.name)<<",\"size\":"<<d<<",\"training_path\":"<<quote(m.training_path)<<",\"validation_path\":"<<quote(m.validation_path)<<",\"training_feature_rows\":"<<fit.training_rows<<",\"feature_training_supported\":"<<(fit.features.supported?"true":"false")<<",\"ridge_parameters\":"<<(d+1)*t<<",\"neural_parameters\":"<<(d+1)*16+17*t<<",\"ridge_fits\":"<<(has_supported?1:0)<<",\"ridge_task_fits\":"<<fit.supported.sum().item<int64_t>()<<",\"neural_fits\":"<<(has_supported?3:0)<<",\"repetitions\":[";
    for(size_t r=0;r<repetitions.size();++r){if(r)report<<',';const auto rep=repetitions[r],seed=stream_seed(rep,d);fit.fit_neural(m.training,run.training,seed);
      const auto folder=fs::path(run.output_directory)/"models"/m.name/("rep-"+std::to_string(rep));fs::create_directories(folder);
      fit.save((folder/"fit.pt").string(),seed);auto train=fit.predict(m.training),validation=fit.predict(m.validation);
      save_predictions(folder/"predictions_train.pt",run.training,m.training,fit,train);save_predictions(folder/"predictions_validation.pt",run.validation,m.validation,fit,validation);
      report<<"{\"repetition\":"<<rep<<",\"actual_seed\":"<<seed<<",\"ridge_fit_reused\":"<<(r&&has_supported?"true":"false")<<",\"neural_updates_completed\":"<<(has_supported?100:0)<<",\"targets\":[";
      for(int64_t j=0;j<t;++j){if(j)report<<',';auto supported=m.validation.valid.logical_and(fit.supported[j]);
        auto ridge=regression_score(validation.first.select(1,j),run.validation.targets.values.select(1,j),run.validation.targets.valid.select(1,j),supported);
        auto neural=regression_score(validation.second.select(1,j),run.validation.targets.values.select(1,j),run.validation.targets.valid.select(1,j),supported);
        report<<"{\"name\":"<<quote(run.target_specs[j].name)<<",\"unit\":"<<quote(run.target_specs[j].unit)<<",\"train_target_eligible\":"<<labels.counts[j].item<int64_t>()<<",\"train_target_supported\":"<<(labels.supported[j].item<bool>()?"true":"false")<<",\"method_training_supported\":"<<(fit.supported[j].item<bool>()?"true":"false")<<",\"train_target_mean\":"<<labels.mean[j].item<double>()<<",\"train_target_scale\":"<<labels.scale[j].item<double>()<<",\"ridge\":"<<score_json(ridge)<<",\"neural\":"<<score_json(neural)<<'}';}
      report<<"]}";
    }report<<"]}";
  }report<<"]}";const auto result=report.str();std::ofstream output(fs::path(run.output_directory)/"report.json");require(bool(output),"cannot create report");output<<result<<'\n';output.close();require(bool(output),"cannot save report");return result;
}
} // namespace cuwacunu::evaluation
