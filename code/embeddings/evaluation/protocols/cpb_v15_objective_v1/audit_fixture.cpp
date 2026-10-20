// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Artificial saved-array cases only. No producer headers, training, autograd,
// market files, old checkpoints or tensor-level matrix arithmetic.
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fixture {
namespace fs=std::filesystem;
using Fields=std::map<std::string,torch::Tensor>;
void need(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
torch::Tensor f64(const std::vector<double>& values,const std::vector<int64_t>& shape){return torch::tensor(values,torch::kFloat64).reshape(shape);}
torch::Tensor i64(const std::vector<int64_t>& values){return torch::tensor(values,torch::kInt64);}
torch::Tensor boolean(const std::vector<int64_t>& values,const std::vector<int64_t>& shape){return torch::tensor(values,torch::kInt64).to(torch::kBool).reshape(shape);}
torch::Tensor text(const std::string& value){return torch::tensor(std::vector<uint8_t>(value.begin(),value.end()),torch::kUInt8);}
void save(const fs::path& file,const Fields& fields){need(!fs::exists(file),"exclusive artificial archive required");torch::serialize::OutputArchive archive;for(const auto& [key,value]:fields)archive.write(key,value,true);archive.save_to(file.string());}
std::string quote(const std::string& value){std::ostringstream out;out<<'"';for(char c:value){if(c=='"'||c=='\\')out<<'\\';need(c>=' ',"fixture path controls refused");out<<c;}return out.str()+'"';}
void write(const fs::path& file,const std::string& value){need(!fs::exists(file),"exclusive fixture text required");std::ofstream out(file,std::ios::binary);out<<value;need(bool(out),"fixture text write failed");}
struct Stats{std::vector<double> mean,scale;int64_t count;};
Stats moments(const std::vector<double>& x,int64_t n,int64_t d,const std::vector<int64_t>& valid){
  Stats result{std::vector<double>(size_t(d),0),std::vector<double>(size_t(d),0),0};
  for(int64_t i=0;i<n;++i)if(valid[size_t(i)]){++result.count;for(int64_t j=0;j<d;++j)result.mean[size_t(j)]+=x[size_t(i*d+j)];}
  need(result.count>=2,"nonvacuous artificial TRAIN population");for(auto& value:result.mean)value/=result.count;
  for(int64_t i=0;i<n;++i)if(valid[size_t(i)])for(int64_t j=0;j<d;++j){const auto delta=x[size_t(i*d+j)]-result.mean[size_t(j)];result.scale[size_t(j)]+=delta*delta;}
  for(auto& value:result.scale)value=std::max(1e-8,std::sqrt(value/result.count));return result;
}
std::vector<double> normalize(const std::vector<double>& x,int64_t n,int64_t d,const std::vector<int64_t>& valid,const Stats& stats){
  auto result=x;for(int64_t i=0;i<n;++i)for(int64_t j=0;j<d;++j)result[size_t(i*d+j)]=valid[size_t(i)]?(x[size_t(i*d+j)]-stats.mean[size_t(j)])/stats.scale[size_t(j)]:0.;return result;
}
std::vector<double> head(const std::vector<double>& x,int64_t n,int64_t d,const std::vector<double>& w1,const std::vector<double>& b1,const std::vector<double>& w2,const std::vector<double>& b2,int64_t outputs,bool hidden_only=false){
  std::vector<double> hidden(size_t(n*16),0),result(size_t(n*outputs),0);
  for(int64_t i=0;i<n;++i)for(int64_t h=0;h<16;++h){double value=b1[size_t(h)];for(int64_t j=0;j<d;++j)value+=x[size_t(i*d+j)]*w1[size_t(j*16+h)];hidden[size_t(i*16+h)]=value;}
  if(hidden_only)return hidden;
  for(int64_t i=0;i<n;++i)for(int64_t j=0;j<outputs;++j){double value=b2[size_t(j)];for(int64_t h=0;h<16;++h)value+=std::tanh(hidden[size_t(i*16+h)])*w2[size_t(h*outputs+j)];result[size_t(i*outputs+j)]=value;}return result;
}
std::string regression(const fs::path& directory,const std::string& mutation){
  constexpr int64_t n=5,d=3,t=4,p=4;
  const std::vector<double> train{-1,-1,7,-1,1,7,1,-1,7,1,1,7,900,800,700};
  const std::vector<int64_t> valid{1,1,1,1,0},ids{100,101,102,103,104},pids{200,201,202,203},pv{1,1,1,0};
  const std::vector<double> probe{0,0,7,2,-1,7,-2,1,7,900,800,700};
  std::vector<double> targets(size_t(n*t)),weights(size_t(d*t),0),bias(static_cast<size_t>(t)),tm(static_cast<size_t>(t)),ts(static_cast<size_t>(t));
  std::vector<int64_t> target_valid(size_t(n*t),1),counts(static_cast<size_t>(t)),supported(size_t(t),1);
  for(int64_t i=0;i<n;++i)for(int64_t j=0;j<t;++j)targets[size_t(i*t+j)]=i==4?20.+j:10.+j+(j+1)*train[size_t(i*d)]+(2.-j)*train[size_t(i*d+1)];
  target_valid[1]=0;targets[1]=0;
  const auto feature=moments(train,n,d,valid);const auto normalized_train=normalize(train,n,d,valid,feature);
  for(int64_t j=0;j<t;++j){std::vector<double> y(static_cast<size_t>(n));std::vector<int64_t> mask(static_cast<size_t>(n));
    for(int64_t i=0;i<n;++i){y[size_t(i)]=targets[size_t(i*t+j)];mask[size_t(i)]=target_valid[size_t(i*t+j)];}
    const auto target_stats=moments(y,n,1,mask);tm[size_t(j)]=target_stats.mean[0];ts[size_t(j)]=target_stats.scale[0];counts[size_t(j)]=target_stats.count;
    double count=0,mx=0,my=0,mz=0;for(int64_t i=0;i<n;++i)if(valid[size_t(i)]&&mask[size_t(i)]){++count;mx+=normalized_train[size_t(i*d)];my+=normalized_train[size_t(i*d+1)];mz+=(y[size_t(i)]-tm[size_t(j)])/ts[size_t(j)];}mx/=count;my/=count;mz/=count;
    // Independent closed-form 2x2 ridge solution; the constant coordinate is
    // zero after normalization and has exactly zero ridge coefficient.
    double a=1,b=0,c=1,u=0,v=0;
    for(int64_t i=0;i<n;++i)if(valid[size_t(i)]&&mask[size_t(i)]){const auto x=normalized_train[size_t(i*d)]-mx,z=normalized_train[size_t(i*d+1)]-my,truth=(y[size_t(i)]-tm[size_t(j)])/ts[size_t(j)]-mz;a+=x*x;b+=x*z;c+=z*z;u+=x*truth;v+=z*truth;}
    const double determinant=a*c-b*b;weights[size_t(j)]=(c*u-b*v)/determinant;weights[size_t(t+j)]=(a*v-b*u)/determinant;bias[size_t(j)]=mz-mx*weights[size_t(j)]-my*weights[size_t(t+j)];
  }
  std::vector<double> w1(size_t(d*16),0),w2(size_t(16*t),0),b1(16,0),b2{.1,-.1,.2,-.2};w1[0]=.2;w1[1]=-.1;w1[16]=-.15;w1[17]=.25;b1[0]=.1;b1[1]=-.2;
  for(int64_t j=0;j<t;++j){w2[size_t(j)]=.4*(j+1);w2[size_t(t+j)]=-.2*(j+1);}
  if(mutation=="ridge_equation")weights[0]+=.02;
  auto normal_probe=normalize(probe,p,d,pv,feature),neural=head(normal_probe,p,d,w1,b1,w2,b2,t);std::vector<double> ridge(size_t(p*t)),truth(size_t(p*t));std::vector<int64_t> eligible(size_t(p*t),1),label_valid(size_t(p*t),1);
  for(int64_t i=0;i<p;++i)for(int64_t j=0;j<t;++j){double value=bias[size_t(j)];for(int64_t k=0;k<d;++k)value+=normal_probe[size_t(i*d+k)]*weights[size_t(k*t+j)];ridge[size_t(i*t+j)]=value*ts[size_t(j)]+tm[size_t(j)];neural[size_t(i*t+j)]=neural[size_t(i*t+j)]*ts[size_t(j)]+tm[size_t(j)];truth[size_t(i*t+j)]=11.+j+i;eligible[size_t(i*t+j)]=pv[size_t(i)];}
  label_valid[2]=0;truth[2]=0;eligible[2]=0;
  Fields feature_fields{{"values",f64(train,{n,d})},{"valid",boolean(valid,{n})},{"row_ids",i64(ids)}};
  Fields target_fields{{"targets",f64(targets,{n,t})},{"valid",boolean(target_valid,{n,t})},{"row_ids",i64(ids)},{"group_ids",i64(ids)}};
  Fields probe_fields{{"values",f64(probe,{p,d})},{"valid",boolean(pv,{p})},{"row_ids",i64(pids)}};
  Fields probe_targets{{"targets",f64(truth,{p,t})},{"valid",boolean(label_valid,{p,t})},{"row_ids",i64(pids)},{"group_ids",i64(pids)}};
  Fields fit{{"feature_mean",f64(feature.mean,{d})},{"feature_scale",f64(feature.scale,{d})},{"feature_training_supported",torch::tensor(true,torch::kBool)},
    {"target_mean",f64(tm,{t})},{"target_scale",f64(ts,{t})},{"target_training_counts",i64(counts)},{"target_training_supported",boolean(supported,{t})},
    {"supported_targets",boolean(supported,{t})},{"training_feature_rows",torch::tensor(int64_t{4},torch::kInt64)},
    {"ridge_weights",f64(weights,{d,t})},{"ridge_intercept",f64(bias,{t})},{"neural_w1",f64(w1,{d,16})},{"neural_b1",f64(b1,{16})},
    {"neural_w2",f64(w2,{16,t})},{"neural_b2",f64(b2,{t})},{"actual_seed",torch::tensor(int64_t{-1},torch::kInt64)}};
  Fields predictions{{"ridge_predictions",f64(ridge,{p,t})},{"neural_predictions",f64(neural,{p,t})},{"targets",f64(truth,{p,t})},
    {"valid",boolean(eligible,{p,t})},{"target_valid",boolean(label_valid,{p,t})},{"row_ids",i64(pids)},{"group_ids",i64(pids)}};
  if(mutation=="feature_stats")fit["feature_mean"]=f64({.25,0,7},{d});
  if(mutation=="target_stats")fit["target_scale"]=f64({ts[0]*2,ts[1],ts[2],ts[3]},{t});
  if(mutation=="ridge_prediction"){ridge[0]+=.05;predictions["ridge_predictions"]=f64(ridge,{p,t});}
  if(mutation=="neural_prediction"){neural[0]+=.05;predictions["neural_predictions"]=f64(neural,{p,t});}
  if(mutation=="mask_shape")predictions["valid"]=boolean(eligible,{2,p*t/2});
  if(mutation=="target_mask_shape")target_fields["valid"]=boolean(target_valid,{t,n});
  if(mutation=="ids")predictions["row_ids"]=i64({201,200,202,203});
  if(mutation=="groups")predictions["group_ids"]=i64({200,201,202,204});
  if(mutation=="unsupported"){eligible[12]=1;predictions["valid"]=boolean(eligible,{p,t});}
  if(mutation=="scalar_shape")fit["training_feature_rows"]=i64({4});
  const std::array<std::pair<const char*,Fields*>,6> archives{{{"train.pt",&feature_fields},{"train-targets.pt",&target_fields},{"fit.pt",&fit},{"probe.pt",&probe_fields},{"probe-targets.pt",&probe_targets},{"predictions.pt",&predictions}}};
  std::ostringstream plan;plan<<"regression";for(const auto& [name,fields]:archives){const auto file=directory/name;save(file,*fields);plan<<'\t'<<file.string();}return plan.str()+"\n";
}
std::string classification(const fs::path& directory,const std::string& mutation,bool outer,bool near_tie=false){
  constexpr int64_t n=5,d=3,p=4;const std::vector<double> train{-1,-1,7,-1,1,7,1,-1,7,1,1,7,900,800,700},probe{0,0,7,2,-1,7,-2,1,7,900,800,700};
  const std::vector<int64_t> valid{1,1,1,1,0},pv{1,1,1,0},labels{0,0,1,1,0},truth{0,1,0,1};
  const std::string train_ids="[\"t0\",\"t1\",\"t0\",\"t1\",\"t2\"]",probe_ids="[\"v0\",\"v1\",\"v0\",\"v1\"]";
  const auto first=moments(train,n,d,valid);const auto prepared=outer?normalize(train,n,d,valid,first):train,prepared_probe=outer?normalize(probe,p,d,pv,first):probe;
  const auto second=moments(prepared,n,d,valid);const auto x=normalize(prepared_probe,p,d,pv,second);
  std::vector<double> weights{-.4,.4,0,0,0,0},bias{.5,.5},w1(size_t(d*16),0),b1(16,0),w2(32,0),b2{0,near_tie?1e-12:0};
  if(mutation=="ridge_equation")weights[0]+=.01;
  auto hidden=head(x,p,d,w1,b1,w2,b2,2,true),tiny=head(x,p,d,w1,b1,w2,b2,2);std::vector<double> ridge(size_t(p*2));std::vector<int64_t> rc(static_cast<size_t>(p)),nc(static_cast<size_t>(p));
  for(int64_t i=0;i<p;++i){for(int64_t j=0;j<2;++j){double v=bias[size_t(j)];for(int64_t k=0;k<d;++k)v+=x[size_t(i*d+k)]*weights[size_t(k*2+j)];ridge[size_t(i*2+j)]=v;}rc[size_t(i)]=ridge[size_t(i*2+1)]>ridge[size_t(i*2)];nc[size_t(i)]=tiny[size_t(i*2+1)]>tiny[size_t(i*2)];}
  Fields training{{"features",f64(train,{n,d})},{"valid",boolean(valid,{n})},{"labels_scoring_only",i64(labels)},{"source_ids_json",text(train_ids)}};
  Fields surface{{"features",f64(probe,{p,d})},{"valid",boolean(pv,{p})},{"labels_scoring_only",i64(truth)},{"source_ids_json",text(probe_ids)}};
  Fields fit{{"feature_mean",f64(outer?first.mean:std::vector<double>(size_t(d),0),{d})},{"feature_scale",f64(outer?first.scale:std::vector<double>(size_t(d),1),{d})},
    {"outer_normalizer_applied",torch::tensor(outer,torch::kBool)},{"outer_fitted_rows",torch::tensor(outer?int64_t{4}:int64_t{0},torch::kInt64)},{"fitted_rows",torch::tensor(int64_t{4},torch::kInt64)},
    {"ridge_mean",f64(second.mean,{d})},{"ridge_scale",f64(second.scale,{d})},{"tiny_mean",f64(second.mean,{d})},{"tiny_scale",f64(second.scale,{d})},
    {"ridge_weights",f64(weights,{d,2})},{"ridge_intercept",f64(bias,{2})},{"tiny_w1",f64(w1,{d,16})},{"tiny_b1",f64(b1,{16})},{"tiny_w2",f64(w2,{16,2})},{"tiny_b2",f64(b2,{2})},
    {"actual_probe_seed_decimal",text("18446744073709551615")},{"training_source_ids_json",text(train_ids)},{"ridge_penalty",torch::tensor(1.,torch::kFloat64)},
    {"tiny_hidden",torch::tensor(int64_t{16},torch::kInt64)},{"tiny_steps",torch::tensor(int64_t{100},torch::kInt64)},{"tiny_learning_rate",torch::tensor(.01,torch::kFloat64)}};
  Fields prediction{{"probe_input_features",f64(prepared_probe,{p,d})},{"ridge_logits",f64(ridge,{p,2})},{"tiny_hidden_preactivation",f64(hidden,{p,16})},{"tiny_logits",f64(tiny,{p,2})},
    {"ridge",i64(rc)},{"tiny_secondary",i64(nc)},{"labels_scoring_only",i64(truth)},{"source_ids_json",text(probe_ids)},{"valid",boolean(pv,{p})}};
  if(mutation=="normalizer")fit["tiny_scale"]=f64({2,1,1e-8},{d});
  if(mutation=="tie_class"){nc[0]=near_tie?0:1;prediction["tiny_secondary"]=i64(nc);}
  if(mutation=="saved_logits"){ridge[2]+=.1;prediction["ridge_logits"]=f64(ridge,{p,2});}
  if(mutation=="source_ids")prediction["source_ids_json"]=text("[\"bad\"]");
  if(mutation=="labels")prediction["labels_scoring_only"]=i64({1,1,0,1});
  if(mutation=="feature_input"){auto corrupt=prepared_probe;corrupt[0]+=.1;prediction["probe_input_features"]=f64(corrupt,{p,d});}
  if(mutation=="seed_decimal")fit["actual_probe_seed_decimal"]=text("18446744073709551616");
  const std::array<std::pair<const char*,Fields*>,4> archives{{{"train.pt",&training},{"fit.pt",&fit},{"probe.pt",&surface},{"predictions.pt",&prediction}}};
  std::ostringstream plan;plan<<"classification";for(const auto& [name,fields]:archives){const auto file=directory/name;save(file,*fields);plan<<'\t'<<file.string();}return plan.str()+"\n";
}
std::string waveform(const fs::path& directory,const std::string& mutation,const std::string& mode){
  constexpr int64_t B=3,C=2,T=2,F=2,N=B*C*T*F;
  std::vector<float> mean(N,std::numeric_limits<float>::quiet_NaN()),variance=mean,target=mean;std::vector<int64_t> mask(N,0),eligible(size_t(B*C),0),cells(size_t(B),0);
  for(int i:{0,1,4,8,9,10}){mask[size_t(i)]=1;mean[size_t(i)]=float((i%5)-2)*.6f;variance[size_t(i)]=.5f+.25f*float(i%3);target[size_t(i)]=float(i%3)*.2f;}
  if(mutation=="negative_nll")for(int64_t i=0;i<N;++i)if(mask[size_t(i)]){mean[size_t(i)]=target[size_t(i)];variance[size_t(i)]=.25f;}
  std::vector<double> row_mae(size_t(B),0),row_nll(size_t(B),0);double mae=0,nll=0,huber=0;int64_t examples=0;
  for(int64_t b=0;b<B;++b){double bm=0,bn=0,bh=0;int64_t channels=0;for(int64_t c=0;c<C;++c){double cm=0,cn=0,ch=0;int64_t count=0;
    for(int64_t t=0;t<T;++t)for(int64_t f=0;f<F;++f){const auto i=size_t(((b*C+c)*T+t)*F+f);if(!mask[i])continue;const double error=double(mean[i])-double(target[i]),a=std::abs(error),v=double(variance[i]);cm+=a;cn+=.5*(std::log(v)+error*error/v);ch+=a<=1.?.5*error*error:a-.5;++count;}
    if(count){++channels;eligible[size_t(b*C+c)]=1;cells[size_t(b)]+=count;bm+=cm/count;bn+=cn/count;bh+=ch/count;}}
    if(channels){++examples;row_mae[size_t(b)]=bm/channels;row_nll[size_t(b)]=bn/channels;mae+=bm/channels;nll+=bn/channels;huber+=bh/channels;}}
  mae/=examples;nll/=examples;huber/=examples;(void)mae;
  Fields fields{{"mean",torch::tensor(mean,torch::kFloat32).reshape({B,C,T,F})},{"variance",torch::tensor(variance,torch::kFloat32).reshape({B,C,T,F})},
    {"targets",torch::tensor(target,torch::kFloat32).reshape({B,C,T,F})},{"target_mask",boolean(mask,{B,C,T,F})}};
  if(mode=="fixed_query"){fields["row_mae"]=f64(row_mae,{B});fields["row_nll"]=f64(row_nll,{B});fields["row_target_cells"]=i64(cells);std::vector<int64_t> channel_counts(size_t(B*C),0);for(int64_t i=0;i<N;++i)channel_counts[size_t(i/(T*F))]+=mask[size_t(i)];fields["target_counts"]=i64(channel_counts).reshape({B,C});}
  else{fields["eligible_channels"]=boolean(eligible,{B,C});fields["reporting_nll"]=torch::tensor(nll,torch::kFloat64);fields["training_objective"]=torch::tensor(mode=="objective_candidate"?huber+nll:nll,torch::kFloat64);
    if(mode=="objective_candidate"){fields["mean_huber"]=torch::tensor(huber,torch::kFloat64);fields["variance_nll"]=torch::tensor(nll,torch::kFloat64);}}
  if(mutation=="nonpositive_variance"){variance[0]=0;fields["variance"]=torch::tensor(variance,torch::kFloat32).reshape({B,C,T,F});}
  if(mutation=="supported_nan"){mean[0]=std::numeric_limits<float>::quiet_NaN();fields["mean"]=torch::tensor(mean,torch::kFloat32).reshape({B,C,T,F});}
  if(mutation=="mask_shape")fields["target_mask"]=boolean(mask,{C,B,T,F});
  if(mutation=="eligibility"){eligible[0]=0;fields["eligible_channels"]=boolean(eligible,{B,C});}
  if(mutation=="objective")fields["training_objective"]=torch::tensor(huber+nll+.1,torch::kFloat64);
  if(mutation=="huber")fields["mean_huber"]=torch::tensor(huber+.1,torch::kFloat64);
  if(mutation=="nll")fields["reporting_nll"]=torch::tensor(nll+.1,torch::kFloat64);
  if(mutation=="query_mae"){row_mae[0]+=.1;fields["row_mae"]=f64(row_mae,{B});}
  if(mutation=="query_nll"){row_nll[0]+=.1;fields["row_nll"]=f64(row_nll,{B});}
  if(mutation=="query_counts"){cells[0]++;fields["row_target_cells"]=i64(cells);}
  if(mutation=="channel_counts")fields["target_counts"]=i64({2,2,3,0,0,0}).reshape({B,C});
  const auto file=directory/"waveform.pt";save(file,fields);return mode+'\t'+file.string()+"\n";
}
} // namespace fixture
int main(int argc,char** argv){using namespace fixture;try{
  need(fs::exists("/.dockerenv")&&fs::canonical("/mityane")=="/mityane","managed Linux container fixture only");torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard guard;
  need(argc==3&&std::string(argv[1])=="--output","audit_fixture --output NEWDIRECTORY");const fs::path root(argv[2]);need(root.is_absolute()&&root.lexically_normal()==root&&fs::canonical(root.parent_path())==root.parent_path()&&!fs::exists(root),"fresh canonical fixture leaf");need(fs::create_directory(root),"exclusive fixture directory");
  struct Case{std::string name,plan;bool pass;};std::vector<Case> cases;
  auto add=[&](const std::string& name,bool pass,auto generate){const auto directory=root/name;need(fs::create_directory(directory),"exclusive case directory");const auto plan=directory/"plan.tsv";write(plan,generate(directory));cases.push_back({name,plan.string(),pass});};
  add("regression-valid",true,[&](const auto& dir){return regression(dir,"");});
  for(const std::string mutation:{"feature_stats","target_stats","ridge_equation","ridge_prediction","neural_prediction","mask_shape","target_mask_shape","ids","groups","unsupported","scalar_shape"})
    add("regression-"+mutation,false,[&](const auto& dir){return regression(dir,mutation);});
  for(bool outer:{false,true})add(outer?"classification-outer":"classification-identity",true,[&](const auto& dir){return classification(dir,"",outer);});
  add("classification-near-tie",true,[&](const auto& dir){return classification(dir,"",true,true);});
  for(const std::string mutation:{"normalizer","ridge_equation","tie_class","saved_logits","source_ids","labels","feature_input","seed_decimal"})
    add("classification-"+mutation,false,[&](const auto& dir){return classification(dir,mutation,true);});
  add("classification-near-tie-wrong-class",false,[&](const auto& dir){return classification(dir,"tie_class",true,true);});
  for(const std::string mode:{"objective_control","objective_candidate","fixed_query"})add(mode+"-valid",true,[&](const auto& dir){return waveform(dir,"",mode);});
  add("objective-negative-nll",true,[&](const auto& dir){return waveform(dir,"negative_nll","objective_candidate");});
  for(const std::string mutation:{"nonpositive_variance","supported_nan","mask_shape","eligibility","objective","huber","nll"})
    add("waveform-"+mutation,false,[&](const auto& dir){return waveform(dir,mutation,"objective_candidate");});
  for(const std::string mutation:{"query_mae","query_nll","query_counts","channel_counts"})add("waveform-"+mutation,false,[&](const auto& dir){return waveform(dir,mutation,"fixed_query");});
  std::ostringstream manifest;manifest<<"{\"kind\":\"cpb_v15_independent_audit_artificial_fixtures_v1\",\"real_dataset_reads\":0,\"head_fits\":0,\"optimizer_steps\":0,\"cases\":[";
  for(size_t i=0;i<cases.size();++i){if(i)manifest<<',';manifest<<"{\"name\":"<<quote(cases[i].name)<<",\"plan\":"<<quote(cases[i].plan)<<",\"expect_pass\":"<<(cases[i].pass?"true":"false")<<'}';}manifest<<"]}\n";write(root/"fixture-manifest.json",manifest.str());
  std::cout<<"Artificial audit fixtures generated: "<<cases.size()<<" cases; no real dataset reads.\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
