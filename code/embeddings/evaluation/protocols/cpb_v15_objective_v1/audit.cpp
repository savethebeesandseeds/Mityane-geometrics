// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Independent saved-fit arithmetic. Torch only reads typed CPU arrays and
// converts storage precision; all normalization/head/metric arithmetic is scalar.
// No producer/evaluation/model header, module forward, optimizer or autograd.
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace independent {
namespace fs=std::filesystem;
constexpr double absolute_tolerance=2e-9,relative_tolerance=2e-8;
uint64_t checks=0;
void need(bool yes,const std::string& message){++checks;if(!yes)throw std::runtime_error(message);}
void close(double actual,double expected,const std::string& message){
  need(std::isfinite(actual)&&std::isfinite(expected)&&std::abs(actual-expected)<=absolute_tolerance+relative_tolerance*std::abs(expected),message);
}
struct Sum{double value=0,correction=0;void add(double x){const double y=x-correction,t=value+y;correction=(t-value)-y;value=t;}};
std::string quoted(const std::string& text){std::ostringstream out;out<<'"';for(unsigned char c:text){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c<' ')out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}return out.str()+'"';}
void write_new(const fs::path& file,const std::string& text){
  const int fd=::open(file.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);need(fd>=0,"exclusive output required: "+file.string());
  size_t offset=0;while(offset<text.size()){const auto n=::write(fd,text.data()+offset,text.size()-offset);if(n<=0){::close(fd);throw std::runtime_error("output write failed");}offset+=size_t(n);}need(::close(fd)==0,"output close failed");
}
struct Archive{
  torch::serialize::InputArchive a;
  explicit Archive(const fs::path& file){need(fs::is_regular_file(file)&&!fs::is_symlink(file),"regular archive required");a.load_from(file.string(),torch::kCPU);}
  torch::Tensor get(const std::string& key){torch::Tensor t;a.read(key,t,true);need(t.defined()&&t.device().is_cpu(),"CPU field required: "+key);return t.contiguous();}
};
struct Matrix{int64_t n=0,d=0;std::vector<double> x;double at(int64_t i,int64_t j)const{return x.at(size_t(i*d+j));}};
Matrix matrix(Archive& a,const std::string& key){auto t=a.get(key);need(t.dim()==2&&t.size(0)>0&&t.size(0)<=4096&&t.size(1)>0&&t.size(1)<=256&&(t.scalar_type()==torch::kFloat64||t.scalar_type()==torch::kFloat32),"bounded floating matrix: "+key);t=t.to(torch::kFloat64).contiguous();const auto p=t.const_data_ptr<double>();return{t.size(0),t.size(1),{p,p+t.numel()}};}
std::vector<double> vector(Archive& a,const std::string& key,int64_t n){auto t=a.get(key);need(t.dim()==1&&t.numel()==n&&t.scalar_type()==torch::kFloat64,"F64 vector: "+key);const auto p=t.const_data_ptr<double>();return{p,p+n};}
std::vector<int64_t> integers(Archive& a,const std::string& key,int64_t n){auto t=a.get(key);need(t.dim()==1&&t.numel()==n&&t.scalar_type()==torch::kInt64,"I64 vector: "+key);const auto p=t.const_data_ptr<int64_t>();return{p,p+n};}
std::vector<uint8_t> boolean(Archive& a,const std::string& key,int64_t n,int rank){auto t=a.get(key);need(t.scalar_type()==torch::kBool&&t.dim()==rank&&t.numel()==n,"boolean shape: "+key);const auto p=t.const_data_ptr<bool>();std::vector<uint8_t> out(size_t(n),0);for(int64_t i=0;i<n;++i)out[size_t(i)]=p[i];return out;}
std::vector<uint8_t> mask_matrix(Archive& a,const std::string& key,int64_t n,int64_t d){auto t=a.get(key);need(t.scalar_type()==torch::kBool&&t.dim()==2&&t.size(0)==n&&t.size(1)==d,"exact row/target mask geometry: "+key);const auto p=t.const_data_ptr<bool>();std::vector<uint8_t> out(size_t(n*d),0);for(int64_t i=0;i<n*d;++i)out[size_t(i)]=p[i];return out;}
int64_t scalar_i(Archive& a,const std::string& key){auto t=a.get(key);need(t.dim()==0&&t.scalar_type()==torch::kInt64,"integer scalar: "+key);return *t.const_data_ptr<int64_t>();}
bool scalar_b(Archive& a,const std::string& key){auto t=a.get(key);need(t.dim()==0&&t.scalar_type()==torch::kBool,"boolean scalar: "+key);return *t.const_data_ptr<bool>();}
double scalar_f(Archive& a,const std::string& key){auto t=a.get(key);need(t.dim()==0&&t.scalar_type()==torch::kFloat64,"F64 scalar: "+key);return *t.const_data_ptr<double>();}
std::string text(Archive& a,const std::string& key){auto t=a.get(key);need(t.dim()==1&&t.scalar_type()==torch::kUInt8&&t.numel()<1000000,"bounded UTF8 bytes: "+key);const auto p=t.const_data_ptr<uint8_t>();return{reinterpret_cast<const char*>(p),size_t(t.numel())};}
std::string seed_decimal(Archive& a,const std::string& key){const auto value=text(a,key);need(!value.empty()&&value.size()<=20&&(value.size()==1||value.front()!='0'),"canonical unsigned seed decimal required");for(char c:value)need(c>='0'&&c<='9',"decimal seed digit required");need(value.size()<20||value<="18446744073709551615","seed exceeds uint64");return value;}
void shape(const Matrix& x,int64_t n,int64_t d){need(x.n==n&&x.d==d,"matrix dimensions differ");}
void finite(const std::vector<double>& x){for(double value:x)need(std::isfinite(value),"nonfinite fitted value");}
struct Stats{std::vector<double> mean,scale;int64_t count;};
Stats stats(const Matrix& x,const std::vector<uint8_t>& valid){
  need(int64_t(valid.size())==x.n,"normalizer row association");Stats s{{}, {},0};for(auto value:valid)s.count+=value;
  need(s.count>=2,"two TRAIN fitting rows required");s.mean.resize(size_t(x.d));s.scale.resize(size_t(x.d));
  for(int64_t j=0;j<x.d;++j){Sum sum;for(int64_t i=0;i<x.n;++i)if(valid[size_t(i)]){need(std::isfinite(x.at(i,j)),"nonfinite supported TRAIN feature");sum.add(x.at(i,j));}s.mean[size_t(j)]=sum.value/s.count;Sum squared;for(int64_t i=0;i<x.n;++i)if(valid[size_t(i)]){const auto delta=x.at(i,j)-s.mean[size_t(j)];squared.add(delta*delta);}s.scale[size_t(j)]=std::max(1e-8,std::sqrt(squared.value/s.count));}return s;
}
void verify_stats(Archive& fit,const Stats& s,const std::string& mean,const std::string& scale){const auto m=vector(fit,mean,int64_t(s.mean.size())),v=vector(fit,scale,int64_t(s.scale.size()));for(size_t j=0;j<m.size();++j){close(m[j],s.mean[j],"TRAIN mean mismatch: "+mean);close(v[j],s.scale[j],"TRAIN scale mismatch: "+scale);need(v[j]>=1e-8,"normalizer scale below floor");}}
Matrix normalized(const Matrix& x,const std::vector<uint8_t>& valid,const std::vector<double>& mean,const std::vector<double>& scale){
  need(int64_t(valid.size())==x.n&&int64_t(mean.size())==x.d&&scale.size()==mean.size(),"normalization geometry");Matrix out{x.n,x.d,std::vector<double>(x.x.size(),0)};
  for(int64_t i=0;i<x.n;++i)if(valid[size_t(i)])for(int64_t j=0;j<x.d;++j){need(std::isfinite(x.at(i,j))&&std::isfinite(mean[size_t(j)])&&std::isfinite(scale[size_t(j)])&&scale[size_t(j)]>=1e-8,"supported normalization operand");out.x[size_t(i*x.d+j)]=(x.at(i,j)-mean[size_t(j)])/scale[size_t(j)];need(std::isfinite(out.at(i,j)),"normalized value overflow");}return out;
}
Matrix affine(const Matrix& x,const Matrix& w,const std::vector<double>& b){
  shape(w,x.d,int64_t(b.size()));finite(w.x);finite(b);Matrix out{x.n,w.d,std::vector<double>(size_t(x.n*w.d))};
  for(int64_t i=0;i<x.n;++i)for(int64_t j=0;j<w.d;++j){Sum s;for(int64_t k=0;k<x.d;++k)s.add(x.at(i,k)*w.at(k,j));s.add(b[size_t(j)]);need(std::isfinite(s.value),"affine overflow");out.x[size_t(i*w.d+j)]=s.value;}return out;
}
void compare(const Matrix& actual,const Matrix& expected,const std::string& why){shape(actual,expected.n,expected.d);for(size_t i=0;i<actual.x.size();++i)close(actual.x[i],expected.x[i],why);}
void ridge_equations(const Matrix& x,const Matrix& y,const std::vector<uint8_t>& eligible,const Matrix& weights,const std::vector<double>& bias,bool center_x){
  shape(y,x.n,weights.d);shape(weights,x.d,y.d);need(eligible.size()==size_t(x.n*y.d),"ridge eligibility shape");
  for(int64_t j=0;j<y.d;++j){int64_t count=0;Sum sy;std::vector<Sum> sx(size_t(x.d));
    for(int64_t i=0;i<x.n;++i)if(eligible[size_t(i*y.d+j)]){++count;sy.add(y.at(i,j));for(int64_t k=0;k<x.d;++k)sx[size_t(k)].add(x.at(i,k));}need(count>=2,"ridge requires supported TRAIN targets");const auto ym=sy.value/count;
    std::vector<double> xm(size_t(x.d),0);for(int64_t k=0;k<x.d;++k)if(center_x)xm[size_t(k)]=sx[size_t(k)].value/count;
    Sum intercept;intercept.add(ym);for(int64_t k=0;k<x.d;++k)intercept.add(-xm[size_t(k)]*weights.at(k,j));close(bias[size_t(j)],intercept.value,"ridge intercept equation");
    for(int64_t k=0;k<x.d;++k){Sum residual;residual.add(weights.at(k,j));double magnitude=std::abs(weights.at(k,j));
      for(int64_t i=0;i<x.n;++i)if(eligible[size_t(i*y.d+j)]){Sum prediction;for(int64_t q=0;q<x.d;++q)prediction.add((x.at(i,q)-xm[size_t(q)])*weights.at(q,j));const double term=(x.at(i,k)-xm[size_t(k)])*(prediction.value-(y.at(i,j)-ym));residual.add(term);magnitude+=std::abs(term);}
      need(std::isfinite(residual.value)&&std::isfinite(magnitude)&&std::abs(residual.value)<=1e-8+1e-8*magnitude,"ridge penalty1 normal equation");}
  }
}
struct Feature{Matrix values;std::vector<uint8_t> valid;std::vector<int64_t> ids;};
Feature regression_features(const fs::path& file){Archive a(file);auto x=matrix(a,"values");auto valid=boolean(a,"valid",x.n,1);return{x,valid,integers(a,"row_ids",x.n)};}
struct Targets{Matrix values;std::vector<uint8_t> valid;std::vector<int64_t> ids,groups;};
Targets targets(const fs::path& file){Archive a(file);auto y=matrix(a,"targets");return{y,mask_matrix(a,"valid",y.n,y.d),integers(a,"row_ids",y.n),integers(a,"group_ids",y.n)};}
std::string metric(const Matrix& p,const Targets& y,const std::vector<uint8_t>& valid,int64_t j){
  int64_t n=0,eligible=0;Sum mean,absolute,squared;for(int64_t i=0;i<y.values.n;++i){eligible+=y.valid[size_t(i*y.values.d+j)];if(valid[size_t(i*y.values.d+j)]){const double truth=y.values.at(i,j),prediction=p.at(i,j);need(std::isfinite(truth)&&std::isfinite(prediction),"finite scored scalar required");++n;mean.add(truth);const auto e=prediction-truth;absolute.add(std::abs(e));squared.add(e*e);}}
  Sum denominator;if(n>=2)for(int64_t i=0;i<y.values.n;++i)if(valid[size_t(i*y.values.d+j)]){const auto delta=y.values.at(i,j)-mean.value/n;denominator.add(delta*delta);}
  need(std::isfinite(absolute.value)&&std::isfinite(squared.value)&&std::isfinite(denominator.value)&&absolute.value>=0&&squared.value>=0&&denominator.value>=0,"finite metric reductions required");
  std::ostringstream out;out<<std::setprecision(17)<<"{\"total\":"<<y.values.n<<",\"target_eligible\":"<<eligible<<",\"valid\":"<<n<<",\"coverage\":"<<double(n)/y.values.n<<",\"mae\":";
  if(n)out<<absolute.value/n;else out<<"null";out<<",\"rmse\":";if(n)out<<std::sqrt(squared.value/n);else out<<"null";out<<",\"r2\":";if(n>=2&&denominator.value>0)out<<1-squared.value/denominator.value;else out<<"null";return out.str()+'}';
}
std::string regression(const std::vector<std::string>& p){
  need(p.size()==6,"regression plan has TRAINfeatures TRAINtargets FIT features targets predictions");const auto train=regression_features(p[0]),f=regression_features(p[3]);const auto ty=targets(p[1]),y=targets(p[4]);
  need(train.ids==ty.ids&&f.ids==y.ids&&train.values.d==f.values.d&&ty.values.d==y.values.d,"regression exact row/width association");Archive fit(p[2]),prediction(p[5]);const auto d=train.values.d,t=ty.values.d;
  const auto feature_stats=stats(train.values,train.valid);verify_stats(fit,feature_stats,"feature_mean","feature_scale");need(scalar_b(fit,"feature_training_supported")&&scalar_i(fit,"training_feature_rows")==feature_stats.count,"TRAIN feature fit count/support");
  const auto fm=vector(fit,"feature_mean",d),fscl=vector(fit,"feature_scale",d),tm=vector(fit,"target_mean",t),ts=vector(fit,"target_scale",t);const auto counts=integers(fit,"target_training_counts",t);const auto supported=boolean(fit,"supported_targets",t,1),target_supported=boolean(fit,"target_training_supported",t,1);
  Matrix standardized_targets{ty.values.n,t,std::vector<double>(ty.values.x.size(),0)};std::vector<uint8_t> eligible(size_t(ty.values.n*t),0);
  for(int64_t j=0;j<t;++j){Matrix one{ty.values.n,1,std::vector<double>(size_t(ty.values.n),0)};std::vector<uint8_t> mask(size_t(ty.values.n));int64_t combined=0;for(int64_t i=0;i<ty.values.n;++i){one.x[size_t(i)]=ty.values.at(i,j);mask[size_t(i)]=ty.valid[size_t(i*t+j)];combined+=mask[size_t(i)]&&train.valid[size_t(i)];}
    const auto s=stats(one,mask);close(tm[size_t(j)],s.mean[0],"TRAIN target mean");close(ts[size_t(j)],s.scale[0],"TRAIN target scale");need(counts[size_t(j)]==s.count&&target_supported[size_t(j)]&&supported[size_t(j)]&&combined>=2,"TRAIN target counts/support");
    for(int64_t i=0;i<ty.values.n;++i)if(mask[size_t(i)]){standardized_targets.x[size_t(i*t+j)]=(ty.values.at(i,j)-tm[size_t(j)])/ts[size_t(j)];eligible[size_t(i*t+j)]=train.valid[size_t(i)];}}
  const auto rw=matrix(fit,"ridge_weights"),w1=matrix(fit,"neural_w1"),w2=matrix(fit,"neural_w2");const auto rb=vector(fit,"ridge_intercept",t),b1=vector(fit,"neural_b1",16),b2=vector(fit,"neural_b2",t);shape(w1,d,16);shape(w2,16,t);shape(rw,d,t);
  ridge_equations(normalized(train.values,train.valid,fm,fscl),standardized_targets,eligible,rw,rb,true);
  auto x=normalized(f.values,f.valid,fm,fscl);auto ridge=affine(x,rw,rb),hidden=affine(x,w1,b1);for(auto& value:hidden.x)value=std::tanh(value);auto neural=affine(hidden,w2,b2);
  for(int64_t i=0;i<x.n;++i)for(int64_t j=0;j<t;++j){ridge.x[size_t(i*t+j)]=ridge.at(i,j)*ts[size_t(j)]+tm[size_t(j)];neural.x[size_t(i*t+j)]=neural.at(i,j)*ts[size_t(j)]+tm[size_t(j)];}
  compare(matrix(prediction,"ridge_predictions"),ridge,"saved ridge prediction");compare(matrix(prediction,"neural_predictions"),neural,"saved neural prediction");compare(matrix(prediction,"targets"),y.values,"saved target association");need(integers(prediction,"row_ids",x.n)==y.ids&&integers(prediction,"group_ids",x.n)==y.groups&&mask_matrix(prediction,"target_valid",x.n,t)==y.valid,"saved target mask/IDs");
  const auto valid=mask_matrix(prediction,"valid",x.n,t);for(int64_t i=0;i<x.n;++i)for(int64_t j=0;j<t;++j)need(valid[size_t(i*t+j)]==bool(f.valid[size_t(i)]&&y.valid[size_t(i*t+j)]&&supported[size_t(j)]),"prediction eligibility mismatch");
  std::ostringstream out;out<<"{\"kind\":\"regression\",\"fit\":"<<quoted(p[2])<<",\"predictions\":"<<quoted(p[5])<<",\"actual_seed_decimal\":"<<quoted(std::to_string(static_cast<uint64_t>(scalar_i(fit,"actual_seed"))))<<",\"tasks\":[";
  for(int64_t j=0;j<t;++j){if(j)out<<',';out<<"{\"ridge\":"<<metric(ridge,y,valid,j)<<",\"neural\":"<<metric(neural,y,valid,j)<<'}';}return out.str()+"]}";
}
struct Surface{Matrix values;std::vector<uint8_t> valid;std::vector<int64_t> labels;std::string ids;};
Surface surface(const fs::path& file){Archive a(file);auto x=matrix(a,"features");return{x,boolean(a,"valid",x.n,1),integers(a,"labels_scoring_only",x.n),text(a,"source_ids_json")};}
std::string classification(const std::vector<std::string>& p){
  need(p.size()==4,"classification plan has TRAINsurface FIT surface predictions");const auto train=surface(p[0]),f=surface(p[2]);Archive fit(p[1]),saved(p[3]);const auto d=train.values.d;need(d==f.values.d,"classification width");
  close(scalar_f(fit,"ridge_penalty"),1,"ridge penalty");need(scalar_i(fit,"tiny_hidden")==16&&scalar_i(fit,"tiny_steps")==100,"fixed tiny dimensions/steps");close(scalar_f(fit,"tiny_learning_rate"),.01,"tiny learning rate");need(text(fit,"training_source_ids_json")==train.ids,"exact TRAIN source IDs");
  const auto outer=scalar_b(fit,"outer_normalizer_applied");const auto fm=vector(fit,"feature_mean",d),fscl=vector(fit,"feature_scale",d);Matrix prepared=train.values,probe=f.values;
  const auto first=stats(train.values,train.valid);if(outer){verify_stats(fit,first,"feature_mean","feature_scale");need(scalar_i(fit,"outer_fitted_rows")==first.count,"outer TRAIN count");prepared=normalized(train.values,train.valid,fm,fscl);probe=normalized(f.values,f.valid,fm,fscl);}else{need(scalar_i(fit,"outer_fitted_rows")==0,"unexpected outer fit");for(int64_t j=0;j<d;++j){close(fm[size_t(j)],0,"identity outer mean");close(fscl[size_t(j)],1,"identity outer scale");}}
  const auto second=stats(prepared,train.valid);verify_stats(fit,second,"ridge_mean","ridge_scale");verify_stats(fit,second,"tiny_mean","tiny_scale");need(scalar_i(fit,"fitted_rows")==second.count,"inner TRAIN count");
  const auto rm=vector(fit,"ridge_mean",d),rs=vector(fit,"ridge_scale",d),nm=vector(fit,"tiny_mean",d),ns=vector(fit,"tiny_scale",d);const auto rw=matrix(fit,"ridge_weights"),w1=matrix(fit,"tiny_w1"),w2=matrix(fit,"tiny_w2");const auto rb=vector(fit,"ridge_intercept",2),b1=vector(fit,"tiny_b1",16),b2=vector(fit,"tiny_b2",2);shape(rw,d,2);shape(w1,d,16);shape(w2,16,2);
  Matrix onehot{train.values.n,2,std::vector<double>(size_t(train.values.n*2),0)};std::vector<uint8_t> eligible(size_t(train.values.n*2),0);int64_t class_rows[2]{0,0};for(int64_t i=0;i<train.values.n;++i){need(train.labels[size_t(i)]==0||train.labels[size_t(i)]==1,"binary TRAIN label");onehot.x[size_t(i*2+train.labels[size_t(i)])]=1;class_rows[train.labels[size_t(i)]]+=train.valid[size_t(i)];for(int64_t j=0;j<2;++j)eligible[size_t(i*2+j)]=train.valid[size_t(i)];}need(class_rows[0]>0&&class_rows[1]>0,"both binary classes required in supported TRAIN rows");
  ridge_equations(normalized(prepared,train.valid,rm,rs),onehot,eligible,rw,rb,false);
  auto ridge=affine(normalized(probe,f.valid,rm,rs),rw,rb),hidden=affine(normalized(probe,f.valid,nm,ns),w1,b1);compare(matrix(saved,"probe_input_features"),probe,"outer-normalized probe input");compare(matrix(saved,"ridge_logits"),ridge,"ridge logit arithmetic");compare(matrix(saved,"tiny_hidden_preactivation"),hidden,"tiny hidden arithmetic");for(auto& value:hidden.x)value=std::tanh(value);auto neural=affine(hidden,w2,b2);compare(matrix(saved,"tiny_logits"),neural,"tiny logit arithmetic");
  const auto actual_ridge=matrix(saved,"ridge_logits"),actual_neural=matrix(saved,"tiny_logits");const auto labels=integers(saved,"labels_scoring_only",f.values.n),ridge_class=integers(saved,"ridge",f.values.n),neural_class=integers(saved,"tiny_secondary",f.values.n);const auto valid=boolean(saved,"valid",f.values.n,1);need(valid==f.valid&&labels==f.labels&&text(saved,"source_ids_json")==f.ids,"classification labels/support/source association");
  int64_t n=0,rc=0,nc=0,ties=0;for(int64_t i=0;i<f.values.n;++i){need(labels[size_t(i)]==0||labels[size_t(i)]==1,"binary scoring label");const int64_t r=actual_ridge.at(i,1)>actual_ridge.at(i,0),q=actual_neural.at(i,1)>actual_neural.at(i,0);need(ridge_class[size_t(i)]==r&&neural_class[size_t(i)]==q,"first argmax tie rule");ties+=(actual_ridge.at(i,0)==actual_ridge.at(i,1))+(actual_neural.at(i,0)==actual_neural.at(i,1));if(valid[size_t(i)]){++n;rc+=r==labels[size_t(i)];nc+=q==labels[size_t(i)];}}
  need(n>0,"classification full abstention");std::ostringstream out;out<<std::setprecision(17)<<"{\"kind\":\"classification\",\"fit\":"<<quoted(p[1])<<",\"predictions\":"<<quoted(p[3])<<",\"actual_seed_decimal\":"<<quoted(seed_decimal(fit,"actual_probe_seed_decimal"))<<",\"total\":"<<f.values.n<<",\"valid\":"<<n<<",\"coverage\":"<<double(n)/f.values.n<<",\"ridge_accuracy\":"<<double(rc)/n<<",\"neural_accuracy\":"<<double(nc)/n<<",\"exact_saved_logit_ties\":"<<ties<<'}';return out.str();
}
struct Volume{std::vector<int64_t> dimensions;std::vector<double> values;};
Volume volume(Archive& archive,const std::string& key){
  auto value=archive.get(key);need(value.dim()==4&&value.scalar_type()==torch::kFloat32&&value.numel()>0&&value.numel()<=8000000,"bounded F32 four-dimensional witness: "+key);
  for(int64_t length:value.sizes())need(length>0&&length<=4096,"bounded witness axis");
  const auto promoted=value.to(torch::kFloat64).contiguous();const auto data=promoted.const_data_ptr<double>();
  return{value.sizes().vec(),{data,data+promoted.numel()}};
}
std::string waveform(const std::string& mode,const std::vector<std::string>& paths){
  need(paths.size()==1,"one explicit saved waveform witness required");Archive archive(paths[0]);
  const auto mean=volume(archive,"mean"),variance=volume(archive,"variance"),target=volume(archive,"targets");
  need(mean.dimensions==variance.dimensions&&mean.dimensions==target.dimensions,"witness distribution/target geometry");
  const auto mask=archive.get("target_mask");need(mask.scalar_type()==torch::kBool&&mask.sizes().vec()==mean.dimensions,"exact witness Bool target-mask shape");
  const auto support=mask.const_data_ptr<bool>();const auto B=mean.dimensions[0],C=mean.dimensions[1],T=mean.dimensions[2],F=mean.dimensions[3];
  std::vector<double> row_mae(size_t(B),0),row_nll(size_t(B),0),row_huber(size_t(B),0);std::vector<int64_t> row_cells(size_t(B),0),channel_cells(size_t(B*C),0);std::vector<uint8_t> eligible(size_t(B*C),0);
  Sum total_mae,total_nll,total_huber;int64_t valid_examples=0,total_cells=0;
  for(int64_t b=0;b<B;++b){Sum example_mae,example_nll,example_huber;int64_t channels=0;
    for(int64_t c=0;c<C;++c){Sum cell_mae,cell_nll,cell_huber;int64_t cells=0;
      for(int64_t t=0;t<T;++t)for(int64_t f=0;f<F;++f){const auto i=size_t(((b*C+c)*T+t)*F+f);if(!support[i])continue;
        const auto mu=mean.values[i],v=variance.values[i],truth=target.values[i];need(std::isfinite(mu)&&std::isfinite(v)&&v>0&&std::isfinite(truth),"finite supported waveform operands and positive variance required");
        const auto error=mu-truth,absolute=std::abs(error);const auto squared=error*error;
        const auto nll=.5*(squared/v+std::log(v)),huber=absolute<=1.?.5*squared:absolute-.5;
        need(std::isfinite(absolute)&&std::isfinite(nll)&&std::isfinite(huber),"finite independent waveform cell arithmetic required");
        cell_mae.add(absolute);cell_nll.add(nll);cell_huber.add(huber);++cells;
      }
      channel_cells[size_t(b*C+c)]=cells;
      if(cells){eligible[size_t(b*C+c)]=1;++channels;row_cells[size_t(b)]+=cells;example_mae.add(cell_mae.value/cells);example_nll.add(cell_nll.value/cells);example_huber.add(cell_huber.value/cells);}
    }
    if(channels){++valid_examples;row_mae[size_t(b)]=example_mae.value/channels;row_nll[size_t(b)]=example_nll.value/channels;row_huber[size_t(b)]=example_huber.value/channels;
      total_mae.add(row_mae[size_t(b)]);total_nll.add(row_nll[size_t(b)]);total_huber.add(row_huber[size_t(b)]);total_cells+=row_cells[size_t(b)];}
  }
  need(valid_examples>0&&total_cells>0,"nonvacuous waveform witness population required");
  const double mae=total_mae.value/valid_examples,nll=total_nll.value/valid_examples,huber=total_huber.value/valid_examples;
  if(mode=="fixed_query"){
    const auto saved_mae=vector(archive,"row_mae",B),saved_nll=vector(archive,"row_nll",B);
    for(int64_t b=0;b<B;++b){close(saved_mae[size_t(b)],row_mae[size_t(b)],"fixed-query row MAE");close(saved_nll[size_t(b)],row_nll[size_t(b)],"fixed-query row NLL");}
    torch::Tensor counts;
    if(archive.a.try_read("row_target_cells",counts,true)){
      need(counts.device().is_cpu()&&counts.scalar_type()==torch::kInt64&&counts.dim()==1&&counts.numel()==B,"fixed-query optional row-count shape");counts=counts.contiguous();
      for(int64_t b=0;b<B;++b)need(counts.const_data_ptr<int64_t>()[b]==row_cells[size_t(b)],"fixed-query row target count");
    }
    torch::Tensor per_channel_counts;
    if(archive.a.try_read("target_counts",per_channel_counts,true)){
      need(per_channel_counts.device().is_cpu()&&per_channel_counts.scalar_type()==torch::kInt64&&per_channel_counts.dim()==2&&per_channel_counts.size(0)==B&&per_channel_counts.size(1)==C,"fixed-query optional channel-count shape");per_channel_counts=per_channel_counts.contiguous();
      for(int64_t i=0;i<B*C;++i)need(per_channel_counts.const_data_ptr<int64_t>()[i]==channel_cells[size_t(i)],"fixed-query channel target count");
    }
  }else{
    need(mask_matrix(archive,"eligible_channels",B,C)==eligible,"actual-step eligible channels match target cells");
    close(scalar_f(archive,"reporting_nll"),nll,"actual-step reporting Gaussian NLL");
    if(mode=="objective_candidate"){
      close(scalar_f(archive,"mean_huber"),huber,"actual-step unweighted Huber1 mean");close(scalar_f(archive,"variance_nll"),nll,"actual-step detached-error variance NLL values");
      close(scalar_f(archive,"training_objective"),huber+nll,"actual-step unit Huber plus variance objective");
    }else close(scalar_f(archive,"training_objective"),nll,"control actual-step Gaussian objective");
  }
  std::ostringstream out;out<<std::setprecision(17)<<"{\"kind\":"<<quoted(mode)<<",\"witness\":"<<quoted(paths[0])<<",\"examples\":"<<B<<",\"valid_examples\":"<<valid_examples<<",\"target_cells\":"<<total_cells
    <<",\"mae\":"<<mae<<",\"gaussian_nll\":"<<nll<<",\"mean_huber\":"<<huber<<",\"arithmetic_only\":true,\"gradient_reconstruction\":false}";return out.str();
}
std::string run(const std::string& mode,const std::vector<std::string>& paths){if(mode=="regression")return regression(paths);if(mode=="classification")return classification(paths);if(mode=="objective_control"||mode=="objective_candidate"||mode=="fixed_query")return waveform(mode,paths);throw std::runtime_error("unknown audit mode");}
} // namespace independent

int main(int argc,char** argv){using namespace independent;fs::path output;bool output_admitted=false;try{
  need(fs::exists("/.dockerenv")&&fs::canonical("/mityane")=="/mityane","managed Linux /mityane container required");
  torch::NoGradGuard no_grad;torch::set_num_threads(1);torch::set_num_interop_threads(1);
  need(argc==5&&std::string(argv[1])=="--plan"&&std::string(argv[3])=="--output","audit --plan TSV --output NEWJSON");output=argv[4];
  need(output.is_absolute()&&output.lexically_normal()==output&&fs::canonical(output.parent_path())==output.parent_path()&&!fs::exists(output),"fresh canonical absolute output required");output_admitted=true;
  const fs::path plan_path(argv[2]);need(plan_path.is_absolute()&&fs::canonical(plan_path)==plan_path&&fs::is_regular_file(plan_path)&&fs::file_size(plan_path)<=16000000,"bounded canonical audit plan required");
  std::ifstream plan(plan_path);need(bool(plan),"existing explicit audit plan");std::vector<std::string> entries;std::string line;
  while(std::getline(plan,line)){need(!line.empty()&&line.size()<=65536&&line.find('\r')==std::string::npos&&line.find('\0')==std::string::npos,"nonempty bounded LF-only plan row");std::vector<std::string> fields;size_t begin=0;for(;;){const auto end=line.find('\t',begin);fields.push_back(line.substr(begin,end==std::string::npos?end:end-begin));if(end==std::string::npos)break;begin=end+1;}need(fields.size()>=2&&entries.size()<10000,"bounded plan row");const auto mode=fields.front();fields.erase(fields.begin());for(const auto& file:fields)need(fs::path(file).is_absolute()&&fs::canonical(file)==fs::path(file),"canonical absolute plan inputs");entries.push_back(run(mode,fields));}
  need(!entries.empty(),"nonempty actual audit population");std::ostringstream result;result<<std::setprecision(17)<<"{\"kind\":\"cpb_v15_independent_saved_fit_audit_v1\",\"status\":\"passed\",\"passed\":true,\"checks\":"<<checks<<",\"array_sets\":"<<entries.size()<<",\"absolute_tolerance\":"<<absolute_tolerance<<",\"relative_tolerance\":"<<relative_tolerance<<",\"model_forwards\":0,\"head_fits\":0,\"optimizer_replays\":0,\"bootstrap_intervals_audited\":false,\"entries\":[";for(size_t i=0;i<entries.size();++i){if(i)result<<',';result<<entries[i];}result<<"]}\n";write_new(output,result.str());std::cout<<"Saved-fit audit passed: "<<entries.size()<<" array sets, "<<checks<<" checks\n";return 0;
}catch(const std::exception& error){if(output_admitted&&!fs::exists(output))try{write_new(output,"{\"kind\":\"cpb_v15_independent_saved_fit_audit_v1\",\"status\":\"failed\",\"passed\":false,\"checks\":"+std::to_string(checks)+",\"error\":"+quoted(error.what())+"}\n");}catch(...){}std::cerr<<error.what()<<'\n';return 1;}}
