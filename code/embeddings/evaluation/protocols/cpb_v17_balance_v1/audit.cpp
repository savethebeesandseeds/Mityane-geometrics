// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Independent scalar ridge witness audit: no encoder, optimizer, Tensor solve,
// Tensor matrix arithmetic or shared regression/scoring implementation executes.
// Torch is used only for archive decoding and tensor shape/type inspection.
#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace fs=std::filesystem;
namespace {
constexpr size_t N=112,M=32,D=36,T=4;
constexpr long double absolute_tolerance=1e-9L,relative_tolerance=1e-8L;
using Values=std::vector<long double>;
std::string quote(const std::string &value){
  std::ostringstream out;out<<'"';for(const unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<c;
    else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}
  out<<'"';return out.str();
}
torch::Tensor read(torch::serialize::InputArchive &archive,const char *name){
  torch::Tensor value;archive.read(name,value,true);return value.contiguous();
}
std::string text(torch::serialize::InputArchive &archive,const char *name){
  const auto value=read(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kUInt8&&value.dim()==1,"[ridge audit] invalid text tensor: ",name);
  return std::string(reinterpret_cast<const char *>(value.const_data_ptr<uint8_t>()),static_cast<size_t>(value.numel()));
}
Values floating(torch::serialize::InputArchive &archive,const char *name,torch::IntArrayRef shape){
  const auto value=read(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kFloat64&&value.sizes()==shape,"[ridge audit] CPU F64 shape mismatch: ",name);
  Values out;out.reserve(value.numel());const auto *p=value.const_data_ptr<double>();
  for(int64_t i=0;i<value.numel();++i){TORCH_CHECK(std::isfinite(p[i]),"[ridge audit] nonfinite tensor: ",name);out.push_back(static_cast<long double>(p[i]));}
  return out;
}
void complete(torch::serialize::InputArchive &archive,const char *name,torch::IntArrayRef shape){
  const auto value=read(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kBool&&value.sizes()==shape,"[ridge audit] support shape mismatch: ",name);
  const auto *p=value.const_data_ptr<bool>();for(int64_t i=0;i<value.numel();++i)TORCH_CHECK(p[i],"[ridge audit] incomplete support: ",name);
}
std::vector<int64_t> ids(torch::serialize::InputArchive &archive,const char *name,int64_t count,bool unique){
  const auto value=read(archive,name);TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kInt64&&value.sizes()==torch::IntArrayRef({count}),"[ridge audit] ID shape mismatch: ",name);
  const auto *p=value.const_data_ptr<int64_t>();std::vector<int64_t> out(p,p+count);std::set<int64_t> seen;
  if(unique)for(const auto id:out)TORCH_CHECK(seen.insert(id).second,"[ridge audit] duplicate row ID");return out;
}
void same_bits(const Values &a,const Values &b,const char *name){
  TORCH_CHECK(a.size()==b.size(),"[ridge audit] equality size mismatch: ",name);
  for(size_t i=0;i<a.size();++i){const double x=static_cast<double>(a[i]),y=static_cast<double>(b[i]);
    TORCH_CHECK(std::memcmp(&x,&y,sizeof(double))==0,"[ridge audit] original legacy target bit mismatch: ",name," index ",i);}
}
struct Difference {size_t count{0};long double maximum_absolute{0},maximum_tolerance_fraction{0};};
Difference compare(const Values &actual,const Values &expected,const char *name){
  TORCH_CHECK(actual.size()==expected.size(),"[ridge audit] comparison size mismatch: ",name);Difference out;
  for(size_t i=0;i<actual.size();++i){
    TORCH_CHECK(std::isfinite(actual[i])&&std::isfinite(expected[i]),"[ridge audit] nonfinite arithmetic: ",name);
    const auto difference=std::abs(actual[i]-expected[i]);
    const auto bound=absolute_tolerance+relative_tolerance*std::max(std::abs(actual[i]),std::abs(expected[i]));
    ++out.count;out.maximum_absolute=std::max(out.maximum_absolute,difference);out.maximum_tolerance_fraction=std::max(out.maximum_tolerance_fraction,difference/bound);
    TORCH_CHECK(difference<=bound,"[ridge audit] mismatch: ",name," index ",i," absolute difference ",static_cast<double>(difference)," bound ",static_cast<double>(bound));
  }return out;
}
struct Standardized {Values mean,scale,values;};
Standardized standardize(const Values &values,size_t rows,size_t columns){
  Standardized out{Values(columns,0),Values(columns,0),Values(values.size(),0)};
  for(size_t j=0;j<columns;++j){
    for(size_t i=0;i<rows;++i)out.mean[j]+=values[i*columns+j];out.mean[j]/=static_cast<long double>(rows);
    for(size_t i=0;i<rows;++i){const auto delta=values[i*columns+j]-out.mean[j];out.scale[j]+=delta*delta;}
    out.scale[j]=std::max(1e-8L,std::sqrt(out.scale[j]/static_cast<long double>(rows)));
    for(size_t i=0;i<rows;++i)out.values[i*columns+j]=(values[i*columns+j]-out.mean[j])/out.scale[j];
  }return out;
}
Values column_mean(const Values &values,size_t rows,size_t columns){
  Values mean(columns,0);for(size_t j=0;j<columns;++j){for(size_t i=0;i<rows;++i)mean[j]+=values[i*columns+j];mean[j]/=static_cast<long double>(rows);}return mean;
}
// A is symmetric positive definite because the ridge penalty is exactly one.
Values cholesky_solve(const Values &a,const Values &b){
  Values lower(D*D,0),solution(D*T,0),forward(D*T,0);
  for(size_t i=0;i<D;++i)for(size_t j=0;j<=i;++j){
    auto value=a[i*D+j];for(size_t k=0;k<j;++k)value-=lower[i*D+k]*lower[j*D+k];
    if(i==j){TORCH_CHECK(std::isfinite(value)&&value>0,"[ridge audit] Cholesky nonpositive pivot");lower[i*D+j]=std::sqrt(value);}
    else lower[i*D+j]=value/lower[j*D+j];
  }
  for(size_t t=0;t<T;++t){
    for(size_t i=0;i<D;++i){auto value=b[i*T+t];for(size_t k=0;k<i;++k)value-=lower[i*D+k]*forward[k*T+t];forward[i*T+t]=value/lower[i*D+i];}
    for(size_t backwards=D;backwards>0;--backwards){const size_t i=backwards-1;auto value=forward[i*T+t];for(size_t k=i+1;k<D;++k)value-=lower[k*D+i]*solution[k*T+t];solution[i*T+t]=value/lower[i*D+i];}
  }return solution;
}
Values metrics(const Values &prediction,const Values &target){
  Values out(T*4,0);for(size_t j=0;j<T;++j){
    long double mean=0,absolute=0,squared=0,denominator=0;
    for(size_t i=0;i<M;++i)mean+=target[i*T+j];mean/=static_cast<long double>(M);
    for(size_t i=0;i<M;++i){const auto delta=prediction[i*T+j]-target[i*T+j],centered=target[i*T+j]-mean;absolute+=std::abs(delta);squared+=delta*delta;denominator+=centered*centered;}
    TORCH_CHECK(std::isfinite(denominator)&&denominator>0,"[ridge audit] unsupported zero-variance validation target");
    out[j*4]=1-squared/denominator;out[j*4+1]=absolute/M;out[j*4+2]=std::sqrt(squared/M);out[j*4+3]=1;
  }return out;
}
std::string artifact(const fs::path &path,const std::string &sha){
  std::ostringstream out;out<<"{\"path\":"<<quote(path.generic_string())<<",\"bytes\":"<<fs::file_size(path)<<",\"sha256\":"<<quote(sha)<<'}';return out.str();
}
int run(int argc,char **argv){
  TORCH_CHECK(argc==15,"[ridge audit] seven explicit flag/value pairs required");
  const std::set<std::string> allowed{"--witness","--witness-sha256","--expected-train-targets","--expected-train-targets-sha256","--expected-validation-targets","--expected-validation-targets-sha256","--output"};
  std::map<std::string,std::string> a;for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&a.emplace(argv[i],argv[i+1]).second,"[ridge audit] unknown or duplicate option");
  for(const auto &key:allowed)TORCH_CHECK(a.count(key),"[ridge audit] missing option: ",key);
  const fs::path out=a.at("--output");TORCH_CHECK(!fs::exists(out)&&fs::is_directory(out.parent_path()),"[ridge audit] new output leaf with existing parent required");
  std::map<std::string,std::string> hashes;
  for(const auto key:{"--witness","--expected-train-targets","--expected-validation-targets"}){
    const auto &expected=a.at(std::string(key)+"-sha256");TORCH_CHECK(expected.size()==64&&expected.find_first_not_of("0123456789abcdef")==std::string::npos,"[ridge audit] lowercase SHA256 required");
    const auto hash=cp::boundary::sha256_file(a.at(key));TORCH_CHECK(hash==expected,"[ridge audit] input SHA256 mismatch: ",key);hashes.emplace(key,hash);
  }
  const auto executable=fs::canonical("/proc/self/exe");const auto executable_sha=cp::boundary::sha256_file(executable);
  torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard guard;
  torch::serialize::InputArchive witness,original_train,original_validation;
  witness.load_from(a.at("--witness"),torch::kCPU);original_train.load_from(a.at("--expected-train-targets"),torch::kCPU);original_validation.load_from(a.at("--expected-validation-targets"),torch::kCPU);
  TORCH_CHECK(text(witness,"artifact_kind")=="cpb_v17_ridge_witness_v1"&&text(witness,"metric_columns")=="r2,mae,rmse,coverage","[ridge audit] unsupported witness schema");
  TORCH_CHECK(text(witness,"expected_train_targets_sha256")==hashes.at("--expected-train-targets")&&text(witness,"expected_validation_targets_sha256")==hashes.at("--expected-validation-targets"),"[ridge audit] witness/legacy artifact association mismatch");
  const auto train=floating(witness,"train_features",{N,D}),validation=floating(witness,"validation_features",{M,D});
  const auto train_target=floating(witness,"train_targets",{N,T}),validation_target=floating(witness,"validation_targets",{M,T});
  complete(witness,"train_target_valid",{N,T});complete(witness,"validation_target_valid",{M,T});complete(witness,"train_feature_valid",{N});complete(witness,"validation_feature_valid",{M});
  complete(original_train,"valid",{N,T});complete(original_validation,"valid",{M,T});
  same_bits(train_target,floating(original_train,"targets",{N,T}),"TRAIN");same_bits(validation_target,floating(original_validation,"targets",{M,T}),"validation");
  const auto train_ids=ids(witness,"train_row_ids",N,true),validation_ids=ids(witness,"validation_row_ids",M,true),train_groups=ids(witness,"train_group_ids",N,false),validation_groups=ids(witness,"validation_group_ids",M,false);
  TORCH_CHECK(train_ids==ids(original_train,"row_ids",N,true)&&validation_ids==ids(original_validation,"row_ids",M,true)&&train_groups==ids(original_train,"group_ids",N,false)&&validation_groups==ids(original_validation,"group_ids",M,false),"[ridge audit] original legacy row/group association mismatch");
  const std::set<int64_t> training_ids(train_ids.begin(),train_ids.end()),training_groups(train_groups.begin(),train_groups.end());
  for(size_t i=0;i<M;++i)TORCH_CHECK(!training_ids.count(validation_ids[i])&&!training_groups.count(validation_groups[i]),"[ridge audit] TRAIN/validation row or group overlap");
  const auto x=standardize(train,N,D),y=standardize(train_target,N,T);
  const auto xm=column_mean(x.values,N,D),ym=column_mean(y.values,N,T);
  for(size_t j=0;j<T;++j){long double variance=0;for(size_t i=0;i<N;++i){const auto delta=train_target[i*T+j]-y.mean[j];variance+=delta*delta;}TORCH_CHECK(variance>0,"[ridge audit] zero TRAIN target variance abstains");}
  Values matrix(D*D,0),rhs(D*T,0);
  for(size_t j=0;j<D;++j){
    for(size_t k=0;k<D;++k){long double sum=j==k?1:0;for(size_t i=0;i<N;++i)sum+=(x.values[i*D+j]-xm[j])*(x.values[i*D+k]-xm[k]);matrix[j*D+k]=sum;}
    for(size_t t=0;t<T;++t){long double sum=0;for(size_t i=0;i<N;++i)sum+=(x.values[i*D+j]-xm[j])*(y.values[i*T+t]-ym[t]);rhs[j*T+t]=sum;}
  }
  const auto weights=cholesky_solve(matrix,rhs);Values intercept(T,0),prediction(M*T,0);
  for(size_t t=0;t<T;++t){intercept[t]=ym[t];for(size_t j=0;j<D;++j)intercept[t]-=xm[j]*weights[j*T+t];}
  for(size_t i=0;i<M;++i)for(size_t t=0;t<T;++t){auto sum=intercept[t];for(size_t j=0;j<D;++j)sum+=(validation[i*D+j]-x.mean[j])/x.scale[j]*weights[j*T+t];prediction[i*T+t]=sum*y.scale[t]+y.mean[t];}
  const auto score=metrics(prediction,validation_target),saved_prediction=floating(witness,"predictions",{M,T});
  std::map<std::string,Difference> differences;
  const auto check=[&](const char *name,const Values &actual,torch::IntArrayRef shape){differences.emplace(name,compare(actual,floating(witness,name,shape),name));};
  check("feature_mean",x.mean,{D});check("feature_scale",x.scale,{D});check("target_mean",y.mean,{T});check("target_scale",y.scale,{T});
  check("weights",weights,{D,T});check("intercept",intercept,{T});check("predictions",prediction,{M,T});check("reported_metrics",score,{T,4});
  differences.emplace("saved_prediction_metrics",compare(metrics(saved_prediction,validation_target),floating(witness,"reported_metrics",{T,4}),"saved_prediction_metrics"));
  for(const auto &item:hashes)TORCH_CHECK(cp::boundary::sha256_file(a.at(item.first))==item.second,"[ridge audit] input bytes changed: ",item.first);
  TORCH_CHECK(cp::boundary::sha256_file(executable)==executable_sha,"[ridge audit] executable changed");
  TORCH_CHECK(fs::create_directory(out),"[ridge audit] exclusive output leaf required");
  std::ofstream report(out/"summary.json",std::ios::binary);TORCH_CHECK(report,"[ridge audit] cannot open report");
  report<<std::setprecision(21)<<"{\"kind\":\"cpb_v17_independent_scalar_ridge_audit_v1\",\"status\":\"passed\",\"arithmetic\":\"scalar-long-double-centered-normal-equations-Cholesky\",\"torch_used_for\":\"archive-codec-and-shape-type-inspection-only\",\"long_double_digits\":"<<std::numeric_limits<long double>::digits<<",\"absolute_tolerance\":"<<absolute_tolerance<<",\"relative_tolerance\":"<<relative_tolerance<<",\"training_rows\":112,\"validation_rows\":32,\"feature_width\":36,\"targets\":4,\"ridge_penalty\":1,\"finite_complete_support\":true,\"original_legacy_target_bits_equal\":true,\"original_row_group_association_equal\":true,\"encoder_executions\":0,\"encoder_updates\":0,\"differences\":{";
  bool comma=false;for(const auto &item:differences){if(comma)report<<',';comma=true;report<<quote(item.first)<<":{\"compared_values\":"<<item.second.count<<",\"maximum_absolute\":"<<item.second.maximum_absolute<<",\"maximum_tolerance_fraction\":"<<item.second.maximum_tolerance_fraction<<'}';}
  report<<"},\"independent_metrics\":[";const char *names[]{"past_volatility","past_trend","past_activity","past_relation"};
  for(size_t t=0;t<T;++t){if(t)report<<',';report<<"{\"task\":"<<quote(names[t])<<",\"r2\":"<<score[t*4]<<",\"mae\":"<<score[t*4+1]<<",\"rmse\":"<<score[t*4+2]<<",\"coverage\":1,\"valid\":32,\"total\":32}";}
  report<<"],\"inputs\":{";comma=false;for(const auto &item:hashes){if(comma)report<<',';comma=true;report<<quote(item.first.substr(2))<<':'<<artifact(a.at(item.first),item.second);}
  report<<"},\"executable\":"<<artifact(executable,executable_sha)<<",\"input_bytes_unchanged\":true,\"fresh_confirmation\":false}\n";report.close();TORCH_CHECK(report,"[ridge audit] report write failed");
  std::cout<<"Independent scalar ridge coefficients, predictions, metrics and legacy target associations passed.\n";return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
