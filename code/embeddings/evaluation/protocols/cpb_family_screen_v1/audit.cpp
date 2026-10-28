// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Dimension-generic scalar ridge witness audit. The audit path does not run an
// encoder, an optimizer, or a Tensor solve. Torch is the archive codec and the
// shape/type inspector. The self-test also compares one float64 torch
// linalg_solve with the scalar Cholesky fit.
#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace fs=std::filesystem;
namespace {
constexpr size_t kTargets=4;
constexpr long double absolute_tolerance=1e-9L,relative_tolerance=1e-8L;
using Values=std::vector<long double>;
std::string quote(const std::string &value){
  std::ostringstream out;out<<'"';
  for(const unsigned char c:value){
    if(c=='"'||c=='\\')out<<'\\'<<c;
    else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<c;
  }
  out<<'"';return out.str();
}
torch::Tensor read_tensor(torch::serialize::InputArchive &archive,const char *name){
  torch::Tensor value;archive.read(name,value,true);return value.contiguous();
}
std::string text(torch::serialize::InputArchive &archive,const char *name){
  const auto value=read_tensor(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kUInt8&&value.dim()==1,"[family screen audit] invalid text tensor: ",name);
  return std::string(reinterpret_cast<const char *>(value.const_data_ptr<uint8_t>()),static_cast<size_t>(value.numel()));
}
int64_t scalar_i64(torch::serialize::InputArchive &archive,const char *name){
  const auto value=read_tensor(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kInt64&&value.dim()==0,"[family screen audit] I64 scalar required: ",name);
  return value.item<int64_t>();
}
Values floating(torch::serialize::InputArchive &archive,const char *name,torch::IntArrayRef shape){
  const auto value=read_tensor(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kFloat64&&value.sizes()==shape,"[family screen audit] CPU F64 shape mismatch: ",name);
  Values out;out.reserve(static_cast<size_t>(value.numel()));const auto *pointer=value.const_data_ptr<double>();
  for(int64_t i=0;i<value.numel();++i){TORCH_CHECK(std::isfinite(pointer[i]),"[family screen audit] nonfinite tensor: ",name);out.push_back(static_cast<long double>(pointer[i]));}
  return out;
}
void complete(torch::serialize::InputArchive &archive,const char *name,torch::IntArrayRef shape){
  const auto value=read_tensor(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kBool&&value.sizes()==shape,"[family screen audit] support shape mismatch: ",name);
  const auto *pointer=value.const_data_ptr<bool>();
  for(int64_t i=0;i<value.numel();++i)TORCH_CHECK(pointer[i],"[family screen audit] incomplete support: ",name);
}
std::vector<int64_t> ids(torch::serialize::InputArchive &archive,const char *name,int64_t count){
  const auto value=read_tensor(archive,name);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kInt64&&value.sizes()==torch::IntArrayRef({count}),"[family screen audit] ID shape mismatch: ",name);
  const auto *pointer=value.const_data_ptr<int64_t>();std::vector<int64_t> out(pointer,pointer+count);std::set<int64_t> seen;
  for(const auto id:out)TORCH_CHECK(seen.insert(id).second,"[family screen audit] duplicate row ID: ",name);
  return out;
}
struct Difference {size_t count{0};long double maximum_absolute{0},maximum_tolerance_fraction{0};};
Difference compare(const Values &actual,const Values &expected,const char *name){
  TORCH_CHECK(actual.size()==expected.size(),"[family screen audit] comparison size mismatch: ",name);Difference out;
  for(size_t i=0;i<actual.size();++i){
    TORCH_CHECK(std::isfinite(static_cast<double>(actual[i]))&&std::isfinite(static_cast<double>(expected[i])),"[family screen audit] nonfinite arithmetic: ",name);
    const auto difference=std::abs(actual[i]-expected[i]);
    const auto bound=absolute_tolerance+relative_tolerance*std::max(std::abs(actual[i]),std::abs(expected[i]));
    ++out.count;out.maximum_absolute=std::max(out.maximum_absolute,difference);out.maximum_tolerance_fraction=std::max(out.maximum_tolerance_fraction,difference/bound);
    TORCH_CHECK(difference<=bound,"[family screen audit] mismatch: ",name," index ",i," absolute difference ",static_cast<double>(difference)," bound ",static_cast<double>(bound));
  }
  return out;
}
struct Standardized {Values mean,scale,values;};
Standardized standardize(const Values &values,size_t rows,size_t columns){
  Standardized out{Values(columns,0),Values(columns,0),Values(values.size(),0)};
  for(size_t column=0;column<columns;++column){
    for(size_t row=0;row<rows;++row)out.mean[column]+=values[row*columns+column];
    out.mean[column]/=static_cast<long double>(rows);
    for(size_t row=0;row<rows;++row){const auto delta=values[row*columns+column]-out.mean[column];out.scale[column]+=delta*delta;}
    out.scale[column]=std::max(1e-8L,std::sqrt(out.scale[column]/static_cast<long double>(rows)));
    for(size_t row=0;row<rows;++row)out.values[row*columns+column]=(values[row*columns+column]-out.mean[column])/out.scale[column];
  }
  return out;
}
Values column_mean(const Values &values,size_t rows,size_t columns){
  Values mean(columns,0);
  for(size_t column=0;column<columns;++column){for(size_t row=0;row<rows;++row)mean[column]+=values[row*columns+column];mean[column]/=static_cast<long double>(rows);}
  return mean;
}
// A is symmetric positive definite because the ridge penalty is exactly one.
Values cholesky_solve(const Values &matrix,const Values &right_hand,size_t width,size_t targets){
  Values lower(width*width,0),solution(width*targets,0),forward(width*targets,0);
  for(size_t row=0;row<width;++row)for(size_t column=0;column<=row;++column){
    auto value=matrix[row*width+column];
    for(size_t k=0;k<column;++k)value-=lower[row*width+k]*lower[column*width+k];
    if(row==column){TORCH_CHECK(std::isfinite(static_cast<double>(value))&&value>0,"[family screen audit] Cholesky nonpositive pivot");lower[row*width+column]=std::sqrt(value);}
    else lower[row*width+column]=value/lower[column*width+column];
  }
  for(size_t target=0;target<targets;++target){
    for(size_t row=0;row<width;++row){
      auto value=right_hand[row*targets+target];
      for(size_t k=0;k<row;++k)value-=lower[row*width+k]*forward[k*targets+target];
      forward[row*targets+target]=value/lower[row*width+row];
    }
    for(size_t backwards=width;backwards>0;--backwards){
      const size_t row=backwards-1;auto value=forward[row*targets+target];
      for(size_t k=row+1;k<width;++k)value-=lower[k*width+row]*solution[k*targets+target];
      solution[row*targets+target]=value/lower[row*width+row];
    }
  }
  return solution;
}
Values score_metrics(const Values &prediction,const Values &target,size_t rows,size_t targets){
  Values out(targets*4,0);
  for(size_t column=0;column<targets;++column){
    long double mean=0,absolute=0,squared=0,denominator=0;
    for(size_t row=0;row<rows;++row)mean+=target[row*targets+column];
    mean/=static_cast<long double>(rows);
    for(size_t row=0;row<rows;++row){
      const auto delta=prediction[row*targets+column]-target[row*targets+column],centered=target[row*targets+column]-mean;
      absolute+=std::abs(delta);squared+=delta*delta;denominator+=centered*centered;
    }
    TORCH_CHECK(std::isfinite(static_cast<double>(denominator))&&denominator>0,"[family screen audit] unsupported zero-variance validation target");
    out[column*4]=1-squared/denominator;out[column*4+1]=absolute/static_cast<long double>(rows);
    out[column*4+2]=std::sqrt(squared/static_cast<long double>(rows));out[column*4+3]=1;
  }
  return out;
}
struct Fit {Values feature_mean,feature_scale,target_mean,target_scale,weights,intercept,predictions,metrics;};
Fit fit_ridge(const Values &train,const Values &train_target,const Values &validation,const Values &validation_target,size_t train_rows,size_t validation_rows,size_t width){
  const auto features=standardize(train,train_rows,width),targets=standardize(train_target,train_rows,kTargets);
  const auto feature_center=column_mean(features.values,train_rows,width),target_center=column_mean(targets.values,train_rows,kTargets);
  for(size_t column=0;column<kTargets;++column){
    long double variance=0;
    for(size_t row=0;row<train_rows;++row){const auto delta=train_target[row*kTargets+column]-targets.mean[column];variance+=delta*delta;}
    TORCH_CHECK(variance>0,"[family screen audit] zero TRAIN target variance abstains");
  }
  Values matrix(width*width,0),right_hand(width*kTargets,0);
  for(size_t column=0;column<width;++column){
    for(size_t other=0;other<width;++other){
      long double sum=column==other?1:0;
      for(size_t row=0;row<train_rows;++row)sum+=(features.values[row*width+column]-feature_center[column])*(features.values[row*width+other]-feature_center[other]);
      matrix[column*width+other]=sum;
    }
    for(size_t target=0;target<kTargets;++target){
      long double sum=0;
      for(size_t row=0;row<train_rows;++row)sum+=(features.values[row*width+column]-feature_center[column])*(targets.values[row*kTargets+target]-target_center[target]);
      right_hand[column*kTargets+target]=sum;
    }
  }
  const auto weights=cholesky_solve(matrix,right_hand,width,kTargets);
  Values intercept(kTargets,0),predictions(validation_rows*kTargets,0);
  for(size_t target=0;target<kTargets;++target){
    intercept[target]=target_center[target];
    for(size_t column=0;column<width;++column)intercept[target]-=feature_center[column]*weights[column*kTargets+target];
  }
  for(size_t row=0;row<validation_rows;++row)for(size_t target=0;target<kTargets;++target){
    auto sum=intercept[target];
    for(size_t column=0;column<width;++column)sum+=(validation[row*width+column]-features.mean[column])/features.scale[column]*weights[column*kTargets+target];
    predictions[row*kTargets+target]=sum*targets.scale[target]+targets.mean[target];
  }
  return {features.mean,features.scale,targets.mean,targets.scale,weights,intercept,predictions,score_metrics(predictions,validation_target,validation_rows,kTargets)};
}
std::string artifact(const fs::path &path,const std::string &sha){
  std::ostringstream out;out<<"{\"path\":"<<quote(path.generic_string())<<",\"bytes\":"<<fs::file_size(path)<<",\"sha256\":"<<quote(sha)<<'}';
  return out.str();
}
torch::Tensor text_tensor(const std::string &value){return torch::tensor(std::vector<uint8_t>(value.begin(),value.end()),torch::kUInt8);}
torch::Tensor f64_tensor(const Values &values,torch::IntArrayRef shape){
  auto out=torch::empty(shape,torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  TORCH_CHECK(static_cast<size_t>(out.numel())==values.size(),"[family screen audit] tensor size mismatch");
  auto *pointer=out.data_ptr<double>();
  for(size_t i=0;i<values.size();++i){TORCH_CHECK(std::isfinite(static_cast<double>(values[i])));pointer[i]=static_cast<double>(values[i]);}
  return out;
}
torch::Tensor row_ids(int64_t count,int64_t first){
  auto out=torch::empty({count},torch::kInt64);auto *pointer=out.data_ptr<int64_t>();
  for(int64_t i=0;i<count;++i)pointer[i]=first+i;
  return out;
}
Values as_values(const torch::Tensor &tensor){
  const auto cpu=tensor.contiguous().to(torch::kCPU,torch::kFloat64);Values out;out.reserve(static_cast<size_t>(cpu.numel()));
  const auto *pointer=cpu.const_data_ptr<double>();
  for(int64_t i=0;i<cpu.numel();++i)out.push_back(static_cast<long double>(pointer[i]));
  return out;
}
void torch_ridge_agrees(const Values &train,const Values &train_target,const Values &validation,const Fit &fit,int64_t train_rows,int64_t validation_rows,int64_t width){
  const auto train_features=f64_tensor(train,{train_rows,width}),validation_features=f64_tensor(validation,{validation_rows,width});
  const auto targets=f64_tensor(train_target,{train_rows,static_cast<int64_t>(kTargets)});
  const auto mean=train_features.mean(0),scale=(train_features-mean).square().mean(0).sqrt().clamp_min(1e-8);
  const auto standardized=(train_features-mean)/scale,validation_standardized=(validation_features-mean)/scale;
  const auto target_mean=targets.mean(0),target_scale=(targets-target_mean).square().mean(0).sqrt().clamp_min(1e-8);
  const auto standardized_targets=(targets-target_mean)/target_scale;
  const auto feature_center=standardized.mean(0),target_center=standardized_targets.mean(0),centered=standardized-feature_center;
  const auto weights=at::linalg_solve(centered.transpose(0,1).matmul(centered)+torch::eye(width,train_features.options()),
      centered.transpose(0,1).matmul(standardized_targets-target_center));
  const auto intercept=target_center-feature_center.matmul(weights);
  const auto predictions=(validation_standardized.matmul(weights)+intercept)*target_scale+target_mean;
  compare(fit.feature_mean,as_values(mean),"torch feature mean");compare(fit.feature_scale,as_values(scale),"torch feature scale");
  compare(fit.weights,as_values(weights),"torch weights");compare(fit.intercept,as_values(intercept),"torch intercept");
  compare(fit.predictions,as_values(predictions),"torch predictions");
}
void unit_checks(){
  const auto weights=cholesky_solve(Values{2,0,0,2},Values{2,0},2,1);
  TORCH_CHECK(weights.size()==2&&std::abs(weights[0]-1)<=1e-12L&&std::abs(weights[1])<=1e-12L,"[family screen audit] Cholesky unit check failed");
  const auto metrics=score_metrics(Values{1,3},Values{1,3},2,1);
  TORCH_CHECK(metrics.size()==4&&metrics[0]==1&&metrics[1]==0&&metrics[2]==0&&metrics[3]==1,"[family screen audit] perfect-prediction metric check failed");
}
fs::path write_synthetic_witness(const fs::path &scratch){
  constexpr size_t train_rows=8,validation_rows=4,width=3;
  Values train(train_rows*width),validation(validation_rows*width),train_target(train_rows*kTargets),validation_target(validation_rows*kTargets);
  for(size_t row=0;row<train_rows;++row)for(size_t column=0;column<width;++column)train[row*width+column]=static_cast<long double>((row+1)*(column+1))*0.25L;
  for(size_t row=0;row<validation_rows;++row)for(size_t column=0;column<width;++column)validation[row*width+column]=static_cast<long double>(row+1)*0.5L+static_cast<long double>(column);
  for(size_t row=0;row<train_rows;++row)for(size_t target=0;target<kTargets;++target)train_target[row*kTargets+target]=static_cast<long double>(row+1)+static_cast<long double>(target)*0.5L;
  for(size_t row=0;row<validation_rows;++row)for(size_t target=0;target<kTargets;++target)validation_target[row*kTargets+target]=static_cast<long double>(row+1)*0.5L+static_cast<long double>(target);
  const auto fit=fit_ridge(train,train_target,validation,validation_target,train_rows,validation_rows,width);
  torch_ridge_agrees(train,train_target,validation,fit,static_cast<int64_t>(train_rows),static_cast<int64_t>(validation_rows),static_cast<int64_t>(width));
  const auto training_ids=row_ids(static_cast<int64_t>(train_rows),1),validation_ids=row_ids(static_cast<int64_t>(validation_rows),101);
  torch::serialize::OutputArchive archive;
  archive.write("artifact_kind",text_tensor("cpb_family_screen_ridge_witness_v1"),true);
  archive.write("metric_columns",text_tensor("r2,mae,rmse,coverage"),true);
  archive.write("recipe",text_tensor("synthetic-self-test"),true);
  archive.write("group_identity",text_tensor("row-id"),true);
  archive.write("native_width",torch::tensor(static_cast<int64_t>(width),torch::kInt64),true);
  archive.write("learned_width",torch::tensor(int64_t{2},torch::kInt64),true);
  archive.write("spectral_width",torch::tensor(int64_t{0},torch::kInt64),true);
  archive.write("change_width",torch::tensor(int64_t{1},torch::kInt64),true);
  archive.write("causal_updates",torch::tensor(int64_t{7},torch::kInt64),true);
  archive.write("train_features",f64_tensor(train,{8,3}),true);
  archive.write("validation_features",f64_tensor(validation,{4,3}),true);
  archive.write("train_targets",f64_tensor(train_target,{8,4}),true);
  archive.write("validation_targets",f64_tensor(validation_target,{4,4}),true);
  archive.write("train_target_valid",torch::ones({8,4},torch::kBool),true);
  archive.write("validation_target_valid",torch::ones({4,4},torch::kBool),true);
  archive.write("train_feature_valid",torch::ones({8},torch::kBool),true);
  archive.write("validation_feature_valid",torch::ones({4},torch::kBool),true);
  archive.write("train_row_ids",training_ids,true);
  archive.write("validation_row_ids",validation_ids,true);
  archive.write("train_group_ids",training_ids.clone(),true);
  archive.write("validation_group_ids",validation_ids.clone(),true);
  archive.write("feature_mean",f64_tensor(fit.feature_mean,{3}),true);
  archive.write("feature_scale",f64_tensor(fit.feature_scale,{3}),true);
  archive.write("target_mean",f64_tensor(fit.target_mean,{4}),true);
  archive.write("target_scale",f64_tensor(fit.target_scale,{4}),true);
  archive.write("weights",f64_tensor(fit.weights,{3,4}),true);
  archive.write("intercept",f64_tensor(fit.intercept,{4}),true);
  archive.write("predictions",f64_tensor(fit.predictions,{4,4}),true);
  archive.write("reported_metrics",f64_tensor(fit.metrics,{4,4}),true);
  const auto path=scratch/"ridge-witness.pt";archive.save_to(path.string());return path;
}
int audit_witness(const fs::path &witness_path,const std::string &expected_sha,const fs::path &out,bool self_test){
  TORCH_CHECK(expected_sha.size()==64&&expected_sha.find_first_not_of("0123456789abcdef")==std::string::npos,"[family screen audit] lowercase SHA256 required");
  TORCH_CHECK(!fs::exists(out)&&fs::is_directory(out.parent_path()),"[family screen audit] new output leaf with existing parent required");
  const auto witness_sha=cp::boundary::sha256_file(witness_path);
  TORCH_CHECK(witness_sha==expected_sha,"[family screen audit] witness SHA256 mismatch");
  const auto executable=fs::canonical("/proc/self/exe");const auto executable_sha=cp::boundary::sha256_file(executable);
  torch::serialize::InputArchive witness;witness.load_from(witness_path.string(),torch::kCPU);
  const auto recipe=text(witness,"recipe");
  TORCH_CHECK(!recipe.empty()&&recipe.size()<=4096,"[family screen audit] recipe text required");
  for(const unsigned char c:recipe)TORCH_CHECK(c>=32&&c<127,"[family screen audit] recipe text must be printable ASCII");
  TORCH_CHECK(text(witness,"artifact_kind")=="cpb_family_screen_ridge_witness_v1"&&text(witness,"metric_columns")=="r2,mae,rmse,coverage"&&
      text(witness,"group_identity")=="row-id","[family screen audit] unsupported witness schema");
  const auto native_width=scalar_i64(witness,"native_width"),learned_width=scalar_i64(witness,"learned_width");
  const auto spectral_width=scalar_i64(witness,"spectral_width"),change_width=scalar_i64(witness,"change_width"),causal_updates=scalar_i64(witness,"causal_updates");
  int64_t train_rows=0,validation_rows=0,width=0;
  {
    const auto train_features_tensor=read_tensor(witness,"train_features"),validation_features_tensor=read_tensor(witness,"validation_features");
    TORCH_CHECK(train_features_tensor.dim()==2&&validation_features_tensor.dim()==2&&train_features_tensor.size(0)>0&&validation_features_tensor.size(0)>0&&
        train_features_tensor.size(1)>0&&train_features_tensor.size(1)==validation_features_tensor.size(1),"[family screen audit] feature matrices required");
    train_rows=train_features_tensor.size(0);validation_rows=validation_features_tensor.size(0);width=train_features_tensor.size(1);
  }
  TORCH_CHECK(native_width==width&&learned_width>0&&spectral_width>=0&&change_width>0&&learned_width+spectral_width+change_width==native_width&&causal_updates>=0,
      "[family screen audit] native/learned/spectral/change layout or update count invalid");
  const auto train=floating(witness,"train_features",{train_rows,width}),validation=floating(witness,"validation_features",{validation_rows,width});
  const auto train_target=floating(witness,"train_targets",{train_rows,static_cast<int64_t>(kTargets)}),validation_target=floating(witness,"validation_targets",{validation_rows,static_cast<int64_t>(kTargets)});
  complete(witness,"train_target_valid",{train_rows,static_cast<int64_t>(kTargets)});
  complete(witness,"validation_target_valid",{validation_rows,static_cast<int64_t>(kTargets)});
  complete(witness,"train_feature_valid",{train_rows});complete(witness,"validation_feature_valid",{validation_rows});
  const auto train_ids=ids(witness,"train_row_ids",train_rows),validation_ids=ids(witness,"validation_row_ids",validation_rows);
  TORCH_CHECK(train_ids==ids(witness,"train_group_ids",train_rows)&&validation_ids==ids(witness,"validation_group_ids",validation_rows),
      "[family screen audit] group ids must repeat row ids");
  const std::set<int64_t> training_ids(train_ids.begin(),train_ids.end());
  for(const auto id:validation_ids)TORCH_CHECK(!training_ids.count(id),"[family screen audit] TRAIN/validation row-id overlap");
  const auto fit=fit_ridge(train,train_target,validation,validation_target,static_cast<size_t>(train_rows),static_cast<size_t>(validation_rows),static_cast<size_t>(width));
  const auto saved_predictions=floating(witness,"predictions",{validation_rows,static_cast<int64_t>(kTargets)});
  std::map<std::string,Difference> differences;
  const auto check_field=[&](const char *name,const Values &actual,torch::IntArrayRef shape){differences.emplace(name,compare(actual,floating(witness,name,shape),name));};
  check_field("feature_mean",fit.feature_mean,{width});check_field("feature_scale",fit.feature_scale,{width});
  check_field("target_mean",fit.target_mean,{static_cast<int64_t>(kTargets)});check_field("target_scale",fit.target_scale,{static_cast<int64_t>(kTargets)});
  check_field("weights",fit.weights,{width,static_cast<int64_t>(kTargets)});check_field("intercept",fit.intercept,{static_cast<int64_t>(kTargets)});
  check_field("predictions",fit.predictions,{validation_rows,static_cast<int64_t>(kTargets)});
  check_field("reported_metrics",fit.metrics,{static_cast<int64_t>(kTargets),4});
  differences.emplace("saved_prediction_metrics",compare(score_metrics(saved_predictions,validation_target,static_cast<size_t>(validation_rows),kTargets),floating(witness,"reported_metrics",{static_cast<int64_t>(kTargets),4}),"saved_prediction_metrics"));
  TORCH_CHECK(cp::boundary::sha256_file(witness_path)==witness_sha,"[family screen audit] witness bytes changed");
  TORCH_CHECK(cp::boundary::sha256_file(executable)==executable_sha,"[family screen audit] executable changed");
  TORCH_CHECK(fs::create_directory(out),"[family screen audit] exclusive output leaf required");
  std::ofstream report(out/"summary.json",std::ios::binary);TORCH_CHECK(report,"[family screen audit] cannot open report");
  report<<std::setprecision(21)
    <<"{\"kind\":\"cpb_family_screen_independent_scalar_ridge_audit_v1\",\"status\":\"passed\""
    <<",\"arithmetic\":\"scalar-long-double-centered-normal-equations-Cholesky\""
    <<",\"torch_used_for\":"<<quote(self_test?"archive-codec-shape-type-inspection-and-self-test-linalg-solve":"archive-codec-and-shape-type-inspection-only")
    <<",\"long_double_digits\":"<<std::numeric_limits<long double>::digits
    <<",\"absolute_tolerance\":"<<static_cast<double>(absolute_tolerance)<<",\"relative_tolerance\":"<<static_cast<double>(relative_tolerance)
    <<",\"training_rows\":"<<train_rows<<",\"validation_rows\":"<<validation_rows<<",\"feature_width\":"<<width
    <<",\"learned_width\":"<<learned_width<<",\"spectral_width\":"<<spectral_width<<",\"change_width\":"<<change_width
    <<",\"native_width\":"<<native_width<<",\"causal_updates\":"<<causal_updates<<",\"targets\":4,\"ridge_penalty\":1"
    <<",\"recipe\":"<<quote(recipe)<<",\"group_identity\":\"row-id\",\"legacy_target_parity\":false"
    <<",\"temporal_separation\":\"A confirmation population must be a disjoint later period. This tool checks row-id disjointness only.\""
    <<",\"finite_complete_support\":true,\"encoder_executions\":0,\"encoder_updates\":0,\"self_test\":"<<(self_test?"true":"false");
  if(self_test)report<<",\"unit_checks\":[\"cholesky-diag-2\",\"metrics-perfect-prediction\",\"torch-linalg-solve-agrees\",\"schema-round-trip\"]";
  report<<",\"differences\":{";
  bool comma=false;
  for(const auto &item:differences){
    if(comma)report<<',';
    comma=true;
    report<<quote(item.first)<<":{\"compared_values\":"<<item.second.count<<",\"maximum_absolute\":"<<item.second.maximum_absolute
      <<",\"maximum_tolerance_fraction\":"<<item.second.maximum_tolerance_fraction<<'}';
  }
  report<<"},\"independent_metrics\":[";
  const char *names[]{"past_volatility","past_trend","past_activity","past_relation"};
  for(size_t target=0;target<kTargets;++target){
    if(target)report<<',';
    report<<"{\"task\":"<<quote(names[target])<<",\"r2\":"<<fit.metrics[target*4]<<",\"mae\":"<<fit.metrics[target*4+1]
      <<",\"rmse\":"<<fit.metrics[target*4+2]<<",\"coverage\":1,\"valid\":"<<validation_rows<<",\"total\":"<<validation_rows<<'}';
  }
  report<<"],\"inputs\":{\"witness\":"<<artifact(witness_path,witness_sha)<<"},\"executable\":"<<artifact(executable,executable_sha)
    <<",\"input_bytes_unchanged\":true}\n";
  report.close();TORCH_CHECK(report,"[family screen audit] report write failed");
  std::cout<<"Independent scalar family-screen ridge audit passed. rows="<<train_rows<<"/"<<validation_rows<<" width="<<width<<"\n";
  return 0;
}
int self_test(const fs::path &out){
  unit_checks();
  const auto scratch=fs::path(out.string()+".scratch");
  TORCH_CHECK(!fs::exists(scratch)&&fs::create_directory(scratch),"[family screen audit] exclusive self-test scratch required");
  const auto witness=write_synthetic_witness(scratch);
  return audit_witness(witness,cp::boundary::sha256_file(witness),out,true);
}
int run(int argc,char **argv){
  torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard guard;
  bool self_test_requested=false;std::map<std::string,std::string> args;
  for(int i=1;i<argc;++i){
    const std::string key=argv[i];
    if(key=="--self-test"){TORCH_CHECK(!self_test_requested,"[family screen audit] duplicate --self-test");self_test_requested=true;continue;}
    TORCH_CHECK(i+1<argc,"[family screen audit] missing value: ",key);
    const std::set<std::string> allowed{"--witness","--witness-sha256","--output"};
    TORCH_CHECK(allowed.count(key)&&args.emplace(key,argv[++i]).second,"[family screen audit] unknown or duplicate option");
  }
  if(self_test_requested){
    TORCH_CHECK(args.size()==1&&args.count("--output"),"[family screen audit] --self-test takes only --output");
    return self_test(args.at("--output"));
  }
  for(const auto key:{"--witness","--witness-sha256","--output"})TORCH_CHECK(args.count(key),"[family screen audit] missing option: ",key);
  return audit_witness(args.at("--witness"),args.at("--witness-sha256"),args.at("--output"),false);
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
