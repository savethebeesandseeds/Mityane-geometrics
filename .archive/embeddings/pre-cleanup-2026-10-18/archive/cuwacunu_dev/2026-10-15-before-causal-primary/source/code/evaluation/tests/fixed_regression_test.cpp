// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#include "cuwacunu/evaluation/fixed_regression.h"
#include <ATen/Context.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

namespace ev=cuwacunu::evaluation;
namespace fs=std::filesystem;
void check(bool value,const char *why){if(!value)throw std::runtime_error(why);}
template<class F>void rejects(F fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}check(failed,"expected rejection");}
int main(){try{
  torch::set_num_threads(1);
  auto values=torch::zeros({2,3,12,9},torch::kFloat64),mask=torch::ones({2,3,12,9},torch::kBool);
  const auto r=torch::tensor({.001,.004,-.002,.005,-.003,.002,-.004,.003,.006,-.001,.002},torch::kFloat64);
  const auto logs=torch::cat({torch::zeros({1},torch::kFloat64),r.cumsum(0)});
  for(int64_t c=0;c<3;++c){const double multiplier=c==0?1:(c==1?2:-1);values.select(1,c).select(2,3).copy_((logs*multiplier).exp().unsqueeze(0).expand({2,12}));}
  values.select(3,4).fill_(9);mask[1].fill_(false);
  const auto labels=ev::past_window_targets(values,mask);
  check(labels.valid[0].all().item<bool>()&&!labels.valid[1].any().item<bool>(),"label eligibility");
  const auto expected_std=(r-r.mean()).square().mean().sqrt().item<double>()*4/3;
  check(std::abs(labels.values[0][0].item<double>()-expected_std)<1e-14,"population return volatility");
  check(std::abs(labels.values[0][1].item<double>()-r.mean().item<double>()*2/3)<1e-14,"past mean-return trend");
  check(std::abs(labels.values[0][2].item<double>()-std::log(10))<1e-14,"past activity");
  check(std::abs(labels.values[0][3].item<double>()+1.0/3)<1e-12,"joint Pearson relation");
  auto hidden=values.clone();hidden[1].fill_(std::numeric_limits<double>::quiet_NaN());
  auto hidden_labels=ev::past_window_targets(hidden,mask);
  check(torch::equal(labels.values,hidden_labels.values)&&torch::equal(labels.valid,hidden_labels.valid),"hidden target storage isolation");
  auto gaps=mask.clone();gaps[0][0][4].fill_(false);auto gaps_values=values.clone();gaps_values[0][0][4].fill_(1e200);
  check(torch::equal(ev::past_window_targets(values,gaps).values,ev::past_window_targets(gaps_values,gaps).values),"adjacent gap storage isolation");
  auto too_few=mask.clone();too_few[0].slice(1,8).fill_(false);
  check(!ev::past_window_targets(values,too_few).valid[0][0].item<bool>(),"need eight adjacent returns in every channel");
  auto constant=values.clone();constant.select(3,3).fill_(1);
  auto constant_labels=ev::past_window_targets(constant,mask);
  check(constant_labels.valid[0][0].item<bool>()&&!constant_labels.valid[0][3].item<bool>(),"constant returns valid volatility, invalid correlation");
  auto truth=torch::tensor({1.,2.,3.},torch::kFloat64),all=torch::ones({3},torch::kBool);
  auto score=ev::regression_score(truth+1,truth,all,all);
  check(score.valid==3&&score.mae==1&&score.rmse==1&&std::abs(score.r2+.5)<1e-14,"raw-unit regression metric");
  check(!ev::regression_score(truth,torch::ones_like(truth),all,all).r2_supported,"constant validation target R2 N/A");
  check(!ev::regression_score(truth,truth,all,torch::zeros_like(all)).supported,"zero coverage unsupported");
  const auto root=fs::temp_directory_path()/("cuwacunu-fixed-regression-test-"+std::to_string(getpid()));check(!fs::exists(root),"exclusive fixture path");fs::create_directories(root);
  auto rows=[](int64_t n,int64_t start){return torch::arange(start,start+n,torch::kInt64);};
  auto features=[](int64_t n,int64_t start,int64_t width){auto x=torch::arange(start,start+n,torch::kFloat64)/10;std::vector<torch::Tensor> cols;
    for(int64_t d=0;d<width;++d)cols.push_back(d%3==0?x:(d%3==1?x.sin():x.cos()));
    return ev::FrozenFeatures{torch::stack(cols,1),torch::ones({n},torch::kBool),torch::arange(start,start+n,torch::kInt64)};};
  auto targets=[&](const ev::FrozenFeatures &f,int64_t group_start){auto x=f.values.select(1,0);auto y=torch::stack({x*2,x.sin(),x.cos(),torch::full_like(x,3)},1);
    return ev::TargetRows{{y,torch::ones_like(y,torch::kBool)},f.row_ids.clone(),rows(x.numel(),group_start)};};
  auto train=features(24,0,3),validation=features(8,100,3);auto train_targets=targets(train,0),validation_targets=targets(validation,100);
  ev::save_features((root/"train.pt").string(),train);ev::save_targets((root/"targets.pt").string(),train_targets);
  check(torch::equal(ev::load_features((root/"train.pt").string()).values,train.values),"feature archive roundtrip");
  check(torch::equal(ev::load_targets((root/"targets.pt").string()).targets.values,train_targets.targets.values),"target archive roundtrip");
  rejects([&]{ev::save_targets((root/"targets.pt").string(),train_targets);});
  auto state=at::globalContext().defaultGenerator(at::Device(at::kCPU)).get_state().clone();
  ev::FixedRegressionRun run;run.output_directory=(root/"readouts").string();run.card_sha256=std::string(64,'a');run.input_manifest_sha256=std::string(64,'b');
  run.training=train_targets;run.validation=validation_targets;run.target_specs=ev::past_window_target_specs();
  run.methods={{"primal","artificial TRAIN","artificial VALIDATION",train,validation},{"dual","artificial TRAIN","artificial VALIDATION",features(24,0,30),features(8,100,30)}};
  auto report=ev::run_fixed_regression(run);check(report.find("\"ridge_fits\":1")!=std::string::npos&&report.find("\"neural_fits\":3")!=std::string::npos,"fixed fit counters");
  check(torch::equal(state,at::globalContext().defaultGenerator(at::Device(at::kCPU)).get_state()),"readout-local RNG isolation");
  for(const auto &method:run.methods){
    torch::serialize::InputArchive fit,pred;fit.load_from((root/"readouts"/"models"/method.name/"rep-2701"/"fit.pt").string());pred.load_from((root/"readouts"/"models"/method.name/"rep-2701"/"predictions_validation.pt").string());
    torch::Tensor mean,scale,tm,ts,rw,rb,w1,b1,w2,b2,expected_ridge,expected_neural;
    fit.read("feature_mean",mean);fit.read("feature_scale",scale);fit.read("target_mean",tm);fit.read("target_scale",ts);fit.read("ridge_weights",rw);fit.read("ridge_intercept",rb);fit.read("neural_w1",w1);fit.read("neural_b1",b1);fit.read("neural_w2",w2);fit.read("neural_b2",b2);
    pred.read("ridge_predictions",expected_ridge);pred.read("neural_predictions",expected_neural);auto x=(method.validation.values-mean)/scale;
    check(torch::equal((x.matmul(rw)+rb)*ts+tm,expected_ridge),"persisted ridge arithmetic exact");
    check(torch::equal((torch::tanh(x.matmul(w1)+b1).matmul(w2)+b2)*ts+tm,expected_neural),"persisted neural arithmetic exact");
    check(torch::allclose(mean,method.training.values.mean(0),1e-14,1e-14),"normalizer TRAIN only");
    check(torch::allclose(tm,train_targets.targets.values.mean(0),1e-14,1e-14),"target scaler TRAIN only");
  }
  auto partial=run;partial.output_directory=(root/"partial-support").string();partial.methods.resize(1);
  partial.training.targets.valid=partial.training.targets.valid.clone();partial.training.targets.valid.select(1,3).fill_(false);
  const auto partial_report=ev::run_fixed_regression(partial);
  check(partial_report.find("\"train_target_eligible\":0,\"train_target_supported\":false,\"method_training_supported\":false")!=std::string::npos,"unsupported target stays named without stopping other targets");
  torch::serialize::InputArchive partial_fit;partial_fit.load_from((root/"partial-support"/"models"/"primal"/"rep-2701"/"fit.pt").string());
  torch::Tensor partial_supported,partial_counts;partial_fit.read("supported_targets",partial_supported);partial_fit.read("target_training_counts",partial_counts);
  check(partial_supported.slice(0,0,3).all().item<bool>()&&!partial_supported[3].item<bool>()&&partial_counts[3].item<int64_t>()==0,"partial TRAIN support persisted");
  auto unsupported=partial;unsupported.output_directory=(root/"unsupported").string();unsupported.training.targets.valid=torch::zeros_like(unsupported.training.targets.valid);
  const auto unsupported_report=ev::run_fixed_regression(unsupported);
  check(unsupported_report.find("\"ridge_fits\":0,\"ridge_task_fits\":0,\"neural_fits\":0")!=std::string::npos&&unsupported_report.find("\"neural_updates_completed\":0")!=std::string::npos,"entirely unsupported target set performs no fit");
  auto thin=run;thin.output_directory=(root/"unsupported-method").string();thin.methods.resize(1);
  auto thin_method=thin.methods.front();thin_method.name="one_training_row";thin_method.training.valid=torch::zeros_like(thin_method.training.valid);thin_method.training.valid[0].fill_(true);thin.methods.push_back(thin_method);
  const auto thin_report=ev::run_fixed_regression(thin);
  check(thin_report.find("\"name\":\"primal\"")!=std::string::npos&&thin_report.find("\"training_feature_rows\":1,\"feature_training_supported\":false")!=std::string::npos,"unsupported method does not abort supported method");
  torch::serialize::InputArchive thin_fit;thin_fit.load_from((root/"unsupported-method"/"models"/"one_training_row"/"rep-2701"/"fit.pt").string());
  torch::Tensor thin_supported,thin_feature_supported;thin_fit.read("supported_targets",thin_supported);thin_fit.read("feature_training_supported",thin_feature_supported);
  check(!thin_supported.any().item<bool>()&&!thin_feature_supported.item<bool>(),"unsupported method fit assets marked unsupported");
  rejects([&]{ev::run_fixed_regression(run);});
  auto bad=run;bad.output_directory=(root/"bad-association").string();bad.methods[0].validation.row_ids=bad.methods[0].validation.row_ids+1;
  rejects([&]{ev::run_fixed_regression(bad);});check(!fs::exists(bad.output_directory),"association rejection before output");
  bad=run;bad.output_directory=(root/"bad-overlap").string();bad.validation.group_ids.select(0,0).copy_(bad.training.group_ids.select(0,0));
  rejects([&]{ev::run_fixed_regression(bad);});check(!fs::exists(bad.output_directory),"split overlap rejection before output");
  fs::remove_all(root);std::cout<<"fixed regression targets/masks/support/metrics/TRAIN-fit/primal-dual/persistence/RNG passed\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
