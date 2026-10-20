// Copyright (c) 2026 Waajacu. See LICENSE.
// Saved fitted arrays only. No Fit constructor, optimizer or normalization fit.
#include "include/model_binding.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace {
namespace ev=cuwacunu::evaluation;
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v4;
namespace fs=std::filesystem;
void require(bool yes,const std::string &why){TORCH_CHECK(yes,"[V16 saved confirmation readout] ",why);}
std::string quote(const std::string &text){std::string out="\"";for(const char c:text){require(c>=32,"printable metadata");if(c=='"'||c=='\\')out+='\\';out+=c;}return out+'"';}
torch::Tensor read(torch::serialize::InputArchive &a,const char *name){torch::Tensor value;a.read(name,value,true);require(value.device().is_cpu()&&!value.requires_grad(),"detached CPU saved fit");return value;}
void floating(const torch::Tensor &x,torch::IntArrayRef shape){require(x.scalar_type()==torch::kFloat64&&x.sizes()==shape&&torch::isfinite(x).all().item<bool>(),"exact finite saved F64 fit shape");}
std::string score(const ev::RegressionScore &s){
  std::ostringstream out;out<<std::setprecision(17)<<"{\"total\":"<<s.total<<",\"target_eligible\":"<<s.target_eligible<<",\"valid\":"<<s.valid<<",\"coverage\":"<<s.coverage<<",\"mae\":";
  if(s.supported)out<<s.mae;else out<<"null";out<<",\"rmse\":";if(s.supported)out<<s.rmse;else out<<"null";out<<",\"r2\":";if(s.r2_supported)out<<s.r2;else out<<"null";return out.str()+"}";
}
int run(int argc,char **argv){
  require(fs::exists("/.dockerenv"),"managed Linux container");torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard guard;
  require(argc==15,"fit/features/targets/output/seal/seal-sha256/fit-sha256 required");std::map<std::string,fs::path> args;
  for(int i=1;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"unique argument");
  for(const auto *key:{"--fit","--features","--targets","--output","--seal","--seal-sha256","--fit-sha256"})require(args.count(key),"required argument");
  require(cp::boundary::sha256_file(args.at("--seal"))==args.at("--seal-sha256").string(),"exact actual all-six development seal");
  require(cp::boundary::sha256_file(args.at("--fit"))==args.at("--fit-sha256").string(),"exact previously sealed fitted arrays");
  const auto f=ev::load_features(args.at("--features").string());
  const auto t=ev::load_targets(args.at("--targets").string());
  require(f.values.sizes()==torch::IntArrayRef({32,36})&&f.values.scalar_type()==torch::kFloat64&&t.targets.values.sizes()==torch::IntArrayRef({32,4})&&torch::equal(f.row_ids,t.row_ids)&&torch::equal(t.row_ids,t.group_ids),"exact32 confirmation rows/native36/four targets");
  require(f.valid.all().item<bool>()&&t.targets.valid.all().item<bool>(),"complete confirmation coverage; no replacement rows");
  torch::serialize::InputArchive fit;fit.load_from(args.at("--fit").string(),torch::kCPU);
  const auto mean=read(fit,"feature_mean"),scale=read(fit,"feature_scale"),target_mean=read(fit,"target_mean"),target_scale=read(fit,"target_scale");
  const auto ridge_w=read(fit,"ridge_weights"),ridge_b=read(fit,"ridge_intercept"),w1=read(fit,"neural_w1"),b1=read(fit,"neural_b1"),w2=read(fit,"neural_w2"),b2=read(fit,"neural_b2");
  floating(mean,{36});floating(scale,{36});floating(target_mean,{4});floating(target_scale,{4});floating(ridge_w,{36,4});floating(ridge_b,{4});floating(w1,{36,16});floating(b1,{16});floating(w2,{16,4});floating(b2,{4});
  require(scale.ge(1e-8).all().item<bool>()&&target_scale.ge(1e-8).all().item<bool>(),"frozen positive TRAIN normalization");
  const auto supported=read(fit,"supported_targets"),target_supported=read(fit,"target_training_supported"),feature_supported=read(fit,"feature_training_supported"),counts=read(fit,"target_training_counts"),training_rows=read(fit,"training_feature_rows"),seed=read(fit,"actual_seed");
  require(supported.scalar_type()==torch::kBool&&supported.sizes()==torch::IntArrayRef({4})&&supported.all().item<bool>()&&target_supported.scalar_type()==torch::kBool&&target_supported.sizes()==supported.sizes()&&target_supported.all().item<bool>(),"supported frozen heads");
  require(feature_supported.scalar_type()==torch::kBool&&feature_supported.numel()==1&&feature_supported.item<bool>()&&counts.scalar_type()==torch::kInt64&&counts.sizes()==torch::IntArrayRef({4})&&counts.eq(112).all().item<bool>()&&training_rows.scalar_type()==torch::kInt64&&training_rows.numel()==1&&training_rows.item<int64_t>()==112&&seed.scalar_type()==torch::kInt64&&seed.numel()==1,"original112 TRAIN support and typed seed");
  const uint64_t unsigned_seed=static_cast<uint64_t>(seed.item<int64_t>());
  require(unsigned_seed==13088411117524848790ULL||unsigned_seed==10018485049408248344ULL||unsigned_seed==9083751155097190455ULL,"unchanged three D36 head streams");
  const auto x=(f.values-mean)/scale;
  const auto ridge=(x.matmul(ridge_w)+ridge_b)*target_scale+target_mean;
  const auto neural=(torch::tanh(x.matmul(w1)+b1).matmul(w2)+b2)*target_scale+target_mean;
  require(torch::isfinite(ridge).all().item<bool>()&&torch::isfinite(neural).all().item<bool>(),"finite predictions from existing fitted arrays");
  const auto output=fs::absolute(args.at("--output"));require(!fs::exists(output)&&fs::is_directory(output.parent_path())&&fs::create_directory(output),"exclusive confirmation readout leaf");
  torch::serialize::OutputArchive predictions;predictions.write("ridge_predictions",ridge,true);predictions.write("neural_predictions",neural,true);predictions.write("valid",t.targets.valid.logical_and(f.valid.unsqueeze(1)).logical_and(supported.unsqueeze(0)),true);predictions.write("targets",t.targets.values,true);predictions.write("target_valid",t.targets.valid,true);predictions.write("row_ids",t.row_ids,true);predictions.write("group_ids",t.group_ids,true);predictions.save_to((output/"predictions_confirmation.pt").string());
  std::ostringstream report;report<<std::setprecision(17)<<"{\"kind\":\"cpb_v16_saved_fit_confirmation_v1\",\"status\":\"passed\",\"head_fits\":0,\"normalizer_fits\":0,\"encoder_calls\":0,\"TRAIN_fitted_arrays_reused\":true,\"rows\":32,\"actual_seed_decimal\":"<<quote(std::to_string(unsigned_seed))<<",\"fit_path\":"<<quote(args.at("--fit").string())<<",\"fit_sha256\":"<<quote(args.at("--fit-sha256").string())<<",\"targets\":[";
  const auto specs=ev::past_window_target_specs();for(int64_t j=0;j<4;++j){if(j)report<<',';const auto valid=f.valid.logical_and(supported[j]);report<<"{\"name\":"<<quote(specs[j].name)<<",\"unit\":"<<quote(specs[j].unit)<<",\"ridge\":"<<score(ev::regression_score(ridge.select(1,j),t.targets.values.select(1,j),t.targets.valid.select(1,j),valid))<<",\"neural\":"<<score(ev::regression_score(neural.select(1,j),t.targets.values.select(1,j),t.targets.valid.select(1,j),valid))<<'}';}report<<"]}";
  require(cp::boundary::sha256_file(args.at("--fit"))==args.at("--fit-sha256").string()&&cp::boundary::sha256_file(args.at("--seal"))==args.at("--seal-sha256").string(),"exact fitted arrays/seal unchanged while scoring");
  std::ofstream result(output/"report.json");result<<report.str()<<'\n';result.close();require(bool(result),"persist confirmation report");return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
