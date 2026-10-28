// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Family-screen linear readout. Width and update count come from the checkpoint.
// No encoder updates, preprocessing fits, or legacy-target parity claim.
#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include "embedding/data/observation_windows.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include <torch/cuda.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace ow=embedding::data::observation_windows;
namespace ev=cuwacunu::evaluation;
namespace fs=std::filesystem;
namespace {
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
std::string artifact(const fs::path &path,const std::string &sha){
  std::ostringstream out;out<<"{\"path\":"<<quote(path.generic_string())<<",\"bytes\":"<<fs::file_size(path)<<",\"sha256\":"<<quote(sha)<<'}';
  return out.str();
}
void finite(const torch::Tensor &value,const char *name){
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kFloat64&&torch::isfinite(value).all().item<bool>(),
      "[family screen] finite CPU F64 required: ",name);
}
cp::RawHistory raw(const ow::Dataset &data){return{data.values,data.observed,data.channel_ids,data.timestamps,data.sampling_interval};}
void market_layout(const ow::Dataset &data,const char *name){
  TORCH_CHECK(data.values.size(3)==9,"[family screen] ",name," feature count must be 9; this readout's past-window targets use close index 3 and volume index 4");
  TORCH_CHECK(data.values.size(2)>=9,"[family screen] ",name," history must be at least 9");
}
torch::Tensor features(cp::Loaded &live,const ow::Dataset &data,int64_t native_width){
  std::vector<torch::Tensor> rows;torch::NoGradGuard guard;live.model->eval();
  for(int64_t first=0;first<data.rows();first+=8){
    std::vector<int64_t> ids;
    for(int64_t i=first;i<std::min(first+8,data.rows());++i)ids.push_back(i);
    const auto batch=ow::subset(data,ids);
    const auto encoded=live.model->encode_at(raw(batch),batch.cutoffs,live.scaler);
    TORCH_CHECK(encoded.valid.all().item<bool>()&&cp::boundary::same_bits(encoded.cutoffs,batch.cutoffs),
        "[family screen] complete declared feature support and cutoff association required");
    rows.push_back(encoded.native.detach().to(torch::kCPU,torch::kFloat64));
  }
  const auto out=torch::cat(rows);
  TORCH_CHECK(out.size(0)==data.rows()&&out.size(1)==native_width,"[family screen] encoded width must match native width");
  return out;
}
std::string bound_recipe(const std::string &path,const cp::Loaded &live,const cp::Config &config){
  torch::serialize::InputArchive training;training.load_from(path,config.level.device);
  TORCH_CHECK(cp::checkpoint_detail::d::read_text(training,"artifact_kind")==cp::training::kTrainingStateKind,
      "[family screen] cpb_family_causal_training_state_v1 required");
  const auto recipe=cp::checkpoint_detail::d::read_text(training,"recipe_identity");
  TORCH_CHECK(!recipe.empty()&&recipe.size()<=4096&&recipe.find('\0')==std::string::npos,"[family screen] training-state recipe_identity required");
  const auto progress=cp::checkpoint_detail::d::read_vector(training,"progress_v1",torch::kInt64,5);
  const auto completed=progress[0].item<int64_t>();
  TORCH_CHECK(completed==live.causal_updates,"[family screen] training-state update count must match the inference checkpoint");
  torch::serialize::InputArchive encoder;training.read("encoder",encoder);
  const auto nested=cp::checkpoint_detail::read(encoder,config.level.device,&config);
  TORCH_CHECK(nested.causal_updates==live.causal_updates&&nested.model->state_identity()==live.model->state_identity()&&
      nested.scaler.identity()==live.scaler.identity()&&nested.model->normalizer().identity()==live.model->normalizer().identity(),
      "[family screen] training state and inference checkpoint must be the same encoder");
  return recipe;
}
ev::TargetValues targets(const ow::Dataset &data){
  // The float32 round trip matches the existing V17 market readout's target preparation.
  auto prepared=ev::past_window_targets(data.values.to(torch::kFloat32).to(torch::kFloat64),data.observed);
  TORCH_CHECK(prepared.values.size(1)==4&&prepared.valid.all().item<bool>(),"[family screen] all four past-window targets must be valid");
  finite(prepared.values,"targets");return prepared;
}
int run(int argc,char **argv){
  TORCH_CHECK(argc==15,"[family screen] seven explicit flag/value pairs required");
  const std::set<std::string> allowed{"--settings","--checkpoint","--training-state","--train","--validation","--output","--expected-updates"};
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&args.emplace(argv[i],argv[i+1]).second,"[family screen] unknown or duplicate option");
  for(const auto &key:allowed)TORCH_CHECK(args.count(key),"[family screen] missing option: ",key);
  torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard guard;
  TORCH_CHECK(torch::cuda::is_available(),"[family screen] CUDA is required to run the readout");
  std::map<std::string,std::string> hashes;
  for(const auto key:{"--settings","--checkpoint","--training-state","--train","--validation"}){
    TORCH_CHECK(fs::is_regular_file(args.at(key)),"[family screen] input must exist: ",key);
    hashes.emplace(key,cp::boundary::sha256_file(args.at(key)));
  }
  const auto executable=fs::canonical("/proc/self/exe");
  const auto executable_sha=cp::boundary::sha256_file(executable);
  const auto config=cp::read_settings(args.at("--settings"));
  auto live=cp::load_checkpoint(args.at("--checkpoint"),config.level.device,&config);
  size_t used=0;const auto updates=std::stoll(args.at("--expected-updates"),&used);
  TORCH_CHECK(used==args.at("--expected-updates").size()&&updates>=0&&live.causal_updates==updates&&live.model->causal_updates()==updates,
      "[family screen] --expected-updates must be a nonnegative integer equal to checkpoint causal_updates");
  const auto dims=cp::dimensions(live.model->config());
  TORCH_CHECK(dims.native_width==cp::implementation::location_scale_decoder_native_width(live.model->config())&&
      dims.shape_width>0&&dims.spectral_width>=0&&dims.change_width>0&&
      dims.shape_width+dims.spectral_width+dims.change_width==dims.native_width,
      "[family screen] saved native width and learned/spectral/change layout disagree");
  const auto before=live.model->state_identity();
  const auto scaler=live.scaler.identity(),bank=live.model->normalizer().identity();
  const auto recipe=bound_recipe(args.at("--training-state"),live,config);
  TORCH_CHECK(live.model->state_identity()==before&&live.scaler.identity()==scaler&&live.model->normalizer().identity()==bank,
      "[family screen] reading the training state must not mutate the inference checkpoint");
  const auto train=ow::read(args.at("--train")),validation=ow::read(args.at("--validation"));
  TORCH_CHECK(train.role=="TRAIN"&&validation.role=="VALIDATION"&&train.rows()>0&&validation.rows()>0,"[family screen] TRAIN and VALIDATION rows required");
  market_layout(train,"TRAIN");market_layout(validation,"validation");
  const auto train_targets=targets(train),validation_targets=targets(validation);
  const auto train_rows=train.rows(),validation_rows=validation.rows();
  std::set<int64_t> train_ids;
  for(int64_t i=0;i<train_rows;++i)TORCH_CHECK(train_ids.insert(train.row_ids[i].item<int64_t>()).second,"[family screen] duplicate TRAIN row id");
  for(int64_t i=0;i<validation_rows;++i)TORCH_CHECK(!train_ids.count(validation.row_ids[i].item<int64_t>()),"[family screen] TRAIN/validation row-id overlap");
  const auto train_features=features(live,train,dims.native_width),validation_features=features(live,validation,dims.native_width);
  TORCH_CHECK(train_targets.values.sizes()==torch::IntArrayRef({train_rows,4})&&validation_targets.values.sizes()==torch::IntArrayRef({validation_rows,4}),
      "[family screen] four target columns required");
  finite(train_features,"TRAIN features");finite(validation_features,"validation features");
  const auto mean=train_features.mean(0),scale=(train_features-mean).square().mean(0).sqrt().clamp_min(1e-8);
  const auto standardized=(train_features-mean)/scale,validation_standardized=(validation_features-mean)/scale;
  const auto target_mean=train_targets.values.mean(0),target_scale=(train_targets.values-target_mean).square().mean(0).sqrt().clamp_min(1e-8);
  TORCH_CHECK((train_targets.values-target_mean).square().sum(0).gt(0).all().item<bool>(),"[family screen] zero TRAIN target variance abstains");
  const auto standardized_targets=(train_targets.values-target_mean)/target_scale;
  const auto feature_center=standardized.mean(0),target_center=standardized_targets.mean(0),centered=standardized-feature_center;
  const auto weights=at::linalg_solve(centered.transpose(0,1).matmul(centered)+torch::eye(standardized.size(1),standardized.options()),
      centered.transpose(0,1).matmul(standardized_targets-target_center));
  const auto intercept=target_center-feature_center.matmul(weights);
  const auto predictions=(validation_standardized.matmul(weights)+intercept)*target_scale+target_mean;
  for(const auto &item:std::vector<std::pair<const char *,torch::Tensor>>{{"feature mean",mean},{"feature scale",scale},{"target mean",target_mean},{"target scale",target_scale},{"weights",weights},{"intercept",intercept},{"predictions",predictions}})finite(item.second,item.first);
  std::vector<ev::RegressionScore> scores;auto reported_metrics=torch::empty({4,4},torch::kFloat64);
  const auto names=ev::past_window_target_specs();TORCH_CHECK(names.size()==4,"[family screen] four fixed targets required");
  for(int64_t task=0;task<4;++task){
    auto score=ev::regression_score(predictions.select(1,task),validation_targets.values.select(1,task),validation_targets.valid.select(1,task),torch::ones({validation_rows},torch::kBool));
    TORCH_CHECK(score.supported&&score.r2_supported&&score.total==validation_rows&&score.target_eligible==validation_rows&&score.valid==validation_rows&&score.coverage==1.,
        "[family screen] strict finite complete scored target required");
    scores.push_back(score);reported_metrics[task].copy_(torch::tensor({score.r2,score.mae,score.rmse,score.coverage},torch::kFloat64));
  }
  TORCH_CHECK(live.model->state_identity()==before&&live.scaler.identity()==scaler&&live.model->normalizer().identity()==bank,
      "[family screen] readout must not mutate encoder or preprocessing");
  for(const auto &item:hashes)TORCH_CHECK(cp::boundary::sha256_file(args.at(item.first))==item.second,"[family screen] input changed during execution: ",item.first);
  TORCH_CHECK(cp::boundary::sha256_file(executable)==executable_sha,"[family screen] executable changed during execution");
  const fs::path out=args.at("--output");TORCH_CHECK(!fs::exists(out)&&fs::is_directory(out.parent_path()),"[family screen] new output leaf with existing parent required");
  fs::create_directory(out);
  torch::serialize::OutputArchive archive;
  archive.write("artifact_kind",ow::text_tensor("cpb_family_screen_ridge_witness_v1"),true);
  archive.write("metric_columns",ow::text_tensor("r2,mae,rmse,coverage"),true);
  archive.write("recipe",ow::text_tensor(recipe),true);
  archive.write("group_identity",ow::text_tensor("row-id"),true);
  for(const auto &item:hashes){auto key=item.first.substr(2);std::replace(key.begin(),key.end(),'-','_');archive.write(key+"_sha256",ow::text_tensor(item.second),true);}
  archive.write("executable_sha256",ow::text_tensor(executable_sha),true);
  archive.write("native_width",torch::tensor(dims.native_width,torch::kInt64),true);
  archive.write("learned_width",torch::tensor(dims.shape_width,torch::kInt64),true);
  archive.write("spectral_width",torch::tensor(dims.spectral_width,torch::kInt64),true);
  archive.write("change_width",torch::tensor(dims.change_width,torch::kInt64),true);
  archive.write("causal_updates",torch::tensor(static_cast<int64_t>(updates),torch::kInt64),true);
  for(const auto &item:std::vector<std::pair<std::string,torch::Tensor>>{
      {"train_features",train_features},{"validation_features",validation_features},{"feature_mean",mean},{"feature_scale",scale},
      {"target_mean",target_mean},{"target_scale",target_scale},{"weights",weights},{"intercept",intercept},{"predictions",predictions},
      {"train_targets",train_targets.values},{"validation_targets",validation_targets.values},
      {"train_target_valid",train_targets.valid},{"validation_target_valid",validation_targets.valid},
      {"train_feature_valid",torch::ones({train_rows},torch::kBool)},{"validation_feature_valid",torch::ones({validation_rows},torch::kBool)},
      {"train_row_ids",train.row_ids},{"validation_row_ids",validation.row_ids},
      {"train_group_ids",train.row_ids.clone()},{"validation_group_ids",validation.row_ids.clone()},
      {"train_cutoffs",train.cutoffs},{"validation_cutoffs",validation.cutoffs},{"reported_metrics",reported_metrics}})archive.write(item.first,item.second,true);
  cp::checkpoint_detail::d::publish_checkpoint(archive,(out/"ridge-witness.pt").string());
  std::ofstream report(out/"summary.json");
  report<<std::setprecision(17)
    <<"{\"kind\":\"cpb_family_screen_linear_readout_v1\",\"recipe\":"<<quote(recipe)<<",\"updates\":"<<updates
    <<",\"learned_width\":"<<dims.shape_width<<",\"spectral_width\":"<<dims.spectral_width<<",\"change_width\":"<<dims.change_width
    <<",\"native_width\":"<<dims.native_width<<",\"input_feature_count\":9,\"history_floor\":9"
    <<",\"market_target_layout\":\"close index 3, volume index 4, H>=9, F==9\""
    <<",\"target_preparation\":\"F64 observation values cast through F32 before past_window_targets\""
    <<",\"head\":\"TRAIN-standardized-ridge1\",\"ridge_penalty\":1,\"encoder_updates\":0,\"preprocessing_fits\":0,\"head_fits\":4"
    <<",\"legacy_target_parity\":false,\"group_identity\":\"row-id\""
    <<",\"temporal_separation\":\"A confirmation population must be a disjoint later period. This tool checks row-id disjointness only.\""
    <<",\"fresh_confirmation\":false,\"reconstruction_evaluated\":false,\"training_feature_rows\":"<<train_rows
    <<",\"validation_feature_rows\":"<<validation_rows<<",\"metrics\":[";
  for(int64_t task=0;task<4;++task){
    const auto &score=scores[static_cast<size_t>(task)];if(task)report<<',';
    report<<"{\"task\":"<<quote(names[static_cast<size_t>(task)].name)<<",\"r2\":"<<score.r2<<",\"mae\":"<<score.mae
      <<",\"rmse\":"<<score.rmse<<",\"coverage\":"<<score.coverage<<",\"valid\":"<<score.valid
      <<",\"target_eligible\":"<<score.target_eligible<<",\"total\":"<<score.total<<'}';
  }
  report<<"],\"encoder_state_unchanged\":true,\"model_state_identity\":"<<quote(before)<<",\"scaler_identity\":"<<quote(scaler)
    <<",\"bank_identity\":"<<quote(bank)<<",\"inputs\":{";
  bool comma=false;
  for(const auto &item:hashes){
    if(comma)report<<',';
    comma=true;
    report<<quote(item.first.substr(2))<<':'<<artifact(args.at(item.first),item.second);
  }
  report<<"},\"executable\":"<<artifact(executable,executable_sha)
    <<",\"witness\":"<<artifact(out/"ridge-witness.pt",cp::boundary::sha256_file(out/"ridge-witness.pt"))
    <<",\"input_bytes_unchanged\":true}";
  report.close();TORCH_CHECK(report,"[family screen] write report");
  std::cout<<"Saved four family-screen linear readouts. native_width="<<dims.native_width<<" updates="<<updates<<"\n";
  return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
