// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Family-screen linear readout for one instrument and timeframe channels.
// Width and update count come from the checkpoint. No encoder updates,
// preprocessing fits, or comparison with the published three-instrument probes.
#include "embedding/encoders/causal_patch_bottleneck_mae/family.h"
#include "embedding/data/observation_windows.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include "timeframe_targets.h"
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
namespace screen=embedding::evaluation::family_screen;
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
screen::FeatureView feature_view(const std::string &text){
  if(text=="raw")return screen::FeatureView::RawPrices;
  if(text=="log-returns")return screen::FeatureView::LogReturns;
  TORCH_CHECK(false,"[family screen] --feature-view must be raw or log-returns");
  return screen::FeatureView::RawPrices;
}
const char *view_name(screen::FeatureView view){return view==screen::FeatureView::LogReturns?"log-returns":"raw";}
void same_channels(const cp::Config &config,const ow::Dataset &data,const char *name){
  const auto expected=cp::implementation::resolved_channel_ids(config.level);
  TORCH_CHECK(data.values.size(1)==config.level.channel_count&&static_cast<int64_t>(expected.size())==config.level.channel_count,
      "[family screen] ",name," channel count must match the encoder");
  TORCH_CHECK(data.values.size(3)==9&&config.level.input_width==9,
      "[family screen] ",name," feature count must be the nine kline fields");
  TORCH_CHECK(data.sampling_interval==config.level.sampling_interval,
      "[family screen] ",name," sampling interval must match the encoder");
  const auto ids=data.channel_ids.dim()==1?data.channel_ids:data.channel_ids.select(0,0);
  TORCH_CHECK(ids.defined()&&ids.dim()==1&&ids.size(0)==config.level.channel_count,"[family screen] ",name," channel ids required");
  if(data.channel_ids.dim()==2){
    for(int64_t row=0;row<data.channel_ids.size(0);++row)
      TORCH_CHECK(torch::equal(data.channel_ids.select(0,row),ids),"[family screen] ",name," channel ids must be the same on every row");
  }
  for(int64_t channel=0;channel<config.level.channel_count;++channel)
    TORCH_CHECK(ids[channel].item<int64_t>()==expected[static_cast<std::size_t>(channel)],
        "[family screen] ",name," channel ids must match the encoder, in order");
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
screen::TimeframeTargets targets(const cp::Config &config,const ow::Dataset &data,screen::FeatureView view,const char *name){
  same_channels(config,data,name);
  const auto window=cp::boundary::retain(raw(data),data.cutoffs,config);
  auto prepared=screen::timeframe_window_targets(window.raw.data,window.raw.observed,window.raw.channel_ids,window.raw.sampling_interval,view);
  TORCH_CHECK(prepared.values.size(1)==4&&prepared.valid.slice(1,0,3).all().item<bool>(),
      "[family screen] ",name," anchor volatility, trend and activity must be valid on every retained row");
  finite(prepared.values,name);return prepared;
}
struct Column{
  torch::Tensor weight,prediction;
  double target_mean{0},target_scale{1},intercept{0};
  ev::RegressionScore score;
  bool solved{false};
};
Column fit_column(const torch::Tensor &train_x,const torch::Tensor &validation_x,const torch::Tensor &train_y,const torch::Tensor &train_valid,
    const torch::Tensor &validation_y,const torch::Tensor &validation_valid,bool required,const char *name){
  const auto index=train_valid.nonzero().flatten();
  Column column;
  column.weight=torch::zeros({train_x.size(1)},train_x.options());
  column.prediction=torch::zeros({validation_x.size(0)},train_x.options());
  column.score.total=validation_y.size(0);
  column.score.target_eligible=validation_valid.sum().item<int64_t>();
  if(index.numel()<2||(required&&index.numel()!=train_y.size(0))){
    TORCH_CHECK(!required,"[family screen] ",name," needs a complete TRAIN column with at least two rows");
    return column;
  }
  const auto y=train_y.index_select(0,index);
  const auto centered=y-y.mean();
  if(!(centered.square().sum().item<double>()>0)){
    TORCH_CHECK(!required,"[family screen] zero TRAIN target variance abstains: ",name);
    return column;
  }
  const auto scale=centered.square().mean().sqrt().clamp_min(1e-8);
  const auto standardized_y=centered/scale;
  const auto rows=train_x.index_select(0,index);
  const auto x_center=rows.mean(0),y_center=standardized_y.mean(),x=rows-x_center;
  column.weight=at::linalg_solve(x.transpose(0,1).matmul(x)+torch::eye(x.size(1),x.options()),
      x.transpose(0,1).matmul((standardized_y-y_center).unsqueeze(1))).squeeze(1);
  column.intercept=(y_center-x_center.matmul(column.weight)).item<double>();
  column.target_mean=y.mean().item<double>();
  column.target_scale=scale.item<double>();
  column.prediction=(validation_x.matmul(column.weight)+column.intercept)*column.target_scale+column.target_mean;
  column.score=ev::regression_score(column.prediction,validation_y,validation_valid,torch::ones({validation_y.size(0)},torch::kBool));
  column.solved=column.score.r2_supported;
  if(required){
    TORCH_CHECK(column.score.supported&&column.score.r2_supported&&column.score.coverage==1.&&column.score.valid==column.score.total,
        "[family screen] strict finite complete anchor target required: ",name);
  }
  if(!column.solved){
    column.weight.zero_();column.prediction.zero_();column.target_mean=0;column.target_scale=1;column.intercept=0;
  }
  finite(column.weight,name);finite(column.prediction,name);return column;
}
void write_score(std::ostream &report,const screen::TargetSpec &spec,const ev::RegressionScore &score,bool solved){
  report<<"{\"task\":"<<quote(spec.name)<<",\"unit\":"<<quote(spec.unit)<<",\"supported\":"<<(solved?"true":"false")<<",\"r2\":";
  if(solved)report<<score.r2;else report<<"null";
  report<<",\"mae\":";if(score.supported&&solved)report<<score.mae;else report<<"null";
  report<<",\"rmse\":";if(score.supported&&solved)report<<score.rmse;else report<<"null";
  report<<",\"coverage\":"<<score.coverage<<",\"valid\":"<<score.valid<<",\"target_eligible\":"<<score.target_eligible
    <<",\"total\":"<<score.total<<'}';
}
int run(int argc,char **argv){
  TORCH_CHECK(argc==17,"[family screen] eight explicit flag/value pairs required");
  const std::set<std::string> allowed{"--settings","--checkpoint","--training-state","--train","--validation","--output","--expected-updates","--feature-view"};
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&args.emplace(argv[i],argv[i+1]).second,"[family screen] unknown or duplicate option");
  for(const auto &key:allowed)TORCH_CHECK(args.count(key),"[family screen] missing option: ",key);
  const auto view=feature_view(args.at("--feature-view"));
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
  const auto train_targets=targets(config,train,view,"TRAIN"),validation_targets=targets(config,validation,view,"validation");
  TORCH_CHECK(train_targets.anchor_id==validation_targets.anchor_id,"[family screen] TRAIN and validation must share the anchor channel");
  const auto train_rows=train.rows(),validation_rows=validation.rows();
  std::set<int64_t> train_ids;
  for(int64_t i=0;i<train_rows;++i)TORCH_CHECK(train_ids.insert(train.row_ids[i].item<int64_t>()).second,"[family screen] duplicate TRAIN row id");
  for(int64_t i=0;i<validation_rows;++i)TORCH_CHECK(!train_ids.count(validation.row_ids[i].item<int64_t>()),"[family screen] TRAIN/validation row-id overlap");
  const auto train_features=features(live,train,dims.native_width),validation_features=features(live,validation,dims.native_width);
  finite(train_features,"TRAIN features");finite(validation_features,"validation features");
  const auto mean=train_features.mean(0),scale=(train_features-mean).square().mean(0).sqrt().clamp_min(1e-8);
  const auto standardized=(train_features-mean)/scale,validation_standardized=(validation_features-mean)/scale;
  std::vector<Column> columns;
  int64_t head_fits=0;
  for(int64_t task=0;task<4;++task){
    auto column=fit_column(standardized,validation_standardized,train_targets.values.select(1,task),train_targets.valid.select(1,task),
        validation_targets.values.select(1,task),validation_targets.valid.select(1,task),task<3,screen::kTimeframeTargets[static_cast<std::size_t>(task)].name);
    if(column.solved)++head_fits;
    columns.push_back(std::move(column));
  }
  auto weights=torch::empty({standardized.size(1),4},torch::kFloat64),predictions=torch::empty({validation_rows,4},torch::kFloat64);
  auto target_mean=torch::empty({4},torch::kFloat64),target_scale=torch::empty({4},torch::kFloat64),intercept=torch::empty({4},torch::kFloat64);
  auto supported=torch::empty({4},torch::kBool),reported_metrics=torch::zeros({4,4},torch::kFloat64);
  for(int64_t task=0;task<4;++task){
    const auto &column=columns[static_cast<std::size_t>(task)];
    weights.select(1,task).copy_(column.weight);predictions.select(1,task).copy_(column.prediction);
    target_mean[task]=column.target_mean;target_scale[task]=column.target_scale;intercept[task]=column.intercept;
    supported[task]=column.solved;
    if(column.solved)reported_metrics[task].copy_(torch::tensor({column.score.r2,column.score.mae,column.score.rmse,column.score.coverage},torch::kFloat64));
  }
  finite(mean,"feature mean");finite(scale,"feature scale");finite(target_mean,"target mean");finite(target_scale,"target scale");
  finite(weights,"weights");finite(intercept,"intercept");finite(predictions,"predictions");
  TORCH_CHECK(live.model->state_identity()==before&&live.scaler.identity()==scaler&&live.model->normalizer().identity()==bank,
      "[family screen] readout must not mutate encoder or preprocessing");
  for(const auto &item:hashes)TORCH_CHECK(cp::boundary::sha256_file(args.at(item.first))==item.second,"[family screen] input changed during execution: ",item.first);
  TORCH_CHECK(cp::boundary::sha256_file(executable)==executable_sha,"[family screen] executable changed during execution");
  const fs::path out=args.at("--output");TORCH_CHECK(!fs::exists(out)&&fs::is_directory(out.parent_path()),"[family screen] new output leaf with existing parent required");
  fs::create_directory(out);
  torch::serialize::OutputArchive archive;
  archive.write("artifact_kind",ow::text_tensor("cpb_family_screen_timeframe_witness_v1"),true);
  archive.write("metric_columns",ow::text_tensor("r2,mae,rmse,coverage"),true);
  archive.write("recipe",ow::text_tensor(recipe),true);
  archive.write("group_identity",ow::text_tensor("row-id"),true);
  archive.write("feature_view",ow::text_tensor(view_name(view)),true);
  archive.write("target_names",ow::text_tensor("anchor_volatility,anchor_trend,anchor_activity,cross_resolution"),true);
  for(const auto &item:hashes){auto key=item.first.substr(2);std::replace(key.begin(),key.end(),'-','_');archive.write(key+"_sha256",ow::text_tensor(item.second),true);}
  archive.write("executable_sha256",ow::text_tensor(executable_sha),true);
  archive.write("native_width",torch::tensor(dims.native_width,torch::kInt64),true);
  archive.write("learned_width",torch::tensor(dims.shape_width,torch::kInt64),true);
  archive.write("spectral_width",torch::tensor(dims.spectral_width,torch::kInt64),true);
  archive.write("change_width",torch::tensor(dims.change_width,torch::kInt64),true);
  archive.write("causal_updates",torch::tensor(static_cast<int64_t>(updates),torch::kInt64),true);
  archive.write("anchor_channel_id",torch::tensor(train_targets.anchor_id,torch::kInt64),true);
  archive.write("history_length",torch::tensor(config.level.history_length,torch::kInt64),true);
  for(const auto &item:std::vector<std::pair<std::string,torch::Tensor>>{
      {"train_features",train_features},{"validation_features",validation_features},{"feature_mean",mean},{"feature_scale",scale},
      {"target_mean",target_mean},{"target_scale",target_scale},{"weights",weights},{"intercept",intercept},{"predictions",predictions},
      {"train_targets",train_targets.values},{"validation_targets",validation_targets.values},
      {"train_target_valid",train_targets.valid},{"validation_target_valid",validation_targets.valid},
      {"target_column_supported",supported},
      {"train_feature_valid",torch::ones({train_rows},torch::kBool)},{"validation_feature_valid",torch::ones({validation_rows},torch::kBool)},
      {"train_row_ids",train.row_ids},{"validation_row_ids",validation.row_ids},
      {"train_group_ids",train.row_ids.clone()},{"validation_group_ids",validation.row_ids.clone()},
      {"train_cutoffs",train.cutoffs},{"validation_cutoffs",validation.cutoffs},{"reported_metrics",reported_metrics}})archive.write(item.first,item.second,true);
  cp::checkpoint_detail::d::publish_checkpoint(archive,(out/"ridge-witness.pt").string());
  std::ofstream report(out/"summary.json");
  report<<std::setprecision(17)
    <<"{\"kind\":\"cpb_family_screen_timeframe_readout_v1\",\"recipe\":"<<quote(recipe)<<",\"updates\":"<<updates
    <<",\"learned_width\":"<<dims.shape_width<<",\"spectral_width\":"<<dims.spectral_width<<",\"change_width\":"<<dims.change_width
    <<",\"native_width\":"<<dims.native_width<<",\"input_feature_count\":9,\"history_length\":"<<config.level.history_length
    <<",\"anchor_channel_id\":"<<train_targets.anchor_id<<",\"feature_view\":"<<quote(view_name(view))
    <<",\"target_span\":\"boundary retain, last HISTORY_LENGTH ticks ending at the cutoff\""
    <<",\"target_contract\":\"anchor_volatility, anchor_trend, anchor_activity, cross_resolution\""
    <<",\"head\":\"TRAIN-standardized-ridge1\",\"ridge_penalty\":1,\"encoder_updates\":0,\"preprocessing_fits\":0,\"head_fits\":"<<head_fits
    <<",\"legacy_target_parity\":false,\"decision_input\":false,\"comparable_to_seed_946103\":false,\"group_identity\":\"row-id\""
    <<",\"temporal_separation\":\"A confirmation population must be a disjoint later period. This tool checks row-id disjointness only.\""
    <<",\"fresh_confirmation\":false,\"reconstruction_evaluated\":false,\"training_feature_rows\":"<<train_rows
    <<",\"validation_feature_rows\":"<<validation_rows<<",\"metrics\":[";
  for(int64_t task=0;task<4;++task){
    if(task)report<<',';
    write_score(report,screen::kTimeframeTargets[static_cast<std::size_t>(task)],columns[static_cast<std::size_t>(task)].score,columns[static_cast<std::size_t>(task)].solved);
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
  std::cout<<"Saved timeframe family-screen readouts. native_width="<<dims.native_width<<" updates="<<updates
    <<" anchor="<<train_targets.anchor_id<<" view="<<view_name(view)<<"\n";
  return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
