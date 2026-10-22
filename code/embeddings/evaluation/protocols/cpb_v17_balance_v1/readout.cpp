// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Bounded chronological development readouts. No encoder updates or fitting.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v17.h"
#include "embedding/data/observation_windows.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace cp=embedding::encoders::causal_patch_bottleneck_mae::family;
namespace version=embedding::encoders::causal_patch_bottleneck_mae::v17;
namespace ow=embedding::data::observation_windows;
namespace ev=cuwacunu::evaluation;
namespace fs=std::filesystem;
namespace {
std::string quote(const std::string &value){
  std::ostringstream out;out<<'"';for(const unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<c;
    else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}
  out<<'"';return out.str();
}
std::string artifact(const fs::path &path,const std::string &sha){
  std::ostringstream out;out<<"{\"path\":"<<quote(path.generic_string())<<",\"bytes\":"<<fs::file_size(path)<<",\"sha256\":"<<quote(sha)<<'}';return out.str();
}
void finite(const torch::Tensor &value,const char *name){
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kFloat64&&torch::isfinite(value).all().item<bool>(),"[V17 readout] finite CPU F64 required: ",name);
}
cp::RawHistory raw(const ow::Dataset &d){return{d.values,d.observed,d.channel_ids,d.timestamps,d.sampling_interval};}
torch::Tensor features(cp::Loaded &live,const ow::Dataset &data){
  std::vector<torch::Tensor> rows;torch::NoGradGuard guard;live.model->eval();
  for(int64_t first=0;first<data.rows();first+=8){std::vector<int64_t> ids;
    for(int64_t i=first;i<std::min(first+8,data.rows());++i)ids.push_back(i);
    const auto batch=ow::subset(data,ids);const auto z=live.model->encode_at(raw(batch),batch.cutoffs,live.scaler);
    TORCH_CHECK(z.valid.all().item<bool>()&&cp::boundary::same_bits(z.cutoffs,batch.cutoffs),"[V17 readout] complete declared feature support and cutoff association required");
    rows.push_back(z.native.detach().to(torch::kCPU,torch::kFloat64));
  }
  return torch::cat(rows);
}
ev::TargetRows targets(const ow::Dataset &data,const fs::path &expected){
  auto t=ev::past_window_targets(data.values.to(torch::kFloat32).to(torch::kFloat64),data.observed);
  const auto original=ev::load_targets(expected.string());
  TORCH_CHECK(t.valid.all().item<bool>()&&torch::equal(original.row_ids,data.row_ids)&&
    torch::equal(original.targets.valid,t.valid)&&cp::boundary::same_bits(original.targets.values,t.values),
    "[V17 readout] exact original frozen January target/ID/support parity required");
  finite(t.values,"targets");return{t,data.row_ids,original.group_ids};
}
int run(int argc,char **argv){
  TORCH_CHECK(argc==19,"[V17 readout] nine explicit flag/value pairs required");
  std::map<std::string,std::string> a;for(int i=1;i<argc;i+=2)TORCH_CHECK(a.emplace(argv[i],argv[i+1]).second,"duplicate flag");
  for(const auto name:{"--settings","--checkpoint","--weight","--train","--validation","--expected-train-targets","--expected-validation-targets","--output","--expected-updates"})TORCH_CHECK(a.count(name),"missing option: ",name);
  torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard guard;
  std::map<std::string,std::string> hashes;
  for(const auto key:{"--settings","--checkpoint","--train","--validation","--expected-train-targets","--expected-validation-targets"}){
    TORCH_CHECK(fs::is_regular_file(a.at(key)),"[V17 readout] input must exist: ",key);hashes.emplace(key,cp::boundary::sha256_file(a.at(key)));
  }
  const auto executable=fs::canonical("/proc/self/exe");const auto executable_sha=cp::boundary::sha256_file(executable);
  const auto config=cp::read_settings(a.at("--settings"));size_t used=0;const double weight=std::stod(a.at("--weight"),&used);
  TORCH_CHECK(used==a.at("--weight").size(),"exact weight parse required");const version::Recipe recipe{weight};version::validate_recipe(recipe);
  auto live=version::load_checkpoint(a.at("--checkpoint"),config.level.device,&config,recipe).numerical;
  used=0;const auto updates=std::stoll(a.at("--expected-updates"),&used);
  TORCH_CHECK(used==a.at("--expected-updates").size()&&updates==256&&live.causal_updates==updates,"exact final256 only");
  const auto before=live.model->state_identity();const auto scaler=live.scaler.identity(),bank=live.model->normalizer().identity();
  const auto train=ow::read(a.at("--train")),validation=ow::read(a.at("--validation"));
  TORCH_CHECK(train.role=="TRAIN"&&validation.role=="VALIDATION"&&train.rows()==112&&validation.rows()==32,"fixed112/32 declared roles");
  const auto ty=targets(train,a.at("--expected-train-targets")),vy=targets(validation,a.at("--expected-validation-targets"));
  std::set<int64_t> train_ids,train_groups;
  for(int64_t i=0;i<112;++i){train_ids.insert(ty.row_ids[i].item<int64_t>());train_groups.insert(ty.group_ids[i].item<int64_t>());}
  for(int64_t i=0;i<32;++i)TORCH_CHECK(!train_ids.count(vy.row_ids[i].item<int64_t>())&&!train_groups.count(vy.group_ids[i].item<int64_t>()),"[V17 readout] TRAIN/validation row or group overlap");
  const auto tx=features(live,train),vx=features(live,validation);
  TORCH_CHECK(tx.sizes()==torch::IntArrayRef({112,36})&&vx.sizes()==torch::IntArrayRef({32,36})&&
      ty.targets.values.sizes()==torch::IntArrayRef({112,4})&&vy.targets.values.sizes()==torch::IntArrayRef({32,4}),"[V17 readout] declared native36/four-target shapes");
  finite(tx,"TRAIN features");finite(vx,"validation features");
  const auto mean=tx.mean(0),scale=(tx-mean).square().mean(0).sqrt().clamp_min(1e-8);
  const auto x=(tx-mean)/scale,v=(vx-mean)/scale;
  const auto ym=ty.targets.values.mean(0),ys=(ty.targets.values-ym).square().mean(0).sqrt().clamp_min(1e-8);
  TORCH_CHECK((ty.targets.values-ym).square().sum(0).gt(0).all().item<bool>(),"[V17 readout] zero TRAIN target variance abstains");
  const auto y=(ty.targets.values-ym)/ys,xm=x.mean(0),yc=y.mean(0);
  const auto centered=x-xm;
  const auto weights=at::linalg_solve(centered.transpose(0,1).matmul(centered)+torch::eye(x.size(1),x.options()),centered.transpose(0,1).matmul(y-yc));
  const auto intercept=yc-xm.matmul(weights),predictions=(v.matmul(weights)+intercept)*ys+ym;
  for(const auto &item:std::vector<std::pair<const char *,torch::Tensor>>{{"feature mean",mean},{"feature scale",scale},{"target mean",ym},{"target scale",ys},{"weights",weights},{"intercept",intercept},{"predictions",predictions}})finite(item.second,item.first);
  std::vector<ev::RegressionScore> scores;auto reported_metrics=torch::empty({4,4},torch::kFloat64);
  const auto names=ev::past_window_target_specs();TORCH_CHECK(names.size()==4,"four fixed targets required");
  for(int64_t j=0;j<4;++j){
    auto score=ev::regression_score(predictions.select(1,j),vy.targets.values.select(1,j),vy.targets.valid.select(1,j),torch::ones({32},torch::kBool));
    TORCH_CHECK(score.supported&&score.r2_supported&&score.total==32&&score.target_eligible==32&&score.valid==32&&score.coverage==1.,"[V17 readout] strict finite complete scored target required");
    scores.push_back(score);reported_metrics[j].copy_(torch::tensor({score.r2,score.mae,score.rmse,score.coverage},torch::kFloat64));
  }
  TORCH_CHECK(live.model->state_identity()==before&&live.scaler.identity()==scaler&&live.model->normalizer().identity()==bank,"readout must not mutate encoder/preprocessing");
  for(const auto &item:hashes)TORCH_CHECK(cp::boundary::sha256_file(a.at(item.first))==item.second,"[V17 readout] input changed during execution: ",item.first);
  TORCH_CHECK(cp::boundary::sha256_file(executable)==executable_sha,"[V17 readout] executable changed during execution");
  const fs::path out=a.at("--output");TORCH_CHECK(!fs::exists(out)&&fs::is_directory(out.parent_path()),"new leaf required");fs::create_directory(out);
  torch::serialize::OutputArchive archive;
  archive.write("artifact_kind",ow::text_tensor("cpb_v17_ridge_witness_v1"),true);
  archive.write("metric_columns",ow::text_tensor("r2,mae,rmse,coverage"),true);
  for(const auto &item:hashes){auto key=item.first.substr(2);std::replace(key.begin(),key.end(),'-','_');archive.write(key+"_sha256",ow::text_tensor(item.second),true);}
  archive.write("executable_sha256",ow::text_tensor(executable_sha),true);
  for(const auto &item:std::vector<std::pair<std::string,torch::Tensor>>{{"train_features",tx},{"validation_features",vx},{"feature_mean",mean},{"feature_scale",scale},{"target_mean",ym},{"target_scale",ys},{"weights",weights},{"intercept",intercept},{"predictions",predictions},{"train_targets",ty.targets.values},{"validation_targets",vy.targets.values},{"train_target_valid",ty.targets.valid},{"validation_target_valid",vy.targets.valid},{"train_feature_valid",torch::ones({112},torch::kBool)},{"validation_feature_valid",torch::ones({32},torch::kBool)},{"train_row_ids",train.row_ids},{"validation_row_ids",validation.row_ids},{"train_group_ids",ty.group_ids},{"validation_group_ids",vy.group_ids},{"train_cutoffs",train.cutoffs},{"validation_cutoffs",validation.cutoffs},{"reported_metrics",reported_metrics}})archive.write(item.first,item.second,true);
  cp::checkpoint_detail::d::publish_checkpoint(archive,(out/"ridge-witness.pt").string());
  std::ofstream report(out/"summary.json");report<<std::setprecision(17)<<"{\"kind\":\"cpb_v17_early_development_linear_readout_v1\",\"updates\":256,\"temporal_weight\":"<<weight<<",\"target_precision_parity\":true,\"head\":\"TRAIN-standardized-ridge1\",\"encoder_updates\":0,\"preprocessing_fits\":0,\"head_fits\":4,\"metrics\":[";
  for(int64_t j=0;j<4;++j){const auto &score=scores[j];if(j)report<<',';
    report<<"{\"task\":"<<quote(names[j].name)<<",\"r2\":"<<score.r2<<",\"mae\":"<<score.mae<<",\"rmse\":"<<score.rmse<<",\"coverage\":"<<score.coverage<<",\"valid\":"<<score.valid<<",\"target_eligible\":"<<score.target_eligible<<",\"total\":"<<score.total<<'}';
  }
  report<<"],\"encoder_state_unchanged\":true,\"fresh_confirmation\":false,\"reconstruction_evaluated\":false,\"training_feature_rows\":112,\"validation_feature_rows\":32,\"native_width\":36,\"model_state_identity\":"<<quote(before)<<",\"scaler_identity\":"<<quote(scaler)<<",\"bank_identity\":"<<quote(bank)<<",\"inputs\":{";
  bool comma=false;for(const auto &item:hashes){if(comma)report<<',';comma=true;report<<quote(item.first.substr(2))<<':'<<artifact(a.at(item.first),item.second);}
  report<<"},\"executable\":"<<artifact(executable,executable_sha)<<",\"witness\":"<<artifact(out/"ridge-witness.pt",cp::boundary::sha256_file(out/"ridge-witness.pt"))<<",\"input_bytes_unchanged\":true}";
  report.close();TORCH_CHECK(report,"write report");
  std::cout<<"Saved four linear development readouts with exact original target parity.\n";return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
