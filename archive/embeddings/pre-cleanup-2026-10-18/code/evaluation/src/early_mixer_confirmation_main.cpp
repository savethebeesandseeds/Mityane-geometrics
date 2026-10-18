// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/early_mixer_confirmation_adapter.h"
#include "embedding/shared/paired_pooling.h"
#include "embedding/shared/fixed_feature_readouts.h"
#include "frozen_role_guard.h"
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <regex>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
namespace {
namespace fs = std::filesystem;
namespace ev = embedding::evaluation;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace ev::frozen_inputs;
using Clock = std::chrono::steady_clock;
const std::array<uint64_t,5> timing_masters{64161,65262,66363,67464,68565};
const std::array<uint64_t,5> amplitude_masters{69666,70767,71868,72969,74070};
const std::string protocol = "early-mixer-confirmation-v1";
const std::string card_relative = "code/evaluation/cards/early_mixer_confirmation_v1.md";
const std::string card_sha = "98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea";
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
std::string plan() {
  return "{\"protocol\":"+quote(protocol)+",\"timing_master_seeds\":[64161,65262,66363,67464,68565],"
      "\"amplitude_data_master_seeds\":[69666,70767,71868,72969,74070],\"tags\":[\"RPB-v7.alt-04\",\"RPB-v10.alt-04\"],"
      "\"encoder_trajectories\":10,\"encoder_updates_each\":512,\"attempt_limit\":1024,\"log_every\":1,\"skipped_attempts_permitted\":false,\"retained_points\":20,"
      "\"decoder_updates\":0,\"extra_decoder_calibration_updates\":0,\"decoder_update_scope\":\"joint reconstruction during encoder updates; no extra decoder-only calibration\",\"batch_size\":8,\"sampled_rows\":40960,"
      "\"train_pairs\":128,\"validation_pairs\":64,\"test_pairs\":0,\"shape\":[3,32,3],\"native_width\":32,"
      "\"parameter_count_each\":225805,\"channel_mixer_placements\":[0,1],\"global_pool_input_sources\":[0,0],\"checkpoint_roles_per_point\":5,"
      "\"external_fit_protocol\":\"early-mixer-confirmation-v1/lag_sign\",\"implementation_fit_protocol\":\"early-mixer-reliability-v1/lag_sign\",\"point_binding_suffix\":\".confirmation.pt\","
      "\"head_repetitions\":[2701,2802,2903],\"planned_pipelines\":210,\"planned_heads\":420,"
      "\"deletion_rate\":0.30,\"methods_each_task\":7,\"separate_initial_controls\":2,\"planned_unique_quality_native_exports\":120,"
      "\"planned_initial_counterpart_exports\":0,\"planned_native_export_calls\":120,\"planned_query_writer_calls\":20,\"planned_query_forwards\":80,"
      "\"driver_raw_outer_fits\":10,\"helper_outer_train_fits\":50,\"raw_outer_fit_shared_with_PCA\":true,\"quality_gain\":false,"
      "\"encoder_device\":\"CUDA\",\"prior_quality_input_roles\":[],\"quality_generated\":false,"
      "\"testing_accessed\":false,\"stress_accessed\":false,\"selection\":false,\"promotion\":false,\"source_fingerprint\":"+
      quote(EVALUATION_SOURCE_ID)+",\"human_card_sha256\":"+quote(card_sha)+"}";
}
void archive(const fs::path &path,torch::serialize::OutputArchive &out) { require(!fs::exists(path),"new archive required");out.save_to(path.string()); }
torch::Tensor text_tensor(const std::string &s) { return torch::tensor(std::vector<uint8_t>(s.begin(),s.end()),torch::kUInt8); }
void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &why) {
  const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
  require(x.scalar_type()==y.scalar_type()&&x.sizes()==y.sizes(),why+" schema");
  require(!x.numel()||std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size())==0,why+" bytes");
}
void sanitize(ev::ControlledDataset &d,const std::string &task) {
  d.observed.data=torch::where(d.observed.feature_mask,d.observed.data,torch::zeros_like(d.observed.data)).detach().clone();
  d.observed.feature_mask=d.observed.feature_mask.clone();d.clean=d.observed;
  for(auto &id:d.source_ids)id=protocol+'/'+task+'/'+id;
}
void save_observations(const fs::path &path,const ev::ControlledDataset &d,const torch::Tensor &erasure={}) {
  torch::serialize::OutputArchive a;a.write("observations",d.observed.data,true);a.write("feature_mask",d.observed.feature_mask,true);
  a.write("labels_scoring_only",d.labels,true);a.write("source_ids_json",text_tensor(strings(d.source_ids)),true);
  if(erasure.defined())a.write("requested_erasure",erasure,true);
  archive(path,a);
}
void initial_pair(const fs::path &late,const fs::path &early,const fs::path &witness) {
  auto x=rpb::load_checkpoint(late.string(),torch::kCUDA),y=rpb::load_checkpoint(early.string(),torch::kCUDA);
  require(x.settings.model.channel_mixer_placement==0&&y.settings.model.channel_mixer_placement==1&&
      x.attempted_steps==0&&y.attempted_steps==0&&x.completed_steps==0&&y.completed_steps==0,"paired initial architecture/counters");
  auto common=y.settings;common.model.channel_mixer_placement=0;
  require(rpb::settings_text(x.settings)==rpb::settings_text(common),"only mixer placement differs initially");
  const auto xp=x.model->named_parameters(),yp=y.model->named_parameters();require(xp.size()==yp.size(),"parameter name count");
  std::vector<std::string> xnames,ynames;
  for(const auto &p:xp)xnames.push_back(p.key());
  for(const auto &p:yp)ynames.push_back(p.key());
  require(xnames==ynames,"exact parameter registration order");
  for(const auto &p:xp){require(yp.contains(p.key()),"paired parameter name");exact(p.value(),yp[p.key()],p.key());}
  const auto xb=x.model->named_buffers(),yb=y.model->named_buffers();require(xb.size()==yb.size(),"buffer name count");
  xnames.clear();ynames.clear();for(const auto &p:xb)xnames.push_back(p.key());for(const auto &p:yb)ynames.push_back(p.key());
  require(xnames==ynames,"exact buffer registration order");
  for(const auto &p:xb){require(yb.contains(p.key()),"paired buffer name");exact(p.value(),yb[p.key()],p.key());}
  for(const auto &[a,b]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{x.scaler.mean,y.scaler.mean},{x.scaler.scale,y.scaler.scale},
      {x.scaler.count,y.scaler.count},{x.scaler.channel_ids,y.scaler.channel_ids},{x.scaler.floor_applied,y.scaler.floor_applied}})exact(a,b,"paired scaler");
  require(x.dataset_id==y.dataset_id&&x.schema_id==y.schema_id&&x.scaler_fit_dataset_id==y.scaler_fit_dataset_id,"paired fitting identity");
  torch::serialize::OutputArchive saved;
  for(const auto &[name,cp]:std::vector<std::pair<std::string,rpb::Checkpoint*>>{{"late",&x},{"early",&y}}){
    torch::serialize::OutputArchive parameters,buffers,scaler;int64_t i=0;
    for(const auto &p:cp->model->named_parameters()){torch::serialize::OutputArchive item;item.write("parameter_name",text_tensor(p.key()),true);
      item.write("value",p.value().detach().to(torch::kCPU).clone(),true);parameters.write("tensor_"+std::to_string(i++),item);}parameters.write("count",torch::tensor(i),true);i=0;
    for(const auto &p:cp->model->named_buffers()){torch::serialize::OutputArchive item;item.write("parameter_name",text_tensor(p.key()),true);
      item.write("value",p.value().detach().to(torch::kCPU).clone(),true);buffers.write("tensor_"+std::to_string(i++),item);}buffers.write("count",torch::tensor(i),true);cp->scaler.save(scaler);
    saved.write(name+"_parameters",parameters);saved.write(name+"_buffers",buffers);saved.write(name+"_scaler",scaler);
  }archive(witness,saved);
}
std::string progress_json(const ev::CurveProgress &p) {
  std::ostringstream s;s<<std::setprecision(17)<<"{\"attempted\":"<<p.attempted<<",\"completed\":"<<p.completed<<",\"sampled_rows\":"<<p.sampled_rows
    <<",\"parameter_count\":"<<p.parameter_count<<",\"cuda_parameter_count\":"<<p.cuda_parameter_count<<",\"training_seconds\":"<<p.training_seconds
    <<",\"training_device\":"<<quote(p.training_device)<<",\"training_dataset_id\":"<<quote(p.training_dataset_id)<<",\"preprocessing_id\":"<<quote(p.preprocessing_id)
    <<",\"last_input_cuda\":"<<(p.last_input_cuda?"true":"false")<<",\"last_loss_cuda\":"<<(p.last_loss_cuda?"true":"false")
    <<",\"finite_gradients\":"<<(p.finite_gradients?"true":"false")<<",\"weights_changed\":"<<(p.weights_changed?"true":"false")<<",\"losses\":[";
  bool first=true;for(const auto &l:p.losses){if(!first)s<<',';first=false;s<<'['<<l.attempted<<','<<l.completed<<','<<l.target_cells<<','<<l.loss<<','<<l.gradient_norm<<']';}return s.str()+"]}";
}
torch::Tensor read(torch::serialize::InputArchive &a,const std::string &key) { torch::Tensor t;a.read(key,t,true);return t; }
void paired_queries(const fs::path &late,const fs::path &early) {
  torch::serialize::InputArchive a,b;a.load_from(late.string(),torch::kCPU);b.load_from(early.string(),torch::kCPU);
  for(const std::string key:{"standardized_target","target_mask","requested_observed_target_mask","visible_mask","trial_channel_eligible",
      "channel_target_counts","channel_valid","example_valid","source_ids_json"})exact(read(a,key),read(b,key),"paired original query "+key);
}
ev::FeatureSurface raw(const ev::ObservationScaler &s,const embedding::Batch &b) {
  const auto z=s.transform(b);return {torch::cat({z.data.flatten(1),z.feature_mask.flatten(1).to(torch::kFloat64)},1),
      b.feature_mask.flatten(1).any(1),"task TRAIN ObservationScaler; legal observed values+visibility"};
}
ev::FeatureSurface mask(const embedding::Batch &b) { return {b.feature_mask.flatten(1).to(torch::kFloat64),b.feature_mask.flatten(1).any(1),"visibility metadata only"}; }
ev::FixedFeatureMethod pca_method(const fs::path &path,const ev::FeatureNormalizer &outer,
    const ev::FeatureSurface &a,const ev::FeatureSurface &b,const ev::FeatureSurface &c) {
  const auto zero=[](const ev::FeatureSurface &x){return ev::FeatureSurface{torch::zeros({x.values.size(0),32},torch::kFloat64),x.valid.clone(),"unsupported raw TRAIN PCA32; no fit"};};
  try {
    ev::TrainPca pca(a,32);torch::serialize::OutputArchive pc;
    pc.write("feature_mean",outer.mean,true);pc.write("feature_scale",outer.scale,true);pc.write("fitted_rows",torch::tensor(outer.fitted_rows),true);
    pc.write("pca_mean",pca.mean,true);pc.write("pca_components",pca.components,true);pc.write("pca_singular_values",pca.singular_values,true);pc.write("pca_numerical_rank",torch::tensor(pca.numerical_rank),true);archive(path/"pca-preprocessing.pt",pc);
    write_new(path/"pca-status.json","{\"status\":\"measured\",\"width\":32,\"native_PCA\":false}\n");
    return {"pca_only",pca.transform(a),pca.transform(b),pca.transform(c),true,{}};
  }catch(const std::exception &e){
    const std::string why=e.what();
    require(why.find("PCA dimensions exceed numerical training rank")!=std::string::npos||
        why.find("PCA dimensions exceed valid centered training-row bound")!=std::string::npos||
        why.find("feature fit requires two valid training rows")!=std::string::npos,"unexpected raw PCA error: "+why);
    require(!fs::exists(path/"pca-preprocessing.pt"),"PCA failed after saving preprocessing; preserve failed capsule");
    write_new(path/"pca-status.json","{\"status\":\"unsupported_fit\",\"width\":32,\"native_PCA\":false,\"reason\":"+quote(e.what())+"}\n");
    return {"pca_only",zero(a),zero(b),zero(c),true,why};
  }
}
int64_t fitted_pipelines(const std::string &json) {
  const auto at=json.rfind("\"fit_counts\"");require(at!=std::string::npos,"shared fit counts required");
  std::smatch result;const auto tail=json.substr(at);require(std::regex_search(tail,result,std::regex("\"ridge_fits\":([0-9]+)")),"ridge fit count");
  const auto ridge=std::stoll(result[1]);require(std::regex_search(tail,result,std::regex("\"tiny_fits\":([0-9]+)"))&&std::stoll(result[1])==ridge,"paired head fit count");return ridge;
}
int64_t outer_fit_count(const std::string &json) {
  const auto at=json.rfind("\"fit_counts\"");require(at!=std::string::npos,"shared fit counts required");
  std::smatch result;const auto tail=json.substr(at);
  require(std::regex_search(tail,result,std::regex("\"outer_train_normalizer_fits\":([0-9]+)")),"outer fit count");
  return std::stoll(result[1]);
}
struct TaskData { ev::ControlledDataset train,val,deleted; };
TaskData generate(ev::Task task,uint64_t master,const fs::path &root,const ev::NativeCurveRun &recipe) {
  const std::string name=task==ev::Task::lag_sign?"lag_sign":"amplitude";
  auto data=ev::make_controlled_development_protocol(task,recipe.card.shape,128,64,master,.1);sanitize(data.training,name);sanitize(data.validation,name);
  require(!data.testing.observed.data.defined()&&data.testing.source_ids.empty(),"no TEST generation");
  const auto stream=task==ev::Task::lag_sign?0x656d633174696d30ULL:0x656d6331616d7030ULL;
  const auto view=ev::make_coordinate_deletion_view(data.validation.observed,data.validation.source_ids,task,ev::stream_seed(master,stream),.30,
      protocol+"/"+name+"/validation-coordinate-dropout");
  auto dropped=data.validation;dropped.observed=view.observations;dropped.clean=dropped.observed;
  save_observations(root/"controlled-training.pt",data.training);save_observations(root/"controlled-validation.pt",data.validation);
  save_observations(root/"controlled-validation-deleted.pt",dropped,view.requested_erasure);
  return {std::move(data.training),std::move(data.validation),std::move(dropped)};
}
} // namespace
int main(int argc,char **argv) { try {
  if(argc==2&&std::string(argv[1])=="--source-id"){std::cout<<EVALUATION_SOURCE_ID<<'\n';return 0;}
  if(argc==2&&std::string(argv[1])=="--plan"){std::cout<<plan()<<'\n';return 0;}
  std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2){require(i+1<argc,"missing option value");require(args.emplace(argv[i],argv[i+1]).second,"duplicate option");}
  require(args.size()==5&&args.count("--output")&&args.count("--admission-log")&&args.count("--admission-sha256")&&
      args.count("--card")&&args.count("--card-sha256"),"five fixed measurement options required");
  const fs::path output=args.at("--output");
  require(output.is_absolute()&&output.filename()=="results"&&
      output.parent_path().parent_path()==fs::path("/embedding/output/runs/rpb-early-mixer-confirmation")&&
      fs::canonical(output.parent_path())==output.parent_path()&&
      output.parent_path().filename().string().rfind("early-mixer-confirmation-",0)==0,"exclusive confirmation capsule bounds");
  require(fs::path(args.at("--admission-log"))==output.parent_path()/"admission/build-and-tests.log","captured admission-log path required");
  for(const auto &path:std::array<fs::path,2>{args.at("--card"),args.at("--admission-log")})
    require(fs::is_regular_file(path)&&!fs::is_symlink(path)&&fs::hard_link_count(path)==1&&fs::canonical(path)==path,
        "whole captured card/log matrix must be regular, canonical and non-aliased before reading bytes");
  require(fs::path(args.at("--card"))==output.parent_path()/"source"/card_relative&&args.at("--card-sha256")==card_sha&&
      sha256(bytes(args.at("--card")))==card_sha&&is_sha(EVALUATION_SOURCE_ID),"captured frozen card/source binding");
  const auto log=bytes(args.at("--admission-log"));require(sha256(log)==args.at("--admission-sha256")&&log.find(EVALUATION_SOURCE_ID)!=std::string::npos&&
      log.find("Early mixer model CUDA admission passed")!=std::string::npos&&log.find("Early mixer CUDA adapter admission passed")!=std::string::npos&&
      log.find("Early mixer confirmation CUDA admission passed")!=std::string::npos&&log.find("Frozen role guard checks passed")!=std::string::npos&&
      log.find("Fixed feature readout tests passed")!=std::string::npos,"actual source-bound CUDA admission");
  require(torch::cuda::is_available(),"CUDA required; no CPU model fallback");torch::set_num_threads(1);
  require(!fs::exists(output)&&fs::create_directory(output),"exclusive new measurement directory");
  write_new(output/"recipe-plan.json",plan()+"\n");write_new(output/"card.json","{\"protocol\":"+quote(protocol)+",\"human_card_sha256\":"+quote(card_sha)+
      ",\"source_fingerprint\":"+quote(EVALUATION_SOURCE_ID)+",\"historical_payload_roles\":0,\"testing_accessed\":false,\"stress_accessed\":false}\n");
  ev::NativeCurveRun recipe;recipe.card.shape={3,32,3,torch::kFloat64,torch::kCPU};recipe.card.channel_ids={0,1,2};recipe.card.feature_units="unitless,unitless,unitless";
  recipe.card.sampling_interval=1;recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=0;recipe.development_only=true;recipe.stress_sweep=false;
  auto settings=rpb::default_settings();settings.steps=512;settings.attempt_limit=1024;settings.log_every=1;settings.batch_size=8;settings.threads=1;
  settings.learning_rate=.001;settings.weight_decay=.0001;settings.gradient_clip_norm=1;settings.model.channel_mixer_layers=1;settings.model.global_bottleneck_mode=2;
  settings.model.export_width=32;settings.model.device=torch::kCUDA;settings.model.channel_mixer_placement=0;const auto late_factory=rpb::make_early_mixer_confirmation_trainer(settings);
  settings.model.channel_mixer_placement=1;const auto early_factory=rpb::make_early_mixer_confirmation_trainer(settings);
  std::ostringstream report;report<<std::setprecision(17)<<"{\"protocol\":"<<quote(protocol)<<",\"source_fingerprint\":"<<quote(EVALUATION_SOURCE_ID)<<",\"cohorts\":[";
  int64_t actual_pipelines=0,actual_helper_outer_fits=0;bool first_cohort=true;
  for(size_t index=0;index<timing_masters.size();++index){
    const auto master=timing_masters[index],amp_master=amplitude_masters[index];const auto root=output/("seed-"+std::to_string(master)+"-lag_sign");
    require(fs::create_directory(root),"new cohort directory");auto start=Clock::now();auto timing=generate(ev::Task::lag_sign,master,root,recipe);double generation_io_seconds=elapsed(start);
    ev::ProviderFitInput fit{timing.train.observed,recipe.card.shape,master,timing.train.source_ids,recipe.card.channel_ids,recipe.card.feature_units,protocol+"/lag_sign",1,31};
    start=Clock::now();auto late=late_factory(fit),early=early_factory(fit);double binding_io_seconds=elapsed(start);
    const std::array<std::string,2> names{"late","early"};const std::array<ev::CurveTrainer*,2> trainers{&late,&early};
    start=Clock::now();for(size_t m=0;m<2;++m){require(fs::create_directory(root/names[m]),"new model directory");require(fs::create_directory(root/names[m]/"point-0"),"new point0");
      trainers[m]->save_checkpoint((root/names[m]/"point-0/checkpoint.pt").string());}
    binding_io_seconds+=elapsed(start);
    const auto options=[&](size_t role,const std::string &path,int64_t updates) {
      return rpb::EarlyMixerConfirmationSnapshotOptions{path,static_cast<int64_t>(role),updates,rpb::EarlyMixerConfirmationScope::quality,
          trainers[role]->audit_fields.at("core_writer_source_fingerprint"),trainers[role]->audit_fields.at("training_producer_source_fingerprint")};
    };
    start=Clock::now();const auto paired_admission=rpb::audit_early_mixer_confirmation_initialization(
        options(0,(root/"late/point-0/checkpoint.pt").string(),0),options(1,(root/"early/point-0/checkpoint.pt").string(),0),fit);
    write_new(root/"initialization-audit.json",fields(paired_admission)+"\n");binding_io_seconds+=elapsed(start);
    start=Clock::now();initial_pair(root/"late/point-0/checkpoint.pt",root/"early/point-0/checkpoint.pt",root/"initial-pair.pt");binding_io_seconds+=elapsed(start);
    std::array<ev::CurveProgress,2> progress;
    for(size_t m=0;m<2;++m){
      progress[m]=trainers[m]->train_to(512);const auto &p=progress[m];require(p.attempted==512&&p.completed==512&&p.sampled_rows==4096&&p.parameter_count==225805&&
          p.cuda_parameter_count==225805&&p.last_input_cuda&&p.last_loss_cuda&&p.finite_gradients&&p.weights_changed&&p.losses.size()==512,"complete CUDA encoder trajectory");
      require(fs::create_directory(root/names[m]/"point-512"),"new point512");start=Clock::now();trainers[m]->save_checkpoint((root/names[m]/"point-512/checkpoint.pt").string());
      write_new(root/names[m]/"encoder-progress.json",progress_json(p)+"\n");write_new(root/names[m]/"trainer-audit.json",fields(trainers[m]->audit_fields)+"\n");binding_io_seconds+=elapsed(start);
      std::cout<<"Early mixer confirmation master="<<master<<" model="<<names[m]<<" CUDA_updates="<<p.completed<<" training_seconds="<<p.training_seconds<<'\n'<<std::flush;
    }
    for(size_t step=0;step<512;++step)require(progress[0].losses[step].attempted==progress[1].losses[step].attempted&&
        progress[0].losses[step].completed==progress[1].losses[step].completed&&progress[0].losses[step].target_cells==progress[1].losses[step].target_cells,"exact paired per-step counters");
    torch::serialize::InputArchive late_audit,early_audit;
    late_audit.load_from((root/"late/point-512/checkpoint.pt.audit.pt").string(),torch::kCPU);
    early_audit.load_from((root/"early/point-512/checkpoint.pt.audit.pt").string(),torch::kCPU);
    for(const std::string key:{"context_requested_deleted_coordinates","context_actual_deleted_coordinates","context_restored_coordinates"})
      exact(read(late_audit,key),read(early_audit,key),"paired cumulative context count "+key);
    start=Clock::now();std::array<ev::CurveSnapshot,4> snapshots{late.snapshot((root/"late/point-0/checkpoint.pt").string()),early.snapshot((root/"early/point-0/checkpoint.pt").string()),
        late.snapshot((root/"late/point-512/checkpoint.pt").string()),early.snapshot((root/"early/point-512/checkpoint.pt").string())};
    for(size_t n=0;n<4;++n){const auto path=root/names[n%2]/(n<2?"point-0":"point-512")/"snapshot-assets";
      require(fs::create_directory(path),"new snapshot audit directory");snapshots[n].features.save_assets(path.string());}
    binding_io_seconds+=elapsed(start);double query_seconds=0;std::array<std::string,4> reconstruction;
    for(size_t m=0;m<2;++m){start=Clock::now();const auto path=root/names[m]/"point-512";
      reconstruction[2*m]=ev::write_native_patch_reconstruction((path/"training-reconstruction.pt").string(),timing.train,snapshots[m+2],recipe);
      reconstruction[2*m+1]=ev::write_native_patch_reconstruction((path/"validation-reconstruction.pt").string(),timing.val,snapshots[m+2],recipe);query_seconds+=elapsed(start);}
    paired_queries(root/"late/point-512/training-reconstruction.pt",root/"early/point-512/training-reconstruction.pt");
    paired_queries(root/"late/point-512/validation-reconstruction.pt",root/"early/point-512/validation-reconstruction.pt");
    require(fs::create_directory(root/"amplitude"),"new amplitude directory");start=Clock::now();auto amplitude=generate(ev::Task::amplitude,amp_master,root/"amplitude",recipe);generation_io_seconds+=elapsed(start);
    require(fs::create_directory(root/"lag_sign"),"new timing readout parent");std::ostringstream tasks;tasks<<'[';double native_seconds=0,baseline_seconds=0,head_seconds=0;
    for(size_t task=0;task<2;++task){auto &data=task?amplitude:timing;const auto path=root/(task?"amplitude":"lag_sign");const auto task_master=task?amp_master:master;
      start=Clock::now();ev::ObservationScaler scaler(data.train.observed);torch::serialize::OutputArchive sc;sc.write("mean",scaler.mean,true);sc.write("scale",scaler.scale,true);sc.write("counts",scaler.counts,true);archive(path/"raw-scaler.pt",sc);
      auto rt=raw(scaler,data.train.observed),rv=raw(scaler,data.val.observed),rd=raw(scaler,data.deleted.observed);
      ev::FeatureNormalizer raw_outer(rt);torch::serialize::OutputArchive outer;
      outer.write("feature_mean",raw_outer.mean,true);outer.write("feature_scale",raw_outer.scale,true);outer.write("fitted_rows",torch::tensor(raw_outer.fitted_rows),true);
      outer.write("fit_count",torch::tensor(int64_t(1)),true);outer.write("training_source_ids_json",text_tensor(strings(data.train.source_ids)),true);archive(path/"raw-outer-normalizer.pt",outer);
      rt=raw_outer.transform(rt);rv=raw_outer.transform(rv);rd=raw_outer.transform(rd);
      std::vector<ev::FixedFeatureMethod> methods{{"raw",rt,rv,rd,true,{}},{"mask_metadata",mask(data.train.observed),mask(data.val.observed),mask(data.deleted.observed),false,{}}};
      methods.push_back(pca_method(path,raw_outer,rt,rv,rd));baseline_seconds+=elapsed(start);
      const std::array<std::string,4> method_names{"untrained_late","untrained_early","native_late","native_early"};
      start=Clock::now();for(size_t n=0;n<4;++n)methods.push_back({method_names[n],ev::extract_native_global(snapshots[n],data.train.observed,recipe),
          ev::extract_native_global(snapshots[n],data.val.observed,recipe),ev::extract_native_global(snapshots[n],data.deleted.observed,recipe),false,{}});native_seconds+=elapsed(start);
      ev::FixedFeatureReadoutRun heads;heads.output_directory=(path/"readouts").string();heads.master_seed=task_master;heads.training_labels=data.train.labels;heads.validation_labels=data.val.labels;
      heads.training_source_ids=data.train.source_ids;heads.validation_source_ids=data.val.source_ids;heads.methods=std::move(methods);heads.comparison_reference="native_late";heads.comparison_candidate="native_early";
      start=Clock::now();const auto scores=ev::run_fixed_feature_readouts(heads);head_seconds+=elapsed(start);actual_pipelines+=fitted_pipelines(scores);
      const auto helper_outer=outer_fit_count(scores);require(helper_outer<=5,"no duplicated helper outer fits");actual_helper_outer_fits+=helper_outer;
      write_new(path/"preprocessing-fit-counts.json","{\"ObservationScaler_fits\":1,\"driver_raw_outer_fits\":1,\"driver_raw_outer_shared_with_PCA\":true,\"helper_outer_train_fits\":"+
          std::to_string(helper_outer)+",\"planned_helper_outer_train_fits\":5,\"validation_fits\":0,\"native_PCA_fits\":0}\n");
      if(task)tasks<<',';
      tasks<<"{\"task\":"<<quote(task?"amplitude":"lag_sign")<<",\"data_master\":"<<task_master<<",\"readouts\":"<<scores<<'}';
    }tasks<<']';
    write_new(root/"initial-pair.json","{\"full_named_parameters_exact\":true,\"buffers_exact\":true,\"scaler_exact\":true,\"initial_features_assumed_equal\":false,\"initial_control_sets\":2,\"per_step_trace_counts_exact\":true,\"initial_shared_state_exact_before_training_and_heads\":true}\n");
    if(!first_cohort)report<<',';
    first_cohort=false;report<<"{\"timing_master\":"<<master<<",\"amplitude_data_master\":"<<amp_master<<",\"encoder_points\":[";
    for(size_t m=0;m<2;++m){if(m)report<<',';report<<"{\"model_tag\":"<<quote(m?"RPB-v10.alt-04":"RPB-v7.alt-04")<<",\"placement\":"<<m<<",\"encoder_progress\":"<<progress_json(progress[m])
        <<",\"training_reconstruction\":"<<reconstruction[2*m]<<",\"validation_reconstruction\":"<<reconstruction[2*m+1]<<'}';}
    report<<"],\"tasks\":"<<tasks.str()<<",\"costs\":{\"generation_and_observation_io_seconds\":"<<generation_io_seconds<<",\"binding_and_checkpoint_io_seconds\":"<<binding_io_seconds
        <<",\"CUDA_query_transfer_verification_io_seconds\":"<<query_seconds<<",\"CUDA_native_transfer_verification_seconds\":"<<native_seconds<<",\"CPU_baseline_preparation_io_seconds\":"<<baseline_seconds
        <<",\"CPU_head_bootstrap_io_seconds\":"<<head_seconds<<"}}";
    std::cout<<"Completed both task readouts for master="<<master<<'\n'<<std::flush;
  }
  write_new(output/"report.json",report.str()+"],\"testing_accessed\":false,\"stress_accessed\":false,\"selection\":false,\"promotion\":false}\n");
  write_new(output/"complete.json","{\"protocol\":"+quote(protocol)+",\"status\":\"complete\",\"cohorts\":5,\"tasks_each\":2,\"encoder_trajectories\":10,\"encoder_updates_each\":512,\"attempt_limit\":1024,\"retained_points\":20,\"skipped_attempts\":0,\"sampled_rows\":40960,"
      "\"head_pipelines\":"+std::to_string(actual_pipelines)+",\"individual_heads\":"+std::to_string(2*actual_pipelines)+",\"planned_pipelines\":210,\"planned_heads\":420,"
      "\"unique_quality_native_exports\":120,\"initial_counterpart_exports\":0,\"full_native_export_calls\":120,\"query_evaluation_calls\":20,\"necessary_query_forwards\":80,"
      "\"driver_raw_outer_fits\":10,\"helper_outer_train_fits\":"+std::to_string(actual_helper_outer_fits)+",\"initial_shared_state_exact_before_training_and_heads\":true,\"separate_initial_controls\":2,"
      "\"extra_decoder_calibration_updates\":0,\"decoder_update_scope\":\"joint reconstruction during encoder updates; no extra decoder-only calibration\",\"selection\":false,"
      "\"CPU_encoder_training\":false,\"CPU_encoder_forward\":false,\"testing_accessed\":false,\"stress_accessed\":false,\"promotion\":false}\n");return 0;
} catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;} }
