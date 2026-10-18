// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/shared/paired_pooling.h"
#include "embedding/shared/fixed_feature_readouts.h"
#include "frozen_role_guard.h"
#include <torch/cuda.h>
#include <cstring>
#include <iostream>
#include <chrono>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
namespace {
namespace fs=std::filesystem;
namespace ev=embedding::evaluation;
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
using namespace ev::frozen_inputs;
const std::vector<uint64_t> masters{9109,10210,11311,12412,13513};
const std::string protocol="fresh-decoder-replication-v1";
const std::string card="code/evaluation/cards/fresh_decoder_replication_v1.md";
const std::string card_sha="2236fb794d608c8f7b8bde81814f199abcc3e27c641cfba5902cdc3f4c4a25cf";
std::string plan(){return "{\"protocol\":"+quote(protocol)+",\"master_seeds\":[9109,10210,11311,12412,13513],\"tags\":[\"RPB-v4\",\"RPB-v7\"],\"encoder_updates\":512,\"decoder_updates\":128,\"train_pairs\":128,\"validation_pairs\":64,\"test_pairs\":0,\"shape\":[3,32,3],\"batch_size\":8,\"native_width\":32,\"head_repetitions\":[2701,2802,2903],\"head_pipelines\":90,\"encoder_device\":\"CUDA\",\"prior_quality_input_roles\":[],\"quality_generated\":false,\"testing_accessed\":false,\"stress_accessed\":false}\n";}
void archive(const fs::path &path,torch::serialize::OutputArchive &a){require(!fs::exists(path),"new archive required");a.save_to(path.string());}
torch::Tensor text_tensor(const std::string &s){return torch::tensor(std::vector<uint8_t>(s.begin(),s.end()),torch::kUInt8);}
void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &why){
 const auto x=a.detach().to(torch::kCPU).contiguous(),y=b.detach().to(torch::kCPU).contiguous();
 require(x.scalar_type()==y.scalar_type() && x.sizes()==y.sizes(),why+" schema");
 require(!x.numel() || std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size())==0,why+" bytes");
}
void same_surface(const ev::FeatureSurface &a,const ev::FeatureSurface &b,const std::string &why){exact(a.values,b.values,why+" values");exact(a.valid,b.valid,why+" support");}
void sanitize(ev::ControlledDataset &d){
 d.observed.data=torch::where(d.observed.feature_mask,d.observed.data,torch::zeros_like(d.observed.data)).detach().clone();
 d.observed.feature_mask=d.observed.feature_mask.clone();d.clean=d.observed;
 for(auto &id:d.source_ids)id=protocol+'/'+id;
}
void save_observations(const fs::path &path,const ev::ControlledDataset &d,const torch::Tensor &erasure={}){
 torch::serialize::OutputArchive a;a.write("observations",d.observed.data,true);a.write("feature_mask",d.observed.feature_mask,true);
 a.write("labels_scoring_only",d.labels,true);a.write("source_ids_json",text_tensor(strings(d.source_ids)),true);
 if(erasure.defined())a.write("requested_erasure",erasure,true);archive(path,a);
}
void initial_pair(const fs::path &a,const fs::path &b,const fs::path &witness){
 auto x=rpb::load_checkpoint(a.string(),torch::kCUDA),y=rpb::load_checkpoint(b.string(),torch::kCUDA);
 require(x.attempted_steps==0 && y.attempted_steps==0 && x.completed_steps==0 && y.completed_steps==0,"exact point0 counters");
 const auto xp=x.model->named_parameters(),yp=y.model->named_parameters();require(xp.size()==yp.size(),"initial parameter count");
 for(const auto &p:xp){require(yp.contains(p.key()),"initial parameter name");exact(p.value(),yp[p.key()],"initial parameter "+p.key());}
 const auto xb=x.model->named_buffers(),yb=y.model->named_buffers();require(xb.size()==yb.size(),"initial buffer count");
 for(const auto &p:xb){require(yb.contains(p.key()),"initial buffer name");exact(p.value(),yb[p.key()],"initial buffer "+p.key());}
 exact(x.scaler.mean,y.scaler.mean,"initial scaler mean");exact(x.scaler.scale,y.scaler.scale,"initial scaler scale");
 exact(x.scaler.count,y.scaler.count,"initial scaler count");exact(x.scaler.channel_ids,y.scaler.channel_ids,"initial scaler channel IDs");exact(x.scaler.floor_applied,y.scaler.floor_applied,"initial scaler floors");
 require(rpb::settings_text(x.settings)==rpb::settings_text(y.settings),"paired complete initial settings");
 require(x.dataset_id==y.dataset_id && x.schema_id==y.schema_id && x.scaler_fit_dataset_id==y.scaler_fit_dataset_id,"paired label-free data/scaler identity");
 torch::serialize::OutputArchive saved;
 for(const auto &[name,cp]:std::vector<std::pair<std::string,rpb::Checkpoint*>>{{"v4",&x},{"v7",&y}}){
  torch::serialize::OutputArchive parameters,buffers,scaler;int64_t i=0;
  for(const auto &p:cp->model->named_parameters()){torch::serialize::OutputArchive item;item.write("parameter_name",text_tensor(p.key()),true);item.write("value",p.value().detach().to(torch::kCPU).clone(),true);parameters.write("tensor_"+std::to_string(i++),item);}parameters.write("count",torch::tensor(i),true);i=0;
  for(const auto &p:cp->model->named_buffers()){torch::serialize::OutputArchive item;item.write("parameter_name",text_tensor(p.key()),true);item.write("value",p.value().detach().to(torch::kCPU).clone(),true);buffers.write("tensor_"+std::to_string(i++),item);}buffers.write("count",torch::tensor(i),true);cp->scaler.save(scaler);
  saved.write(name+"_parameters",parameters);saved.write(name+"_buffers",buffers);saved.write(name+"_scaler",scaler);
 }archive(witness,saved);
}
ev::FeatureSurface raw(const ev::ObservationScaler &s,const embedding::Batch &b){const auto z=s.transform(b);return {torch::cat({z.data.flatten(1),z.feature_mask.flatten(1).to(torch::kFloat64)},1),b.feature_mask.flatten(1).any(1),"TRAIN ObservationScaler; legal observed values+visibility"};}
ev::FeatureSurface mask(const embedding::Batch &b){return {b.feature_mask.flatten(1).to(torch::kFloat64),b.feature_mask.flatten(1).any(1),"visibility metadata only"};}
std::string encoder_progress(const ev::CurveProgress &p){
 std::ostringstream s;s << std::setprecision(17) << "{\"attempted\":" << p.attempted << ",\"completed\":" << p.completed << ",\"sampled_rows\":" << p.sampled_rows << ",\"parameter_count\":" << p.parameter_count << ",\"cuda_parameter_count\":" << p.cuda_parameter_count << ",\"training_seconds\":" << p.training_seconds << ",\"training_device\":" << quote(p.training_device) << ",\"training_dataset_id\":" << quote(p.training_dataset_id) << ",\"preprocessing_id\":" << quote(p.preprocessing_id) << ",\"last_input_cuda\":" << (p.last_input_cuda?"true":"false") << ",\"last_loss_cuda\":" << (p.last_loss_cuda?"true":"false") << ",\"finite_gradients\":" << (p.finite_gradients?"true":"false") << ",\"weights_changed\":" << (p.weights_changed?"true":"false") << ",\"losses\":[";
 bool first=true;for(const auto &l:p.losses){if(!first)s << ',';first=false;s << "[" << l.attempted << ',' << l.completed << ',' << l.target_cells << ',' << l.loss << ',' << l.gradient_norm << ']';}return s.str()+"]}";
}
std::string decoder_progress(const rpb::DecoderCalibrationProgress &p){
 std::ostringstream s;s << std::setprecision(17) << "{\"original_encoder_attempted\":" << p.original_encoder_attempted << ",\"original_encoder_completed\":" << p.original_encoder_completed << ",\"decoder_attempted\":" << p.attempted << ",\"decoder_completed\":" << p.completed << ",\"sampled_rows\":" << p.sampled_rows << ",\"next_absolute_attempt\":" << p.next_absolute_attempt << ",\"decoder_parameters\":" << p.decoder_parameter_count << ",\"frozen_parameters\":" << p.frozen_parameter_count << ",\"decoder_training_loop_seconds\":" << p.training_seconds << ",\"encoder_parameters_exact\":" << (p.encoder_parameters_exact?"true":"false") << ",\"encoder_buffers_exact\":" << (p.encoder_buffers_exact?"true":"false") << ",\"scaler_exact\":" << (p.scaler_exact?"true":"false") << ",\"native_exports_exact\":" << (p.native_exports_exact?"true":"false") << ",\"zero_encoder_gradients\":" << (p.zero_encoder_gradients?"true":"false") << ",\"last_input_cuda\":" << (p.last_input_cuda?"true":"false") << ",\"last_latent_cuda\":" << (p.last_latent_cuda?"true":"false") << ",\"last_loss_cuda\":" << (p.last_loss_cuda?"true":"false") << ",\"finite_decoder_gradients\":" << (p.finite_decoder_gradients?"true":"false") << ",\"decoder_weights_changed\":" << (p.decoder_weights_changed?"true":"false") << '}';return s.str();
}
torch::Tensor read(torch::serialize::InputArchive &a,const std::string &key){torch::Tensor t;a.read(key,t,true);return t;}
void query_support(const fs::path &a,const fs::path &b){
 torch::serialize::InputArchive x,y;x.load_from(a.string(),torch::kCPU);y.load_from(b.string(),torch::kCPU);
 for(const std::string key:{"standardized_target","target_mask","requested_observed_target_mask","visible_mask","trial_channel_eligible","channel_target_counts","channel_valid","example_valid","source_ids_json"})exact(read(x,key),read(y,key),"original query "+key);
}
} // namespace
int main(int argc,char **argv){try{
 if(argc==2 && std::string(argv[1])=="--source-id"){std::cout << EVALUATION_SOURCE_ID << '\n';return 0;}
 if(argc==2 && std::string(argv[1])=="--plan"){std::cout << plan();return 0;}
 std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2){require(i+1<argc,"missing option value");require(args.emplace(argv[i],argv[i+1]).second,"duplicate option");}
 require(args.size()==3 && args.count("--output") && args.count("--admission-log") && args.count("--admission-sha256"),"three fixed measurement options required");
 require(sha256(bytes(card))==card_sha && is_sha(EVALUATION_SOURCE_ID),"frozen card/source binding");
 const auto log=bytes(args.at("--admission-log"));require(sha256(log)==args.at("--admission-sha256") && log.find("Fresh decoder replication CUDA admission passed")!=std::string::npos && log.find(EVALUATION_SOURCE_ID)!=std::string::npos,"actual source-bound CUDA admission");
 require(torch::cuda::is_available(),"CUDA required; no CPU encoder fallback");torch::set_num_threads(1);
 const fs::path output=args.at("--output");require(!fs::exists(output) && fs::create_directory(output),"exclusive new result directory");write_new(output/"recipe-plan.json",plan());
 write_new(output/"card.json","{\"protocol\":"+quote(protocol)+",\"human_card_sha256\":"+quote(card_sha)+",\"source_fingerprint\":"+quote(EVALUATION_SOURCE_ID)+",\"historical_payload_roles\":0,\"testing_accessed\":false,\"stress_accessed\":false}\n");
 ev::NativeCurveRun recipe;recipe.card.shape={3,32,3,torch::kFloat64,torch::kCPU};recipe.card.channel_ids={0,1,2};recipe.card.feature_units="unitless,unitless,unitless";recipe.card.sampling_interval=1;recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=0;recipe.development_only=true;recipe.stress_sweep=false;
 rpb::Settings settings=rpb::default_settings();settings.steps=512;settings.attempt_limit=1000;settings.log_every=1;settings.batch_size=8;settings.threads=1;settings.learning_rate=.001;settings.weight_decay=.0001;settings.gradient_clip_norm=1;settings.model.channel_mixer_layers=1;settings.model.global_bottleneck_mode=2;settings.model.export_width=32;settings.model.device=torch::kCUDA;
 const auto ordinary=rpb::make_learning_curve_trainer(settings),lighter=rpb::make_learning_curve_trainer(settings,{true,rpb::ContextDeletionRecipe::coordinate15_v1});
 std::ostringstream report;report << std::setprecision(17) << "{\"protocol\":" << quote(protocol) << ",\"source_fingerprint\":" << quote(EVALUATION_SOURCE_ID) << ",\"cohorts\":[";bool first_cohort=true;
 for(auto master:masters){
  const auto root=output/("seed-"+std::to_string(master)+"-lag_sign");require(fs::create_directory(root),"new cohort");
  auto development=ev::make_controlled_development_protocol(ev::Task::lag_sign,recipe.card.shape,128,64,master,.1);sanitize(development.training);sanitize(development.validation);
  require(!development.testing.observed.data.defined() && development.testing.source_ids.empty(),"TEST not generated");
  auto &train=development.training;auto &val=development.validation;
  auto view=ev::make_coordinate_deletion_view(val.observed,val.source_ids,ev::Task::lag_sign,ev::stream_seed(master,0x6672642d76616c30ULL),.30,protocol+"/validation-coordinate-dropout");
  auto dropped=val;dropped.observed=view.observations;dropped.clean=dropped.observed;
  save_observations(root/"controlled-training.pt",train);save_observations(root/"controlled-validation.pt",val);save_observations(root/"controlled-validation-deleted.pt",dropped,view.requested_erasure);
  ev::ProviderFitInput fit{train.observed,recipe.card.shape,master,train.source_ids,recipe.card.channel_ids,recipe.card.feature_units,protocol+"/lag_sign",1,31};
  auto v4=ordinary(fit),v7=lighter(fit);
  for(const auto &[name,trainer]:std::vector<std::pair<std::string,ev::CurveTrainer*>>{{"v4",&v4},{"v7",&v7}}){require(fs::create_directory(root/name),"new model directory");require(fs::create_directory(root/name/"point-0"),"new point0");trainer->save_checkpoint((root/name/"point-0/checkpoint.pt").string());}
  initial_pair(root/"v4/point-0/checkpoint.pt",root/"v7/point-0/checkpoint.pt",root/"initial-pair.pt");
  const auto initial_start=std::chrono::steady_clock::now();
  const auto initial=rpb::make_fresh_initial_snapshot((root/"v4/point-0/checkpoint.pt").string(),rpb::FrozenDecoderParentPolicy::ordinary_v4,fit);
  const auto paired_initial=rpb::make_fresh_initial_snapshot((root/"v7/point-0/checkpoint.pt").string(),rpb::FrozenDecoderParentPolicy::coordinate15_v7,fit);
  const auto initial_train=ev::extract_native_global(initial,train.observed,recipe);same_surface(initial_train,ev::extract_native_global(paired_initial,train.observed,recipe),"same CUDA paired point0");
  std::vector<ev::FixedFeatureMethod> methods;
  methods.push_back({"untrained_native",initial_train,ev::extract_native_global(initial,val.observed,recipe),ev::extract_native_global(initial,dropped.observed,recipe),false});
  const double initial_inference_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-initial_start).count();
  ev::ObservationScaler scaler(train.observed);torch::serialize::OutputArchive sc;sc.write("mean",scaler.mean,true);sc.write("scale",scaler.scale,true);sc.write("counts",scaler.counts,true);archive(root/"raw-scaler.pt",sc);
  auto raw_train=raw(scaler,train.observed),raw_val=raw(scaler,val.observed),raw_drop=raw(scaler,dropped.observed);
  methods.push_back({"raw",raw_train,raw_val,raw_drop,false});ev::FeatureNormalizer outer(raw_train);ev::TrainPca pca(outer.transform(raw_train),32);
  torch::serialize::OutputArchive pc;pc.write("feature_mean",outer.mean,true);pc.write("feature_scale",outer.scale,true);pc.write("fitted_rows",torch::tensor(outer.fitted_rows),true);pc.write("pca_mean",pca.mean,true);pc.write("pca_components",pca.components,true);pc.write("pca_singular_values",pca.singular_values,true);pc.write("pca_numerical_rank",torch::tensor(pca.numerical_rank),true);archive(root/"pca-preprocessing.pt",pc);
  methods.push_back({"pca_only",pca.transform(outer.transform(raw_train)),pca.transform(outer.transform(raw_val)),pca.transform(outer.transform(raw_drop)),true});methods.push_back({"mask_metadata",mask(train.observed),mask(val.observed),mask(dropped.observed),false});
  std::ostringstream points;points << '[';bool first_point=true;std::vector<ev::CurveProgress> parent_progress;
  for(const auto &[name,trainer,policy]:std::vector<std::tuple<std::string,ev::CurveTrainer*,rpb::FrozenDecoderParentPolicy>>{{"v4",&v4,rpb::FrozenDecoderParentPolicy::ordinary_v4},{"v7",&v7,rpb::FrozenDecoderParentPolicy::coordinate15_v7}}){
   const auto progress=trainer->train_to(512);require(progress.attempted==512 && progress.completed==512 && progress.sampled_rows==4096 && progress.parameter_count==225805 && progress.cuda_parameter_count==225805 && progress.last_input_cuda && progress.last_loss_cuda && progress.finite_gradients && progress.weights_changed,"actual complete encoder CUDA trajectory");parent_progress.push_back(progress);
   const auto model_dir=root/name;require(fs::create_directory(model_dir/"point-512"),"new point512");const auto parent=model_dir/"point-512/checkpoint.pt";trainer->save_checkpoint(parent.string());write_new(model_dir/"encoder-progress.json",encoder_progress(progress)+"\n");write_new(model_dir/"trainer-audit.json",fields(trainer->audit_fields)+"\n");
   rpb::FrozenDecoderCalibrationOptions options;options.parent_checkpoint_path=parent.string();options.parent_policy=policy;auto decoder=rpb::make_frozen_decoder_calibration(options,fit);write_new(model_dir/"decoder-audit.json",fields(decoder.audit_fields)+"\n");
   std::map<std::string,ev::FeatureSurface> frozen;
   for(int stage:{0,128}){
    const auto p=decoder.train_to(stage);require(p.original_encoder_attempted==512 && p.original_encoder_completed==512 && p.attempted==stage && p.completed==stage && p.next_absolute_attempt==512+stage && p.frozen_parameter_count==214277 && p.decoder_parameter_count==11528,"equal decoder budget/counters");require(p.encoder_parameters_exact && p.encoder_buffers_exact && p.scaler_exact && p.native_exports_exact && p.zero_encoder_gradients,"frozen encoder/native invariant");if(stage)require(p.last_input_cuda && p.last_latent_cuda && p.last_loss_cuda && p.finite_decoder_gradients && p.decoder_weights_changed,"actual CUDA decoder updates");
    const auto dir=model_dir/("decoder-"+std::to_string(stage));require(fs::create_directory(dir),"new decoder stage");const auto asset=dir/"decoder-calibration.pt";decoder.save(asset.string());const auto inference_start=std::chrono::steady_clock::now();const auto snapshot=decoder.snapshot(asset.string());
    for(const auto &[split_name,split]:std::vector<std::pair<std::string,ev::ControlledDataset>>{{"training",train},{"validation",val},{"validation-deleted",dropped}}){const auto f=ev::extract_native_global(snapshot,split.observed,recipe);require(f.valid.all().item<bool>(),"full native coverage");if(!stage)frozen[split_name]=f;else same_surface(f,frozen.at(split_name),"same CUDA frozen "+name+'/'+split_name);ev::save_native_feature_archive((dir/("native-"+split_name+".pt")).string(),f,split,recipe);}
    const auto tq=dir/"training-reconstruction.pt",vq=dir/"validation-reconstruction.pt";const auto ts=ev::write_native_patch_reconstruction(tq.string(),train,snapshot,recipe),vs=ev::write_native_patch_reconstruction(vq.string(),val,snapshot,recipe);
    if(stage){query_support(tq,model_dir/"decoder-0/training-reconstruction.pt");query_support(vq,model_dir/"decoder-0/validation-reconstruction.pt");}if(name=="v7"){query_support(tq,root/"v4/decoder-0/training-reconstruction.pt");query_support(vq,root/"v4/decoder-0/validation-reconstruction.pt");}
    const double inference_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-inference_start).count();
    const auto record="{\"model_tag\":"+quote(name=="v4"?"RPB-v4":"RPB-v7")+",\"master_seed\":"+std::to_string(master)+",\"decoder_stage\":"+std::to_string(stage)+",\"encoder_progress\":"+encoder_progress(progress)+",\"decoder_progress\":"+decoder_progress(p)+",\"feature_query_inference_and_evidence_seconds\":"+std::to_string(inference_seconds)+",\"training_reconstruction\":"+ts+",\"validation_reconstruction\":"+vs+"}";
    write_new(dir/"point.json",record+"\n");if(!first_point)points << ',';first_point=false;points << record;std::cout << "Fresh replication master=" << master << " model=" << name << " encoder_updates=512 decoder_updates=" << stage << " encoder_seconds=" << progress.training_seconds << " decoder_seconds=" << p.training_seconds << '\n' << std::flush;
   }
   methods.push_back({"native_"+name,frozen.at("training"),frozen.at("validation"),frozen.at("validation-deleted"),false});
  }
  require(parent_progress[0].losses.size()==512 && parent_progress[1].losses.size()==512,"full parent trace");for(size_t i=0;i<512;++i)require(parent_progress[0].losses[i].attempted==parent_progress[1].losses[i].attempted && parent_progress[0].losses[i].target_cells==parent_progress[1].losses[i].target_cells,"paired original counters/targets");
  ev::FixedFeatureReadoutRun heads;heads.output_directory=(root/"readouts").string();heads.master_seed=master;heads.training_labels=train.labels;heads.validation_labels=val.labels;heads.training_source_ids=train.source_ids;heads.validation_source_ids=val.source_ids;heads.methods=methods;
  const auto head_start=std::chrono::steady_clock::now();const auto scores=ev::run_fixed_feature_readouts(heads);const double head_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-head_start).count();
  write_new(root/"classification-reuse.json","{\"encoder_updates\":512,\"post_decoder_updates\":128,\"native_same_CUDA_exact\":true,\"readout_pipelines\":18,\"post_decoder_head_refits\":0,\"post_decoder_score_repetitions\":0,\"binding\":\"decoder0 saved native features and TRAIN-only fitted assets; exact decoder128 feature witnesses\"}\n");
  write_new(root/"initial-pair.json","{\"full_named_parameters_exact\":true,\"buffers_exact\":true,\"scaler_exact\":true,\"same_CUDA_native_exact\":true,\"original_counter_target_counts_paired\":true,\"initial_control_sets\":1}\n");
  if(!first_cohort)report << ',';first_cohort=false;report << "{\"master_seed\":" << master << ",\"points\":" << points.str() << "],\"readouts\":" << scores << ",\"initial_CUDA_inference_and_evidence_seconds\":" << initial_inference_seconds << ",\"CPU_readout_loop_seconds\":" << head_seconds << '}';
 }
 write_new(output/"report.json",report.str()+"],\"testing_accessed\":false,\"stress_accessed\":false,\"selection\":false,\"promotion\":false}\n");write_new(output/"complete.json","{\"protocol\":"+quote(protocol)+",\"status\":\"complete\",\"masters\":5,\"encoder_trajectories\":10,\"decoder_trajectories\":10,\"head_pipelines\":90,\"individual_heads\":180,\"CPU_encoder_training\":false,\"CPU_encoder_forward\":false,\"testing_accessed\":false,\"stress_accessed\":false}\n");return 0;
}catch(const std::exception &e){std::cerr << e.what() << '\n';return 1;}}
