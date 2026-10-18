// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/decoder_calibration.h"
#include "embedding/shared/paired_pooling.h"
#include "frozen_role_guard.h"
#include <torch/cuda.h>
#include <iostream>
#include <cstring>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
namespace {
namespace fs=std::filesystem;
namespace ev=embedding::evaluation;
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
using namespace embedding::evaluation::frozen_inputs;
const std::vector<uint64_t> masters{4404,5505,6606,7707,8808};
const std::string protocol="v7-decoder-calibration-v1";
const std::string card="code/evaluation/cards/v7_decoder_calibration_v1.md";
const std::string card_sha="5ae05a3b5ec253d1743842f82fc28b45e0f20c8c84da1b58bf4ab9c514d292e9";
const std::map<std::string,fs::path> parents{
 {"v4","output/runs/rpb-context-replication/context-replication-JjNEUc"},
 {"v7","output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2"}};
std::string cohort(uint64_t seed){return "seed-"+std::to_string(seed)+"-lag_sign";}
std::string prefix(const std::string &id){return id=="v4"?"reference":"results/candidate-development";}
std::map<std::string,fs::path> roles() {
 std::map<std::string,fs::path> result;
 for(const std::string id:{"v4","v7"}) {
   auto add=[&](const std::string &path){require(result.emplace(id+'/'+path,parents.at(id)/path).second,"duplicate role template");};
   add(prefix(id)+"/native-development-card.json"); add(prefix(id)+"/validation-report.json");
   if(id=="v7")add(prefix(id)+"/development-complete.json");
   for(auto master:masters) {
     const auto base=prefix(id)+'/'+cohort(master),point=base+"/milestone-512";
     if(id=="v7") {
       for(const std::string file:{"controlled-training.pt","controlled-validation.pt","development-manifest.json","trainer-audit.json"})add(base+'/'+file);
       for(const std::string file:{"checkpoint.pt","checkpoint.pt.audit.pt","checkpoint.pt.scaler.pt","checkpoint.pt.training-raw.pt","native-training.pt","native-validation.pt"})add(point+'/'+file);
     }
     for(const std::string file:{"point.json","training-reconstruction.pt","validation-reconstruction.pt"})add(point+'/'+file);
   }
 }
 require(result.size()==85,"closed85 roles");return result;
}
std::string role_json() {
 std::ostringstream out;out << "{\"protocol\":" << quote(protocol) << ",\"payload_access\":false,\"testing_accessed\":false,\"stress_accessed\":false,\"inputs\":[";
 bool first=true;for(const auto &[key,path]:roles()) {if(!first)out << ',';first=false;out << "{\"manifest_path\":" << quote(key) << ",\"relative_path\":" << quote(path.generic_string()) << '}';}
 return out.str()+"]}\n";
}
torch::Tensor tensor(torch::serialize::InputArchive &in,const std::string &key){torch::Tensor t;in.read(key,t,true);return t;}
void exact(const torch::Tensor &a,const torch::Tensor &b,const std::string &why) {
 require(a.defined() && b.defined() && a.device().is_cpu() && b.device().is_cpu() && a.scalar_type()==b.scalar_type() && a.sizes()==b.sizes(),why+" schema");
 const auto x=a.contiguous(),y=b.contiguous();require(!x.numel() || std::memcmp(x.const_data_ptr(),y.const_data_ptr(),x.numel()*x.element_size())==0,why+" exact bytes");
}
void query_parity(const fs::path &a,const fs::path &b,bool predictions) {
 torch::serialize::InputArchive x,y;x.load_from(a.string(),torch::kCPU);y.load_from(b.string(),torch::kCPU);
 for(const std::string key:{"standardized_target","target_mask","requested_observed_target_mask","visible_mask","trial_channel_eligible","channel_target_counts","channel_valid","example_valid","source_ids_json"})exact(tensor(x,key),tensor(y,key),"fixed query "+key);
 if(predictions)for(const std::string key:{"standardized_prediction","channel_standardized_mae","channel_standardized_huber","example_standardized_mae","example_standardized_huber"})exact(tensor(x,key),tensor(y,key),"parent query "+key);
}
std::string progress_json(const rpb::DecoderCalibrationProgress &p) {
 std::ostringstream s;s << std::setprecision(17) << "{\"original_encoder_attempted\":" << p.original_encoder_attempted << ",\"original_encoder_completed\":" << p.original_encoder_completed
 << ",\"decoder_attempted\":" << p.attempted << ",\"decoder_completed\":" << p.completed << ",\"sampled_rows\":" << p.sampled_rows << ",\"next_absolute_attempt\":" << p.next_absolute_attempt
 << ",\"decoder_parameters\":" << p.decoder_parameter_count << ",\"frozen_parameters\":" << p.frozen_parameter_count << ",\"decoder_training_loop_seconds\":" << p.training_seconds
 << ",\"encoder_parameters_exact\":" << (p.encoder_parameters_exact?"true":"false") << ",\"encoder_buffers_exact\":" << (p.encoder_buffers_exact?"true":"false")
 << ",\"scaler_exact\":" << (p.scaler_exact?"true":"false") << ",\"native_exports_exact\":" << (p.native_exports_exact?"true":"false") << ",\"zero_encoder_gradients\":" << (p.zero_encoder_gradients?"true":"false")
 << ",\"last_input_cuda\":" << (p.last_input_cuda?"true":"false") << ",\"last_latent_cuda\":" << (p.last_latent_cuda?"true":"false") << ",\"last_loss_cuda\":" << (p.last_loss_cuda?"true":"false")
 << ",\"finite_decoder_gradients\":" << (p.finite_decoder_gradients?"true":"false") << ",\"decoder_weights_changed\":" << (p.decoder_weights_changed?"true":"false") << '}';return s.str();
}
} // namespace
int main(int argc,char **argv) {
 try {
  if(argc==2 && std::string(argv[1])=="--source-id"){std::cout << EVALUATION_SOURCE_ID << '\n';return 0;}
  if(argc==2 && std::string(argv[1])=="--parent-roles"){std::cout << role_json();return 0;}
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2){require(i+1<argc,"missing option value");require(args.emplace(argv[i],argv[i+1]).second,"duplicate option");}
  require(args.size()==4 && args.count("--output") && args.count("--parent-input-manifest") && args.count("--admission-log") && args.count("--admission-sha256"),"four fixed options required");
  require(sha256(bytes(card))==card_sha && sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","card and digest binding");
  const auto log=bytes(args.at("--admission-log"));require(sha256(log)==args.at("--admission-sha256") && log.find("V7 decoder calibration CUDA admission passed")!=std::string::npos && log.find(EVALUATION_SOURCE_ID)!=std::string::npos,"actual source-bound decoder CUDA admission required");
  require(torch::cuda::is_available() && is_sha(EVALUATION_SOURCE_ID),"CUDA/source required");
  const fs::path output=args.at("--output");require(!fs::exists(output) && fs::create_directory(output),"exclusive output required");
  Guard guard(args.at("--parent-input-manifest"),roles());write_new(output/"input-manifest.json",guard.json()+"\n");
  write_new(output/"card.json","{\"protocol\":"+quote(protocol)+",\"human_card_sha256\":"+quote(card_sha)+",\"source_fingerprint\":"+quote(EVALUATION_SOURCE_ID)+",\"encoder_updates\":512,\"decoder_updates\":128,\"head_fits\":0,\"CPU_encoder_training\":false,\"testing_accessed\":false,\"stress_accessed\":false,\"selection\":false,\"promotion\":false}\n");
  ev::NativeCurveRun recipe;recipe.card.shape={3,32,3,torch::kFloat64,torch::kCPU};recipe.card.channel_ids={0,1,2};recipe.card.feature_units="unitless,unitless,unitless";recipe.card.sampling_interval=1;
  recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=0;recipe.development_only=true;recipe.stress_sweep=false;
  std::ostringstream report;report << std::setprecision(17) << "{\"protocol\":" << quote(protocol) << ",\"classifiers\":\"cached results reused, encoder invariant; no fitting or new accuracy measurements\",\"points\":[";
  bool first=true;
  for(auto master:masters) {
   const auto base="v7/"+prefix("v7")+'/'+cohort(master),point=base+"/milestone-512";
   const auto train=ev::load_native_development_observations(guard.bind(base+"/controlled-training.pt")),val=ev::load_native_development_observations(guard.bind(base+"/controlled-validation.pt"));
   require(train.observed.data.size(0)==256 && val.observed.data.size(0)==128,"fixed cohorts required");
   auto legal=train.observed;legal.data=torch::where(legal.feature_mask,legal.data,torch::zeros_like(legal.data)).detach().clone();legal.feature_mask=legal.feature_mask.clone();
   ev::ProviderFitInput fit{legal,recipe.card.shape,master,train.source_ids,recipe.card.channel_ids,recipe.card.feature_units,"native-development-v1/lag_sign",1.0,31.0};
   rpb::DecoderCalibrationOptions options;options.parent_checkpoint_path=guard.bind(point+"/checkpoint.pt");
   auto run=rpb::make_decoder_calibration(options,fit);
   const auto root=output/cohort(master);require(fs::create_directory(root),"cohort must be new");write_new(root/"trainer-audit.json",fields(run.audit_fields)+"\n");
   std::map<std::string,torch::Tensor> initial_features,initial_valid;
   for(const int stage:{0,128}) {
    const auto progress=run.train_to(stage);require(progress.original_encoder_completed==512 && progress.completed==stage && progress.attempted==stage && progress.next_absolute_attempt==512+stage && progress.decoder_parameter_count==11528 && progress.frozen_parameter_count==214277,"separate frozen encoder/decoder counters");
    require(progress.encoder_parameters_exact && progress.encoder_buffers_exact && progress.scaler_exact && progress.native_exports_exact && progress.zero_encoder_gradients,"frozen encoder invariant");
    if(stage)require(progress.last_input_cuda && progress.last_latent_cuda && progress.last_loss_cuda && progress.finite_decoder_gradients && progress.decoder_weights_changed,"actual decoder-only CUDA update");
    const auto directory=root/("stage-"+std::to_string(stage));require(fs::create_directory(directory),"stage must be new");const auto artifact=directory/"decoder-calibration.pt";run.save(artifact.string());const auto snapshot=run.snapshot(artifact.string());
    for(const auto &[name,split]:std::vector<std::pair<std::string,ev::ControlledDataset>>{{"training",train},{"validation",val}}) {
      const auto surface=ev::extract_native_global(snapshot,split.observed,recipe);
      if(stage==0){initial_features[name]=surface.values.clone();initial_valid[name]=surface.valid.clone();}
      else {exact(surface.values,initial_features.at(name),"same CUDA native32 "+name);exact(surface.valid,initial_valid.at(name),"same native support "+name);}
      ev::save_native_feature_archive((directory/("native-"+name+"-witness.pt")).string(),surface,split,recipe);
    }
    const auto training_query=directory/"training-reconstruction.pt",validation_query=directory/"validation-reconstruction.pt";
    const auto training_summary=ev::write_native_patch_reconstruction(training_query.string(),train,snapshot,recipe),validation_summary=ev::write_native_patch_reconstruction(validation_query.string(),val,snapshot,recipe);
    query_parity(training_query,guard.bind(point+"/training-reconstruction.pt"),stage==0);query_parity(validation_query,guard.bind(point+"/validation-reconstruction.pt"),stage==0);
    const auto v4point="v4/reference/"+cohort(master)+"/milestone-512";query_parity(training_query,guard.bind(v4point+"/training-reconstruction.pt"),false);query_parity(validation_query,guard.bind(v4point+"/validation-reconstruction.pt"),false);
    const auto record="{\"master_seed\":"+std::to_string(master)+",\"model_tag\":\"RPB-v7\",\"decoder_stage\":"+std::to_string(stage)+",\"progress\":"+progress_json(progress)+",\"training_reconstruction\":"+training_summary+",\"validation_reconstruction\":"+validation_summary+"}";
    write_new(directory/"point.json",record+"\n");if(!first)report << ',';first=false;report << record;
    std::cout << "V7 decoder calibration master=" << master << " decoder_updates=" << stage << " encoder_updates=512 decoder_loop_seconds=" << progress.training_seconds << '\n' << std::flush;
   }
  }
  write_new(output/"report.json",report.str()+"]}\n");guard.verify();write_new(output/"input-integrity-after.json",guard.json()+"\n");write_new(output/"complete.json","{\"protocol\":"+quote(protocol)+",\"status\":\"complete\",\"masters\":5,\"points\":10,\"new_encoder_updates\":0,\"decoder_updates_each\":128,\"head_fits\":0,\"testing_accessed\":false,\"stress_accessed\":false}\n");return 0;
 }catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
