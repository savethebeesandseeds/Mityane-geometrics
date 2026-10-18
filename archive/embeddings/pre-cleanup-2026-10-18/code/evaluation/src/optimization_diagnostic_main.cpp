// SPDX-License-Identifier: MIT
#include "embedding/shared/paired_pooling.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/optimization_diagnostic_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
#ifndef EVALUATION_GIT_HEAD
#define EVALUATION_GIT_HEAD "unrecorded"
#endif
#ifndef EVALUATION_GIT_DIRTY
#define EVALUATION_GIT_DIRTY "unrecorded"
#endif
namespace {
namespace fs=std::filesystem;
namespace ev=embedding::evaluation;
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
const std::string retained="output/runs/rpb-paired-pooling/paired-pooling-BFFPX5";
const std::map<uint64_t,std::string> parent_hashes{
  {3101,"fc9da5eceffc252cb69f5e9b4a92e6464711cfcc8c1e7c225bb42936f24a1207"},
  {3202,"7a7d3949f9757ba7594c00cb6231e700d7ce19e822194b660126eeff582c3d1e"},
  {3303,"02c45cc957fa6beae6730df66bee0bdb5c22297ce18d0f620c4890fcb3044de0"}};
void require(bool value,const std::string &message) {
  if(!value) throw std::runtime_error("[optimization diagnostic] "+message);
}
bool sha(const std::string &s) { return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos; }
std::string quote(const std::string &value) {
  std::ostringstream out;out << '"';
  for(unsigned char c:value) {
    if(c=='"' || c=='\\')out << '\\' << c;
    else if(c=='\n')out << "\\n";
    else { require(c>=32,"unsupported control character");out << c; }
  }
  out << '"';return out.str();
}
void write_new(const fs::path &path,const std::string &value) {
  const int file=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  require(file>=0,"output must be new: "+path.string());
  size_t written=0;
  while(written<value.size()) {
    const auto count=::write(file,value.data()+written,value.size()-written);
    if(count<=0){::close(file);throw std::runtime_error("output write failed");}
    written+=static_cast<size_t>(count);
  }
  const int synced=::fsync(file),closed=::close(file);require(synced==0 && closed==0,"output fsync failed");
  const int directory=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);require(directory>=0,"output directory unavailable");
  const int ds=::fsync(directory),dc=::close(directory);require(ds==0 && dc==0,"directory fsync failed");
}
std::map<std::string,std::string> hashes(const fs::path &path) {
  std::ifstream in(path);require(bool(in),"explicit frozen retained manifest required");
  std::map<std::string,std::string> out;std::string line;
  while(std::getline(in,line)) {
    require(line.size()>66 && line.substr(64,2)=="  " && sha(line.substr(0,64)),"bad frozen hash record");
    const fs::path relative(line.substr(66));
    require(!relative.is_absolute() && relative.lexically_normal()==relative && relative.generic_string().find("..") == std::string::npos,
        "retained path outside capsule");
    require(out.emplace(fs::absolute(fs::path(retained)/relative).string(),line.substr(0,64)).second,"duplicate retained input");
  }
  require(in.eof() && !out.empty(),"empty/unreadable frozen manifest");return out;
}
torch::Tensor tensor(torch::serialize::InputArchive &archive,const std::string &key) {
  torch::Tensor out;archive.read(key,out,true);return out;
}
void feature_parity(const ev::FeatureSurface &actual,const std::string &old) {
  torch::serialize::InputArchive archive;archive.load_from(old,torch::kCPU);
  require(torch::equal(actual.values,tensor(archive,"features")) && torch::equal(actual.valid,tensor(archive,"valid")),
      "original512 native export differs");
}
void query_parity(const std::string &actual,const std::string &old) {
  torch::serialize::InputArchive a,b;a.load_from(actual,torch::kCPU);b.load_from(old,torch::kCPU);
  for(const auto &key:a.keys()) require(torch::equal(tensor(a,key),tensor(b,key)),"original512 reconstruction witness differs: "+key);
  require(a.keys()==b.keys(),"original512 reconstruction schema differs");
}
ev::ProviderFitInput metadata(const ev::ControlledDataset &train,uint64_t seed,const ev::NativeCurveRun &recipe) {
  auto legal=train.observed;
  legal.data=torch::where(legal.feature_mask,legal.data,torch::zeros_like(legal.data)).detach().clone();
  legal.feature_mask=legal.feature_mask.clone();
  return {legal,recipe.card.shape,seed,train.source_ids,recipe.card.channel_ids,recipe.card.feature_units,
      "native-curve-v1/lag_sign",recipe.card.sampling_interval,31*recipe.card.sampling_interval};
}
} // namespace
int main(int argc,char **argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--source-id"){std::cout << EVALUATION_SOURCE_ID << '\n';return 0;}
    if(argc==1 || (argc==2 && std::string(argv[1])=="--help")) {
      std::cout << "embedding_optimization_diagnostic --output NEW_DIRECTORY --retained-hashes FROZEN_MANIFEST\n"
          "Fixed optimization-validation-v1; RPB-v5 exact512 parents continue to1024/2048; TRAIN/VALIDATION only.\n";
      return argc==1?1:0;
    }
    std::map<std::string,std::string> options;
    for(int i=1;i<argc;i+=2)require(i+1<argc && (std::string(argv[i])=="--output" || std::string(argv[i])=="--retained-hashes") &&
        options.emplace(argv[i],argv[i+1]).second,"unknown/duplicate option or missing value");
    require(options.size()==2 && !options.at("--output").empty(),"new output and frozen manifest required");
    require(sha(EVALUATION_SOURCE_ID) && torch::cuda::is_available(),"source fingerprint and actual CUDA required");
    const fs::path output=fs::absolute(options.at("--output")).lexically_normal();
    require(!fs::exists(output),"output already exists");fs::create_directories(output.parent_path());
    const auto expected=hashes(options.at("--retained-hashes"));
    const auto bind=[&](const fs::path &p) {
      const auto full=fs::absolute(p).lexically_normal().string();require(expected.count(full),"undeclared retained input: "+full);return full;
    };
    ev::NativeCurveRun recipe;recipe.source_fingerprint=EVALUATION_SOURCE_ID;recipe.model_tag="RPB-v5";
    recipe.card.seeds={3101,3202,3303};recipe.card.tasks={ev::Task::lag_sign};
    recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=0;recipe.stress_sweep=false;
    recipe.card.id="optimization-validation-v1";recipe.milestones={512,1024,2048};
    auto settings=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/learned_patch_global.conf");settings.model.device=torch::kCUDA;
    std::ostringstream plan;plan << "{\"protocol\":\"optimization-validation-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\","
      "\"model_tag\":\"RPB-v5\",\"masters\":[3101,3202,3303],\"absolute_budgets\":[512,1024,2048],"
      "\"additional_command_updates\":[512,1024],\"source_pairs\":[128,64],\"batch_size\":8,\"device\":\"cuda\","
      "\"native_size\":32,\"post_encoder_pca\":false,\"test_access\":false,\"stress_access\":false,\"selection\":false,\"promotion\":false,"
      "\"readouts\":\"fixed ridge1; tanh16 Adam0.01x100; base seeds2701,2802,2903; TRAIN-only new points; cached512 scores\","
      "\"raw_pca_control_policy\":\"same TRAIN refits at each new point through archive-readout-v1\","
      "\"timing\":\"synchronized whole ordinary resume command; GPU-update-only unmeasured\","
      "\"retained_inventory_sha256\":\"610976e666468ac2f80404bd56884e36f9275645305e83a9d0e50b762e702be3\","
      "\"source_fingerprint\":" << quote(EVALUATION_SOURCE_ID) << ",\"git_head\":" << quote(EVALUATION_GIT_HEAD)
      << ",\"git_dirty\":" << quote(EVALUATION_GIT_DIRTY) << ",\"retained_hash_manifest\":" << quote(fs::absolute(options.at("--retained-hashes")).string())
      << ",\"base_design_config\":" << quote(rpb::settings_text(settings)) << ",\"actual_parent_settings\":\"pinned per-master checkpoint producer audit; restored and verified by continuation adapter\",\"parents\":[";
    bool first=true;
    for(const auto &[master,checksum]:parent_hashes) {
      const auto cp=bind(fs::path(retained)/"results"/("seed-"+std::to_string(master)+"-lag_sign")/"candidate-milestone-512/checkpoint.pt");
      require(expected.at(cp)==checksum,"wrong exact512 parent");if(!first)plan << ',';first=false;
      plan << "{\"master_seed\":" << master << ",\"checkpoint\":" << quote(cp) << ",\"sha256\":" << quote(checksum) << '}';
    }
    plan << "]}\n";write_new(output.string()+".launch-plan.json",plan.str());
    require(fs::create_directory(output),"cannot claim exclusive output");
    torch::set_num_threads(1);rpb::run_native_curve_cuda_gate(settings,output.string()+"-cuda-gate");
    std::cout << "Actual CUDA gate passed; exact-parent validation-only continuation starts.\n" << std::flush;
    ev::ArchiveReadoutRun readouts;readouts.output_directory=(output/"readouts").string();
    readouts.source_fingerprint=EVALUATION_SOURCE_ID;readouts.git_head=EVALUATION_GIT_HEAD;readouts.git_dirty=EVALUATION_GIT_DIRTY;
    std::ostringstream report;report << "{\"protocol\":\"optimization-validation-v1\",\"model_tag\":\"RPB-v5\",\"test_access\":false,\"selection\":false,\"points\":[";first=true;
    std::set<std::string> all_sources;
    for(const auto &[master,checksum]:parent_hashes) {
      (void)checksum;const auto old=fs::path(retained)/"results"/("seed-"+std::to_string(master)+"-lag_sign");
      const auto directory=output/("seed-"+std::to_string(master)+"-lag_sign");require(fs::create_directory(directory),"cohort output exists");
      const auto train_path=bind(old/"controlled-training.pt"),val_path=bind(old/"controlled-validation.pt");
      const auto train=ev::load_native_development_observations(train_path),val=ev::load_native_development_observations(val_path);
      require(train.labels.numel()==256 && val.labels.numel()==128,"wrong frozen split size");
      for(const auto *split:{&train,&val}) {
        const std::set<std::string> sources(split->source_ids.begin(),split->source_ids.end());
        for(const auto &id:sources)require(all_sources.insert(id).second,"TRAIN/VALIDATION/master source overlap");
      }
      const auto fit=metadata(train,master,recipe);std::string parent=bind(old/"candidate-milestone-512/checkpoint.pt");
      const auto raw=bind(parent+".training-raw.pt");bind(parent+".audit.pt");bind(parent+".scaler.pt");
      for(const std::string repetition:{"rep-1","rep-2","rep-3"}) {
        bind(old/"candidate-milestone-512"/repetition/"fit.pt");
        bind(old/"candidate-milestone-512"/repetition/"validation-predictions.pt");
      }
      const auto initial=rpb::make_optimization_diagnostic_snapshot(parent,fit);
      const auto original_train=ev::extract_native_global(initial,train.observed,recipe),original_val=ev::extract_native_global(initial,val.observed,recipe);
      feature_parity(original_train,bind(old/"candidate-milestone-512/native-training.pt"));
      feature_parity(original_val,bind(old/"candidate-milestone-512/native-validation.pt"));
      ev::save_native_feature_archive((directory/"original512-native-training.pt").string(),original_train,train,recipe);
      ev::save_native_feature_archive((directory/"original512-native-validation.pt").string(),original_val,val,recipe);
      for(const auto &[name,split]:std::vector<std::pair<std::string,const ev::ControlledDataset *>>{{"training",&train},{"validation",&val}}) {
        const auto witness=directory/("original512-"+name+"-reconstruction.pt");
        write_new(directory/("original512-"+name+"-reconstruction.json"),ev::write_native_patch_reconstruction(witness.string(),*split,initial,recipe));
        query_parity(witness.string(),bind(old/("candidate-"+name+"-reconstruction.pt")));
      }
      write_new(directory/"original512-parity.json","{\"native_training_exact\":true,\"native_validation_exact\":true,\"fixed_queries_exact\":true,\"head_refits\":0}\n");
      for(const int64_t budget:{1024,2048}) {
        const auto point=directory/("milestone-"+std::to_string(budget));require(fs::create_directory(point),"point output exists");
        const auto cp=(point/"checkpoint.pt").string();const auto result=rpb::resume_optimization_diagnostic({parent,raw,cp,budget==1024?512:1024},fit);
        require(result.completed_after==budget && result.attempted_after==budget,"continuation skipped updates or wrong absolute budget");
        const auto snapshot=rpb::make_optimization_diagnostic_snapshot(cp,fit);
        const auto training=ev::extract_native_global(snapshot,train.observed,recipe),validation=ev::extract_native_global(snapshot,val.observed,recipe);
        require(training.provenance==validation.provenance,"TRAIN/VALIDATION feature producer lineage differs");
        const auto tf=(point/"native-training.pt").string(),vf=(point/"native-validation.pt").string();
        ev::save_native_feature_archive(tf,training,train,recipe);ev::save_native_feature_archive(vf,validation,val,recipe);
        const auto te=ev::write_native_patch_reconstruction((point/"training-reconstruction.pt").string(),train,snapshot,recipe);
        const auto ve=ev::write_native_patch_reconstruction((point/"validation-reconstruction.pt").string(),val,snapshot,recipe);
        const auto json="{\"master_seed\":"+std::to_string(master)+",\"completed_updates\":"+std::to_string(budget)+",\"training_reconstruction\":"+te+",\"validation_reconstruction\":"+ve+",\"continuation\":"+result.audit_json+'}';
        write_new(point/"point.json",json+'\n');if(!first)report << ',';first=false;report << json;
        ev::ArchiveReadoutInput input;input.id="seed-"+std::to_string(master)+"-updates-"+std::to_string(budget);
        input.tag="RPB-v5";input.task="lag_sign";input.master_seed=master;input.checkpoint_steps=budget;
        input.producer_source_fingerprint=EVALUATION_SOURCE_ID;input.cohort_provenance="paired-pooling-v1/BFFPX5 retained TRAIN/VALIDATION; optimization-validation-v1";
        input.training_observations=train_path;input.validation_observations=val_path;input.training_features=tf;input.validation_features=vf;
        input.training_observations_sha256=expected.at(train_path);input.validation_observations_sha256=expected.at(val_path);
        input.expected_feature_provenance=training.provenance;readouts.inputs.push_back(input);parent=cp;
        std::cout << "Master " << master << " completed " << budget << " updates; frozen native32 and fixed-query measurements saved.\n" << std::flush;
      }
      const auto after_train=ev::extract_native_global(initial,train.observed,recipe),after_val=ev::extract_native_global(initial,val.observed,recipe);
      feature_parity(after_train,bind(old/"candidate-milestone-512/native-training.pt"));
      feature_parity(after_val,bind(old/"candidate-milestone-512/native-validation.pt"));
      ev::save_native_feature_archive((directory/"after-continuation-original512-native-training.pt").string(),after_train,train,recipe);
      ev::save_native_feature_archive((directory/"after-continuation-original512-native-validation.pt").string(),after_val,val,recipe);
      for(const auto &[name,split]:std::vector<std::pair<std::string,const ev::ControlledDataset *>>{{"training",&train},{"validation",&val}}) {
        const auto witness=directory/("after-continuation-original512-"+name+"-reconstruction.pt");
        write_new(directory/("after-continuation-original512-"+name+"-reconstruction.json"),ev::write_native_patch_reconstruction(witness.string(),*split,initial,recipe));
        query_parity(witness.string(),bind(old/("candidate-"+name+"-reconstruction.pt")));
      }
      write_new(directory/"after-continuation-original512-parity.json","{\"native_training_exact\":true,\"native_validation_exact\":true,\"fixed_queries_exact\":true,\"head_refits\":0,\"retained_head_byte_witness\":\"runner SHA256 before/after; independent prediction replay\"}\n");
    }
    report << "]}\n";write_new(output/"continuation-report.json",report.str());
    ev::run_archive_readout(readouts);std::cout << "All declared TRAIN/VALIDATION points measured; no TEST, selection or promotion.\n";
    return 0;
  } catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
