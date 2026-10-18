// SPDX-License-Identifier: MIT
#include "embedding/shared/paired_pooling.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replication_adapter.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
namespace ev=embedding::evaluation;
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
namespace fs=std::filesystem;
void require(bool value,const std::string &message) {
  if (!value) throw std::runtime_error("[context replication CLI] "+message);
}
bool sha256(const std::string &value) {
  return value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
std::string quote(const std::string &value) {
  std::ostringstream out;out << '"';
  for (const unsigned char c:value) {
    if (c=='"' || c=='\\') out << '\\' << c;
    else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str()+'"';
}
void durable_plan(const fs::path &path,const std::string &value) {
  const int file=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  require(file>=0,"launch plan must be new");
  size_t offset=0;
  while (offset<value.size()) {
    const auto count=::write(file,value.data()+offset,value.size()-offset);
    if (count<=0) { ::close(file);throw std::runtime_error("launch plan write failed"); }
    offset+=static_cast<size_t>(count);
  }
  const int synced=::fsync(file),closed=::close(file);
  require(synced==0 && closed==0,"launch plan fsync failed");
  const int directory=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
  require(directory>=0,"launch plan parent missing");
  const int directory_synced=::fsync(directory),directory_closed=::close(directory);
  require(directory_synced==0 && directory_closed==0,"launch plan parent fsync failed");
}
std::map<std::string,std::string> retained_hashes(const fs::path &root,const std::string &path) {
  std::ifstream input(path);require(bool(input),"frozen reference manifest required");
  std::map<std::string,std::string> result;std::string line;
  while (std::getline(input,line)) {
    require(line.size()>66 && line.substr(64,2)=="  " && sha256(line.substr(0,64)),"bad reference hash record");
    const fs::path relative(line.substr(66));
    require(!relative.is_absolute() && relative.lexically_normal()==relative &&
        relative.generic_string().find("..") == std::string::npos,"reference path escapes its preparation directory");
    const auto full=fs::absolute(root/relative).lexically_normal().string();
    require(result.emplace(full,line.substr(0,64)).second,"duplicate reference path");
  }
  require(input.eof() && !result.empty(),"reference manifest empty/unreadable");
  return result;
}
ev::RetainedPoolingCohort cohort(uint64_t master,const fs::path &root,
    const std::map<std::string,std::string> &hashes) {
  ev::RetainedPoolingCohort result;result.master_seed=master;
  result.lineage="context-replication-v1/native-development-v1/source-"+std::string(EVALUATION_SOURCE_ID);
  const auto base=fs::absolute(root/("seed-"+std::to_string(master)+"-lag_sign"));
  const auto bind=[&](const fs::path &path) {
    const auto full=path.lexically_normal().string();
    require(hashes.count(full),"reference role missing from frozen manifest: "+full);
    result.expected_sha256.emplace(full,hashes.at(full));return full;
  };
  result.training_observations=bind(base/"controlled-training.pt");
  result.validation_observations=bind(base/"controlled-validation.pt");
  bind(base/"development-manifest.json");bind(base/"trainer-audit.json");
  result.reference_checkpoint=bind(base/"milestone-512/checkpoint.pt");
  result.reference_initial_checkpoint=bind(base/"milestone-0/checkpoint.pt");
  for (const auto budget:{0,512}) {
    const auto point=base/("milestone-"+std::to_string(budget));bind(point/"point.json");
    for (const std::string suffix:{".audit.pt",".scaler.pt",".training-raw.pt"}) bind(point/("checkpoint.pt"+suffix));
  }
  result.reference_training_features=bind(base/"milestone-512/native-training.pt");
  result.reference_validation_features=bind(base/"milestone-512/native-validation.pt");
  result.reference_initial_training_features=bind(base/"milestone-0/native-training.pt");
  result.reference_initial_validation_features=bind(base/"milestone-0/native-validation.pt");
  result.raw_scaler=bind(base/"controls/raw-scaler.pt");
  for (const std::string rep:{"rep-1","rep-2","rep-3"}) {
    result.reference_fits.push_back(bind(base/"milestone-512"/rep/"fit.pt"));
    result.reference_initial_fits.push_back(bind(base/"milestone-0"/rep/"fit.pt"));
    result.reference_validation_predictions.push_back(bind(base/"milestone-512"/rep/"validation-predictions.pt"));
    result.reference_initial_validation_predictions.push_back(bind(base/"milestone-0"/rep/"validation-predictions.pt"));
    result.raw_fits.push_back(bind(base/"controls/raw"/rep/"fit.pt"));
    result.pca_fits.push_back(bind(base/"controls/pca_only"/rep/"fit.pt"));
    result.mask_fits.push_back(bind(base/"controls/mask_metadata"/rep/"fit.pt"));
    result.raw_validation_predictions.push_back(bind(base/"controls/raw"/rep/"validation-predictions.pt"));
    result.pca_validation_predictions.push_back(bind(base/"controls/pca_only"/rep/"validation-predictions.pt"));
    result.mask_validation_predictions.push_back(bind(base/"controls/mask_metadata"/rep/"validation-predictions.pt"));
  }
  bind(fs::absolute(root/"native-development-card.json"));
  bind(fs::absolute(root/"development-complete.json"));
  bind(fs::absolute(root/"validation-report.json"));
  return result;
}
} // namespace
int main(int argc,char **argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--source-id") { std::cout << EVALUATION_SOURCE_ID << '\n';return 0; }
    if (argc==1 || (argc==2 && std::string(argv[1])=="--help")) {
      std::cout << "embedding_context_replication --phase gate|prepare|compare --output NEW_DIRECTORY\n"
          "Prepare/compare require --admission-log LOG --admission-log-sha256 SHA256; compare also --reference DIRECTORY --retained-hashes FROZEN_MANIFEST\n"
          "Fixed context-replication-v1: five fresh paired v4/v6 trainers, native32, 512 updates, fixed heads.\n";
      return argc==1?1:0;
    }
    std::map<std::string,std::string> options;
    for (int i=1;i<argc;i+=2) {
      const std::string name=argv[i];
      require(i+1<argc && (name=="--output" || name=="--phase" || name=="--reference" ||
          name=="--retained-hashes" || name=="--admission-log" || name=="--admission-log-sha256") &&
          options.emplace(name,argv[i+1]).second,"unknown/duplicate option or missing value");
    }
    require(options.count("--output") && options.count("--phase"),"output and phase required");
    const auto phase=options.at("--phase");
    require(phase=="gate" || phase=="prepare" || phase=="compare","unknown phase");
    const bool gate=phase=="gate",compare=phase=="compare";
    require(compare ? (options.count("--reference") && options.count("--retained-hashes")) :
        (!options.count("--reference") && !options.count("--retained-hashes")),"reference roles belong only to compare");
    require(gate ? (!options.count("--admission-log") && !options.count("--admission-log-sha256")) :
        (options.count("--admission-log") && options.count("--admission-log-sha256") &&
         fs::is_regular_file(options.at("--admission-log")) && sha256(options.at("--admission-log-sha256"))),
        "measurement requires preserved coordinated CUDA admission");
    require(torch::cuda::is_available() && sha256(EVALUATION_SOURCE_ID),"actual CUDA and compiled source identity required");
    ev::PairedPoolingRun run;run.protocol_id="context-replication-v1";
    run.training_protocol_namespace="native-development-v1";
    run.fresh_test_namespace="context-replication-v1/fresh-testing";
    run.fresh_test_stream=0x6372763174657374ULL;run.candidate_tag="RPB-v6";
    auto &recipe=run.recipe;recipe.output_directory=fs::absolute(options.at("--output")).lexically_normal().string();
    recipe.source_fingerprint=EVALUATION_SOURCE_ID;recipe.git_head=EVALUATION_GIT_HEAD;recipe.git_dirty=EVALUATION_GIT_DIRTY;
    recipe.model_tag=compare?"RPB-v6":"RPB-v4";recipe.card.id=compare?"context-replication-v1":"native-development-v1";
    recipe.card.version=2;recipe.card.policy_version="1.2";recipe.card.stage="development";
    recipe.card.seeds={4404,5505,6606,7707,8808};recipe.card.tasks={ev::Task::lag_sign};
    recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=compare?64:0;
    recipe.card.comparisons.clear();recipe.milestones={0,512};recipe.sanity_budget=512;
    recipe.fresh_test_stream=run.fresh_test_stream;recipe.stress_sweep=compare;recipe.development_only=!compare;
    require(!fs::exists(recipe.output_directory),"output already exists");
    fs::create_directories(fs::path(recipe.output_directory).parent_path());
    auto settings=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf");
    settings.model.device=torch::kCUDA;settings.steps=512;rpb::validate_settings(settings);
    require(settings.model.channel_count==3 && settings.model.history_length==32 && settings.model.input_width==3 &&
        settings.model.patch_length==8 && settings.model.encoder_width==64 && settings.model.export_width==32 &&
        settings.model.channel_mixer_layers==1 && settings.model.global_bottleneck_mode==2 && settings.batch_size==8 &&
        settings.model.mask_ratio==0.25 && settings.model.huber_delta==1.0 && settings.model.dropout==0 &&
        settings.learning_rate==0.001 && settings.weight_decay==0.0001 && settings.gradient_clip_norm==1.0,
        "resolved recipe differs from frozen geometry/optimizer");
    if (compare) {
      const fs::path reference=fs::absolute(options.at("--reference")).lexically_normal();
      require(fs::is_regular_file(reference/"development-complete.json"),"reference preparation must complete before comparison");
      const auto hashes=retained_hashes(reference,options.at("--retained-hashes"));
      for (const auto master:recipe.card.seeds) run.cohorts.push_back(cohort(master,reference,hashes));
    }
    std::ostringstream plan;
    plan << "{\"protocol\":\"context-replication-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\",\"phase\":" << quote(phase)
        << ",\"reference\":\"RPB-v4\",\"candidate\":\"RPB-v6\",\"masters\":[4404,5505,6606,7707,8808],"
        "\"geometry\":[3,32,3],\"source_pairs\":[128,64," << (compare?64:0) << "],\"batch_size\":8,\"completed_updates\":512,"
        "\"parameters_each\":225805,\"native_size\":32,\"post_encoder_projection\":false,\"natural_missing_rate\":0.1,"
        "\"training_protocol_namespace\":\"native-development-v1\",\"baseline_preparation\":\"TRAIN and VALIDATION only; point0/512; no selection or TEST; durable all-point witnesses\","
        "\"comparison\":\"baseline files hashed before candidate training and fresh TEST; equal initialization/scaler/data/original counter streams\","
        "\"training_policy_id\":\"rpb-training-context-deletion-v1\",\"context_coordinate_deletion_rate\":0.3,\"context_rng_stream\":7166485043407384432,"
        "\"heads\":{\"ridge_penalty\":1,\"neural_hidden\":16,\"neural_steps\":100,\"neural_learning_rate\":0.01,\"base_seeds\":[2701,2802,2903]},"
        "\"primary_cases\":[\"intact\",\"additional_coordinate_dropout_30_percent\"],\"secondary_stress_cases\":12,"
        "\"lead_rule\":\"both mean ridge effects positive; equal coverage; no lower worst master; mean TRAIN and VALIDATION fixed-query MAE no worse; no automatic promotion\","
        "\"fresh_test_namespace\":\"context-replication-v1/fresh-testing\",\"fresh_test_stream\":" << run.fresh_test_stream
        << ",\"testing_in_this_phase\":" << (compare?"true":"false") << ",\"reference_directory\":" << quote(compare?fs::absolute(options.at("--reference")).string():"not-applicable")
        << ",\"retained_hash_manifest\":" << quote(compare?fs::absolute(options.at("--retained-hashes")).string():"not-applicable")
        << ",\"source_fingerprint\":" << quote(EVALUATION_SOURCE_ID) << ",\"git_head\":" << quote(EVALUATION_GIT_HEAD)
        << ",\"git_dirty\":" << quote(EVALUATION_GIT_DIRTY) << ",\"base_encoder_settings_before_master_override\":" << quote(rpb::settings_text(settings))
        << ",\"actual_seed_policy\":\"each declared master overrides base seed; per-master checkpoint and training producer audits retain actual settings\""
        << ",\"admission_log\":" << quote(gate?"engineering-only":fs::absolute(options.at("--admission-log")).string())
        << ",\"admission_log_sha256\":" << quote(gate?"engineering-only":options.at("--admission-log-sha256")) << "}\n";
    durable_plan(recipe.output_directory+".launch-plan.json",plan.str());
    torch::set_num_threads(recipe.card.threads);
    rpb::run_native_curve_cuda_gate(settings,gate?recipe.output_directory:recipe.output_directory+"-cuda-gate");
    std::cout << "Replication launch plan and actual CUDA/native32 serving gate passed.\n" << std::flush;
    if (phase=="prepare") ev::run_native_curve(recipe,{"RPB-v4",rpb::settings_text(settings),rpb::make_learning_curve_trainer(settings)});
    if (compare) ev::run_paired_pooling(run,{"RPB-v6",rpb::settings_text(settings)+"\ntraining_policy_id=rpb-training-context-deletion-v1\ncontext_coordinate_deletion_rate=0.30\ncontext_rng_stream=7166485043407384432\n",rpb::make_learning_curve_trainer(settings,rpb::ContextDeletionOptions{true})},
        rpb::make_retained_curve_snapshot,[](const auto &path,const auto &retained,const auto &fit) {
          return rpb::audit_context_replication_initialization(path,retained,fit,512);
        });
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n';return 1; }
}
