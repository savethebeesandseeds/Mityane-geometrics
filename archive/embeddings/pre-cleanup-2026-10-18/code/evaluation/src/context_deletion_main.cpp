// SPDX-License-Identifier: MIT
#include "embedding/shared/paired_pooling.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion_adapter.h"
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
const std::string reference_capsule="output/runs/rpb-native-curve/native-curve-5G8O5c";
const std::string reference_inventory_sha="c9922d3c817630da3d8609b7ba8d2b47cab7434a17fe28bf5ae272a841da0d95";
void require(bool value,const std::string &message) {
  if (!value) throw std::runtime_error("[context deletion CLI] " + message);
}
bool sha256(const std::string &value) {
  return value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (const unsigned char character:value) {
    if (character=='"' || character=='\\') out << '\\' << character;
    else if (character<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(character) << std::dec;
    else out << character;
  }
  return out.str()+'"';
}
void durable_plan(const std::string &path,const std::string &value) {
  const int file=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  require(file>=0,"launch plan must be a new file");
  size_t written=0;
  while (written<value.size()) {
    const auto count=::write(file,value.data()+written,value.size()-written);
    if (count<=0) { ::close(file); throw std::runtime_error("launch plan write failed"); }
    written+=static_cast<size_t>(count);
  }
  const int synced=::fsync(file),closed=::close(file);
  require(synced==0 && closed==0,"launch plan fsync failed");
  const int directory=::open(fs::path(path).parent_path().c_str(),O_RDONLY|O_DIRECTORY);
  require(directory>=0,"launch plan directory unavailable");
  const int directory_synced=::fsync(directory),directory_closed=::close(directory);
  require(directory_synced==0 && directory_closed==0,"launch plan directory fsync failed");
}
std::map<std::string,std::string> retained_hashes(const std::string &path) {
  std::ifstream input(path); require(bool(input),"retained hash manifest required");
  std::map<std::string,std::string> result; std::string line;
  while (std::getline(input,line)) {
    require(line.size()>66 && line.substr(64,2)=="  " && sha256(line.substr(0,64)),"invalid retained hash record");
    const auto relative=fs::path(line.substr(66));
    require(!relative.is_absolute() && relative.lexically_normal()==relative &&
        relative.generic_string().find("..") == std::string::npos,"retained path must stay inside the capsule");
    const auto full=fs::absolute(fs::path(reference_capsule)/relative).lexically_normal().string();
    require(result.emplace(full,line.substr(0,64)).second,"duplicate retained hash path");
  }
  require(input.eof() && !result.empty(),"cannot read retained hash manifest");
  return result;
}
ev::RetainedPoolingCohort cohort(uint64_t master,const std::map<std::string,std::string> &hashes) {
  ev::RetainedPoolingCohort result; result.master_seed=master;
  result.lineage="native-curve-v1/source-5a2c39d343e399c549b89f880c7e3a9e4b823d17b541775b7352c2de6ce8d643";
  const auto base=fs::absolute(fs::path(reference_capsule)/"results"/("seed-"+std::to_string(master)+"-lag_sign"));
  const auto bind=[&](const fs::path &path) {
    const auto full=path.lexically_normal().string();
    require(hashes.count(full),"declared retained input missing from frozen inventory: "+full);
    result.expected_sha256.emplace(full,hashes.at(full)); return full;
  };
  result.training_observations=bind(base/"controlled-training.pt");
  result.validation_observations=bind(base/"controlled-validation.pt");
  bind(base/"development-manifest.json");
  bind(base/"trainer-audit.json");
  result.reference_checkpoint=bind(base/"milestone-512/checkpoint.pt");
  result.reference_initial_checkpoint=bind(base/"milestone-0/checkpoint.pt");
  for (const auto budget:{0,512}) {
    bind(base/("milestone-"+std::to_string(budget))/"point.json");
    for (const std::string suffix:{".audit.pt",".scaler.pt",".training-raw.pt"})
      bind(base/("milestone-"+std::to_string(budget))/("checkpoint.pt"+suffix));
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
  const auto inventory=fs::absolute(fs::path(reference_capsule)/"artifact-integrity.json").string();
  result.expected_sha256.emplace(inventory,reference_inventory_sha);
  bind(fs::absolute(fs::path(reference_capsule)/"results/native-curve-card.json"));
  bind(fs::absolute(fs::path(reference_capsule)/"results/selection.json"));
  const std::map<uint64_t,std::string> checkpoints{{3101,"71db828e63884956024a20b8223ef89a3da1b3f031a7fe84d36cc6c1cdac351b"},
      {3202,"8cbce9f90c3c36146c2102955b279ab587b9569a0519f7815dfd0eec9e44ab73"},
      {3303,"8bdf10f69f7c927546707f9fcabf6269d4401fb63425fb1da5c4877c9b303904"}};
  require(result.expected_sha256.at(result.reference_checkpoint)==checkpoints.at(master),"reference selected checkpoint identity differs");
  return result;
}
} // namespace
int main(int argc,char **argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--source-id") { std::cout << EVALUATION_SOURCE_ID << '\n'; return 0; }
    if (argc==1 || (argc==2 && std::string(argv[1])=="--help")) {
      std::cout << "embedding_context_deletion --output NEW_DIRECTORY --retained-hashes FROZEN_MANIFEST --context-training-log LOG --context-training-log-sha256 SHA256\n"
          "Engineering gate only: --output NEW_DIRECTORY --gpu-check true\n"
          "Fixed context-deletion-v1: candidate RPB-v6 vs retained RPB-v4; timing; native32; 512 updates; fixed heads.\n";
      return argc==1?1:0;
    }
    std::map<std::string,std::string> options;
    for (int index=1;index<argc;index+=2) {
      const std::string name=argv[index];
      require(index+1<argc && (name=="--output" || name=="--retained-hashes" || name=="--gpu-check" ||
          name=="--context-training-log" || name=="--context-training-log-sha256") &&
          options.emplace(name,argv[index+1]).second,"unknown/duplicate option or missing value");
    }
    require(options.count("--output") && !options.at("--output").empty(),"new output required");
    const bool gate_only=options.count("--gpu-check") && options.at("--gpu-check")=="true";
    require(!options.count("--gpu-check") || gate_only,"gpu-check accepts true only");
    require(gate_only ? !options.count("--retained-hashes") : bool(options.count("--retained-hashes")),"use retained manifest only for the full paired experiment");
    require(gate_only ? (!options.count("--context-training-log") && !options.count("--context-training-log-sha256")) :
        (options.count("--context-training-log") && options.count("--context-training-log-sha256") &&
         fs::is_regular_file(options.at("--context-training-log")) && sha256(options.at("--context-training-log-sha256"))),
        "full measurement requires preserved enabled-context CUDA admission evidence");
    require(torch::cuda::is_available(),"actual CUDA required; no CPU fallback");
    require(sha256(EVALUATION_SOURCE_ID),"compiled source fingerprint required");
    ev::PairedPoolingRun run;
    run.protocol_id="context-deletion-v1";run.fresh_test_namespace="context-deletion-v1/fresh-testing";
    run.fresh_test_stream=0x6374763174657374ULL;run.candidate_tag="RPB-v6";
    run.recipe.output_directory=fs::absolute(options.at("--output")).lexically_normal().string();
    run.recipe.source_fingerprint=EVALUATION_SOURCE_ID;
    run.recipe.git_head=EVALUATION_GIT_HEAD; run.recipe.git_dirty=EVALUATION_GIT_DIRTY;
    run.recipe.model_tag="RPB-v6"; run.recipe.card.id="context-deletion-v1";
    run.recipe.card.version=2; run.recipe.card.policy_version="1.2"; run.recipe.card.stage="development";
    run.recipe.card.seeds={3101,3202,3303}; run.recipe.card.tasks={ev::Task::lag_sign};
    run.recipe.card.train_pairs=128; run.recipe.card.validation_pairs=64; run.recipe.card.test_pairs=64;
    run.recipe.card.comparisons.clear(); run.recipe.milestones={0,512};
    run.recipe.fresh_test_stream=run.fresh_test_stream;
    require(!fs::exists(run.recipe.output_directory),"output already exists");
    fs::create_directories(fs::path(run.recipe.output_directory).parent_path());
    auto settings=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf");
    settings.model.device=torch::kCUDA; settings.steps=512; rpb::validate_settings(settings);
    auto expected=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf");
    expected.model.device=torch::kCUDA; expected.steps=512;
    require(rpb::settings_text(settings)==rpb::settings_text(expected),"candidate changed the retained v4 inference/optimizer configuration");
    require(settings.model.channel_count==3 && settings.model.history_length==32 && settings.model.input_width==3 &&
        settings.model.patch_length==8 && settings.model.encoder_width==64 && settings.model.export_width==32 &&
        settings.model.channel_mixer_layers==1 && settings.model.global_bottleneck_mode==2 && settings.batch_size==8 &&
        settings.model.mask_ratio==0.25 && settings.model.huber_delta==1.0 && settings.model.dropout==0 &&
        settings.learning_rate==0.001 && settings.weight_decay==0.0001 && settings.gradient_clip_norm==1.0,
        "resolved recipe differs from frozen production geometry/optimizer");
    if (!gate_only) {
      const auto hashes=retained_hashes(options.at("--retained-hashes"));
      for (const auto master:run.recipe.card.seeds) run.cohorts.push_back(cohort(master,hashes));
    }
    std::ostringstream plan;
    plan << "{\"protocol\":\"context-deletion-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\","
        "\"candidate\":\"RPB-v6\",\"reference\":\"RPB-v4\",\"completed_updates\":512,\"geometry\":[3,32,3],"
        "\"masters\":[3101,3202,3303],\"source_pairs\":[128,64,64],\"natural_missing_rate\":0.1,"
        "\"primary_cases\":[\"intact\",\"additional_coordinate_dropout_30_percent\"],"
        "\"primary_metric\":\"native32 ridge accuracy with equal coverage; every master retained\","
        "\"lead_rule\":\"both mean primary ridge effects positive; worst master not lower on either; equal coverage; mean TRAIN/VALIDATION fixed-query reconstruction MAE no worse; conditional intervals retained; no automatic promotion\","
        "\"heads\":{\"ridge_penalty\":1,\"neural_hidden\":16,\"neural_steps\":100,\"neural_learning_rate\":0.01,\"base_seeds\":[2701,2802,2903]},"
        "\"native_size\":32,\"post_encoder_projection\":false,\"reference_control_fit\":\"retained tensors; no fitting constructor\","
        "\"candidate_parameters\":225805,\"reference_parameters\":225805,\"fresh_test_namespace\":\"context-deletion-v1/fresh-testing\","
        "\"training_policy_id\":\"rpb-training-context-deletion-v1\",\"context_coordinate_deletion_rate\":0.30,\"context_rng_stream\":\"7166485043407384432\","
        "\"target_policy\":\"original O&A, original eligibility and Huber; erase only visible context; repair only extra deletion to retain two original visible patch groups\","
        "\"initialization\":\"all mode2 point0 parameters exactly paired with retained RPB-v4; no new modules\",\"context_resume\":\"fresh continuous session only; ordinary resume rejects nonempty training policy\","
        "\"fresh_test_stream\":" << run.fresh_test_stream << ",\"testing\":\"only after durable fixed-budget manifest; no test or stress fitting\","
        "\"reference_capsule\":" << quote(fs::absolute(reference_capsule).string()) << ",\"reference_inventory_sha256\":" << quote(reference_inventory_sha)
        << ",\"retained_hash_manifest\":" << quote(gate_only ? "engineering-only" : fs::absolute(options.at("--retained-hashes")).string())
        << ",\"source_fingerprint\":" << quote(run.recipe.source_fingerprint) << ",\"git_head\":" << quote(run.recipe.git_head)
        << ",\"git_dirty\":" << quote(run.recipe.git_dirty) << ",\"resolved_encoder_settings\":" << quote(rpb::settings_text(settings))
        << ",\"context_training_log\":" << quote(gate_only ? "engineering-only" : fs::absolute(options.at("--context-training-log")).string())
        << ",\"context_training_log_sha256\":" << quote(gate_only ? "engineering-only" : options.at("--context-training-log-sha256"))
        << ",\"gpu_gate_only\":" << (gate_only?"true":"false") << "}\n";
    durable_plan(run.recipe.output_directory+".launch-plan.json",plan.str());
    std::cout << "Context deletion launch plan frozen.\n" << std::flush;
    torch::set_num_threads(run.recipe.card.threads);
    rpb::run_native_curve_cuda_gate(settings,gate_only?run.recipe.output_directory:run.recipe.output_directory+"-cuda-gate");
    std::cout << "Mode2 actual CUDA and exact native32 serving gate passed.\n" << std::flush;
    if (!gate_only) ev::run_paired_pooling(run,{"RPB-v6",rpb::settings_text(settings)+"\ntraining_policy_id=rpb-training-context-deletion-v1\ncontext_coordinate_deletion_rate=0.30\ncontext_rng_stream=7166485043407384432\n",rpb::make_learning_curve_trainer(settings,rpb::ContextDeletionOptions{true})},
        rpb::make_retained_curve_snapshot,rpb::audit_context_deletion_initialization);
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
