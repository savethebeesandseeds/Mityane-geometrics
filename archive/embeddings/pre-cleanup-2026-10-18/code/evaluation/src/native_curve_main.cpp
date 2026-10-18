// SPDX-License-Identifier: MIT
#include "embedding/shared/native_curve.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <torch/cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

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
namespace ev = embedding::evaluation;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace fs = std::filesystem;
void require(bool ok,const std::string &message) {
  if (!ok) throw std::runtime_error("[native curve CLI] " + message);
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (const unsigned char byte:value) {
    if (byte=='"' || byte=='\\') out << '\\' << byte;
    else if (byte=='\n') out << "\\n";
    else if (byte=='\r') out << "\\r";
    else if (byte=='\t') out << "\\t";
    else { require(byte>=32,"unsupported JSON control byte"); out << byte; }
  }
  return out.str()+'"';
}
void save_exclusive(const fs::path &path,const std::string &text) {
  const int file = ::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0644);
  require(file>=0,"cannot create new launch plan: " + path.string());
  size_t written=0;
  while (written<text.size()) {
    const auto count=::write(file,text.data()+written,text.size()-written);
    if (count<=0) { ::close(file); throw std::runtime_error("launch plan write failed"); }
    written+=size_t(count);
  }
  const bool synced=::fsync(file)==0;
  const bool closed=::close(file)==0;
  require(synced && closed,"launch plan durable save failed");
}
}

int main(int argc,char **argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--source-id") {
      std::cout << EVALUATION_SOURCE_ID << '\n';
      return 0;
    }
    if (argc==1 || (argc==2 && std::string(argv[1])=="--help")) {
      std::cout << "embedding_native_curve --output NEW_DIRECTORY [--gpu-check true]\n"
        "Fixed native-curve-v1: RPB-v4 CUDA; fresh masters3101/3202/3303;\n"
        "timing0/128/512/2048, other tasks0/512, fixed raw/PCA-only/native heads.\n"
        "Every full curve passes the actual CUDA/export/decoder gate first.\n";
      return argc==1?1:0;
    }
    std::map<std::string,std::string> options;
    for (int index=1;index<argc;index+=2) {
      const std::string name=argv[index];
      require(index+1<argc && (name=="--output" || name=="--gpu-check") &&
        options.emplace(name,argv[index+1]).second,"unknown/duplicate option or missing value");
    }
    require(options.count("--output") && !options.at("--output").empty(),"--output required");
    const bool gate_only=options.count("--gpu-check") && options.at("--gpu-check")=="true";
    require(!options.count("--gpu-check") || gate_only,"--gpu-check accepts true only");
    require(torch::cuda::is_available(),"CUDA unavailable; training requires the managed GPU container");

    ev::NativeCurveRun run;
    run.output_directory=fs::absolute(options.at("--output")).lexically_normal().string();
    run.source_fingerprint=EVALUATION_SOURCE_ID;
    run.git_head=EVALUATION_GIT_HEAD; run.git_dirty=EVALUATION_GIT_DIRTY;
    require(!fs::exists(run.output_directory),"output already exists");
    fs::create_directories(fs::path(run.output_directory).parent_path());
    auto settings=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf");
    settings.model.device=torch::kCUDA;
    settings.steps=2048;
    rpb::validate_settings(settings);
    require(settings.model.channel_mixer_layers==1 && settings.model.global_bottleneck_mode==2 &&
      settings.model.export_width==32 && settings.model.channel_count==3 &&
      settings.model.history_length==32 && settings.model.input_width==3 && settings.model.patch_length==8 &&
      settings.model.huber_delta==1.0 && settings.model.mask_ratio==0.25 &&
      settings.model.encoder_width==64 && settings.model.num_layers==3 && settings.model.num_heads==4 &&
      settings.model.feedforward_width==256 && settings.model.decoder_hidden_width==128 &&
      settings.model.dropout==0 && settings.model.layer_norm_epsilon==1e-5 &&
      settings.model.scale_floor==1e-6 && settings.model.sampling_interval==1 &&
      settings.batch_size==8 && settings.threads==1 && settings.learning_rate==0.001 &&
      settings.weight_decay==0.0001 && settings.gradient_clip_norm==1.0,
      "resolved settings differ from the frozen RPB-v4 recipe");
    torch::set_num_threads(run.card.threads);
    const std::string gate_directory=gate_only?run.output_directory:run.output_directory+"-cuda-gate";
    const std::string recipe=rpb::settings_text(settings);
    std::ostringstream plan;
    plan << "{\"protocol\":\"native-curve-v1\",\"policy_version\":\"1.2\",\"stage\":\"development\","
      "\"model_tag\":\"RPB-v4\",\"source_fingerprint\":" << quote(run.source_fingerprint)
      << ",\"git_head\":" << quote(run.git_head) << ",\"git_dirty\":" << quote(run.git_dirty)
      << ",\"resolved_encoder_settings\":" << quote(recipe)
      << ",\"development_masters\":[3101,3202,3303],\"geometry\":[3,32,3],\"natural_missing_rate\":0.1,"
      "\"split_source_pairs\":{\"training\":128,\"validation\":64,\"testing\":64},"
      "\"legal_raw_oracle_gate\":{\"minimum_accuracy\":0.95,\"nonzero_support_required\":true,\"failure\":\"abort and preserve artifacts; no replacement draws\"},"
      "\"timing_milestones\":[0,128,512,2048],\"sanity_milestones\":[0,512],"
      "\"sanity_tasks\":[\"reversal\",\"level\",\"amplitude\"],"
      "\"selection\":\"mean native32 ridge validation over every master; identical checkpoint supports; positive budgets; unsupported budget excluded; exact ties smaller\","
      "\"controls\":\"raw576 and standalone raw PCA32 fitted once per cohort; same-path point0 native32\","
      "\"heads\":{\"ridge_penalty\":1,\"neural_hidden\":16,\"neural_updates\":100,\"neural_learning_rate\":0.01,"
      "\"base_seeds\":[2701,2802,2903]},\"fresh_test_namespace\":\"native-curve-v1/fresh-testing\","
      "\"fresh_test_stream\":" << run.fresh_test_stream << ",\"test_generation\":\"only after selection.json durable save\","
      "\"stress\":\"shared twelve-case fixed-readout protocol; no refitting\",\"gpu_gate_only\":"
      << (gate_only?"true":"false") << ",\"gpu_gate_file\":" << quote(gate_directory+"/gpu-check.json")
      << ",\"results_directory\":" << quote(run.output_directory) << "}\n";
    save_exclusive(run.output_directory+".launch-plan.json",plan.str());
    std::cout << "Launch plan frozen: " << run.output_directory << ".launch-plan.json\n" << std::flush;
    rpb::run_native_curve_cuda_gate(settings,gate_directory);
    std::cout << "RPB-v4 actual CUDA and exact32 bottleneck gate passed.\n" << std::flush;
    if (gate_only) return 0;
    ev::run_native_curve(run,{"RPB-v4",recipe,rpb::make_learning_curve_trainer(settings)});
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
