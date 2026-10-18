// SPDX-License-Identifier: MIT
#include "embedding/shared/learning_curve.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <torch/cuda.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
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
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[learning curve] " + message);
}
std::vector<int64_t> integers(const std::string &value) {
  std::vector<int64_t> out;
  std::istringstream in(value); std::string field;
  require(!value.empty() && value.back() != ',', "empty integer list");
  while (std::getline(in, field, ',')) {
    require(!field.empty() && field.find_first_not_of("0123456789") == std::string::npos,
            "expected nonnegative integer list");
    out.push_back(std::stoll(field));
  }
  return out;
}
void gpu_check(const std::string &output, const std::vector<ev::NamedCurveFactory> &factories) {
  require(!fs::exists(output), "GPU check requires a new output directory");
  fs::create_directories(output);
  const embedding::input_shape_t shape{3, 32, 3, torch::kFloat64, torch::kCPU};
  const auto data = ev::make_controlled_development_protocol(ev::Task::lag_sign,
      shape, 4, 2, 424242, 0.1);
  ev::ProviderFitInput fit{data.training.observed, shape, 424242,
      data.training.source_ids, {0,1,2}, "unitless,unitless,unitless",
      "learning-curve-gpu-check-v1", 1.0, 31.0};
  std::ofstream report(fs::path(output) / "gpu-check.json");
  require(bool(report), "cannot write GPU verification");
  report << "{\"cuda_available\":true,\"checks\":[";
  bool first = true;
  for (const auto &named : factories) {
    auto trainer = named.factory(fit);
    const auto point = trainer.train_to(4);
    require(point.completed == 4 && point.parameter_count > 0 &&
        point.parameter_count == point.cuda_parameter_count &&
        point.last_input_cuda && point.last_loss_cuda && point.finite_gradients &&
        point.weights_changed, "actual CUDA optimizer verification failed for " + named.name);
    const auto checkpoint = fs::path(output) / (named.name + ".pt");
    trainer.save_checkpoint(checkpoint.string());
    const auto frozen = trainer.snapshot(checkpoint.string());
    const auto features = frozen.features.extract(data.validation.observed);
    require(features.size() == 2, "snapshot must expose active global and channel concatenation");
    for (const auto &[name, surface] : features) {
      ev::validate_features(surface);
      require(torch::isfinite(surface.values).all().item<bool>(), "nonfinite GPU checkpoint features");
    }
    const auto patch_observed=data.validation.observed.feature_mask.reshape({4,3,4,24}).any(-1);
    const auto eligible=patch_observed.sum(-1).ge(3).logical_and(patch_observed.select(2,0));
    auto hidden=torch::zeros_like(data.validation.observed.feature_mask);
    hidden.narrow(2,0,8).copy_(eligible.unsqueeze(-1).unsqueeze(-1).expand({4,3,8,3}));
    const auto reconstruction=frozen.reconstruct(data.validation.observed,hidden);
    require(reconstruction.prediction.device().is_cpu() && reconstruction.target.device().is_cpu() &&
      reconstruction.prediction.sizes()==data.validation.observed.data.sizes() &&
      reconstruction.target.sizes()==reconstruction.prediction.sizes() &&
      torch::isfinite(reconstruction.prediction).all().item<bool>() &&
      torch::isfinite(reconstruction.target).all().item<bool>() &&
      torch::equal(reconstruction.eligible,eligible), "GPU checkpoint reconstruction verification failed");
    if (!first) report << ',';
    first = false;
    report << "{\"architecture\":\"" << named.name << "\",\"completed\":" << point.completed
        << ",\"parameters\":" << point.parameter_count
        << ",\"cuda_parameters\":" << point.cuda_parameter_count
        << ",\"input_cuda\":true,\"loss_cuda\":true,\"finite_gradients\":true,"
           "\"weights_changed\":true,\"checkpoint_reconstruction_cuda_verified\":true,\"training_seconds\":" << point.training_seconds << '}';
    std::cout << "GPU verified: " << named.name << ", all " << point.parameter_count
        << " parameters CUDA; input/loss CUDA; four finite updates changed weights\n";
  }
  report << "],\"status\":\"passed\"}\n";
  require(bool(report), "failed to save GPU verification");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help") {
      std::cout << "embedding_learning_curve --output NEW_DIRECTORY [--gpu-check true]\n"
        "  [--independent-config FILE] [--mixer-config FILE]\n"
        "  [--seeds 901,1002,1103] [--milestones 0,128,512,2048]\n"
        "  [--train-pairs 32] [--validation-pairs 64] [--test-pairs 64]\n"
        "CUDA is required for encoder training; frozen probes run on CPU.\n"
        "Development lag-sign curve; common budget selected before fresh test generation.\n";
      return argc < 2 ? 1 : 0;
    }
    const std::set<std::string> permitted{"--output", "--gpu-check", "--independent-config",
      "--mixer-config", "--seeds", "--milestones", "--train-pairs", "--validation-pairs", "--test-pairs"};
    std::map<std::string,std::string> options;
    for (int i=1; i<argc; i+=2) {
      require(i+1<argc && permitted.count(argv[i]), "unknown option or missing value");
      require(options.emplace(argv[i],argv[i+1]).second, "duplicate option");
    }
    auto get = [&](const std::string &key,const std::string &fallback) {
      const auto found=options.find(key); return found==options.end()?fallback:found->second;
    };
    const auto output=get("--output", ""); require(!output.empty(), "--output is required");
    const auto check=get("--gpu-check", "false");
    require(check=="true" || check=="false", "--gpu-check must be true or false");
    require(torch::cuda::is_available(), "CUDA unavailable; GPU training is required");
    ev::LearningCurveRun run;
    torch::set_num_threads(run.threads);
    run.output_directory=output; run.source_fingerprint=EVALUATION_SOURCE_ID;
    run.git_head=EVALUATION_GIT_HEAD; run.git_dirty=EVALUATION_GIT_DIRTY;
    run.seeds.clear();
    for(const auto seed:integers(get("--seeds","901,1002,1103"))) run.seeds.push_back(seed);
    run.milestones=integers(get("--milestones","0,128,512,2048"));
    auto count=[&](const std::string &key,const std::string &fallback) {
      const auto values=integers(get(key,fallback)); require(values.size()==1,"expected one integer"); return values[0];
    };
    run.train_pairs=count("--train-pairs","32");
    run.validation_pairs=count("--validation-pairs","64"); run.test_pairs=count("--test-pairs","64");
    auto independent=rpb::read_settings(get("--independent-config",
      "code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf"));
    auto mixer=rpb::read_settings(get("--mixer-config",
      "code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf"));
    require(independent.model.channel_mixer_layers==0 && mixer.model.channel_mixer_layers==1,
      "expected unchanged independent and one-layer mixer architectures");
    auto comparable=mixer; comparable.model.channel_mixer_layers=0;
    require(rpb::settings_text(comparable)==rpb::settings_text(independent),
      "architecture comparison requires identical settings except channel_mixer_layers");
    require(independent.model.channel_count==run.shape.channel_count &&
      independent.model.history_length==run.shape.history_length &&
      independent.model.input_width==run.shape.input_width &&
      independent.model.patch_length==run.patch_length &&
      independent.model.huber_delta==run.huber_delta && independent.batch_size==8,
      "configs must agree with the frozen geometry, target recipe and batch size");
    std::vector<ev::NamedCurveFactory> factories;
    for(auto entry:std::vector<std::pair<std::string,rpb::Settings>>{{"independent",independent},{"mixer",mixer}}) {
      entry.second.model.device=torch::kCUDA;
      entry.second.steps=std::max<int64_t>(1,*std::max_element(run.milestones.begin(),run.milestones.end()));
      rpb::validate_settings(entry.second);
      factories.push_back({entry.first,rpb::settings_text(entry.second),
        rpb::make_learning_curve_trainer(entry.second)});
    }
    if(check=="true") gpu_check(output,factories);
    else ev::run_learning_curve(run,factories);
    return 0;
  } catch(const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
