// SPDX-License-Identifier: MIT
#include "embedding/shared/global_bottleneck_experiment.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <torch/cuda.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
namespace ev=embedding::evaluation;
namespace rpb=embedding::encoders::raw_patch_bottleneck_mae;
namespace fs=std::filesystem;
void require(bool ok,const std::string &message) {
  if(!ok) throw std::runtime_error("[global bottleneck] "+message);
}
std::vector<int64_t> integers(const std::string &value) {
  require(!value.empty() && value.back()!=',',"empty integer list");
  std::vector<int64_t> result;std::istringstream input(value);std::string field;
  while(std::getline(input,field,',')) {
    require(!field.empty() && field.find_first_not_of("0123456789")==std::string::npos,
            "expected nonnegative integer list");
    result.push_back(std::stoll(field));
  }
  return result;
}
std::vector<ev::Task> tasks(const std::string &value) {
  std::vector<ev::Task> result;std::istringstream input(value);std::string field;
  while(std::getline(input,field,',')) {
    if(field=="reversal") result.push_back(ev::Task::reversal);
    else if(field=="level") result.push_back(ev::Task::level);
    else if(field=="amplitude") result.push_back(ev::Task::amplitude);
    else if(field=="lag_sign") result.push_back(ev::Task::lag_sign);
    else throw std::runtime_error("unknown/empty task");
  }
  return result;
}
void gpu_check(const std::string &output,const std::vector<ev::NamedCurveFactory> &factories) {
  require(!fs::exists(output),"GPU check requires a new output directory");fs::create_directories(output);
  const embedding::input_shape_t shape{3,32,3,torch::kFloat64,torch::kCPU};
  const auto protocol=ev::make_controlled_development_protocol(ev::Task::lag_sign,shape,4,2,424243,.1);
  ev::ProviderFitInput fit{protocol.training.observed,shape,424243,protocol.training.source_ids,
      {0,1,2},"unitless,unitless,unitless","global-bottleneck-gpu-check-v1",1,31};
  std::ofstream report(fs::path(output)/"gpu-check.json");require(bool(report),"cannot write GPU verification");
  report << std::setprecision(17) << "{\"cuda_available\":true,\"checks\":[";
  bool first=true;
  for(const auto &named:factories) {
    auto trainer=named.factory(fit);const auto progress=trainer.train_to(4);
    require(progress.completed==4 && progress.parameter_count>0 &&
      progress.parameter_count==progress.cuda_parameter_count && progress.last_input_cuda &&
      progress.last_loss_cuda && progress.finite_gradients && progress.weights_changed,
      "actual CUDA update verification failed for "+named.name);
    const auto path=(fs::path(output)/(named.name+".pt")).string();trainer.save_checkpoint(path);
    const auto snapshot=trainer.snapshot(path);
    const auto features=snapshot.features.extract(protocol.validation.observed);
    require(features.size()==2,"expected exact global and channel surfaces");
    for(const auto &[name,surface]:features) { (void)name;ev::validate_features(surface); }
    auto checkpoint=rpb::load_checkpoint(path,torch::kCUDA);checkpoint.model->eval();
    for(auto &parameter:checkpoint.model->parameters()) parameter.set_requires_grad(false);
    const auto patches=protocol.validation.observed.feature_mask.reshape({4,3,4,24}).any(-1);
    const auto eligible=patches.sum(-1).ge(3).logical_and(patches.select(2,0));
    auto hidden=torch::zeros_like(protocol.validation.observed.feature_mask);
    hidden.narrow(2,0,8).copy_(eligible.unsqueeze(-1).unsqueeze(-1).expand({4,3,8,3}));
    rpb::Input raw{protocol.validation.observed.data,protocol.validation.observed.feature_mask,
        torch::tensor({0,1,2},torch::kInt64),torch::full({4},31,torch::kFloat64),1};
    torch::NoGradGuard no_grad;
    const auto expected=checkpoint.model->forward(checkpoint.scaler.transform(raw,checkpoint.settings.model),hidden);
    require(expected.reconstruction.is_cuda(),"checkpoint reconstruction must actually reside on CUDA");
    const auto reconstructed=snapshot.reconstruct(protocol.validation.observed,hidden);
    require(torch::equal(reconstructed.prediction,expected.reconstruction.to(torch::kCPU)) &&
      torch::equal(reconstructed.eligible,expected.eligible_channels.to(torch::kCPU)),
      "exact frozen CUDA checkpoint reconstruction parity failed");
    const auto exported=rpb::compact_reconstruction_export(expected.encoding,checkpoint.settings.model);
    require(exported.is_cuda() && exported.dim()==(checkpoint.settings.model.global_bottleneck_mode==0?3:2),
      "declared sole reconstruction export mismatch");
    if(!first) report << ',';
    first=false;
    report << "{\"architecture\":\"" << named.name << "\",\"global_bottleneck_mode\":"
      << checkpoint.settings.model.global_bottleneck_mode << ",\"completed\":4,\"parameters\":"
      << progress.parameter_count << ",\"cuda_parameters\":" << progress.cuda_parameter_count
      << ",\"input_cuda\":true,\"loss_cuda\":true,\"finite_gradients\":true,\"weights_changed\":true,"
         "\"checkpoint_reconstruction_cuda_verified\":true,\"checkpoint_parity_exact\":true,\"training_seconds\":"
      << progress.training_seconds << '}';
    std::cout << "GPU verified: " << named.name << "; CUDA parameters=" << progress.cuda_parameter_count
      << "; four finite updates; exact checkpoint decoder/export parity\n";
  }
  report << "],\"status\":\"passed\"}\n";require(bool(report),"failed to save GPU gate");
}
}

int main(int argc,char **argv) {
  try {
    if(argc<2 || std::string(argv[1])=="--help") {
      std::cout << "embedding_global_bottleneck --output NEW_DIRECTORY [--gpu-check true]\n"
        "  [--seeds 1401,1502,1603] [--milestones 0,128,512]\n"
        "  [--train-pairs 32] [--validation-pairs 64] [--test-pairs 64]\n"
        "  [--tasks reversal,level,amplitude,lag_sign] [--stress true]\n"
        "  [--fixed-test-budget 0] [--validation-stream 0]\n"
        "  [--training-reservoir-pairs 0]\n"
        "CUDA training; current mixer / mean-global / learned-global comparison.\n";
      return argc<2?1:0;
    }
    const std::set<std::string> permitted{"--output","--gpu-check","--seeds","--milestones",
      "--train-pairs","--validation-pairs","--test-pairs","--tasks","--stress","--card-id",
      "--fixed-test-budget","--validation-stream","--training-reservoir-pairs"};
    std::map<std::string,std::string> options;
    for(int i=1;i<argc;i+=2) require(i+1<argc && permitted.count(argv[i]) &&
      options.emplace(argv[i],argv[i+1]).second,"unknown/duplicate option or missing value");
    const auto get=[&](const std::string &key,const std::string &fallback) {
      const auto found=options.find(key);return found==options.end()?fallback:found->second;
    };
    ev::GlobalBottleneckRun run;run.output_directory=get("--output","");
    require(!run.output_directory.empty(),"--output is required");
    const auto check=get("--gpu-check","false"),stress=get("--stress","true");
    require((check=="true" || check=="false") && (stress=="true" || stress=="false"),"boolean options require true/false");
    require(torch::cuda::is_available(),"CUDA unavailable; no CPU training fallback");
    run.source_fingerprint=EVALUATION_SOURCE_ID;run.git_head=EVALUATION_GIT_HEAD;run.git_dirty=EVALUATION_GIT_DIRTY;
    run.card.id=get("--card-id",run.card.id);run.card.seeds.clear();
    for(const auto seed:integers(get("--seeds","1401,1502,1603"))) run.card.seeds.push_back(seed);
    run.milestones=integers(get("--milestones","0,128,512"));require(!run.milestones.empty(),"empty budgets");
    const auto count=[&](const std::string &key,const std::string &fallback) {
      const auto values=integers(get(key,fallback));require(values.size()==1,"expected one count");return values[0];
    };
    run.card.train_pairs=count("--train-pairs","32");run.card.validation_pairs=count("--validation-pairs","64");
    run.card.test_pairs=count("--test-pairs","64");
    run.fixed_test_budget=count("--fixed-test-budget","0");
    run.validation_seed_stream=count("--validation-stream","0");
    run.training_reservoir_pairs=count("--training-reservoir-pairs","0");
    run.card.tasks=tasks(get("--tasks","reversal,level,amplitude,lag_sign"));run.stress_sweep=stress=="true";
    torch::set_num_threads(run.card.threads);
    const auto base=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf");
    std::vector<ev::NamedCurveFactory> factories;
    const std::vector<std::string> names{"current_mixer","mean_global","learned_global"};
    for(int64_t mode=0;mode<3;++mode) {
      auto settings=base;settings.model.global_bottleneck_mode=mode;settings.model.device=torch::kCUDA;
      settings.steps=std::max<int64_t>(4,*std::max_element(run.milestones.begin(),run.milestones.end()));
      rpb::validate_settings(settings);
      require(settings.model.channel_count==run.card.shape.channel_count && settings.model.history_length==run.card.shape.history_length &&
        settings.model.input_width==run.card.shape.input_width && settings.model.patch_length==run.patch_length &&
        settings.model.huber_delta==run.huber_delta && settings.batch_size==8 && settings.model.channel_mixer_layers==1,
        "configuration differs from frozen geometry/recipe");
      factories.push_back({names[mode],rpb::settings_text(settings),rpb::make_learning_curve_trainer(settings)});
    }
    if(check=="true") gpu_check(run.output_directory,factories);
    else ev::run_global_bottleneck_experiment(run,factories);
    return 0;
  } catch(const std::exception &error) {std::cerr << error.what() << '\n';return 1;}
}
