// SPDX-License-Identifier: MIT
#include "embedding/shared/projection_diagnostic.h"
#include <filesystem>
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
std::vector<int64_t> integers(const std::string &value) {
  if(value.empty() || value.back()==',') throw std::runtime_error("empty integer list");
  std::vector<int64_t> result; std::istringstream input(value); std::string field;
  while(std::getline(input,field,',')) {
    if(field.empty() || field.find_first_not_of("0123456789")!=std::string::npos)
      throw std::runtime_error("expected nonnegative integer list");
    result.push_back(std::stoll(field));
  }
  return result;
}
}

int main(int argc,char **argv) {
  try {
    if(argc<2 || std::string(argv[1])=="--help") {
      std::cout << "embedding_projection_diagnostic --input-root CURVE_RESULTS --output NEW_DIRECTORY\n"
        "  [--seeds 901,1002,1103] [--milestones 128,512,2048]\n"
        "  [--widths 12,24,36,48]\n"
        "Frozen training/validation features only; CPU probes; no encoder training or test inputs.\n";
      return argc<2?1:0;
    }
    const std::set<std::string> permitted{"--input-root","--output","--seeds","--milestones","--widths"};
    std::map<std::string,std::string> options;
    for(int i=1;i<argc;i+=2) {
      if(i+1>=argc || !permitted.count(argv[i]) || !options.emplace(argv[i],argv[i+1]).second)
        throw std::runtime_error("unknown/duplicate option or missing value");
    }
    const auto get=[&](const std::string &key,const std::string &fallback) {
      const auto found=options.find(key);return found==options.end()?fallback:found->second;
    };
    namespace fs=std::filesystem; namespace ev=embedding::evaluation;
    const fs::path root=get("--input-root","");
    ev::ProjectionDiagnosticRun run;
    run.output_directory=get("--output","");
    if(root.empty() || !fs::is_directory(root) || run.output_directory.empty())
      throw std::runtime_error("--input-root must be an existing curve results directory; --output is required");
    run.source_fingerprint=EVALUATION_SOURCE_ID;run.git_head=EVALUATION_GIT_HEAD;
    run.git_dirty=EVALUATION_GIT_DIRTY;run.widths=integers(get("--widths","12,24,36,48"));
    for(const auto seed:integers(get("--seeds","901,1002,1103"))) {
      const auto cohort=root/("seed-"+std::to_string(seed));
      for(const std::string architecture:{"independent","mixer"})
        for(const auto budget:integers(get("--milestones","128,512,2048"))) {
          const auto point=cohort/architecture/("milestone-"+std::to_string(budget));
          ev::ProjectionArchiveInput input;
          input.id=architecture+"-"+std::to_string(seed)+"-"+std::to_string(budget);
          input.architecture=architecture;input.master_seed=seed;input.checkpoint_steps=budget;
          input.training_features=(point/"curve_channel_concatenation-training.pt").string();
          input.validation_features=(point/"curve_channel_concatenation-validation.pt").string();
          input.training_labels=(cohort/"controlled-training.pt").string();
          input.validation_labels=(cohort/"controlled-validation.pt").string();
          run.inputs.push_back(std::move(input));
        }
    }
    ev::run_projection_diagnostic(run);
    return 0;
  } catch(const std::exception &error) {std::cerr << error.what() << '\n';return 1;}
}
