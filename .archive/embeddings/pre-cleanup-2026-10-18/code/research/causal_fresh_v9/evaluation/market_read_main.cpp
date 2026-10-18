// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#include "cuwacunu/evaluation/fixed_regression.h"
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv){
  try {
    torch::set_num_threads(1);
    cuwacunu::evaluation::FixedRegressionRun run;std::string training,validation;
    struct Paths {std::string name,training,validation;};std::vector<Paths> methods;
    auto next=[&](int &i){if(++i>=argc)throw std::runtime_error("missing argument value");return std::string(argv[i]);};
    for(int i=1;i<argc;++i){const std::string option=argv[i];
      if(option=="--train-targets")training=next(i);
      else if(option=="--validation-targets")validation=next(i);
      else if(option=="--output-directory")run.output_directory=next(i);
      else if(option=="--card-sha256")run.card_sha256=next(i);
      else if(option=="--input-manifest-sha256")run.input_manifest_sha256=next(i);
      else if(option=="--dataset-id")run.dataset_id=next(i);
      else if(option=="--method"){Paths m;m.name=next(i);m.training=next(i);m.validation=next(i);methods.push_back(m);}
      else throw std::runtime_error("unknown argument: "+option);
    }
    auto hash=[](const std::string &s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;};
    if(training.empty()||validation.empty()||run.output_directory.empty()||methods.empty()||!hash(run.card_sha256)||!hash(run.input_manifest_sha256)||run.dataset_id.empty()||run.dataset_id.size()>64||run.dataset_id.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")!=std::string::npos)
      throw std::runtime_error("usage: market_fixed_regression --train-targets FILE --validation-targets FILE --method NAME TRAIN_FILE VAL_FILE [--method ...] --output-directory NEW_DIR --card-sha256 HEX --input-manifest-sha256 HEX [--dataset-id NAME]");
    run.training=cuwacunu::evaluation::load_targets(training);run.validation=cuwacunu::evaluation::load_targets(validation);run.target_specs=cuwacunu::evaluation::past_window_target_specs();
    for(const auto &m:methods)run.methods.push_back({m.name,m.training,m.validation,cuwacunu::evaluation::load_features(m.training),cuwacunu::evaluation::load_features(m.validation)});
    std::cout<<cuwacunu::evaluation::run_fixed_regression(run)<<'\n';return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
