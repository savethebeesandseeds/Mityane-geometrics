// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Waajacu. See the repository LICENSE.
// Test-only variants of an artificial archive; no application data migration.
#include "embedding/data/observation_windows.h"
#include <iostream>
#include <map>
namespace ow=embedding::data::observation_windows;
int main(int argc,char **argv){try{
  TORCH_CHECK(argc==7,"usage: rpb_dataset_variant --input FILE --output NEWFILE --mode TEST|allmissing|ineligible|nonfinite|wrong-ids");
  std::map<std::string,std::string> options;const std::set<std::string> allowed{"--input","--output","--mode"};
  for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&options.emplace(argv[i],argv[i+1]).second,"unique supported options required");
  auto data=ow::read(options.at("--input"));const auto mode=options.at("--mode");
  if(mode=="TEST")data.role="TEST";
  else if(mode=="allmissing")data.observed=torch::zeros_like(data.observed);
  else if(mode=="ineligible"){
    data.observed=torch::zeros_like(data.observed);
    data.observed.select(2,data.values.size(2)-1).fill_(true);
  }else if(mode=="nonfinite"){
    data.observed.select(2,data.values.size(2)-1).fill_(true);
    data.values.select(2,data.values.size(2)-1).fill_(std::numeric_limits<double>::quiet_NaN());
  }else if(mode=="wrong-ids")data.channel_ids=data.channel_ids+100000;
  else TORCH_CHECK(false,"unknown artificial variant");
  ow::write(options.at("--output"),data);
  std::cout<<"{\"kind\":\"rpb_artificial_dataset_variant_v1\",\"status\":\"written\",\"mode\":\""<<mode<<"\"}"<<std::endl;return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
