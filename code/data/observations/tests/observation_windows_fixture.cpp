// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
// Generates an explicitly synthetic TRAIN archive for bounded tooling checks.
#include "embedding/data/observation_windows.h"
#include <iostream>
#include <map>
#include <sstream>

namespace ow=embedding::data::observation_windows;
int main(int argc,char **argv){try{
  if(argc==2&&std::string(argv[1])=="--help"){
    std::cout<<"observation_windows_fixture --output NEWFILE --channels C --history N --features F --rows B --sampling-interval S\n"
      <<"Generates only artificial TRAIN observations; no application quality or real split membership. Semantic IDs are 11+7*c.\n";return 0;}
  TORCH_CHECK(argc==13,"[window fixture] use --help for six required options");
  std::map<std::string,std::string> options;
  const std::set<std::string> allowed{"--output","--channels","--history","--features","--rows","--sampling-interval"};
  for(int i=1;i<argc;i+=2)TORCH_CHECK(allowed.count(argv[i])&&options.emplace(argv[i],argv[i+1]).second,"[window fixture] unknown or duplicate option");
  const auto integer=[&](const char *key){size_t n=0;const auto value=std::stoll(options.at(key),&n);TORCH_CHECK(n==options.at(key).size()&&value>0,"[window fixture] positive integer required");return value;};
  const auto C=integer("--channels"),N=integer("--history"),F=integer("--features"),B=integer("--rows");
  size_t consumed=0;const auto interval=std::stod(options.at("--sampling-interval"),&consumed);
  TORCH_CHECK(consumed==options.at("--sampling-interval").size()&&std::isfinite(interval)&&interval>0,"[window fixture] positive finite interval required");
  TORCH_CHECK(C<=1000000&&N<=1000000&&F<=1000000&&B<=1000000&&C<=10000000/N/F/B,
      "[window fixture] fixture bounded to ten million cells; choose a small engineering cohort");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  auto values=torch::zeros({B,C,N,F},torch::kFloat64),observed=torch::ones({B,C,N,F},torch::kBool);
  auto x=values.accessor<double,4>();auto m=observed.accessor<bool,4>();
  for(int64_t b=0;b<B;++b)for(int64_t c=0;c<C;++c)for(int64_t t=0;t<N;++t)for(int64_t f=0;f<F;++f){
    x[b][c][t][f]=10.+c*.7+f*.2+b*.3+.03*t+.6*std::sin(t*.27+f*.31+c*.13+b*.22);
    m[b][c][t][f]=(t+f+c+b)%29!=0;
  }
  std::vector<int64_t> ids;for(int64_t c=0;c<C;++c)ids.push_back(11+7*c);
  ow::Dataset data{values,observed,torch::tensor(ids,torch::kInt64),torch::arange(N,torch::kFloat64)*interval,
      torch::full({B},(N-1)*interval,torch::kFloat64),torch::arange(B,torch::kInt64),interval,
      "synthetic-observation-window-fixture-v1","synthetic-TRAIN-only-no-quality-population","TRAIN"};
  ow::write(options.at("--output"),data);
  std::cout<<"{\"kind\":\"synthetic_observation_window_fixture_v1\",\"status\":\"written\",\"role\":\"TRAIN\",\"rows\":"<<B<<",\"channels\":"<<C<<",\"history\":"<<N<<",\"features\":"<<F<<",\"synthetic_only\":true}"<<std::endl;return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
