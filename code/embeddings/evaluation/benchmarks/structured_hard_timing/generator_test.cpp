// SPDX-License-Identifier: MIT
#include "structured_hard_timing.h"
#include <torch/torch.h>
#include <cmath>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ev=embedding::evaluation;
namespace {
void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F call){bool rejected=false;try{call();}catch(const std::exception&){rejected=true;}check(rejected,"invalid input accepted");}
embedding::Batch signal(double P,double L,double phase,int label) {
  auto x=torch::zeros({1,3,32,3},torch::kFloat64);
  auto a=x.accessor<double,4>();constexpr double pi=3.14159265358979323846;
  for(int64_t c=0;c<3;++c)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f){
    const double gain=(c==0?.55:c==1?1.35:1.0)*(1+.1*f);
    const double offset=c==0?.71:c==1?-.73:.3;
    const double shift=c==1?(label?L:-L):0;
    a[0][c][t][f]=offset+gain*std::sin(2*pi*(t+shift)/P+phase);
  }
  return{x,torch::ones_like(x,torch::kBool)};
}
void dataset_checks(const ev::ControlledDataset &data) {
  check(data.observed.data.scalar_type()==torch::kFloat64,"F64 observations");
  check(torch::equal(data.clean.data,data.observed.data)&&torch::equal(data.clean.feature_mask,data.observed.feature_mask),"legal-only clean field");
  check(data.clean.data.data_ptr()!=data.observed.data.data_ptr(),"clean must be a separate clone");
  check(data.observed.data.masked_select(~data.observed.feature_mask).eq(0).all().item<bool>(),"zero hidden storage");
  const auto n=data.labels.size(0);check(n%2==0&&data.source_ids.size()==static_cast<size_t>(n),"paired source count");
  check(data.labels.eq(0).sum().item<int64_t>()==n/2&&data.labels.eq(1).sum().item<int64_t>()==n/2,"balanced paired labels");
  for(int64_t row=0;row<n;row+=2){
    check(data.source_ids[row]==data.source_ids[row+1]&&data.labels[row].item<int64_t>()!=data.labels[row+1].item<int64_t>(),"source-paired variants");
    check(torch::equal(data.observed.feature_mask[row],data.observed.feature_mask[row+1]),"label-independent paired mask");
    for(int64_t c:{0,2})check(torch::equal(data.observed.data[row][c],data.observed.data[row+1][c]),"paired nonlabel channels");
    for(int64_t c=0;c<3;++c){bool gap=false;for(int64_t t=0;t<30;++t)
      gap=gap||!data.observed.feature_mask[row][c].slice(0,t,t+3).any().item<bool>();
      check(gap,"three-tick whole-feature channel gap");}
  }
}
void fixtures(){
  torch::set_num_threads(1);
  auto data=ev::make_structured_hard_timing_development(12,7,901901);
  auto same=ev::make_structured_hard_timing_development(12,7,901901);
  auto other=ev::make_structured_hard_timing_development(12,7,901902);
  dataset_checks(data.training);dataset_checks(data.validation);
  check(torch::equal(data.training.observed.data,same.training.observed.data)&&data.training.source_ids==same.training.source_ids,"deterministic local generation");
  check(!torch::equal(data.training.observed.data,other.training.observed.data),"seed changes observations");
  std::set<std::string> training(data.training.source_ids.begin(),data.training.source_ids.end());
  for(const auto&id:data.validation.source_ids)check(!training.count(id),"source-disjoint splits");
  check(training.size()==12&&std::set<std::string>(data.validation.source_ids.begin(),data.validation.source_ids.end()).size()==7,"split before pairing");
  torch::manual_seed(123);auto expected=torch::rand({8});torch::manual_seed(123);
  (void)ev::make_structured_hard_timing_development(3,2,44);check(torch::equal(expected,torch::rand({8})),"unchanged Torch RNG");
  rejects([]{ev::make_structured_hard_timing_development(0,2,1);});
  rejects([]{ev::make_structured_hard_timing_development(1,-1,1);});
  rejects([]{ev::make_structured_hard_timing_development(std::numeric_limits<int64_t>::max(),1,1);});
  constexpr double pi=3.14159265358979323846;int cases=0;
  for(double P:{10.,17.,24.})for(double L:{.25,.6,1.})for(double phase:{0.,1.31,5.98})for(int label:{0,1}){
    auto observations=signal(P,L,phase,label);auto result=ev::structured_hard_timing_solvability(observations);
    check(result.valid.item<bool>()&&result.predictions.item<int64_t>()==label,"affine noiseless analytic sign");
    check(result.supported_time_positions.item<int64_t>()==30,"distinct centre support");
    double theoretical=0;
    for(int centre=1;centre<31;++centre){double centre_sum=0;int spacings=0;
      for(int k=1;k<=4;++k){if(centre-k<0||centre+k>=32)continue;double features=0;
        for(int f=0;f<3;++f)features+=4*.55*1.35*(1+.1*f)*(1+.1*f)*std::pow(std::sin(pi*k/P),2)*std::sin(2*pi*k/P)*std::sin(2*pi*L/P);
        centre_sum+=features/3;++spacings;}
      theoretical+=centre_sum/spacings;}
    theoretical=(label?1:-1)*theoretical/30;
    check(std::abs(result.margins.item<double>()-theoretical)<2e-12,"independent centred-difference formula");++cases;
  }
  auto sparse=signal(16,.6,.3,1);sparse.feature_mask.zero_();sparse.feature_mask.slice(2,0,6).fill_(true);
  auto four=ev::structured_hard_timing_solvability(sparse);check(four.valid.item<bool>()&&four.supported_time_positions.item<int64_t>()==4,"four centre support valid");
  sparse.feature_mask.select(2,5).zero_();auto three=ev::structured_hard_timing_solvability(sparse);
  check(!three.valid.item<bool>()&&three.supported_time_positions.item<int64_t>()==3,"three centre support abstains");
  auto hidden=sparse;hidden.data=sparse.data.clone();hidden.data.masked_fill_(~hidden.feature_mask,std::numeric_limits<double>::quiet_NaN());
  auto hidden_result=ev::structured_hard_timing_solvability(hidden);check(torch::equal(three.margins,hidden_result.margins)&&torch::equal(three.valid,hidden_result.valid),"hidden NaN independence");
  auto constant=signal(16,.6,.3,1);constant.data.fill_(.2);check(!ev::structured_hard_timing_solvability(constant).valid.item<bool>(),"zero margin abstains");
  auto observed_inf=signal(16,.6,.3,1);observed_inf.data[0][0][0][0]=std::numeric_limits<double>::infinity();rejects([&]{ev::structured_hard_timing_solvability(observed_inf);});
  auto wrong=signal(16,.6,.3,1);wrong.feature_mask=wrong.feature_mask.to(torch::kFloat64);rejects([&]{ev::structured_hard_timing_solvability(wrong);});
  auto f32=signal(16,.6,.3,1);f32.data=f32.data.to(torch::kFloat32);check(ev::structured_hard_timing_solvability(f32).predictions.item<int64_t>()==1,"F32 legal observations");
  auto deleted=ev::make_coordinate_deletion_view(data.validation.observed,data.validation.source_ids,ev::Task::lag_sign,
      ev::stream_seed(901901,ev::kStructuredHardTimingDeletionStream),.30,"structured-hard-timing-engineering/deletion");
  for(int64_t row=0;row<data.validation.labels.size(0);row+=2)check(torch::equal(deleted.observations.feature_mask[row],deleted.observations.feature_mask[row+1]),"source-paired extra deletion");
  check(deleted.observations.data.masked_select(~deleted.observations.feature_mask).eq(0).all().item<bool>(),"deleted hidden storage zero");
  std::cout<<"Structured hard timing data-only fixtures passed: "<<cases<<" affine noiseless cases; no encoder/head fitting or quality evaluation\n";
}
std::string stats(const ev::ControlledDataset &data,const embedding::Batch &observations,bool deleted,bool &passed){
  auto result=ev::structured_hard_timing_solvability(observations);
  const auto total=data.labels.size(0),valid=result.valid.sum().item<int64_t>();
  const auto correct=((result.predictions==data.labels)&result.valid).sum().item<int64_t>();
  const double accuracy=valid?double(correct)/valid:0,coverage=double(valid)/total;
  const bool ok=accuracy>=(deleted?.98:.99)&&coverage>=(deleted?.95:.99);passed=passed&&ok;
  std::ostringstream out;out.precision(17);out<<"{\"total\":"<<total<<",\"valid\":"<<valid<<",\"correct\":"<<correct
    <<",\"accuracy\":"<<accuracy<<",\"coverage\":"<<coverage<<",\"full_population_correctness\":"<<double(correct)/total<<",\"passed\":"<<(ok?"true":"false")<<'}';return out.str();
}
void information(const std::filesystem::path &path){
  check(path.is_absolute()&&path.parent_path().parent_path()==std::filesystem::path("/embedding/output/runs/rpb-structured-hard-timing/information-admission")&&
    path.filename()=="information.json"&&!std::filesystem::exists(path)&&std::filesystem::canonical(path.parent_path())==path.parent_path(),"exclusive information output bounds");
  bool passed=true;std::ostringstream out;out<<"{\"protocol\":\"structured-hard-timing-information-v1\",\"dataset_id\":\"TEMPO-3\",\"complexity\":4,\"complexity_maximum\":5,\"seeds\":[";
  bool first=true;for(uint64_t seed:{80383ULL,81484ULL}){
    auto data=ev::make_structured_hard_timing_development(512,512,seed);
    auto view=ev::make_coordinate_deletion_view(data.validation.observed,data.validation.source_ids,ev::Task::lag_sign,
      ev::stream_seed(seed,ev::kStructuredHardTimingDeletionStream),.30,"structured-hard-timing-information-v1/lag_sign/deletion");
    if(!first)out<<',';first=false;out<<"{\"seed\":"<<seed<<",\"training_source_pairs\":512,\"validation_source_pairs\":512,\"intact\":"
      <<stats(data.validation,data.validation.observed,false,passed)<<",\"deleted\":"<<stats(data.validation,view.observations,true,passed)<<'}';
  }
  out<<"],\"passed\":"<<(passed?"true":"false")<<",\"encoder_or_head_fits\":0,\"TEST_generated\":false}\n";
  const auto body=out.str();
  const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
  check(fd>=0,"exclusive new information output");
  const auto written=::write(fd,body.data(),body.size());const int closed=::close(fd);
  check(written==static_cast<ssize_t>(body.size())&&closed==0,"complete information record write");
  std::cout<<out.str();check(passed,"structured information admission failed; preserve recipe and result, do not train encoders");
}
} // namespace
int main(int argc,char**argv){try{torch::set_num_threads(1);if(argc==1)fixtures();else{check(argc==3&&std::string(argv[1])=="--information-admission","closed test/information modes");information(argv[2]);}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
