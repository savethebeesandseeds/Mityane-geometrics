// SPDX-License-Identifier: MIT
#include "cross_feature_solvability.h"
#include <torch/torch.h>
#include <cmath>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ev=embedding::evaluation;
namespace {
void check(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}check(caught,"invalid input accepted");}
embedding::Batch signal(double P,double L,double phase,int label){
  auto x=torch::zeros({1,3,32,3},torch::kFloat64);
  auto a=x.accessor<double,4>();constexpr double pi=3.14159265358979323846;
  for(int c=0;c<3;++c)for(int t=0;t<32;++t)for(int f=0;f<3;++f){
    const double gain=(c==0?.55:c==1?1.35:1.)*(1+.1*f);
    a[0][c][t][f]=(c==0?.71:c==1?-.73:.3)+gain*std::sin(2*pi*(t+(c==1?(label?L:-L):0))/P+phase);
  }
  return{x,torch::ones_like(x,torch::kBool)};
}
void fixtures(){
  constexpr double pi=3.14159265358979323846;int cases=0;
  for(double P:{10.,17.,24.})for(double L:{.25,.6,1.})for(double phase:{0.,1.31,5.98})for(int label:{0,1}){
    auto batch=signal(P,L,phase,label);const auto result=ev::cross_feature_solvability(batch);
    check(result.valid.item<bool>()&&result.predictions.item<int64_t>()==label&&result.supported_time_positions.item<int64_t>()==30,"cross-feature noiseless sign/support");
    double theoretical=0;
    for(int centre=1;centre<31;++centre){double sum=0;int spacings=0;
      for(int k=1;k<=4;++k){if(centre-k<0||centre+k>=32)continue;
        sum+=4*(.55*1.1)*(1.35*1.1)*std::pow(std::sin(pi*k/P),2)*std::sin(2*pi*k/P)*std::sin(2*pi*L/P);++spacings;}
      theoretical+=sum/spacings;
    }
    theoretical=(label?1:-1)*theoretical/30;
    check(std::abs(result.margins.item<double>()-theoretical)<2e-12,"independent Cartesian feature-pair formula");
    ++cases;
  }
  for(int label:{0,1}){
    auto batch=signal(17,.4,5.1,label);batch.feature_mask.zero_();
    batch.feature_mask.select(1,0).select(2,0).fill_(true);
    batch.feature_mask.select(1,1).select(2,1).fill_(true);
    check(!ev::structured_hard_timing_solvability(batch).valid.item<bool>(),"old same-feature rule has no tuple");
    const auto full=ev::cross_feature_solvability(batch);
    check(full.valid.item<bool>()&&full.predictions.item<int64_t>()==label&&full.supported_time_positions.item<int64_t>()==30,"independent-feature legal evidence");
    batch.feature_mask.slice(2,6,32).zero_();
    const auto four=ev::cross_feature_solvability(batch);
    check(four.valid.item<bool>()&&four.predictions.item<int64_t>()==label&&four.supported_time_positions.item<int64_t>()==4,"four supported centres valid");
    batch.feature_mask.select(2,5).zero_();
    const auto three=ev::cross_feature_solvability(batch);
    check(!three.valid.item<bool>()&&three.predictions.item<int64_t>()==0&&three.supported_time_positions.item<int64_t>()==3,"three centres abstain with zero prediction");
    auto hidden=batch;hidden.data=batch.data.clone();hidden.data.masked_fill_(~hidden.feature_mask,std::numeric_limits<double>::quiet_NaN());
    const auto ignored=ev::cross_feature_solvability(hidden);
    check(torch::equal(three.margins,ignored.margins)&&torch::equal(three.valid,ignored.valid),"hidden-value independence");
  }
  auto bad=signal(16,.5,1.1,1);bad.data[0][0][0][0]=std::numeric_limits<double>::infinity();rejects([&]{ev::cross_feature_solvability(bad);});
  bad=signal(16,.5,1.1,1);bad.feature_mask=bad.feature_mask.to(torch::kFloat64);rejects([&]{ev::cross_feature_solvability(bad);});
  auto constant=signal(16,.5,1.1,1);constant.data.fill_(.2);check(!ev::cross_feature_solvability(constant).valid.item<bool>(),"zero margin abstains");
  auto f32=signal(16,.5,1.1,1);f32.data=f32.data.to(torch::kFloat32);check(ev::cross_feature_solvability(f32).predictions.item<int64_t>()==1,"F32 observations accepted");
  std::cout<<"Cross-feature timing data-only fixtures passed: "<<cases<<" affine noiseless cases; no encoder/head fitting or quality evaluation\n";
}
std::string stats(const ev::ControlledDataset &data,const embedding::Batch &batch,bool deleted,bool &passed){
  const auto result=ev::cross_feature_solvability(batch);
  const auto total=data.labels.size(0),valid=result.valid.sum().item<int64_t>();
  const auto correct=result.predictions.eq(data.labels).logical_and(result.valid).sum().item<int64_t>();
  const double accuracy=valid?double(correct)/valid:0,coverage=double(valid)/total;
  const bool ok=accuracy>=(deleted?.98:.99)&&coverage>=(deleted?.95:.99);passed=passed&&ok;
  std::ostringstream out;out.precision(17);out<<"{\"total\":"<<total<<",\"valid\":"<<valid<<",\"correct\":"<<correct
    <<",\"accuracy\":"<<accuracy<<",\"coverage\":"<<coverage<<",\"full_population_correctness\":"<<double(correct)/total<<",\"passed\":"<<(ok?"true":"false")<<'}';return out.str();
}
void information(const std::filesystem::path &path){
  check(path.is_absolute()&&path.parent_path().parent_path()==std::filesystem::path("/embedding/output/runs/rpb-structured-hard-timing/information-admission")&&
    path.filename()=="information.json"&&!std::filesystem::exists(path)&&std::filesystem::canonical(path.parent_path())==path.parent_path(),"exclusive information output bounds");
  bool passed=true;std::ostringstream out;out<<"{\"protocol\":\"structured-hard-timing-information-v2\",\"rule_id\":\"observed-cross-feature-determinant-v2\",\"dataset_id\":\"TEMPO-3\",\"complexity\":4,\"complexity_maximum\":5,\"seeds\":[";
  bool first=true;for(uint64_t seed:{82585ULL,83686ULL}){
    auto data=ev::make_structured_hard_timing_development(512,512,seed);
    const auto view=ev::make_coordinate_deletion_view(data.validation.observed,data.validation.source_ids,ev::Task::lag_sign,
      ev::stream_seed(seed,ev::kStructuredHardTimingDeletionStream),.30,"structured-hard-timing-information-v2/lag_sign/deletion");
    if(!first)out<<',';
    first=false;out<<"{\"seed\":"<<seed<<",\"training_source_pairs\":512,\"validation_source_pairs\":512,\"intact\":"
      <<stats(data.validation,data.validation.observed,false,passed)<<",\"deleted\":"<<stats(data.validation,view.observations,true,passed)<<'}';
  }
  out<<"],\"passed\":"<<(passed?"true":"false")<<",\"encoder_or_head_fits\":0,\"TEST_generated\":false}\n";
  const auto body=out.str();const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);check(fd>=0,"exclusive information record");
  const auto written=::write(fd,body.data(),body.size());const int closed=::close(fd);check(written==static_cast<ssize_t>(body.size())&&closed==0,"complete information record");
  std::cout<<body;check(passed,"cross-feature information admission failed; preserve recipe and result, do not train encoders");
}
}
int main(int argc,char**argv){try{torch::set_num_threads(1);if(argc==1)fixtures();else{check(argc==3&&std::string(argv[1])=="--information-admission","closed test/information modes");information(argv[2]);}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
