// SPDX-License-Identifier: MIT
#include "configurable_capacity.h"
#include "embedding/shared/data.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cc=embedding::evaluation::configurable_capacity;
namespace fs=std::filesystem;
namespace {
void require(bool value,const std::string &message) { if(!value)throw std::runtime_error(message); }
struct Evidence {int64_t total=0,supported=0,correct=0;double seconds=0;};
Evidence information(const fs::path &path,const cc::Split &split) {
  const auto start=std::chrono::steady_clock::now();
  const auto decisions=cc::infer_information(split.observed,split.mask);
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  const auto n=split.observed.size(0);
  auto prediction=torch::zeros({n},torch::kInt64),valid=torch::zeros({n},torch::kBool);
  auto diagnostics=torch::zeros({n,4},torch::kFloat64),counts=torch::zeros({n},torch::kInt64);
  auto coefficients=torch::zeros({n,6,5},torch::kFloat64),eligible=torch::zeros({n,6},torch::kBool);
  auto p=prediction.accessor<int64_t,1>();auto v=valid.accessor<bool,1>();auto d=diagnostics.accessor<double,2>();
  auto c=counts.accessor<int64_t,1>();auto beta=coefficients.accessor<double,3>();auto e=eligible.accessor<bool,2>();
  std::ostringstream reasons;reasons<<'[';Evidence score{n,0,0,seconds};
  auto labels=split.scoring_labels.accessor<int64_t,1>();
  for(int64_t row=0;row<n;++row) {
    const auto &decision=decisions[row];p[row]=decision.predicted_label;v[row]=decision.supported;
    d[row][0]=decision.slow_frequency;d[row][1]=decision.fast_frequency;
    d[row][2]=decision.normalized_residual;d[row][3]=decision.margin;c[row]=decision.frequency_pairs_evaluated;
    for(int f=0;f<6;++f) {
      e[row][f]=decision.eligible[f];for(int col=0;col<5;++col)beta[row][f][col]=decision.coefficients[f][col];
    }
    if(row)reasons<<',';
    reasons<<'"'<<decision.reason<<'"';
    if(v[row]){++score.supported;if(p[row]==labels[row])++score.correct;}
  }
  reasons<<']';torch::serialize::OutputArchive out;
  out.write("predictions",prediction,true);out.write("valid",valid,true);out.write("diagnostics",diagnostics,true);
  out.write("diagnostics_columns",embedding::archive::text_tensor("slow_frequency,fast_frequency,normalized_residual,slow_phase_margin"),true);
  out.write("frequency_pairs_evaluated",counts,true);out.write("coefficients",coefficients,true);out.write("eligible",eligible,true);
  out.write("reasons_json",embedding::archive::text_tensor(reasons.str()),true);
  out.write("rule",embedding::archive::text_tensor(cc::kInformationRule),true);
  out.write("labels_scoring_only",split.scoring_labels,true);out.write("source_ids",split.source_ids,true);
  out.write("source_ids_json",embedding::archive::text_tensor(cc::source_ids_json(split)),true);
  out.write("cpu_information_seconds",torch::tensor(seconds,torch::kFloat64),true);
  cc::publish_archive(path.string(),out);
  return score;
}
bool master(const fs::path &root,uint64_t seed,int64_t history,std::ostream &report,bool first) {
  const auto directory=root/("master-"+std::to_string(seed));
  require(fs::create_directory(directory),"master output directory already exists");
  const auto data=cc::make_development(256,128,seed,{3,history,9});
  const auto deleted=cc::delete_observations(data.validation,.30,seed);
  cc::save_split_archive((directory/"training.pt").string(),data.training,"TRAIN",seed);
  cc::save_split_archive((directory/"validation.pt").string(),data.validation,"VALIDATION",seed);
  cc::save_split_archive((directory/"validation-deleted.pt").string(),deleted,"VALIDATION_DELETED",seed,.30);
  bool passed=true;std::map<std::string,const cc::Split *> views{
      {"training",&data.training},{"validation",&data.validation},{"validation-deleted",&deleted}};
  if(!first)report<<',';
  report<<"{\"master\":"<<seed<<",\"history\":"<<history<<",\"views\":[";
  bool first_view=true;
  for(const auto &[name,split]:views) {
    const auto evidence=information(directory/(name+"-information.pt"),*split);
    if(!first_view)report<<',';
    first_view=false;
    const bool gate=evidence.total==evidence.supported&&evidence.total==evidence.correct;passed=passed&&gate;
    report<<"{\"view\":\""<<name<<"\",\"total\":"<<evidence.total<<",\"supported\":"<<evidence.supported
        <<",\"correct\":"<<evidence.correct<<",\"accuracy\":"<<(evidence.supported?double(evidence.correct)/evidence.supported:0)
        <<",\"coverage\":"<<double(evidence.supported)/evidence.total<<",\"cpu_information_seconds\":"<<evidence.seconds
        <<",\"passed\":"<<(gate?"true":"false")<<'}';
    std::cout<<"TEMPO-5 master="<<seed<<" H="<<history<<" "<<name<<" correct="<<evidence.correct
        <<" supported="<<evidence.supported<<" total="<<evidence.total<<" CPU information seconds="<<evidence.seconds<<'\n';
  }
  report<<"],\"passed\":"<<(passed?"true":"false")<<'}';return passed;
}
}
int main(int argc,char **argv) {
  try {
    require(argc==3&&(std::string(argv[1])=="--engineering"||std::string(argv[1])=="--quality"),
        "usage: configurable_capacity_data --engineering|--quality NEW_OUTPUT_DIRECTORY");
    const bool engineering=std::string(argv[1])=="--engineering";
    torch::set_num_threads(1);torch::set_num_interop_threads(1);fs::path root=argv[2];
    require(!root.empty()&&!fs::exists(root),"a new output directory is required; failures are preserved");
    if(!root.parent_path().empty())fs::create_directories(root.parent_path());
    require(fs::create_directory(root),"cannot create new output directory");
    std::ostringstream report;report<<std::setprecision(17)<<"{\"protocol\":\""<<cc::kProtocolId
        <<"\",\"dataset\":\""<<cc::kDatasetId<<"\",\"complexity\":5,\"complexity_maximum\":5,\"stage\":\""
        <<(engineering?"engineering":"development-quality")<<"\",\"rule\":\""<<cc::kInformationRule
        <<"\",\"channel_count\":3,\"input_width\":9,\"training_pairs\":256,\"validation_pairs\":128"
        <<",\"additional_validation_deletion\":0.30,\"observation_archive_keys\":[\"observed\",\"feature_mask\",\"labels_scoring_only\",\"source_ids\",\"source_ids_json\"]"
        <<",\"gate\":\"all rows correct and supported; no redraw\",\"masters\":[";
    bool passed=true;
    if(engineering) {
      passed=master(root,941000,128,report,true);
      const auto extension=root/"history-256";fs::create_directory(extension);
      passed=master(extension,941000,256,report,false)&&passed;
    } else {
      bool first=true;
      for(const uint64_t seed:{941101ULL,941102ULL,941103ULL}) {
        passed=master(root,seed,128,report,first)&&passed;first=false;
      }
    }
    report<<"],\"passed\":"<<(passed?"true":"false")<<"}\n";
    std::ofstream out(root/"information-report.json",std::ios::binary);out<<report.str();out.close();
    require(bool(out),"cannot publish information report");
    require(passed,"information gate failed; all evidence retained; stop quality training");
    std::cout<<"TEMPO-5 observed-information admission passed; encoder training not executed\n";return 0;
  } catch(const std::exception &error) {
    std::cerr<<"configurable_capacity_data: "<<error.what()<<'\n';return 1;
  }
}
