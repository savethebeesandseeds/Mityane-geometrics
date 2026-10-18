// SPDX-License-Identifier: MIT
// Data-only engineering admission: no encoder, optimizer, head or hidden truth.
#include "two_component.h"
#include "observed_information.h"
#include "frozen_role_guard.h"
#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sys/stat.h>

#ifndef TWO_COMPONENT_SOURCE_ID
#define TWO_COMPONENT_SOURCE_ID "unrecorded"
#endif
namespace {
namespace tc = embedding::evaluation::two_component;
namespace fs = std::filesystem;
using namespace embedding::evaluation::frozen_inputs;
using Clock = std::chrono::steady_clock;
constexpr uint64_t masters[] = {910901, 910902};

void direct_directory(const fs::path &p) {
  require(p.is_absolute() && fs::is_directory(p) && fs::canonical(p) == p,
          "canonical directory required");
  for (auto q=p; !q.empty() && q!=q.root_path(); q=q.parent_path())
    require(!fs::is_symlink(q), "direct directory ancestry required");
}
void new_leaf(const fs::path &p) {
  require(p.is_absolute() && p.lexically_normal()==p && !p.filename().empty(),
          "absolute normalized output required");
  direct_directory(p.parent_path());
  const fs::path bound("/embedding/output/runs/two-component-information-v1");
  const auto rel=p.lexically_relative(bound);
  require(!rel.empty() && !rel.is_absolute() && *rel.begin()!=".." && p!=bound,
          "output stays in the named data-only run directory");
  require(!fs::exists(p) && !fs::is_symlink(p) && fs::create_directory(p),
          "exclusive new output directory required");
}
void validate(const tc::Split &s, int64_t rows, uint64_t master, tc::Task task) {
  const std::vector<int64_t> shape{rows,3,32,3};
  require(s.observed.device().is_cpu() && s.observed.scalar_type()==torch::kFloat64 &&
          s.observed.sizes()==torch::IntArrayRef(shape) && s.observed.is_contiguous(),
          "legal CPU F64 B3H32F3 observations");
  require(s.mask.device().is_cpu() && s.mask.scalar_type()==torch::kBool &&
          s.mask.sizes()==torch::IntArrayRef(shape) && s.mask.is_contiguous(),
          "legal CPU Bool observation mask");
  require(s.scoring_labels.device().is_cpu() && s.scoring_labels.scalar_type()==torch::kInt64 &&
          s.scoring_labels.sizes()==torch::IntArrayRef({rows}) && s.scoring_labels.is_contiguous(),
          "CPU Long scoring-only labels");
  require(s.source_ids.device().is_cpu() && s.source_ids.scalar_type()==torch::kInt64 &&
          s.source_ids.sizes()==torch::IntArrayRef({rows}) && s.source_ids.is_contiguous(),
          "CPU Long source identifiers");
  require(torch::isfinite(s.observed).all().item<bool>() &&
          s.observed.masked_select(~s.mask).eq(0).all().item<bool>(), "finite legal values and hidden zeros");
  const auto ids=s.source_ids.accessor<int64_t,1>();
  const auto labels=s.scoring_labels.accessor<int64_t,1>();
  int64_t previous=-1;
  for(int64_t i=0;i<rows;i+=2) {
    const auto id=ids[i];
    require(id>=0 && ids[i+1]==id && id>previous && (uint64_t(id)>>31)==master &&
            (id&1)==(task==tc::Task::ComponentBalance?1:0), "packed paired source identity");
    require(labels[i]+labels[i+1]==1 && (labels[i]==0 || labels[i]==1), "balanced opposite pair labels");
    require(torch::equal(s.mask[i],s.mask[i+1]), "pair-shared observation masks");
    previous=id;
  }
}
void save_data(const fs::path &p, const tc::Split &s) {
  require(!fs::exists(p), "new data archive only");
  torch::serialize::OutputArchive a;
  a.write("observations",s.observed,true); a.write("feature_mask",s.mask,true);
  a.write("labels_scoring_only",s.scoring_labels,true); a.write("source_ids",s.source_ids,true);
  a.save_to(p.string());
}
void save_decisions(const fs::path &p, const std::vector<tc::InformationDecision> &d) {
  const auto n=int64_t(d.size());
  auto predictions=torch::zeros({n},torch::kInt64), supported=torch::zeros({n},torch::kBool);
  auto slow=torch::zeros({n},torch::kFloat64), fast=torch::zeros_like(slow);
  auto residual=torch::zeros_like(slow), margin=torch::zeros_like(slow);
  auto coefficients=torch::zeros({n,6,5},torch::kFloat64);
  auto eligible=torch::zeros({n,6},torch::kBool), searchcounts=torch::zeros({n},torch::kInt64);
  auto coef=coefficients.accessor<double,3>();auto active=eligible.accessor<bool,2>();
  for(int64_t i=0;i<n;++i) {
    predictions[i]=d[i].predicted_label; supported[i]=d[i].supported;
    slow[i]=d[i].slow_frequency; fast[i]=d[i].fast_frequency;
    residual[i]=d[i].normalized_residual; margin[i]=d[i].margin;
    searchcounts[i]=d[i].frequency_pairs_evaluated;
    for(int f=0;f<6;++f){active[i][f]=d[i].eligible[f];for(int j=0;j<5;++j)coef[i][f][j]=d[i].coefficients[f][j];}
  }
  torch::serialize::OutputArchive a;
  a.write("predictions",predictions,true); a.write("supported",supported,true);
  a.write("slow_frequency",slow,true); a.write("fast_frequency",fast,true);
  a.write("normalized_residual",residual,true); a.write("margin",margin,true);
  a.write("coefficients",coefficients,true);a.write("eligible",eligible,true);
  a.write("frequency_pairs_evaluated",searchcounts,true);
  require(!fs::exists(p), "new decision archive only"); a.save_to(p.string());
}
std::string score_view(const fs::path &dir, const std::string &view, const tc::Split &s,
                       tc::Task task, bool &cohort_pass) {
  const auto information_task=task==tc::Task::SlowLagSign ?
      tc::InformationTask::SlowLagSign : tc::InformationTask::ComponentBalance;
  const auto started=Clock::now();
  // This is the only inference call. Labels and source IDs cannot enter it.
  const auto decisions=tc::infer_information(s.observed,s.mask,information_task);
  require(decisions.size()==size_t(s.observed.size(0)), "entire declared population retained");
  save_data(dir/(view+".pt"),s); save_decisions(dir/(view+"-decisions.pt"),decisions);
  int64_t supported=0,correct=0;
  std::map<std::string,int64_t> reasons;
  std::vector<std::string> row_reasons;
  const auto labels=s.scoring_labels.accessor<int64_t,1>();
  for(size_t i=0;i<decisions.size();++i) {
    const auto &d=decisions[i];
    require(d.predicted_label==0 || d.predicted_label==1, "binary prediction encoding");
    require(std::isfinite(d.slow_frequency) && std::isfinite(d.fast_frequency) &&
            std::isfinite(d.normalized_residual) && std::isfinite(d.margin), "finite saved diagnostics");
    row_reasons.push_back(d.reason); ++reasons[d.reason];
    if(d.supported) { ++supported; if(d.predicted_label==labels[i]) ++correct; }
  }
  const auto total=int64_t(decisions.size());
  const double coverage=double(supported)/total;
  const double accuracy=supported ? double(correct)/supported : 0;
  const bool deleted=view=="validation-deleted";
  const bool passed=supported>0 && accuracy>=(deleted?.98:.99) && coverage>=(deleted?.95:.99);
  cohort_pass=cohort_pass && passed;
  std::ostringstream out; out.precision(17);
  out<<"{\"view\":"<<quote(view)<<",\"total\":"<<total<<",\"valid\":"<<supported
     <<",\"correct\":"<<correct<<",\"abstained\":"<<total-supported<<",\"accuracy\":";
  if(supported)out<<accuracy; else out<<"null";
  out<<",\"coverage\":"<<coverage<<",\"full_population_correctness\":"<<double(correct)/total
     <<",\"gate_passed\":"<<(passed?"true":"false")
     <<",\"diagnostic_seconds\":"<<std::chrono::duration<double>(Clock::now()-started).count()
     <<",\"data_archive\":"<<quote(view+".pt")<<",\"decisions_archive\":"<<quote(view+"-decisions.pt")
     <<",\"row_reasons\":"<<strings(row_reasons)<<",\"reason_counts\":{";
  bool first=true;for(const auto &[reason,count]:reasons){if(!first)out<<',';first=false;out<<quote(reason)<<':'<<count;}
  return out.str()+"}}";
}
void run(const fs::path &output, const fs::path &card, const std::string &card_sha,
         const fs::path &sources, const fs::path &sdk, const std::string &sdk_sha) {
  require(is_sha(TWO_COMPONENT_SOURCE_ID) && sha256(bytes(sources))==TWO_COMPONENT_SOURCE_ID,
          "compiled closed source identity required");
  require(is_sha(card_sha) && sha256(bytes(card))==card_sha, "frozen card required");
  require(sdk_sha=="e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb" &&
          sha256(bytes(sdk))==sdk_sha, "verified SDK proof required");
  new_leaf(output);
  bool passed=true; std::set<int64_t> all_ids;
  std::ostringstream cohorts; cohorts.precision(17);cohorts<<'[';bool first=true;
  const auto started=Clock::now();
  for(const auto master:masters) for(const auto task:{tc::Task::SlowLagSign,tc::Task::ComponentBalance}) {
    const auto dir=output/("seed-"+std::to_string(master)+"-"+tc::task_name(task));
    require(fs::create_directory(dir), "new task/cohort directory");
    const auto data=tc::make_development(128,64,master,task);
    const auto deleted=tc::delete_observations(data.validation,.30,master);
    validate(data.training,256,master,task);validate(data.validation,128,master,task);validate(deleted,128,master,task);
    for(const auto *split:{&data.training,&data.validation}) {
      const auto ids=split->source_ids.accessor<int64_t,1>();
      for(int64_t i=0;i<split->source_ids.size(0);i+=2)
        require(all_ids.insert(ids[i]).second, "independent tasks/masters/splits have disjoint sources");
    }
    require(torch::equal(data.validation.source_ids,deleted.source_ids) &&
            torch::equal(data.validation.scoring_labels,deleted.scoring_labels) &&
            !deleted.mask.logical_and(~data.validation.mask).any().item<bool>() &&
            torch::equal(deleted.observed.masked_select(deleted.mask),data.validation.observed.masked_select(deleted.mask)),
            "deleted validation retains original rows, labels and legal values");
    bool cohort_pass=true;
    const auto t=score_view(dir,"training",data.training,task,cohort_pass);
    const auto v=score_view(dir,"validation-intact",data.validation,task,cohort_pass);
    const auto d=score_view(dir,"validation-deleted",deleted,task,cohort_pass);
    passed=passed && cohort_pass;
    if(!first)cohorts<<',';
    first=false;
    cohorts<<"{\"master\":"<<master<<",\"task\":"<<quote(tc::task_name(task))
           <<",\"dataset_id\":"<<quote(tc::dataset_id(task))
           <<",\"designed_complexity_level\":5,\"complexity_scale_max\":5,\"directory\":"<<quote(dir.filename().string())
           <<",\"gate_passed\":"<<(cohort_pass?"true":"false")<<",\"views\":["<<t<<','<<v<<','<<d<<"]}";
    std::cout<<"Completed data-only "<<tc::dataset_id(task)<<" engineering cohort "<<master
             <<": "<<(cohort_pass?"gate passed":"gate failed")<<std::endl;
  }
  cohorts<<']';require(all_ids.size()==768,"all engineering independent source groups retained");
  std::ostringstream inventory;inventory<<'[';bool first_file=true;
  for(const auto &entry:fs::recursive_directory_iterator(output))if(entry.is_regular_file()) {
    if(!first_file)inventory<<',';
    first_file=false;const auto payload=bytes(entry.path());
    inventory<<"{\"path\":"<<quote(entry.path().lexically_relative(output).generic_string())
             <<",\"bytes\":"<<payload.size()<<",\"sha256\":"<<quote(sha256(payload))<<'}';
  }
  inventory<<']';require(sha256(bytes(card))==card_sha && sha256(bytes(sources))==TWO_COMPONENT_SOURCE_ID,
                         "card/source manifest unchanged after admission");
  std::ostringstream report;report.precision(17);
  report<<"{\"protocol\":\"two-component-information-v1\",\"data_recipe\":\"two-component-v1\",\"status\":"
        <<quote(passed?"passed":"failed")<<",\"source_fingerprint\":"<<quote(TWO_COMPONENT_SOURCE_ID)
        <<",\"card_sha256\":"<<quote(card_sha)<<",\"sdk_proof_sha256\":"<<quote(sdk_sha)
        <<",\"engineering_masters\":[910901,910902],\"cohorts\":"<<cohorts.str()<<",\"files\":"<<inventory.str()
        <<",\"generator_calls\":4,\"information_rows\":2048,\"data_archives\":12,\"decision_archives\":12"
        <<",\"encoder_calls\":0,\"optimizer_updates\":0,\"head_fits\":0,\"PCA_fits\":0,\"TEST_generated\":false"
        <<",\"quality_masters_generated\":false,\"seconds\":"<<std::chrono::duration<double>(Clock::now()-started).count()<<"}\n";
  write_new(output/"report.json",report.str());
  write_new(output/"complete.json","{\"report_sha256\":"+quote(sha256(report.str()))+"}\n");
  std::cout<<"INFORMATION_"<<(passed?"PASSED":"FAILED")<<' '<<output<<std::endl;
}
} // namespace
int main(int argc,char **argv) {
  try {
    torch::set_num_threads(1);torch::NoGradGuard no_grad;
    std::map<std::string,std::string> options;
    require(argc==13,"six exact named arguments required");
    for(int i=1;i<argc;i+=2)require(options.emplace(argv[i],argv[i+1]).second,"unique named arguments");
    require(options.size()==6 && options.count("--output") && options.count("--card") &&
            options.count("--card-sha256") && options.count("--sources") &&
            options.count("--sdk-proof") && options.count("--sdk-sha256"),"closed argument schema");
    run(options.at("--output"),options.at("--card"),options.at("--card-sha256"),
        options.at("--sources"),options.at("--sdk-proof"),options.at("--sdk-sha256"));
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
