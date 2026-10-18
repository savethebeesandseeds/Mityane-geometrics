// SPDX-License-Identifier: MIT
#include "fresh_tempo3_data.h"
#include <ATen/Context.h>
#include <chrono>
#include <iostream>
#include <map>
#include <sys/stat.h>

#ifndef FIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID
#define FIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID "unrecorded"
#endif
namespace {
namespace fresh = fixed_prior_fresh_confirmation;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace ev::frozen_inputs;
using Clock = std::chrono::steady_clock;
constexpr const char *output_parent = "/embedding/output/runs/rpb-fixed-prior-fresh-confirmation";

void direct_directory(const fs::path &path) {
  require(path.is_absolute()&&fs::is_directory(path)&&fs::canonical(path)==path,"canonical directory");
  for(auto p=path;!p.empty()&&p!=p.root_path();p=p.parent_path())require(!fs::is_symlink(p),"direct directory ancestors");
}
void direct_input(const fs::path &path) {
  require(path.is_absolute()&&fs::is_regular_file(path)&&!fs::is_symlink(path)&&fs::canonical(path)==path,"canonical regular card");
  direct_directory(path.parent_path());struct stat s{};
  require(::stat(path.c_str(),&s)==0&&s.st_nlink==1,"single-link card input");
}
void new_leaf(const fs::path &path) {
  require(path.is_absolute()&&path.lexically_normal()==path&&path.filename()!="."&&path.filename()!=".."&&
      !path.filename().empty(),"absolute normalized output leaf");
  direct_directory(path.parent_path());const fs::path bound(output_parent);
  const auto relative=path.lexically_relative(bound);
  require(!relative.empty()&&!relative.is_absolute()&&*relative.begin()!=".."&&path!=bound,"output stays in fresh protocol root");
  require(!fs::exists(path)&&!fs::is_symlink(path)&&fs::create_directory(path),"exclusive absent output leaf");
  direct_directory(path);
}
std::vector<uint64_t> parse_masters(const std::string &value) {
  std::istringstream in(value);std::string part;std::vector<uint64_t> out;
  while(std::getline(in,part,',')) {
    require(!part.empty()&&part.size()<=20&&part.find_first_not_of("0123456789")==std::string::npos,"canonical master integer");
    const auto seed=std::stoull(part);require(std::to_string(seed)==part,"no alternate master spelling");out.push_back(seed);
  }
  require(out==std::vector<uint64_t>(fresh::kMasters.begin(),fresh::kMasters.end()),"exact ordered fresh five masters, no subset selection");
  return out;
}
std::string masters_json() {return "[80787,81888,82989,84090,85191]";}
void save_cpu_archive(const fs::path &path,const ev::ControlledDataset &data,const torch::Tensor &erasure={}) {
  require(!fs::exists(path)&&!fs::is_symlink(path),"new data archive only");direct_directory(path.parent_path());
  torch::serialize::OutputArchive a;
  a.write("observations",data.observed.data,true);a.write("feature_mask",data.observed.feature_mask,true);
  a.write("labels_scoring_only",data.labels,true);
  a.write("source_ids_json",embedding::archive::text_tensor(strings(data.source_ids)),true);
  if(erasure.defined())a.write("requested_erasure",erasure,true);
  const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
  require(fd>=0,"exclusive new data archive");require(::close(fd)==0,"archive reservation close");
  embedding::archive::save_archive(path.string(),a);
}
void admit_written(const std::vector<fs::path> &paths) {
  std::set<fs::path> names;std::set<std::pair<dev_t,ino_t>> nodes;
  for(const auto &path:paths) {
    require(path.is_absolute()&&fs::is_regular_file(path)&&!fs::is_symlink(path)&&fs::canonical(path)==path,"complete regular archive matrix");
    direct_directory(path.parent_path());struct stat s{};
    require(::stat(path.c_str(),&s)==0&&s.st_nlink==1&&names.insert(path).second&&nodes.emplace(s.st_dev,s.st_ino).second,
        "all data roles distinct before hashing");
  }
}
std::string plan() {
  return "{\"protocol\":\"fixed-prior-fresh-confirmation-v1\",\"dataset_id\":\"TEMPO-3\",\"generator_protocol\":\"structured-hard-timing-v1\","
      "\"designed_complexity\":4,\"complexity_maximum\":5,\"masters\":[80787,81888,82989,84090,85191],"
      "\"train_pairs\":128,\"validation_pairs\":64,\"testing_pairs\":0,\"shape\":[3,32,3],\"archives\":15,"
      "\"encoder_inputs\":[\"observations\",\"feature_mask\"],\"scoring_only_labels\":true,"
      "\"extra_validation_deletion_rate\":0.3,\"extra_validation_deletion_stream\":\"0x746d70332d64656c\","
      "\"quality_generator_calls\":5,\"encoder_calls\":0,\"optimizer_updates\":0,\"head_fits\":0,\"TEST_generated\":false}";
}
template<class F>void rejects(F call,int &count) {
  bool rejected=false;try{call();}catch(const std::exception &){rejected=true;}
  require(rejected,"malformed fresh data must reject");++count;
}
void engineering(const fs::path &output) {
  new_leaf(output);const auto start=Clock::now();int rejected=0;
  // These are already declared artificial generator-fixture seeds, not any of
  // the five prospective confirmation masters. No quality scores are computed.
  auto &generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));
  const auto rng_before=generator.get_state().clone();
  const auto first=fresh::make_fresh_tempo3_data(901901),again=fresh::make_fresh_tempo3_data(901901);
  const auto other=fresh::make_fresh_tempo3_data(901902);
  require(fresh::exact_cpu(rng_before,generator.get_state().contiguous()),"generator/deletion preserve CPU Torch RNG");
  for(const auto pair:{std::make_pair(&first.training,&again.training),std::make_pair(&first.validation,&again.validation),
                       std::make_pair(&first.validation_deleted,&again.validation_deleted)}) {
    require(fresh::exact_cpu(pair.first->observed.data,pair.second->observed.data)&&
        fresh::exact_cpu(pair.first->observed.feature_mask,pair.second->observed.feature_mask)&&
        fresh::exact_cpu(pair.first->labels,pair.second->labels)&&pair.first->source_ids==pair.second->source_ids,
        "exact deterministic generator, labels, split and deletion");
  }
  require(fresh::exact_cpu(first.requested_erasure,again.requested_erasure)&&
      !fresh::exact_cpu(first.training.observed.data,other.training.observed.data),"deterministic deletion and distinct generator seed");
  const auto unqualified=ev::make_structured_hard_timing_development(128,64,901901);
  require(fresh::exact_cpu(first.training.observed.data,unqualified.training.observed.data)&&
      fresh::exact_cpu(first.training.labels,unqualified.training.labels)&&
      fresh::exact_cpu(first.validation.observed.data,unqualified.validation.observed.data)&&
      fresh::exact_cpu(first.validation.labels,unqualified.validation.labels),"literal authoritative observations/order unchanged");
  for(size_t row=0;row<first.training.source_ids.size();++row)
    require(first.training.source_ids[row]==std::string(fresh::kProtocol)+"/lag_sign/"+unqualified.training.source_ids[row],"only source qualification changes");
  auto relabeled=first.training;relabeled.labels=1-first.training.labels;
  require(fresh::exact_cpu(fresh::model_observations(first.training).data,fresh::model_observations(relabeled).data)&&
      fresh::exact_cpu(fresh::model_observations(first.training).feature_mask,fresh::model_observations(relabeled).feature_mask),
      "scoring labels excluded from observation-only encoder interface");
  const auto view=ev::make_coordinate_deletion_view(first.validation.observed,first.validation.source_ids,ev::Task::lag_sign,
      ev::stream_seed(901901,ev::kStructuredHardTimingDeletionStream),.30,fresh::kDeletionPurpose);
  require(fresh::exact_cpu(view.requested_erasure,first.requested_erasure)&&
      fresh::exact_cpu(view.observations.data,first.validation_deleted.observed.data),"exact unchanged deletion API/stream/purpose");
  auto invalid=first;invalid.training=first.training;invalid.training.labels=first.training.labels.clone();invalid.training.labels[1]=invalid.training.labels[0];
  rejects([&]{fresh::validate(invalid);},rejected);
  invalid=first;invalid.training.source_ids[0]="wrong-namespace";rejects([&]{fresh::validate(invalid);},rejected);
  invalid=first;invalid.training.source_ids[1]=invalid.training.source_ids[2];rejects([&]{fresh::validate(invalid);},rejected);
  invalid=first;invalid.training.observed.data=first.training.observed.data.to(torch::kFloat32);rejects([&]{fresh::validate(invalid);},rejected);
  invalid=first;invalid.training.observed.feature_mask=first.training.observed.feature_mask.to(torch::kInt64);rejects([&]{fresh::validate(invalid);},rejected);
  invalid=first;invalid.validation.source_ids=first.training.source_ids;rejects([&]{fresh::validate(invalid);},rejected);
  invalid=first;invalid.requested_erasure=first.requested_erasure.to(torch::kInt64);rejects([&]{fresh::validate(invalid);},rejected);
  invalid=first;invalid.validation_deleted.observed.data=first.validation_deleted.observed.data.clone();
  invalid.validation_deleted.observed.data[0][0][0][0]=std::numeric_limits<double>::infinity();rejects([&]{fresh::validate(invalid);},rejected);
  rejects([&]{(void)parse_masters("80787,81888");},rejected);
  require(rejected==9,"all nine artificial malformed data cases");
  // Serialize the exact quality writer envelope, then independently inspect
  // keys/types/values; no encoder, optimizer, fit or synthetic model is used.
  std::vector<fs::path> paths;
  for(const auto view_name:{"training","validation","validation-deleted"}) {
    const auto path=output/(std::string(view_name)+".serialized-archive");paths.push_back(path);
    const auto &data=std::string(view_name)=="training"?first.training:
        std::string(view_name)=="validation"?first.validation:first.validation_deleted;
    const bool deleted=std::string(view_name)=="validation-deleted";
    save_cpu_archive(path,data,deleted?first.requested_erasure:torch::Tensor{});
  }
  admit_written(paths);
  for(size_t i=0;i<paths.size();++i) {
    torch::serialize::InputArchive a;a.load_from(paths[i].string(),torch::kCPU);const auto keys=a.keys();
    std::set<std::string> expected{"observations","feature_mask","labels_scoring_only","source_ids_json"};
    if(i==2)expected.insert("requested_erasure");
    require(std::set<std::string>(keys.begin(),keys.end())==expected,"exact4/5 serialized scoring-only envelope");
    const auto &data=i==0?first.training:i==1?first.validation:first.validation_deleted;
    for(const auto &key:{"observations","feature_mask","labels_scoring_only","source_ids_json"}) {
      torch::Tensor actual;a.read(key,actual,true);
      const auto wanted=std::string(key)=="observations"?data.observed.data:std::string(key)=="feature_mask"?data.observed.feature_mask:
          std::string(key)=="labels_scoring_only"?data.labels:embedding::archive::text_tensor(strings(data.source_ids));
      require(fresh::exact_cpu(actual,wanted),"exact CPU serialized observation/label/source identity");
    }
    if(i==2){torch::Tensor actual;a.read("requested_erasure",actual,true);require(fresh::exact_cpu(actual,first.requested_erasure),"exact serialized requested erasure");}
  }
  std::ostringstream record;record.precision(17);
  record<<"{\"status\":\"PASS\",\"protocol\":"<<quote(fresh::kProtocol)<<",\"source_fingerprint\":"<<quote(FIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID)
      <<",\"engineering_seeds\":[901901,901902],\"quality_masters_generated\":false,\"negative_cases\":"<<rejected
      <<",\"serialized_cpu_fixture_archives\":3,\"generator_calls\":4,\"encoder_calls\":0,\"optimizer_updates\":0,\"head_fits\":0,\"TEST_generated\":false,\"seconds\":"
      <<std::chrono::duration<double>(Clock::now()-start).count()<<"}\n";
  write_new(output/"engineering.json",record.str());std::cout<<"Fresh TEMPO-3 data engineering passed\n"<<record.str();
}
void generate(const fs::path &output,const fs::path &card,const std::string &card_sha,const std::string &master_text) {
  const auto masters=parse_masters(master_text);
  require(card_sha==fresh::kCardSha256&&is_sha(FIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID),"frozen card and recorded compiled source required");
  direct_input(card);const auto card_body=bytes(card);require(sha256(card_body)==card_sha,"exact frozen card bytes before generation");
  new_leaf(output);require(fs::create_directory(output/"results"),"new data results directory");
  std::vector<fs::path> paths;std::set<fs::path> names;
  for(const auto master:masters) {
    const auto dir=output/"results"/("seed-"+std::to_string(master)+"-lag_sign");require(fs::create_directory(dir),"new cohort data directory");
    for(const auto name:{"training","validation","validation-deleted"}) {
      const auto path=dir/(std::string("controlled-")+name+".pt");require(!fs::exists(path)&&names.insert(path).second,"entire absent data role matrix");paths.push_back(path);
    }
  }
  require(paths.size()==15,"exact fifteen fresh data roles before generation");
  const auto start=Clock::now();
  for(size_t i=0;i<masters.size();++i) {
    const auto data=fresh::make_fresh_tempo3_data(masters[i]);
    save_cpu_archive(paths[3*i],data.training);save_cpu_archive(paths[3*i+1],data.validation);
    save_cpu_archive(paths[3*i+2],data.validation_deleted,data.requested_erasure);
  }
  admit_written(paths);std::ostringstream files;files<<'[';
  for(size_t i=0;i<paths.size();++i) {
    const auto body=bytes(paths[i]);if(i)files<<',';
    files<<"{\"path\":"<<quote(paths[i].lexically_relative(output).generic_string())<<",\"bytes\":"<<body.size()<<",\"sha256\":"<<quote(sha256(body))<<'}';
  }
  files<<']';require(sha256(bytes(card))==card_sha,"card unchanged after all generation");
  std::ostringstream report;report.precision(17);
  report<<"{\"protocol\":"<<quote(fresh::kProtocol)<<",\"dataset_id\":\"TEMPO-3\",\"generator_protocol\":"<<quote(ev::kStructuredHardTimingProtocol)
      <<",\"data_recipe\":\"structured-hard-timing-v1\",\"source_fingerprint\":"<<quote(FIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID)
      <<",\"card_sha256\":"<<quote(card_sha)<<",\"human_card_sha256\":"<<quote(card_sha)<<",\"masters\":"<<masters_json()
      <<",\"designed_complexity\":4,\"complexity_maximum\":5,\"train_pairs\":128,\"validation_pairs\":64,\"test_pairs\":0,\"shape\":[3,32,3]"
      <<",\"source_namespace\":\"fixed-prior-fresh-confirmation-v1/lag_sign/structured-hard-timing-v1\",\"source_disjoint\":true,\"randomized_pair_label_order\":true"
      <<",\"deletion_rate\":0.3,\"deletion_stream\":\"0x746d70332d64656c\",\"deletion_purpose\":"<<quote(fresh::kDeletionPurpose)
      <<",\"pair_shared_deletion\":true,\"engineering_separate\":true,\"generator_calls\":5,\"archive_count\":15,\"encoder_calls\":0,\"optimizer_updates\":0,\"head_fits\":0,\"TEST_generated\":false"
      <<",\"CPU_generation_validation_and_archive_IO_seconds\":"<<std::chrono::duration<double>(Clock::now()-start).count()<<",\"files\":"<<files.str()<<"}\n";
  write_new(output/"data-report.json",report.str());
  write_new(output/"data-complete.json","{\"complete\":true,\"protocol\":"+quote(fresh::kProtocol)+",\"masters\":"+masters_json()+
      ",\"source_fingerprint\":"+quote(FIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID)+",\"card_sha256\":"+quote(card_sha)+
      ",\"data_report_sha256\":"+quote(sha256(report.str()))+",\"archive_count\":15,\"encoder_calls\":0,\"head_fits\":0}\n");
  std::cout<<"Fresh TEMPO-3 data generation completed; fifteen CPU archives; no encoder or head execution\n";
}
} // namespace
int main(int argc,char **argv) {
  try {
    if(argc==2&&std::string(argv[1])=="--source-id"){std::cout<<FIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID<<'\n';return 0;}
    if(argc==2&&std::string(argv[1])=="--plan"){std::cout<<plan()<<'\n';return 0;}
    torch::set_num_threads(1);
    if(argc==3&&std::string(argv[1])=="--engineering"){engineering(argv[2]);return 0;}
    std::map<std::string,std::string> args;
    require(argc==9,"closed data writer arguments");
    for(int i=1;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"no duplicate writer argument");
    require(args.size()==4&&args.count("--output")&&args.count("--card")&&args.count("--card-sha256")&&args.count("--masters"),"exact writer option matrix");
    generate(args.at("--output"),args.at("--card"),args.at("--card-sha256"),args.at("--masters"));return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
