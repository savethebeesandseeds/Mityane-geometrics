// SPDX-License-Identifier: MIT
#include "embedding/shared/archive_readout.h"
#include "embedding/shared/paired_pooling.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replication_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

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
namespace fs = std::filesystem;
namespace ev = embedding::evaluation;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
const std::string protocol = "context-lighter-validation-v1";
const fs::path parent = "output/runs/rpb-context-replication/context-replication-JjNEUc";
const std::string parent_source = "9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828";
const std::vector<uint64_t> masters{4404,5505,6606,7707,8808};
constexpr uint64_t validation_stream = 0x636c763164726f70ULL;
const std::string view_id = "validation-dropout-030";
const std::string training_namespace = "native-development-v1/lag_sign";

void require(bool value,const std::string &message) {
  if(!value)throw std::runtime_error("[context lighter validation] "+message);
}
std::string quote(const std::string &value) {
  std::ostringstream out;out << '"';
  for(const unsigned char c:value) {
    if(c=='"' || c=='\\')out << '\\' << c;
    else if(c<32)out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str()+'"';
}
std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out;out << '[';
  for(size_t i=0;i<values.size();++i){if(i){out << ',';}out << quote(values[i]);}
  return out.str()+']';
}
std::string fields(const std::map<std::string,std::string> &values) {
  std::ostringstream out;out << '{';bool first=true;
  for(const auto &[key,value]:values){if(!first){out << ',';}first=false;out << quote(key) << ':' << quote(value);}
  return out.str()+'}';
}
std::string bytes(const fs::path &path) {
  std::ifstream in(path,std::ios::binary);require(bool(in),"cannot read declared file: "+path.string());
  std::ostringstream out;out << in.rdbuf();require(!in.bad(),"declared file read failed: "+path.string());return out.str();
}
bool is_sha(const std::string &value) {
  return value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
// Same dependency-free exact-byte digest used by the shared archive evaluator.
std::string sha256(const std::string &input) {
  static constexpr std::array<uint32_t,64> constants{
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  require(input.size() <= std::numeric_limits<uint64_t>::max()/8, "file too large for SHA-256");
  std::vector<uint8_t> padded(input.begin(),input.end()); padded.push_back(0x80);
  while (padded.size()%64 != 56) padded.push_back(0);
  const uint64_t bits=uint64_t(input.size())*8;
  for (int shift=56;shift>=0;shift-=8) padded.push_back(uint8_t(bits >> shift));
  std::array<uint32_t,8> state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  auto rotate=[](uint32_t value,int count) { return (value>>count)|(value<<(32-count)); };
  for (size_t offset=0;offset<padded.size();offset+=64) {
    std::array<uint32_t,64> schedule{};
    for (size_t i=0;i<16;++i) for (size_t j=0;j<4;++j) schedule[i]=(schedule[i]<<8)|padded[offset+4*i+j];
    for (size_t i=16;i<64;++i) {
      const auto x=schedule[i-15], y=schedule[i-2];
      schedule[i]=schedule[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+schedule[i-7]+(rotate(y,17)^rotate(y,19)^(y>>10));
    }
    auto a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for (size_t i=0;i<64;++i) {
      const auto first=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+constants[i]+schedule[i];
      const auto second=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
      h=g;g=f;f=e;e=d+first;d=c;c=b;b=a;a=first+second;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
  }
  std::ostringstream out; out << std::hex << std::setfill('0');
  for (const auto value:state) out << std::setw(8) << value;
  return out.str();
}
void write_new(const fs::path &path,const std::string &value) {
  const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  require(fd>=0,"output must be new: "+path.string());size_t offset=0;
  while(offset<value.size()) {
    const auto count=::write(fd,value.data()+offset,value.size()-offset);
    if(count<=0){::close(fd);throw std::runtime_error("output write failed");}
    offset+=static_cast<size_t>(count);
  }
  const int synced=::fsync(fd),closed=::close(fd);
  require(synced==0 && closed==0,"output fsync failed");
  const int directory=::open(path.parent_path().c_str(),O_RDONLY|O_DIRECTORY);
  require(directory>=0,"output parent unavailable");
  const int ds=::fsync(directory),dc=::close(directory);
  require(ds==0 && dc==0,"output directory fsync failed");
}
struct RuntimeIsolation {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for(size_t i=0;i<at::getNumGPUs();++i)generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(i))));
    for(const auto &generator:generators)states.push_back(generator.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try{for(size_t i=0;i<generators.size();++i)generators[i].set_state(states[i]);at::set_num_threads(threads);}catch(...){std::terminate();}
  }
};
torch::Tensor tensor(torch::serialize::InputArchive &archive,const std::string &key) {
  torch::Tensor value;archive.read(key,value,true);return value;
}
void same_tensor(const torch::Tensor &left,const torch::Tensor &right,const std::string &why) {
  require(left.defined() && right.defined() && left.device().is_cpu() && right.device().is_cpu() &&
      left.scalar_type()==right.scalar_type() && left.sizes()==right.sizes(),why+" tensor schema");
  const auto a=left.contiguous(),b=right.contiguous();
  require(a.numel()==0 || std::memcmp(a.const_data_ptr(),b.const_data_ptr(),static_cast<size_t>(a.numel()*a.element_size()))==0,why+" exact tensor bytes");
}
std::string cohort_name(uint64_t master){return "seed-"+std::to_string(master)+"-lag_sign";}
std::set<std::string> allowed_roles() {
  std::set<std::string> roles{"reference/native-development-card.json","reference/development-complete.json","reference/validation-report.json"};
  for(const auto master:masters) {
    for(const std::string role:{"reference","results"}) {
      const auto base=role+'/'+cohort_name(master);
      for(const std::string file:{"controlled-training.pt","controlled-validation.pt","development-manifest.json"})roles.insert(base+'/'+file);
      roles.insert(base+(role=="reference"?"/trainer-audit.json":"/candidate-trainer-audit.json"));
      for(const int budget:{0,512}) {
        const auto point=base+(role=="reference"?"/milestone-":"/candidate-milestone-")+std::to_string(budget);
        for(const std::string suffix:{"",".audit.pt",".scaler.pt",".training-raw.pt"})roles.insert(point+"/checkpoint.pt"+suffix);
        for(const std::string file:{"native-training.pt","native-validation.pt","point.json"})roles.insert(point+'/'+file);
        if(role=="reference")for(const std::string split:{"training","validation"})roles.insert(point+'/'+split+"-reconstruction.pt");
        if(budget==512)for(const std::string rep:{"rep-1","rep-2","rep-3"})
          for(const std::string file:{"fit.pt","training-predictions.pt","validation-predictions.pt"})roles.insert(point+'/'+rep+'/'+file);
      }
      if(role=="results")for(const std::string name:{"candidate_initial","candidate"})
        for(const std::string split:{"training","validation"})roles.insert(base+'/'+name+'-'+split+"-reconstruction.pt");
    }
  }
  return roles;
}
std::string parent_roles_json() {
  std::ostringstream out;out << "{\"protocol\":" << quote(protocol) << ",\"parent_capsule\":" << quote(parent.generic_string()) << ",\"masters\":[4404,5505,6606,7707,8808],\"testing_accessed\":false,\"stress_accessed\":false,\"payload_access\":false,\"inputs\":[";
  bool first=true;
  for(const auto &path:allowed_roles()) {
    const bool baseline=path.rfind("reference/",0)==0;std::string role="development-metadata";
    if(path.find("checkpoint.pt.training-raw.pt")!=std::string::npos)role="checkpoint-TRAIN-observations";
    else if(path.find("checkpoint.pt.audit.pt")!=std::string::npos)role="TRAIN-producer-audit";
    else if(path.find("checkpoint.pt.scaler.pt")!=std::string::npos)role="TRAIN-fitted-model-scaler";
    else if(path.find("checkpoint.pt")!=std::string::npos)role="checkpoint";
    else if(path.find("reconstruction.pt")!=std::string::npos)role=path.find("training-reconstruction")!=std::string::npos?"TRAIN-fixed-query":"VALIDATION-fixed-query";
    else if(path.find("predictions.pt")!=std::string::npos)role=path.find("training-predictions")!=std::string::npos?"TRAIN-head-predictions":"VALIDATION-head-predictions";
    else if(path.find("/fit.pt")!=std::string::npos)role="TRAIN-fitted-native-heads";
    else if(path.find("native-training.pt")!=std::string::npos)role="TRAIN-native-export";
    else if(path.find("native-validation.pt")!=std::string::npos)role="VALIDATION-native-export";
    else if(path.find("controlled-training.pt")!=std::string::npos)role="TRAIN-observations-labels-source-order";
    else if(path.find("controlled-validation.pt")!=std::string::npos)role="VALIDATION-observations-labels-source-order";
    if(!first){out << ',';}first=false;
    out << "{\"path\":" << quote(path) << ",\"role\":" << quote(role) << ",\"tag\":" << quote(baseline?"RPB-v4":"RPB-v6") << ",\"master_seed\":";
    bool found=false;for(const auto master:masters)if(path.find('/'+cohort_name(master)+'/')!=std::string::npos){out << master;found=true;break;}if(!found){out << "null";}
    out << ",\"budget\":";
    if(path.find("milestone-0/")!=std::string::npos || path.find("candidate_initial-")!=std::string::npos){out << 0;}
    else if(path.find("milestone-512/")!=std::string::npos || path.find("/candidate-training-reconstruction")!=std::string::npos || path.find("/candidate-validation-reconstruction")!=std::string::npos){out << 512;}
    else {out << "null";}out << '}';
  }
  return out.str()+"]}";
}
struct InputGuard {
  std::map<std::string,std::string> expected,original;
  explicit InputGuard(const fs::path &manifest) {
    const auto allowed=allowed_roles();std::ifstream in(manifest);require(bool(in),"frozen retained manifest required");
    std::string line;
    while(std::getline(in,line)) {
      if(!line.empty() && line.back()=='\r')line.pop_back();
      require(line.size()>66 && is_sha(line.substr(0,64)) && line.substr(64,2)=="  ","malformed retained SHA record");
      const auto relative=line.substr(66);
      require(allowed.count(relative),"retained role is not an explicitly permitted TRAIN/VALIDATION role: "+relative);
      const auto path=fs::absolute(parent/fs::path(relative)).lexically_normal();
      require(fs::is_regular_file(path) && fs::canonical(path)==path,"missing or redirected parent role: "+relative);
      require(expected.emplace(path.string(),line.substr(0,64)).second,"duplicate retained role");
      const auto content=bytes(path);require(sha256(content)==line.substr(0,64),"retained SHA mismatch: "+relative);
      original.emplace(path.string(),content);
    }
    require(in.eof() && expected.size()==allowed.size(),"manifest must contain every explicit permitted role exactly once");
  }
  std::string bind(const std::string &relative) const {
    const auto path=fs::absolute(parent/fs::path(relative)).lexically_normal().string();require(expected.count(path),"unbound retained role");return path;
  }
  void verify() const {for(const auto &[path,value]:original)require(bytes(path)==value,"retained input bytes changed: "+path);}
  std::string json() const {
    std::ostringstream out;out << "{\"checksum_algorithm\":\"sha256-file-bytes\",\"exact_byte_preservation\":true,\"files\":[";bool first=true;
    for(const auto &[path,digest]:expected){if(!first){out << ',';}first=false;out << "{\"path\":" << quote(path) << ",\"sha256\":" << quote(digest) << ",\"bytes\":" << original.at(path).size() << '}';}
    return out.str()+"]}";
  }
};
void same_split(const ev::ControlledDataset &left,const ev::ControlledDataset &right,const std::string &why) {
  same_tensor(left.observed.data,right.observed.data,why+" observations");
  same_tensor(left.observed.feature_mask,right.observed.feature_mask,why+" support");same_tensor(left.labels,right.labels,why+" labels");
  require(left.source_ids==right.source_ids,why+" exact source order");
}
ev::ProviderFitInput metadata(const ev::ControlledDataset &training,uint64_t seed,const ev::NativeCurveRun &recipe) {
  const auto &o=training.observed;embedding::Batch legal{torch::where(o.feature_mask,o.data,torch::zeros_like(o.data)).detach().clone(),o.feature_mask.clone()};
  return {legal,recipe.card.shape,seed,training.source_ids,recipe.card.channel_ids,recipe.card.feature_units,training_namespace,recipe.card.sampling_interval,31*recipe.card.sampling_interval};
}
void same_fit_input(const ev::ProviderFitInput &fit,const ev::ProviderFitInput &expected) {
  same_tensor(fit.training_observations.data,expected.training_observations.data,"factory permitted TRAIN");
  same_tensor(fit.training_observations.feature_mask,expected.training_observations.feature_mask,"factory permitted TRAIN mask");
  require(fit.training_source_ids==expected.training_source_ids && fit.seed==expected.seed && fit.channel_ids==expected.channel_ids &&
      fit.protocol_id==expected.protocol_id && fit.feature_units==expected.feature_units && fit.endpoint==expected.endpoint &&
      fit.sampling_interval==expected.sampling_interval && fit.shape.channel_count==3 && fit.shape.history_length==32 && fit.shape.input_width==3 &&
      fit.shape.dtype==torch::kFloat64 && fit.shape.device.is_cpu(),"factory TRAIN/schema/namespace differs");
}
ev::RetainedPoolingCohort reference(uint64_t master,const InputGuard &guard) {
  ev::RetainedPoolingCohort out;out.master_seed=master;const auto base="reference/"+cohort_name(master);
  out.training_observations=guard.bind(base+"/controlled-training.pt");out.validation_observations=guard.bind(base+"/controlled-validation.pt");
  out.reference_checkpoint=guard.bind(base+"/milestone-512/checkpoint.pt");out.reference_initial_checkpoint=guard.bind(base+"/milestone-0/checkpoint.pt");
  return out;
}
void feature_parity(const ev::FeatureSurface &value,const fs::path &saved) {
  torch::serialize::InputArchive archive;archive.load_from(saved.string(),torch::kCPU);
  same_tensor(value.values,tensor(archive,"features"),"retained feature values");same_tensor(value.valid,tensor(archive,"valid"),"retained feature support");
}
void archive_parity(const fs::path &left,const fs::path &right,const std::vector<std::string> &keys) {
  torch::serialize::InputArchive a,b;a.load_from(left.string(),torch::kCPU);b.load_from(right.string(),torch::kCPU);
  for(const auto &key:keys)same_tensor(tensor(a,key),tensor(b,key),"archive parity "+key);
}
void query_parity(const fs::path &left,const fs::path &right) {
  torch::serialize::InputArchive a,b;a.load_from(left.string(),torch::kCPU);b.load_from(right.string(),torch::kCPU);
  require(a.keys()==b.keys(),"fixed-query reconstruction schema differs");
  for(const auto &key:a.keys())same_tensor(tensor(a,key),tensor(b,key),"fixed-query parity "+key);
}
void numeric_archive_parity(const fs::path &left,const fs::path &right) {
  torch::serialize::InputArchive a,b;a.load_from(left.string(),torch::kCPU);b.load_from(right.string(),torch::kCPU);
  const auto ak=a.keys(),bk=b.keys();
  require(std::set<std::string>(ak.begin(),ak.end())==std::set<std::string>(bk.begin(),bk.end()),"raw/PCA archive schema differs across versions");
  for(const auto &key:ak)same_tensor(tensor(a,key),tensor(b,key),"raw/PCA across-version parity "+key);
}
const std::vector<std::string> prediction_keys{"ridge","tiny_secondary","valid","labels_scoring_only","source_ids_json"};
const std::vector<std::string> fit_keys{"feature_mean","feature_scale","fitted_rows","ridge_mean","ridge_scale","ridge_weights","ridge_intercept",
    "tiny_mean","tiny_scale","tiny_w1","tiny_b1","tiny_w2","tiny_b2","actual_probe_seed_decimal"};
void fit_parity(const fs::path &left,const fs::path &right) {
  torch::serialize::InputArchive a,b;a.load_from(left.string(),torch::kCPU);b.load_from(right.string(),torch::kCPU);
  const std::set<std::string> expected(fit_keys.begin(),fit_keys.end());
  const auto ak=a.keys(),bk=b.keys();require(std::set<std::string>(ak.begin(),ak.end())==expected && std::set<std::string>(bk.begin(),bk.end())==expected,"native fit schema changed or contains PCA");
  for(const auto &key:fit_keys)same_tensor(tensor(a,key),tensor(b,key),"ordinary TRAIN fit parity "+key);
}
void save_view(const fs::path &path,const ev::ControlledDataset &ordinary,const ev::CoordinateDeletionView &view,uint64_t seed) {
  require(!fs::exists(path),"view destination exists");torch::serialize::OutputArchive out;
  out.write("observed",view.observations.data,true);out.write("feature_mask",view.observations.feature_mask,true);
  out.write("labels_scoring_only",ordinary.labels,true);out.write("source_ids_json",embedding::archive::text_tensor(strings(ordinary.source_ids)),true);
  out.write("requested_erasure",view.requested_erasure,true);out.write("base_observed",ordinary.observed.data,true);out.write("base_feature_mask",ordinary.observed.feature_mask,true);
  out.write("rng_namespace",embedding::archive::text_tensor(protocol),true);out.write("view_id",embedding::archive::text_tensor(view_id),true);
  out.write("actual_corruption_seed",embedding::archive::text_tensor(std::to_string(seed)),true);out.write("corruption_stream",embedding::archive::text_tensor(std::to_string(validation_stream)),true);
  out.write("rate",embedding::archive::text_tensor("0.30"),true);out.write("artifact_kind",embedding::archive::text_tensor("validation_coordinate_deletion_v1"),true);
  const auto oracle=ev::raw_oracle(ev::Task::lag_sign,view.observations);out.write("raw_oracle_predictions",oracle.predictions,true);out.write("raw_oracle_valid",oracle.valid,true);
  out.save_to(path.string());
}
std::string score_json(const ev::Score &s) {
  std::ostringstream out;out << std::setprecision(17) << "{\"total\":" << s.total << ",\"valid\":" << s.valid << ",\"correct\":" << s.correct
      << ",\"coverage\":" << s.coverage << ",\"full_population_correctness\":" << (s.total?double(s.correct)/s.total:0) << ",\"accuracy\":";
  if(s.supported){out << s.accuracy;}else {out << "null";}return out.str()+'}';
}
std::string interval_json(const ev::GroupedInterval &value) {
  std::ostringstream out;out << std::setprecision(17) << "{\"supported\":" << (value.supported?"true":"false") << ",\"source_groups\":" << value.source_groups << ",\"replicates\":1000,\"confidence\":0.95,\"estimate\":" << value.estimate << ",\"lower\":";
  if(value.supported){out << value.lower;}else {out << "null";}out << ",\"upper\":";if(value.supported){out << value.upper;}else {out << "null";}return out.str()+'}';
}
struct Prediction {torch::Tensor ridge,tiny,valid;};
Prediction load_prediction(const fs::path &path,const ev::ControlledDataset &split) {
  torch::serialize::InputArchive in;in.load_from(path.string(),torch::kCPU);Prediction p{tensor(in,"ridge"),tensor(in,"tiny_secondary"),tensor(in,"valid")};
  same_tensor(tensor(in,"labels_scoring_only"),split.labels,"prediction labels");same_tensor(tensor(in,"source_ids_json"),embedding::archive::text_tensor(strings(split.source_ids)),"prediction source order");
  require(p.valid.scalar_type()==torch::kBool && p.valid.sizes()==split.labels.sizes(),"prediction support schema");
  for(const auto &value:{p.ridge,p.tiny})require(value.scalar_type()==torch::kInt64 && value.sizes()==split.labels.sizes() && value.ge(0).logical_and(value.le(1)).all().item<bool>(),"prediction class/schema");
  return p;
}
std::string paired_json(const Prediction &candidate,const Prediction &comparator,const ev::ControlledDataset &split,uint64_t seed) {
  const auto common=candidate.valid.logical_and(comparator.valid);const auto all=torch::ones_like(common),truth=torch::ones_like(split.labels);
  std::ostringstream out;out << "{\"total_rows\":" << split.labels.numel() << ",\"common_valid_rows\":" << common.sum().item<int64_t>() << ",\"probes\":{";
  for(int i=0;i<2;++i) {
    if(i){out << ',';}const auto &a=i?candidate.tiny:candidate.ridge;const auto &b=i?comparator.tiny:comparator.ridge;
    const auto ac=a.eq(split.labels).logical_and(candidate.valid).to(torch::kInt64),bc=b.eq(split.labels).logical_and(comparator.valid).to(torch::kInt64);
    const auto actual=ev::stream_seed(seed,i?0x74696e79ULL:0x7269646765ULL);
    out << quote(i?"tiny_secondary":"ridge") << ":{\"bootstrap_seed_decimal\":" << quote(std::to_string(actual))
        << ",\"candidate\":" << score_json(ev::score(a,split.labels,candidate.valid)) << ",\"comparator\":" << score_json(ev::score(b,split.labels,comparator.valid))
        << ",\"candidate_on_common\":" << score_json(ev::score(a,split.labels,common)) << ",\"comparator_on_common\":" << score_json(ev::score(b,split.labels,common))
        << ",\"conditional_candidate_minus_comparator\":" << interval_json(ev::grouped_accuracy_interval(a,split.labels,common,split.source_ids,actual,1000,b))
        << ",\"full_population_candidate_minus_comparator\":" << interval_json(ev::grouped_accuracy_interval(ac,truth,all,split.source_ids,ev::stream_seed(actual,1),1000,bc)) << '}';
  }
  return out.str()+"}}";
}
struct Cohort {ev::ControlledDataset training,validation;ev::RetainedPoolingCohort reference;};
} // namespace

int main(int argc,char **argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--source-id"){std::cout << EVALUATION_SOURCE_ID << '\n';return 0;}
    if(argc==2 && std::string(argv[1])=="--parent-roles"){std::cout << parent_roles_json() << '\n';return 0;}
    if(argc==1 || (argc==2 && std::string(argv[1])=="--help")) {
      std::cout << "embedding_context_lighter_validation --phase gate|measure --output NEW_DIRECTORY\n"
          "Measure also requires --retained-hashes FROZEN_MANIFEST --admission-log LOG --admission-log-sha256 SHA256\n"
          "Fixed RPB-v7 .15, five known TRAIN/VALIDATION cohorts, point0/512, native32; no TEST or stress access.\n";return argc==1?1:0;
    }
    std::map<std::string,std::string> options;
    for(int i=1;i<argc;i+=2) {
      const std::string name=argv[i];require(i+1<argc && (name=="--phase" || name=="--output" || name=="--retained-hashes" || name=="--admission-log" || name=="--admission-log-sha256") && options.emplace(name,argv[i+1]).second,"unknown/duplicate option or missing value");
    }
    require(options.count("--phase") && options.count("--output") && !options.at("--output").empty(),"phase and new output required");
    const bool measure=options.at("--phase")=="measure";require(measure || options.at("--phase")=="gate","unknown phase");
    require(options.size()==(measure?5u:2u) && (!measure || (options.count("--retained-hashes") && options.count("--admission-log") && options.count("--admission-log-sha256"))),"measure requires frozen roles and preserved admission; gate accepts only phase/output");
    require(is_sha(EVALUATION_SOURCE_ID) && torch::cuda::is_available(),"compiled source identity and actual CUDA required");
    require(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","private SHA256 regression");
    const RuntimeIsolation runtime;torch::set_num_threads(1);
    const auto output=fs::absolute(options.at("--output")).lexically_normal();require(!fs::exists(output),"output already exists");fs::create_directories(output.parent_path());
    auto settings=rpb::read_settings("code/encoders/raw_patch_bottleneck_mae/config/learned_global.conf");settings.model.device=torch::kCUDA;settings.steps=512;rpb::validate_settings(settings);
    require(settings.model.channel_count==3 && settings.model.history_length==32 && settings.model.input_width==3 && settings.model.patch_length==8 && settings.model.encoder_width==64 && settings.model.export_width==32 && settings.model.channel_mixer_layers==1 && settings.model.global_bottleneck_mode==2 && settings.batch_size==8 && settings.model.mask_ratio==.25 && settings.model.huber_delta==1 && settings.model.dropout==0 && settings.learning_rate==.001 && settings.weight_decay==.0001 && settings.gradient_clip_norm==1,"resolved frozen model/training recipe differs");
    if(!measure) {
      write_new(output.string()+".launch-plan.json","{\"protocol\":"+quote(protocol)+",\"phase\":\"gate\",\"test_access\":false,\"source_fingerprint\":"+quote(EVALUATION_SOURCE_ID)+"}\n");
      rpb::run_native_curve_cuda_gate(settings,output.string());std::cout << "Native32 CUDA serving gate passed; the measure phase additionally requires coordinated coordinate15 training admission.\n";return 0;
    }
    const auto admission_path=fs::absolute(options.at("--admission-log")).lexically_normal();require(is_sha(options.at("--admission-log-sha256")) && fs::is_regular_file(admission_path) && sha256(bytes(admission_path))==options.at("--admission-log-sha256"),"preserved admission log hash differs");
    InputGuard guard(options.at("--retained-hashes"));
    std::ostringstream plan;plan << "{\"protocol\":" << quote(protocol) << ",\"policy_version\":\"1.2\",\"stage\":\"development\",\"model_tag\":\"RPB-v7\",\"phase\":\"measure\",\"masters\":[4404,5505,6606,7707,8808],\"milestones\":[0,512],\"source_pairs\":[128,64],\"batch_size\":8,\"parameters_each\":225805,\"native_size\":32,\"post_encoder_pca\":false,\"training_policy_id\":\"rpb-training-context-deletion-015-v1\",\"training_context_rate\":0.15,\"training_namespace\":" << quote(training_namespace)
        << ",\"validation_view\":{\"id\":" << quote(view_id) << ",\"rate\":0.30,\"stream\":" << validation_stream << ",\"rng_namespace\":" << quote(protocol) << ",\"seed_policy\":\"stream_seed(master,stream)\",\"pair_shared\":true,\"repair\":false},\"test_access\":false,\"stress_access\":false,\"selection\":false,\"promotion\":false,\"parent_capsule\":" << quote(fs::absolute(parent).string()) << ",\"parent_source_fingerprint\":" << quote(parent_source)
        << ",\"heads\":{\"ridge_penalty\":1,\"tiny_hidden\":16,\"tiny_steps\":100,\"tiny_learning_rate\":0.01,\"base_seeds\":[2701,2802,2903]},\"readout_policy\":\"new deterministic ordinary TRAIN refits; exact old/cache native fit and intact prediction parity; same TRAIN fits score added VALIDATION view\",\"source_fingerprint\":" << quote(EVALUATION_SOURCE_ID) << ",\"git_head\":" << quote(EVALUATION_GIT_HEAD) << ",\"git_dirty\":" << quote(EVALUATION_GIT_DIRTY) << ",\"base_settings\":" << quote(rpb::settings_text(settings)) << ",\"admission_log\":" << quote(admission_path.string()) << ",\"admission_log_sha256\":" << quote(options.at("--admission-log-sha256")) << ",\"retained_hash_manifest\":" << quote(fs::absolute(options.at("--retained-hashes")).string()) << "}\n";
    write_new(output.string()+".launch-plan.json",plan.str());require(fs::create_directory(output),"cannot claim exclusive output root");write_new(output/"context-lighter-validation-card.json",plan.str());write_new(output/"input-manifest.json",guard.json()+"\n");
    ev::NativeCurveRun recipe;recipe.output_directory=(output/"candidate-development").string();recipe.source_fingerprint=EVALUATION_SOURCE_ID;recipe.git_head=EVALUATION_GIT_HEAD;recipe.git_dirty=EVALUATION_GIT_DIRTY;recipe.model_tag="RPB-v7";recipe.card.id=protocol;recipe.card.seeds=masters;recipe.card.tasks={ev::Task::lag_sign};recipe.card.train_pairs=128;recipe.card.validation_pairs=64;recipe.card.test_pairs=0;recipe.card.comparisons.clear();recipe.milestones={0,512};recipe.development_only=true;recipe.stress_sweep=false;
    std::map<uint64_t,Cohort> cohorts;std::set<std::string> all_sources;
    std::ostringstream preparation;preparation << "{\"protocol\":" << quote(protocol) << ",\"before_any_candidate_fit\":true,\"cohorts\":[";bool first=true;
    for(const auto master:masters) {
      const RuntimeIsolation isolation;Cohort cohort;cohort.reference=reference(master,guard);cohort.training=ev::load_native_development_observations(cohort.reference.training_observations);cohort.validation=ev::load_native_development_observations(cohort.reference.validation_observations);
      const auto generated=ev::make_controlled_development_protocol(ev::Task::lag_sign,recipe.card.shape,128,64,master,.1);
      same_split(generated.training,cohort.training,"pregenerated original TRAIN");same_split(generated.validation,cohort.validation,"pregenerated known VALIDATION");require(!generated.testing.observed.data.defined() && generated.testing.source_ids.empty(),"development generator created testing observations");
      const auto base="results/"+cohort_name(master);same_split(cohort.training,ev::load_native_development_observations(guard.bind(base+"/controlled-training.pt")),"v4/v6 TRAIN");same_split(cohort.validation,ev::load_native_development_observations(guard.bind(base+"/controlled-validation.pt")),"v4/v6 VALIDATION");
      for(const auto *split:{&cohort.training,&cohort.validation})for(const auto &id:std::set<std::string>(split->source_ids.begin(),split->source_ids.end()))require(all_sources.insert(id).second,"development source roles/master overlap");
      const auto audit=rpb::audit_context_replication_initialization(guard.bind(base+"/candidate-milestone-0/checkpoint.pt"),cohort.reference,metadata(cohort.training,master,recipe),512);
      if(!first){preparation << ',';}first=false;preparation << "{\"master_seed\":" << master << ",\"pregenerated_observations_masks_labels_source_order_exact\":true,\"reference_v6_point0_audit\":" << fields(audit) << '}';cohorts.emplace(master,std::move(cohort));
    }
    preparation << "]}\n";write_new(output/"pregeneration-parity.json",preparation.str());guard.verify();
    const auto actual_factory=rpb::make_learning_curve_trainer(settings,rpb::ContextDeletionOptions{true,rpb::ContextDeletionRecipe::coordinate15_v1});
    auto gated=[&](const ev::ProviderFitInput &fit) {
      require(cohorts.count(fit.seed),"undeclared factory master");const auto &cohort=cohorts.at(fit.seed);same_fit_input(fit,metadata(cohort.training,fit.seed,recipe));
      auto trainer=actual_factory(fit);struct State{bool admitted{false};int64_t completed{0};};const auto state=std::make_shared<State>();
      const auto train=trainer.train_to;const auto save=trainer.save_checkpoint;const auto original_fit=fit;
      trainer.train_to=[train,state](int64_t budget){require(budget==0 || budget==512,"undeclared absolute budget");if(budget)require(state->admitted && state->completed==0,"point0 initialization admission must precede training");const auto progress=train(budget);require(progress.attempted==budget && progress.completed==budget && progress.sampled_rows==budget*8 && progress.parameter_count==225805 && progress.cuda_parameter_count==225805 && progress.training_device=="cuda","unmatched/skipped/non-CUDA trajectory");if(budget)require(progress.last_input_cuda && progress.last_loss_cuda && progress.finite_gradients && progress.weights_changed,"actual CUDA update evidence missing");state->completed=budget;return progress;};
      trainer.save_checkpoint=[save,state,original_fit,retained=cohort.reference](const std::string &path){save(path);if(state->completed==0){const RuntimeIsolation isolation;const auto audit=rpb::audit_context_replication_initialization(path,retained,original_fit,rpb::ContextDeletionRecipe::coordinate15_v1,512);write_new(fs::path(path).parent_path()/"initialization-audit.json",fields(audit)+"\n");state->admitted=true;}};
      return trainer;
    };
    ev::run_native_curve(recipe,{"RPB-v7",rpb::settings_text(settings)+"\ntraining_policy_id=rpb-training-context-deletion-015-v1\ncontext_coordinate_deletion_rate=0.15\n",gated});guard.verify();
    require(fs::is_regular_file(fs::path(recipe.output_directory)/"development-complete.json"),"candidate all-point development completion missing");
    require(fs::create_directory(output/"exports") && fs::create_directory(output/"validation-views"),"export/view directory exists");
    ev::ArchiveReadoutRun readouts;readouts.output_directory=(output/"readouts").string();readouts.source_fingerprint=EVALUATION_SOURCE_ID;readouts.git_head=EVALUATION_GIT_HEAD;readouts.git_dirty=EVALUATION_GIT_DIRTY;
    std::map<std::string,fs::path> original_points;std::ostringstream points;points << "{\"protocol\":" << quote(protocol) << ",\"test_access\":false,\"stress_access\":false,\"selection\":false,\"promotion\":false,\"positive_inputs\":15,\"points\":[";first=true;
    for(const auto master:masters) {
      const auto &cohort=cohorts.at(master);const RuntimeIsolation isolation;const auto fit=metadata(cohort.training,master,recipe);
      const auto view_root=output/"validation-views"/cohort_name(master);require(fs::create_directory(view_root),"view cohort exists");const auto seed=ev::stream_seed(master,validation_stream);
      const auto deletion=ev::make_coordinate_deletion_view(cohort.validation.observed,cohort.validation.source_ids,ev::Task::lag_sign,seed,.30,protocol);auto deleted=cohort.validation;deleted.observed=deletion.observations;deleted.clean=deletion.observations;
      const auto view_path=view_root/(view_id+".pt");save_view(view_path,cohort.validation,deletion,seed);
      write_new(view_root/(view_id+".json"),"{\"view_id\":"+quote(view_id)+",\"master_seed\":"+std::to_string(master)+",\"actual_seed\":"+quote(std::to_string(seed))+",\"requested_rate\":0.30,\"rng_namespace\":"+quote(protocol)+",\"pair_shared\":true,\"repair\":false,\"requested_erasure\":"+std::to_string(deletion.requested_erasure.sum().item<int64_t>())+",\"remaining_observed\":"+std::to_string(deletion.observations.feature_mask.sum().item<int64_t>())+"}\n");
      const auto candidate0=fs::path(recipe.output_directory)/cohort_name(master)/"milestone-0";
      const auto initial=rpb::make_retained_curve_snapshot((candidate0/"checkpoint.pt").string(),fit);
      const auto initial_train=ev::extract_native_global(initial,cohort.training.observed,recipe),initial_val=ev::extract_native_global(initial,cohort.validation.observed,recipe);
      feature_parity(initial_train,candidate0/"native-training.pt");feature_parity(initial_val,candidate0/"native-validation.pt");
      for(const std::string role:{"reference","results"}) {
        const auto prefix=role+'/'+cohort_name(master)+(role=="reference"?"/milestone-0":"/candidate-milestone-0");feature_parity(initial_train,guard.bind(prefix+"/native-training.pt"));feature_parity(initial_val,guard.bind(prefix+"/native-validation.pt"));
      }
      for(const std::string tag:{"RPB-v4","RPB-v6","RPB-v7"}) {
        const auto id="seed-"+std::to_string(master)+'-'+tag+"-updates-512";const auto directory=output/"exports"/id;require(fs::create_directory(directory),"export input exists");
        fs::path point,checkpoint,old_train_query,old_val_query;std::string train_path,val_path;
        if(tag=="RPB-v7"){point=fs::path(recipe.output_directory)/cohort_name(master)/"milestone-512";checkpoint=point/"checkpoint.pt";train_path=(point.parent_path()/"controlled-training.pt").string();val_path=(point.parent_path()/"controlled-validation.pt").string();old_train_query=point/"training-reconstruction.pt";old_val_query=point/"validation-reconstruction.pt";}
        else {const auto role=tag=="RPB-v4"?"reference":"results";const auto base=std::string(role)+'/'+cohort_name(master);const auto relative=base+(tag=="RPB-v4"?"/milestone-512":"/candidate-milestone-512");point=fs::path(guard.bind(relative+"/checkpoint.pt")).parent_path();checkpoint=point/"checkpoint.pt";train_path=guard.bind(base+"/controlled-training.pt");val_path=guard.bind(base+"/controlled-validation.pt");old_train_query=guard.bind(tag=="RPB-v4"?relative+"/training-reconstruction.pt":base+"/candidate-training-reconstruction.pt");old_val_query=guard.bind(tag=="RPB-v4"?relative+"/validation-reconstruction.pt":base+"/candidate-validation-reconstruction.pt");}
        original_points.emplace(id,point);const auto snapshot=rpb::make_retained_curve_snapshot(checkpoint.string(),fit);
        const auto training=ev::extract_native_global(snapshot,cohort.training.observed,recipe),validation=ev::extract_native_global(snapshot,cohort.validation.observed,recipe),corrupted=ev::extract_native_global(snapshot,deleted.observed,recipe);
        require(training.provenance==validation.provenance && training.provenance==corrupted.provenance,"snapshot export lineage differs across views");feature_parity(training,point/"native-training.pt");feature_parity(validation,point/"native-validation.pt");
        const auto tf=directory/"native-training.pt",vf=directory/"native-validation.pt",df=directory/"native-validation-dropout-030.pt";
        ev::save_native_feature_archive(tf.string(),training,cohort.training,recipe);ev::save_native_feature_archive(vf.string(),validation,cohort.validation,recipe);ev::save_native_feature_archive(df.string(),corrupted,deleted,recipe);
        const auto tr=directory/"training-reconstruction.pt",vr=directory/"validation-reconstruction.pt";const auto te=ev::write_native_patch_reconstruction(tr.string(),cohort.training,snapshot,recipe),ve=ev::write_native_patch_reconstruction(vr.string(),cohort.validation,snapshot,recipe);query_parity(tr,old_train_query);query_parity(vr,old_val_query);
        write_new(directory/"provider-audit.json","{\"checkpoint\":"+quote(checkpoint.string())+",\"checkpoint_sha256\":"+quote(sha256(bytes(checkpoint)))+",\"snapshot_provenance\":"+quote(training.provenance)+",\"audit_fields\":"+fields(snapshot.features.audit_fields)+"}\n");
        ev::ArchiveReadoutInput input;input.id=id;input.tag=tag;input.task="lag_sign";input.master_seed=master;input.checkpoint_steps=512;input.producer_source_fingerprint=EVALUATION_SOURCE_ID;input.cohort_provenance=protocol+"; known JjNEUc TRAIN/VALIDATION; exact snapshot exports; checkpoint="+checkpoint.string()+"; parent-source="+parent_source;
        input.training_observations=train_path;input.validation_observations=val_path;input.training_features=tf.string();input.validation_features=vf.string();input.training_observations_sha256=sha256(bytes(train_path));input.validation_observations_sha256=sha256(bytes(val_path));input.training_features_sha256=sha256(bytes(tf));input.validation_features_sha256=sha256(bytes(vf));input.expected_feature_provenance=training.provenance;
        ev::ArchiveReadoutValidationView view;view.id=view_id;view.observations=view_path.string();view.features=df.string();view.observations_sha256=sha256(bytes(view_path));view.features_sha256=sha256(bytes(df));view.expected_feature_provenance=training.provenance;view.corruption_provenance=protocol+"; .30; stream="+std::to_string(validation_stream)+"; seed="+std::to_string(seed)+"; pair-shared; no repair";input.validation_views.push_back(view);readouts.inputs.push_back(input);
        if(!first){points << ',';}first=false;points << "{\"master_seed\":" << master << ",\"model_tag\":" << quote(tag) << ",\"updates\":512,\"input_id\":" << quote(id) << ",\"original_point\":" << quote(point.string()) << ",\"export_directory\":" << quote(directory.string()) << ",\"training_reconstruction\":" << te << ",\"validation_reconstruction\":" << ve << '}';
      }
      std::cout << "Lighter validation master=" << master << " frozen positive exports and fixed shared VALIDATION view saved.\n" << std::flush;
    }
    points << "]}\n";write_new(output/"report.json",points.str());guard.verify();ev::run_archive_readout(readouts);
    std::map<std::string,int64_t> fit_counts{{"native",0},{"raw",0},{"pca_only",0}};
    std::ostringstream refits;refits << "{\"protocol\":" << quote(protocol) << ",\"deterministic_refits_disclosed\":true,\"declared_fit_repetitions_per_method\":45,\"same_fit_for_both_validation_views\":true,\"native_fits_contain_no_PCA\":true,\"inputs\":[";first=true;
    for(const auto &input:readouts.inputs) {
      const RuntimeIsolation isolation;const auto point=original_points.at(input.id);
      for(const auto &rep:readouts.repetitions) {
        const auto root=fs::path(readouts.output_directory)/input.id/rep.id;fit_parity(root/"native-fit.pt",point/rep.id/"fit.pt");archive_parity(root/"native-training-predictions.pt",point/rep.id/"training-predictions.pt",prediction_keys);archive_parity(root/"native-validation-predictions.pt",point/rep.id/"validation-predictions.pt",prediction_keys);
        for(auto &[method,count]:fit_counts)if(fs::is_regular_file(root/(method+"-fit.pt"))){++count;}
        if(!first){refits << ',';}first=false;refits << "{\"input_id\":" << quote(input.id) << ",\"repetition\":" << quote(rep.id) << ",\"original_fit\":" << quote((point/rep.id/"fit.pt").string()) << ",\"new_fit\":" << quote((root/"native-fit.pt").string()) << ",\"fit_tensors_exact\":true,\"ordinary_training_predictions_exact\":true,\"ordinary_validation_predictions_exact\":true}";
      }
    }
    require(fit_counts.at("native")==45,"native fit matrix incomplete");
    std::ostringstream counts;counts << "{\"native\":" << fit_counts.at("native") << ",\"raw\":" << fit_counts.at("raw") << ",\"pca_only\":" << fit_counts.at("pca_only") << '}';
    write_new(output/"refit-parity.json",refits.str()+"],\"ordinary_TRAIN_fit_counts_by_method\":"+counts.str()+",\"unsupported_methods_retained_in_archive_report\":true}\n");
    std::ostringstream controls;controls << "{\"protocol\":" << quote(protocol) << ",\"scope\":\"raw/PCA numeric transforms, TRAIN fits and intact/deleted VALIDATION predictions equal across identical per-master inputs; no old raw/PCA fit parity claim\",\"comparisons\":[";first=true;
    int64_t map_comparisons=0,control_fit_comparisons=0,unsupported_controls=0;
    for(const auto master:masters) {
      const RuntimeIsolation isolation;
      const auto root_for=[&](const std::string &tag){return fs::path(readouts.output_directory)/("seed-"+std::to_string(master)+'-'+tag+"-updates-512");};
      const auto base=root_for("RPB-v4");
      for(const std::string tag:{"RPB-v6","RPB-v7"}) {
        const auto other=root_for(tag);
        for(const std::string file:{"raw-scaler.pt","raw-normalizer.pt","pca-only.pt","raw-training.pt","raw-validation.pt","pca-training.pt","pca-validation.pt"}) {
          const bool present=fs::is_regular_file(base/file);require(present==fs::is_regular_file(other/file),"raw/PCA transform support differs across versions");
          if(present){numeric_archive_parity(base/file,other/file);++map_comparisons;}
        }
        for(const std::string method:{"raw","pca"}) {
          const auto file=fs::path("validation-views")/view_id/(method+"-features.pt");
          const bool present=fs::is_regular_file(base/file);require(present==fs::is_regular_file(other/file),"raw/PCA view transform support differs across versions");
          if(present){numeric_archive_parity(base/file,other/file);++map_comparisons;}
        }
        for(const auto &rep:readouts.repetitions)for(const std::string method:{"raw","pca_only"}) {
          const auto a=base/rep.id,b=other/rep.id;const bool supported=fs::is_regular_file(a/(method+"-fit.pt"));
          require(supported==fs::is_regular_file(b/(method+"-fit.pt")),"raw/PCA head fit support differs across versions");
          if(supported) {
            numeric_archive_parity(a/(method+"-fit.pt"),b/(method+"-fit.pt"));
            for(const std::string split:{"training","validation"})numeric_archive_parity(a/(method+'-'+split+"-predictions.pt"),b/(method+'-'+split+"-predictions.pt"));
            numeric_archive_parity(a/"validation-views"/view_id/(method+"-predictions.pt"),b/"validation-views"/view_id/(method+"-predictions.pt"));++control_fit_comparisons;
          }else {++unsupported_controls;}
          if(!first){controls << ',';}first=false;
          controls << "{\"master_seed\":" << master << ",\"repetition\":" << quote(rep.id) << ",\"method\":" << quote(method) << ",\"reference_tag\":\"RPB-v4\",\"comparator_tag\":" << quote(tag) << ",\"status\":" << quote(supported?"exact_numeric_fit_and_prediction_parity":"unsupported_consistently") << '}';
        }
      }
    }
    write_new(output/"raw-pca-parity.json",controls.str()+"],\"transform_archive_comparisons\":"+std::to_string(map_comparisons)+",\"supported_fit_and_prediction_comparisons\":"+std::to_string(control_fit_comparisons)+",\"consistent_unsupported_comparisons\":"+std::to_string(unsupported_controls)+",\"ordinary_TRAIN_fit_counts_by_method\":"+counts.str()+"}\n");
    std::ostringstream pairs;pairs << "{\"protocol\":" << quote(protocol) << ",\"split\":\"known VALIDATION\",\"test_access\":false,\"selection\":false,\"uncertainty\":\"within-master paired source-group; conditional common support and full-population correctness; no across-master CI\",\"comparisons\":[";first=true;
    for(const auto master:masters)for(const auto &rep:readouts.repetitions)for(const bool dropout:{false,true}) {
      const auto &split=cohorts.at(master).validation;
      const auto path=[&](const std::string &tag){const auto root=fs::path(readouts.output_directory)/("seed-"+std::to_string(master)+'-'+tag+"-updates-512")/rep.id;return dropout?root/"validation-views"/view_id/"native-predictions.pt":root/"native-validation-predictions.pt";};
      const auto candidate=load_prediction(path("RPB-v7"),split);
      for(const std::string tag:{"RPB-v4","RPB-v6"}) {
        const auto comparator=load_prediction(path(tag),split);const auto seed=ev::stream_seed(master,ev::stream_seed(rep.probe_seed,(dropout?0x30ULL:0ULL)+(tag=="RPB-v4"?4ULL:6ULL)));
        if(!first){pairs << ',';}first=false;pairs << "{\"master_seed\":" << master << ",\"repetition\":" << quote(rep.id) << ",\"view\":" << quote(dropout?view_id:"intact") << ",\"candidate\":\"RPB-v7\",\"comparator\":" << quote(tag) << ",\"candidate_prediction_file\":" << quote(path("RPB-v7").string()) << ",\"comparator_prediction_file\":" << quote(path(tag).string()) << ",\"paired\":" << paired_json(candidate,comparator,split,seed) << '}';
      }
    }
    write_new(output/"paired-validation-report.json",pairs.str()+"]}\n");guard.verify();require(sha256(bytes(admission_path))==options.at("--admission-log-sha256"),"admission log changed during measurement");write_new(output/"input-integrity-after.json",guard.json()+"\n");
    write_new(output/"complete.json","{\"protocol\":"+quote(protocol)+",\"status\":\"complete\",\"candidate_development_points\":10,\"positive_readout_inputs\":15,\"native_refit_parity_checks\":45,\"ordinary_TRAIN_fit_counts_by_method\":"+counts.str()+",\"paired_version_view_repetition_records\":60,\"paired_head_effects\":120,\"test_access\":false,\"stress_access\":false,\"selection\":false,\"promotion\":false,\"source_fingerprint\":"+quote(EVALUATION_SOURCE_ID)+"}\n");
    std::cout << "Fixed .15 TRAIN/known VALIDATION diagnostic complete; native fits and original predictions exactly retained; no TEST/stress or selection.\n";return 0;
  }catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
