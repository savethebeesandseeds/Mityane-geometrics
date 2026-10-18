// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/training_objective_diagnostic.h"
#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif
namespace fs = std::filesystem;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace {
void require(bool value,const std::string &message) {
  if(!value)throw std::runtime_error("[TRAIN objective diagnostic] "+message);
}
std::string bytes(const fs::path &path) {
  std::ifstream in(path,std::ios::binary);require(bool(in),"cannot read declared file: "+path.string());
  std::ostringstream out;out << in.rdbuf();require(!in.bad(),"declared file read failed");return out.str();
}
bool is_sha(const std::string &value) {
  return value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
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

std::vector<std::string> split(const std::string &line,char delimiter) {
  std::vector<std::string> result;size_t begin=0;
  while(true){const auto end=line.find(delimiter,begin);result.push_back(line.substr(begin,end-begin));
    if(end==std::string::npos)break;begin=end+1;}
  return result;
}
const std::vector<uint64_t> masters{4404,5505,6606,7707,8808};
const std::array<std::string,3> tags{"RPB-v4","RPB-v7","RPB-v8"};
const std::array<std::array<std::string,3>,3> producer_ids{{
  {"9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828",
   "2fff50c48605ee21dbbfd28d6989bdcf6b5cd057785c2acb19419571af48b895",
   "587f2423758c3e70c2c7665d9e7b59bfe10af4c7f9aaba23f5318b3c55b13bca"},
  {"f22cc8d5f2ac8af3e6a6056c6ad2693ffcc82cf39e8744c32fb13038710c5311",
   "d8ed2771d1615beabccbead081847813b2562b7b1f9b5347d4825c075ccc27a7",
   "e379b8f8abb102c256fb843466b4fbfbecb6fdc21298b862fd0b6dfeb47b853e"},
  {"78ab6ffdb0a3edf141e0ad81e58fe10f36e276d4b7d8af1288859004b5f0f38c",
   "3532deae194828f2640ff4fadacbcd94bbccfb8af6eaf37182cb8137ae84cf41",
   "073ec84b7c781cf48a578e58a4c9602c6b765c9c23f6065f27338780be8a1835"}
}};
const std::array<std::string,3> parents{
  "output/runs/rpb-context-replication/context-replication-JjNEUc/reference",
  "output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2/results/candidate-development",
  "output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV/results/candidate-development"};
const std::string heads="output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV/results/readouts/";
const std::array<std::string,7> files{"controlled-training.pt","milestone-512/checkpoint.pt",
  "milestone-512/checkpoint.pt.audit.pt","milestone-512/checkpoint.pt.scaler.pt",
  "milestone-512/checkpoint.pt.training-raw.pt","milestone-512/native-training.pt",
  "milestone-512/training-reconstruction.pt"};
std::string input_id(uint64_t master,size_t tag) {
  return "seed-"+std::to_string(master)+"-"+tags.at(tag)+"-updates-512";
}
std::array<std::string,13> paths(const fs::path &repo,uint64_t master,size_t tag) {
  std::array<std::string,13> result;
  const auto base=repo/parents.at(tag)/("seed-"+std::to_string(master)+"-lag_sign");
  for(size_t i=0;i<7;++i)result[i]=(base/files[i]).string();
  for(size_t rep=0;rep<3;++rep){const auto p=repo/heads/input_id(master,tag)/("rep-"+std::to_string(rep+1));
    result[7+rep*2]=(p/"native-fit.pt").string();
    result[8+rep*2]=(p/"native-training-predictions.pt").string();}
  return result;
}
const std::string tsv_header="input_id\ttag\tmaster_seed\tbudget\texpected_policy_id\ttraining_namespace\torchestration_source_id\ttraining_producer_source_id\tcore_writer_source_id\tcontrolled_training_path\tcheckpoint_path\tcheckpoint_audit_path\tcheckpoint_scaler_path\tcheckpoint_raw_training_path\tnative_training_features_path\ttraining_reconstruction_path\trep_1_native_fit_path\trep_1_training_predictions_path\trep_2_native_fit_path\trep_2_training_predictions_path\trep_3_native_fit_path\trep_3_training_predictions_path";
std::vector<rpb::TrainingObjectiveDiagnosticInstance> instances(const std::string &text,const fs::path &repo) {
  std::istringstream in(text);std::string line;require(bool(std::getline(in,line)) && line==tsv_header,"exact22 TSV header required");
  std::vector<rpb::TrainingObjectiveDiagnosticInstance> result;std::set<std::string> ids;
  while(std::getline(in,line)){
    const auto row=split(line,'\t');require(row.size()==22,"exact22 TSV fields required");
    rpb::TrainingObjectiveDiagnosticInstance out;out.input_id=row[0];out.model_tag=row[1];
    out.master_seed=std::stoull(row[2]);out.budget=std::stoll(row[3]);
    require(row[2]==std::to_string(out.master_seed) && row[3]=="512" &&
      std::find(masters.begin(),masters.end(),out.master_seed)!=masters.end(),"known master and budget512 required");
    const auto found=std::find(tags.begin(),tags.end(),out.model_tag);require(found!=tags.end(),"known tag required");
    const size_t tag=static_cast<size_t>(found-tags.begin());
    require(out.input_id==input_id(out.master_seed,tag) && ids.insert(out.input_id).second,"unique exact instance identity");
    out.expected_policy_id=row[4];out.training_namespace=row[5];
    const std::array<std::string,3> policies{"","rpb-training-context-deletion-015-v1","rpb-training-context-balanced-030-v1"};
    require(row[4]==policies[tag] && row[5]=="native-development-v1/lag_sign","fixed policy and TRAIN namespace");
    require(is_sha(row[6]) && is_sha(row[7]) && is_sha(row[8]),"original producer identities required");
    for(size_t i=0;i<3;++i)require(row[6+i]==producer_ids[tag][i],"pinned original producer scope differs");
    out.orchestration_source_id=row[6];out.training_producer_source_id=row[7];out.core_writer_source_id=row[8];
    const auto expected=paths(repo,out.master_seed,tag);
    for(size_t i=0;i<13;++i)require(row[9+i]==expected[i],"undeclared path in TRAIN TSV");
    out.controlled_training_path=row[9];out.checkpoint_path=row[10];out.checkpoint_audit_path=row[11];
    out.scaler_path=row[12];out.training_raw_path=row[13];out.native_training_path=row[14];out.training_reconstruction_path=row[15];
    for(size_t rep=0;rep<3;++rep){out.ridge_fit_paths[rep]=row[16+rep*2];out.saved_training_prediction_paths[rep]=row[17+rep*2];}
    result.push_back(out);
  }
  require(in.eof() && result.size()==15,"all fifteen unique instances required");return result;
}
struct InputGuard {
  std::map<std::string,std::string> expected;
  InputGuard(const std::string &manifest,const fs::path &repo) {
    std::set<std::string> allowed;
    for(const auto master:masters)for(size_t tag=0;tag<3;++tag)for(const auto &path:paths(repo,master,tag))allowed.insert(path);
    require(allowed.size()==195,"internal exact195 TRAIN role count");
    std::istringstream in(manifest);std::string line;
    // Validate the entire path set before the first TRAIN payload hash.
    while(std::getline(in,line)){
      require(line.size()>66 && is_sha(line.substr(0,64)) && line.substr(64,2)=="  ","malformed input SHA line");
      const auto path=line.substr(66);require(allowed.count(path) && expected.emplace(path,line.substr(0,64)).second,"undeclared or duplicate TRAIN role");
    }
    require(in.eof() && expected.size()==allowed.size(),"exact195 input manifest required");verify();
  }
  void verify() const {
    for(const auto &[path,digest]:expected){require(fs::is_regular_file(path) && fs::canonical(path)==fs::path(path),"redirected/missing input");
      require(sha256(bytes(path))==digest,"input SHA mismatch: "+path);}
  }
};
}
int main(int argc,char **argv) {
  try {
    require(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA implementation self-check");
    if(argc==2 && std::string(argv[1])=="--source-id"){std::cout << EVALUATION_SOURCE_ID << '\n';return 0;}
    if(argc==2 && std::string(argv[1])=="--metadata-self-test"){
      std::set<std::string> allowed;for(const auto master:masters)for(size_t tag=0;tag<3;++tag)for(const auto &path:paths("/embedding",master,tag))allowed.insert(path);
      require(allowed.size()==195 && split(tsv_header,'\t').size()==22,"metadata role schema");
      std::cout << "Metadata/SHA self-check passed; payloads read0.\n";return 0;
    }
    std::map<std::string,std::string> args;
    require(argc==17,"required args: --repo-root --output --instances --instances-sha256 --input-manifest --input-manifest-sha256 --admission-log --admission-log-sha256");
    for(int i=1;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"duplicate argument");
    for(const auto *key:{"--repo-root","--output","--instances","--instances-sha256","--input-manifest","--input-manifest-sha256","--admission-log","--admission-log-sha256"})require(args.count(key),"missing required argument");
    require(is_sha(EVALUATION_SOURCE_ID) && is_sha(args.at("--admission-log-sha256")),"compiled source and admitted log binding required");
    const auto repo=fs::canonical(args.at("--repo-root"));require(repo==fs::path("/embedding"),"managed repository mount required");
    const auto output=fs::absolute(args.at("--output")).lexically_normal();
    require(output.string().rfind((repo/"output/runs/rpb-training-objective-diagnostic/").string(),0)==0 && !fs::exists(output),"new diagnostic output required");
    const auto admission=bytes(args.at("--admission-log"));
    require(sha256(admission)==args.at("--admission-log-sha256") && admission.find("TRAIN objective CUDA admission passed")!=std::string::npos,"actual CUDA admission log missing/mismatched");
    const auto tsv=bytes(args.at("--instances")),manifest=bytes(args.at("--input-manifest"));
    require(sha256(tsv)==args.at("--instances-sha256") && sha256(manifest)==args.at("--input-manifest-sha256"),"frozen metadata binding mismatch");
    auto parsed=instances(tsv,repo);InputGuard guard(manifest,repo);
    rpb::TrainingObjectiveDiagnosticRun run{output.string(),EVALUATION_SOURCE_ID,std::move(parsed)};
    rpb::run_training_objective_diagnostic(run);
    guard.verify();
    require(bytes(args.at("--instances"))==tsv && bytes(args.at("--input-manifest"))==manifest,"input metadata changed");
    require(bytes(args.at("--admission-log"))==admission,"admission log changed");
    std::cout << "TRAIN diagnostic completed; instances15, encoder updates0, head fits0, input hashes195 preserved.\n";
    return 0;
  } catch(const std::exception &error){std::cerr << error.what() << '\n';return 1;}
}
