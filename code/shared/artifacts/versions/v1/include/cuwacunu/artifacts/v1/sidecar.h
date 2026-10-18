// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../LICENSE.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cuwacunu::artifacts::v1 {
inline constexpr std::string_view encoder_component = "Encoder";
inline constexpr std::string_view state_representation_component = "State Representation Network";
inline constexpr std::string_view target_state_representation_component = "Target State Representation Network";
inline constexpr std::string_view dynamics_component = "Dynamics Model";
inline constexpr std::string_view portfolio_policy_component = "Portfolio Policy";

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("Artifact sidecar: " + message);
}
inline bool is_sha256(std::string_view value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

// SHA256 is kept local to this version; no archive representation is trusted.
class Sha256 {
  std::array<uint32_t, 8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::array<uint8_t, 64> pending_{};
  uint64_t length_ = 0;
  size_t used_ = 0;
  static uint32_t rotate(uint32_t x, unsigned n) { return (x >> n) | (x << (32-n)); }
  void block(const std::array<uint8_t, 64>& bytes) {
    constexpr std::array<uint32_t,64> k{
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<uint32_t, 64> w{};
    for (size_t i=0;i<16;++i) w[i]=(uint32_t(bytes[i*4])<<24)|(uint32_t(bytes[i*4+1])<<16)|(uint32_t(bytes[i*4+2])<<8)|bytes[i*4+3];
    for (size_t i=16;i<64;++i) {
      const auto a=w[i-15],b=w[i-2];
      w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));
    }
    auto [a,b,c,d,e,f,g,h]=state_;
    for (size_t i=0;i<64;++i) {
      const auto t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
      const auto t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
      h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }
public:
  void update(std::string_view bytes) {
    require(bytes.size() <= UINT64_MAX/8 - length_, "SHA256 input exceeds supported length");
    length_ += bytes.size();
    for (unsigned char byte : bytes) {
      pending_[used_++] = byte;
      if (used_ == 64) { block(pending_); used_ = 0; }
    }
  }
  std::string finish() const {
    auto copy=*this;
    copy.pending_[copy.used_++]=0x80;
    if (copy.used_>56) {std::fill(copy.pending_.begin()+copy.used_,copy.pending_.end(),0);copy.block(copy.pending_);copy.used_=0;}
    std::fill(copy.pending_.begin()+copy.used_,copy.pending_.begin()+56,0);
    for (unsigned j=0;j<8;++j) copy.pending_[63-j]=static_cast<uint8_t>((copy.length_*8)>>(j*8));
    copy.block(copy.pending_);
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for (auto value:copy.state_) out<<std::setw(8)<<value;
    return out.str();
  }
};
inline std::string sha256_bytes(std::string_view bytes) {Sha256 hash;hash.update(bytes);return hash.finish();}
inline std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path,std::ios::binary);
  require(bool(input),"cannot read " + path.string());
  Sha256 hash;std::array<char,65536> buffer{};
  while (input) {input.read(buffer.data(),buffer.size());hash.update(std::string_view(buffer.data(),static_cast<size_t>(input.gcount())));}
  require(input.eof()&&!input.bad(),"read failure for " + path.string());
  return hash.finish();
}
inline void hash_field(Sha256& hash,std::string_view field) {
  std::array<char,8> size{};
  for (unsigned j=0;j<8;++j) size[j]=static_cast<char>((uint64_t(field.size())>>(8*j))&255);
  hash.update(std::string_view(size.data(),size.size()));hash.update(field);
}
inline std::string digest_fields(std::string_view domain,const std::vector<std::string>& fields) {
  Sha256 hash;hash_field(hash,"cuwacunu_artifact_record_v1");hash_field(hash,domain);
  hash_field(hash,std::to_string(fields.size()));
  for (const auto& field:fields) hash_field(hash,field);
  return hash.finish();
}
inline std::string source_closure_sha256(const std::filesystem::path& root,std::vector<std::filesystem::path> files) {
  require(!files.empty(),"empty source closure");
  const auto canonical_root=std::filesystem::canonical(root);
  std::sort(files.begin(),files.end(),[](const auto& a,const auto& b){return a.generic_string()<b.generic_string();});
  std::vector<std::string> fields;
  std::string previous;
  for (const auto& file:files) {
    const auto name=file.generic_string();
    require(!file.empty()&&!file.is_absolute()&&file.lexically_normal()==file&&name!=previous,"invalid or duplicate source path " + name);
    for (const auto& part:file) require(part!=".."&&part!=".","source path escapes closure");
    const auto actual=std::filesystem::canonical(canonical_root/file);
    const auto relative=actual.lexically_relative(canonical_root);
    require(!relative.empty()&&*relative.begin()!=".."&&std::filesystem::is_regular_file(actual),"source file escapes closure " + name);
    fields.push_back(name);fields.push_back(sha256_file(actual));previous=name;
  }
  return digest_fields("source_closure_v1",fields);
}

struct ModuleRef {
  std::string component,implementation_version,source_sha256;
  bool operator==(const ModuleRef&) const = default;
};
struct ParentRef {
  std::string role,component,model_ref;
  bool operator==(const ParentRef&) const = default;
};
struct Component {
  ModuleRef module_ref;
  std::string state_sha256,inference_config_sha256,model_ref;
  std::vector<ParentRef> parents;
  bool operator==(const Component&) const = default;
};
struct Manifest {
  std::string archive_kind,archive_sha256,resume_identity_sha256;
  std::vector<Component> components;
  bool operator==(const Manifest&) const = default;
};
inline bool supported_component(std::string_view name) {
  return name==encoder_component||name==state_representation_component||name==target_state_representation_component||name==dynamics_component;
}
inline std::vector<std::pair<std::string,std::string>> required_parents(std::string_view component) {
  if(component==encoder_component)return {};
  if(component==state_representation_component||component==target_state_representation_component)return {{"embedding",std::string(encoder_component)}};
  if(component==dynamics_component)return {{"input_history",std::string(state_representation_component)},{"prediction_target",std::string(target_state_representation_component)}};
  throw std::runtime_error("Artifact sidecar: unsupported component " + std::string(component));
}
inline std::string derive_model_ref(const Component& component) {
  std::vector<std::string> fields{component.module_ref.component,component.module_ref.implementation_version,
    component.module_ref.source_sha256,component.state_sha256,component.inference_config_sha256,std::to_string(component.parents.size())};
  for(const auto& parent:component.parents) {fields.push_back(parent.role);fields.push_back(parent.component);fields.push_back(parent.model_ref);}
  return digest_fields("ModelRef",fields);
}
inline void validate_component(const Component& component) {
  const auto& module=component.module_ref;
  require(supported_component(module.component),"unsupported component " + module.component);
  require(module.implementation_version.size()>1&&module.implementation_version[0]=='v'&&module.implementation_version[1]>='1'&&module.implementation_version[1]<='9'&&
    std::all_of(module.implementation_version.begin()+2,module.implementation_version.end(),[](char c){return c>='0'&&c<='9';}),"invalid implementation version for " + module.component);
  require(is_sha256(module.source_sha256)&&is_sha256(component.state_sha256)&&is_sha256(component.inference_config_sha256)&&is_sha256(component.model_ref),"invalid digest for " + module.component);
  const auto expected=required_parents(module.component);
  require(component.parents.size()==expected.size(),"wrong parent count for " + module.component);
  for(size_t i=0;i<expected.size();++i) {
    const auto& parent=component.parents[i];
    require(parent.role==expected[i].first&&parent.component==expected[i].second&&is_sha256(parent.model_ref),"invalid parent role/component for " + module.component);
  }
  require(derive_model_ref(component)==component.model_ref,"ModelRef does not match content for " + module.component);
}
inline Component make_component(ModuleRef module,std::string state,std::string config,std::vector<ParentRef> parents={}) {
  std::sort(parents.begin(),parents.end(),[](const auto& a,const auto& b){return a.role<b.role;});
  Component result{std::move(module),std::move(state),std::move(config),{},std::move(parents)};
  result.model_ref=derive_model_ref(result);validate_component(result);return result;
}
inline void validate_manifest(const Manifest& manifest) {
  require(manifest.archive_kind=="training"||manifest.archive_kind=="inference","unsupported archive kind");
  require(is_sha256(manifest.archive_sha256),"invalid archive SHA256");
  require(manifest.archive_kind=="training"?is_sha256(manifest.resume_identity_sha256):manifest.resume_identity_sha256.empty(),"resume identity must exist only for training archives");
  require(!manifest.components.empty()&&manifest.components.size()<=4,"invalid component count");
  std::set<std::string> names;
  for(const auto& component:manifest.components) {validate_component(component);require(names.insert(component.module_ref.component).second,"duplicate component " + component.module_ref.component);}
}
inline void validate_parents(const std::vector<Component>& components) {
  require(!components.empty()&&components.size()<=4,"invalid selected component count");
  std::map<std::string,const Component*> selected;
  for(const auto& component:components) {validate_component(component);require(selected.emplace(component.module_ref.component,&component).second,"duplicate selected component " + component.module_ref.component);}
  for(const auto& component:components)for(const auto& parent:component.parents) {
    const auto actual=selected.find(parent.component);
    require(actual!=selected.end(),component.module_ref.component + " missing " + parent.role + " parent " + parent.component);
    require(actual->second->model_ref==parent.model_ref,component.module_ref.component + " " + parent.role + " parent " + parent.component + " expected " + parent.model_ref + " actual " + actual->second->model_ref);
  }
}
inline void verify_module_ref(const Component& component,const ModuleRef& supported) {
  require(component.module_ref==supported,"unsupported ModuleRef for " + component.module_ref.component+
    "; expected " + supported.component+"/"+supported.implementation_version+"/"+supported.source_sha256+
    " actual " + component.module_ref.component+"/"+component.module_ref.implementation_version+"/"+component.module_ref.source_sha256);
}

namespace detail {
// The sidecar grammar has strings, objects and arrays only. Limits are enforced
// before allocation; unknown/duplicate keys never silently override a binding.
struct Json {
  enum class Kind { string,object,array } kind;
  std::string string;
  std::map<std::string,Json> object;
  std::vector<Json> array;
};
class JsonReader {
  std::string_view text_;size_t at_=0,nodes_=0;
  void whitespace(){while(at_<text_.size()&&(text_[at_]==' '||text_[at_]=='\r'||text_[at_]=='\n'||text_[at_]=='\t'))++at_;}
  char take(){require(at_<text_.size(),"truncated JSON");return text_[at_++];}
  std::string string() {
    require(take()=='"',"expected JSON string");std::string value;
    for(;;) {
      auto c=static_cast<unsigned char>(take());
      if(c=='"')return value;
      require(c>=32&&c<127,"sidecar strings must be ASCII");
      if(c=='\\') {const auto escaped=take();require(escaped=='"'||escaped=='\\'||escaped=='/',"unsupported JSON escape");c=static_cast<unsigned char>(escaped);}
      require(value.size()<4096,"JSON string too long");value+=static_cast<char>(c);
    }
  }
  Json value(unsigned depth) {
    require(depth<=8&&++nodes_<=256,"sidecar JSON nesting/size limit");whitespace();require(at_<text_.size(),"missing JSON value");
    if(text_[at_]=='"')return {Json::Kind::string,string(),{}, {}};
    const auto opening=take();require(opening=='{'||opening=='[',"unsupported JSON value");
    Json result{opening=='{'?Json::Kind::object:Json::Kind::array,{},{},{}};
    const char closing=opening=='{'?'}':']';whitespace();
    if(at_<text_.size()&&text_[at_]==closing){++at_;return result;}
    for(;;) {
      whitespace();
      if(opening=='{') {
        const auto key=string();whitespace();require(take()==':',"expected JSON colon");
        auto child=value(depth+1);require(result.object.emplace(key,std::move(child)).second,"duplicate JSON field " + key);
      } else result.array.push_back(value(depth+1));
      whitespace();const auto separator=take();if(separator==closing)return result;require(separator==',',"expected JSON comma");
    }
  }
public:
  explicit JsonReader(std::string_view text):text_(text){require(text.size()<=262144,"sidecar exceeds 256 KiB");}
  Json read(){auto result=value(0);whitespace();require(at_==text_.size(),"trailing JSON data");return result;}
};
inline void keys(const Json& value,std::initializer_list<std::string_view> required,std::initializer_list<std::string_view> optional={}) {
  require(value.kind==Json::Kind::object,"expected JSON object");
  for(auto key:required)require(value.object.count(std::string(key)),"missing JSON field " + std::string(key));
  for(const auto& [key,unused]:value.object) {bool known=false;for(auto field:required)known|=key==field;for(auto field:optional)known|=key==field;require(known,"unknown JSON field " + key);}
}
inline std::string str(const Json& value){require(value.kind==Json::Kind::string,"expected JSON string");return value.string;}
inline const std::vector<Json>& arr(const Json& value){require(value.kind==Json::Kind::array,"expected JSON array");return value.array;}
inline std::string quote(std::string_view value) {
  std::string out="\"";
  for(unsigned char c:value){require(c>=32&&c<127,"sidecar string is not printable ASCII");if(c=='"'||c=='\\')out+='\\';out+=static_cast<char>(c);}
  return out+'"';
}
inline std::string read_bytes(const std::filesystem::path& path,uintmax_t limit) {
  std::ifstream input(path,std::ios::binary);require(bool(input),"cannot read " + path.string());
  std::string bytes;std::array<char,65536> buffer{};
  while(input){input.read(buffer.data(),buffer.size());const auto count=static_cast<size_t>(input.gcount());require(count<=limit-bytes.size(),"artifact exceeds byte limit");bytes.append(buffer.data(),count);}
  require(input.eof()&&!input.bad(),"read failure for " + path.string());return bytes;
}
} // namespace detail

inline std::string serialize_manifest(const Manifest& manifest) {
  validate_manifest(manifest);using detail::quote;
  std::string out="{\"schema\":\"model_artifact_sidecar_v1\",\"archive_kind\":"+quote(manifest.archive_kind)+",\"archive_sha256\":"+quote(manifest.archive_sha256);
  if(manifest.archive_kind=="training")out+=",\"resume_identity_sha256\":"+quote(manifest.resume_identity_sha256);
  out+=",\"components\":[";
  for(size_t i=0;i<manifest.components.size();++i) {
    if(i)out+=',';
    const auto& c=manifest.components[i];
    out+="{\"module_ref\":{\"component\":"+quote(c.module_ref.component)+",\"implementation_version\":"+quote(c.module_ref.implementation_version)+",\"source_sha256\":"+quote(c.module_ref.source_sha256)+"},\"weights_ref\":{\"state_sha256\":"+quote(c.state_sha256)+"},\"inference_config_sha256\":"+quote(c.inference_config_sha256)+",\"model_ref\":"+quote(c.model_ref)+",\"parents\":[";
    for(size_t j=0;j<c.parents.size();++j){if(j)out+=',';const auto& p=c.parents[j];out+="{\"role\":"+quote(p.role)+",\"component\":"+quote(p.component)+",\"model_ref\":"+quote(p.model_ref)+"}";}
    out+="]}";
  }
  return out+"]}\n";
}
inline Manifest parse_manifest(std::string_view bytes) {
  using namespace detail;
  const auto root=JsonReader(bytes).read();keys(root,{"schema","archive_kind","archive_sha256","components"},{"resume_identity_sha256"});
  require(str(root.object.at("schema"))=="model_artifact_sidecar_v1","unsupported schema");
  Manifest result{str(root.object.at("archive_kind")),str(root.object.at("archive_sha256")),{}, {}};
  if(root.object.count("resume_identity_sha256"))result.resume_identity_sha256=str(root.object.at("resume_identity_sha256"));
  require(result.archive_kind=="training"||!root.object.count("resume_identity_sha256"),"inference sidecar contains resume identity");
  for(const auto& entry:arr(root.object.at("components"))) {
    keys(entry,{"module_ref","weights_ref","inference_config_sha256","model_ref","parents"});
    const auto& module=entry.object.at("module_ref");keys(module,{"component","implementation_version","source_sha256"});
    const auto& weights=entry.object.at("weights_ref");keys(weights,{"state_sha256"});
    Component c{{str(module.object.at("component")),str(module.object.at("implementation_version")),str(module.object.at("source_sha256"))},str(weights.object.at("state_sha256")),str(entry.object.at("inference_config_sha256")),str(entry.object.at("model_ref")),{}};
    for(const auto& parent:arr(entry.object.at("parents"))) {keys(parent,{"role","component","model_ref"});c.parents.push_back({str(parent.object.at("role")),str(parent.object.at("component")),str(parent.object.at("model_ref"))});}
    result.components.push_back(std::move(c));
  }
  validate_manifest(result);return result;
}
inline std::filesystem::path sidecar_path(const std::filesystem::path& archive) {return std::filesystem::path(archive.string()+".manifest.json");}
inline std::string publish_sidecar(const std::filesystem::path& archive,const Manifest& manifest) {
  validate_manifest(manifest);require(sha256_file(archive)==manifest.archive_sha256,"archive bytes disagree during sidecar publication");
  const auto destination=sidecar_path(archive);require(!std::filesystem::exists(destination),"sidecar already exists " + destination.string());
  const auto bytes=serialize_manifest(manifest);
  std::filesystem::path temporary;
  for(unsigned attempt=0;attempt<100;++attempt) {
    temporary=std::filesystem::path(destination.string()+".publishing-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(attempt));
    if(std::filesystem::create_directory(temporary))break;
    temporary.clear();
  }
  require(!temporary.empty(),"cannot create exclusive sidecar publication directory");
  try {
    const auto staged=temporary/"manifest.json";
    {std::ofstream output(staged,std::ios::binary|std::ios::trunc);require(bool(output),"cannot stage sidecar");output.write(bytes.data(),bytes.size());output.close();require(bool(output),"cannot finish sidecar write");}
    std::filesystem::permissions(staged,std::filesystem::perms::owner_read|std::filesystem::perms::group_read|std::filesystem::perms::others_read);
    // Hard-link publication is atomic and refuses to replace an existing name.
    std::filesystem::create_hard_link(staged,destination);
    std::filesystem::remove_all(temporary);
  } catch(...) {std::error_code ignored;std::filesystem::remove_all(temporary,ignored);throw;}
  return sha256_bytes(bytes);
}
struct VerifiedArtifact {std::string bytes;Manifest manifest;std::string sidecar_sha256;};
inline VerifiedArtifact read_verified(const std::filesystem::path& archive,const std::string& expected_sidecar_sha256,const std::string& expected_kind) {
  require(is_sha256(expected_sidecar_sha256),"missing/invalid pinned sidecar SHA256");
  const auto sidecar=detail::read_bytes(sidecar_path(archive),262144);
  require(sha256_bytes(sidecar)==expected_sidecar_sha256,"pinned sidecar SHA256 mismatch for " + archive.string());
  auto manifest=parse_manifest(sidecar);require(manifest.archive_kind==expected_kind,"archive kind expected " + expected_kind + " actual " + manifest.archive_kind);
  auto bytes=detail::read_bytes(archive,uintmax_t{2}*1024*1024*1024);
  require(sha256_bytes(bytes)==manifest.archive_sha256,"archive SHA256 mismatch for " + archive.string());
  return {std::move(bytes),std::move(manifest),expected_sidecar_sha256};
}
struct BundleSelection {std::string sidecar_sha256,component,model_ref;};
inline std::string bundle_ref(std::vector<BundleSelection> selected) {
  require(!selected.empty()&&selected.size()<=4,"invalid bundle selection");
  std::sort(selected.begin(),selected.end(),[](const auto& a,const auto& b){return a.component<b.component;});
  std::vector<std::string> fields;std::string previous;
  for(const auto& item:selected){require(supported_component(item.component)&&item.component!=previous&&is_sha256(item.sidecar_sha256)&&is_sha256(item.model_ref),"invalid bundle item " + item.component);fields.push_back(item.sidecar_sha256);fields.push_back(item.component);fields.push_back(item.model_ref);previous=item.component;}
  return digest_fields("BundleRef",fields);
}
} // namespace cuwacunu::artifacts::v1
