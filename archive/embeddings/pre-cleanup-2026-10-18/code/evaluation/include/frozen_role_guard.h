// SPDX-License-Identifier: MIT
#pragma once
// Exact-byte helpers retained from the frozen native-view evaluator. New
// protocols can share role binding without rewriting historical drivers.
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace embedding::evaluation::frozen_inputs {
namespace fs = std::filesystem;
inline void require(bool value,const std::string &message) {
  if(!value)throw std::runtime_error("[frozen input roles] "+message);
}
inline std::string quote(const std::string &value) {
  std::ostringstream out;out << '"';
  for(const unsigned char c:value) {
    if(c=='"' || c=='\\')out << '\\' << c;
    else if(c<32)out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str()+'"';
}
inline std::string strings(const std::vector<std::string> &values) {
  std::ostringstream out;out << '[';
  for(size_t i=0;i<values.size();++i){if(i){out << ',';}out << quote(values[i]);}
  return out.str()+']';
}
inline std::string fields(const std::map<std::string,std::string> &values) {
  std::ostringstream out;out << '{';bool first=true;
  for(const auto &[key,value]:values){if(!first){out << ',';}first=false;out << quote(key) << ':' << quote(value);}
  return out.str()+'}';
}
inline std::string bytes(const fs::path &path) {
  std::ifstream in(path,std::ios::binary);require(bool(in),"cannot read declared file: "+path.string());
  std::ostringstream out;out << in.rdbuf();require(!in.bad(),"declared file read failed: "+path.string());return out.str();
}
inline bool is_sha(const std::string &value) {
  return value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
// Same dependency-free exact-byte digest used by the shared archive evaluator.
inline std::string sha256(const std::string &input) {
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
inline void write_new(const fs::path &path,const std::string &value) {
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

struct Guard {
  std::map<std::string,fs::path> allowed;
  std::map<std::string,std::string> expected;
  std::map<std::string,size_t> byte_counts;
  Guard(const fs::path &manifest,std::map<std::string,fs::path> roles):allowed(std::move(roles)) {
    std::ifstream in(manifest); require(bool(in),"closed parent SHA manifest required");
    std::string line; std::set<fs::path> physical;
    while(std::getline(in,line)) {
      if(!line.empty() && line.back()=='\r')line.pop_back();
      require(line.size()>66 && is_sha(line.substr(0,64)) && line.substr(64,2)=="  ","malformed SHA role");
      const auto key=line.substr(66); require(allowed.count(key),"undeclared role: "+key);
      const auto path=fs::absolute(allowed.at(key)).lexically_normal();
      require(fs::is_regular_file(path) && !fs::is_symlink(path) && fs::canonical(path)==path,"missing/redirected role: "+key);
      require(expected.emplace(key,line.substr(0,64)).second && physical.insert(path).second,"duplicate role/path");
      allowed[key]=path;
    }
    require(in.eof() && expected.size()==allowed.size(),"entire closed role set required before payload access");
    for(const auto &[key,digest]:expected) {
      const auto value=bytes(allowed.at(key)); require(sha256(value)==digest,"parent SHA differs: "+key);
      byte_counts[key]=value.size();
    }
  }
  std::string bind(const std::string &key) const {
    require(expected.count(key),"unbound role: "+key); return allowed.at(key).string();
  }
  void verify() const {
    for(const auto &[key,digest]:expected) {
      const auto value=bytes(allowed.at(key));
      require(value.size()==byte_counts.at(key) && sha256(value)==digest,"parent changed: "+key);
    }
  }
  std::string json() const {
    std::ostringstream out; out << "{\"checksum_algorithm\":\"sha256-file-bytes\",\"closed_roles\":true,\"files\":[";
    bool first=true;
    for(const auto &[key,digest]:expected) {
      if(!first){out << ',';} first=false;
      out << "{\"manifest_path\":" << quote(key) << ",\"path\":" << quote(allowed.at(key).string())
          << ",\"sha256\":" << quote(digest) << ",\"bytes\":" << byte_counts.at(key) << '}';
    }
    return out.str()+"]}";
  }
};
} // namespace embedding::evaluation::frozen_inputs
