// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuwacunu::latent_market::v1 {
// Streaming SHA256: admission verifies actual bytes, rather than trusting an ID
// stored in an archive. This application utility has no Encoder dependency.
inline std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot hash artifact: " + path.string());
  constexpr std::array<uint32_t,64> k{
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  std::array<uint32_t,8> state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  const auto rotate=[](uint32_t x, unsigned n){return (x>>n)|(x<<(32-n));};
  const auto block=[&](const std::array<uint8_t,64>& b) {
    std::array<uint32_t,64> w{};
    for(size_t i=0;i<16;++i) w[i]=(uint32_t(b[i*4])<<24)|(uint32_t(b[i*4+1])<<16)|(uint32_t(b[i*4+2])<<8)|b[i*4+3];
    for(size_t i=16;i<64;++i) {
      auto a=w[i-15],c=w[i-2];
      w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(c,17)^rotate(c,19)^(c>>10));
    }
    auto [a,b0,c,d,e,f,g,h]=state;
    for(size_t i=0;i<64;++i) {
      const auto t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
      const auto t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b0)^(a&c)^(b0&c));
      h=g;g=f;f=e;e=d+t1;d=c;c=b0;b0=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b0;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
  };
  std::array<uint8_t,64> bytes{}; uint64_t length=0;
  while(true) {
    in.read(reinterpret_cast<char*>(bytes.data()),64);
    const auto n=static_cast<size_t>(in.gcount());
    if (in.bad() || length>UINT64_MAX/8-n) throw std::runtime_error("Artifact hash read/length failure");
    length+=n;
    if(n==64) {block(bytes);continue;}
    bytes[n]=0x80;
    for(size_t j=n+1;j<64;++j)bytes[j]=0;
    if(n>=56){block(bytes);bytes.fill(0);}
    for(unsigned j=0;j<8;++j)bytes[63-j]=static_cast<uint8_t>((length*8)>>(j*8));
    block(bytes);break;
  }
  std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto s:state)out<<std::setw(8)<<s;return out.str();
}
inline void verify_sha256(const std::filesystem::path& path,const std::string& expected) {
  if(expected.size()!=64 || sha256_file(path)!=expected)throw std::runtime_error("SHA256 admission failed: "+path.string());
}
struct ArtifactPin {std::filesystem::path path;std::string hash;};
inline std::vector<ArtifactPin> read_pins(const std::filesystem::path& manifest) {
  std::ifstream in(manifest);if(!in)throw std::runtime_error("Cannot read SHA256 manifest");
  std::vector<ArtifactPin> pins;std::string line;
  while(std::getline(in,line)) {
    if(line.empty() || line[0]=='#')continue;
    if(line.size()<67 || line.substr(64,2)!="  ")throw std::runtime_error("Manifest requires SHA256, two spaces, path");
    auto p=std::filesystem::path(line.substr(66));if(p.is_relative())p=manifest.parent_path()/p;
    pins.push_back({std::filesystem::weakly_canonical(p),line.substr(0,64)});
    verify_sha256(p,pins.back().hash);
  }
  if(!in.eof() || pins.empty())throw std::runtime_error("Empty/unreadable manifest");
  return pins;
}
inline void verify_pins(const std::vector<ArtifactPin>& pins) {for(const auto& p:pins)verify_sha256(p.path,p.hash);}
inline void require_pinned(const std::vector<ArtifactPin>& pins,const std::filesystem::path& path) {
  const auto canonical=std::filesystem::weakly_canonical(path);
  for(const auto& p:pins)if(p.path==canonical){verify_sha256(p.path,p.hash);return;}
  throw std::runtime_error("Artifact missing from independently verified manifest: "+path.string());
}
inline void new_leaf(const std::filesystem::path& leaf) {
  std::filesystem::create_directories(leaf.parent_path());
  if(!std::filesystem::create_directory(leaf))throw std::runtime_error("Output leaf already exists: "+leaf.string());
}
} // namespace cuwacunu::latent_market::v1
