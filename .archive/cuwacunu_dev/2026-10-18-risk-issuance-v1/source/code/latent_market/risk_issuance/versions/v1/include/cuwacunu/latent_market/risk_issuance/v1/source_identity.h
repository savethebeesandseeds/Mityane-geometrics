// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/artifacts/v1/sidecar.h"
#ifndef CUWACUNU_RISK_ISSUANCE_V1_SOURCE_SHA256
#define CUWACUNU_RISK_ISSUANCE_V1_SOURCE_SHA256 ""
#endif
namespace cuwacunu::latent_market::risk_issuance::v1 {
inline std::vector<std::filesystem::path> source_files(const std::filesystem::path& root) {
  const auto bytes=cuwacunu::artifacts::v1::detail::read_bytes(root/"code/latent_market/risk_issuance/versions/v1/source-files.json",262144);
  const auto json=cuwacunu::artifacts::v1::detail::JsonReader(bytes).read();
  std::vector<std::filesystem::path> paths;
  for(const auto& value:cuwacunu::artifacts::v1::detail::arr(json))paths.emplace_back(cuwacunu::artifacts::v1::detail::str(value));
  return paths;
}
inline std::string source_identity(const std::filesystem::path& root=std::filesystem::current_path()) {
  return cuwacunu::artifacts::v1::source_closure_sha256(root,source_files(root));
}
inline void verify_runtime_source(const std::filesystem::path& root=std::filesystem::current_path()) {
  const std::string compiled=CUWACUNU_RISK_ISSUANCE_V1_SOURCE_SHA256;
  cuwacunu::artifacts::v1::require(cuwacunu::artifacts::v1::is_sha256(compiled) && compiled==source_identity(root),
    "risk issuance compiled implementation identity differs from current sources");
}
}
