// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include "cuwacunu/artifacts/selected_encoder.h"
#include <torch/cuda.h>
#include <iostream>
#include <set>

namespace {
constexpr const char *usage =
    "Usage: adopt_encoder_snapshot [--repo-root PATH] --source-digest\n"
    "       adopt_encoder_snapshot [--repo-root PATH] --source-sha256 SHA256 "
    "[--settings config/encoder.conf] [--runtime config/encoder_runtime.conf]\n"
    "Adopts only the already selected Encoder; it never rewrites archive bytes or runtime selection.";
}

int main(int argc, char **argv) try {
  namespace adapter = cuwacunu::artifacts::selected_encoder;
  std::filesystem::path root{"."}, settings{"config/encoder.conf"},
      runtime{"config/encoder_runtime.conf"};
  std::string expected_source;
  bool inspect = false;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    TORCH_CHECK(seen.insert(flag).second, "[Encoder adoption] duplicate argument: ", flag);
    if (flag == "--help") { std::cout << usage << '\n'; return 0; }
    if (flag == "--source-digest") { inspect = true; continue; }
    TORCH_CHECK(i + 1 < argc, usage);
    const std::string value = argv[++i];
    if (flag == "--repo-root") root = value;
    else if (flag == "--settings") settings = value;
    else if (flag == "--runtime") runtime = value;
    else if (flag == "--source-sha256") expected_source = value;
    else TORCH_CHECK(false, usage);
  }
  if (inspect) {
    TORCH_CHECK(expected_source.empty(), "[Encoder adoption] inspect and adoption are separate operations");
    std::cout << adapter::source_digest(root) << '\n';
    return 0;
  }
  TORCH_CHECK(!expected_source.empty(), usage);
  TORCH_CHECK(torch::cuda::is_available(),
      "[Encoder adoption] selected Encoder requires the managed CUDA runtime");
  torch::set_num_threads(1);
  const auto result = adapter::adopt_selected(settings, runtime, root, expected_source);
  std::cout << "{\"component\":\"Encoder\",\"sidecar_sha256\":\""
      << result.sidecar_sha256 << "\",\"model_ref\":\"" << result.model_ref
      << "\",\"archive_sha256\":\"" << result.archive_sha256 << "\"}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
