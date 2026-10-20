// SPDX-License-Identifier: MIT
// Engineering fixture only. No project tensor or historical artifact is read.
#include "../include/historical_parent_paths.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
namespace hp = embedding::evaluation::historical_paths;
namespace {
void fails(const std::function<void()> &operation) {
  bool rejected = false;
  try { operation(); } catch (const std::exception &) { rejected = true; }
  assert(rejected);
}
}
int main() {
  assert(fs::exists("/.dockerenv"));
  const fs::path base="/opt/mityane-geometrics/migration/historical-path-tests";
  fs::create_directories(base);
  std::string pattern=(base/"fixture-XXXXXX").string();
  std::vector<char> writable(pattern.begin(),pattern.end());writable.push_back('\0');
  const char *created=::mkdtemp(writable.data());assert(created);
  const fs::path fixture(created),root=fixture/"preserved";
  fs::create_directory(root);
  ::unsetenv("MITYANE_HISTORICAL_PARENT_ROOT");
  fails([] { hp::parent_root(); });
  const auto redirected=fixture/"redirected";fs::create_directory_symlink(root,redirected);
  ::setenv("MITYANE_HISTORICAL_PARENT_ROOT",redirected.c_str(),1);
  fails([] { hp::parent_root(); });
  ::setenv("MITYANE_HISTORICAL_PARENT_ROOT",root.c_str(),1);
  assert(hp::parent_root()==root);
  assert(hp::physical("output/runs/rpb-parent/capsule/results/train.pt")==root/"output/runs/rpb-parent/capsule/results/train.pt");
  fails([] { hp::physical("/absolute/artifact.pt"); });
  fails([] { hp::physical("output/runs/../outside.pt"); });
  fails([] { hp::physical("output/runs/./same.pt"); });
  fails([] { hp::physical("output/runs/"); });
  const auto ledger=fixture/"mapping.json";hp::write_ledger(ledger);
  fails([&] { hp::write_ledger(ledger); });
  assert(hp::ledger().find("\"historical_prefix\":\"output/runs/\"")!=std::string::npos);
  assert(hp::ledger().find(root.string())!=std::string::npos);
  assert(fixture.parent_path()==base && fixture.filename().string().rfind("fixture-",0)==0);
  fs::remove_all(fixture);
  std::cout << "Historical identity and physical root mapping checks passed.\n";
}
