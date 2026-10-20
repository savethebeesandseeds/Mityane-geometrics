// SPDX-License-Identifier: MIT
#pragma once
// Physical location mapping only. Original receipt/plan identities remain the
// frozen output/runs/... names. No mapping is consulted by metadata-only plans.
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

namespace embedding::evaluation::historical_paths {
namespace fs = std::filesystem;
inline const fs::path &parent_root() {
  static const fs::path root = [] {
    const char *configured = std::getenv("MITYANE_HISTORICAL_PARENT_ROOT");
    if (!configured || !*configured) throw std::runtime_error("explicit MITYANE_HISTORICAL_PARENT_ROOT required for saved parent access");
    const fs::path value(configured);
    if (!value.is_absolute() || value.lexically_normal() != value || !fs::is_directory(value) ||
        fs::is_symlink(value) || fs::canonical(value) != value)
      throw std::runtime_error("historical parent root must be an existing canonical directory");
    for (auto ancestor = value; !ancestor.empty(); ancestor = ancestor.parent_path()) {
      if (fs::is_symlink(ancestor)) throw std::runtime_error("historical parent root has a redirected ancestor");
      if (ancestor == ancestor.parent_path()) break;
    }
    return value;
  }();
  return root;
}
inline fs::path physical(const fs::path &identity) {
  const std::string prefix = "output/runs/", name = identity.generic_string();
  if (name.rfind(prefix, 0) != 0) throw std::runtime_error("historical identity must use the exact output/runs/ prefix");
  const fs::path relative(name.substr(prefix.size()));
  if (relative.empty() || relative.is_absolute()) throw std::runtime_error("missing historical parent identity");
  for (const auto &part : relative) if (part == "." || part == "..")
    throw std::runtime_error("historical identity contains a path redirect");
  return (parent_root() / identity).lexically_normal();
}
inline std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
inline std::string ledger() {
  return "{\"format\":\"historical-parent-path-mapping-v1\",\"historical_prefix\":\"output/runs/\",\"physical_parent_root\":" +
    quote(parent_root().generic_string()) + "}\n";
}
inline void write_ledger(const fs::path &filename) {
  const auto value = ledger();
  const int descriptor = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  if (descriptor < 0) throw std::runtime_error("historical path ledger output must be new");
  size_t offset = 0;
  while (offset < value.size()) {
    const auto count = ::write(descriptor, value.data() + offset, value.size() - offset);
    if (count <= 0) { ::close(descriptor); throw std::runtime_error("historical path ledger write failed"); }
    offset += static_cast<size_t>(count);
  }
  const int synced = ::fsync(descriptor), closed = ::close(descriptor);
  if (synced || closed) throw std::runtime_error("historical path ledger fsync failed");
  const int directory = ::open(filename.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
  if (directory < 0) throw std::runtime_error("historical path ledger parent unavailable");
  const int parent_synced = ::fsync(directory), parent_closed = ::close(directory);
  if (parent_synced || parent_closed) throw std::runtime_error("historical path ledger parent fsync failed");
}
} // namespace embedding::evaluation::historical_paths
