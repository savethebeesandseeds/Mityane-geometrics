// SPDX-License-Identifier: MIT
// CPB-v2 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include <torch/torch.h>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v2::configurable_checkpoint_detail {

// Linux managed-container runtime. Stage beside the destination so publication
// is on the same filesystem. Never truncate or replace an existing artifact.
class StagedCheckpoint {
public:
  explicit StagedCheckpoint(const std::string &destination) : destination_(destination) {
    const auto target = std::filesystem::path(destination);
    TORCH_CHECK(!target.filename().empty(), "[rpb-v20 checkpoint] filename required");
    auto pattern = (target.parent_path() / (target.filename().string() + ".tmp-XXXXXX")).string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
    fd_ = ::mkstemp(name.data());
    TORCH_CHECK(fd_ >= 0, "[rpb-v20 checkpoint] cannot create staging file: ", std::strerror(errno));
    staged_ = name.data();
  }
  StagedCheckpoint(const StagedCheckpoint &) = delete;
  StagedCheckpoint &operator=(const StagedCheckpoint &) = delete;
  ~StagedCheckpoint() {
    if (fd_ >= 0) ::close(fd_);
    if (!staged_.empty()) ::unlink(staged_.c_str());
  }
  const std::string &path() const { return staged_; }
  void publish() {
    TORCH_CHECK(::fsync(fd_) == 0, "[rpb-v20 checkpoint] cannot sync staged file: ", std::strerror(errno));
    const auto descriptor = fd_; fd_ = -1;
    TORCH_CHECK(::close(descriptor) == 0, "[rpb-v20 checkpoint] cannot close staged file: ", std::strerror(errno));
    if (::syscall(SYS_renameat2, AT_FDCWD, staged_.c_str(), AT_FDCWD,
                  destination_.c_str(), RENAME_NOREPLACE) == 0) {
      staged_.clear();
      return;
    }
    const auto rename_error = errno;
    if (rename_error == ENOSYS || rename_error == EINVAL || rename_error == EOPNOTSUPP) {
      // Docker Desktop Windows binds may support link but not rename flags.
      // link creates one complete inode atomically and refuses any existing
      // destination. The destructor removes our staging name after publication.
      TORCH_CHECK(::link(staged_.c_str(), destination_.c_str()) == 0,
                  "[rpb-v20 checkpoint] cannot publish new checkpoint (existing files are preserved): ",
                  std::strerror(errno));
      return;
    }
    TORCH_CHECK(false,
                "[rpb-v20 checkpoint] cannot publish new checkpoint (existing files are preserved): ",
                std::strerror(rename_error));
  }
private:
  std::string destination_, staged_;
  int fd_{-1};
};

inline void publish_checkpoint(torch::serialize::OutputArchive &archive, const std::string &path) {
  StagedCheckpoint staged(path);
  archive.save_to(staged.path());
  staged.publish();
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v2::configurable_checkpoint_detail
