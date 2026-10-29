// SPDX-License-Identifier: MIT
#pragma once

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include "memory_mapped_error.h"

namespace mityane::data::detail {

// Serialize the complete raw/normalized writer chain. Keep the lock inode:
// unlinking it can let a new writer bypass a waiter holding the old inode.
class cache_write_lock {
  int fd_{-1};
public:
  explicit cache_write_lock(const std::string &raw_path) {
    const auto path = raw_path + ".lock";
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR);
    if (fd_ == -1)
      throw_memory_mapped_errno_error({.where = "cache_write_lock", .file = path,
                                     .reason = "could not open cache writer lock"}, errno);
    struct stat status{};
    if (::fstat(fd_, &status) == -1) {
      const auto saved_errno = errno; ::close(fd_); fd_ = -1;
      throw_memory_mapped_errno_error({.where = "cache_write_lock", .file = path,
                                     .reason = "could not inspect cache writer lock"}, saved_errno);
    }
    if (!S_ISREG(status.st_mode)) {
      ::close(fd_); fd_ = -1;
      throw_memory_mapped_error({.where = "cache_write_lock", .file = path,
                               .reason = "cache writer lock must be a regular file"});
    }
    while (::flock(fd_, LOCK_EX) == -1) {
      if (errno == EINTR) continue;
      const auto saved_errno = errno; ::close(fd_); fd_ = -1;
      throw_memory_mapped_errno_error({.where = "cache_write_lock", .file = path,
                                     .reason = "could not acquire cache writer lock"}, saved_errno);
    }
  }
  cache_write_lock(const cache_write_lock &) = delete;
  cache_write_lock &operator=(const cache_write_lock &) = delete;
  ~cache_write_lock() { if (fd_ != -1) ::close(fd_); }
};

// Stage beside the destination so POSIX rename publishes a complete new inode.
// Existing mmap readers retain their old inode instead of observing truncation.
class cache_staging_file {
  std::string destination_, path_;
public:
  explicit cache_staging_file(const std::string &destination) : destination_(destination) {
    auto pattern = destination + ".tmp.XXXXXX";
    std::vector<char> bytes(pattern.begin(), pattern.end()); bytes.push_back('\0');
    const auto fd = ::mkstemp(bytes.data());
    if (fd == -1)
      throw_memory_mapped_errno_error({.where = "cache_staging_file", .file = destination,
                                     .reason = "could not create cache staging file"}, errno);
    path_ = bytes.data();
    if (::close(fd) == -1) {
      const auto saved_errno = errno; ::unlink(path_.c_str());
      throw_memory_mapped_errno_error({.where = "cache_staging_file", .file = path_,
                                     .reason = "could not close cache staging file"}, saved_errno);
    }
  }
  cache_staging_file(const cache_staging_file &) = delete;
  cache_staging_file &operator=(const cache_staging_file &) = delete;
  const std::string &path() const { return path_; }
  void publish() {
    std::error_code error;
    std::filesystem::rename(path_, destination_, error);
    if (error)
      throw_memory_mapped_error({.where = "cache_staging_file::publish", .file = destination_,
                               .reason = "atomic cache publication failed: " + error.message()});
    path_.clear();
  }
  ~cache_staging_file() { if (!path_.empty()) { std::error_code error; std::filesystem::remove(path_, error); } }
};

} // namespace mityane::data::detail
