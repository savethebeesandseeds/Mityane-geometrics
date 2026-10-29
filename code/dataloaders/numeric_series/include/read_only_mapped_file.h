// Extracted from cuwacunu_torch; see code/data/README.md for provenance.
// Shared ownership of the loader's read-only POSIX mapping; no record decoding.
#pragma once

#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "memory_mapped_error.h"

namespace mityane::data {

class ReadOnlyMappedFile {
  int fd_{-1};
  void* data_ptr_{nullptr};
  std::size_t file_size_{0};

  void map_(const std::string& bin_filename) {
    struct stat st {};
    if (fstat(fd_, &st) == -1) {
      const int saved_errno = errno;
      close(fd_);
      fd_ = -1;
      throw_memory_mapped_errno_error(
          {.where = "MemoryMappedDataset::MappedData::fstat",
           .file = bin_filename,
           .reason = "failed to determine file size"}, saved_errno);
    }
    file_size_ = static_cast<std::size_t>(st.st_size);
    if (file_size_ == 0) {
      close(fd_);
      fd_ = -1;
      throw_memory_mapped_error({.where = "MemoryMappedDataset::MappedData",
                                .file = bin_filename,
                                .reason = "binary file is empty"});
    }

    data_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ptr_ == MAP_FAILED) {
      const int saved_errno = errno;
      data_ptr_ = nullptr;
      close(fd_);
      fd_ = -1;
      throw_memory_mapped_errno_error(
          {.where = "MemoryMappedDataset::MappedData::mmap",
           .file = bin_filename,
           .reason = "failed to memory-map binary file"}, saved_errno);
    }
  }

public:
  explicit ReadOnlyMappedFile(const std::string& bin_filename) {
    fd_ = open(bin_filename.c_str(), O_RDONLY);
    if (fd_ == -1) {
      const int saved_errno = errno;
      throw_memory_mapped_errno_error(
          {.where = "MemoryMappedDataset::MappedData::open",
           .file = bin_filename,
           .reason = "could not open binary file"}, saved_errno);
    }
    map_(bin_filename);
  }

  // The caller retains its already admitted descriptor. This mapping owns only
  // an independent read-only duplicate and never reopens the supplied path.
  ReadOnlyMappedFile(int borrowed_fd, const std::string& name) {
    const int flags = fcntl(borrowed_fd,F_GETFL);
    if (flags == -1) {
      const int saved_errno = errno;
      throw_memory_mapped_errno_error({.where = "ReadOnlyMappedFile::descriptor",.file = name,
        .reason = "could not inspect borrowed descriptor"},saved_errno);
    }
    if ((flags&O_ACCMODE) != O_RDONLY)
      throw_memory_mapped_error({.where = "ReadOnlyMappedFile::descriptor",.file = name,
        .reason = "borrowed descriptor must be read-only"});
    fd_ = fcntl(borrowed_fd,F_DUPFD_CLOEXEC,0);
    if (fd_ == -1) {
      const int saved_errno = errno;
      throw_memory_mapped_errno_error({.where = "ReadOnlyMappedFile::dup",.file = name,
        .reason = "could not duplicate read-only descriptor"},saved_errno);
    }
    map_(name);
  }

  ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;

  ~ReadOnlyMappedFile() {
    if (data_ptr_ && data_ptr_ != MAP_FAILED) munmap(data_ptr_, file_size_);
    if (fd_ != -1) close(fd_);
  }

  const void* data() const noexcept { return data_ptr_; }
  std::size_t size() const noexcept { return file_size_; }
  int descriptor() const noexcept { return fd_; }
};

} // namespace mityane::data
