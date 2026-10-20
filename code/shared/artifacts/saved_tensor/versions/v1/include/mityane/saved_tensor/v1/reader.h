#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace mityane::saved_tensor::v1 {

class DecodeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class Role { training, validation, engineering_fixture, test };
enum class DType { float64, float32, int64, boolean, byte };

struct Limits {
  std::uint64_t archive_bytes = 8ULL << 30;
  std::uint64_t member_bytes = 1ULL << 30;
  std::uint64_t pickle_bytes = 32ULL << 20;
  std::uint64_t tensor_elements = 16ULL << 20;
  std::uint64_t nodes = 1ULL << 20;
  std::uint64_t opcodes = 8ULL << 20;
  std::uint64_t members = 100000;
  std::size_t depth = 128;
};

// A caller supplies an already admitted artifact role and an explicit root.
// This codec is not an experiment admission mechanism or a hash verifier.
struct ReadPolicy {
  Role role = Role::engineering_fixture;
  std::filesystem::path allowed_root;
  std::set<std::string> selected_fields;
  bool permit_legacy_missing_byteorder = false;
  Limits limits;
};

struct TensorInfo {
  DType dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides;
  std::int64_t storage_offset = 0;
  std::string storage_key;
  std::int64_t storage_elements = 0;
};

struct Tensor {
  DType dtype;
  std::vector<std::int64_t> shape;
  // Logical row-major, little-endian bytes, including original NaN/signed-zero bits.
  std::vector<std::uint8_t> raw;
  std::vector<double> floating() const;
  std::vector<std::int64_t> integers() const;
  std::string utf8() const;
};

// No Torch, model loading, Python interpreter or general pickle execution.
// Metadata is parsed at open; only read(field) materializes a tensor storage.
class Archive {
 public:
  Archive(const std::filesystem::path& path, ReadPolicy policy);
  ~Archive();
  Archive(Archive&&) noexcept;
  Archive& operator=(Archive&&) noexcept;
  Archive(const Archive&) = delete;
  Archive& operator=(const Archive&) = delete;
  std::vector<std::string> fields() const;
  const TensorInfo& info(const std::string& field) const;
  Tensor read(const std::string& field) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::size_t element_bytes(DType dtype);

}  // namespace mityane::saved_tensor::v1
