// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/mtf_jepa_mae_vicreg/workflow.h"
#include <memory>
#include <vector>

namespace embedding::encoders::mtf_jepa_mae_vicreg::family {

inline constexpr const char *kPreparation = "identity-cpu-f32-observed-v1";
inline constexpr const char *kReplacementSampling = "replacement-step-seeded-v1";
inline constexpr const char *kFullPartitionSampling = "full-partition-step-seeded-v1";
inline constexpr const char *kSeedAlgorithm = "portable-absolute-step-seed-plus-one-v1";
inline constexpr const char *kTrainingKind = "mtf-family-training-v1";
inline constexpr const char *kInferenceKind = "mtf-family-inference-v1";

// Fixed-budget portable recipe. Source/data declarations bind numerical state;
// they are not authenticated application publication or data admission.
struct Recipe {
  Settings settings{default_settings()};
  std::string source_sha256;
  std::string dataset_sha256;
  std::string input_source_identity;
  std::string split_identity;
  std::string recipe_identity;
  std::string sampling{kReplacementSampling};
  double adam_beta1{0.9};
  double adam_beta2{0.999};
  double adam_epsilon{1e-8};
  bool adam_amsgrad{false};
};

// CPU floating BCHF with explicit matching CPU bool support. Observed values
// must survive F32 conversion; missing values are zeroed. Returned storage is
// cloned. There is no fitted scaler. Sessions require every TRAIN row observed.
::embedding::Batch prepare(const ::embedding::Batch &raw, const Config &config);
std::string dataset_identity(const ::embedding::Batch &prepared);
void validate_recipe(const Recipe &recipe);
std::string recipe_sha256(const Recipe &recipe);
std::string runtime_identity();
std::string file_sha256(const std::string &path);

struct ParameterRecord {
  std::string name;
  std::vector<int64_t> shape;
  int64_t elements{0};
  bool trainable{false};
};
struct ParameterInventory {
  std::vector<ParameterRecord> records;
  int64_t total{0};
  int64_t trainable{0};
  int64_t frozen{0};
};
ParameterInventory parameter_inventory(const Model &model);

struct UpdateResult {
  int64_t completed_updates{0};
  double loss{0.0};
  int64_t target_tokens{0};
};

class Session {
public:
  Session(Session &&) noexcept = default;
  Session &operator=(Session &&) noexcept = default;
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  static Session fresh(const Recipe &recipe, const ::embedding::Batch &observations);
  static Session resume(const std::string &path, const Recipe &expected_recipe,
                        const ::embedding::Batch &observations);
  UpdateResult update();
  int64_t completed_updates() const { return completed_; }
  const Recipe &recipe() const { return recipe_; }
  Model &model() { return model_; }
  const Model &model() const { return model_; }
  torch::optim::AdamW &optimizer() { return *optimizer_; }
  const torch::optim::AdamW &optimizer() const { return *optimizer_; }
  std::string state_sha256() const;
  std::string optimizer_sha256() const;
  const std::string &initial_state_sha256() const { return initial_state_; }
  void save_training(const std::string &path) const;
  void export_inference(const std::string &path) const;

private:
  Session() = default;
  void validate_state() const;
  Recipe recipe_;
  ::embedding::Batch observations_;
  Model model_{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer_;
  int64_t completed_{0};
  std::string initial_state_;
  std::string runtime_;
  std::string schema_;
};

struct InferenceExpectation {
  Config config;
  std::string source_sha256;
  std::string state_sha256;
  std::string preparation{kPreparation};
};
struct Inference {
  Model model{nullptr};
  Config config;
  std::string source_sha256;
  std::string state_sha256;
  std::string preparation;
  int64_t completed_updates{0};
  // Online encode in eval/no-grad mode. Codec contains registered model state
  // (including inactive teacher/heads), no optimizer or resume capability.
  EncodeOutput encode(const ::embedding::Batch &raw);
};
Inference load_inference(const std::string &path, const InferenceExpectation &expected);

} // namespace embedding::encoders::mtf_jepa_mae_vicreg::family
