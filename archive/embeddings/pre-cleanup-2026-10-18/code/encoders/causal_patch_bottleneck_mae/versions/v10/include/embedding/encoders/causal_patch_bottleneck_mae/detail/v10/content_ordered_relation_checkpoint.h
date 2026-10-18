// SPDX-License-Identifier: MIT
// CPB-v10 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v10/configurable_spectral_checkpoint.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v10/content_ordered_relation_model.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10 {

inline constexpr const char *kContentOrderedRelationCheckpointKind =
    "rpb_content_ordered_relation_inference_checkpoint_v1";

struct LoadedContentOrderedRelationEncoder {
  Config config;
  ContentOrderedRelationModel model{nullptr};
  FrozenScaler scaler;
  int64_t training_updates{0};
};

inline void save_content_ordered_relation_checkpoint(const std::string &path,
    const ContentOrderedRelationModel &model, const FrozenScaler &scaler, int64_t updates) {
  TORCH_CHECK(model && updates >= 0, "[rpb-v22 checkpoint] model and nonnegative update count required");
  const auto &c = model->config();
  validate_content_ordered_relation_config(c);
  model->validate_state();
  scaler.validate(configurable_spectral_input_config(c));
  namespace d = configurable_checkpoint_detail;
  torch::serialize::OutputArchive a, m, s;
  a.write("artifact_kind", d::text_tensor(kContentOrderedRelationCheckpointKind), true);
  a.write("model_tag", d::text_tensor(kContentOrderedRelationModelTag), true);
  a.write("architecture_id", d::text_tensor(kContentOrderedRelationArchitectureId), true);
  a.write("native_layout", d::text_tensor(kContentOrderedRelationLayout), true);
  a.write("dtype", d::text_tensor("float32"), true);
  a.write("training_device", d::text_tensor(c.device.str()), true);
  a.write("config_integer_v1", torch::tensor(d::integers(c), torch::kInt64), true);
  a.write("config_number_v1", torch::tensor(d::numbers(c), torch::kFloat64), true);
  a.write("channel_ids", torch::tensor(resolved_channel_ids(c), torch::kInt64), true);
  a.write("relation_lags_v1", torch::tensor(std::vector<int64_t>(
      kContentOrderedRelationLags.begin(), kContentOrderedRelationLags.end()), torch::kInt64), true);
  a.write("relation_energy_floor", torch::tensor(kContentOrderedRelationEnergyFloor, torch::kFloat64), true);
  a.write("relation_minimum_common", torch::tensor(kContentOrderedRelationMinimumCommon, torch::kInt64), true);
  a.write("training_updates", torch::tensor(updates, torch::kInt64), true);
  a.write("scaler_identity", d::text_tensor(scaler.identity()), true);
  model->save(m); scaler.save(s);
  a.write("model", m); a.write("scaler", s);
  d::publish_checkpoint(a, path);
}

inline LoadedContentOrderedRelationEncoder load_content_ordered_relation_checkpoint(
    const std::string &path, torch::Device device = torch::Device(torch::kCUDA, 0),
    const Config *expected = nullptr) {
  TORCH_CHECK(device.is_cuda(), "[rpb-v22 checkpoint] CUDA target device required");
  namespace d = configurable_checkpoint_detail;
  torch::serialize::InputArchive a;
  a.load_from(path, device);
  TORCH_CHECK(d::read_text(a, "artifact_kind") == kContentOrderedRelationCheckpointKind &&
                  d::read_text(a, "model_tag") == kContentOrderedRelationModelTag &&
                  d::read_text(a, "architecture_id") == kContentOrderedRelationArchitectureId &&
                  d::read_text(a, "native_layout") == kContentOrderedRelationLayout &&
                  d::read_text(a, "dtype") == "float32" &&
                  torch::Device(d::read_text(a, "training_device")).is_cuda(),
              "[rpb-v22 checkpoint] schema/tag/architecture/layout/device mismatch");
  auto c = d::read_config(a, device);
  validate_content_ordered_relation_config(c);
  if (expected) {
    validate_content_ordered_relation_config(*expected);
    TORCH_CHECK(d::same_config(c, *expected), "[rpb-v22 checkpoint] requested configuration mismatch");
  }
  const auto lags = d::read_vector(a, "relation_lags_v1", torch::kInt64, 4);
  const auto ptr = lags.const_data_ptr<int64_t>();
  TORCH_CHECK(std::equal(kContentOrderedRelationLags.begin(), kContentOrderedRelationLags.end(), ptr),
              "[rpb-v22 checkpoint] relation lag contract mismatch");
  torch::Tensor floor, minimum, updates;
  a.read("relation_energy_floor", floor, true);
  a.read("relation_minimum_common", minimum, true);
  a.read("training_updates", updates, true);
  TORCH_CHECK(floor.scalar_type() == torch::kFloat64 && floor.dim() == 0 &&
                  floor.item<double>() == kContentOrderedRelationEnergyFloor &&
                  minimum.scalar_type() == torch::kInt64 && minimum.dim() == 0 &&
                  minimum.item<int64_t>() == kContentOrderedRelationMinimumCommon,
              "[rpb-v22 checkpoint] relation support/energy contract mismatch");
  TORCH_CHECK(updates.scalar_type() == torch::kInt64 && updates.dim() == 0 && updates.item<int64_t>() >= 0,
              "[rpb-v22 checkpoint] malformed update count");
  LoadedContentOrderedRelationEncoder out{c, ContentOrderedRelationModel(c), {}, updates.item<int64_t>()};
  torch::serialize::InputArchive m, s;
  a.read("model", m); a.read("scaler", s);
  out.model->load(m);
  out.scaler = FrozenScaler::load(s);
  out.scaler.validate(configurable_spectral_input_config(c));
  TORCH_CHECK(out.scaler.identity() == d::read_text(a, "scaler_identity"),
              "[rpb-v22 checkpoint] scaler identity mismatch");
  out.model->eval();
  return out;
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v10
