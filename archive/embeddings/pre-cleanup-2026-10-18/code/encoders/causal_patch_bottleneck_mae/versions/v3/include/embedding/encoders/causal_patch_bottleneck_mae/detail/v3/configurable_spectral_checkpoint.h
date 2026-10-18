// SPDX-License-Identifier: MIT
// CPB-v3 namespace/include and causal adaptations Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/configurable_spectral_model.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/configurable_spectral_checkpoint_io.h"
#include <string>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3 {

inline constexpr const char *kConfigurableSpectralCheckpointKind = "rpb_configurable_inference_checkpoint_v1";

namespace configurable_checkpoint_detail {
inline torch::Tensor text_tensor(const std::string &text) {
  return torch::tensor(std::vector<uint8_t>(text.begin(), text.end()), torch::kUInt8);
}
inline std::string read_text(torch::serialize::InputArchive &archive, const char *key) {
  torch::Tensor t;
  archive.read(key, t, true);
  TORCH_CHECK(t.scalar_type() == torch::kUInt8 && t.dim() == 1,
              "[rpb-v20 checkpoint] malformed text: ", key);
  t = t.to(torch::kCPU).contiguous();
  if (!t.numel()) return {};
  return std::string(reinterpret_cast<const char *>(t.const_data_ptr<uint8_t>()), t.numel());
}
inline std::vector<int64_t> integers(const Config &c) {
  return {c.channel_count, c.history_length, c.input_width, c.patch_length,
      c.encoder_width, c.export_width, c.num_layers, c.num_heads,
      c.channel_mixer_layers, c.channel_mixer_placement, c.global_bottleneck_mode,
      c.global_pool_input_source, c.temporal_difference_input,
      c.feedforward_width, c.decoder_hidden_width};
}
inline std::vector<double> numbers(const Config &c) {
  return {c.dropout, c.layer_norm_epsilon, c.mask_ratio, c.huber_delta,
      c.scale_floor, c.sampling_interval};
}
inline torch::Tensor read_vector(torch::serialize::InputArchive &a, const char *key,
                                 torch::Dtype dtype, int64_t length) {
  torch::Tensor t;
  a.read(key, t, true);
  TORCH_CHECK(t.scalar_type() == dtype && t.dim() == 1 && t.size(0) == length,
              "[rpb-v20 checkpoint] malformed config vector: ", key);
  return t.to(torch::kCPU).contiguous();
}
inline Config read_config(torch::serialize::InputArchive &a, const torch::Device &device) {
  const auto i = read_vector(a, "config_integer_v1", torch::kInt64, 15);
  const auto n = read_vector(a, "config_number_v1", torch::kFloat64, 6);
  const auto x = i.accessor<int64_t, 1>();
  const auto y = n.accessor<double, 1>();
  Config c;
  c.channel_count=x[0]; c.history_length=x[1]; c.input_width=x[2]; c.patch_length=x[3];
  c.encoder_width=x[4]; c.export_width=x[5]; c.num_layers=x[6]; c.num_heads=x[7];
  c.channel_mixer_layers=x[8]; c.channel_mixer_placement=x[9]; c.global_bottleneck_mode=x[10];
  c.global_pool_input_source=x[11]; c.temporal_difference_input=x[12];
  c.feedforward_width=x[13]; c.decoder_hidden_width=x[14];
  c.dropout=y[0]; c.layer_norm_epsilon=y[1]; c.mask_ratio=y[2]; c.huber_delta=y[3];
  c.scale_floor=y[4]; c.sampling_interval=y[5];
  c.device=device; c.dtype=torch::kFloat32;
  validate_configurable_spectral_config(c);
  const auto ids = read_vector(a, "channel_ids", torch::kInt64, c.channel_count);
  const auto ptr = ids.const_data_ptr<int64_t>();
  c.channel_ids.assign(ptr, ptr+c.channel_count);
  validate_configurable_spectral_config(c);
  return c;
}
inline bool same_config(const Config &a, const Config &b) {
  return integers(a) == integers(b) && numbers(a) == numbers(b) &&
      resolved_channel_ids(a) == resolved_channel_ids(b) && a.dtype == b.dtype;
}
} // namespace configurable_checkpoint_detail

struct LoadedConfigurableSpectralEncoder {
  Config config;
  ConfigurableSpectralModel model{nullptr};
  FrozenScaler scaler;
  int64_t training_updates{0};
};

// Model/scaler export for inference. Optimizer, RNG and data-split provenance
// belong to a training run artifact; this format does not claim resume support.
inline void save_configurable_spectral_checkpoint(const std::string &path,
    const ConfigurableSpectralModel &model, const FrozenScaler &scaler, int64_t updates) {
  TORCH_CHECK(model && updates >= 0, "[rpb-v20 checkpoint] model and nonnegative updates required");
  const auto &c = model->config();
  validate_configurable_spectral_config(c);
  model->validate_state();
  scaler.validate(configurable_spectral_input_config(c));
  namespace d = configurable_checkpoint_detail;
  torch::serialize::OutputArchive a, m, s;
  a.write("artifact_kind", d::text_tensor(kConfigurableSpectralCheckpointKind), true);
  a.write("model_tag", d::text_tensor(kConfigurableSpectralModelTag), true);
  a.write("architecture_id", d::text_tensor(kConfigurableSpectralArchitectureId), true);
  a.write("native_layout", d::text_tensor(kConfigurableSpectralLayout), true);
  a.write("dtype", d::text_tensor("float32"), true);
  a.write("training_device", d::text_tensor(c.device.str()), true);
  a.write("config_integer_v1", torch::tensor(d::integers(c), torch::kInt64), true);
  a.write("config_number_v1", torch::tensor(d::numbers(c), torch::kFloat64), true);
  a.write("channel_ids", torch::tensor(resolved_channel_ids(c), torch::kInt64), true);
  a.write("training_updates", torch::tensor(updates, torch::kInt64), true);
  a.write("scaler_identity", d::text_tensor(scaler.identity()), true);
  model->save(m); scaler.save(s);
  a.write("model", m); a.write("scaler", s);
  d::publish_checkpoint(a, path);
}

inline LoadedConfigurableSpectralEncoder load_configurable_spectral_checkpoint(
    const std::string &path, torch::Device device = torch::Device(torch::kCUDA, 0),
    const Config *expected = nullptr) {
  TORCH_CHECK(device.is_cuda(), "[rpb-v20 checkpoint] CUDA target device required");
  namespace d = configurable_checkpoint_detail;
  torch::serialize::InputArchive a;
  a.load_from(path, device);
  TORCH_CHECK(d::read_text(a, "artifact_kind") == kConfigurableSpectralCheckpointKind &&
      d::read_text(a, "model_tag") == kConfigurableSpectralModelTag &&
      d::read_text(a, "architecture_id") == kConfigurableSpectralArchitectureId &&
      d::read_text(a, "native_layout") == kConfigurableSpectralLayout &&
      d::read_text(a, "dtype") == "float32" &&
      torch::Device(d::read_text(a, "training_device")).is_cuda(),
      "[rpb-v20 checkpoint] schema/tag/architecture/layout/device mismatch");
  auto c = d::read_config(a, device);
  if (expected) {
    validate_configurable_spectral_config(*expected);
    TORCH_CHECK(d::same_config(c, *expected), "[rpb-v20 checkpoint] requested configuration mismatch");
  }
  torch::Tensor updates;
  a.read("training_updates", updates, true);
  TORCH_CHECK(updates.scalar_type() == torch::kInt64 && updates.dim() == 0 && updates.item<int64_t>() >= 0,
              "[rpb-v20 checkpoint] malformed training update count");
  LoadedConfigurableSpectralEncoder out{c, ConfigurableSpectralModel(c), {}, updates.item<int64_t>()};
  torch::serialize::InputArchive m, s;
  a.read("model", m); a.read("scaler", s);
  out.model->load(m);
  out.scaler = FrozenScaler::load(s);
  out.scaler.validate(configurable_spectral_input_config(c));
  TORCH_CHECK(out.scaler.identity() == d::read_text(a, "scaler_identity"),
              "[rpb-v20 checkpoint] scaler content identity mismatch");
  out.model->eval();
  return out;
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3
