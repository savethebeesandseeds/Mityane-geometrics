// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/fixed_prior_temporal_relation.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "rpb_test_support.h"
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

#ifndef FIXED_PRIOR_RELATION_SCREEN_SOURCE_ID
#define FIXED_PRIOR_RELATION_SCREEN_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;

rpb::Config configuration() {
  rpb::Config c; c.device = torch::kCUDA; c.channel_ids = {101, 202, 303};
  c.channel_mixer_layers = 1; c.channel_mixer_placement = 1; c.global_bottleneck_mode = 2;
  return c;
}
void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &why) {
  check(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() && torch::equal(a, b), why);
}
void initial_contract(rpb::PartitionedTemporalRelationModel &v16, rpb::FixedPriorTemporalRelationModel &v17) {
  const auto a = v16->named_parameters(), b = v17->named_parameters();
  check(a.keys() == b.keys(), "v16/v17 parameter registration order exact");
  int64_t registered = 0, trainable = 0, frozen = 0;
  for (const auto &p : b) {
    exact(p.value(), a[p.key()], "all initial v16/v17 values exact");
    check(p.value().is_cuda(), "all registered model values actual CUDA"); registered += p.value().numel();
    if (p.key() == "grouped_odd_relation_projection.weight") {
      check(!p.value().requires_grad() && p.value().sizes() == torch::IntArrayRef({3, 4, 36}) &&
            !p.value().grad().defined(), "only bias-free grouped432 frozen"); frozen += p.value().numel();
    } else { check(p.value().requires_grad(), "shape/backbone/decoder remain trainable"); trainable += p.value().numel(); }
  }
  check(registered == 226877 && trainable == 226445 && frozen == 432 &&
        registered == rpb::kFixedPriorTemporalRelationParameterCount &&
        trainable == rpb::kFixedPriorTemporalRelationTrainableParameterCount &&
        frozen == rpb::kFixedPriorTemporalRelationFrozenParameterCount, "exact registered/trainable/frozen counts");
  const auto ab = v16->named_buffers(), bb = v17->named_buffers(); check(ab.keys() == bb.keys(), "buffer registration order exact");
  for (const auto &p : ab) exact(p.value(), bb[p.key()], "initial buffers exact");
  v17->validate_fixed_prior();
  const auto w = v17->grouped_odd_relation_projection->weight.to(torch::kCPU).contiguous();
  auto values = w.accessor<float, 3>(); int64_t nonzero = 0;
  for (int64_t pair = 0; pair < 3; ++pair)
    for (int64_t row = 0; row < 4; ++row)
      for (int64_t column = 0; column < 36; ++column) {
        check(values[pair][row][column] == (column % 4 == row ? 1.f / 9.f : 0.f), "all pairs/features/spacings literal prior");
        nonzero += values[pair][row][column] != 0;
      }
  check(nonzero == 108, "nine feature pairs at each of all12 pair/spacings");
}
void cuda_contract() {
  const auto c = configuration(); auto raw = input(c, 3);
  raw.observed[0][2].narrow(0, 13, 3).fill_(false);
  raw.data = torch::where(raw.observed, raw.data, torch::zeros_like(raw.data));
  const auto scaler = rpb::fit_scaler(raw, c);
  const auto scaler_mean = scaler.mean.clone(), scaler_scale = scaler.scale.clone();
  const auto x = scaler.transform(raw, c);
  torch::manual_seed(17101); rpb::PartitionedTemporalRelationModel v16(c);
  const auto cpu_rng = torch::rand({8}), cuda_rng = torch::rand({8}, torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(17101); rpb::FixedPriorTemporalRelationModel v17(c);
  exact(cpu_rng, torch::rand({8}), "v17 preserves exact v16 CPU initializer RNG");
  exact(cuda_rng, torch::rand({8}, torch::TensorOptions().device(torch::kCUDA)), "v17 preserves exact v16 CUDA initializer RNG");
  v16->eval(); v17->eval(); initial_contract(v16, v17);
  const auto before = v17->encode(x); const auto v16_initial = v16->encode(x);
  exact(before.z_contextual_global, v16_initial.z_contextual_global, "initial native32 identical to v16");
  exact(before.z_local, v16_initial.z_local, "independent local diagnostic identical");
  const auto odd_initial = before.z_contextual_global.narrow(1, 20, 12).detach().clone();
  const auto prior_initial = v17->grouped_odd_relation_projection->weight.detach().clone();
  const auto shape_initial = v17->native_shape_projection->weight.detach().clone();
  const auto backbone_weight = v17->named_parameters()["backbone.patch_projection.weight"];
  const auto decoder_weight = v17->named_parameters()["backbone.decoder_first.weight"];
  const auto backbone_initial = backbone_weight.detach().clone(), decoder_initial = decoder_weight.detach().clone();
  const auto q = rpb::make_training_mask(x.observed, c, 17103);
  const auto old_q = v16->forward(x, q.hidden), new_q = v17->forward(x, q.hidden);
  exact(new_q.encoding.z_contextual_global, old_q.encoding.z_contextual_global, "initial original-Q native parity");
  exact(new_q.reconstruction, old_q.reconstruction, "initial original-Q prediction parity");
  exact(new_q.target_counts, old_q.target_counts, "original query support unchanged");
  exact(new_q.reconstruction, v17->decode(new_q.encoding.z_contextual_global, x.channel_ids), "sole original decoder receives concatenated32");
  auto changed = x; changed.data = torch::where(q.target, x.data + 1000., x.data);
  exact(v17->forward(changed, q.hidden).encoding.z_contextual_global, new_q.encoding.z_contextual_global, "hidden targets cannot enter fixed prior or shape");
  auto poison = x; poison.data = torch::where(x.observed, x.data, torch::full_like(x.data, std::numeric_limits<float>::quiet_NaN()));
  exact(v17->encode(poison).z_contextual_global, before.z_contextual_global, "naturally hidden NaNs isolated");
  torch::optim::AdamW optimizer(v17->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for (int64_t step = 0; step < 4; ++step) {
    const auto masks = rpb::make_training_mask(x.observed, c, 17110 + step);
    const auto context = rpb::context_deletion::make_plan(masks, x.channel_ids, c, 17107, step, rpb::ContextDeletionRecipe::coordinate15_v1);
    const rpb::Input visible{torch::where(context.visible, x.data, torch::zeros_like(x.data)), context.visible, x.channel_ids, x.endpoints, x.sampling_interval};
    optimizer.zero_grad(); const auto e = v17->encode(visible); const auto z = rpb::compact_reconstruction_export(e, c);
    const auto prediction = v17->decode(z, x.channel_ids);
    const auto loss = rpb::hierarchical_huber(prediction, x.data.detach(), masks.target, masks.eligible_channels, 1.);
    check(z.sizes() == torch::IntArrayRef({3, 32}) && z.is_cuda() && loss.loss.is_cuda() && torch::isfinite(loss.loss).item<bool>(), "CUDA original waveform Huber1 through sole32");
    loss.loss.backward();
    check(!v17->grouped_odd_relation_projection->weight.grad().defined(), "frozen odd weight has no gradient");
    for (const auto &w : {v17->native_shape_projection->weight, backbone_weight, decoder_weight}) {
      finite(w.grad(), "trainable shape/backbone/decoder gradient finite CUDA");
      check(w.grad().is_cuda() && w.grad().abs().sum().item<double>() > 0, "trainable graph still learns from original loss");
    }
    torch::nn::utils::clip_grad_norm_(v17->parameters(), 1., 2., true); optimizer.step(); v17->validate_fixed_prior();
    exact(v17->grouped_odd_relation_projection->weight, prior_initial, "frozen432 unchanged after each AdamW step including weight decay");
    exact(v17->encode(x).z_contextual_global.narrow(1, 20, 12), odd_initial, "odd coordinates exactly preserved on unchanged context");
  }
  check(!torch::equal(v17->native_shape_projection->weight, shape_initial) && !torch::equal(backbone_weight, backbone_initial) &&
        !torch::equal(decoder_weight, decoder_initial), "four CUDA updates change shape/backbone/decoder");
  check(optimizer.state().find(v17->grouped_odd_relation_projection->weight.unsafeGetTensorImpl()) == optimizer.state().end(), "no fabricated frozen odd AdamW state");
  for (const auto &w : {v17->native_shape_projection->weight, backbone_weight, decoder_weight}) {
    const auto it = optimizer.state().find(w.unsafeGetTensorImpl()); check(it != optimizer.state().end(), "actual trainable AdamW state present");
    const auto *adam = dynamic_cast<const torch::optim::AdamWParamState *>(it->second.get());
    check(adam && adam->step() == 4 && adam->exp_avg().is_cuda(), "actual four-step CUDA moments");
  }
  exact(scaler.mean, scaler_mean, "original TRAIN scaler mean unchanged"); exact(scaler.scale, scaler_scale, "original TRAIN scaler scale unchanged");
  {
    torch::NoGradGuard guard; auto reverse = x; reverse.data = x.data.flip({2}); reverse.observed = x.observed.flip({2});
    close(v17->encode(reverse).z_contextual_global.narrow(1, 20, 12), -odd_initial, "fixed odd partition reverses sign", 4e-6, 1e-6);
    const auto order = torch::tensor({2, 0, 1}, torch::kInt64).to(torch::kCUDA); auto perm = x;
    perm.data = x.data.index_select(1, order); perm.observed = x.observed.index_select(1, order); perm.channel_ids = x.channel_ids.to(torch::kCUDA).index_select(0, order);
    close(v17->encode(perm).z_contextual_global, v17->encode(x).z_contextual_global, "semantic permutation native equality", 2e-5, 2e-6);
    auto absent = poison; absent.observed = torch::zeros_like(x.observed); const auto empty = v17->encode(absent);
    check(!empty.sample_valid_mask.any().item<bool>(), "all absent remains invalid"); exact(empty.z_contextual_global, torch::zeros_like(empty.z_contextual_global), "all-absent native32 zero");
  }
  torch::Tensor saved_native;
  { torch::NoGradGuard guard; saved_native = v17->encode(x).z_contextual_global.detach().clone(); }
  std::stringstream bytes; torch::serialize::OutputArchive output; v17->save(output); output.save_to(bytes);
  rpb::FixedPriorTemporalRelationModel restored(c);
  restored->grouped_odd_relation_projection->weight.set_requires_grad(true);
  std::shared_ptr<torch::nn::Module> erased = restored.ptr(); torch::serialize::InputArchive archive; archive.load_from(bytes, torch::kCUDA); erased->load(archive);
  restored->eval(); restored->validate_fixed_prior();
  check(!restored->grouped_odd_relation_projection->weight.requires_grad(), "type-erased load restores frozen flag");
  { torch::NoGradGuard guard; exact(restored->encode(x).z_contextual_global, saved_native, "trained CUDA save/load exact native32"); }
  const auto original_parameters = v17->named_parameters(), restored_parameters = restored->named_parameters();
  check(original_parameters.keys() == restored_parameters.keys(), "save/load named association exact");
  for (const auto &p : original_parameters) exact(p.value(), restored_parameters[p.key()], "save/load parameter exact");
  rpb::FixedPriorTemporalRelationModel corrupted(c);
  { torch::NoGradGuard guard; corrupted->grouped_odd_relation_projection->weight[0][0][0] = 0; }
  std::stringstream bad_bytes; torch::serialize::OutputArchive bad_output; corrupted->save(bad_output); bad_output.save_to(bad_bytes);
  torch::serialize::InputArchive bad_archive; bad_archive.load_from(bad_bytes, torch::kCUDA);
  rejects([&] { restored->load(bad_archive); }, "changed checkpoint prior cannot silently load");
}
} // namespace

int main() try {
  check(torch::cuda::is_available(), "fixed prior admission requires actual CUDA"); torch::set_num_threads(1); cuda_contract();
  std::cout << "Fixed prior temporal relation CUDA admission passed\n" << FIXED_PRIOR_RELATION_SCREEN_SOURCE_ID << '\n'; return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
