// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <iostream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ctx = rpb::context_deletion;
using namespace rpb_test;

void counter_support_and_semantics() {
  auto c = config(); auto raw = input(c, 16);
  const auto original = rpb::make_training_mask(raw.observed, c, 37);
  const auto saved_visible = original.visible.clone(), saved_target = original.target.clone();
  auto generator = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  const auto rng_before = generator.get_state().clone();
  const auto plan = ctx::make_plan(original, raw.channel_ids, c, 202, 9);
  const auto explicit30 = ctx::make_plan(original, raw.channel_ids, c, 202, 9,
      rpb::ContextDeletionRecipe::coordinate30_v1);
  close(plan.visible, explicit30.visible, "old public plan is exact explicit30", 0, 0);
  close(plan.requested_deleted, explicit30.requested_deleted, "old public requests are exact explicit30", 0, 0);
  check(torch::equal(generator.get_state(), rng_before), "context mask consumes no Torch RNG");
  const auto repeated = ctx::make_plan(original, raw.channel_ids, c, 202, 9);
  close(plan.visible, repeated.visible, "deterministic fixed counter plan", 0, 0);
  close(plan.requested_deleted, repeated.requested_deleted, "deterministic requests", 0, 0);
  close(original.visible, saved_visible, "original visibility never mutated", 0, 0);
  close(original.target, saved_target, "original targets never mutated", 0, 0);
  check(!plan.deleted.logical_and(original.visible.logical_not()).any().item<bool>() &&
      !plan.deleted.logical_and(original.target).any().item<bool>() &&
      !plan.visible.logical_and(original.visible.logical_not()).any().item<bool>(),
      "only originally visible coordinates can be deleted or retained");
  check(torch::equal(plan.visible, original.visible.logical_and(plan.deleted.logical_not())) &&
      plan.requested_count == plan.requested_deleted.sum().item<int64_t>() &&
      plan.actual_count == plan.deleted.sum().item<int64_t>() &&
      plan.requested_count == plan.actual_count + plan.restored_count && plan.actual_count > 0,
      "requested actual restored counts describe the exact mask");
  check(!torch::equal(plan.requested_deleted, ctx::make_plan(original, raw.channel_ids, c, 202, 10).requested_deleted) &&
      !torch::equal(plan.requested_deleted, ctx::make_plan(original, raw.channel_ids, c, 303, 9).requested_deleted),
      "absolute attempt and training seed independently change requests");
  const auto order = torch::tensor({1, 0}, torch::kInt64);
  const rpb::MaskPlan permuted{original.hidden.index_select(1, order), original.visible.index_select(1, order),
      original.target.index_select(1, order), original.eligible_channels.index_select(1, order)};
  const auto reordered = ctx::make_plan(permuted, raw.channel_ids.index_select(0, order), c, 202, 9);
  close(reordered.requested_deleted, plan.requested_deleted.index_select(1, order), "semantic canonical counter request order", 0, 0);
  close(reordered.deleted, plan.deleted.index_select(1, order), "semantic storage permutation deletion", 0, 0);
  close(reordered.visible, plan.visible.index_select(1, order), "semantic storage permutation retained support", 0, 0);
  const auto rows = raw.channel_ids.unsqueeze(0).expand({16, c.channel_count});
  close(ctx::make_plan(original, rows, c, 202, 9).deleted, plan.deleted, "per-row IDs preserve canonical counters", 0, 0);

  // Fixed protocol ordinal: B, ascending semantic ID, original H, F. The
  // draws exist even for target coordinates; only original visible cells count.
  const auto base = rpb::training_detail::counter_seed(202, 9, 0x6374782d64726f70ULL);
  for (const int64_t b : {0, 7, 15})
    for (const int64_t channel : {0, 1})
      for (const int64_t h : {0, 8, 19, 31})
        for (const int64_t f : {0, 1}) {
          const auto ordinal = static_cast<uint64_t>(((b * c.channel_count + channel) * c.history_length + h) * c.input_width + f);
          const auto u = static_cast<double>(rpb::training_detail::mixed(base + ordinal) >> 11) * 0x1.0p-53;
          check(plan.requested_deleted[b][channel][h][f].item<bool>() ==
              (original.visible[b][channel][h][f].item<bool>() && u < .30), "frozen stream/ordinal/threshold contract");
        }
  rejects([&] { ctx::make_plan(original, raw.channel_ids, c, -1, 9); }, "negative counter seed");
  rejects([&] { ctx::make_plan(original, torch::tensor({101, 101}, torch::kInt64), c, 202, 9); }, "duplicate semantic IDs");
}

void bounded_lighter_counter_contract() {
  auto c = config(); const auto raw = input(c, 16);
  const auto original = rpb::make_training_mask(raw.observed, c, 37);
  const auto heavy = ctx::make_plan(original, raw.channel_ids, c, 202, 9);
  const auto light = ctx::make_plan(original, raw.channel_ids, c, 202, 9,
      rpb::ContextDeletionRecipe::coordinate15_v1);
  const auto &old = ctx::descriptor(rpb::ContextDeletionRecipe::coordinate30_v1);
  const auto &selected = ctx::descriptor(rpb::ContextDeletionRecipe::coordinate15_v1);
  check(old.ratio == .30 && std::string(old.ratio_text) == "0.30" && old.policy_id == ctx::policy_id &&
      std::string(old.model_tag) == "RPB-v6" && selected.ratio == .15 &&
      std::string(selected.ratio_text) == "0.15" && std::string(selected.model_tag) == "RPB-v7" &&
      std::string(selected.policy_id) == "rpb-training-context-deletion-015-v1", "bounded exact recipe identities");
  check(light.requested_count > 0 && light.requested_count < heavy.requested_count &&
      !light.requested_deleted.logical_and(heavy.requested_deleted.logical_not()).any().item<bool>(),
      "lighter requested masks are nested using the same counter draws");
  close(light.visible, ctx::make_plan(original, raw.channel_ids, c, 202, 9,
      rpb::ContextDeletionRecipe::coordinate15_v1).visible, "lighter counter determinism", 0, 0);
  const auto order = torch::tensor({1, 0}, torch::kInt64);
  const rpb::MaskPlan permuted{original.hidden.index_select(1, order), original.visible.index_select(1, order),
      original.target.index_select(1, order), original.eligible_channels.index_select(1, order)};
  const auto reordered = ctx::make_plan(permuted, raw.channel_ids.index_select(0, order), c, 202, 9,
      rpb::ContextDeletionRecipe::coordinate15_v1);
  close(reordered.requested_deleted, light.requested_deleted.index_select(1, order), "lighter semantic request permutation", 0, 0);
  close(reordered.visible, light.visible.index_select(1, order), "lighter semantic support permutation", 0, 0);
  const auto base = rpb::training_detail::counter_seed(202, 9, ctx::stream);
  for (int64_t b = 0; b < 16; ++b)
    for (int64_t channel = 0; channel < c.channel_count; ++channel)
      for (int64_t h = 0; h < c.history_length; ++h)
        for (int64_t f = 0; f < c.input_width; ++f) {
          const auto ordinal = static_cast<uint64_t>(((b * c.channel_count + channel) * c.history_length + h) * c.input_width + f);
          const auto u = static_cast<double>(rpb::training_detail::mixed(base + ordinal) >> 11) * 0x1.0p-53;
          check(light.requested_deleted[b][channel][h][f].item<bool>() ==
              (original.visible[b][channel][h][f].item<bool>() && u < .15), "lighter exact unchanged stream/ordinal threshold");
        }
  auto generator = at::globalContext().defaultGenerator(at::Device(at::kCPU));
  const auto before = generator.get_state().clone(); const auto threads = at::get_num_threads();
  rejects([&] { ctx::validate_options({true, static_cast<rpb::ContextDeletionRecipe>(99)}); }, "unknown recipe");
  rejects([&] { ctx::validate_options({false, rpb::ContextDeletionRecipe::coordinate15_v1}); }, "disabled nondefault selector");
  rejects([&] { ctx::make_plan(original, raw.channel_ids, c, 202, 9,
      static_cast<rpb::ContextDeletionRecipe>(99)); }, "unknown mask recipe");
  check(torch::equal(before, generator.get_state()) && threads == at::get_num_threads(),
      "recipe validation and masks do not consume runtime RNG or change threads");
}

void sparse_repair_and_ineligible_support() {
  auto c = config(); auto observed = torch::zeros({2, c.channel_count, c.history_length, c.input_width}, torch::kBool);
  auto hidden = torch::zeros_like(observed);
  for (int64_t channel = 0; channel < c.channel_count; ++channel) {
    for (const int64_t h : {1, 9, 17}) observed[0][channel][h][0] = true;
    hidden[0][channel].narrow(0, 0, c.patch_length).fill_(true);
    observed[1][channel][9][0] = true; // Ineligible: one original visible group, no query.
  }
  const auto original = rpb::mask_from_hidden(observed, hidden, c);
  check(original.eligible_channels.select(0, 0).all().item<bool>() &&
      !original.eligible_channels.select(0, 1).any().item<bool>(), "sparse original eligibility");
  bool repaired = false, deleted_ineligible = false;
  for (int64_t attempt = 0; attempt < 64; ++attempt) {
    const auto plan = ctx::make_plan(original, torch::tensor({101, 202}, torch::kInt64), c, 202, attempt);
    const auto groups = plan.visible.reshape({2, c.channel_count, 4, -1}).any(-1).sum(-1);
    check(groups.select(0, 0).eq(2).all().item<bool>(), "repair retains the two original singleton groups");
    close(plan.visible.select(0, 0), original.visible.select(0, 0), "repair only undoes E on original visible coordinates", 0, 0);
    close(plan.deleted.select(0, 1), plan.requested_deleted.select(0, 1), "ineligible channels are never repaired or promoted", 0, 0);
    check(plan.restored_count == plan.requested_deleted.select(0, 0).sum().item<int64_t>(),
        "every restored coordinate corresponds to an original eligible erasure");
    repaired |= plan.restored_count > 0;
    deleted_ineligible |= plan.deleted.select(0, 1).any().item<bool>();
  }
  check(repaired && deleted_ineligible, "fixture exercises repair and un-repaired ineligible context");
  const auto none = torch::zeros_like(observed);
  const auto empty = ctx::make_plan(rpb::mask_from_hidden(none, none, c), torch::tensor({101, 202}, torch::kInt64), c, 202, 9);
  check(empty.requested_count == 0 && empty.actual_count == 0 && empty.restored_count == 0 &&
      !empty.visible.any().item<bool>(), "allmissing never invents any support");
}

void lighter_repair_preserves_queries_without_final_nesting_claim() {
  auto c = config(); auto observed = torch::zeros({1, c.channel_count, c.history_length, c.input_width}, torch::kBool);
  auto hidden = torch::zeros_like(observed);
  for (int64_t channel = 0; channel < c.channel_count; ++channel) {
    for (const int64_t h : {1, 9, 17, 25}) observed[0][channel][h][0] = true;
    hidden[0][channel].narrow(0, 0, c.patch_length).fill_(true);
  }
  const auto original = rpb::mask_from_hidden(observed, hidden, c);
  const auto ids = torch::tensor({101, 202}, torch::kInt64);
  bool repaired = false, actual_not_nested = false;
  for (int64_t attempt = 0; attempt < 4096 && !(repaired && actual_not_nested); ++attempt) {
    const auto light = ctx::make_plan(original, ids, c, 202, attempt, rpb::ContextDeletionRecipe::coordinate15_v1);
    const auto heavy = ctx::make_plan(original, ids, c, 202, attempt);
    check(!light.requested_deleted.logical_and(heavy.requested_deleted.logical_not()).any().item<bool>(),
        "sparse requested masks retain nesting before repair");
    check(light.visible.reshape({1, c.channel_count, 4, -1}).any(-1).sum(-1).ge(2).all().item<bool>() &&
        !light.visible.logical_and(original.visible.logical_not()).any().item<bool>() &&
        !light.deleted.logical_and(original.target).any().item<bool>() &&
        light.requested_count == light.actual_count + light.restored_count,
        "lighter repair clears only E and retains legal original eligible groups");
    repaired |= light.restored_count > 0;
    actual_not_nested |= light.deleted.logical_and(heavy.deleted.logical_not()).any().item<bool>();
  }
  check(repaired && actual_not_nested, "deterministic sparse fixture demonstrates repair can break actual-mask nesting");
  const auto none = torch::zeros_like(observed);
  const auto empty = ctx::make_plan(rpb::mask_from_hidden(none, none, c), ids, c, 202, 9,
      rpb::ContextDeletionRecipe::coordinate15_v1);
  check(empty.actual_count == 0 && empty.requested_count == 0 && empty.restored_count == 0 &&
      !empty.visible.any().item<bool>(), "lighter allmissing remains unsupported");
}

void exact_bottleneck_targets_and_isolation(rpb::ContextDeletionRecipe recipe) {
  auto c = config(); c.global_bottleneck_mode = 2; c.channel_mixer_layers = 1; c.export_width = 32;
  auto raw = input(c, 4); raw.observed[0][0][14][1] = false;
  raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const auto normalized = rpb::fit_scaler(raw, c).transform(raw, c);
  const auto original = rpb::make_training_mask(normalized.observed, c, 37);
  const auto plan = ctx::make_plan(original, normalized.channel_ids, c, 202, 9, recipe);
  torch::manual_seed(707); rpb::Model model(c); model->eval();
  const auto output = ctx::training_forward(*model, normalized, original, plan);
  rpb::Input visible{torch::where(plan.visible, normalized.data, torch::zeros_like(normalized.data)), plan.visible,
      normalized.channel_ids, normalized.endpoints, normalized.sampling_interval};
  const auto exported = model->encode(visible);
  close(output.encoding.z_contextual_global, exported.z_contextual_global, "training serves exact context-visible global32", 0, 0);
  close(output.reconstruction, model->decode(exported.z_contextual_global, normalized.channel_ids), "sole decoder receives exact served global32", 0, 0);
  const auto direct_loss = rpb::hierarchical_huber(output.reconstruction, normalized.data.detach(),
      original.target, original.eligible_channels, c.huber_delta);
  close(output.loss, direct_loss.loss, "existing original-Q hierarchical Huber", 0, 0);
  close(output.target_counts, original.target.sum(std::vector<int64_t>{2, 3}), "original target counts unchanged", 0, 0);
  close(output.eligible_channels, original.eligible_channels, "original eligibility unchanged", 0, 0);
  auto changed_query = normalized; changed_query.data = torch::where(original.target, normalized.data + 100, normalized.data);
  const auto query_result = ctx::training_forward(*model, changed_query, original, plan);
  close(query_result.reconstruction, output.reconstruction, "query values cannot enter context encoder", 0, 0);
  check(query_result.loss.item<double>() != output.loss.item<double>(), "query change affects scoring loss only");
  auto changed_context = normalized; changed_context.data = torch::where(plan.deleted, normalized.data + 10000, normalized.data);
  const auto context_result = ctx::training_forward(*model, changed_context, original, plan);
  close(context_result.reconstruction, output.reconstruction, "deleted context cannot leak into encoding", 0, 0);
  close(context_result.loss, output.loss, "deleted context never becomes a reconstruction target", 0, 0);
  auto illegal = plan; illegal.visible = plan.visible.logical_or(original.target);
  rejects([&] { ctx::training_forward(*model, normalized, original, illegal); }, "query bypass into encoder");
  illegal = plan; illegal.visible = torch::zeros_like(plan.visible);
  rejects([&] { ctx::training_forward(*model, normalized, original, illegal); }, "loss eligibility cannot survive erased required context");
  output.loss.backward();
  for (const auto prefix : {"global_pool_first", "global_pool_second", "decoder_first", "patch_projection"}) {
    double gradient = 0;
    for (const auto &parameter : model->named_parameters())
      if (parameter.key().rfind(prefix, 0) == 0 && parameter.value().grad().defined()) {
        finite(parameter.value().grad(), parameter.key()); gradient += parameter.value().grad().abs().sum().item<double>();
      }
    check(gradient > 0, std::string("context exact-bottleneck gradient missing from ") + prefix);
  }
}
} // namespace

int main() {
  try {
    torch::set_num_threads(1); counter_support_and_semantics(); bounded_lighter_counter_contract();
    sparse_repair_and_ineligible_support(); lighter_repair_preserves_queries_without_final_nesting_claim();
    exact_bottleneck_targets_and_isolation(rpb::ContextDeletionRecipe::coordinate30_v1);
    exact_bottleneck_targets_and_isolation(rpb::ContextDeletionRecipe::coordinate15_v1);
  }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  std::cout << "RPB context-deletion support/counter/loss tests passed\n";
}
