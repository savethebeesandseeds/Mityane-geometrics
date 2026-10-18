#include "embedding/encoders/raw_patch_bottleneck_mae/model.h"
#include "rpb_test_support.h"
#include <iostream>
#include <sstream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;
using torch::indexing::Slice;

void semantics_and_missingness() {
  auto c = config(); auto raw = input(c); auto scaler = rpb::fit_scaler(raw, c);
  auto x = scaler.transform(raw, c); rpb::Model model(c); model->eval();
  torch::NoGradGuard no_grad;
  auto encoded = model->encode(x);
  check(encoded.z_local.sizes() == torch::IntArrayRef({3, 2, 8}) &&
            encoded.z_global.sizes() == torch::IntArrayRef({3, 8}), "export shapes");
  auto changed = x; changed.data = x.data.clone(); changed.data.select(1, 1).add_(17.0);
  close(model->encode(changed).z_local.select(1, 0), encoded.z_local.select(1, 0), "other-channel isolation", 0, 1e-7);
  auto order = torch::tensor({1, 0}, torch::kInt64);
  auto permuted = x;
  permuted.data = x.data.index_select(1, order); permuted.observed = x.observed.index_select(1, order);
  permuted.channel_ids = x.channel_ids.index_select(0, order);
  auto p = model->encode(permuted);
  close(p.z_local, encoded.z_local.index_select(1, order), "semantic permutation", 1e-5, 1e-6);
  close(p.z_global, encoded.z_global, "global permutation", 1e-5, 1e-6);

  auto sparse = x; sparse.data = torch::zeros_like(x.data); sparse.observed = torch::zeros_like(x.observed);
  sparse.observed.index_put_({0, 0, 5, 0}, true);
  for (auto &parameter : model->named_parameters())
    if (parameter.key().find("bias") != std::string::npos) parameter.value().fill_(0.75);
  auto s = model->encode(sparse);
  finite(s.z_local, "singleton/all-invalid encoding");
  check(s.channel_valid_mask.index({0, 0}).item<bool>() && s.visible_observation_counts.index({0, 0}).item<int64_t>() == 1,
        "observed zero must remain valid with one observation");
  check(!s.sample_valid_mask.index({1}).item<bool>() && !s.channel_valid_mask.index({0, 1}).item<bool>(), "absent validity");
  close(s.z_local.index({0, 1}), torch::zeros({8}), "absent channel after biases", 0, 0);
  close(s.z_global.index({1}), torch::zeros({8}), "absent sample after biases", 0, 0);
  check(s.z_local.index({0, 0}).abs().sum().item<double>() > 0, "valid zero signal differs from absent signal");
  auto poisoned = sparse; poisoned.data = sparse.data.masked_fill(sparse.observed.logical_not(), std::numeric_limits<float>::quiet_NaN());
  close(model->encode(poisoned).z_local, s.z_local, "absent NaN isolation", 0, 0);
}

void bottleneck_and_gradients() {
  auto c = config(); auto raw = input(c); auto x = rpb::fit_scaler(raw, c).transform(raw, c);
  rpb::Model model(c); model->eval();
  auto masks = rpb::make_training_mask(x.observed, c, 37);
  auto out = model->forward(x, masks.hidden);
  auto visible = x; visible.observed = masks.visible;
  close(out.encoding.z_local, model->encode(visible).z_local, "served-vector training identity", 0, 0);
  close(out.reconstruction, model->decode(out.encoding.z_local, x.channel_ids), "decoder receives exact export", 0, 0);
  auto changed = x; changed.data = torch::where(masks.target, x.data + 1000, x.data);
  close(model->forward(changed, masks.hidden).encoding.z_local, out.encoding.z_local, "hidden observed value isolation", 0, 0);
  check(out.eligible_example_count == 3 && out.eligible_channel_count == 6 && out.target_cell_count == 96,
        "training denominators");
  out.loss.backward();
  for (const auto &prefix : {"patch_projection", "block_0", "pool_score", "export_projection", "decoder_first"}) {
    double gradient = 0;
    for (const auto &parameter : model->named_parameters())
      if (parameter.key().rfind(prefix, 0) == 0 && parameter.value().grad().defined()) {
        finite(parameter.value().grad(), parameter.key());
        gradient += parameter.value().grad().abs().sum().item<double>();
      }
    check(gradient > 0, std::string("bottleneck gradient missing from ") + prefix);
  }
  model->zero_grad();
  auto differentiable = x; differentiable.data = x.data.detach().clone().set_requires_grad(true);
  model->forward(differentiable, masks.hidden).encoding.z_local.sum().backward();
  close(differentiable.data.grad().masked_select(masks.target), torch::zeros({96}), "forbidden input gradients", 0, 0);
  check(differentiable.data.grad().masked_select(masks.visible).abs().sum().item<double>() > 0, "visible input gradients");

  std::stringstream stream; torch::serialize::OutputArchive output; model->save(output); output.save_to(stream);
  rpb::Model restored(c); torch::serialize::InputArchive archive; archive.load_from(stream); restored->load(archive); restored->eval();
  close(restored->encode(x).z_local, model->encode(x).z_local, "model archive roundtrip", 0, 0);
}

void optimization_smoke() {
  auto c = config(); auto raw = input(c); auto x = rpb::fit_scaler(raw, c).transform(raw, c);
  rpb::Model model(c); auto mask = rpb::make_training_mask(x.observed, c, 5);
  torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(0.01).weight_decay(0));
  const auto first = model->forward(x, mask.hidden).loss.item<double>();
  for (int i = 0; i < 16; ++i) { optimizer.zero_grad(); auto loss = model->forward(x, mask.hidden).loss; loss.backward(); optimizer.step(); }
  const auto last = model->forward(x, mask.hidden).loss.item<double>();
  check(std::isfinite(last) && last < first * 0.8, "fixed-batch optimization did not learn");
  std::cout << "fixed-batch Huber " << first << " -> " << last << '\n';
}

void channel_mixer_defaults_and_initialization() {
  auto independent_config = config();
  check(independent_config.channel_mixer_layers == 0, "channel mixer must default off");
  auto invalid = independent_config; invalid.channel_mixer_layers = -1;
  rejects([&] { rpb::validate_config(invalid); }, "negative channel mixer layers");
  torch::manual_seed(901);
  rpb::Model independent(independent_config); independent->eval();
  const auto independent_rng = torch::rand({8});
  auto explicit_zero = independent_config; explicit_zero.channel_mixer_layers = 0;
  torch::manual_seed(901);
  rpb::Model zero(explicit_zero); zero->eval();
  close(torch::rand({8}), independent_rng, "disabled mixer consumes no extra RNG", 0, 0);
  for (const auto &parameter : independent->named_parameters())
    check(parameter.key().rfind("channel_mixer_block_", 0) != 0,
          "disabled mixer registered parameters");
  auto contextual_config = independent_config; contextual_config.channel_mixer_layers = 2;
  torch::manual_seed(901);
  rpb::Model contextual(contextual_config); contextual->eval();
  const auto contextual_parameters = contextual->named_parameters();
  for (const auto &parameter : independent->named_parameters()) {
    close(contextual_parameters[parameter.key()], parameter.value(),
          "enabled common initialization/" + parameter.key(), 0, 0);
    close(zero->named_parameters()[parameter.key()], parameter.value(),
          "explicit disabled initialization/" + parameter.key(), 0, 0);
  }
  check(contextual_parameters.size() > independent->named_parameters().size(),
        "enabled mixer parameters missing");
  auto raw = input(independent_config);
  auto x = rpb::fit_scaler(raw, independent_config).transform(raw, independent_config);
  torch::NoGradGuard no_grad;
  const auto old = independent->encode(x), enabled = contextual->encode(x);
  check(!old.z_contextual.defined() && !old.z_contextual_global.defined(),
        "disabled contextual exports must be undefined");
  close(enabled.z_local, old.z_local, "enabled independent local path unchanged", 0, 0);
  close(enabled.z_global, old.z_global, "enabled independent global path unchanged", 0, 0);
  close(rpb::compact_reconstruction_export(old, independent_config), old.z_local,
        "disabled reconstruction export", 0, 0);
  close(rpb::compact_reconstruction_export(enabled, contextual_config), enabled.z_contextual,
        "enabled reconstruction export", 0, 0);
}

void channel_mixer_semantics_and_support() {
  auto c = config(); c.channel_mixer_layers = 2;
  auto raw = input(c); auto x = rpb::fit_scaler(raw, c).transform(raw, c);
  rpb::Model model(c); model->eval(); torch::NoGradGuard no_grad;
  const auto encoded = model->encode(x);
  check(encoded.z_contextual.sizes() == torch::IntArrayRef({3, 2, 8}) &&
            encoded.z_contextual_global.sizes() == torch::IntArrayRef({3, 8}),
        "contextual export shapes");
  auto changed = x; changed.data = x.data.clone(); changed.data.select(1, 1).add_(17.0);
  const auto affected = model->encode(changed);
  close(affected.z_local.select(1, 0), encoded.z_local.select(1, 0),
        "contextual model preserves local other-channel isolation", 0, 0);
  check((affected.z_contextual.select(1, 0) - encoded.z_contextual.select(1, 0))
            .abs().max().item<double>() > 1e-6,
        "contextual export ignores the other observed channel");
  auto order = torch::tensor({1, 0}, torch::kInt64);
  auto permuted = x;
  permuted.data = x.data.index_select(1, order); permuted.observed = x.observed.index_select(1, order);
  permuted.channel_ids = x.channel_ids.index_select(0, order);
  const auto shuffled = model->encode(permuted);
  close(shuffled.z_contextual, encoded.z_contextual.index_select(1, order),
        "contextual semantic channel permutation", 1e-5, 1e-6);
  close(shuffled.z_contextual_global, encoded.z_contextual_global,
        "contextual global semantic permutation", 1e-5, 1e-6);

  auto sparse = x; sparse.data = torch::zeros_like(x.data); sparse.observed = torch::zeros_like(x.observed);
  sparse.observed.index_put_({0, 0, 5, 0}, true);
  for (auto &parameter : model->named_parameters())
    if (parameter.key().find("bias") != std::string::npos) parameter.value().fill_(0.75);
  const auto supported = model->encode(sparse);
  finite(supported.z_contextual, "biased singleton/all-invalid contextual encoding");
  check(supported.channel_valid_mask.index({0, 0}).item<bool>() &&
            supported.visible_patch_counts.index({0, 0}).item<int64_t>() == 1 &&
            !supported.channel_valid_mask.index({0, 1}).item<bool>() &&
            !supported.sample_valid_mask.index({1}).item<bool>(),
        "mixer must preserve observed-only support");
  close(supported.z_contextual.index({0, 1}), torch::zeros({8}),
        "mixer cannot infer an absent channel under biased weights", 0, 0);
  close(supported.z_contextual_global.index({1}), torch::zeros({8}),
        "all-invalid contextual sample under biased weights", 0, 0);
  check(supported.z_contextual.index({0, 0}).abs().sum().item<double>() > 0,
        "observed zero singleton lost contextual support");
  auto poisoned = sparse;
  poisoned.data = sparse.data.masked_fill(sparse.observed.logical_not(),
                                          std::numeric_limits<float>::quiet_NaN());
  close(model->encode(poisoned).z_contextual, supported.z_contextual,
        "contextual absent NaN isolation", 0, 0);
  auto entirely_absent = sparse; entirely_absent.observed = torch::zeros_like(sparse.observed);
  entirely_absent.data = torch::full_like(sparse.data, std::numeric_limits<float>::quiet_NaN());
  const auto absent = model->encode(entirely_absent);
  close(absent.z_contextual, torch::zeros_like(absent.z_contextual),
        "all-invalid contextual early path", 0, 0);
  close(absent.z_contextual_global, torch::zeros_like(absent.z_contextual_global),
        "all-invalid contextual global early path", 0, 0);
}

void channel_mixer_original_patch_alignment() {
  auto c = config(); c.channel_mixer_layers = 1;
  auto raw = input(c, 1); auto x = rpb::fit_scaler(raw, c).transform(raw, c);
  rpb::Model model(c); model->eval(); torch::NoGradGuard no_grad;
  // Make the temporal block exactly residual so only the aligned mixer can
  // transmit another patch's signal. Keep projection/position/semantic metadata.
  for (auto &parameter : model->named_parameters())
    if (parameter.key().rfind("block_", 0) == 0) parameter.value().zero_();
  auto disjoint = x; disjoint.observed = torch::zeros_like(x.observed);
  disjoint.observed.select(1, 0).narrow(1, 3 * c.patch_length, c.patch_length).fill_(true);
  disjoint.observed.select(1, 1).narrow(1, 0, c.patch_length).fill_(true);
  disjoint.observed.select(1, 1).narrow(1, 2 * c.patch_length, c.patch_length).fill_(true);
  const auto before = model->encode(disjoint);
  auto off_grid_change = disjoint; off_grid_change.data = disjoint.data.clone();
  off_grid_change.data.select(1, 1).narrow(1, 0, 3 * c.patch_length).add_(19.0);
  close(model->encode(off_grid_change).z_contextual.select(1, 0), before.z_contextual.select(1, 0),
        "mixer aligned original patch indices rather than packed rank", 0, 0);
  close(before.visible_patch_counts, torch::tensor({{1, 2}}, torch::kInt64),
        "unequal packed row counts", 0, 0);
  auto aligned = disjoint; aligned.observed = disjoint.observed.clone();
  aligned.observed.select(1, 1).narrow(1, 3 * c.patch_length, c.patch_length).fill_(true);
  const auto aligned_before = model->encode(aligned);
  auto aligned_change = aligned; aligned_change.data = aligned.data.clone();
  aligned_change.data.select(1, 1).narrow(1, 3 * c.patch_length, c.patch_length).add_(19.0);
  const auto aligned_after = model->encode(aligned_change);
  close(aligned_after.z_local.select(1, 0), aligned_before.z_local.select(1, 0),
        "aligned mixer keeps independent local path", 0, 0);
  check((aligned_after.z_contextual.select(1, 0) - aligned_before.z_contextual.select(1, 0))
            .abs().max().item<double>() > 1e-6,
        "mixer failed to transmit an aligned visible patch");
}

void channel_mixer_bottleneck_and_gradients() {
  auto c = config(); c.channel_mixer_layers = 2;
  auto raw = input(c); auto x = rpb::fit_scaler(raw, c).transform(raw, c);
  rpb::Model model(c); model->eval();
  const auto masks = rpb::make_training_mask(x.observed, c, 37);
  const auto out = model->forward(x, masks.hidden);
  auto visible = x; visible.observed = masks.visible;
  close(out.encoding.z_contextual, model->encode(visible).z_contextual,
        "contextual served-vector training identity", 0, 0);
  close(out.reconstruction,
        model->decode(rpb::compact_reconstruction_export(out.encoding, c), x.channel_ids),
        "decoder receives exact contextual bottleneck", 0, 0);
  check(!torch::allclose(out.reconstruction, model->decode(out.encoding.z_local, x.channel_ids), 1e-6, 1e-6),
        "decoder bypassed contextual export");
  auto hidden_change = x; hidden_change.data = torch::where(masks.target, x.data + 1000, x.data);
  close(model->forward(hidden_change, masks.hidden).encoding.z_contextual, out.encoding.z_contextual,
        "contextual hidden observed value isolation", 0, 0);
  out.loss.backward();
  for (const auto &prefix : {"patch_projection", "block_0", "pool_score", "export_projection",
                             "channel_mixer_block_0", "channel_mixer_block_1", "decoder_first"}) {
    double gradient = 0;
    for (const auto &parameter : model->named_parameters())
      if (parameter.key().rfind(prefix, 0) == 0 && parameter.value().grad().defined()) {
        finite(parameter.value().grad(), parameter.key());
        gradient += parameter.value().grad().abs().sum().item<double>();
      }
    check(gradient > 0, std::string("contextual bottleneck gradient missing from ") + prefix);
  }
  model->zero_grad();
  auto differentiable = x; differentiable.data = x.data.detach().clone().set_requires_grad(true);
  model->forward(differentiable, masks.hidden).encoding.z_contextual.sum().backward();
  close(differentiable.data.grad().masked_select(masks.target), torch::zeros({96}),
        "contextual forbidden input gradients", 0, 0);
  check(differentiable.data.grad().masked_select(masks.visible).abs().sum().item<double>() > 0,
        "contextual visible input gradients");
  std::stringstream stream; torch::serialize::OutputArchive output; model->save(output); output.save_to(stream);
  rpb::Model restored(c); torch::serialize::InputArchive archive; archive.load_from(stream);
  restored->load(archive); restored->eval();
  close(restored->encode(x).z_contextual, model->encode(x).z_contextual,
        "contextual model archive roundtrip", 0, 0);
  close(restored->forward(x, masks.hidden).reconstruction, out.reconstruction,
        "contextual decoder archive roundtrip", 0, 0);
}

void global_bottleneck_initialization(int64_t mixer_layers) {
  auto baseline_config = config(); baseline_config.channel_mixer_layers = mixer_layers;
  check(baseline_config.global_bottleneck_mode == 0, "global bottleneck must default off");
  for (const int64_t invalid_mode : {-1, 4}) {
    auto invalid = baseline_config; invalid.global_bottleneck_mode = invalid_mode;
    rejects([&] { rpb::validate_config(invalid); }, "invalid global bottleneck mode");
  }
  torch::manual_seed(719);
  rpb::Model baseline(baseline_config); baseline->eval();
  const auto baseline_rng = torch::rand({8});
  auto raw = input(baseline_config);
  const auto x = rpb::fit_scaler(raw, baseline_config).transform(raw, baseline_config);
  torch::NoGradGuard no_grad;
  const auto old = baseline->encode(x);
  for (const int64_t mode : {0, 1, 2, 3}) {
    auto c = baseline_config; c.global_bottleneck_mode = mode;
    torch::manual_seed(719);
    rpb::Model model(c); model->eval();
    const auto next_rng = torch::rand({8});
    if (mode < 2)
      close(next_rng, baseline_rng, "mode0/mean-global consumes no extra initialization RNG", 0, 0);
    const auto parameters = model->named_parameters();
    for (const auto &parameter : baseline->named_parameters())
      close(parameters[parameter.key()], parameter.value(),
            "global-mode common initialization/" + parameter.key(), 0, 0);
    check(parameters.size() == baseline->named_parameters().size() + (mode >= 2 ? 4 : 0),
          "only learned-global mode may register pooling parameters");
    const auto encoded = model->encode(x);
    close(encoded.z_local, old.z_local, "global modes preserve initial local diagnostic vectors", 0, 0);
    if (mixer_layers > 0)
      close(encoded.z_contextual, old.z_contextual,
            "global modes preserve initial contextual diagnostic vectors", 0, 0);
    if (mode < 2) {
      close(encoded.z_global, old.z_global, "mode0/mean-global retains exact mean", 0, 0);
      if (mixer_layers > 0)
        close(encoded.z_contextual_global, old.z_contextual_global,
              "mode0/mean-global retains exact contextual mean", 0, 0);
    }
  }
}

void global_bottleneck_semantics_and_gradients(int64_t mixer_layers, int64_t mode) {
  auto c = config(); c.channel_mixer_layers = mixer_layers; c.global_bottleneck_mode = mode;
  auto raw = input(c); const auto x = rpb::fit_scaler(raw, c).transform(raw, c);
  rpb::Model model(c); model->eval();
  const auto masks = rpb::make_training_mask(x.observed, c, 81);
  const auto out = model->forward(x, masks.hidden);
  const auto served = rpb::compact_reconstruction_export(out.encoding, c);
  check(served.sizes() == torch::IntArrayRef({3, 8}), "global reconstruction export must be BD");
  close(served, mixer_layers > 0 ? out.encoding.z_contextual_global : out.encoding.z_global,
        "configured global is the exact served reconstruction export", 0, 0);
  close(out.reconstruction, model->decode(served, x.channel_ids),
        "sole-global decoder receives the exact served vector", 0, 0);
  rejects([&] { model->decode(out.encoding.z_local, x.channel_ids); },
          "global decoder accepting a per-channel local bypass");
  if (mixer_layers > 0)
    rejects([&] { model->decode(out.encoding.z_contextual, x.channel_ids); },
            "global decoder accepting a contextual per-channel bypass");
  auto altered_exports = out.encoding;
  altered_exports.z_local = torch::full_like(out.encoding.z_local, 999);
  if (mixer_layers > 0)
    altered_exports.z_contextual = torch::full_like(out.encoding.z_contextual, -999);
  close(model->decode(rpb::compact_reconstruction_export(altered_exports, c), x.channel_ids),
        out.reconstruction, "diagnostic per-channel vectors cannot bypass global reconstruction", 0, 0);
  check(!torch::allclose(model->decode(torch::zeros_like(served), x.channel_ids),
                         out.reconstruction, 1e-6, 1e-6),
        "global decoder ignores exported-vector intervention");
  auto visible = x; visible.observed = masks.visible;
  close(rpb::compact_reconstruction_export(model->encode(visible), c), served,
        "global served-vector masking identity", 0, 0);
  auto hidden_change = x; hidden_change.data = torch::where(masks.target, x.data + 1000, x.data);
  close(rpb::compact_reconstruction_export(model->forward(hidden_change, masks.hidden).encoding, c),
        served, "global hidden target value isolation", 0, 0);
  served.retain_grad();
  out.loss.backward();
  finite(served.grad(), "sole global reconstruction gradient");
  check(served.grad().abs().sum().item<double>() > 0, "reconstruction loss does not train served global");
  std::vector<std::string> prefixes{"patch_projection", "block_0", "decoder_first"};
  if (mode != 3) { prefixes.push_back("pool_score"); prefixes.push_back("export_projection"); }
  if (mixer_layers > 0) prefixes.push_back("channel_mixer_block_0");
  if (mode == 2) { prefixes.push_back("global_pool_first"); prefixes.push_back("global_pool_second"); }
  if (mode == 3) { prefixes.push_back("global_patch_pool_first"); prefixes.push_back("global_patch_pool_second"); }
  for (const auto &prefix : prefixes) {
    double gradient = 0;
    for (const auto &parameter : model->named_parameters())
      if (parameter.key().rfind(prefix, 0) == 0 && parameter.value().grad().defined()) {
        finite(parameter.value().grad(), parameter.key());
        gradient += parameter.value().grad().abs().sum().item<double>();
      }
    check(gradient > 0, "global reconstruction gradient missing from " + prefix);
  }
  if (mode == 3)
    for (const auto &parameter : model->named_parameters())
      if (parameter.key().rfind("pool_score", 0) == 0 || parameter.key().rfind("pool_positions", 0) == 0 ||
          parameter.key().rfind("export_projection", 0) == 0)
        check(!parameter.value().grad().defined() || parameter.value().grad().abs().sum().item<double>() == 0,
              "patch-state global reconstruction routed through diagnostic D compression");
  model->zero_grad();
  auto differentiable = x; differentiable.data = x.data.detach().clone().set_requires_grad(true);
  rpb::compact_reconstruction_export(model->forward(differentiable, masks.hidden).encoding, c)
      .sum().backward();
  const auto hidden_gradients = differentiable.data.grad().masked_select(masks.target);
  close(hidden_gradients, torch::zeros_like(hidden_gradients), "global forbidden input gradients", 0, 0);
  check(differentiable.data.grad().masked_select(masks.visible).abs().sum().item<double>() > 0,
        "global visible input gradients missing");

  torch::NoGradGuard no_grad;
  const auto encoded = model->encode(x);
  // Different physical storage permutations in each example, with semantic IDs
  // travelling with values/masks. Neither global pool may depend on storage rank.
  const auto order = torch::tensor({{1, 0}, {0, 1}, {1, 0}}, torch::kInt64);
  auto permuted = x;
  const auto gather = order.unsqueeze(-1).unsqueeze(-1).expand_as(x.data);
  permuted.data = x.data.gather(1, gather); permuted.observed = x.observed.gather(1, gather);
  permuted.channel_ids = x.channel_ids.unsqueeze(0).expand({3, 2}).gather(1, order);
  const auto shuffled = model->encode(permuted);
  close(shuffled.z_global, encoded.z_global, "global semantic-ID storage permutation", 1e-5, 1e-6);
  close(shuffled.z_local, encoded.z_local.gather(1, order.unsqueeze(-1).expand({3, 2, 8})),
        "global mode preserves local semantic permutation", 1e-5, 1e-6);
  if (mixer_layers > 0)
    close(shuffled.z_contextual_global, encoded.z_contextual_global,
          "contextual global semantic-ID storage permutation", 1e-5, 1e-6);
  close(model->decode(rpb::compact_reconstruction_export(shuffled, c), permuted.channel_ids),
        model->decode(rpb::compact_reconstruction_export(encoded, c), x.channel_ids).gather(1, gather),
        "global decoder semantic channel permutation", 1e-5, 1e-6);
  auto sparse = x; sparse.data = torch::zeros_like(x.data); sparse.observed = torch::zeros_like(x.observed);
  sparse.observed.index_put_({0, 0, 5, 0}, true);
  for (auto &parameter : model->named_parameters())
    if (parameter.key().find("bias") != std::string::npos) parameter.value().fill_(0.75);
  const auto supported = model->encode(sparse);
  const auto sparse_global = rpb::compact_reconstruction_export(supported, c);
  finite(sparse_global, "partial-support learned/mean global");
  check(supported.sample_valid_mask[0].item<bool>() && !supported.sample_valid_mask[1].item<bool>() &&
      !supported.channel_valid_mask[0][1].item<bool>(), "global mode invented observed support");
  close(supported.z_local[0][1], torch::zeros({8}), "global pooling cannot infer missing local", 0, 0);
  close(sparse_global[1], torch::zeros({8}), "biased all-absent global must be exact zero", 0, 0);
  if (mixer_layers > 0)
    close(supported.z_contextual[0][1], torch::zeros({8}),
          "global pooling cannot infer missing contextual channel", 0, 0);
  auto poisoned = sparse;
  poisoned.data = sparse.data.masked_fill(sparse.observed.logical_not(),
                                          std::numeric_limits<float>::quiet_NaN());
  close(rpb::compact_reconstruction_export(model->encode(poisoned), c), sparse_global,
        "global absent NaN storage isolation", 0, 0);
  auto absent = poisoned; absent.observed = torch::zeros_like(sparse.observed);
  absent.data = torch::full_like(sparse.data, std::numeric_limits<float>::quiet_NaN());
  const auto no_signal = model->encode(absent);
  close(rpb::compact_reconstruction_export(no_signal, c), torch::zeros({3, 8}),
        "all-absent global early path exact zero", 0, 0);
  check(!no_signal.sample_valid_mask.any().item<bool>(), "all-absent global marked valid");
  std::stringstream stream; torch::serialize::OutputArchive output; model->save(output); output.save_to(stream);
  rpb::Model restored(c); torch::serialize::InputArchive archive; archive.load_from(stream);
  restored->load(archive); restored->eval();
  close(rpb::compact_reconstruction_export(restored->encode(x), c),
        rpb::compact_reconstruction_export(model->encode(x), c), "global model archive export roundtrip", 0, 0);
  close(restored->forward(x, masks.hidden).reconstruction, model->forward(x, masks.hidden).reconstruction,
        "sole-global decoder archive roundtrip", 0, 0);
}

void patch_global_original_slots_and_diagnostic_bypass(int64_t mixer_layers) {
  auto c = config(); c.channel_ids = {202, 101};
  c.channel_mixer_layers = mixer_layers; c.global_bottleneck_mode = 3;
  auto raw = input(c, 1); const auto x = rpb::fit_scaler(raw, c).transform(raw, c);
  rpb::Model model(c); model->eval(); torch::NoGradGuard no_grad;
  const auto before = model->encode(x);
  for (auto &parameter : model->named_parameters()) {
    if (parameter.key().rfind("export_projection", 0) == 0) parameter.value().fill_(7);
    if (parameter.key().rfind("pool_score", 0) == 0 || parameter.key().rfind("pool_positions", 0) == 0)
      parameter.value().fill_(-9);
  }
  const auto after = model->encode(x);
  close(after.z_global, before.z_global, "patch-state global bypasses local D readout weights", 0, 0);
  check(!torch::equal(after.z_local, before.z_local), "diagnostic D intervention failed");
  if (mixer_layers > 0)
    close(after.z_contextual_global, before.z_contextual_global,
          "patch-state contextual global bypasses contextual D readout weights", 0, 0);

  const auto K = c.history_length / c.patch_length, W = c.encoder_width;
  auto parameters = model->named_parameters();
  for (auto &parameter : model->named_parameters())
    if (parameter.key().rfind("global_patch_pool_", 0) == 0) parameter.value().zero_();
  // Read one declared support bit exactly. Unequal packed lengths make packed
  // rank an invalid surrogate for this semantic-channel/original-patch slot.
  parameters["global_patch_pool_first.weight"].index_put_({0, c.channel_count * K * W + 3}, 1);
  parameters["global_patch_pool_second.weight"].index_put_({0, 0}, 1);
  auto sparse = x; sparse.observed = torch::zeros_like(x.observed);
  sparse.observed.select(1, 0).narrow(1, 3 * c.patch_length, c.patch_length).fill_(true);
  sparse.observed.select(1, 1).narrow(1, 0, c.patch_length).fill_(true);
  sparse.observed.select(1, 1).narrow(1, 2 * c.patch_length, c.patch_length).fill_(true);
  const auto selected = model->encode(sparse);
  auto expected = torch::zeros({1, c.export_width});
  expected.index_put_({0, 0}, torch::gelu(torch::ones({1})).item<float>());
  close(rpb::compact_reconstruction_export(selected, c), expected,
        "patch-state pool restores original patch3 support rather than packed rank0", 0, 0);
  const auto permutation = torch::tensor({1, 0}, torch::kInt64);
  auto reordered = sparse; reordered.data = sparse.data.index_select(1, permutation);
  reordered.observed = sparse.observed.index_select(1, permutation);
  reordered.channel_ids = sparse.channel_ids.index_select(0, permutation);
  close(rpb::compact_reconstruction_export(model->encode(reordered), c), expected,
        "patch-state slots follow custom semantic IDs rather than physical channel rank", 0, 0);
  parameters["global_patch_pool_first.weight"].zero_();
  parameters["global_patch_pool_first.weight"].index_put_({0, c.channel_count * K * W}, 1);
  close(rpb::compact_reconstruction_export(model->encode(sparse), c), torch::zeros_like(expected),
        "absent original patch0 cannot inherit packed patch3 support", 0, 0);
  auto poison = sparse; poison.data = sparse.data.masked_fill(sparse.observed.logical_not(),
      std::numeric_limits<float>::quiet_NaN());
  close(rpb::compact_reconstruction_export(model->encode(poison), c), torch::zeros_like(expected),
        "patch-state padded absent values remain zero", 0, 0);
  // A hand-set original-position state gives a second witness independent of
  // support bits. Residual blocks and projection metadata simplify the known
  // patch3 state without changing the scatter/semantic-order/pooling route.
  for (auto &parameter : model->named_parameters())
    if (parameter.key().rfind("block_", 0) == 0 ||
        parameter.key().rfind("channel_mixer_block_", 0) == 0 ||
        parameter.key().rfind("patch_projection", 0) == 0 ||
        parameter.key().rfind("channels", 0) == 0 ||
        parameter.key().rfind("positions", 0) == 0)
      parameter.value().zero_();
  parameters["positions.weight"].index_put_({3, 0}, 2);
  parameters["positions.weight"].index_put_({2, 0}, -3);
  parameters["final_norm.weight"].fill_(1); parameters["final_norm.bias"].zero_();
  parameters["global_patch_pool_first.weight"].zero_();
  parameters["global_patch_pool_first.weight"].index_put_({0, 3 * W}, 1);
  auto state = torch::zeros({1, W}); state.index_put_({0, 0}, 2);
  const auto normalized = torch::layer_norm(state, {W}, torch::ones({W}), torch::zeros({W}),
                                            c.layer_norm_epsilon);
  expected.zero_(); expected.index_put_({0, 0}, torch::gelu(normalized)[0][0]);
  close(rpb::compact_reconstruction_export(model->encode(sparse), c), expected,
        "patch-state value goes to original patch3 rather than packed rank0", 0, 0);
  close(rpb::compact_reconstruction_export(model->encode(reordered), c), expected,
        "original-position W state uses canonical semantic channel order", 0, 0);
}

void patch_global_parameter_count() {
  rpb::Config c; c.channel_mixer_layers = 1; c.global_bottleneck_mode = 2;
  rpb::Model summary(c); c.global_bottleneck_mode = 3; rpb::Model direct(c);
  int64_t summary_count = 0, direct_count = 0, head_count = 0;
  for (const auto &parameter : summary->parameters()) summary_count += parameter.numel();
  for (const auto &parameter : direct->named_parameters()) {
    direct_count += parameter.value().numel();
    if (parameter.key().rfind("global_patch_pool_", 0) == 0) head_count += parameter.value().numel();
  }
  check(summary_count == 225805 && direct_count == 269389 && head_count == 52064 &&
            direct_count - summary_count == 43584,
        "production patch-state candidate parameter budget changed");
  auto overflow = c; overflow.encoder_width = std::numeric_limits<int64_t>::max();
  rejects([&] { rpb::validate_config(overflow); }, "patch-state input width overflow");
}
}

int main() {
  try {
    torch::set_num_threads(1); torch::manual_seed(71);
    semantics_and_missingness(); bottleneck_and_gradients(); optimization_smoke();
    channel_mixer_defaults_and_initialization(); channel_mixer_semantics_and_support();
    channel_mixer_original_patch_alignment(); channel_mixer_bottleneck_and_gradients();
    for (const int64_t mixer_layers : {0, 1}) {
      global_bottleneck_initialization(mixer_layers);
      for (const int64_t mode : {1, 2, 3}) global_bottleneck_semantics_and_gradients(mixer_layers, mode);
      patch_global_original_slots_and_diagnostic_bypass(mixer_layers);
    }
    patch_global_parameter_count();
    std::cout << "RPB-MAE model tests passed\n";
  }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
