// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/temporal_relation_bank.h"
#include "rpb_test_support.h"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#ifndef ARCHITECTURE_SCREEN_SOURCE_ID
#define ARCHITECTURE_SCREEN_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;

rpb::Config config() {
  rpb::Config c;
  c.device = torch::kCUDA; c.channel_mixer_layers = 1;
  c.channel_mixer_placement = 1; c.global_bottleneck_mode = 2;
  c.channel_ids = {101, 202, 303};
  return c;
}
void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &why) {
  check(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() && torch::equal(a, b), why);
}
void bank_reference() {
  const auto c = config();
  auto x = input(c, 2);
  x.data = x.data.to(torch::kCUDA, torch::kFloat32);
  x.observed = x.observed.to(torch::kCUDA);
  x.observed[0][0].select(1, 0).fill_(false);
  x.observed[0][1].select(1, 1).fill_(false);
  x.observed[0][2].fill_(false);
  x.observed[1][0].narrow(0, 6, 3).fill_(false);
  x.observed[1][1].narrow(0, 16, 3).fill_(false);
  const auto actual = rpb::temporal_relation_bank(x, c);
  const auto data = x.data.to(torch::kCPU).contiguous();
  const auto mask = x.observed.to(torch::kCPU).contiguous();
  auto v = data.accessor<float, 4>(); auto m = mask.accessor<bool, 4>();
  auto expected = torch::zeros({2, 324}, torch::kFloat32);
  auto counts = torch::zeros({2, 108}, torch::kInt64);
  auto out = expected.accessor<float, 2>(); auto nout = counts.accessor<int64_t, 2>();
  constexpr std::array<std::array<int64_t, 2>, 3> pairs{{{0, 1}, {0, 2}, {1, 2}}};
  for (int64_t b = 0; b < 2; ++b)
    for (int64_t p = 0; p < 3; ++p)
      for (int64_t fi = 0; fi < 3; ++fi)
        for (int64_t fj = 0; fj < 3; ++fj)
          for (int64_t k = 1; k <= 4; ++k) {
            const auto i = pairs[p][0], j = pairs[p][1];
            double a = 0, s = 0, ei = 0, ej = 0; int64_t n = 0;
            for (int64_t t = k; t < 32 - k; ++t) {
              bool legal = true;
              for (const auto pos : {t - k, t, t + k})
                legal = legal && m[b][i][pos][fi] && m[b][j][pos][fj];
              if (!legal) continue;
              const double li = double(v[b][i][t][fi]) - v[b][i][t - k][fi];
              const double ri = double(v[b][i][t + k][fi]) - v[b][i][t][fi];
              const double lj = double(v[b][j][t][fj]) - v[b][j][t - k][fj];
              const double rj = double(v[b][j][t + k][fj]) - v[b][j][t][fj];
              a += li * rj - ri * lj; s += li * lj + ri * rj;
              ei += li * li + ri * ri; ej += lj * lj + rj * rj; ++n;
            }
            const auto column = (p * 9 + fi * 3 + fj) * 4 + k - 1;
            nout[b][column] = n; out[b][216 + column] = float(double(n) / (32 - 2 * k));
            if (n && ei > 0 && ej > 0) {
              const auto denominator = std::max(std::sqrt((ei / n) * (ej / n)),
                                                rpb::kTemporalRelationDenominatorFloor);
              out[b][column] = float(a / n / denominator);
              out[b][108 + column] = float(s / n / denominator);
            }
          }
  close(actual.values.to(torch::kCPU), expected, "independent original-coordinate moments", 2e-6, 2e-7);
  exact(actual.common_counts.to(torch::kCPU), counts, "all108 exact common support counts");
  check(counts[0][(0 * 9 + 1 * 3 + 0) * 4].item<int64_t>() == 30,
        "cross-feature triplet survives when matching features do not");
  check(counts[0][0].item<int64_t>() == 0, "missing feature is not invented");
  check(actual.values.narrow(1, 0, 216).abs().max().item<double>() <= 1.000001,
        "normalized symmetric and oriented moments bounded");
  auto poison = x;
  poison.data = torch::where(x.observed, x.data,
      torch::full_like(x.data, std::numeric_limits<float>::quiet_NaN()));
  exact(rpb::temporal_relation_bank(poison, c).values, actual.values, "hidden NaNs never enter products");
  auto affine = x;
  const auto gains = torch::tensor({.7, 1.2, 1.5, 1.1, .8, 1.4, .6, 1.3, 1.7}, torch::kFloat32)
                         .reshape({1, 3, 1, 3}).to(torch::kCUDA);
  const auto offsets = torch::tensor({-.3, .2, .5, .4, -.1, .6, .1, .3, -.2}, torch::kFloat32)
                           .reshape({1, 3, 1, 3}).to(torch::kCUDA);
  affine.data = x.data * gains + offsets;
  close(rpb::temporal_relation_bank(affine, c).values, actual.values,
        "positive per-feature affine invariance away from denominator floor", 2e-5, 3e-6);
  const auto permutation = torch::tensor({2, 0, 1}, torch::kInt64).to(torch::kCUDA);
  auto permuted = x;
  permuted.data = x.data.index_select(1, permutation);
  permuted.observed = x.observed.index_select(1, permutation);
  permuted.channel_ids = x.channel_ids.to(torch::kCUDA).index_select(0, permutation);
  exact(rpb::temporal_relation_bank(permuted, c).values, actual.values, "semantic permutation bank equality");
  auto absent = poison; absent.observed = torch::zeros_like(x.observed);
  exact(rpb::temporal_relation_bank(absent, c).values, torch::zeros_like(actual.values), "all absent bank zero");
  auto flat = x; flat.observed = torch::ones_like(x.observed); flat.data = torch::zeros_like(x.data).set_requires_grad(true);
  const auto flat_bank = rpb::temporal_relation_bank(flat, c);
  exact(flat_bank.values.narrow(1, 0, 216), torch::zeros({2, 216}, x.data.options()), "zero energy moments zero");
  exact(flat_bank.values.narrow(1, 216, 108), torch::ones({2, 108}, x.data.options()), "flat observed support retained");
  flat_bank.values.sum().backward(); finite(flat.data.grad(), "zero energy input derivatives finite");
  exact(flat.data.grad(), torch::zeros_like(flat.data), "zero energy derivatives zero");
}

void common_initialization(rpb::Model &old, rpb::TemporalRelationModel &candidate) {
  const auto op = old->named_parameters(), cp = candidate->named_parameters();
  check(cp.size() == op.size() + 1, "one last bias-free relation weight"); int64_t values = 0;
  for (size_t i = 0; i < op.size(); ++i) {
    check(cp[i].key() == "backbone." + op[i].key(), "exact common registration prefix");
    exact(cp[i].value(), op[i].value(), "exact common initialized CUDA values");
    check(cp[i].value().is_cuda(), "common parameters CUDA"); values += cp[i].value().numel();
  }
  check(values == 225805 && cp[op.size()].key() == "temporal_relation_projection.weight" &&
        cp[op.size()].value().sizes() == torch::IntArrayRef({32, 324}), "common225805 plus10368 last values");
  const auto ob = old->named_buffers(), cb = candidate->named_buffers(); check(ob.size() == cb.size(), "common buffer count");
  for (size_t i = 0; i < ob.size(); ++i) {
    check(cb[i].key() == "backbone." + ob[i].key(), "common buffer order");
    exact(ob[i].value(), cb[i].value(), "common buffers exact");
  }
}
void compare_state(rpb::TemporalRelationModel &a, rpb::TemporalRelationModel &b,
                   torch::optim::AdamW &ao, torch::optim::AdamW &bo) {
  const auto ap = a->named_parameters(), bp = b->named_parameters();
  check(ap.keys() == bp.keys() && ao.state().size() == bo.state().size(), "same named state association");
  for (const auto &p : ap) {
    exact(p.value(), bp[p.key()], "direct/split parameter exact");
    const auto x = ao.state().find(p.value().unsafeGetTensorImpl());
    const auto y = bo.state().find(bp[p.key()].unsafeGetTensorImpl());
    check((x == ao.state().end()) == (y == bo.state().end()), "actual optimizer presence");
    if (x == ao.state().end()) continue;
    const auto *xs = dynamic_cast<const torch::optim::AdamWParamState *>(x->second.get());
    const auto *ys = dynamic_cast<const torch::optim::AdamWParamState *>(y->second.get());
    check(xs && ys && xs->step() == ys->step(), "actual optimizer step");
    exact(xs->exp_avg(), ys->exp_avg(), "first moment exact");
    exact(xs->exp_avg_sq(), ys->exp_avg_sq(), "second moment exact");
  }
  const auto ab = a->named_buffers(), bb = b->named_buffers(); check(ab.keys() == bb.keys(), "named buffers association");
  for (const auto &v : ab) exact(v.value(), bb[v.key()], "direct/split buffers exact");
}
void model_graph() {
  const auto c = config(); const auto raw = input(c, 3);
  const auto scaler = rpb::fit_scaler(raw, c); auto x = scaler.transform(raw, c);
  torch::manual_seed(14101); rpb::Model old(c);
  const auto cpu_rng = torch::rand({8});
  const auto cuda_rng = torch::rand({8}, torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(14101); rpb::TemporalRelationModel candidate(c);
  exact(cpu_rng, torch::rand({8}), "new constructor preserves backbone CPU RNG");
  exact(cuda_rng, torch::rand({8}, torch::TensorOptions().device(torch::kCUDA)), "new constructor preserves backbone CUDA RNG");
  old->eval(); candidate->eval(); common_initialization(old, candidate);
  const auto weight = candidate->temporal_relation_projection->weight;
  check(weight.is_cuda() && weight.abs().sum().item<double>() > 0, "deterministic nonzero CUDA initializer");
  check(candidate->parameters().back().numel() == 10368, "declared added capacity");
  {
    torch::NoGradGuard guard;
    const auto base = old->encode(x), modified = candidate->encode(x);
    exact(base.z_local, modified.z_local, "independent local output unchanged");
    exact(base.z_contextual, modified.z_contextual, "old contextual channel output unchanged");
    exact(base.z_global, modified.z_global, "independent global diagnostic unchanged");
    exact(base.channel_valid_mask, modified.channel_valid_mask, "no new support filter");
    close(modified.z_contextual_global, base.z_contextual_global +
          candidate->temporal_relation_projection(rpb::temporal_relation_bank(x, c).values),
          "exact declared native residual", 0., 0.);
    check(!torch::equal(base.z_contextual_global, modified.z_contextual_global), "distinct untrained candidate necessary");
    const auto saved = weight.clone(); weight.zero_();
    exact(candidate->encode(x).z_contextual_global, base.z_contextual_global, "zero intervention recovers unchanged backbone");
    weight.copy_(saved);
  }
  const auto q = rpb::make_training_mask(x.observed, c, 14103);
  const auto baseline = old->forward(x, q.hidden), initial = candidate->forward(x, q.hidden);
  exact(initial.target_counts, baseline.target_counts, "original query target denominator unchanged");
  exact(initial.eligible_examples, baseline.eligible_examples, "original eligibility unchanged");
  exact(initial.reconstruction, candidate->decode(initial.encoding.z_contextual_global, x.channel_ids),
        "decoder receives the same combined native32 only");
  auto changed = x; changed.data = torch::where(q.target, x.data + 1000., x.data);
  exact(candidate->forward(changed, q.hidden).encoding.z_contextual_global,
        initial.encoding.z_contextual_global, "query targets cannot enter relation bank");
  candidate->zero_grad(); initial.loss.backward(); finite(weight.grad(), "CUDA relation reconstruction gradient");
  check(weight.grad().abs().sum().item<double>() > 0, "relation projection learns through native32");
  candidate->zero_grad(); auto leaf = x; leaf.data = x.data.detach().clone().set_requires_grad(true);
  candidate->forward(leaf, q.hidden).encoding.z_contextual_global.sum().backward();
  finite(leaf.data.grad(), "visible native input gradient finite");
  check(leaf.data.grad().masked_select(q.target).abs().sum().item<double>() == 0, "hidden target input gradient zero");
  check(leaf.data.grad().masked_select(q.visible).abs().sum().item<double>() > 0, "visible gradient remains");
  torch::manual_seed(14107); rpb::TemporalRelationModel direct(c);
  torch::manual_seed(14107); rpb::TemporalRelationModel split(c);
  direct->eval(); split->eval();
  torch::optim::AdamW da(direct->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  torch::optim::AdamW sa(split->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  auto step = [&](rpb::TemporalRelationModel &model, torch::optim::AdamW &optimizer, int64_t attempt) {
    const auto mask = rpb::make_training_mask(x.observed, c, 14110 + attempt);
    optimizer.zero_grad(); auto out = model->forward(x, mask.hidden);
    check(out.loss.is_cuda() && out.encoding.z_contextual_global.sizes() == torch::IntArrayRef({3, 32}), "CUDA native32 loss path");
    out.loss.backward(); finite(model->temporal_relation_projection->weight.grad(), "actual CUDA branch gradients");
    torch::nn::utils::clip_grad_norm_(model->parameters(), 1., 2., true); optimizer.step();
  };
  for (int64_t t = 0; t < 4; ++t) step(direct, da, t);
  for (int64_t t = 0; t < 2; ++t) step(split, sa, t);
  torch::Tensor preserved_native;
  { torch::NoGradGuard guard; preserved_native = split->encode(x).z_contextual_global.detach().clone(); }
  std::stringstream stream; torch::serialize::OutputArchive saved; split->save(saved); saved.save_to(stream);
  const auto preserved_bytes = stream.str();
  // Intervening CPU fixture work uses cloned results, not a CPU model/head fit.
  auto cpu_fixture = preserved_native.to(torch::kCPU).clone(); cpu_fixture.square_();
  for (int64_t t = 2; t < 4; ++t) step(split, sa, t);
  compare_state(direct, split, da, sa); check(stream.str() == preserved_bytes, "earlier serialized state bytes unchanged");
  rpb::TemporalRelationModel restored(c); torch::serialize::InputArchive archive;
  archive.load_from(stream, torch::kCUDA); restored->load(archive); restored->eval();
  { torch::NoGradGuard guard; exact(restored->encode(x).z_contextual_global, preserved_native, "earlier CUDA snapshot immutable after continuation"); }
  check(!torch::equal(split->temporal_relation_projection->weight, weight), "actual trained branch differs from initial");
  const auto state = sa.state().find(split->temporal_relation_projection->weight.unsafeGetTensorImpl());
  check(state != sa.state().end(), "actual relation AdamW state present");
  const auto *adam = dynamic_cast<const torch::optim::AdamWParamState *>(state->second.get());
  check(adam && adam->step() == 4 && adam->exp_avg().is_cuda(), "actual four-step CUDA moments");
  {
    torch::NoGradGuard guard;
    const auto order = torch::tensor({2, 0, 1}, torch::kInt64).to(torch::kCUDA); auto perm = x;
    perm.data = x.data.index_select(1, order); perm.observed = x.observed.to(torch::kCUDA).index_select(1, order);
    perm.channel_ids = x.channel_ids.to(torch::kCUDA).index_select(0, order);
    close(split->encode(perm).z_contextual_global, split->encode(x).z_contextual_global, "semantic native permutation", 2e-5, 2e-6);
    auto absent = x; absent.observed = torch::zeros_like(x.observed);
    const auto out = split->encode(absent); check(!out.sample_valid_mask.any().item<bool>(), "all absent remains invalid");
    exact(out.z_contextual_global, torch::zeros_like(out.z_contextual_global), "all absent native zero");
  }
}
void closed_contract() {
  const auto c = config();
  for (int change = 0; change < 5; ++change) {
    auto invalid = c;
    if (change == 0) invalid.channel_mixer_placement = 0;
    if (change == 1) invalid.global_pool_input_source = 1;
    if (change == 2) invalid.temporal_difference_input = 1;
    if (change == 3) invalid.global_bottleneck_mode = 3;
    if (change == 4) invalid.device = torch::kCPU;
    rejects([&] { rpb::TemporalRelationModel model(invalid); }, "closed architecture/CUDA contract");
  }
}
} // namespace

int main() try {
  check(torch::cuda::is_available(), "temporal relation admission requires actual CUDA");
  torch::set_num_threads(1);
  closed_contract(); bank_reference(); model_graph();
  std::cout << "Temporal relation bank CUDA admission passed\n" << ARCHITECTURE_SCREEN_SOURCE_ID << '\n';
  return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
