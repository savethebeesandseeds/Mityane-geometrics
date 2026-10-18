// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/partitioned_temporal_relation.h"
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

rpb::Config configuration() {
  rpb::Config c; c.device = torch::kCUDA; c.channel_ids = {101, 202, 303};
  c.channel_mixer_layers = 1; c.channel_mixer_placement = 1; c.global_bottleneck_mode = 2;
  return c;
}
void exact(const torch::Tensor &a, const torch::Tensor &b, const std::string &why) {
  check(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() && torch::equal(a, b), why);
}
rpb::Input reverse_time(const rpb::Input &x) {
  auto out = x; out.data = x.data.flip({2}); out.observed = x.observed.flip({2}); return out;
}
void odd_contract() {
  const auto c = configuration(); auto x = input(c, 3);
  x.data = x.data.to(torch::kCUDA, torch::kFloat32); x.observed = x.observed.to(torch::kCUDA);
  x.observed[0][0].narrow(0, 6, 3).fill_(false);
  x.observed[0][1].narrow(0, 17, 3).fill_(false);
  x.observed[1][2].select(1, 1).fill_(false);
  x.data = torch::where(x.observed, x.data, torch::full_like(x.data, std::numeric_limits<float>::quiet_NaN()));
  const auto bank = rpb::temporal_relation_bank(x, c), reversed = rpb::temporal_relation_bank(reverse_time(x), c);
  const auto a = bank.values.narrow(1, 0, 108);
  close(reversed.values.narrow(1, 0, 108), -a, "all108 original-coordinate moments are time odd", 2e-6, 2e-7);
  close(reversed.values.narrow(1, 108, 108), bank.values.narrow(1, 108, 108), "symmetric block is time even", 2e-6, 2e-7);
  exact(reversed.values.narrow(1, 216, 108), bank.values.narrow(1, 216, 108), "support block is time even");
  exact(reversed.common_counts, bank.common_counts, "reversal retains identical common support counts");
  rpb::GroupedOddRelationProjection grouped; grouped->to(c.device, c.dtype);
  const auto expected = a.reshape({3, 3, 9, 4}).mean(2).reshape({3, 12});
  close(grouped(a), expected, "literal generic feature means at all pair spacings", 2e-6, 2e-7);
  {
    torch::NoGradGuard guard;
    const auto arbitrary = torch::sin(torch::arange(432, torch::TensorOptions().dtype(c.dtype).device(c.device)) * .17).reshape({3, 4, 36});
    grouped->weight.copy_(arbitrary);
  }
  close(grouped(reversed.values.narrow(1, 0, 108)), -grouped(a),
        "time-odd subspace for arbitrary dense trained weights", 4e-6, 1e-6);
  const auto one_pair = torch::zeros_like(a); one_pair.narrow(1, 36, 36).copy_(a.narrow(1, 36, 36));
  const auto isolated = grouped(one_pair);
  exact(isolated.narrow(1, 0, 4), torch::zeros({3, 4}, a.options()), "pair0 cannot receive pair1 input");
  exact(isolated.narrow(1, 8, 4), torch::zeros({3, 4}, a.options()), "pair2 cannot receive pair1 input");
  rejects([&] { grouped(bank.values); }, "symmetric/support bank cannot enter odd-only API");
  rpb::NativeShapeProjection shape; shape->to(c.device, c.dtype);
  const auto raw32 = torch::arange(96, a.options()).reshape({3, 32});
  exact(shape(raw32), raw32.narrow(1, 0, 20), "shape initializer selects first20 without RNG");
  rejects([&] { shape(torch::zeros({3, 31}, a.options())); }, "closed raw native width");
  auto absent = x; absent.observed = torch::zeros_like(x.observed);
  const auto absent_a = rpb::temporal_relation_bank(absent, c).values.narrow(1, 0, 108);
  exact(grouped(absent_a), torch::zeros({3, 12}, a.options()), "all-absent odd coordinates zero for arbitrary weights");
  auto flat = x; flat.observed = torch::ones_like(x.observed); flat.data = torch::zeros_like(x.data).set_requires_grad(true);
  grouped(rpb::temporal_relation_bank(flat, c).values.narrow(1, 0, 108)).sum().backward();
  finite(flat.data.grad(), "zero-energy odd gradient finite");
  exact(flat.data.grad(), torch::zeros_like(flat.data), "zero-energy odd gradient zero");
}
void common_initialization(rpb::Model &old, rpb::PartitionedTemporalRelationModel &candidate) {
  const auto op = old->named_parameters(), cp = candidate->named_parameters();
  check(cp.size() == op.size() + 2, "two last bias-free projection weights"); int64_t common = 0, total = 0;
  for (size_t i = 0; i < op.size(); ++i) {
    check(cp[i].key() == "backbone." + op[i].key(), "exact common registration prefix");
    exact(cp[i].value(), op[i].value(), "old225805 common initialized values exact"); common += cp[i].value().numel();
  }
  check(common == 225805 && cp[op.size()].key() == "native_shape_projection.weight" &&
        cp[op.size()].value().sizes() == torch::IntArrayRef({20, 32}) &&
        cp[op.size() + 1].key() == "grouped_odd_relation_projection.weight" &&
        cp[op.size() + 1].value().sizes() == torch::IntArrayRef({3, 4, 36}), "common225805 plus640 shape plus432 grouped odd");
  for (const auto &p : cp) { check(p.value().is_cuda(), "all registered parameters actual CUDA"); total += p.value().numel(); }
  check(total == rpb::kPartitionedTemporalRelationParameterCount && total == 226877, "exact registered capacity");
  const auto ob = old->named_buffers(), cb = candidate->named_buffers(); check(ob.size() == cb.size(), "common buffer count");
  for (size_t i = 0; i < ob.size(); ++i) {
    check(cb[i].key() == "backbone." + ob[i].key(), "common buffer registration order");
    exact(cb[i].value(), ob[i].value(), "common buffers exact");
  }
}
void compare_state(rpb::PartitionedTemporalRelationModel &a, rpb::PartitionedTemporalRelationModel &b,
                   torch::optim::AdamW &ao, torch::optim::AdamW &bo) {
  const auto ap = a->named_parameters(), bp = b->named_parameters();
  check(ap.keys() == bp.keys() && ao.state().size() == bo.state().size(), "same named optimizer association");
  for (const auto &p : ap) {
    exact(p.value(), bp[p.key()], "direct/split parameter exact");
    const auto x = ao.state().find(p.value().unsafeGetTensorImpl()), y = bo.state().find(bp[p.key()].unsafeGetTensorImpl());
    check((x == ao.state().end()) == (y == bo.state().end()), "actual optimizer presence");
    if (x == ao.state().end()) continue;
    const auto *xs = dynamic_cast<const torch::optim::AdamWParamState *>(x->second.get());
    const auto *ys = dynamic_cast<const torch::optim::AdamWParamState *>(y->second.get());
    check(xs && ys && xs->step() == ys->step(), "actual AdamW step");
    exact(xs->exp_avg(), ys->exp_avg(), "first moment exact"); exact(xs->exp_avg_sq(), ys->exp_avg_sq(), "second moment exact");
  }
  const auto ab = a->named_buffers(), bb = b->named_buffers(); check(ab.keys() == bb.keys(), "named buffer association");
  for (const auto &v : ab) exact(v.value(), bb[v.key()], "direct/split buffers exact");
}
void model_graph() {
  const auto c = configuration(); const auto raw = input(c, 3); const auto scaler = rpb::fit_scaler(raw, c);
  const auto x = scaler.transform(raw, c);
  torch::manual_seed(16101); rpb::Model old(c);
  const auto cpu_rng = torch::rand({8}), cuda_rng = torch::rand({8}, torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(16101); rpb::PartitionedTemporalRelationModel candidate(c);
  exact(cpu_rng, torch::rand({8}), "partition constructor preserves common CPU RNG");
  exact(cuda_rng, torch::rand({8}, torch::TensorOptions().device(torch::kCUDA)), "partition constructor preserves common CUDA RNG");
  old->eval(); candidate->eval(); common_initialization(old, candidate);
  {
    torch::NoGradGuard guard; const auto base = old->encode(x), out = candidate->encode(x);
    exact(base.z_local, out.z_local, "local diagnostic unchanged"); exact(base.z_contextual, out.z_contextual, "per-channel contextual diagnostic unchanged");
    exact(base.z_global, out.z_global, "independent global diagnostic unchanged"); exact(base.sample_valid_mask, out.sample_valid_mask, "no scoring support filter");
    exact(out.z_contextual_global.narrow(1, 0, 20), base.z_contextual_global.narrow(1, 0, 20), "first20 common shape initializer");
    const auto a = rpb::temporal_relation_bank(x, c).values.narrow(1, 0, 108);
    exact(out.z_contextual_global.narrow(1, 20, 12), candidate->grouped_odd_relation_projection(a), "last12 contain only grouped odd projection");
    check(!torch::equal(base.z_contextual_global, out.z_contextual_global), "separate untrained candidate required");
    close(candidate->encode(reverse_time(x)).z_contextual_global.narrow(1, 20, 12), -out.z_contextual_global.narrow(1, 20, 12),
          "relation partition reverses sign independent of raw shape", 4e-6, 1e-6);
  }
  const auto q = rpb::make_training_mask(x.observed, c, 16103);
  const auto base = old->forward(x, q.hidden), out = candidate->forward(x, q.hidden);
  exact(out.target_counts, base.target_counts, "original query target denominator unchanged");
  exact(out.eligible_examples, base.eligible_examples, "original query eligibility unchanged");
  exact(out.reconstruction, candidate->decode(out.encoding.z_contextual_global, x.channel_ids), "sole original decoder receives exact partitioned32");
  auto changed = x; changed.data = torch::where(q.target, x.data + 1000., x.data);
  exact(candidate->forward(changed, q.hidden).encoding.z_contextual_global, out.encoding.z_contextual_global, "hidden Q values cannot enter either partition");
  auto poison = x; poison.observed = x.observed.clone(); poison.observed[0][1][7][2] = false;
  poison.data = torch::where(poison.observed, x.data, torch::full_like(x.data, std::numeric_limits<float>::quiet_NaN()));
  auto sanitized = poison; sanitized.data = torch::where(poison.observed, x.data, torch::zeros_like(x.data));
  exact(candidate->encode(poison).z_contextual_global, candidate->encode(sanitized).z_contextual_global, "naturally hidden NaNs cannot enter partitions");
  candidate->zero_grad(); out.loss.backward();
  for (const auto &w : {candidate->native_shape_projection->weight, candidate->grouped_odd_relation_projection->weight}) {
    finite(w.grad(), "both partition projection gradients finite CUDA");
    check(w.grad().is_cuda() && w.grad().abs().sum().item<double>() > 0, "both new modules learn through native32");
  }
  candidate->zero_grad(); auto leaf = x; leaf.data = x.data.detach().clone().set_requires_grad(true);
  candidate->forward(leaf, q.hidden).encoding.z_contextual_global.sum().backward();
  finite(leaf.data.grad(), "visible native input gradient finite");
  check(leaf.data.grad().masked_select(q.target).abs().sum().item<double>() == 0, "Q input gradient zero");
  check(leaf.data.grad().masked_select(q.visible).abs().sum().item<double>() > 0, "visible gradient remains");
  torch::manual_seed(16107); rpb::PartitionedTemporalRelationModel direct(c);
  torch::manual_seed(16107); rpb::PartitionedTemporalRelationModel split(c);
  direct->eval(); split->eval(); const auto initial_shape = split->native_shape_projection->weight.detach().clone();
  const auto initial_odd = split->grouped_odd_relation_projection->weight.detach().clone();
  torch::optim::AdamW da(direct->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  torch::optim::AdamW sa(split->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  auto step = [&](rpb::PartitionedTemporalRelationModel &model, torch::optim::AdamW &optimizer, int64_t attempt) {
    const auto mask = rpb::make_training_mask(x.observed, c, 16110 + attempt); optimizer.zero_grad();
    const auto value = model->forward(x, mask.hidden);
    check(value.loss.is_cuda() && value.encoding.z_contextual_global.sizes() == torch::IntArrayRef({3, 32}), "actual CUDA loss and sole32");
    value.loss.backward(); finite(model->grouped_odd_relation_projection->weight.grad(), "actual odd branch gradient");
    torch::nn::utils::clip_grad_norm_(model->parameters(), 1., 2., true); optimizer.step();
  };
  for (int64_t t = 0; t < 4; ++t) step(direct, da, t);
  for (int64_t t = 0; t < 2; ++t) step(split, sa, t);
  torch::Tensor preserved_native;
  { torch::NoGradGuard guard; preserved_native = split->encode(x).z_contextual_global.detach().clone(); }
  std::stringstream stream; torch::serialize::OutputArchive saved; split->save(saved); saved.save_to(stream); const auto preserved_bytes = stream.str();
  auto cpu_fixture = preserved_native.to(torch::kCPU).clone(); cpu_fixture.square_();
  for (int64_t t = 2; t < 4; ++t) step(split, sa, t);
  compare_state(direct, split, da, sa); check(stream.str() == preserved_bytes, "earlier serialized model bytes unchanged");
  rpb::PartitionedTemporalRelationModel restored(c); torch::serialize::InputArchive archive; archive.load_from(stream, torch::kCUDA); restored->load(archive); restored->eval();
  { torch::NoGradGuard guard; exact(restored->encode(x).z_contextual_global, preserved_native, "earlier CUDA snapshot immutable after continuation"); }
  check(!torch::equal(split->native_shape_projection->weight, initial_shape) && !torch::equal(split->grouped_odd_relation_projection->weight, initial_odd), "four updates change both projections");
  for (const auto &w : {split->native_shape_projection->weight, split->grouped_odd_relation_projection->weight}) {
    const auto it = sa.state().find(w.unsafeGetTensorImpl()); check(it != sa.state().end(), "both actual AdamW states present");
    const auto *adam = dynamic_cast<const torch::optim::AdamWParamState *>(it->second.get());
    check(adam && adam->step() == 4 && adam->exp_avg().is_cuda(), "both actual four-step CUDA moments");
  }
  {
    torch::NoGradGuard guard; const auto trained = split->encode(x);
    close(split->encode(reverse_time(x)).z_contextual_global.narrow(1, 20, 12), -trained.z_contextual_global.narrow(1, 20, 12), "time oddness survives actual updates", 4e-6, 1e-6);
    const auto order = torch::tensor({2, 0, 1}, torch::kInt64).to(torch::kCUDA); auto perm = x;
    perm.data = x.data.index_select(1, order); perm.observed = x.observed.index_select(1, order); perm.channel_ids = x.channel_ids.to(torch::kCUDA).index_select(0, order);
    close(split->encode(perm).z_contextual_global, trained.z_contextual_global, "semantic native permutation", 2e-5, 2e-6);
    auto absent = x; absent.observed = torch::zeros_like(x.observed); const auto empty = split->encode(absent);
    check(!empty.sample_valid_mask.any().item<bool>(), "all absent stays invalid"); exact(empty.z_contextual_global, torch::zeros_like(empty.z_contextual_global), "all-absent native32 zero");
  }
}
void closed_contract() {
  const auto c = configuration();
  for (int change = 0; change < 5; ++change) {
    auto invalid = c;
    if (change == 0) invalid.channel_mixer_placement = 0;
    if (change == 1) invalid.global_pool_input_source = 1;
    if (change == 2) invalid.temporal_difference_input = 1;
    if (change == 3) invalid.global_bottleneck_mode = 3;
    if (change == 4) invalid.device = torch::kCPU;
    rejects([&] { rpb::PartitionedTemporalRelationModel model(invalid); }, "closed early/CUDA contract");
  }
}
} // namespace

int main() try {
  check(torch::cuda::is_available(), "partitioned relation admission requires actual CUDA"); torch::set_num_threads(1);
  closed_contract(); odd_contract(); model_graph();
  std::cout << "Partitioned temporal relation CUDA admission passed\n" << ARCHITECTURE_SCREEN_SOURCE_ID << '\n'; return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
