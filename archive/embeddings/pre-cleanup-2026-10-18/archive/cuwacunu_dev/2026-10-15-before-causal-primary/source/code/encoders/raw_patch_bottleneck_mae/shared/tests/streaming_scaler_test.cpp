// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Artificial preprocessing checks. No model, classifier or market quality run.
#include "embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h"
#include <torch/cuda.h>
#include <iostream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
void check(bool condition, const char *message) { TORCH_CHECK(condition,"[streaming scaler test] ",message); }
template<class F> void rejects(F action, const char *message) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  check(rejected,message);
}
rpb::Config config(int64_t C=3, int64_t H=17, int64_t F=4) {
  rpb::Config c; c.channel_count=C; c.history_length=H; c.input_width=F;
  c.patch_length=1; c.device=torch::kCPU; c.scale_floor=.001;
  if (C == 3) c.channel_ids={30,-10,90};
  rpb::validate_config(c); return c;
}
rpb::Input slice(const rpb::Input &input, int64_t first, int64_t last) {
  return {input.data.slice(0,first,last),input.observed.slice(0,first,last),
      input.channel_ids.dim() == 2 ? input.channel_ids.slice(0,first,last) : input.channel_ids,
      input.endpoints.defined() ? input.endpoints.slice(0,first,last) : torch::Tensor(),input.sampling_interval};
}
rpb::Input fixture(const rpb::Config &c, bool reorder) {
  constexpr int64_t B=7;
  auto values = torch::empty({B,c.channel_count,c.history_length,c.input_width},torch::kFloat64);
  auto observed = torch::ones_like(values,torch::kBool);
  auto x = values.accessor<double,4>(); auto mask = observed.accessor<bool,4>();
  for (int64_t b=0; b<B; ++b) for (int64_t j=0; j<c.channel_count; ++j)
    for (int64_t t=0; t<c.history_length; ++t) for (int64_t f=0; f<c.input_width; ++f) {
      mask[b][j][t][f] = b != 0 && (b*7+j*3+t+f*2)%11 != 0;
      x[b][j][t][f] = f == 3 ? 1000+j : 100*j+3*f+.25*b+.5*t+((b+t+j)%3)*.125;
      if (!mask[b][j][t][f]) x[b][j][t][f] = (b+t+f)%2 ?
          std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    }
  auto ids = torch::tensor(rpb::resolved_channel_ids(c),torch::kInt64);
  if (reorder) {
    std::vector<torch::Tensor> rows, supports, identity;
    for (int64_t b=0; b<B; ++b) {
      const auto permutation = b%2 ? torch::tensor({2,0,1},torch::kInt64) : torch::tensor({1,2,0},torch::kInt64);
      rows.push_back(values[b].index_select(0,permutation));
      supports.push_back(observed[b].index_select(0,permutation));
      identity.push_back(ids.index_select(0,permutation));
    }
    values=torch::stack(rows); observed=torch::stack(supports); ids=torch::stack(identity);
  }
  return {values,observed,ids,torch::arange(B,torch::kFloat64),c.sampling_interval};
}
void same(const rpb::FrozenScaler &a, const rpb::FrozenScaler &b, const rpb::Config &c,
          double absolute=1e-11) {
  a.validate(c); b.validate(c);
  check(torch::equal(a.count,b.count) && torch::equal(a.channel_ids,b.channel_ids) &&
            torch::equal(a.floor_applied,b.floor_applied),"exact semantic counts/order/floor flags");
  check(torch::allclose(a.mean,b.mean,0.,absolute) && torch::allclose(a.scale,b.scale,0.,absolute),
        "population means/deviations agree within declared float64 reduction tolerance");
}
void masked_partition_and_merge() {
  const auto c=config(); const auto input=fixture(c,true);
  const auto reference=rpb::fit_scaler(input,c);
  rpb::StreamingScaler all(c); all.add_train_batch(input); same(all.finish(),reference,c);
  rpb::StreamingScaler partitioned(c);
  partitioned.add_train_batch(slice(input,0,1));
  check(partitioned.count().eq(0).all().item<bool>(),"an allmissing batch adds zero count without poison");
  rejects([&] { partitioned.finish(); },"empty support cannot produce a fitted scaler");
  partitioned.add_train_batch(slice(input,1,3)); partitioned.add_train_batch(slice(input,3,4));
  partitioned.add_train_batch(slice(input,4,7));
  same(partitioned.finish(),reference,c);
  rpb::StreamingScaler first(c),second(c),empty(c);
  first.add_train_batch(slice(input,0,3)); second.add_train_batch(slice(input,3,7));
  first.merge(second); first.merge(empty); same(first.finish(),reference,c);
  const auto canonical=fixture(c,false);
  rpb::StreamingScaler in_order(c); in_order.add_train_batch(canonical);
  same(in_order.finish(),reference,c);
  auto global_permutation=canonical;
  const auto order=torch::tensor({2,1,0},torch::kInt64);
  global_permutation.data=canonical.data.index_select(1,order);
  global_permutation.observed=canonical.observed.index_select(1,order);
  global_permutation.channel_ids=canonical.channel_ids.index_select(0,order);
  rpb::StreamingScaler reordered(c); reordered.add_train_batch(global_permutation);
  same(reordered.finish(),reference,c);
  check(reference.floor_applied.select(1,3).all().item<bool>() &&
            reference.scale.select(1,3).eq(c.scale_floor).all().item<bool>(),
        "constant coordinates apply the configured population-standard-deviation floor");
  auto count_copy=first.count(); count_copy.zero_();
  check(first.count().gt(0).all().item<bool>(),"count snapshots cannot mutate accumulator state");
  const auto frozen=first.finish(); const auto identity=frozen.identity();
  first.add_train_batch(slice(input,1,2));
  check(frozen.identity() == identity && !torch::equal(frozen.count,first.count()),
        "finished scaler owns a snapshot independent of later TRAIN accumulation");
}
void large_offset_and_population_std() {
  const auto c=config(1,4,1); const double offset=1099511627776.; // 2^40, exact in float64.
  const auto local=torch::tensor({-1.5,-.5,.5,1.5},torch::kFloat64).view({1,1,4,1});
  const auto shift=torch::tensor({-3.,-1.,1.,3.},torch::kFloat64).view({4,1,1,1});
  rpb::Input input{offset+local+shift,torch::ones({4,1,4,1},torch::kBool),
      torch::tensor({0},torch::kInt64),{},1.};
  rpb::StreamingScaler stream(c); stream.add_train_batch(slice(input,0,1));
  stream.add_train_batch(slice(input,1,3)); stream.add_train_batch(slice(input,3,4));
  const auto frozen=stream.finish();
  check(frozen.mean.item<double>() == offset && std::abs(frozen.scale.item<double>()-2.5) < 1e-12 &&
            frozen.count.item<int64_t>() == 16 && !frozen.floor_applied.item<bool>(),
        "large-offset float64 Chan moments retain exact mean and population variance6.25");
  same(frozen,rpb::fit_scaler(input,c),c,1e-12);
  // A finite constant near the float64 limit is valid. An empty stream must
  // install it without summing the offset or forming 0 * infinity.
  input.data=torch::full_like(input.data,1e308);
  rpb::StreamingScaler huge(c); huge.add_train_batch(input);
  check(huge.finish().mean.item<double>() == 1e308 && huge.finish().floor_applied.item<bool>(),
        "finite large constant has zero empirical deviation and a valid floor");
}
void invalid_and_incomplete() {
  const auto c=config(); const auto input=fixture(c,true);
  rpb::StreamingScaler stream(c); stream.add_train_batch(input); const auto identity=stream.finish().identity();
  auto nonfinite=input; nonfinite.data=input.data.clone(); nonfinite.observed=input.observed.clone();
  nonfinite.data[1][0][0][0]=std::numeric_limits<double>::quiet_NaN(); nonfinite.observed[1][0][0][0]=true;
  rejects([&] { stream.add_train_batch(nonfinite); },"observed nonfinite values must be rejected");
  auto duplicate=input; duplicate.channel_ids=input.channel_ids.clone(); duplicate.channel_ids[1][1]=duplicate.channel_ids[1][0];
  rejects([&] { stream.add_train_batch(duplicate); },"duplicate per-row semantic IDs must be rejected");
  auto unknown=input; unknown.channel_ids=input.channel_ids.clone(); unknown.channel_ids[1][0]=123456;
  rejects([&] { stream.add_train_batch(unknown); },"unknown semantic IDs must be rejected");
  auto wrong_shape=input; wrong_shape.data=input.data.slice(2,0,16); wrong_shape.observed=input.observed.slice(2,0,16);
  rejects([&] { stream.add_train_batch(wrong_shape); },"history geometry mismatch must be rejected");
  auto bad_mask=input; bad_mask.observed=input.observed.to(torch::kUInt8);
  rejects([&] { stream.add_train_batch(bad_mask); },"nonboolean support must be rejected");
  auto wrong_interval=input; wrong_interval.sampling_interval=2.;
  rejects([&] { stream.add_train_batch(wrong_interval); },"sampling interval mismatch must be rejected");
  rejects([&] { stream.add_train_batch(slice(input,0,0)); },"empty batch must be rejected");
  auto wrong_order=c; wrong_order.channel_ids={-10,30,90}; rpb::StreamingScaler incompatible(wrong_order);
  rejects([&] { stream.merge(incompatible); },"different configured semantic order must refuse merge");
  auto wrong_floor=c; wrong_floor.scale_floor=.01; rpb::StreamingScaler floor_mismatch(wrong_floor);
  rejects([&] { stream.merge(floor_mismatch); },"different floor must refuse merge");
  check(stream.finish().identity() == identity,"rejected input/merge leaves every accumulated coordinate unchanged");
  auto invalid=c; invalid.scale_floor=0;
  rejects([&] { rpb::StreamingScaler rejected(invalid); },"constructor validates existing configuration schema");
  auto incomplete=input; incomplete.observed=input.observed.clone(); incomplete.observed.select(3,2).fill_(false);
  rpb::StreamingScaler partial(c); partial.add_train_batch(incomplete);
  rejects([&] { partial.finish(); },"a globally unseen semantic feature cannot be silently floored");
  auto overflow_config=config(1,1,1);
  rpb::Input one{torch::ones({1,1,1,1},torch::kFloat64),torch::ones({1,1,1,1},torch::kBool),
      torch::tensor({0},torch::kInt64),{},1.};
  rpb::StreamingScaler overflow(overflow_config); overflow.add_train_batch(one);
  for (int i=0; i<62; ++i) overflow.merge(overflow);
  const auto count=overflow.count().item<int64_t>();
  check(count == (int64_t(1)<<62),"counts stay exact int64 beyond float64 integer precision");
  rejects([&] { overflow.merge(overflow); },"int64 count overflow must be rejected before mutation");
  check(overflow.count().item<int64_t>() == count,"overflow rejection preserves the previous count");
  auto explosive=one; explosive.data=torch::tensor({1e308,-1e308},torch::kFloat64).view({2,1,1,1});
  explosive.observed=torch::ones_like(explosive.data,torch::kBool);
  rejects([&] { overflow.add_train_batch(explosive); },"finite observations with unrepresentable float64 moments must be rejected");
  check(overflow.count().item<int64_t>() == count,"moment overflow rejection preserves the accumulator");
}
void logical_history_and_transform() {
  const auto c=config(1,129,2);
  auto raw=torch::arange(2*129*2,torch::kFloat64).view({2,1,129,2});
  auto observed=torch::ones_like(raw,torch::kBool); observed[0][0][7][1]=false;
  raw[0][0][7][1]=std::numeric_limits<double>::quiet_NaN();
  rpb::Input input{raw,observed,torch::tensor({0},torch::kInt64),{},1.};
  rpb::StreamingScaler stream(c); stream.add_train_batch(slice(input,0,1)); stream.add_train_batch(slice(input,1,2));
  const auto frozen=stream.finish(); same(frozen,rpb::fit_scaler(input,c),c);
  const auto cpu=frozen.transform(input,c);
  check(cpu.data.scalar_type() == torch::kFloat32 && cpu.data.sizes() == raw.sizes() &&
            torch::isfinite(cpu.data).all().item<bool>() && cpu.data[0][0][7][1].item<float>() == 0,
        "existing FrozenScaler transforms logical H129 and zeros masked storage");
  check(torch::cuda::is_available(),"actual CUDA required only for the existing scaler transfer witness");
  auto gpu_config=c; gpu_config.device=torch::Device(torch::kCUDA,0);
  const auto gpu=frozen.transform(input,gpu_config);
  check(gpu.data.is_cuda() && gpu.observed.is_cuda() &&
            torch::allclose(cpu.data,gpu.data.to(torch::kCPU),1e-6,1e-6) &&
            torch::equal(cpu.observed,gpu.observed.to(torch::kCPU)),
        "finished existing-schema scaler performs the one configured CUDA transfer without any encoder call");
  torch::cuda::synchronize();
}
} // namespace

int main() try {
  torch::set_num_threads(1);
  masked_partition_and_merge(); large_offset_and_population_std(); invalid_and_incomplete(); logical_history_and_transform();
  std::cout << "PASS bounded streaming TRAIN scaler: masked semantic float64 moments, variable batches/per-row reorder, "
      "partition/merge/reference equivalence, large offsets, population floor, strict incomplete/overflow guards, "
      "logical H129 and existing CUDA transfer; no encoder/head/quality execution\n";
  return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
