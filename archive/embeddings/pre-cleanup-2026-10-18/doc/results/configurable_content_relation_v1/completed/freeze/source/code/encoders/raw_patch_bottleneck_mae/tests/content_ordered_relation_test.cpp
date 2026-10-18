// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learned_ordered_relation_checkpoint.h"
#include <torch/cuda.h>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <unistd.h>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace d = rpb::configurable_checkpoint_detail;
namespace fs = std::filesystem;
void check(bool value, const char *message) { TORCH_CHECK(value, "[rpb-v22 CUDA test] ", message); }
template<class F> void rejects(F action, const char *message) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  check(rejected, message);
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  const auto x = a.detach().to(torch::kCPU).contiguous();
  const auto y = b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type() == y.scalar_type() && x.sizes() == y.sizes() &&
      (!x.numel() || !std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel() * x.element_size()));
}
void close(const torch::Tensor &a, const torch::Tensor &b, double tolerance, const char *message) {
  check(a.sizes() == b.sizes() && (a - b).abs().le(tolerance).all().item<bool>(), message);
}
struct FixtureDirectory {
  fs::path path;
  FixtureDirectory() {
    const auto pattern = (fs::temp_directory_path() / "rpb-v22-test-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
    const auto result = ::mkdtemp(name.data()); check(result, "private temporary directory creation");
    path = result;
  }
  FixtureDirectory(const FixtureDirectory &) = delete;
  FixtureDirectory &operator=(const FixtureDirectory &) = delete;
  ~FixtureDirectory() { std::error_code error; fs::remove_all(path, error); }
};
std::vector<char> file_bytes(const fs::path &path) {
  std::ifstream f(path, std::ios::binary); check(f.good(), "read artificial saved checkpoint");
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
rpb::Input fixture(const rpb::Config &c, int64_t batch = 2) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(c.device);
  const auto t = torch::arange(c.history_length, options).view({1, 1, c.history_length, 1});
  const auto ch = torch::arange(c.channel_count, options).view({1, c.channel_count, 1, 1});
  const auto f = torch::arange(c.input_width, options).view({1, 1, 1, c.input_width});
  const auto b = torch::arange(batch, options).view({batch, 1, 1, 1});
  auto values = (t * .19 + ch * .42 + f * .11 + b * .23).sin() * (1. + f * .07) +
      (t * .67 - ch * .27 + f * .12).cos() * .4 + ch * .05;
  auto mask = torch::ones_like(values, torch::kBool);
  mask[0][c.channel_count - 1].narrow(0, 4, 3).fill_(false);
  mask[1][0][2][0] = false;
  return {values, mask, torch::tensor(rpb::resolved_channel_ids(c), torch::kInt64),
          torch::full({batch}, double(c.history_length - 1), torch::kFloat64), c.sampling_interval};
}
rpb::Config small_config() {
  auto c = rpb::default_content_ordered_relation_config();
  c.history_length = 33; c.input_width = 3; c.encoder_width = 16;
  c.num_layers = 1; c.feedforward_width = 32; c.decoder_hidden_width = 16;
  c.channel_ids = {31, 11, 21};
  return c;
}

// Scalar CPU calculation over arrays is an independent numerical witness,
// not a CPU model forward or a second training execution.
void scalar_bank_witness() {
  auto c = small_config(); c.channel_count = 2; c.channel_ids = {91, 17};
  c.history_length = 17; c.input_width = 1; c.export_width = 8;
  auto input = fixture(c);
  const auto actual = rpb::content_ordered_relation_bank(input, c);
  const auto x = input.data.to(torch::kCPU).contiguous();
  const auto m = input.observed.to(torch::kCPU).contiguous();
  const auto v = x.accessor<float, 4>(); const auto mask = m.accessor<bool, 4>();
  auto expected = torch::zeros({2, 8}, torch::kFloat64);
  auto counts = torch::zeros({2, 4}, torch::kInt64);
  auto eligible = torch::zeros({2, 4}, torch::kBool);
  auto e = expected.accessor<double, 2>(); auto n = counts.accessor<int64_t, 2>();
  auto supported = eligible.accessor<bool, 2>();
  for (int64_t b = 0; b < 2; ++b) {
    for (size_t k = 0; k < rpb::kContentOrderedRelationLags.size(); ++k) {
      const auto lag = rpb::kContentOrderedRelationLags[k];
      std::vector<std::array<double, 4>> endpoints;
      double am = 0., bm = 0.;
      for (int64_t t = 0; t < c.history_length - lag; ++t) {
        if (!(mask[b][0][t][0] && mask[b][0][t + lag][0] &&
              mask[b][1][t][0] && mask[b][1][t + lag][0])) continue;
        endpoints.push_back({v[b][0][t][0], v[b][0][t + lag][0],
                             v[b][1][t][0], v[b][1][t + lag][0]});
        am += endpoints.back()[0] + endpoints.back()[1];
        bm += endpoints.back()[2] + endpoints.back()[3];
      }
      const auto count = static_cast<int64_t>(endpoints.size());
      n[b][k] = count;
      am /= 2. * double(std::max(int64_t{1}, count));
      bm /= 2. * double(std::max(int64_t{1}, count));
      double ea = 0., eb = 0., odd = 0., even = 0.;
      for (const auto &p : endpoints) {
        const auto a0 = p[0] - am, a1 = p[1] - am, b0 = p[2] - bm, b1 = p[3] - bm;
        ea += a0*a0 + a1*a1; eb += b0*b0 + b1*b1;
        odd += a0*b1 - a1*b0; even += a0*b0 + a1*b1;
      }
      const auto divisor = double(std::max(int64_t{1}, count)); ea /= divisor; eb /= divisor;
      supported[b][k] = count >= 2 && ea > 1e-12 && eb > 1e-12;
      if (supported[b][k]) { e[b][k] = odd / divisor / std::sqrt(ea*eb); e[b][4+k] = even / divisor / std::sqrt(ea*eb); }
    }
  }
  close(actual.values.to(torch::kFloat64), expected.to(c.device), 1e-6, "CUDA moments agree with independent scalar array calculation");
  check(exact(actual.common_counts, counts) && exact(actual.eligible, eligible), "scalar common support and eligibility agree");
  auto reversed = input; reversed.data = input.data.flip({2}); reversed.observed = input.observed.flip({2});
  const auto reverse = rpb::content_ordered_relation_bank(reversed, c);
  close(actual.values.narrow(1,0,4), -reverse.values.narrow(1,0,4), 1e-6, "odd moments reverse sign under joint original-time reversal");
  close(actual.values.narrow(1,4,4), reverse.values.narrow(1,4,4), 1e-6, "even moments survive time reversal");
  check(exact(actual.common_counts, reverse.common_counts) && exact(actual.eligible, reverse.eligible), "time reversal preserves support");
  auto swapped = input; swapped.data = input.data.flip({1}); swapped.observed = input.observed.flip({1});
  const auto swap = rpb::content_ordered_relation_bank(swapped,c);
  close(actual.values.narrow(1,0,4), -swap.values.narrow(1,0,4), 1e-6, "semantic value swap changes odd sign");
  close(actual.values.narrow(1,4,4), swap.values.narrow(1,4,4), 1e-6, "semantic value swap preserves even moments");
  auto affine = input; affine.data = input.data.clone();
  affine.data.select(1,0).mul_(3.).add_(5.); affine.data.select(1,1).mul_(.4).add_(-2.);
  close(actual.values, rpb::content_ordered_relation_bank(affine,c).values, 2e-6, "joint centering and energy normalization reject positive gains and offsets");
  auto constant = input; constant.data = torch::ones_like(input.data) * 4.;
  const auto zero_energy = rpb::content_ordered_relation_bank(constant,c);
  check(zero_energy.values.narrow(1,0,8).eq(0).all().item<bool>() &&
            !zero_energy.eligible.any().item<bool>() &&
            exact(zero_energy.common_counts, actual.common_counts), "zero-energy moments are ineligible finite zero but counts remain honest");
  auto scarce = input; scarce.observed = torch::zeros_like(input.observed);
  scarce.observed.narrow(2,0,2).fill_(true);
  check(!rpb::content_ordered_relation_bank(scarce,c).eligible.any().item<bool>(), "fewer than two common positions is ineligible");
}

// Perfect alternating correlations have the same nonzero normalized content
// after redundant observations are removed, despite changed counts at every lag.
// This tests an active learned branch, not equality hidden by zero initialization.
void content_without_density_witness() {
  auto c = small_config(); c.history_length = 128;
  auto input = fixture(c);
  const auto x = input.data.options();
  const auto t = torch::arange(c.history_length, x).view({1,1,c.history_length,1});
  const auto ch = torch::arange(c.channel_count, x).view({1,c.channel_count,1,1});
  const auto f = torch::arange(c.input_width, x).view({1,1,1,c.input_width});
  const auto sign = 1. - (ch + f).remainder(2) * 2.;
  input.data = (1. - t.remainder(2) * 2.) * sign;
  input.data = input.data.expand({2,c.channel_count,c.history_length,c.input_width}).contiguous();
  input.observed = torch::ones_like(input.data, torch::kBool);
  auto reduced = input; reduced.observed = input.observed.clone();
  for (int64_t tick = 0; tick < c.history_length; tick += 16)
    reduced.observed.select(2,tick).fill_(false);
  const auto full = rpb::content_ordered_relation_bank(input,c);
  const auto sparse = rpb::content_ordered_relation_bank(reduced,c);
  const auto N = rpb::content_ordered_relation_moment_width(c);
  check(full.values.size(1) == N * 2 && full.eligible.all().item<bool>() &&
            sparse.eligible.all().item<bool>(), "both content views retain valid finite-energy comparisons");
  check(full.common_counts.gt(sparse.common_counts).all().item<bool>(),
        "redundant masking changes common support at every pair/feature/lag");
  check(full.values.narrow(1,0,N).eq(0).all().item<bool>() &&
            full.values.narrow(1,N,N).abs().eq(1).all().item<bool>(),
        "alternating witness has zero odd and nonzero unit even content");
  check(exact(full.values,sparse.values), "identical normalized content is byte-exact across differing support");
  const auto historical_full = rpb::learned_ordered_relation_bank(input,c);
  const auto historical_sparse = rpb::learned_ordered_relation_bank(reduced,c);
  check(exact(full.values,historical_full.values.narrow(1,0,N*2)) &&
            exact(sparse.values,historical_sparse.values.narrow(1,0,N*2)),
        "content arithmetic preserves v21 odd/even bytes while omitting its density block");
  check(!exact(historical_full.values.narrow(1,N*2,N),
               historical_sparse.values.narrow(1,N*2,N)),
        "the removed historical density block distinguishes these masks");
  rpb::ContentOrderedRelationModel model(c);
  torch::NoGradGuard guard;
  auto weights = torch::arange(model->relation_projection->weight.numel(), x)
      .reshape(model->relation_projection->weight.sizes());
  model->relation_projection->weight.copy_(.01 + (weights.remainder(17)+1.) * .002);
  const auto a = model->relation_projection(full.values);
  const auto b = model->relation_projection(sparse.values);
  check(a.abs().max().item<double>() > 0 && exact(a,b),
        "nonzero content projection cannot learn an additive density shift for equal moments");
  model->validate_state();
}

void initialized_parity_and_masks() {
  auto c = small_config(); const auto input = fixture(c);
  torch::manual_seed(22101); rpb::ConfigurableSpectralModel reference(c);
  const auto cpu_next = torch::rand({8}); const auto cuda_next = torch::rand({8}, input.data.options());
  torch::manual_seed(22101); rpb::ContentOrderedRelationModel candidate(c);
  check(exact(cpu_next,torch::rand({8})) && exact(cuda_next,torch::rand({8}, input.data.options())),
        "zero projection preserves both CPU and CUDA initialization RNG streams");
  for (const auto &p : reference->named_parameters())
    check(candidate->base->named_parameters().contains(p.key()) &&
              exact(p.value(),candidate->base->named_parameters()[p.key()]), "every common initialized parameter is byte-identical");
  check(candidate->relation_projection->weight.eq(0).all().item<bool>(), "new projection is literal zero without bias");
  reference->eval(); candidate->eval(); torch::NoGradGuard guard;
  auto deleted = input; deleted.observed = input.observed.clone(); deleted.observed.narrow(2,9,5).fill_(false);
  auto absent = input; absent.observed = torch::zeros_like(input.observed);
  absent.data = torch::full_like(input.data, -0.);
  for (const auto &view : {input, deleted, absent}) {
    const auto base = reference->encode(view), next = candidate->encode(view);
    check(exact(base.z_contextual_global,next.z_contextual_global), "zero-start complete export matches v20 bytes in intact/deleted/allmissing views");
    check(exact(reference->decode(base.z_contextual_global,view.channel_ids),
                candidate->decode(next.z_contextual_global,view.channel_ids)), "zero-start reconstruction matches v20 bytes");
  }
  const auto before = candidate->encode(input).z_contextual_global;
  auto poison = input; poison.data = torch::where(input.observed,input.data,
      torch::full_like(input.data,std::numeric_limits<float>::quiet_NaN()));
  check(exact(before,candidate->encode(poison).z_contextual_global), "masked NaN storage never enters native export");
  check(exact(rpb::content_ordered_relation_bank(input,c).values,
              rpb::content_ordered_relation_bank(poison,c).values), "masked poison never enters relation bank");
  auto missing = poison; missing.observed = torch::zeros_like(input.observed);
  const auto empty = candidate->encode(missing);
  check(empty.z_contextual_global.eq(0).all().item<bool>() && !empty.sample_valid_mask.any().item<bool>(), "allmissing window is finite invalid zero");
  check(rpb::content_ordered_relation_bank(missing,c).values.eq(0).all().item<bool>(), "allmissing relation bank is exact zero");
  // Give the new branch nonzero weights so invariance cannot pass by zero gating.
  candidate->relation_projection->weight.fill_(.0125);
  const auto active_empty = candidate->encode(missing);
  check(active_empty.z_contextual_global.eq(0).all().item<bool>() &&
            !active_empty.sample_valid_mask.any().item<bool>(),
        "allmissing remains invalid zero with a nonzero relation projection");
  auto permuted = input;
  const auto order = torch::tensor({2,0,1},input.channel_ids.options());
  permuted.data = input.data.index_select(1,order.to(c.device));
  permuted.observed = input.observed.index_select(1,order.to(c.device));
  permuted.channel_ids = input.channel_ids.index_select(0,order);
  check(exact(rpb::content_ordered_relation_bank(input,c).values,
              rpb::content_ordered_relation_bank(permuted,c).values),
        "canonical relation bank is byte-identical across storage permutation");
  const auto canonical_native=candidate->encode(input).z_contextual_global;
  const auto permuted_native=candidate->encode(permuted).z_contextual_global;
  const auto base_difference=(candidate->base->encode(input).z_contextual_global-
      candidate->base->encode(permuted).z_contextual_global).abs().max().item<double>();
  std::cout<<"Permutation F32 roundoff: historical base "<<base_difference
           <<", candidate "<<(canonical_native-permuted_native).abs().max().item<double>()<<'\n';
  // Attention sums physical storage order; its mathematical permutation
  // invariance permits bounded F32 roundoff. Initial/control reuse above still
  // requires byte equality at identical storage order, including signed zeros.
  close(canonical_native,permuted_native,2e-6,
        "semantic identity preserves export across storage permutation within F32 roundoff");
  auto batched = permuted;
  batched.channel_ids = permuted.channel_ids.unsqueeze(0).repeat({2,1});
  batched.data = permuted.data.clone(); batched.observed = permuted.observed.clone();
  const auto second_order = torch::tensor({1,2,0},order.options()).to(c.device);
  batched.data[1].copy_(permuted.data[1].index_select(0,second_order));
  batched.observed[1].copy_(permuted.observed[1].index_select(0,second_order));
  batched.channel_ids[1].copy_(permuted.channel_ids.index_select(0,second_order.to(torch::kCPU)));
  check(exact(rpb::content_ordered_relation_bank(input,c).values,
              rpb::content_ordered_relation_bank(batched,c).values),
        "independent batched semantic permutations preserve exact relation bank");
  close(canonical_native,candidate->encode(batched).z_contextual_global,2e-6,
        "independent batched semantic permutations preserve native output within F32 roundoff");
}

void generic_shapes() {
  for (const auto &geometry : {std::array<int64_t,4>{1,17,1,8}, {2,65,3,16}, {4,129,9,48}}) {
    auto c = small_config(); c.channel_count=geometry[0]; c.history_length=geometry[1];
    c.input_width=geometry[2]; c.export_width=geometry[3]; c.channel_ids.clear();
    const auto input = fixture(c); rpb::ContentOrderedRelationModel model(c);
    torch::NoGradGuard guard; model->eval(); const auto encoded = model->encode(input);
    check(encoded.z_contextual_global.is_cuda() && encoded.z_contextual_global.sizes()==torch::IntArrayRef({2,c.export_width}), "generic C/H/F/D native shape on CUDA");
    check(model->decode(encoded.z_contextual_global,input.channel_ids).sizes()==input.data.sizes(), "nondivisible logical history crops decoder output");
    check(rpb::content_ordered_relation_bank(input,c).values.size(1)==rpb::content_ordered_relation_bank_width(c), "configured bank dimension");
    if (c.channel_count==1) {
      check(model->relation_projection->weight.numel()==0, "C1 has no cross-channel projection parameters");
      check(exact(encoded.z_contextual_global,model->base->encode(input).z_contextual_global), "C1 remains its exact v20 base");
    }
  }
}

void write_inconsistent(const fs::path &path, const rpb::Config &metadata,
                        const rpb::ContentOrderedRelationModel &model, const rpb::FrozenScaler &scaler,
                        bool wrong_lags=false, const std::string &mutation={}) {
  torch::serialize::OutputArchive a,m,s;
  a.write("artifact_kind",d::text_tensor(rpb::kContentOrderedRelationCheckpointKind),true);
  a.write("model_tag",d::text_tensor(mutation=="tag"?"RPB-v21":rpb::kContentOrderedRelationModelTag),true);
  a.write("architecture_id",d::text_tensor(mutation=="architecture"?"wrong-architecture":rpb::kContentOrderedRelationArchitectureId),true);
  a.write("native_layout",d::text_tensor(mutation=="layout"?"wrong-layout":rpb::kContentOrderedRelationLayout),true);
  a.write("dtype",d::text_tensor(mutation=="dtype"?"float64":"float32"),true);
  a.write("training_device",d::text_tensor(mutation=="device"?"cpu":metadata.device.str()),true);
  a.write("config_integer_v1",torch::tensor(d::integers(metadata),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(metadata),torch::kFloat64),true);
  a.write("channel_ids",torch::tensor(rpb::resolved_channel_ids(metadata),torch::kInt64),true);
  a.write("relation_lags_v1",torch::tensor(std::vector<int64_t>{1,2,4,wrong_lags?7:8},torch::kInt64),true);
  a.write("relation_energy_floor",torch::tensor(mutation=="floor"?1e-6:1e-12,torch::kFloat64),true);
  a.write("relation_minimum_common",torch::tensor(mutation=="minimum"?int64_t{3}:int64_t{2},torch::kInt64),true);
  a.write("training_updates",torch::tensor(mutation=="updates"?int64_t{-1}:int64_t{2},torch::kInt64),true);
  a.write("scaler_identity",d::text_tensor(scaler.identity()),true);
  model->save(m); scaler.save(s); a.write("model",m); a.write("scaler",s); a.save_to(path.string());
}

void gradients_and_checkpoints() {
  auto c = small_config(); c.history_length=65;
  const auto raw = fixture(c); const auto scaler = rpb::fit_configurable_spectral_scaler(raw,c);
  auto input = rpb::transform_configurable_spectral_input(scaler,raw,c);
  input.data = input.data.to(c.device).detach().set_requires_grad(true);
  rpb::ContentOrderedRelationModel model(c);
  const auto initial_fixed = model->encode(input).z_contextual_global.narrow(1,20,12).detach().clone();
  const auto initial = model->relation_projection->weight.detach().clone();
  torch::optim::AdamW optimizer(model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  for (int64_t step=0;step<2;++step) {
    optimizer.zero_grad(); input.data.mutable_grad()=torch::Tensor();
    const auto mask = rpb::configurable_spectral_make_training_mask(input.observed,c,22110+step);
    const auto output = model->forward(input,mask.hidden);
    check(output.loss.is_cuda() && torch::isfinite(output.loss).item<bool>(), "masked Huber loss executes on actual CUDA");
    output.loss.backward();
    const auto grad = model->relation_projection->weight.grad();
    check(grad.defined() && grad.is_cuda() && torch::isfinite(grad).all().item<bool>() &&
              grad.abs().sum().item<double>()>0, "zero-start relation projection receives finite nonzero Huber gradient");
    check(input.data.grad().defined() && input.data.grad().is_cuda() &&
              torch::isfinite(input.data.grad()).all().item<bool>(), "input derivatives remain finite on CUDA");
    const auto actual_observed = input.observed.to(c.device);
    check(input.data.grad().masked_select(actual_observed.logical_not()).eq(0).all().item<bool>(), "missing input coordinates have zero derivative");
    for (const auto &p:model->named_parameters()) {
      check(p.value().is_cuda(), "every model parameter remains on CUDA");
      if(p.value().grad().defined()) check(p.value().grad().is_cuda() &&
          torch::isfinite(p.value().grad()).all().item<bool>(), "every actual parameter gradient is finite CUDA");
    }
    torch::nn::utils::clip_grad_norm_(model->parameters(),1.); optimizer.step(); model->validate_state();
  }
  check(!exact(initial,model->relation_projection->weight), "optimizer changes the new relation projection");
  torch::NoGradGuard guard; model->eval();
  const auto native=model->encode(input).z_contextual_global;
  check(exact(initial_fixed,native.narrow(1,20,12)), "fixed spectral coordinates stay exact after both optimizer steps");
  model->validate_fixed_prior();
  FixtureDirectory directory; const auto file=directory.path/"trained.pt";
  rpb::save_content_ordered_relation_checkpoint(file.string(),model,scaler,2);
  const auto original=file_bytes(file);
  rejects([&]{rpb::save_content_ordered_relation_checkpoint(file.string(),model,scaler,3);}, "existing inference checkpoint refuses overwrite");
  check(file_bytes(file)==original,"rejected overwrite preserves bytes");
  auto loaded=rpb::load_content_ordered_relation_checkpoint(file.string(),c.device,&c);
  check(loaded.training_updates==2 && loaded.scaler.identity()==scaler.identity(), "typed checkpoint retains updates/scaler");
  check(d::same_config(c,loaded.config),"full resolved config and semantic order round trip");
  check(exact(native,loaded.model->encode(input).z_contextual_global),"trained native coordinates reload byte-identically");
  check(exact(model->decode(native,input.channel_ids),loaded.model->decode(native,input.channel_ids)),"trained reconstruction reloads byte-identically");
  check(exact(model->relation_projection->weight,loaded.model->relation_projection->weight),"new relation weights reload exactly");
  rejects([&]{rpb::load_configurable_spectral_checkpoint(file.string(),c.device,&c);},"v20 loader refuses v22 typed archive");
  const auto base_file=directory.path/"base.pt";
  rpb::save_configurable_spectral_checkpoint(base_file.string(),model->base,scaler,2);
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(base_file.string(),c.device,&c);},"v22 loader refuses v20 typed archive");
  const auto learned_file=directory.path/"v21.pt";
  rpb::LearnedOrderedRelationModel stopped_design(c);
  rpb::save_learned_ordered_relation_checkpoint(learned_file.string(),stopped_design,scaler,0);
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(learned_file.string(),c.device,&c);},
          "v22 loader refuses the distinct v21 typed archive");
  rejects([&]{rpb::load_learned_ordered_relation_checkpoint(file.string(),c.device,&c);},
          "v21 loader refuses the distinct v22 typed archive");
  for(const std::string field : {"tag","architecture","layout","dtype","device","floor","minimum","updates"}) {
    const auto malformed=directory.path/("bad-"+field+".pt");
    write_inconsistent(malformed,c,model,scaler,false,field);
    rejects([&]{rpb::load_content_ordered_relation_checkpoint(malformed.string(),c.device,&c);},
            "malformed typed schema/support/update contract refuses load");
  }
  auto wrong=c; wrong.history_length=66;
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(file.string(),c.device,&wrong);},"same-private-padding but different logical history refuses load");
  wrong=c; wrong.channel_ids={11,31,21};
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(file.string(),c.device,&wrong);},"different semantic order refuses load");
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(file.string(),torch::kCPU);},"CPU model loader refuses execution");
  const auto bad_lags=directory.path/"bad-lags.pt"; write_inconsistent(bad_lags,c,model,scaler,true);
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(bad_lags.string(),c.device,&c);},"wrong relation lag metadata refuses load");
  const auto weight_backup=model->relation_projection->weight.clone();
  model->relation_projection->weight[0][0]=std::numeric_limits<float>::quiet_NaN();
  const auto bad_state=directory.path/"bad-state.pt"; write_inconsistent(bad_state,c,model,scaler);
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(bad_state.string(),c.device,&c);},"nonfinite relation weight refuses load");
  const auto bad_save=directory.path/"must-not-exist.pt";
  rejects([&]{rpb::save_content_ordered_relation_checkpoint(bad_save.string(),model,scaler,2);},"nonfinite relation weight refuses save");
  check(!fs::exists(bad_save),"invalid state never publishes a checkpoint");
  model->relation_projection->weight.copy_(weight_backup);
  // Corrupt only the added module, with an otherwise correct historical child.
  // This exercises the new wrapper's own geometry contract independently.
  const auto good_geometry=model->relation_projection->weight.clone();
  model->relation_projection->weight.set_data(torch::zeros(
      {good_geometry.size(0),good_geometry.size(1)-1},good_geometry.options()));
  const auto bad_projection=directory.path/"bad-projection-shape.pt";
  write_inconsistent(bad_projection,c,model,scaler);
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(bad_projection.string(),c.device,&c);},
          "wrong added projection geometry refuses load with correct base geometry");
  model->relation_projection->weight.set_data(good_geometry);
  model->validate_state();
  auto wide=c; wide.export_width=48; rpb::ContentOrderedRelationModel wrong_width(wide);
  const auto bad_shape=directory.path/"bad-shape.pt"; write_inconsistent(bad_shape,c,wrong_width,scaler);
  rejects([&]{rpb::load_content_ordered_relation_checkpoint(bad_shape.string(),c.device,&c);},"matching typed header with wrong learned geometry refuses load");
}
} // namespace

int main(int argc,char **argv) try {
  check(torch::cuda::is_available(),"actual CUDA is required"); torch::set_num_threads(1);
  const auto defaults=rpb::content_ordered_relation_settings(cuwacunu::config::Document::parse("ENCODER {}"));
  check(defaults.history_length==128 && defaults.input_width==9 && defaults.export_width==32,"candidate settings defaults");
  check(rpb::content_ordered_relation_bank_width(defaults)==1944,"C3/F9 has 1944 content-only relation coordinates");
  check(rpb::configurable_spectral_shape_width(defaults) *
            rpb::content_ordered_relation_bank_width(defaults) == 38880,
        "default content-only projection has 38880 trainable values");
  if(argc>1) check(d::same_config(defaults,rpb::read_content_ordered_relation_settings(argv[1])),"settings file agrees with candidate defaults");
  for(const auto *text:{"ENCODER { MODEL_TAG=RPB-v21; }","ENCODER { MODEL_TAG=RPB-v20; }","ENCODER { DEVICE=cpu; }",
      "ENCODER { HISTORY_LENGTH=16; }","ENCODER { CHANNEL_IDS=1,1,2; }",
      "ENCODER { INPUT_WIDTH=0; }","ENCODER { RELATION_LAGS=1,3; }"})
    rejects([&]{rpb::content_ordered_relation_settings(cuwacunu::config::Document::parse(text));},"unsupported candidate settings accepted");
  scalar_bank_witness(); content_without_density_witness(); initialized_parity_and_masks(); generic_shapes(); gradients_and_checkpoints();
  torch::cuda::synchronize();
  std::cout<<"PASS RPB-v22 artificial CUDA engineering: independent scalar moments, odd/even reversal, "
      "original common masks, equal content/different support, semantic order, zero-start v20 bytes/RNG, generic geometry, actual Huber gradients/updates, "
      "fixed-prior parity and separate immutable typed checkpoints\n";
  return 0;
} catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
