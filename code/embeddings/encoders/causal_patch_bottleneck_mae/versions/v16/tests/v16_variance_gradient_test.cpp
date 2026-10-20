// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Synthetic CUDA decoder mechanism proof only; no checkpoint, old statistics,
// encoder forward, real training trajectory or quality readout is executed.
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v16.h"
#include <torch/cuda.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace cp = embedding::encoders::causal_patch_bottleneck_mae::v16;
namespace impl = cp::implementation;
int64_t checks = 0, cuda_decode_calls = 0;

void check(bool condition, const char *message) {
  ++checks;
  TORCH_CHECK(condition, "[CPB-v16 variance gradient proof] ", message);
}

bool nonzero_gradient(const torch::Tensor &value) {
  return value.grad().defined() && value.grad().is_cuda() &&
      torch::isfinite(value.grad()).all().item<bool>() &&
      value.grad().abs().sum().item<double>() > 0.;
}

bool zero_or_absent_gradient(const torch::Tensor &value) {
  return !value.grad().defined() ||
      (value.grad().is_cuda() && torch::isfinite(value.grad()).all().item<bool>() &&
       value.grad().eq(0).all().item<bool>());
}

void clear(impl::LocationScaleDecoderModel &model) {
  for (auto &value : model->parameters()) value.mutable_grad() = torch::Tensor();
}

struct FixtureStatistics {
  cp::FrozenScaler scaler;
  cp::BankStandardizer bank;
};

FixtureStatistics statistics(const cp::Config &config) {
  constexpr int64_t B = 8, C = 3, H = 128, F = 9;
  auto raw = torch::zeros({B,C,H,F}, torch::kFloat64);
  auto values = raw.accessor<double,4>();
  for (int64_t b=0;b<B;++b) for (int64_t c=0;c<C;++c)
    for (int64_t h=0;h<H;++h) for (int64_t f=0;f<F;++f)
      values[b][c][h][f] = 10.+2.*c+.2*f+.01*h+.07*b+
          std::sin(.13*h+.19*f+.31*c+.17*b);
  const auto observed = torch::ones_like(raw, torch::kBool);
  const auto ids = torch::tensor({0,1,2}, torch::kInt64);
  const auto times = torch::arange(H, torch::kFloat64)*60000.;
  const cp::RawHistory history{raw,observed,ids,times,60000.};
  const auto retained = cp::boundary::retain(history,
      torch::full({B},127.*60000.,torch::kFloat64),config);
  const impl::Input rounded{retained.raw.data.to(torch::kFloat32),
      retained.raw.observed,retained.raw.channel_ids,retained.raw.endpoints,
      retained.raw.sampling_interval};
  auto scaler = impl::fit_configurable_spectral_scaler(rounded,config.level);
  cp::BankStandardizer bank(config.level,
      "v16-variance-gradient-engineering-generated-TRAIN-only",scaler.identity());
  bank.add(impl::relative_change_targets(retained.raw,config.level));
  bank.freeze();
  const auto scaler_id = scaler.identity();
  bank.validate(config.level,nullptr,&scaler_id);
  check(bank.frozen && bank.support.all().item<bool>(),
      "synthetic TRAIN-only bank calibrated without any model or old data");
  return {scaler,bank};
}

torch::Tensor native_leaf(const torch::TensorOptions &options) {
  return torch::linspace(-.4,.7,72,options).reshape({2,36}).clone().set_requires_grad(true);
}

impl::LocationScaleDecoderTrace decode(impl::LocationScaleDecoderModel &model,
    const torch::Tensor &native,const torch::Tensor &ids) {
  ++cuda_decode_calls;
  auto out = model->decode_with_trace(native,ids);
  check(out.mean.is_cuda() && out.variance.is_cuda() &&
      out.mean.scalar_type()==torch::kFloat32 && out.variance.scalar_type()==torch::kFloat32,
      "actual CUDA F32 conditional distribution");
  return out;
}

// Independent expression of the numerically unchanged ATTACHED v3 formula.
// This positive control intentionally has the native gradient that v16 blocks.
torch::Tensor attached_variance(impl::LocationScaleDecoderModel &model,
    const torch::Tensor &native,const torch::Tensor &ids) {
  const auto z = native.narrow(1,32,4);
  const auto score = torch::matmul(z,model->variance_projection.transpose(0,1))+
      model->variance_bias;
  const auto constants = impl::location_scale_decoder_constants();
  const auto epsilon = torch::tensor(constants[0],native.options());
  const auto shift = torch::tensor(constants[1],native.options());
  const auto coefficient = torch::ones({},native.options())-epsilon;
  const auto u = torch::softplus(score+shift,
      impl::kLocationScaleDecoderSoftplusBeta,impl::kLocationScaleDecoderSoftplusThreshold);
  const auto u0 = torch::softplus(shift,
      impl::kLocationScaleDecoderSoftplusBeta,impl::kLocationScaleDecoderSoftplusThreshold);
  const auto canonical = (epsilon+coefficient*(u/u0)).reshape({2,3,1,9});
  const auto semantic = impl::channel_indices(ids,model->config().level,2,native.device());
  return canonical.expand({2,3,128,9}).gather(1,
      semantic.unsqueeze(-1).unsqueeze(-1).expand({2,3,128,9}));
}

torch::Tensor support(const torch::Device &device) {
  auto result = torch::zeros({2,3,128,9},torch::kBool);
  auto values = result.accessor<bool,4>();
  for(int64_t b=0;b<2;++b) for(int64_t c=0;c<3;++c)
    for(int64_t h=0;h<128;++h) for(int64_t f=0;f<9;++f)
      values[b][c][h][f] = !(b==1 && c==2) && (h+2*f+c+b)%(3+c)==0;
  return result.to(device);
}

torch::Tensor manual_mean_path(impl::LocationScaleDecoderModel &model,
    const impl::LocationScaleDecoderTrace &distribution,const torch::Tensor &target,
    const torch::Tensor &mask,const torch::Tensor &ids) {
  torch::NoGradGuard guard;
  const auto counts = mask.sum(std::vector<int64_t>{2,3});
  const auto eligible = counts.gt(0);
  const auto examples = eligible.any(1).sum().item<int64_t>();
  const auto channel_count = eligible.sum(1).clamp_min(1).to(torch::kFloat64);
  const auto denominator = counts.clamp_min(1).to(torch::kFloat64)*
      channel_count.unsqueeze(1)*double(examples);
  const auto error = distribution.mean.to(torch::kFloat64)-target.to(torch::kFloat64);
  const auto raw_derivative = error/distribution.variance.to(torch::kFloat64);
  const auto dmean = torch::where(mask,raw_derivative,torch::zeros_like(raw_derivative))/
      denominator.unsqueeze(-1).unsqueeze(-1);
  const auto semantic = impl::channel_indices(ids,model->config().level,2,target.device());
  const auto canonical = model->encoder->change_decoder.detach()
      .reshape({1,3,128,9,4}).expand({2,3,128,9,4});
  const auto decoder_in_storage_order = canonical.gather(1,
      semantic.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand({2,3,128,9,4}));
  return (dmean.unsqueeze(-1)*decoder_in_storage_order.to(torch::kFloat64))
      .sum(std::vector<int64_t>{1,2,3}).to(torch::kFloat32);
}
} // namespace

int main(int argc,char **argv) try {
  check(argc==3 && std::string_view(argv[1])=="--settings",
      "usage: v16_variance_gradient_test --settings CONFIG");
  check(std::filesystem::is_regular_file(argv[2]),"existing version-owned configuration");
  check(torch::cuda::is_available(),"actual managed CUDA runtime required");
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  const auto config = cp::read_settings(argv[2]);
  check(config.level.device==torch::Device(torch::kCUDA,0) &&
      config.level.channel_count==3 && config.level.history_length==128 &&
      config.level.input_width==9 && config.level.patch_length==8 &&
      config.level.export_width==32 && config.change_width==4,
      "declared C3/H128/F9/P8/native36 fixture geometry");
  check(std::string_view(cp::kGradientPolicy)=="v4-gaussian-mean-live;temporal-hidden-difference-mean-live;variance-input-gradient-stop;shared-global-clip1-v16",
      "selected explicit gradient policy");
  const auto stats = statistics(config);
  const auto scaler_identity = stats.scaler.identity(),bank_identity = stats.bank.identity();
  torch::manual_seed(943001);
  auto model = impl::LocationScaleDecoderModel(config,stats.bank);
  model->train();
  int64_t learned_tensors=0,learned_values=0,fixed_values=0;
  std::vector<std::pair<torch::Tensor,torch::Tensor>> prior;
  for(const auto &parameter:model->named_parameters()) {
    check(parameter.value().is_cuda(),"all fixture parameters actually reside on CUDA");
    if(parameter.value().requires_grad()) {++learned_tensors;learned_values+=parameter.value().numel();}
    else {fixed_values+=parameter.value().numel();prior.push_back({parameter.value(),parameter.value().detach().clone()});}
  }
  check(learned_tensors==75 && learned_values==293672 && fixed_values==3888,
      "unchanged75 learned tensors/293672 values and3888 fixed literal prior");
  check(model->variance_projection.sizes()==torch::IntArrayRef({27,4}) &&
      model->variance_bias.sizes()==torch::IntArrayRef({27}) &&
      model->encoder->change_decoder.sizes()==torch::IntArrayRef({3456,4}),
      "unchanged108+27 conditional variance parameters and waveform residual");
  const auto options = model->variance_projection.options();
  {
    torch::NoGradGuard guard;
    // Engineering-only nonzero values. The production constructor retains its
    // original zero initializers; these deliberate fixture interventions make
    // the path tests impossible to pass just because a projection starts zero.
    model->variance_projection.copy_(torch::linspace(.04,.20,108,options).reshape({27,4}));
    model->variance_bias.copy_(torch::linspace(-.2,.2,27,options));
    model->encoder->change_decoder.copy_(torch::linspace(.01,.025,3456*4,options).reshape({3456,4}));
  }
  check(model->variance_projection.ne(0).all().item<bool>() &&
      model->encoder->change_decoder.ne(0).all().item<bool>(),"nonvacuous nonzero engineering projections");
  const auto ids = torch::tensor({0,1,2,2,0,1},torch::kInt64).reshape({2,3});

  auto native = native_leaf(options);
  const auto initial_distribution = decode(model,native,ids);
  {
    torch::NoGradGuard guard;
    check(torch::equal(initial_distribution.variance,attached_variance(model,native,ids)),
        "stopped and attached variance formulas are numerically identical");
    auto changed = native.detach().clone();
    changed.narrow(1,32,4).add_(.5);
    const auto changed_distribution = decode(model,changed,ids);
    check(!torch::equal(initial_distribution.variance,changed_distribution.variance) &&
        (initial_distribution.variance-changed_distribution.variance).abs().max().item<double>()>1e-4,
        "variance still conditions numerically on changed native input");
  }

  clear(model);
  native = native_leaf(options);
  decode(model,native,ids).variance.sum().backward();
  check(zero_or_absent_gradient(native),"variance-only loss gives no native gradient");
  check(nonzero_gradient(model->variance_projection) && nonzero_gradient(model->variance_bias),
      "variance-only loss gives nonzero actual CUDA projection and bias gradients");
  check(zero_or_absent_gradient(model->encoder->change_decoder),
      "variance-only loss does not touch the mean residual decoder");

  clear(model);
  native = native_leaf(options);
  attached_variance(model,native,ids).sum().backward();
  check(nonzero_gradient(native) &&
      native.grad().narrow(1,32,4).abs().sum().item<double>()>0.,
      "attached-formula positive control has a nonzero native change4 gradient");

  clear(model);
  native = native_leaf(options);
  decode(model,native,ids).mean.sum().backward();
  check(nonzero_gradient(native) &&
      native.grad().narrow(1,32,4).abs().sum().item<double>()>0.,
      "mean-only loss preserves nonzero native change4 gradient");
  check(nonzero_gradient(model->encoder->change_decoder) &&
      zero_or_absent_gradient(model->variance_projection) && zero_or_absent_gradient(model->variance_bias),
      "mean-only loss learns residual while leaving conditional variance head unused");

  clear(model);
  native = native_leaf(options);
  const auto distribution = decode(model,native,ids);
  torch::Tensor target;
  {
    torch::NoGradGuard guard;
    target = distribution.mean.detach()+torch::linspace(.1,.5,2*3*128*9,options).reshape({2,3,128,9});
  }
  const auto mask = support(config.level.device);
  const auto eligible = mask.sum(std::vector<int64_t>{2,3}).gt(0);
  check(eligible[1][2].logical_not().item<bool>() && eligible.sum().item<int64_t>()==5,
      "manual-gradient fixture includes unequal support and an ineligible channel");
  const auto expected = manual_mean_path(model,distribution,target,mask,ids);
  const auto nll = impl::location_scale_decoder_nll(distribution.mean,
      distribution.variance,target,mask,eligible);
  check(nll.loss.is_cuda() && nll.loss.scalar_type()==torch::kFloat64 &&
      torch::isfinite(nll.loss).item<bool>(),"actual CUDA F64 Gaussian NLL");
  nll.loss.backward();
  const auto actual = native.grad().narrow(1,32,4);
  check(actual.is_cuda() && expected.is_cuda() && expected.abs().sum().item<double>()>0. &&
      torch::allclose(actual,expected,1e-4,1e-7),
      "full NLL native change4 gradient equals only manual mean path with correct semantic/loss reductions");
  check(nonzero_gradient(model->encoder->change_decoder) &&
      nonzero_gradient(model->variance_projection) && nonzero_gradient(model->variance_bias),
      "full NLL supplies live mean/residual and conditional variance-head learning gradients");
  check(model->variance_projection.grad().ne(0).all().item<bool>() &&
      model->variance_bias.grad().ne(0).all().item<bool>(),
      "all108+27 conditional variance values receive actual nonzero CUDA gradients");
  const auto before_mean = model->encoder->change_decoder.detach().clone();
  const auto before_projection = model->variance_projection.detach().clone();
  const auto before_bias = model->variance_bias.detach().clone();
  torch::optim::SGD optimizer({model->encoder->change_decoder,
      model->variance_projection,model->variance_bias},torch::optim::SGDOptions(.01));
  optimizer.step();
  check(!torch::equal(before_mean,model->encoder->change_decoder),"actual synthetic CUDA optimizer step moves mean residual decoder");
  check(!torch::equal(before_projection,model->variance_projection),"actual synthetic CUDA optimizer step moves variance projection");
  check(!torch::equal(before_bias,model->variance_bias),"actual synthetic CUDA optimizer step moves variance bias");
  check(before_projection.ne(model->variance_projection).all().item<bool>() &&
      before_bias.ne(model->variance_bias).all().item<bool>(),
      "all108+27 conditional variance values actually move in the synthetic optimizer step");
  clear(model);
  model->validate_state();
  model->validate_fixed_prior();
  for(const auto &[tensor,saved]:prior)
    check(!tensor.requires_grad() && !tensor.grad().defined() && torch::equal(tensor,saved),
        "literal analytic prior stays exact without gradients or updates");
  check(stats.scaler.identity()==scaler_identity && stats.bank.identity()==bank_identity,
      "synthetic fitted input and bank statistics remain frozen");
  torch::cuda::synchronize();
  std::cout << "PASS CPB-v16 variance_input_gradient_stop-v1: " << checks
      << " checks, " << cuda_decode_calls << " actual CUDA decoder calls; "
      << "forward conditioning preserved, variance-only native gradient blocked, live mean gradient, "
      << "manual full-NLL change4 gradient parity, conditional variance head still learns; "
      << "synthetic optimizer steps=1, parameter tensors moved=3, learned tensors=75, "
      << "encoder forwards=0, old checkpoint/statistic loads=0, real encoder training updates=0, quality head fits=0\n";
  return 0;
} catch(const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
