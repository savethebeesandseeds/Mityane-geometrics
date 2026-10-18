// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/native_view_agreement.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace nv = rpb::native_view_agreement;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;
struct RuntimeWitness {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeWitness() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &g : generators) states.push_back(g.get_state().clone());
  }
  void unchanged(const std::string &label) const {
    check(threads == at::get_num_threads(), label + " thread state");
    for (size_t i = 0; i < generators.size(); ++i) check(torch::equal(states[i], generators[i].get_state()), label + " RNG state");
  }
};
void same_model(const rpb::Model &a, const rpb::Model &b) {
  const auto x = a->named_parameters(), y = b->named_parameters(); check(x.size() == y.size(), "named parameter size");
  for (const auto &p : x) check(y.contains(p.key()) && torch::equal(p.value(), y[p.key()]), "exact named parameter " + p.key());
  const auto u = a->named_buffers(), v = b->named_buffers(); check(u.size() == v.size(), "named buffer size");
  for (const auto &p : u) check(v.contains(p.key()) && torch::equal(p.value(), v[p.key()]), "exact named buffer " + p.key());
}
torch::Tensor read(torch::serialize::InputArchive &archive, const std::string &key) {
  torch::Tensor out; archive.read(key, out, true); return out;
}
std::string text(torch::serialize::InputArchive &archive, const std::string &key) { return embedding::archive::tensor_text(read(archive, key)); }
void same_optimizer(rpb::Checkpoint &a, const std::string &ap, rpb::Checkpoint &b, const std::string &bp) {
  torch::optim::AdamW x(a.model->parameters(), torch::optim::AdamWOptions(a.settings.learning_rate).weight_decay(a.settings.weight_decay));
  torch::optim::AdamW y(b.model->parameters(), torch::optim::AdamWOptions(b.settings.learning_rate).weight_decay(b.settings.weight_decay));
  rpb::load_optimizer(ap, x, a.settings.model.device); rpb::load_optimizer(bp, y, b.settings.model.device);
  check(!x.state().empty() && x.state().size() == y.state().size(), "same nonempty CUDA AdamW state");
  const auto yp = b.model->named_parameters();
  for (const auto &p : a.model->named_parameters()) {
    const auto i = x.state().find(p.value().unsafeGetTensorImpl()), j = y.state().find(yp[p.key()].unsafeGetTensorImpl());
    check((i == x.state().end()) == (j == y.state().end()), "same active optimizer parameter set");
    if (i == x.state().end()) continue;
    const auto *u = dynamic_cast<const torch::optim::AdamWParamState *>(i->second.get());
    const auto *v = dynamic_cast<const torch::optim::AdamWParamState *>(j->second.get());
    check(u && v && u->step() == a.completed_steps && v->step() == b.completed_steps && u->exp_avg().is_cuda() &&
        u->exp_avg_sq().is_cuda() && torch::equal(u->exp_avg(), v->exp_avg()) && torch::equal(u->exp_avg_sq(), v->exp_avg_sq()),
        "exact named CUDA AdamW step and moments " + p.key());
  }
}
void prefix(const ev::CurveProgress &a, const ev::CurveProgress &b) {
  check(a.losses.size() <= b.losses.size(), "loss prefix size");
  for (size_t i = 0; i < a.losses.size(); ++i) {
    const auto &x = a.losses[i], &y = b.losses[i];
    check(x.attempted == y.attempted && x.completed == y.completed && x.target_cells == y.target_cells &&
        x.loss == y.loss && x.gradient_norm == y.gradient_norm, "queried milestone exact trace prefix");
  }
}
void freeze(rpb::Checkpoint &checkpoint) {
  checkpoint.model->eval(); for (auto &p : checkpoint.model->parameters()) p.set_requires_grad(false);
}
nv::RowIdentity identity(const rpb::Input &input, const std::vector<std::string> &sources) {
  return {torch::arange(input.data.size(0), torch::kInt64),
      input.channel_ids.dim() == 1 ? input.channel_ids.unsqueeze(0).expand({input.data.size(0), input.data.size(1)}) : input.channel_ids,
      input.endpoints, sources};
}
rpb::EncodeOutput encoding(const torch::Tensor &z) {
  rpb::EncodeOutput out; out.z_contextual_global = z;
  out.sample_valid_mask = torch::ones({z.size(0)}, z.options().dtype(torch::kBool));
  out.channel_valid_mask = torch::ones({z.size(0), 3}, z.options().dtype(torch::kBool));
  out.channel_ids = torch::tensor({101, 202, 303}, torch::kInt64).unsqueeze(0).expand({z.size(0), 3}).to(z.device());
  return out;
}
void primitive_cuda() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true);
  auto zo = torch::zeros({4, 32}, options), zs = torch::ones({4, 32}, options);
  auto a = encoding(zo), b = encoding(zs);
  auto ids = nv::RowIdentity{torch::arange(4, torch::kInt64), a.channel_ids, torch::full({4}, 31., torch::kFloat64), {"a", "b", "a", "d"}};
  auto eligible = torch::tensor({true, false, true, true}, torch::kBool).to(torch::kCUDA);
  b.channel_valid_mask[3][1] = false;
  const auto scale = torch::full({32}, 2., torch::kFloat32);
  const auto output = nv::view_loss(a, b, ids, ids, eligible, scale);
  check(output.supported_count == 2 && output.unique_source_count == 1 &&
      torch::equal(output.variance_rows.to(torch::kCPU), torch::tensor({0}, torch::kInt64)), "eligibility/mask support and first duplicate-source row");
  check(output.agreement.is_cuda() && output.variance.is_cuda() && output.agreement.item<float>() == .25f &&
      output.variance.item<float>() == 0.f, "exact agreement and differentiable unsupported variance");
  output.agreement.backward();
  check(!zo.grad().defined() && zs.grad().defined() && zs.grad().is_cuda(), "agreement-only target path is detached");
  check(zs.grad()[1].eq(0).all().item<bool>() && zs.grad()[3].eq(0).all().item<bool>(), "unsupported agreement rows have zero gradients");
  auto wrong = ids; wrong.row_indices = ids.row_indices.roll(1);
  rejects([&] { nv::view_loss(a, b, ids, wrong, eligible, scale); }, "different sampled row identities");
  wrong = ids; wrong.source_ids[0] = "opposite-sibling";
  rejects([&] { nv::view_loss(a, b, ids, wrong, eligible, scale); }, "source-only positive substitution");
  wrong = ids; wrong.endpoints = ids.endpoints + 1;
  rejects([&] { nv::view_loss(a, b, ids, wrong, eligible, scale); }, "different endpoint positive");

  auto constant = torch::ones({4, 32}, options); auto c = encoding(constant); auto unique = ids;
  unique.source_ids = {"a", "b", "c", "d"};
  const auto variance = nv::view_loss(encoding(torch::zeros_like(constant)), c, unique, unique,
      torch::ones({4}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)), torch::ones({32}, torch::kFloat32));
  close(variance.variance, torch::tensor(.49f).to(torch::kCUDA), "constant native variance safeguard", 0, 1e-7);
  variance.variance.backward(); finite(constant.grad(), "constant variance gradient");
}

void core_cuda() {
  auto settings = rpb::default_settings(); auto &c = settings.model;
  c.device = torch::Device(torch::kCUDA, 0); c.channel_mixer_layers = 1; c.global_bottleneck_mode = 2; c.channel_ids = {101, 202, 303};
  auto raw = input(c, 6); const auto scaler = rpb::fit_scaler(raw, c); auto normalized = scaler.transform(raw, c);
  torch::manual_seed(991); rpb::Model model(c); torch::manual_seed(991); rpb::Model control(c);
  const std::vector<std::string> sources{"a", "a", "b", "b", "c", "c"};
  const RuntimeWitness calibration_runtime;
  const auto scale = nv::calibrate_loss_scale(*model, normalized, sources); calibration_runtime.unchanged("calibration");
  check(model->is_training() && model->parameters().front().requires_grad(), "calibration restores original model flags");
  same_model(model, control); nv::validate_loss_scale(scale);
  rejects([&] { nv::calibrate_loss_scale(*model, normalized, std::vector<std::string>(6, "only-source")); }, "calibration needs two valid source groups");
  auto one_valid_group = normalized; one_valid_group.observed = normalized.observed.clone();
  one_valid_group.observed.index_put_({torch::indexing::Slice(2, 6)}, false);
  rejects([&] { nv::calibrate_loss_scale(*model, one_valid_group, sources); }, "invalid rows cannot supply a second calibration source group");
  auto changed_scale = scale; changed_scale.values = scale.values * 2;
  rejects([&] { nv::validate_loss_scale(changed_scale); }, "loss-only scale identity and population arithmetic");
  auto ordinary = rpb::make_training_mask(normalized.observed, c, 177);
  const auto student = rpb::context_deletion::make_plan(ordinary, normalized.channel_ids, c, 220, 3, rpb::ContextDeletionRecipe::coordinate15_v1);
  const auto rows = identity(normalized, sources);
  auto exact = control->forward(normalized, ordinary.hidden);
  auto disabled = nv::training_forward(*model, normalized, ordinary, student, rows, scale.values, false);
  check(torch::equal(exact.loss, disabled.total_loss) && torch::equal(exact.reconstruction, disabled.ordinary.reconstruction), "disabled admission branch exact ordinary scalar/predictions");
  exact.loss.backward(); disabled.total_loss.backward();
  const auto cg = control->named_parameters();
  for (const auto &p : model->named_parameters())
    check(p.value().grad().defined() == cg[p.key()].grad().defined() && (!p.value().grad().defined() ||
        torch::equal(p.value().grad(), cg[p.key()].grad())), "disabled exact ordinary parameter gradient " + p.key());
  model->zero_grad(); control->zero_grad();
  auto active = nv::training_forward(*model, normalized, ordinary, student, rows, scale.values);
  check(torch::equal(active.ordinary.loss, exact.loss) && torch::equal(active.ordinary.target_counts, exact.target_counts) &&
      torch::equal(active.ordinary.eligible_channels, exact.eligible_channels) && active.ordinary.target_cell_count == exact.target_cell_count,
      "active ordinary reconstruction preserves original Q/reduction/eligibility");
  auto fresh_core = control->forward(normalized, ordinary.hidden);
  active.ordinary.loss.backward({}, true); fresh_core.loss.backward();
  for (const auto &p : model->named_parameters())
    check(p.value().grad().defined() == cg[p.key()].grad().defined() && (!p.value().grad().defined() ||
        torch::equal(p.value().grad(), cg[p.key()].grad())), "production ordinary-anchor exact core parameter gradient " + p.key());
  model->zero_grad(); control->zero_grad();
  active.total_loss.backward();
  check(active.total_loss.is_cuda() && active.ordinary.loss.is_cuda() && active.auxiliary.agreement.is_cuda() && active.auxiliary.variance.is_cuda(), "all actual loss components CUDA");
  bool encoder_gradient = false, decoder_gradient = false;
  for (const auto &p : model->named_parameters()) {
    if (!p.value().grad().defined()) continue;
    check(p.value().grad().is_cuda(), "CUDA parameter gradient"); finite(p.value().grad(), "finite actual full gradient");
    if (p.value().grad().abs().sum().item<double>() > 0)
      (p.key().rfind("decoder_", 0) == 0 ? decoder_gradient : encoder_gradient) = true;
  }
  check(encoder_gradient && decoder_gradient, "shared encoder auxiliaries plus ordinary decoder receive gradients");
  model->eval(); for (auto &p : model->parameters()) p.set_requires_grad(false);
  torch::NoGradGuard no_grad;
  const auto reference = nv::training_forward(*model, normalized, ordinary, student, rows, scale.values);
  auto poisoned = normalized; poisoned.data = normalized.data.clone(); poisoned.data.masked_fill_(ordinary.target, 99999.f);
  const auto hidden = nv::training_forward(*model, poisoned, ordinary, student, rows, scale.values);
  check(torch::equal(reference.ordinary.encoding.z_contextual_global, hidden.ordinary.encoding.z_contextual_global) &&
      torch::equal(reference.student.z_contextual_global, hidden.student.z_contextual_global), "query values never enter either encoder branch");
  poisoned.data = normalized.data.clone(); poisoned.data.masked_fill_(student.deleted, 88888.f);
  const auto deleted = nv::training_forward(*model, poisoned, ordinary, student, rows, scale.values);
  check(torch::equal(reference.student.z_contextual_global, deleted.student.z_contextual_global), "student deleted storage physically hidden");
}

void live_cuda() {
  auto settings = rpb::default_settings(); auto &c = settings.model;
  c.device = torch::Device(torch::kCUDA, 0); c.channel_mixer_layers = 1; c.global_bottleneck_mode = 2;
  c.channel_ids = {101, 202, 303}; c.dropout = .1;
  settings.steps = 4; settings.batch_size = 2; settings.log_every = 3; settings.attempt_limit = 8;
  auto raw = input(c, 6); const auto order = torch::tensor({2, 0, 1}, torch::kInt64);
  raw.data = raw.data.index_select(1, order); raw.observed = raw.observed.index_select(1, order); raw.channel_ids = raw.channel_ids.index_select(0, order);
  raw.observed[0][0][9][1] = false; raw.data.masked_fill_(raw.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  const embedding::Batch legal{raw.data.clone(), raw.observed.clone()};
  ev::ProviderFitInput fit{legal, {3, 32, 3, torch::kFloat64, torch::kCPU}, 4404,
      {"a", "a", "b", "b", "c", "c"}, {303, 101, 202}, "volts,amperes,kelvin", "native-development-v1/lag_sign", 1., 31.};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-native-view-agreement-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directories(directory), "exclusive test directory");
  const auto factory = rpb::make_native_view_agreement_trainer(settings);
  const RuntimeWitness runtime;
  auto split = factory(fit); runtime.unchanged("factory");
  const auto p0 = (directory / "split0.pt").string(); split.train_to(0); split.save_checkpoint(p0);
  auto zero = split.snapshot(p0); auto zero_features = zero.features.extract(legal).at("curve_global").values.clone();
  auto query = torch::zeros_like(raw.observed); query.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 8), torch::indexing::Slice()}, true);
  const auto zero_reconstruction = zero.reconstruct(legal, query);
  const auto p1 = (directory / "split1.pt").string(); const auto one = split.train_to(1); split.save_checkpoint(p1);
  auto point1 = split.snapshot(p1); const auto point1_features = point1.features.extract(legal).at("curve_global").values.clone();
  const auto point1_reconstruction = point1.reconstruct(legal, query);
  const auto p2 = (directory / "split2.pt").string(); const auto two = split.train_to(2); split.save_checkpoint(p2);
  auto point2 = split.snapshot(p2); const auto point2_features = point2.features.extract(legal).at("curve_global").values.clone();
  const auto witness = point2.reconstruct(legal, query);
  const auto four = split.train_to(4); const auto ps = (directory / "split4.pt").string(); split.save_checkpoint(ps);
  runtime.unchanged("training/save/snapshot callbacks"); prefix(one, two); prefix(two, four);
  check(four.parameter_count == 225805 && four.cuda_parameter_count == 225805 && four.last_input_cuda && four.last_loss_cuda &&
      four.finite_gradients && four.weights_changed && four.attempted == 4 && four.completed == 4 && four.sampled_rows == 8,
      "actual CUDA continuous update and parameter/input/loss witnesses");
  check(torch::equal(zero_features, zero.features.extract(legal).at("curve_global").values) &&
      torch::equal(point1_features, point1.features.extract(legal).at("curve_global").values) &&
      torch::equal(point2_features, point2.features.extract(legal).at("curve_global").values) &&
      torch::equal(zero_reconstruction.prediction, zero.reconstruct(legal, query).prediction) &&
      torch::equal(point1_reconstruction.prediction, point1.reconstruct(legal, query).prediction) &&
      torch::equal(witness.prediction, point2.reconstruct(legal, query).prediction), "earlier saved snapshots immutable after later updates");
  auto direct = factory(fit); direct.train_to(4); const auto pd = (directory / "direct4.pt").string(); direct.save_checkpoint(pd);
  auto a = rpb::load_checkpoint(ps, torch::kCUDA), b = rpb::load_checkpoint(pd, torch::kCUDA);
  same_model(a.model, b.model); same_optimizer(a, ps, b, pd);
  check(a.scaler.identity() == b.scaler.identity() && torch::equal(a.scaler.mean, b.scaler.mean) &&
      torch::equal(a.scaler.scale, b.scaler.scale) && a.training_policy_id == nv::policy_id, "fixed scaler and distinct policy persist");
  torch::serialize::InputArchive sa, da; sa.load_from(ps + ".audit.pt", torch::kCPU); da.load_from(pd + ".audit.pt", torch::kCPU);
  for (const auto &key : {"loss_scale", "loss_scale_calibration_native", "loss_scale_calibration_valid", "loss_components", "sampled_indices",
      "ordinary_native", "student_native", "auxiliary_supported", "variance_selected", "original_eligible", "ordinary_prediction", "query_support",
      "context_requested_deleted_coordinates", "context_actual_deleted_coordinates", "context_restored_coordinates"})
    check(torch::equal(read(sa, key), read(da, key)), "split/direct exact audit evidence " + std::string(key));
  check(text(sa, "artifact_kind") == "rpb_native_view_agreement_training_audit_v1" && text(sa, "model_tag") == "RPB-v9" &&
      text(sa, "training_policy_id") == nv::policy_id && text(sa, "loss_scale_identity") == text(da, "loss_scale_identity"), "new companion kind/tag/scale identity");
  auto retained = rpb::make_native_view_agreement_snapshot(ps, fit);
  const auto positive = split.snapshot(ps);
  check(torch::equal(retained.features.extract(legal).at("curve_global").values, positive.features.extract(legal).at("curve_global").values) &&
      torch::equal(retained.reconstruct(legal, query).prediction, positive.reconstruct(legal, query).prediction) &&
      retained.features.audit_fields.at("training_policy_id") == nv::policy_id &&
      retained.features.audit_fields.at("loss_scale_serving_policy") == "loss-only;ordinary-native32-and-decoder-unchanged", "retained wrapper exact inference and new policy metadata");
  {
    freeze(a); torch::NoGradGuard no_grad;
    const auto exact = a.model->forward(a.scaler.transform(raw, a.settings.model), query);
    check(torch::equal(exact.reconstruction.to(torch::kCPU), retained.reconstruct(legal, query).prediction), "ordinary checkpoint exact CUDA reconstruction remains unchanged");
  }
  // Point0 and positive ordinary reference use the original untouched factory.
  auto ordinary = rpb::make_learning_curve_trainer(settings)(fit);
  const auto o0 = (directory / "ordinary0.pt").string(), o4 = (directory / "ordinary4.pt").string();
  ordinary.train_to(0); ordinary.save_checkpoint(o0); ordinary.train_to(4); ordinary.save_checkpoint(o4);
  ev::RetainedPoolingCohort reference; reference.master_seed = fit.seed; reference.reference_initial_checkpoint = o0; reference.reference_checkpoint = o4;
  const auto init = rpb::audit_native_view_agreement_initialization(p0, reference, fit, 4);
  check(init.at("all_parameters_exact_including_global_pool") == "true" && init.at("loss_scale_calibration_exact") == "true", "complete ordinary point0 common initialization and exact calibration gate");
  auto wrong = fit; wrong.training_source_ids[0] = "changed";
  rejects([&] { rpb::audit_native_view_agreement_initialization(p0, reference, wrong, 4); }, "changed source order");
  wrong = fit; wrong.seed++;
  rejects([&] { rpb::audit_native_view_agreement_initialization(p0, reference, wrong, 4); }, "wrong counter seed");
  wrong = fit; wrong.protocol_id = "historical-selected";
  rejects([&] { rpb::make_native_view_agreement_trainer(settings)(wrong); }, "wrong training namespace");
  rejects([&] { split.save_checkpoint(ps); }, "checkpoint overwrite");
  std::vector<std::string> args{"rpb", "train", "--resume", ps, "--input", (directory / "never-read.pt").string(), "--checkpoint", (directory / "forbidden.pt").string(), "--steps", "1", "--device", "cuda"};
  std::vector<char *> argv; for (auto &arg : args) argv.push_back(arg.data());
  bool policy_rejected = false;
  try { rpb::run_cli(argv.size(), argv.data()); } catch (const std::exception &error) {
    policy_rejected = std::string(error.what()).find("ordinary train cannot resume this saved training policy") != std::string::npos;
  }
  check(policy_rejected && !fs::exists(directory / "forbidden.pt") && !fs::exists(directory / "never-read.pt"), "ordinary tagged resume rejected before TRAIN access");
  std::cout << "Native view agreement preserved admission artifacts: " << directory << '\n';
}
} // namespace

int main() {
  try {
    check(torch::cuda::is_available(), "actual CUDA required; CPU fallback forbidden"); at::set_num_threads(1);
    primitive_cuda(); core_cuda(); live_cuda();
    std::cout << "Native view agreement CUDA admission passed\n"; return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
