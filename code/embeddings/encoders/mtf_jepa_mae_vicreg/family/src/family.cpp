// SPDX-License-Identifier: MIT
#include "embedding/encoders/mtf_jepa_mae_vicreg/family.h"
#include "cuwacunu/artifacts/v1/tensor_state.h"
#include <ATen/Context.h>
#include <torch/version.h>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace embedding::encoders::mtf_jepa_mae_vicreg::family {
namespace {
namespace a = ::cuwacunu::artifacts::v1;
constexpr const char *kMechanism = "mtf-v1-preserved-core-portable-session-v1";

void require(bool condition, const std::string &message) {
  TORCH_CHECK(condition, "[mtf-family] ", message);
}
void identity_string(const std::string &value, const char *name) {
  require(!value.empty() && value.size() <= 4096 &&
              std::all_of(value.begin(), value.end(), [](unsigned char ch) {
                return ch >= 32 && ch < 127;
              }), std::string("nonempty printable identity required: ") + name);
}
std::string real(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}
void cpu_config(const Config &config) {
  validate_config(config);
  require(config.device.is_cpu() && config.dtype == torch::kFloat32,
          "portable v1 supports CPU/F32 only");
}
std::string config_text(const Config &config) {
  cpu_config(config);
  auto settings = default_settings();
  settings.model = config;
  return settings_text(settings);
}
std::vector<torch::Tensor> learned(const Model &model) {
  std::vector<torch::Tensor> out;
  for (const auto &p : model->parameters())
    if (p.requires_grad()) out.push_back(p);
  return out;
}
torch::optim::AdamWOptions options(const Recipe &recipe) {
  return torch::optim::AdamWOptions(recipe.settings.learning_rate)
      .betas(std::make_tuple(recipe.adam_beta1, recipe.adam_beta2))
      .eps(recipe.adam_epsilon)
      .weight_decay(recipe.settings.weight_decay)
      .amsgrad(recipe.adam_amsgrad);
}
std::string schema(const Model &model) {
  std::vector<std::string> fields;
  for (const auto &p : model->named_parameters()) {
    fields.push_back(p.key());
    fields.push_back(p.value().requires_grad() ? "learned" : "frozen");
    fields.push_back(c10::toString(p.value().scalar_type()));
    fields.push_back(p.value().device().str());
    fields.push_back(std::to_string(p.value().dim()));
    for (const auto n : p.value().sizes()) fields.push_back(std::to_string(n));
  }
  for (const auto &b : model->named_buffers()) {
    fields.push_back("buffer." + b.key());
    fields.push_back(c10::toString(b.value().scalar_type()));
    fields.push_back(b.value().device().str());
    fields.push_back(std::to_string(b.value().dim()));
    for (const auto n : b.value().sizes()) fields.push_back(std::to_string(n));
  }
  return a::digest_fields("mtf_family_parameter_schema_v1", fields);
}
void finite_model(const Model &model) {
  require(bool(model), "model required");
  for (const auto &p : model->named_parameters()) {
    require(p.value().device().is_cpu() && p.value().scalar_type() == torch::kFloat32 &&
                torch::isfinite(p.value()).all().item<bool>(),
            "finite CPU/F32 parameter required: " + p.key());
    const bool teacher = p.key().rfind("target_tokenizer.", 0) == 0 ||
                         p.key().rfind("target_encoder.", 0) == 0;
    require(p.value().requires_grad() != teacher,
            "online/teacher gradient registration changed: " + p.key());
    if (teacher) require(!p.value().grad().defined(), "teacher received gradient");
  }
  for (const auto &b : model->named_buffers())
    require(b.value().device().is_cpu() && torch::isfinite(b.value()).all().item<bool>(),
            "finite CPU buffer required: " + b.key());
}
std::vector<int64_t> validate_optimizer(const Model &model,
                                      const torch::optim::AdamW &optimizer,
                                      const Recipe &recipe, int64_t completed) {
  const auto expected = options(recipe);
  const auto *defaults = dynamic_cast<const torch::optim::AdamWOptions *>(&optimizer.defaults());
  require(defaults && *defaults == expected && optimizer.param_groups().size() == 1,
          "single-group AdamW with bound options required");
  const auto &group = optimizer.param_groups().front();
  const auto *actual = dynamic_cast<const torch::optim::AdamWOptions *>(&group.options());
  const auto parameters = learned(model);
  require(actual && *actual == expected && group.params().size() == parameters.size(),
          "optimizer options/parameter count mismatch");
  std::set<void *> known;
  std::vector<int64_t> steps;
  int64_t maximum = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto &p = parameters[i];
    auto *key = p.unsafeGetTensorImpl();
    require(group.params()[i].unsafeGetTensorImpl() == key && known.insert(key).second,
            "optimizer parameter order/alias mismatch");
    const auto found = optimizer.state().find(key);
    if (found == optimizer.state().end()) { steps.push_back(-1); continue; }
    const auto *state = dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
    require(state && state->step() > 0 && state->step() <= completed,
            "optimizer state/step mismatch");
    for (const auto &m : {state->exp_avg(), state->exp_avg_sq()})
      require(m.defined() && m.device().is_cpu() && m.scalar_type() == torch::kFloat32 &&
                  m.sizes() == p.sizes() && torch::isfinite(m).all().item<bool>(),
              "finite matching AdamW moments required");
    require(state->exp_avg_sq().ge(0).all().item<bool>(), "negative second moment");
    const auto &maximum_moment = state->max_exp_avg_sq();
    if (recipe.adam_amsgrad)
      require(maximum_moment.defined() && maximum_moment.device().is_cpu() &&
                  maximum_moment.scalar_type() == torch::kFloat32 &&
                  maximum_moment.sizes() == p.sizes() &&
                  torch::isfinite(maximum_moment).all().item<bool>() &&
                  maximum_moment.ge(state->exp_avg_sq()).all().item<bool>(),
              "matching AMSGrad maximum moments required");
    else require(!maximum_moment.defined(), "unexpected AMSGrad moments");
    maximum = std::max(maximum, state->step());
    steps.push_back(state->step());
  }
  for (const auto &entry : optimizer.state())
    require(known.count(entry.first) != 0, "orphan/frozen optimizer state");
  require(maximum == completed && (completed != 0 || optimizer.state().empty()),
          "optimizer maximum step disagrees with completed updates");
  return steps;
}
std::string optimizer_digest(const Model &model, const torch::optim::AdamW &optimizer,
                             const Recipe &recipe, int64_t completed) {
  const auto steps = validate_optimizer(model, optimizer, recipe, completed);
  std::vector<a::NamedTensor> tensors;
  tensors.emplace_back("steps", torch::tensor(steps, torch::kInt64));
  tensors.emplace_back("options", torch::tensor(std::vector<double>{
      recipe.settings.learning_rate, recipe.settings.weight_decay, recipe.adam_beta1,
      recipe.adam_beta2, recipe.adam_epsilon, recipe.adam_amsgrad ? 1.0 : 0.0}, torch::kFloat64));
  for (const auto &p : model->named_parameters()) {
    if (!p.value().requires_grad()) continue;
    const auto found = optimizer.state().find(p.value().unsafeGetTensorImpl());
    if (found == optimizer.state().end()) continue;
    const auto *s = dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
    tensors.emplace_back(p.key() + ".exp_avg", s->exp_avg());
    tensors.emplace_back(p.key() + ".exp_avg_sq", s->exp_avg_sq());
    if (recipe.adam_amsgrad) tensors.emplace_back(p.key() + ".max_exp_avg_sq", s->max_exp_avg_sq());
  }
  return a::tensor_state_sha256(std::move(tensors));
}
void write_text(torch::serialize::OutputArchive &out, const char *name, const std::string &value) {
  out.write(name, ::embedding::archive::text_tensor(value), true);
}
std::string read_text(torch::serialize::InputArchive &in, const char *name) {
  torch::Tensor value;
  in.read(name, value, true);
  require(value.device().is_cpu() && value.scalar_type() == torch::kUInt8 && value.dim() == 1 &&
              value.numel() <= 1024 * 1024, std::string("invalid text field: ") + name);
  return ::embedding::archive::tensor_text(value);
}
void write_integer(torch::serialize::OutputArchive &out, const char *name, int64_t value) {
  out.write(name, torch::tensor(value, torch::kInt64), true);
}
int64_t read_integer(torch::serialize::InputArchive &in, const char *name) {
  torch::Tensor value;
  in.read(name, value, true);
  require(value.device().is_cpu() && value.scalar_type() == torch::kInt64 && value.dim() == 0,
          std::string("invalid integer field: ") + name);
  return value.item<int64_t>();
}
void kind(torch::serialize::InputArchive &in, const char *expected) {
  require(read_text(in, "artifact_kind") == expected && read_integer(in, "format_version") == 1 &&
              read_text(in, "mechanism") == kMechanism && read_text(in, "preparation") == kPreparation,
          "typed portable artifact kind/version/mechanism/preparation mismatch (MTF-v1 is separate)");
}
void write_common(torch::serialize::OutputArchive &out, const char *artifact_kind,
                  const Recipe &recipe, int64_t completed, const Model &model,
                  const std::string &runtime) {
  write_text(out, "artifact_kind", artifact_kind);
  write_integer(out, "format_version", 1);
  write_text(out, "mechanism", kMechanism);
  write_text(out, "preparation", kPreparation);
  write_text(out, "source_sha256", recipe.source_sha256);
  write_text(out, "config", config_text(recipe.settings.model));
  write_text(out, "state_sha256", a::module_state_sha256(*model));
  write_text(out, "parameter_schema", schema(model));
  write_text(out, "runtime", runtime);
  write_integer(out, "completed_updates", completed);
  torch::serialize::OutputArchive weights;
  model->save(weights);
  out.write("model", weights);
}
} // namespace

::embedding::Batch prepare(const ::embedding::Batch &raw, const Config &config) {
  cpu_config(config);
  require(raw.data.defined() && raw.data.device().is_cpu() && raw.data.is_floating_point() &&
              raw.data.dim() == 4 && raw.data.size(0) > 0 &&
              raw.data.size(1) == config.channel_count &&
              raw.data.size(2) == config.history_length && raw.data.size(3) == config.input_width,
          "nonempty CPU floating BCHF matching config required");
  require(raw.feature_mask.defined() && raw.feature_mask.device().is_cpu() &&
              raw.feature_mask.scalar_type() == torch::kBool &&
              raw.feature_mask.sizes() == raw.data.sizes(), "explicit matching CPU bool support required");
  const auto safe = torch::where(raw.feature_mask, raw.data, torch::zeros_like(raw.data));
  require(torch::isfinite(safe).all().item<bool>(), "observed nonfinite input");
  const auto data = safe.to(torch::kFloat32);
  require(torch::isfinite(data).all().item<bool>(), "observed input overflows F32");
  return {data.detach().clone().contiguous(), raw.feature_mask.detach().clone().contiguous()};
}
std::string dataset_identity(const ::embedding::Batch &prepared) {
  require(prepared.data.defined() && prepared.data.device().is_cpu() &&
              prepared.data.scalar_type() == torch::kFloat32 && prepared.data.dim() == 4 &&
              prepared.data.size(0) > 0 && prepared.feature_mask.defined() &&
              prepared.feature_mask.device().is_cpu() && prepared.feature_mask.scalar_type() == torch::kBool &&
              prepared.feature_mask.sizes() == prepared.data.sizes() &&
              torch::isfinite(prepared.data).all().item<bool>() &&
              prepared.data.masked_select(prepared.feature_mask.logical_not()).eq(0).all().item<bool>(),
          "canonical prepared F32 observations required for hashing");
  return a::tensor_state_sha256({{"data", prepared.data}, {"observed", prepared.feature_mask}});
}
void validate_recipe(const Recipe &recipe) {
  cpu_config(recipe.settings.model);
  (void)settings_text(recipe.settings);
  require(recipe.settings.threads == 1 && recipe.settings.model.strict_finite_loss,
          "single-thread and strict finite loss required");
  require(recipe.settings.seed <= std::numeric_limits<int64_t>::max() - recipe.settings.steps - 1,
          "seed/progress would overflow");
  require(a::is_sha256(recipe.source_sha256) && a::is_sha256(recipe.dataset_sha256),
          "lowercase source and prepared-dataset SHA256 required");
  identity_string(recipe.input_source_identity, "input source");
  identity_string(recipe.split_identity, "TRAIN split");
  identity_string(recipe.recipe_identity, "recipe");
  require(recipe.sampling == kReplacementSampling || recipe.sampling == kFullPartitionSampling,
          "unsupported sampling algorithm");
  require(std::isfinite(recipe.adam_beta1) && recipe.adam_beta1 >= 0 && recipe.adam_beta1 < 1 &&
              std::isfinite(recipe.adam_beta2) && recipe.adam_beta2 >= 0 && recipe.adam_beta2 < 1 &&
              std::isfinite(recipe.adam_epsilon) && recipe.adam_epsilon > 0,
          "valid explicit AdamW beta/epsilon options required");
  const auto &c = recipe.settings.model;
  require((c.use_jepa_loss && c.lambda_jepa > 0) || (c.use_mae_decoder && c.lambda_mae > 0) ||
              (c.use_tf_align_loss && c.use_frequency_tokens && c.lambda_tf_align > 0) ||
              (c.use_vicreg_loss && c.lambda_vicreg > 0 &&
               ((c.use_global_vicreg && c.lambda_global_vicreg > 0) ||
                (c.use_channel_vicreg && c.lambda_channel_vicreg > 0))),
          "at least one effective objective required");
}
std::string recipe_sha256(const Recipe &recipe) {
  validate_recipe(recipe);
  return a::digest_fields("mtf_family_recipe_v1", {kMechanism, kPreparation, kSeedAlgorithm,
      settings_text(recipe.settings), recipe.source_sha256, recipe.dataset_sha256,
      recipe.input_source_identity, recipe.split_identity, recipe.recipe_identity, recipe.sampling,
      real(recipe.adam_beta1), real(recipe.adam_beta2), real(recipe.adam_epsilon),
      recipe.adam_amsgrad ? "true" : "false"});
}
std::string runtime_identity() {
  // Exact continuation deliberately pins this executable, not merely its
  // reported compiler/Torch version. SDK/hardware remain a declared same-runtime
  // requirement; inference loading does not require this training binary pin.
  static const std::string executable_sha256 = a::sha256_file("/proc/self/exe");
  std::ostringstream out;
  out << "mtf-portable-cpu-f32-v1;torch=" << TORCH_VERSION << ";compiler=" << __VERSION__
      << ";cxx=" << __cplusplus << ";abi=" << _GLIBCXX_USE_CXX11_ABI
      << ";little_endian=" << (std::endian::native == std::endian::little)
      << ";threads=" << torch::get_num_threads() << ";interop=" << torch::get_num_interop_threads()
      << ";deterministic=" << at::globalContext().deterministicAlgorithms()
      << ";deterministic_warn_only=" << at::globalContext().deterministicAlgorithmsWarnOnly()
      << ";mkldnn=" << at::globalContext().userEnabledMkldnn()
      << ";executable_sha256=" << executable_sha256;
  return out.str();
}
std::string file_sha256(const std::string &path) { return a::sha256_file(path); }
ParameterInventory parameter_inventory(const Model &model) {
  require(bool(model), "model required for inventory");
  ParameterInventory out;
  for (const auto &p : model->named_parameters()) {
    require(p.value().numel() <= std::numeric_limits<int64_t>::max() - out.total,
            "parameter inventory overflow");
    const auto count = p.value().numel();
    out.records.push_back({p.key(), p.value().sizes().vec(), count, p.value().requires_grad()});
    out.total += count;
    if (p.value().requires_grad()) out.trainable += count;
    else out.frozen += count;
  }
  return out;
}
Session Session::fresh(const Recipe &recipe, const ::embedding::Batch &observations) {
  validate_recipe(recipe);
  Session out;
  out.recipe_ = recipe;
  out.observations_ = prepare(observations, recipe.settings.model);
  require(out.observations_.feature_mask.flatten(1).any(1).all().item<bool>(),
          "each TRAIN row must contain observed support");
  require(dataset_identity(out.observations_) == recipe.dataset_sha256,
          "actual prepared observations disagree with dataset binding");
  torch::set_num_threads(1);
  torch::manual_seed(static_cast<uint64_t>(recipe.settings.seed));
  out.model_ = Model(recipe.settings.model);
  out.optimizer_ = std::make_unique<torch::optim::AdamW>(learned(out.model_), options(recipe));
  out.initial_state_ = a::module_state_sha256(*out.model_);
  out.schema_ = schema(out.model_);
  out.runtime_ = runtime_identity();
  out.validate_state();
  return out;
}
void Session::validate_state() const {
  validate_recipe(recipe_);
  require(model_ && optimizer_ && completed_ >= 0 && completed_ <= recipe_.settings.steps &&
              torch::get_num_threads() == 1 && runtime_identity() == runtime_,
          "session progress/runtime mismatch");
  require(schema(model_) == schema_ && config_text(model_->config()) == config_text(recipe_.settings.model),
          "model configuration/schema changed");
  finite_model(model_);
  (void)validate_optimizer(model_, *optimizer_, recipe_, completed_);
}
std::string Session::state_sha256() const { return a::module_state_sha256(*model_); }
std::string Session::optimizer_sha256() const {
  return optimizer_digest(model_, *optimizer_, recipe_, completed_);
}
UpdateResult Session::update() {
  validate_state();
  require(completed_ < recipe_.settings.steps, "fixed total update budget exhausted");
  require(dataset_identity(observations_) == recipe_.dataset_sha256, "owned prepared data changed");
  model_->train();
  torch::manual_seed(static_cast<uint64_t>(recipe_.settings.seed) + static_cast<uint64_t>(completed_) + 1);
  ::embedding::Batch batch = observations_;
  if (recipe_.sampling == kReplacementSampling) {
    const auto indices = torch::randint(observations_.data.size(0), {recipe_.settings.batch_size}, torch::kInt64);
    batch = {observations_.data.index_select(0, indices), observations_.feature_mask.index_select(0, indices)};
  }
  optimizer_->zero_grad();
  auto output = model_->forward(batch.data, batch.feature_mask);
  const auto targets = output.jepa_target_mask.sum().item<int64_t>();
  const auto &c = recipe_.settings.model;
  require(targets > 0 || !((c.use_jepa_loss && c.lambda_jepa > 0) ||
                            (c.use_mae_decoder && c.lambda_mae > 0)),
          "no selected JEPA/MAE target support in this batch; update refused");
  require(output.loss.requires_grad() && torch::isfinite(output.loss).all().item<bool>(),
          "finite differentiable objective required");
  output.loss.backward();
  bool nonzero_gradient = false;
  for (const auto &p : model_->parameters()) {
    const auto gradient = p.grad();
    if (!gradient.defined()) continue;
    require(p.requires_grad() && torch::isfinite(gradient).all().item<bool>(),
            "finite online gradients required");
    nonzero_gradient |= gradient.ne(0).any().item<bool>();
  }
  require(nonzero_gradient, "no nonzero learning gradient; update refused");
  torch::nn::utils::clip_grad_norm_(learned(model_), recipe_.settings.gradient_clip_norm > 0 ?
      recipe_.settings.gradient_clip_norm : std::numeric_limits<double>::infinity(),
      2.0, /*error_if_nonfinite=*/true);
  optimizer_->step();
  model_->update_target_network();
  ++completed_;
  validate_state();
  return {completed_, output.loss.item<double>(), targets};
}
void Session::save_training(const std::string &path) const {
  validate_state();
  torch::serialize::OutputArchive out, optimizer;
  write_common(out, kTrainingKind, recipe_, completed_, model_, runtime_);
  write_text(out, "settings", settings_text(recipe_.settings));
  write_text(out, "recipe_sha256", recipe_sha256(recipe_));
  write_text(out, "dataset_sha256", recipe_.dataset_sha256);
  write_text(out, "input_source_identity", recipe_.input_source_identity);
  write_text(out, "split_identity", recipe_.split_identity);
  write_text(out, "recipe_identity", recipe_.recipe_identity);
  write_text(out, "sampling", recipe_.sampling);
  write_text(out, "seed_algorithm", kSeedAlgorithm);
  write_text(out, "initial_state_sha256", initial_state_);
  write_text(out, "optimizer_sha256", optimizer_sha256());
  out.write("adam_options_f64", torch::tensor(std::vector<double>{
      recipe_.adam_beta1, recipe_.adam_beta2, recipe_.adam_epsilon, recipe_.adam_amsgrad ? 1.0 : 0.0}, torch::kFloat64), true);
  optimizer_->save(optimizer);
  out.write("optimizer", optimizer);
  ::embedding::archive::save_archive(path, out);
}
Session Session::resume(const std::string &path, const Recipe &expected_recipe,
                        const ::embedding::Batch &observations) {
  auto out = fresh(expected_recipe, observations);
  torch::serialize::InputArchive in, model, optimizer;
  in.load_from(path, torch::kCPU);
  kind(in, kTrainingKind);
  require(read_text(in, "source_sha256") == expected_recipe.source_sha256 &&
              read_text(in, "settings") == settings_text(expected_recipe.settings) &&
              read_text(in, "config") == config_text(expected_recipe.settings.model) &&
              read_text(in, "recipe_sha256") == recipe_sha256(expected_recipe) &&
              read_text(in, "dataset_sha256") == expected_recipe.dataset_sha256 &&
              read_text(in, "input_source_identity") == expected_recipe.input_source_identity &&
              read_text(in, "split_identity") == expected_recipe.split_identity &&
              read_text(in, "recipe_identity") == expected_recipe.recipe_identity &&
              read_text(in, "sampling") == expected_recipe.sampling &&
              read_text(in, "seed_algorithm") == kSeedAlgorithm &&
              read_text(in, "runtime") == out.runtime_ &&
              read_text(in, "parameter_schema") == out.schema_ &&
              read_text(in, "initial_state_sha256") == out.initial_state_,
          "expected source/config/data/split/recipe/initialization/runtime binding mismatch");
  torch::Tensor adam;
  in.read("adam_options_f64", adam, true);
  require(adam.device().is_cpu() && adam.scalar_type() == torch::kFloat64 &&
              adam.sizes() == torch::IntArrayRef({4}) && torch::equal(adam, torch::tensor(std::vector<double>{
                expected_recipe.adam_beta1, expected_recipe.adam_beta2, expected_recipe.adam_epsilon,
                expected_recipe.adam_amsgrad ? 1.0 : 0.0}, torch::kFloat64)), "AdamW option metadata mismatch");
  out.completed_ = read_integer(in, "completed_updates");
  require(out.completed_ >= 0 && out.completed_ <= expected_recipe.settings.steps,
          "invalid completed update count");
  in.read("model", model);
  out.model_->load(model);
  in.read("optimizer", optimizer);
  out.optimizer_->load(optimizer);
  out.validate_state();
  require(read_text(in, "state_sha256") == out.state_sha256() &&
              read_text(in, "optimizer_sha256") == out.optimizer_sha256(),
          "model or optimizer tensor content binding mismatch");
  out.model_->train();
  return out;
}
void Session::export_inference(const std::string &path) const {
  validate_state();
  torch::serialize::OutputArchive out;
  write_common(out, kInferenceKind, recipe_, completed_, model_, runtime_);
  ::embedding::archive::save_archive(path, out);
}
Inference load_inference(const std::string &path, const InferenceExpectation &expected) {
  cpu_config(expected.config);
  require(a::is_sha256(expected.source_sha256) && a::is_sha256(expected.state_sha256) &&
              expected.preparation == kPreparation, "expected source/state/preparation pins required");
  torch::serialize::InputArchive in, weights;
  in.load_from(path, torch::kCPU);
  kind(in, kInferenceKind);
  require(read_text(in, "source_sha256") == expected.source_sha256 &&
              read_text(in, "state_sha256") == expected.state_sha256 &&
              read_text(in, "config") == config_text(expected.config),
          "expected inference source/state/config mismatch");
  Inference out;
  out.model = Model(expected.config);
  const auto expected_schema = schema(out.model);
  require(read_text(in, "parameter_schema") == expected_schema, "inference parameter schema mismatch");
  in.read("model", weights);
  out.model->load(weights);
  finite_model(out.model);
  require(schema(out.model) == expected_schema && a::module_state_sha256(*out.model) == expected.state_sha256,
          "inference loaded tensor state mismatch");
  out.completed_updates = read_integer(in, "completed_updates");
  require(out.completed_updates >= 0, "invalid inference progress");
  out.model->eval();
  out.config = expected.config;
  out.source_sha256 = expected.source_sha256;
  out.state_sha256 = expected.state_sha256;
  out.preparation = expected.preparation;
  return out;
}
EncodeOutput Inference::encode(const ::embedding::Batch &raw) {
  require(model && preparation == kPreparation && config_text(model->config()) == config_text(config),
          "inference configuration/preparation changed");
  finite_model(model);
  require(a::module_state_sha256(*model) == state_sha256, "inference weights changed");
  auto input = prepare(raw, config);
  model->eval();
  torch::NoGradGuard no_grad;
  return model->encode(input.data, input.feature_mask);
}
} // namespace embedding::encoders::mtf_jepa_mae_vicreg::family
