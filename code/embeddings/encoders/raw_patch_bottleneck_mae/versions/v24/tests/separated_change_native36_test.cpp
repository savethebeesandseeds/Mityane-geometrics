// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
// Additive native36 fixture; the preserved native40 fixture stays unchanged.
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_frozen_level.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_settings.h"
#include <torch/cuda.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace {
namespace r = embedding::encoders::raw_patch_bottleneck_mae;
namespace tr = r::separated_change_training;

void check(bool value, const char *why) {
  TORCH_CHECK(value, "[separate native36 test] ", why);
}
template<class F> void rejects(F operation, const char *why) {
  bool caught = false;
  try { operation(); } catch (const std::exception &) { caught = true; }
  check(caught, why);
}
struct Temporary {
  std::filesystem::path path;
  Temporary() {
    auto candidate = (std::filesystem::temp_directory_path()/"rpb-separate-native36-XXXXXX").string();
    std::vector<char> buffer(candidate.begin(), candidate.end());
    buffer.push_back(0);
    check(::mkdtemp(buffer.data()), "exclusive fixture directory");
    path = buffer.data();
  }
  ~Temporary() { std::error_code error; std::filesystem::remove_all(path, error); }
};

void protected_state(const r::SeparatedChangeModel &model,
    const torch::optim::AdamW &optimizer, const tr::FrozenLevelGuard &guard) {
  guard.validate(model, optimizer);
  for (const auto &parameter : model->level->parameters()) {
    check(parameter.is_cuda() && !parameter.grad().defined(), "protected CUDA parameters have no gradients");
    check(!optimizer.state().count(parameter.unsafeGetTensorImpl()), "protected parameters have no AdamW moments");
  }
}

torch::Tensor update(const r::Input &input, r::SeparatedChangeModel &model,
    torch::optim::AdamW &optimizer, tr::TrainingProgress &progress,
    const tr::FrozenLevelGuard &guard) {
  check(input.data.is_cuda() && input.observed.is_cuda(), "actual training data and masks are on CUDA");
  check(model->change_projection.is_cuda() && model->change_decoder.is_cuda(), "both learned branch parameters are on CUDA");
  model->train();
  optimizer.zero_grad();
  const auto masks = r::configurable_spectral_make_training_mask(input.observed, model->config().level,
      tr::training_counter_seed(progress.mask_seed, progress.attempted_batches));
  const auto out = model->forward(input, masks.hidden);
  check(out.loss.is_cuda() && torch::isfinite(out.loss).item<bool>(), "finite actual CUDA loss");
  out.loss.backward();
  const auto projection_gradient = model->change_projection.grad();
  const auto decoder_gradient = model->change_decoder.grad();
  for (const auto &gradient : {projection_gradient, decoder_gradient})
    check(gradient.defined() && gradient.is_cuda() && torch::isfinite(gradient).all().item<bool>(),
        "finite actual CUDA branch gradients");
  if (progress.completed_updates == 0)
    check(projection_gradient.eq(0).all().item<bool>() && decoder_gradient.abs().sum().item<double>() > 0,
        "zero residual decoder first opens through a nonzero CUDA decoder gradient");
  else
    check(projection_gradient.abs().sum().item<double>() > 0 && decoder_gradient.abs().sum().item<double>() > 0,
        "both branches receive nonzero CUDA gradients after the first update");
  for (const auto &parameter : model->level->parameters())
    check(!parameter.grad().defined(), "parent is structurally absent from the gradient graph");
  guard.discard_gradients(model, optimizer);
  const auto old_projection = model->change_projection.detach().clone();
  const auto old_decoder = model->change_decoder.detach().clone();
  torch::nn::utils::clip_grad_norm_(model->parameters(), 1.);
  optimizer.step();
  check(!torch::equal(old_decoder, model->change_decoder.detach()), "actual CUDA decoder weights moved");
  if (progress.completed_updates > 0)
    check(!torch::equal(old_projection, model->change_projection.detach()), "actual CUDA projection weights moved with nonzero gradients");
  ++progress.completed_updates;
  ++progress.attempted_batches;
  protected_state(model, optimizer, guard);
  return out.loss.detach().clone();
}
} // namespace

int main() try {
  check(torch::cuda::is_available(), "CUDA required");
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  auto level_config = r::default_content_ordered_relation_config();
  level_config.channel_ids = {7, 41, 99};
  level_config.history_length = 32;
  level_config.encoder_width = 32;
  level_config.num_layers = 1;
  level_config.feedforward_width = 64;
  level_config.decoder_hidden_width = 64;
  const r::SeparatedChangeConfig native36{level_config, 4};
  const r::SeparatedChangeConfig native40{level_config, 8};
  auto invalid = native36;
  invalid.change_width = 0;
  rejects([&] { r::validate_separated_change_config(invalid); }, "zero change width rejected");
  check(r::separated_change_native_width(native36) == 36, "declared native36 geometry");

  torch::manual_seed(941001);
  r::ContentOrderedRelationModel parent(level_config);
  { torch::NoGradGuard no_grad; parent->relation_projection->weight.fill_(.003); }
  const auto time = torch::arange(32, torch::kFloat64).view({1,1,32,1});
  const auto feature = torch::arange(9, torch::kFloat64).view({1,1,1,9});
  const auto channel = torch::arange(3, torch::kFloat64).view({1,3,1,1});
  const auto batch = torch::arange(3, torch::kFloat64).view({3,1,1,1});
  auto values = (.13*time + .11*feature + .2*channel + .3*batch).sin() + .02*time;
  auto observed = torch::ones(values.sizes(), torch::kBool);
  observed[0][1][7][3] = false;
  observed[1][2][11][2] = false;
  r::Input raw{values, observed, torch::tensor(level_config.channel_ids, torch::kInt64),
      torch::tensor({1000.,2000.,3000.}, torch::kFloat64), level_config.sampling_interval};
  const auto scaler = r::fit_configurable_spectral_scaler(raw, level_config);
  const auto input = r::transform_configurable_spectral_input(scaler, raw, level_config);
  check(r::consecutive_change_bank(input, level_config).counts[0][1][3].item<int64_t>() == 29,
      "one missing tick removes both adjacent pairs without bridging");
  auto isolated = input;
  isolated.observed = torch::arange(32, input.data.options().dtype(torch::kInt64)).remainder(2).eq(0)
      .view({1,1,32,1}).expand_as(input.observed).clone();
  const auto isolated_bank = r::consecutive_change_bank(isolated, level_config);
  check(isolated_bank.counts.eq(0).all().item<bool>() && isolated_bank.values.eq(0).all().item<bool>(),
      "isolated visible endpoints never create changes across gaps");

  Temporary temporary;
  const auto parent_file = (temporary.path/"parent.pt").string();
  r::save_content_ordered_relation_checkpoint(parent_file, parent, scaler, 2048);
  auto loaded_parent = r::load_content_ordered_relation_checkpoint(parent_file, level_config.device, &level_config);
  torch::manual_seed(942001);
  r::SeparatedChangeModel model(native36);
  tr::warm_start_level(model, loaded_parent.model);
  auto optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), tr::content_training_adamw_options());
  tr::FrozenLevelGuard guard(model);
  protected_state(model, *optimizer, guard);
  {
    torch::NoGradGuard no_grad;
    model->eval();
    const auto native = model->encode(input).z_contextual_global;
    const auto old = loaded_parent.model->encode(input).z_contextual_global;
    check(native.sizes() == torch::IntArrayRef({3,36}) && torch::equal(native.narrow(1,0,32), old),
        "rank2 native36 with exact saved parent32 prefix");
    check(native.narrow(1,32,4).abs().sum().item<double>() > 0, "initial separate projection exposes four change coordinates");
    check(torch::equal(model->decode(native,input.channel_ids), loaded_parent.model->decode(old,input.channel_ids)),
        "zero residual decoder preserves initial reconstruction exactly");
    auto poisoned = input;
    poisoned.data = input.data.clone();
    poisoned.data.masked_fill_(input.observed.logical_not(), std::numeric_limits<float>::quiet_NaN());
    check(torch::equal(model->encode(poisoned).z_contextual_global,native), "unobserved NaN storage cannot affect either block");
    auto missing = poisoned;
    missing.observed = torch::zeros_like(input.observed);
    check(model->encode(missing).z_contextual_global.eq(0).all().item<bool>(), "all-missing output is zero");
    const auto masks = r::configurable_spectral_make_training_mask(input.observed, level_config, 942005);
    auto hidden_poison = input;
    hidden_poison.data = input.data.clone();
    hidden_poison.data.masked_fill_(masks.hidden, 10000.);
    const auto clean_forward = model->forward(input,masks.hidden);
    const auto poisoned_forward = model->forward(hidden_poison,masks.hidden);
    check(torch::equal(clean_forward.encoding.z_contextual_global,poisoned_forward.encoding.z_contextual_global) &&
        torch::equal(clean_forward.reconstruction,poisoned_forward.reconstruction),
        "hidden target storage cannot leak into the parent or new change coordinates");
    auto partial_config = native36;
    partial_config.level.history_length = 33;
    r::SeparatedChangeModel partial_model(partial_config);
    r::Input partial_input{torch::cat({input.data,torch::zeros({3,3,1,9},input.data.options())},2),
        torch::cat({input.observed,torch::zeros({3,3,1,9},input.observed.options())},2),
        input.channel_ids,input.endpoints,input.sampling_interval};
    const auto partial = partial_model->forward(partial_input,
        r::configurable_spectral_make_training_mask(partial_input.observed,partial_config.level,1101).hidden);
    check(partial.reconstruction.sizes() == torch::IntArrayRef({3,3,33,9}) && torch::isfinite(partial.loss).item<bool>(),
        "partial history retained by both decoder paths");
  }
  const auto settings = (temporary.path/"settings.conf").string();
  {
    std::ofstream file(settings);
    file << "ENCODER { MODEL_TAG=RPB-v24; LEVEL_EMBEDDING_SIZE=32; EMBEDDING_SIZE=36; HISTORY_LENGTH=32; CHANNEL_IDS=7,41,99; }\n";
  }
  const auto parsed = r::read_separated_change_settings(settings);
  check(parsed.change_width == 4 && parsed.level.export_width == 32, "settings preserve native36 and parent32 dimensions");
  tr::TrainingIdentity identity{"artificial-source","artificial-split","v24-separate-4;parent-fixture2048"};
  tr::TrainingProgress progress{0,0,942002,942003,942004};
  update(input,model,*optimizer,progress,guard);
  rejects([&] { tr::warm_start_level(model,parent); }, "trained residual cannot silently restart");
  const auto state = (temporary.path/"training.pt").string();
  tr::save_training_state(state,model,scaler,*optimizer,identity,progress);
  rejects([&] { tr::save_training_state(state,model,scaler,*optimizer,identity,progress); }, "immutable training state refuses overwrite");
  rejects([&] { tr::load_training_state(state,native40,identity,scaler.identity()); }, "native40 expected geometry rejects native36 typed training state");
  for (int field = 0; field < 3; ++field) {
    auto wrong_identity = identity;
    if (field == 0) wrong_identity.input_source_identity += "-wrong";
    if (field == 1) wrong_identity.split_identity += "-wrong";
    if (field == 2) wrong_identity.recipe_identity += "-wrong";
    rejects([&] { tr::load_training_state(state,native36,wrong_identity,scaler.identity()); }, "typed resume rejects wrong source, split or recipe identity");
  }
  rejects([&] { tr::load_training_state(state,native36,identity,scaler.identity()+"-wrong"); }, "typed resume rejects wrong TRAIN scaler");
  auto restored = tr::load_training_state(state,native36,identity,scaler.identity());
  tr::FrozenLevelGuard resumed_guard(restored.model);
  check(restored.progress.completed_updates == progress.completed_updates &&
      restored.progress.attempted_batches == progress.attempted_batches &&
      restored.progress.row_seed == progress.row_seed && restored.progress.mask_seed == progress.mask_seed &&
      restored.progress.context_seed == progress.context_seed, "typed resume retains counters and seed identity exactly");
  protected_state(restored.model,*restored.optimizer,resumed_guard);
  const auto next_loss = update(input,model,*optimizer,progress,guard);
  const auto resumed_loss = update(input,restored.model,*restored.optimizer,restored.progress,resumed_guard);
  check(torch::equal(next_loss,resumed_loss), "resumed next CUDA loss is exact");
  const auto first_parameters = model->named_parameters();
  const auto second_parameters = restored.model->named_parameters();
  for (const auto &parameter : first_parameters)
    check(torch::equal(parameter.value(),second_parameters[parameter.key()]), "resumed next CUDA parameters are exact");
  const auto second_buffers = restored.model->named_buffers();
  for (const auto &buffer : model->named_buffers())
    check(torch::equal(buffer.value(),second_buffers[buffer.key()]), "resumed CUDA buffers are exact");
  {
    torch::NoGradGuard no_grad;
    model->eval();
    const auto native = model->encode(input).z_contextual_global;
    check(torch::equal(native.narrow(1,0,32),loaded_parent.model->encode(input).z_contextual_global), "trained parent prefix stays exact");
    const auto permutation = torch::tensor({2,0,1},torch::TensorOptions().device(level_config.device).dtype(torch::kInt64));
    r::Input shuffled{input.data.index_select(1,permutation),input.observed.index_select(1,permutation),
        input.channel_ids.index_select(0,permutation),input.endpoints,input.sampling_interval};
    const auto shuffled_native = model->encode(shuffled).z_contextual_global;
    const auto parent_shuffled = loaded_parent.model->encode(shuffled).z_contextual_global;
    check(torch::equal(shuffled_native.narrow(1,0,32),parent_shuffled), "permuted prefix matches parent in the same physical layout exactly");
    check(torch::equal(native.narrow(1,32,4),shuffled_native.narrow(1,32,4)), "four change coordinates are exactly permutation invariant");
    // Keep the inherited v20 learned-path tolerance for physical GPU row reorder.
    check(torch::allclose(native.narrow(1,0,32),parent_shuffled,1e-5,1e-6), "inherited prefix follows semantic permutation tolerance");
    const auto reordered = model->decode(shuffled_native,shuffled.channel_ids);
    const auto expected = model->decode(native,input.channel_ids).index_select(1,permutation);
    check(torch::allclose(reordered,expected,1e-5,1e-6), "combined decoder follows inherited semantic permutation tolerance");
    std::cout << "permutation_prefix_max_abs=" << (native.narrow(1,0,32)-parent_shuffled).abs().max().item<double>()
        << " permutation_decoder_max_abs=" << (reordered-expected).abs().max().item<double>() << '\n';
  }
  const auto inference = (temporary.path/"inference.pt").string();
  r::save_separated_change_checkpoint(inference,model,scaler,2);
  auto reloaded = r::load_separated_change_checkpoint(inference,level_config.device,&native36);
  {
    torch::NoGradGuard no_grad;
    check(torch::equal(model->encode(input).z_contextual_global,reloaded.model->encode(input).z_contextual_global), "typed native36 inference reload is exact");
  }
  rejects([&] { r::load_separated_change_checkpoint(inference,level_config.device,&native40); }, "native40 expected geometry rejects native36 typed inference");
  const auto inference40 = (temporary.path/"inference40.pt").string();
  r::SeparatedChangeModel reference40(native40);
  tr::warm_start_level(reference40,loaded_parent.model);
  r::save_separated_change_checkpoint(inference40,reference40,scaler,0);
  rejects([&] { r::load_separated_change_checkpoint(inference40,level_config.device,&native36); }, "native36 expected geometry rejects native40 typed inference");
  rejects([&] { r::load_consecutive_change_checkpoint(inference,level_config.device,&level_config); }, "v23 reader rejects v24 inference");
  rejects([&] { r::load_separated_change_checkpoint(parent_file,level_config.device,&native36); }, "v24 reader rejects v22 inference");
  rejects([&] { r::load_separated_change_checkpoint(state,level_config.device,&native36); }, "inference reader rejects training archive");
  rejects([&] { tr::load_training_state(inference,native36,identity,scaler.identity()); }, "training reader rejects inference archive");
  rejects([&] { r::save_separated_change_checkpoint(inference,model,scaler,2); }, "immutable inference refuses overwrite");
  { torch::NoGradGuard no_grad; model->level->relation_projection->weight.add_(.001); }
  rejects([&] { guard.validate(model,*optimizer); }, "protected parent corruption detected");
  std::cout << "PASS separate native36: exact parent32 plus learned4, zero initial residual, masked poison/hidden targets/missingness, no gap bridging, partial history, settings dimensions, semantic permutation, actual CUDA decoder then projection learning, protected values/no moments, typed immutable inference and training/refusal, D36/D40 rejection, exact resumed next loss/weights\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
