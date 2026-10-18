#include "embedding/encoders/mtf_jepa_mae_vicreg/model.h"
#include "test_support.h"

#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;

namespace {

using test::check;
using test::close;
using test::finite;
using torch::indexing::Slice;

void test_shapes_and_masks() {
  auto config = test::small_config<mtf::Config>();
  torch::manual_seed(11);
  auto model = mtf::Model(config);
  model->eval();
  torch::NoGradGuard no_grad;
  auto data = test::input();
  auto mask = torch::ones_like(data, torch::kBool);
  mask.index_put_({0, 1}, false);
  mask.index_put_({1}, false);
  mask.index_put_({2, 0, Slice(4, 8), 1}, false);

  auto encoded = model->encode(data, mask);
  check(encoded.embeddings.sizes() == torch::IntArrayRef({4, 28, 8}),
        "token embeddings must be [batch, tokens, latent]");
  check(encoded.pooled_embedding.sizes() == torch::IntArrayRef({4, 8}),
        "global embeddings have the wrong shape");
  check(encoded.pooled_by_channel.sizes() == torch::IntArrayRef({4, 2, 8}),
        "channel embeddings have the wrong shape");
  check(torch::equal(encoded.sample_valid_mask,
                     torch::tensor({true, false, true, true}, torch::kBool)),
        "sample validity does not reflect missing input");
  check(torch::equal(encoded.channel_valid_mask,
                     torch::tensor({{true, false}, {false, false},
                                    {true, true}, {true, true}}, torch::kBool)),
        "channel validity does not reflect missing input");
  finite(encoded.embeddings, "embeddings");
  close(encoded.pooled_embedding.index({1}), torch::zeros({8}),
        "empty sample embedding");
  close(encoded.pooled_by_channel.index({0, 1}), torch::zeros({8}),
        "empty channel embedding");

  const auto poisoned = data.masked_fill(
      mask.logical_not(), std::numeric_limits<float>::quiet_NaN());
  const auto masked = model->encode(poisoned, mask);
  close(masked.embeddings, encoded.embeddings, "masked NaN invariance", 0, 0);
  const auto inferred = model->encode(poisoned);
  close(inferred.embeddings, encoded.embeddings, "implicit NaN mask", 0, 0);
  check(torch::equal(inferred.token_mask, encoded.token_mask),
        "implicit NaN mask changes token validity");

  auto tokens = model->tokenize(data, mask);
  torch::manual_seed(37);
  const auto masks = model->create_masks(tokens);
  check(!masks.context_mask.logical_and(masks.target_mask).any().item<bool>(),
        "JEPA context and target tokens overlap");
  check(!masks.context_mask.logical_or(masks.target_mask)
             .logical_and(tokens.token_mask.logical_not()).any().item<bool>(),
        "JEPA selects invalid tokens");
  check(!masks.context_mask.index({1}).any().item<bool>() &&
            !masks.target_mask.index({1}).any().item<bool>(),
        "JEPA selects tokens from an empty sample");
  for (int64_t batch : {0, 2, 3}) {
    check(masks.context_mask.index({batch}).any().item<bool>(),
          "valid sample has no JEPA context");
    check(masks.target_mask.index({batch}).any().item<bool>(),
          "valid sample has no JEPA targets");
  }
  // Paired time/frequency views of an identical patch must not leak targets.
  for (int64_t i = 0; i < tokens.token_mask.size(1); ++i) {
    for (int64_t j = 0; j < tokens.token_mask.size(1); ++j) {
      if (tokens.metadata.channel_id[i].item<int64_t>() ==
              tokens.metadata.channel_id[j].item<int64_t>() &&
          tokens.metadata.scale_id[i].item<int64_t>() ==
              tokens.metadata.scale_id[j].item<int64_t>() &&
          tokens.metadata.start_index[i].item<int64_t>() ==
              tokens.metadata.start_index[j].item<int64_t>() &&
          tokens.metadata.domain_id[i].item<int64_t>() !=
              tokens.metadata.domain_id[j].item<int64_t>()) {
        check(!masks.target_mask.select(1, i)
                   .logical_and(masks.context_mask.select(1, j)).any().item<bool>(),
              "paired time/frequency target leaks into context");
      }
    }
  }
  torch::manual_seed(41);
  auto output = model->forward(poisoned, mask);
  finite(output.loss, "partially missing input loss");

  config.channel_count = 1;
  auto single = mtf::Model(config);
  single->eval();
  const auto one_channel = data.slice(1, 0, 1);
  close(single->encode(one_channel).pooled_embedding,
        single->encode(one_channel.squeeze(1)).pooled_embedding,
        "rank-3 single-channel input", 0, 0);
}

void test_training_ema_and_checkpoint() {
  const auto config = test::small_config<mtf::Config>();
  torch::manual_seed(17);
  auto model = mtf::Model(config);
  auto data = test::input();
  auto parameters = model->named_parameters();
  std::map<std::string, torch::Tensor> teacher_before;
  for (const auto &parameter : parameters) {
    if (test::teacher(parameter.key())) {
      check(!parameter.value().requires_grad(), "teacher must be frozen");
      teacher_before.emplace(parameter.key(), parameter.value().detach().clone());
      close(parameter.value(), parameters[parameter.key().substr(7)],
            "initial teacher copy", 0, 0);
    }
  }
  check(!teacher_before.empty(), "model has no teacher parameters");
  check(!model->target_encode(data).requires_grad(),
        "teacher output must not require gradients");

  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.003));
  const auto online_before = parameters["encoder.latent_projection.weight"].detach().clone();
  torch::manual_seed(59);
  auto output = model->forward(data);
  const double initial_loss = output.loss.item<double>();
  finite(output.loss, "initial training loss");
  output.loss.backward();
  std::set<std::string> active_groups;
  for (const auto &parameter : parameters) {
    const auto gradient = parameter.value().grad();
    if (test::teacher(parameter.key())) {
      check(!gradient.defined(), "teacher received a gradient");
    } else if (gradient.defined()) {
      finite(gradient, "gradient " + parameter.key());
      if (gradient.abs().sum().item<double>() > 0.0)
        active_groups.insert(parameter.key().substr(0, parameter.key().find('.')));
    }
  }
  for (const auto &group : {"tokenizer", "encoder", "predictor", "mae_decoder",
                            "vicreg_stability_head"})
    check(active_groups.count(group) != 0, std::string(group) + " has no gradient");
  optimizer.step();
  check(!torch::equal(online_before, parameters["encoder.latent_projection.weight"]),
        "optimizer did not update the encoder");
  for (const auto &[name, before] : teacher_before)
    close(parameters[name], before, "optimizer changed teacher", 0, 0);

  model->update_target_network();
  for (const auto &[name, before] : teacher_before)
    close(parameters[name], before * config.target_ema_tau +
              parameters[name.substr(7)].detach() * (1.0 - config.target_ema_tau),
          "EMA update " + name);

  for (int step = 1; step < 12; ++step) {
    optimizer.zero_grad();
    torch::manual_seed(59);
    auto current = model->forward(data);
    finite(current.loss, "training loss");
    current.loss.backward();
    for (const auto &parameter : model->parameters())
      if (parameter.grad().defined())
        finite(parameter.grad(), "training gradient");
    optimizer.step();
    model->update_target_network();
  }
  torch::manual_seed(59);
  const double final_loss = model->forward(data).loss.item<double>();
  check(final_loss < initial_loss, "short fixed-batch training did not reduce loss");

  model->eval();
  torch::NoGradGuard no_grad;
  const auto before = model->encode(data);
  torch::serialize::OutputArchive output_archive;
  model->save(output_archive);
  std::stringstream checkpoint(std::ios::in | std::ios::out | std::ios::binary);
  output_archive.save_to(checkpoint);
  checkpoint.seekg(0);
  auto restored = mtf::Model(config);
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(checkpoint);
  restored->load(input_archive);
  restored->eval();
  const auto restored_parameters = restored->named_parameters();
  for (const auto &parameter : parameters)
    close(restored_parameters[parameter.key()], parameter.value(),
          "checkpoint parameter " + parameter.key(), 0, 0);
  close(restored->encode(data).pooled_embedding, before.pooled_embedding,
        "checkpoint global embedding", 0, 0);
  close(restored->encode(data).pooled_by_channel, before.pooled_by_channel,
        "checkpoint channel embeddings", 0, 0);
  std::cout << "12 training steps: loss " << initial_loss << " -> " << final_loss << '\n';
}

void test_nonfinite_encoding_rejected() {
  auto model = mtf::Model(test::small_config<mtf::Config>());
  model->eval();
  torch::NoGradGuard no_grad;
  model->named_parameters()["encoder.latent_projection.weight"].fill_(
      std::numeric_limits<float>::infinity());
  bool rejected = false;
  try { (void)model->encode(test::input()); }
  catch (const std::exception &error) {
    rejected = std::string(error.what()).find("non-finite encoding") != std::string::npos;
  }
  check(rejected, "non-finite inference must fail explicitly");
}

} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    test_shapes_and_masks();
    test_training_ema_and_checkpoint();
    test_nonfinite_encoding_rejected();
    std::cout << "PASS: shapes, missing data, masks, gradients, EMA, training, checkpoint\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
