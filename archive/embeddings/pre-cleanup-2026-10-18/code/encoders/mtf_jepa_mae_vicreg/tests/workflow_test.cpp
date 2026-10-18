#include "embedding/encoders/mtf_jepa_mae_vicreg/workflow.h"
#include "test_support.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;

namespace {

struct TemporaryFiles {
  std::filesystem::path directory;

  TemporaryFiles() {
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() /
                ("embedding-workflow-test-" + std::to_string(id));
    test::check(std::filesystem::create_directory(directory),
                "could not create a unique test directory");
  }

  ~TemporaryFiles() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }

  std::string file(const std::string &name) const {
    return (directory / name).string();
  }
};

template <typename Function>
void rejects(Function function, const std::string &label) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception &) {
    rejected = true;
  }
  test::check(rejected, label + " was accepted");
}

mtf::Settings settings() {
  auto value = mtf::default_settings();
  value.model = test::small_config<mtf::Config>();
  value.model.dropout = 0.1;
  value.model.use_channel_vicreg = true;
  value.model.target_ema_tau = 0.97;
  value.model.strict_jepa_support = true;
  value.steps = 5;
  value.batch_size = 3;
  value.seed = 71;
  value.threads = 1;
  value.log_every = 2;
  value.checkpoint_every = 2;
  value.learning_rate = 0.0023;
  value.weight_decay = 0.007;
  value.gradient_clip_norm = 0.75;
  return value;
}

void check_settings(const mtf::Settings &actual,
                    const mtf::Settings &expected) {
  test::check(mtf::settings_text(actual) == mtf::settings_text(expected),
              "serialized settings changed on round-trip");
  test::check(actual.model.channel_count == expected.model.channel_count &&
                  actual.model.history_length == expected.model.history_length &&
                  actual.model.input_width == expected.model.input_width &&
                  actual.model.latent_dim == expected.model.latent_dim &&
                  actual.model.time_scales == expected.model.time_scales &&
                  actual.model.scale_strides == expected.model.scale_strides &&
                  actual.model.dropout == expected.model.dropout &&
                  actual.model.use_channel_vicreg == expected.model.use_channel_vicreg &&
                  actual.model.strict_jepa_support == expected.model.strict_jepa_support &&
                  actual.model.target_ema_tau == expected.model.target_ema_tau,
              "model settings changed on round-trip");
  test::check(actual.steps == expected.steps && actual.batch_size == expected.batch_size &&
                  actual.seed == expected.seed && actual.threads == expected.threads &&
                  actual.log_every == expected.log_every &&
                  actual.checkpoint_every == expected.checkpoint_every &&
                  actual.learning_rate == expected.learning_rate &&
                  actual.weight_decay == expected.weight_decay &&
                  actual.gradient_clip_norm == expected.gradient_clip_norm,
              "training settings changed on round-trip");
}

void test_config(const TemporaryFiles &files) {
  const auto original = settings();
  check_settings(mtf::parse_settings(mtf::settings_text(original)), original);
  auto implicit_strides = original;
  implicit_strides.model.scale_strides.clear();
  check_settings(mtf::parse_settings(mtf::settings_text(implicit_strides)),
                 implicit_strides);
  const auto path = files.file("settings.conf");
  {
    std::ofstream output(path);
    output << mtf::settings_text(original);
  }
  check_settings(mtf::read_settings(path), original);
  for (const auto &invalid : {"unknown_option = 1\n", "time_scales=\n", "batch_size = 0\n",
                             "learning_rate = nan\n", "dropout = 2\n",
                             "checkpoint_every = -1\n", "strict_jepa_support = sometimes\n",
                             "history_length = sixteen\n", "use_frequency_tokens = perhaps\n",
                             "channel_count = 1\nchannel_count = 2\n"}) {
    rejects([&] { mtf::parse_settings(invalid); },
            std::string("invalid config: ") + invalid);
  }
  auto old_text = mtf::settings_text(original);
  for (const auto &key : {"strict_jepa_support=", "checkpoint_every="}) {
    const auto begin = old_text.find(key);
    const auto end = old_text.find('\n', begin);
    old_text.erase(begin, end - begin + 1);
  }
  const auto old_settings = mtf::parse_settings(old_text);
  test::check(!old_settings.model.strict_jepa_support && old_settings.checkpoint_every == 0,
              "old checkpoints must retain legacy masking and final-only saves");
}

double train_step(mtf::Model &model, torch::optim::AdamW &optimizer,
                  const mtf::Settings &config,
                  const embedding::Batch &batch, int64_t seed) {
  model->train();
  torch::manual_seed(seed);
  optimizer.zero_grad();
  auto output = model->forward(batch.data, batch.feature_mask);
  test::finite(output.loss, "resume test loss");
  output.loss.backward();
  torch::nn::utils::clip_grad_norm_(model->parameters(), config.gradient_clip_norm);
  optimizer.step();
  model->update_target_network();
  return output.loss.item<double>();
}

std::vector<torch::Tensor> trainable(mtf::Model &model) {
  std::vector<torch::Tensor> result;
  for (const auto &parameter : model->parameters())
    if (parameter.requires_grad())
      result.push_back(parameter);
  return result;
}

void test_cli_embeddings(const TemporaryFiles &files,
                         const std::string &checkpoint_path,
                         const std::string &input_path,
                         mtf::Model &model,
                         const embedding::Batch &batch) {
  model->eval();
  torch::NoGradGuard no_grad;
  const auto expected = model->encode(batch.data, batch.feature_mask);
  const auto output_path = files.file("embeddings.pt");
  std::vector<std::string> arguments = {"embedding", "embed", "--checkpoint",
                                         checkpoint_path, "--input", input_path,
                                         "--output", output_path, "--batch-size", "2"};
  std::vector<char *> argv;
  for (auto &argument : arguments)
    argv.push_back(argument.data());
  test::check(mtf::run_cli(static_cast<int>(argv.size()), argv.data()) == 0,
              "embed command failed");
  torch::serialize::InputArchive archive;
  archive.load_from(output_path, torch::kCPU);
  torch::Tensor global, channels, sample_valid, channel_valid;
  archive.read("pooled_embedding", global, true);
  archive.read("pooled_by_channel", channels, true);
  archive.read("sample_valid_mask", sample_valid, true);
  archive.read("channel_valid_mask", channel_valid, true);
  test::close(global, expected.pooled_embedding, "CLI global embeddings");
  test::close(channels, expected.pooled_by_channel, "CLI channel embeddings");
  test::check(torch::equal(sample_valid, expected.sample_valid_mask) &&
                  torch::equal(channel_valid, expected.channel_valid_mask),
              "CLI validity masks changed");
}

void test_checkpoint_and_batch(const TemporaryFiles &files) {
  const auto config = settings();
  torch::manual_seed(config.seed);
  auto batch = mtf::synthetic_batch(config.model, 5, 2);
  batch.feature_mask.index_put_({0, 1}, false);
  batch.feature_mask.index_put_({1}, false);
  batch.data.masked_fill_(batch.feature_mask.logical_not(), 0.0);
  const auto input_path = files.file("input.pt");
  embedding::save_batch(input_path, batch);
  const auto restored_batch = mtf::load_batch(input_path, config.model);
  test::close(restored_batch.data, batch.data, "input archive data", 0, 0);
  test::check(torch::equal(restored_batch.feature_mask, batch.feature_mask),
              "input archive feature mask changed");
  auto wrong_shape = config.model;
  wrong_shape.history_length += 1;
  rejects([&] { mtf::load_batch(input_path, wrong_shape); },
          "input archive with incompatible shape");

  auto model = mtf::Model(config.model);
  const auto options = torch::optim::AdamWOptions(config.learning_rate)
                           .weight_decay(config.weight_decay);
  torch::optim::AdamW optimizer(trainable(model), options);
  train_step(model, optimizer, config, batch, 83);
  const auto checkpoint_path = files.file("checkpoint.pt");
  mtf::save_checkpoint(checkpoint_path, config, model, optimizer, 1);
  auto restored = mtf::load_checkpoint(checkpoint_path);
  check_settings(restored.settings, config);
  test::check(restored.completed_steps == 1, "checkpoint lost completed steps");
  const auto parameters = model->named_parameters();
  const auto restored_parameters = restored.model->named_parameters();
  for (const auto &parameter : parameters)
    test::close(restored_parameters[parameter.key()], parameter.value(),
                "public checkpoint parameter " + parameter.key(), 0, 0);

  test_cli_embeddings(files, checkpoint_path, input_path, restored.model, batch);
  torch::optim::AdamW restored_optimizer(trainable(restored.model), options);
  mtf::load_optimizer(checkpoint_path, restored_optimizer, torch::kCPU);
  const double uninterrupted_loss = train_step(model, optimizer, config, batch, 89);
  const double resumed_loss = train_step(restored.model, restored_optimizer,
                                         restored.settings, batch, 89);
  test::check(uninterrupted_loss == resumed_loss,
              "resumed forward does not match uninterrupted training");
  for (const auto &parameter : parameters)
    test::close(restored_parameters[parameter.key()], parameter.value(),
                "optimizer resume parameter " + parameter.key(), 0, 0);
}

void test_cli_training_continuation(const TemporaryFiles &files) {
  auto config = settings();
  config.model.strict_jepa_support = false;
  const auto config_path = files.file("training.conf");
  { std::ofstream output(config_path); output << mtf::settings_text(config); }
  const auto input_path = files.file("training.pt");
  torch::manual_seed(300);
  embedding::save_batch(input_path, mtf::synthetic_batch(config.model, 6, 1));
  const auto uninterrupted = files.file("uninterrupted.pt");
  const auto split = files.file("split.pt");
  auto invoke = [](std::vector<std::string> arguments) {
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    test::check(mtf::run_cli(static_cast<int>(argv.size()), argv.data()) == 0,
                "training CLI failed");
  };
  invoke({"embedding", "train", "--config", config_path, "--input", input_path,
          "--checkpoint", uninterrupted, "--steps", "3", "--checkpoint-every", "1"});
  invoke({"embedding", "train", "--config", config_path, "--input", input_path,
          "--checkpoint", split, "--steps", "1", "--checkpoint-every", "1"});
  invoke({"embedding", "train", "--resume", split, "--input", input_path,
          "--checkpoint", split, "--steps", "2"});
  auto whole = mtf::load_checkpoint(uninterrupted);
  auto continued = mtf::load_checkpoint(split);
  test::check(whole.completed_steps == 3 && continued.completed_steps == 3 &&
                  continued.settings.checkpoint_every == 1,
              "CLI resume lost steps or periodic-save settings");
  const auto expected = whole.model->named_parameters();
  for (const auto &parameter : continued.model->named_parameters())
    test::close(parameter.value(), expected[parameter.key()],
                "CLI uninterrupted/continued " + parameter.key(), 0, 0);
}

} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    TemporaryFiles files;
    test_config(files);
    test_checkpoint_and_batch(files);
    test_cli_training_continuation(files);
    std::cout << "PASS: configuration, input archives, checkpoint settings, optimizer resume, CLI embeddings\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
