// Optional extraction audit: reference_model.h is a staged, unmodified source
// header. It is not a dependency of the standalone model or its regular tests.
#include "embedding/encoders/mtf_jepa_mae_vicreg/model.h"
#include "reference_model.h"
#include "test_support.h"

#include <iostream>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;

namespace reference =
    cuwacunu::wikimyei::representation::encoding::mtf_jepa_mae_vicreg;

int main() {
  try {
    torch::set_num_threads(1);
    auto config = test::small_config<mtf::Config>();
    auto old_config = test::small_config<reference::mtf_jepa_mae_vicreg_config_t>();
    old_config.return_diagnostics = false;
    torch::manual_seed(101);
    auto model = mtf::Model(config);
    torch::manual_seed(101);
    auto baseline = reference::MtfJepaMaeVicreg(old_config);
    const auto parameters = model->named_parameters();
    const auto old_parameters = baseline->named_parameters();
    test::check(parameters.size() == old_parameters.size(), "parameter count changed");
    for (const auto &parameter : parameters) {
      test::check(old_parameters.contains(parameter.key()),
                  "parameter name changed: " + parameter.key());
      test::close(parameter.value(), old_parameters[parameter.key()],
                  "initial parameter " + parameter.key(), 0, 0);
    }
    auto data = test::input();
    auto mask = torch::ones_like(data, torch::kBool);
    mask.index_put_({0, 1}, false);
    mask.index_put_({2, 0, torch::indexing::Slice(4, 8), 1}, false);
    const auto encoded = model->encode(data, mask);
    const auto old_encoded = baseline->encode(data, mask);
    test::close(encoded.embeddings, old_encoded.embeddings, "token embeddings");
    test::close(encoded.pooled_embedding, old_encoded.pooled_embedding, "global embeddings");
    test::close(encoded.pooled_by_channel, old_encoded.pooled_by_channel, "channel embeddings");
    test::check(torch::equal(encoded.token_mask, old_encoded.token_mask), "token mask changed");

    torch::manual_seed(103);
    auto output = model->forward(data, mask);
    torch::manual_seed(103);
    auto old_output = baseline->forward(data, mask);
#define COMPARE(field) test::close(output.field, old_output.field, #field)
    COMPARE(loss);
    COMPARE(loss_jepa);
    COMPARE(loss_mae);
    COMPARE(loss_mae_time);
    COMPARE(loss_mae_frequency);
    COMPARE(loss_tf_align);
    COMPARE(loss_vicreg);
    COMPARE(loss_vicreg_global);
    COMPARE(loss_vicreg_channel);
    COMPARE(embeddings);
#undef COMPARE
    const auto old_tokens = baseline->tokenize(data, mask);
    torch::manual_seed(103);
    const auto old_masks = baseline->create_masks(old_tokens);
    test::check(torch::equal(output.jepa_context_mask, old_masks.context_mask),
                "JEPA context mask changed");
    test::check(torch::equal(output.jepa_target_mask, old_masks.target_mask),
                "JEPA target mask changed");

    output.loss.backward();
    old_output.loss.backward();
    for (const auto &parameter : parameters) {
      const auto gradient = parameter.value().grad();
      const auto old_gradient = old_parameters[parameter.key()].grad();
      test::check(gradient.defined() == old_gradient.defined(),
                  "gradient availability changed: " + parameter.key());
      if (gradient.defined())
        test::close(gradient, old_gradient, "gradient " + parameter.key(), 1e-4, 1e-6);
    }
    std::cout << "PASS: original baseline parameters, embeddings, masks, losses, gradients\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
