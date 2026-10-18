// SPDX-License-Identifier: MIT
#include "embedding/shared/objectives.h"
#include "shared_test_support.h"

#include <iostream>

int main() {
  try {
    torch::set_num_threads(1);
    auto one = torch::tensor({{{1.0f, 2.0f, 3.0f}}},
                              torch::TensorOptions().requires_grad(true));
    auto mask = torch::ones({1, 1}, torch::kBool);
    embedding::vicreg_stability_loss_options_t options;
    options.invariance_weight = 0.0;
    auto singleton = embedding::compute_vicreg_stability_loss(one, mask, one, mask, options);
    test::check(singleton.valid_rows == 1 && !singleton.statistics_supported,
                "singleton statistics must be diagnosed as unsupported");
    test::close(singleton.covariance_loss, torch::zeros({}, one.options()),
                "singleton covariance");
    singleton.loss.backward();
    test::finite(one.grad(), "singleton gradient");
    test::check(one.grad().abs().sum().item<double>() == 0.0,
                "singleton variance should have no diversity gradient");

    auto diverse = torch::tensor({{{-0.2f, 0.1f, 0.0f}},
                                  {{0.2f, -0.1f, 0.3f}},
                                  {{50.0f, 50.0f, 50.0f}}},
                                 torch::TensorOptions().requires_grad(true));
    auto joint = torch::tensor({{true}, {true}, {false}}, torch::kBool);
    auto result = embedding::compute_vicreg_stability_loss(diverse, joint, diverse, joint, options);
    test::check(result.valid_rows == 2 && result.statistics_supported,
                "joint valid rows must determine statistical support");
    result.loss.backward();
    test::finite(diverse.grad(), "diverse gradient");
    test::check(diverse.grad().narrow(0, 0, 2).abs().sum().item<double>() > 0,
                "multiple rows should receive a diversity gradient");
    test::check(diverse.grad()[2].abs().sum().item<double>() == 0,
                "invalid rows must not receive gradients");
    auto none = embedding::compute_vicreg_stability_loss(
        diverse.detach(), torch::zeros({3, 1}, torch::kBool),
        diverse.detach(), torch::zeros({3, 1}, torch::kBool));
    test::check(none.valid_rows == 0 && !none.statistics_supported && none.loss.item<double>() == 0,
                "empty statistics must be safe and diagnosed");
    std::cout << "PASS: VICReg empty/singleton/multiple valid-row diagnostics and gradients\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
