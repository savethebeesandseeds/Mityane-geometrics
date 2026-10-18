#pragma once

#include <torch/torch.h>

#include <stdexcept>
#include <string>

namespace test {

inline void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

inline void finite(const torch::Tensor &tensor, const std::string &label) {
  check(tensor.defined() && torch::isfinite(tensor).all().item<bool>(),
        label + " contains a non-finite value");
}

inline void close(const torch::Tensor &actual, const torch::Tensor &expected,
                  const std::string &label, double rtol = 1e-5,
                  double atol = 1e-6) {
  check(actual.defined() && expected.defined(), label + " is undefined");
  check(actual.sizes() == expected.sizes(), label + " shape mismatch");
  check(torch::allclose(actual, expected, rtol, atol), label + " mismatch");
}

} // namespace test
