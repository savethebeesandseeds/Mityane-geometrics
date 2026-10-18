// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  try {
    return embedding::encoders::raw_patch_bottleneck_mae::run_cli(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
