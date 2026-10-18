// SPDX-License-Identifier: MIT
#pragma once

// Receives the original argument vector: argv[1] is "reconstruct".
// Full evaluation registers RPB; the minimum executable rejects this command.
int run_reconstruction_cli(int argc, char **argv);
