// SPDX-License-Identifier: MIT
#pragma once

// Single-task paired timing orchestration. Numerical encoder, query and readout
// implementations remain in their existing public APIs. The CLI only dispatches
// this closed protocol; no historical evaluator entrypoint is imported.
namespace embedding::evaluation::structured_timing {
int run(int argc, char **argv);
}
