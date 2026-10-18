// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/learning_curve.h"
#include <cstdint>
#include <string>

namespace embedding::encoders::raw_patch_bottleneck_mae {

struct ContextReplayOptions {
  std::string audited_checkpoint, training_archive, replay_checkpoint;
  int64_t expected_replay_updates{512}, max_completed_updates{2048};
};

// Fresh CUDA training under the unchanged context-deletion policy. Before any
// update beyond expected_replay_updates, a new replay checkpoint must exactly
// match the audited parent model, buffers, AdamW, scaler and counter witnesses.
// The same live optimizer then continues; ordinary tagged resume stays blocked.
// All checkpoint paths are new. Each saved point has .replay-audit.pt/.json
// companions, with CPU state witnesses and replay/continuation costs separated.
embedding::evaluation::CurveTrainerFactory
make_context_replay_trainer(const ContextReplayOptions &options);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
