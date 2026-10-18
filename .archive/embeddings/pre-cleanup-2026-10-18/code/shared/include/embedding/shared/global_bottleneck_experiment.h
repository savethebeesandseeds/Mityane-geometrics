// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/learning_curve.h"

namespace embedding::evaluation {

struct GlobalBottleneckRun {
  GlobalBottleneckRun();
  // Geometry, legal schema, task list, source-pair budgets and development seeds.
  // Task subsets and larger training cohorts require a separately frozen card.
  EvaluationCard card;
  std::string output_directory;
  std::string source_fingerprint{"unrecorded"}, git_head{"unrecorded"}, git_dirty{"unrecorded"};
  int64_t patch_length{8};
  double huber_delta{1.0};
  std::vector<int64_t> milestones{0, 128, 512};
  Task selection_task{Task::lag_sign};
  // Zero retains validation-budget selection. A positive declared milestone is
  // used for testing only if its complete primary validation population is supported.
  int64_t fixed_test_budget{0};
  // Zero retains the ordinary development partition. A named nonzero stream
  // supplies an independent validation cohort unaffected by training-pair count.
  uint64_t validation_seed_stream{0};
  // With an independent validation stream, draw a fixed training reservoir and
  // use its first complete source pairs. Equal reservoirs give nested cohorts.
  int64_t training_reservoir_pairs{0};
  uint64_t fresh_test_stream{0x67626f7474657374ULL};
  // The shared fixed-readout 12-case stress protocol is applied only to selected
  // checkpoints/readouts on fresh testing observations. It never fits under stress.
  bool stress_sweep{true};
};

// Factories are label-free CurveTrainer factories, registered in this fixed order:
// current_mixer, mean_global, learned_global. Each snapshot exposes one typed
// global and one typed channel-concatenation surface, renamed by the runner.
// Selection uses the mean validation matched-global ridge accuracy over every
// variant/master, on within-master validity common to all three variants.
// One unsupported focal fit/class population excludes the entire nonzero budget;
// exact ties select the smaller budget unless fixed_test_budget is declared.
// No testing data exists before the supported budget is persisted.
void run_global_bottleneck_experiment(const GlobalBottleneckRun &run,
                                     const std::vector<NamedCurveFactory> &factories);

} // namespace embedding::evaluation
