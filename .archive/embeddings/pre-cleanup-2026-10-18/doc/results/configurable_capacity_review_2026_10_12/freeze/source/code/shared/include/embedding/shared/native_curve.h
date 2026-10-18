// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/archive_readout.h"
#include "embedding/shared/learning_curve.h"

namespace embedding::evaluation {

struct NativeCurveRun {
  NativeCurveRun();
  // New policy1.2 card. Historical feature/curve card validators are unchanged.
  EvaluationCard card;
  std::string output_directory;
  std::string source_fingerprint{"unrecorded"};
  std::string git_head{"unrecorded"},git_dirty{"unrecorded"};
  std::string model_tag{"RPB-v4"};
  int64_t export_width{32},patch_length{8},sanity_budget{512};
  double huber_delta{1.0},missing_rate{0.1};
  Task selection_task{Task::lag_sign};
  // Selection-task curve only. Other declared tasks retain point0 and the
  // separately frozen sanity budget, and never choose a budget from their tests.
  std::vector<int64_t> milestones{0,128,512,2048};
  std::vector<ArchiveReadoutRepetition> repetitions{
      {"rep-1",2701},{"rep-2",2802},{"rep-3",2903}};
  double ridge_penalty{1.0};
  int64_t tiny_hidden{16},tiny_steps{100},bootstrap_replicates{1000};
  double tiny_learning_rate{0.01};
  uint64_t fresh_test_stream{0x6e63763174657374ULL}; // named ncv1test stream.
  bool stress_sweep{true};
  // TRAIN/VALIDATION asset preparation only. Requires test_pairs==0 and
  // stress_sweep==false; uses native-development-v1 and every task's declared
  // milestones, checks every retained point, then durably records completion.
  // No budget selection, TEST generation, or stress evaluation occurs.
  bool development_only{false};
};

// Sole label-free continuous trainer; no encoder dependency in this driver.
// A snapshot supplies exactly one typed global export of export_width. Extra
// diagnostic surfaces are ignored. Native heads fit TRAIN at each point with
// fixed paired recipes; no PCA after an encoder. Raw/PCA/mask controls fit once
// per cohort and remain frozen. Point0 is the same path's untrained comparator.
// Selection uses mean native ridge validation accuracy over all declared masters
// with identical validation support across checkpoints; positive budgets only,
// unsupported entire budget excluded, exact ties choose the smaller. Persisted
// selection precedes ALL fresh TEST generation, including sanity tasks. Selected
// snapshots, reconstruction witnesses and fitted readouts are checked unchanged
// after later training, then reused without test/stress fitting.
// development_only instead returns after all TRAIN/VALIDATION assets and
// immutable witnesses, before the selection phase.
void run_native_curve(const NativeCurveRun &run,const NamedCurveFactory &factory);

} // namespace embedding::evaluation
