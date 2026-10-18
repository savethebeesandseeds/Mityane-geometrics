// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/native_curve.h"
#include <functional>
#include <map>

namespace embedding::evaluation {

// Explicit immutable TRAIN/VALIDATION archives and already fitted readouts.
// Vector entries follow run.repetitions order; no directory discovery or TEST
// input exists in the generic driver. Feature archives embed ordered source IDs
// and scoring labels, permitting an exact association check against observations.
struct RetainedPoolingCohort {
  uint64_t master_seed{0};
  std::string lineage;
  std::string training_observations,validation_observations;
  std::string reference_checkpoint,reference_initial_checkpoint;
  std::string reference_training_features,reference_validation_features;
  std::string reference_initial_training_features,reference_initial_validation_features;
  std::string raw_scaler;
  std::vector<std::string> reference_fits,reference_initial_fits,raw_fits,pca_fits,mask_fits;
  std::vector<std::string> reference_validation_predictions,reference_initial_validation_predictions;
  std::vector<std::string> raw_validation_predictions,pca_validation_predictions,mask_validation_predictions;
  // All explicit inputs are recorded and protected by exact bytes before/after;
  // production adapters additionally pin every file with an expected SHA-256.
  std::map<std::string,std::string> expected_sha256;
};

struct PairedPoolingRun {
  NativeCurveRun recipe;
  std::vector<RetainedPoolingCohort> cohorts;
  int64_t completed_updates{512};
  std::string reference_tag{"RPB-v4"},candidate_tag{"RPB-v5"};
  // Historical defaults preserve paired-pooling-v1; new training-view
  // candidates must declare their own truthful protocol and fresh namespace.
  std::string protocol_id{"paired-pooling-v1"};
  std::string fresh_test_namespace{"paired-pooling-v1/fresh-testing"};
  uint64_t fresh_test_stream{0x7070763174657374ULL}; // ppv1test.
  // Label-free TRAIN cohort namespace, independent of this comparison's TEST
  // namespace. Nondefault values are explicitly frozen in the card/manifest.
  std::string training_protocol_namespace{"native-curve-v1"};
};

using RetainedCurveSnapshotLoader = std::function<CurveSnapshot(
    const std::string &checkpoint_path,const ProviderFitInput &metadata)>;
// Encoder adapter verifies common named-module initialization, frozen scaler,
// schema/dataset/counter-seed identity. The shared driver never imports encoders.
using PoolingInitializationAudit = std::function<std::map<std::string,std::string>(
    const std::string &candidate_point0,const RetainedPoolingCohort &reference,
    const ProviderFitInput &metadata)>;

// Small archive/measurement helpers for explicitly bound TRAIN/VALIDATION
// cohorts. The caller owns role binding, source provenance and exclusive output
// directory reservation. These helpers do not discover files, generate data,
// fit preprocessing/readouts, or expose a TEST protocol. Loaded clean is only a
// zero-masked legal-observation schema placeholder, never hidden ground truth.
ControlledDataset load_native_development_observations(const std::string &path);
FeatureSurface extract_native_global(const CurveSnapshot &snapshot,const Batch &batch,
                                     const NativeCurveRun &recipe);
void save_native_feature_archive(const std::string &new_archive_path,
                                 const FeatureSurface &surface,const ControlledDataset &split,
                                 const NativeCurveRun &recipe);
// Writes the unchanged fixed original-patch query arrays and hierarchical
// reductions, returning their existing JSON summary. Destinations must be new;
// callback inputs are independent legal clones and RNG/thread state is restored.
std::string write_native_patch_reconstruction(const std::string &new_archive_path,
                                             const ControlledDataset &split,
                                             const CurveSnapshot &snapshot,
                                             const NativeCurveRun &recipe);

// Fixed-budget label-free candidate only. Retained reference/control readouts
// load as tensors without any fitting constructor. Candidate point0/positive
// heads fit TRAIN only using the same paired recipe. All validation and exact
// snapshot/decoder/readout witnesses precede a durable fixed-budget manifest;
// every fresh TEST draw follows it. Intact and additional30% deletion are the
// declared primary cases; the existing full fixed-readout stress sweep is reused.
void run_paired_pooling(const PairedPoolingRun &run,const NamedCurveFactory &candidate,
                        const RetainedCurveSnapshotLoader &load_reference,
                        const PoolingInitializationAudit &audit_initialization);

} // namespace embedding::evaluation
