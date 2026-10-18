#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Report immutable saved v29 parent-modulated decoder evidence using JSON/CSV and hashes only.

This program loads no tensor archive, calls no encoder and fits no readout.
Run exclusively in the existing managed cuwacunu container.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
TAG = "RPB-v29.market-01"
NATIVE36 = "RPB-v24.market-01.alt-01"
CARD = "fd900adeaf93fef6258ca2bfe344cd1dbb69ffa6cc03e00f697e2000b455f666"
FIXED = "5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14"
PARENT_SHA = "e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c"
INIT_SHA = "2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108"
SHARED_SHA = "dbc208fb7c858909ff6f04e70b939adaab1c3e8c27f8e72f261a620cf1b55e10"
MANIFEST_SHA = "828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8"
OLD_TABLE = ROOT / "doc/results/market_relative_changes_v1/table_rows.json"
OLD_TABLE_SHA = "61c5de99f0527fce8252cf579fefba0e6a9de22add193eb068d3a93aefb77890"
PARENT = ROOT / "output/runs/market-capacity-study/run-base-gryiAV"
OLD36 = ROOT / "output/runs/market-separate-changes-native36/run-4SauBf"
SHARED = ROOT / "output/runs/market-capacity-study/shared-MKT-HIST-2-v1"
REPS = [2701, 2802, 2903]
SEEDS = [13088411117524848790, 10018485049408248344, 9083751155097190455]
TASKS = ["past_volatility", "past_trend", "past_activity", "past_relation"]
TOL = 1e-12
FLOORS = {
    "past_volatility": (.6781069298378846, .3926141918082198),
    "past_trend": (.6549112553681312, .1342955201053261),
    "past_activity": (.7514296819362523, .5631529010630465),
    "past_relation": (.3406853885730756, .3495834881128966),
}
LEGEND = "MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent128 past-only targets · complexity 4/5"
TEXT_SUFFIXES = {".json", ".csv", ".log", ".conf", ".md", ".sha256", ".txt"}
BANK_FAMILY = "raw-symmetric-relative-f64-rounded32-global-quarter-v1"
RAW_CONTRACT = "genuine-raw-f64;parent-cpu-raw-f32-before-scaler-v1"
INCREMENT_RECIPE = "cuda-f64-maxabs-scaled-symmetric-relative;u64-to-f32-to-f64;visible-adjacent-zero-safe-v1"
MOMENT_LAYOUT = "canonical-channel-feature-moment;global-mean,rms;quarter0-mean,rms;quarter1-mean,rms;quarter2-mean,rms;quarter3-mean,rms;quarter-normalized-by-global-rms;visible-adjacent-min2-v1"
NORMALIZATION_RECIPE = "valid-f32-bank-cpu-f64-count-sum-sumsq-population-center-std-min2-v1"
RAW_LOADER = ROOT / "code/training/include/cuwacunu/training/market_raw_f64_data.h"
REUSED = ROOT / "output/runs/market-relative-changes/run-vaPC4X"
REUSED_CARD = "f4378d4a76185fc17b6c0adbadab5af27accd772ed66e5a8bc628ab7848f94a2"
REUSED_READOUT_SHA = "0f9199d88e7d3a90039d8a555873fb74399b8b79f6505798a58c3fcc5bae4156"
NORMALIZER_SHA = "c3e4f9e850ebbb355ae58e8b434173e4db0d1ff98055c844af4fe16ba7879fbc"
NORMALIZER_JSON_SHA = "782128503036956beb1ec87e39ceb3c502395f00f1d0e8a6138e9d8e52d933ee"
REUSED_AUDIT_SHA = "b8278365f60f95eb1a611b8ed34fe4a144a77be59de686e1891eda9e30b25d97"
INITIAL_FEATURE_SHA = {"train":"953c9967515f2de6c709e1651fab7a0b464410f1f62bd25e06c2a534d72cc593",
                       "validation":"a774470823dc2d0aad0e5f622fee4acd52383f05aabf11e193071d88f5b8aea0"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def exclusive(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def dump(path, value):
    exclusive(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def asset(path):
    return dict(source=relative(path), sha256=sha(path), bytes=path.stat().st_size)


def copy_exclusive(source, destination, expected=None):
    expected = expected or sha(source)
    require(sha(source) == expected, "source changed before capture: " + str(source))
    exclusive(destination, source.read_bytes())
    require(sha(destination) == expected, "captured hash mismatch: " + str(destination))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def table(lines, headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines.extend(["| " + " | ".join(map(cell, headers)) + " |",
                  "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |"])
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    lines.append("")


def validate_readouts(report, run, *, reused_initial=False, new_run=None):
    require(report["kind"] == "fixed_past_window_regression_v1" and report["dataset_id"] == "MKT-HIST-2",
            "fixed evaluator dataset/kind mismatch")
    require(report["card_sha256"] == (REUSED_CARD if reused_initial else CARD) and report["input_manifest_sha256"] == MANIFEST_SHA,
            "readout card/input binding mismatch")
    require((report["training_rows"], report["validation_rows"]) == (112, 32), "readout split changed")
    heads = {"ridge_penalty": 1, "neural_hidden": 16, "neural_learning_rate": .01,
             "neural_updates": 100, "outputs": 4, "loss": "masked TRAIN-target-standardized MSE",
             "device": "CPU (frozen readouts only)"}
    require(all(report["heads"][key] == value for key, value in heads.items()), "fixed head recipe changed")
    parent_targets = load(PARENT / "readouts/report.json")["methods"][0]["repetitions"][0]["targets"]
    names = ["initialized_v28", "trained_v28"] if reused_initial else ["trained_v29"]
    require([method["name"] for method in report["methods"]] == names,
            "Only final V29 heads may be newly fitted; initial scores come from pinned V28")
    coverage = True
    rows = []
    for method in (report["methods"][:1] if reused_initial else report["methods"]):
        require((method["size"], method["ridge_parameters"], method["neural_parameters"], method["ridge_fits"],
                 method["ridge_task_fits"], method["neural_fits"], method["training_feature_rows"],
                 method["feature_training_supported"]) == (36, 148, 660, 1, 4, 3, 112, True),
                "new native36 head size/fit count/support mismatch")
        for role, suffix in [("training", "train"), ("validation", "validation")]:
            require(Path(method[role + "_path"]).resolve() == run / "features" / f"{method['name']}-{suffix}.pt",
                    "readout does not bind the declared new bank")
        require([rep["repetition"] for rep in method["repetitions"]] == REPS and
                [rep["actual_seed"] for rep in method["repetitions"]] == SEEDS, "width-derived repetitions/seeds changed")
        for index, repetition in enumerate(method["repetitions"]):
            require(repetition["neural_updates_completed"] == 100 and
                    repetition["ridge_fit_reused"] == (index != 0), "readout update/fit reuse changed")
            require([target["name"] for target in repetition["targets"]] == TASKS, "declared target order changed")
            for target, old in zip(repetition["targets"], parent_targets):
                require(target["train_target_eligible"] == 112 and target["train_target_supported"] and
                        target["method_training_supported"], "TRAIN target support changed")
                require(all(target[key] == old[key] for key in
                            ("name", "unit", "train_target_mean", "train_target_scale")), "target normalization identity changed")
                for head in ("ridge", "neural"):
                    metrics = target[head]
                    require(metrics["total"] == metrics["target_eligible"] == 32 and 0 < metrics["valid"] <= 32,
                            "unscorable/invalid validation population")
                    require(metrics["coverage"] == metrics["valid"] / 32, "coverage arithmetic mismatch")
                    require(all(finite(metrics[key]) for key in ("r2", "mae", "rmse")), "nonfinite quality metric")
                    coverage &= metrics["valid"] == 32 and metrics["coverage"] == 1
        for index, task in enumerate(TASKS):
            targets = [rep["targets"][index] for rep in method["repetitions"]]
            ridge = targets[0]["ridge"]
            require(all(target["ridge"] == ridge for target in targets), "ridge values changed across repetitions")
            row = dict(dataset_id="MKT-HIST-2", task=task, recipe_id="market-parent-modulated-decoder-v1",
                       designed_complexity_level=4, complexity_scale_max=5, evidence_stage="known-source development",
                       view="intact chronological VALIDATION", method=TAG + (".init" if reused_initial else ""),
                       source_method=method["name"], reused=reused_initial, size=36, linear_r2=ridge["r2"],
                       linear_mae=ridge["mae"], linear_rmse=ridge["rmse"], valid=ridge["valid"], total=32,
                       coverage=min(target[head]["coverage"] for target in targets for head in ("ridge", "neural")),
                       unit=targets[0]["unit"], report_path=relative(run / "readouts/report.json"),
                       report_sha256=sha(run / "readouts/report.json"), repetitions=[])
            row.update(new_head_measurement=not reused_initial,
                       representation_method="initialized_v29" if reused_initial else "trained_v29",
                       scoring_provenance=("reused V28 initial scoring after exact new full36/row/support parity"
                                           if reused_initial else "new final saved-bank fixed-head fits"))
            if reused_initial:
                require(new_run is not None, "New representation provenance required for reused initial scoring")
                row.update(scoring_source_model="RPB-v28.market-01.init",
                           initial_scoring_reuse=asset(new_run / "initial-scoring-reuse.json"),
                           newly_extracted_feature_assets=[asset(new_run / "features" / f"initialized_v29-{role}.pt")
                                                          for role in ("train", "validation")])
            for metric in ("r2", "mae", "rmse"):
                row["neural_" + metric + "_each"] = [target["neural"][metric] for target in targets]
                row["neural_mean_" + metric] = statistics.mean(row["neural_" + metric + "_each"])
            row["repetitions"] = [dict(repetition=rep["repetition"], actual_seed=rep["actual_seed"],
                                       neural_updates_completed=rep["neural_updates_completed"], target=rep["targets"][index])
                                  for rep in method["repetitions"]]
            rows.append(row)
    return rows, coverage


def decisions(lookup, full_coverage):
    original, strict = [], []
    trend = lookup[TAG, "past_trend"]
    for key in ("linear_r2", "neural_mean_r2"):
        value = trend[key]
        original.append(dict(criterion="positive_" + key + "_trend", candidate=value, floor=0.,
                             difference=value, comparison=">", passed=value > 0.))
    for task in ("past_volatility", "past_activity", "past_relation"):
        for key in ("linear_r2", "neural_mean_r2"):
            baseline, value = lookup["RPB-v22.market-02", task][key], lookup[TAG, task][key]
            original.append(dict(criterion=task + "_" + key + "_v22_retention", baseline=baseline,
                                 candidate=value, floor=baseline - .05, difference=value - baseline,
                                 difference_from_floor=value - (baseline - .05), maximum_allowed_loss=.05,
                                 comparison=">=", passed=value + TOL >= baseline - .05))
    reference, value = lookup["RPB-v23.market-01", "past_trend"]["linear_r2"], trend["linear_r2"]
    require(reference == .3871309972495327, "saved v23 trend reference changed")
    original.append(dict(criterion="linear_trend_exceeds_saved_v23", baseline=reference, floor=reference,
                         candidate=value, difference=value - reference, comparison=">", passed=value + TOL > reference))
    for task in TASKS:
        for index, key in enumerate(("linear_r2", "neural_mean_r2")):
            floor, value = FLOORS[task][index], lookup[TAG, task][key]
            if task != "past_volatility" or index != 0:
                require(lookup[NATIVE36, task][key] == floor, "frozen native36 floor differs from declaration")
            strict.append(dict(criterion=task + "_" + key + "_native36_guard", candidate=value, floor=floor,
                               difference=value - floor, comparison=">=", passed=value + TOL >= floor))
    individual = [dict(criterion=f"final_neural_trend_rep_{rep}_positive", repetition=rep,
                       actual_seed=seed, candidate=value, floor=0., difference=value, comparison=">", passed=value > 0.)
                  for rep, seed, value in zip(REPS, SEEDS, trend["neural_r2_each"])]
    coverage = dict(criterion="every_new_task_head_view_has_32_of_32_coverage", passed=full_coverage,
                    required_valid=32, required_total=32, required_coverage=1.)
    return original, strict, individual, coverage


def validate_saved_run(run, summary, initial):
    expected = dict(kind="market_parent_modulated_decoder_v1", protocol="market-parent-modulated-decoder-v1",
                    dataset="MKT-HIST-2", instance=TAG, H=128, W=64, D=36,
                    level_width=32, change_width=4, parameters=298289, optimizer_updated_parameters=15768,
                    protected_parameters=282521, completed_updates=2048, attempted_batches=2048,
                    window_presentations=16384, parent_updates=2048, probe_train_rows=112, probe_validation_rows=32,
                    CPU_encoder_calls=0, cuda_learning_verified=True, protected_level_exact=True, protected_optimizer_states=0,
                    initial_prefix_matches_saved_trained_v22=True, final_prefix_matches_saved_trained_v22=True,
                    initial_weights_match_saved_v24_init=True, initial_all36_equality_to_v24_not_expected=True,
                    gain_parameters=864, first_gain_gradient_zero=True, first_gain_weights_zero=True,
                    initial_all36_equal_to_v28=True, gain_weights_finite=True,
                    new_bank_normalizer_fits=0, new_initial_head_fits=0,
                    new_final_ridge_pipelines=1, new_final_ridge_task_fits=4, new_final_neural_fits=3,
                    card_sha256=CARD, parent_inference_sha256=PARENT_SHA, shared_ledger_sha256=SHARED_SHA,
                    input_manifest_sha256=MANIFEST_SHA, scaler_identity="rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969",
                    bank_normalizer_TRAIN_anchors=4096, bank_normalizer_validation_anchors=0, bank_normalizer_encoder_calls=0,
                    raw_source_dtype="float64",parent_rounding_dtype="float32",bank_family=BANK_FAMILY,raw_contract=RAW_CONTRACT,
                    increment_recipe=INCREMENT_RECIPE,moment_layout=MOMENT_LAYOUT,historical_parent_preprocessing_exact=True,
                    training_seconds_includes_raw_loader_audit=True)
    require(all(summary[key] == value for key, value in expected.items()), "Declared V29 run contract mismatch")
    require(summary["bank_normalizer_fit_seconds_including_io"] == 0, "V29 must reuse calibration without fitting")
    require(all(finite(summary[key]) for key in ("final_gain_weight_min", "final_gain_weight_max", "final_gain_weight_l2")) and
            summary["final_gain_weight_min"] <= summary["final_gain_weight_max"] and summary["final_gain_weight_l2"] > 0,
            "New gain tensor must be finite and actually learned")
    for filename, key in [("study.conf", "config_sha256"), ("encoder.conf", "settings_sha256"),
                          ("prospective-card.md", "card_sha256"), ("source-closure.sha256", "source_closure_sha256")]:
        require(sha(run / filename) == summary[key], "Saved binding changed: " + filename)
    require(sha(Path("/opt/cuwacunu/build/market-parent-modulated/market_parent_modulated_decoder")) == summary["binary_sha256"],
            "New producer binary changed")
    require(sha(PARENT / "checkpoints/inference-2048.pt") == PARENT_SHA and
            sha(OLD36 / "checkpoints/inference-0.pt") == INIT_SHA, "Exact preserved parent/initial checkpoint changed")
    names = ["inference-0.pt", "inference-2048.pt", "training-512.pt", "training-1024.pt", "training-2048.pt"]
    require(sorted(path.name for path in (run / "checkpoints").iterdir()) == sorted(names), "Five immutable checkpoints required")
    normalizer = load(run / "bank-standardizer.json")
    require(sha(run / "bank-standardizer.pt") == sha(REUSED / "bank-standardizer.pt") == NORMALIZER_SHA and
            sha(run / "bank-standardizer.json") == sha(REUSED / "bank-standardizer.json") == NORMALIZER_JSON_SHA,
            "Exact unchanged V28 normalizer and original fit receipt required")
    normalizer_contract = dict(kind="market_relative_changes_TRAIN_fit_v1", batch_size=8, bank_width=270,
                               channel_count=3, history_length=128, input_width=9, minimum_count=2, floor=1e-6,
                               bank_family=BANK_FAMILY,raw_contract=RAW_CONTRACT,increment_recipe=INCREMENT_RECIPE,
                               layout_identity=MOMENT_LAYOUT,recipe_identity=NORMALIZATION_RECIPE,
                               input_geometry=[3,128,9,0,1,2],sampling_interval=60000)
    require(all(normalizer[key] == value for key, value in normalizer_contract.items()),
            "Frozen normalizer geometry/support/floor changed")
    require(normalizer["TRAIN_anchors"] == 4096 and normalizer["validation_anchors"] == normalizer["encoder_calls"] == 0,
            "TRAIN-only calibration population changed")
    require(normalizer["identity"] == summary["bank_normalizer_identity"] and
            normalizer["source_identity"] == summary["bank_normalizer_source_identity"] and
            normalizer["input_scaler_identity"] == summary["bank_normalizer_input_scaler_identity"] == summary["scaler_identity"] and
            normalizer["archive_sha256"] == summary["bank_normalizer_sha256"], "Normalizer source/input-scaler identity changed")
    require(normalizer["identity"].startswith("rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-") and
            normalizer["source_identity"].startswith("RPB-v28;raw-symmetric-relative-f64-v1;"),
            "Distinct relative-bank family/content/source identity required")
    require(all(len(normalizer[key]) == 270 for key in ("counts", "sum", "sumsq", "mean", "scale", "support")), "Normalizer width changed")
    require(sha(run / "bank-standardizer.pt") == normalizer["archive_sha256"], "Exact local bank normalizer archive changed")
    for count, summed, square, mean, scale, supported in zip(*(normalizer[key] for key in
                                                            ("counts", "sum", "sumsq", "mean", "scale", "support"))):
        require(type(count) is int and 0 <= count <= 4096 and all(finite(x) for x in (summed, square, mean, scale)) and
                square >= 0 and scale >= 1e-6 and bool(supported) == (count >= 2), "Invalid normalizer sufficient statistics")
        expected_mean = summed / count if count >= 2 else 0.
        variance = max(square / count - expected_mean * expected_mean, 0.) if count >= 2 else 0.
        expected_scale = max(math.sqrt(variance), 1e-6) if count >= 2 else 1.
        require(math.isclose(mean, expected_mean, rel_tol=1e-12, abs_tol=1e-15) and
                math.isclose(scale, expected_scale, rel_tol=1e-12, abs_tol=1e-15), "Declared conditioning arithmetic mismatch")
    for filename in ("initialization-parity.json", "final-prefix-parity.json"):
        parity = load(run / filename)
        require(parity["saved_v22_train_and_validation_prefix32_equal"] and
                parity["old_encoder_replay_calls"] == parity["old_head_refits"] == 0, "Exact parent-prefix parity receipt missing")
    require(load(run / "initialization-parity.json")["saved_v28_initial_train_and_validation_native36_equal"] is True,
            "Full36 initialized TRAIN/VALIDATION parity must precede initial score reuse")
    parameters = load(run / "initial-parameter-parity.json")
    require(parameters == dict(saved_v24_initial_encoder_parameters_and_buffers_exact=True, zero_wave_decoder=True,
                               zero_waveform_gain=True, full_initial36_equality_to_v24_not_expected=True),
            "Fresh saved V24 initialization and literal-zero gain receipt missing")
    validate_reuse_contract(run)
    parent = load(PARENT / "summary.json")
    for role, count in (("train", 112), ("validation", 32)):
        for key in ("mae_train_scaled", "row_mae", "row_target_cells", "row_eligible_channels", "eligible_rows",
                    "total_rows", "target_cells", "target_history"):
            require(initial[role][key] == parent["trained_query"][role][key], "Zero residual initial reconstruction differs from parent")
        require(summary["trained_query"][role]["eligible_rows"] == summary["trained_query"][role]["total_rows"] == count,
                "Final fixed-query coverage changed")
    with (run / "training.csv").open(newline="") as handle:
        trace = list(csv.DictReader(handle))
    with (OLD36 / "training.csv").open(newline="") as handle:
        old_trace = list(csv.DictReader(handle))
    require(len(trace) == len(old_trace) == 2048, "Matched trace lengths changed")
    fields = ("attempt", "update", "eligible_examples", "target_cells", "row_seed", "mask_seed", "anchors")
    require(all(all(row[key] == old[key] for key in fields) for row, old in zip(trace, old_trace)), "Matched row/mask/anchor streams changed")
    counts = [("projection_gradient_nonzero", "nonzero_change_gradient_updates"),
              ("wave_decoder_gradient_nonzero", "nonzero_decoder_gradient_updates"),
              ("gain_gradient_nonzero", "nonzero_gain_gradient_updates"),
              ("projection_weight_changed", "projection_weight_updates"),
              ("wave_decoder_weight_changed", "decoder_weight_updates"),
              ("gain_weight_changed", "gain_weight_updates")]
    for field, key in counts:
        count = sum(int(row[field]) for row in trace)
        require(0 < count <= 2048 and count == summary[key], "CUDA gradient/weight count mismatch")
    require(trace[0]["projection_gradient_nonzero"] == "0" and trace[0]["wave_decoder_gradient_nonzero"] == "1" and
            trace[0]["wave_decoder_weight_changed"] == "1" and trace[0]["gain_gradient_nonzero"] == "0" and
            trace[0]["gain_weight_changed"] == "0" and all(trace[1][key] == "1" for key, _ in counts),
            "First decoder then projection/gain CUDA learning proof missing")
    for row in trace:
        lo, hi, mean = (float(row[key]) for key in ("gain_min", "gain_max", "gain_mean"))
        require(all(finite(x) for x in (lo, hi, mean)) and lo <= hi and lo - 1e-5 <= mean <= hi + 1e-5,
                "Nonfinite or inconsistent descriptive gain trace")
    require(all(float(trace[0][key]) == 1 for key in ("gain_min", "gain_max", "gain_mean")),
            "The first actual gain must be unity")
    losses = [float(row["loss"]) for row in trace]
    require(all(finite(value) and value >= 0 for value in losses), "Nonfinite waveform loss")
    raw = load(run / "raw-loading-provenance.json")
    raw_contract = dict(kind="market_parent_modulated_decoder_raw_loading_provenance_v1",raw_source_dtype="float64",parent_rounding_dtype="float32",
                        loader_recipe="pinned-raw-kline-cache88-direct-double-registry;strict-minute-past-role;legacy-f32-equality-v1",
                        record_type="kline_cache_t",record_size=88,original_record_to_F64_exact=True,raw_F64_to_legacy_F32_exact=True,
                        historical_parent_preprocessing_exact=True,parent_preprocessing_checks=2,old_encoder_calls=0,cache_mutations=0,
                        difference_count_scope="observed coordinate presentations across all loader calls",
                        record_comparison_scope="observed finite coordinates; hidden storage zero")
    require(all(raw[key] == value for key,value in raw_contract.items()), "Genuine raw-loader/parent preprocessing contract changed")
    require(sha(RAW_LOADER) == raw["loader_source_sha256"] == summary["raw_loader_source_sha256"] and
            raw["genuine_F64_differs_from_F32_count"] == summary["genuine_F64_differs_from_F32_count"] > 0 and
            0 < raw["load_calls"] and raw["record_values_checked"] == raw["legacy_values_checked"] > 0 and
            raw["genuine_F64_differs_from_F32_count"] <= raw["record_values_checked"], "Raw precision/counter/source proof missing")
    require(len(raw["sources"]) == 3 and [item["channel"] for item in raw["sources"]] == [0,1,2], "Raw source channel association changed")
    for item in raw["sources"]:
        path=Path(item["path"])
        path=path if path.is_absolute() else ROOT/path
        require(path.resolve().is_relative_to(ROOT) and sha(path) == item["sha256"] and path.stat().st_size == item["bytes"] == item["records"]*88,
                "Exact original raw-cache input changed")
    require((run/"raw-relative-bank-witness.json").is_file() and (run/"raw-loader-record-witness.json").is_file(),
            "Local raw increment/record witnesses missing")
    require(len(list((run / "features").glob("*-decoder-witness.json"))) == 4,
            "Local initial/final TRAIN/VALIDATION decoder witnesses missing")
    return dict(first=losses[0], last=losses[-1], first64_mean=statistics.mean(losses[:64]), last64_mean=statistics.mean(losses[-64:])), normalizer, raw


def validate_reuse_contract(run):
    reuse = load(run / "calibration-reuse.json")
    expected = dict(kind="market_parent_modulated_decoder_calibration_reuse_v1", TRAIN_anchors=4096,
                    new_bank_normalizer_fits=0, new_calibration_fits=0, new_calibration_encoder_calls=0,
                    new_calibration_encoder_updates=0, normalizer_source_path=relative(REUSED / "bank-standardizer.pt"),
                    normalizer_source_sha256=NORMALIZER_SHA, normalizer_source_json_sha256=NORMALIZER_JSON_SHA,
                    normalizer_identity="rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476",
                    original_audit_sha256=REUSED_AUDIT_SHA, reused_raw_calibration_audit_checks=1549413)
    require(all(reuse[key] == value for key, value in expected.items()), "Reused calibration provenance/counters changed")
    require(sha(REUSED / "independent-audit.json") == REUSED_AUDIT_SHA and
            load(REUSED / "independent-audit.json")["status"] == "passed", "Original raw/calibration audit changed")
    scoring = load(run / "initial-scoring-reuse.json")
    expected = dict(kind="market_parent_modulated_decoder_initial_scoring_reuse_v1",
                    source_report_path=relative(REUSED / "readouts/report.json"), source_report_sha256=REUSED_READOUT_SHA,
                    source_method="initialized_v28", new_method="initialized_v29",
                    train_feature_sha256=INITIAL_FEATURE_SHA["train"], validation_feature_sha256=INITIAL_FEATURE_SHA["validation"],
                    full_native_values_valid_row_ids_exact=True, new_initial_head_fits=0,
                    query_mask_parity=True, target_bank_parity=True, shared_ledger_sha256=SHARED_SHA,
                    new_initial_ridge_pipelines=0, new_initial_neural_fits=0)
    require(all(scoring[key] == value for key, value in expected.items()), "Initial scoring reuse/parity contract changed")
    require(sha(REUSED / "readouts/report.json") == REUSED_READOUT_SHA, "Original initial head report changed")
    for role, digest in INITIAL_FEATURE_SHA.items():
        require(sha(REUSED / "features" / f"initialized_v28-{role}.pt") == digest,
                "Original initial feature archive changed")
    return reuse, scoring


def main():
    started = time.perf_counter()
    require(Path("/.dockerenv").is_file() and ROOT == Path("/cuwacunu"), "Use existing managed cuwacunu container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/results/market_parent_modulated_decoder_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "doc/MARKET_PARENT_MODULATED_DECODER_RESULTS.md")
    parser.add_argument("--admission-log", type=Path)
    parser.add_argument("--producer-build-log", type=Path)
    args = parser.parse_args()
    run, out, report_path = args.run.resolve(), args.output_dir.resolve(), args.report.resolve()
    engineering_receipt = out / "engineering-admission.json"
    if args.admission_log:
        admission = args.admission_log.resolve()
    else:
        engineering = load(engineering_receipt)
        require(engineering["status"] == "passed", "Passed artificial engineering receipt required")
        passed = engineering["pass_log"]
        if isinstance(passed, dict):
            passed = passed.get("source", passed.get("path", passed.get("Path")))
        require(isinstance(passed, str), "Engineering admission must identify its actual passed log")
        prefix = "C:/Work/cuwacunu/cuwacunu_dev/"
        passed = passed.replace("\\", "/")
        admission = (ROOT / passed[len(prefix):] if passed.startswith(prefix) else Path(passed))
        if not admission.is_absolute():
            admission = ROOT / admission if (ROOT / admission).exists() else out / admission
        admission = admission.resolve()
    require(admission.is_relative_to(out) and admission.is_file(), "Preserved passed engineering log required")
    require(run.is_dir() and run.parent == ROOT / "output/runs/market-parent-modulated-decoder" and run.name.startswith("run-"), "New V29 run required")
    require(out.is_relative_to(ROOT / "doc/results") and report_path.is_relative_to(ROOT / "doc"), "Repository documentation outputs required")
    require(not any(path.exists() for path in (report_path, out / "completion.json", out / "table_rows.json", out / "asset_hashes.json",
                                              out / "durable_files.json", out / "candidate-record.json", out / "capture")), "Exclusive outputs already exist")
    require(sha(OLD_TABLE) == OLD_TABLE_SHA, "Frozen V28 historical table changed")
    require(sha(ROOT / "doc/MARKET_PARENT_MODULATED_DECODER_CARD.md") == CARD, "Prospective V29 card changed")
    require(sha(Path("/opt/cuwacunu/build/market-study/market_fixed_regression")) == FIXED, "Preserved fixed evaluator changed")
    require("PASS" in admission.read_text() and "RPB-v29" in admission.read_text(), "Artificial CUDA admission missing")
    verification, audit = load(run / "verification.json"), load(run / "independent-audit.json")
    require(verification["status"] == audit["status"] == "passed", "Passed wrapper verification and independent arithmetic audit required")
    require(verification["kind"] == "market_parent_modulated_decoder_verification_v1" and
            verification["protocol"] == "market-parent-modulated-decoder-v1" and verification["instance"] == TAG and
            verification["card_sha256"] == CARD and verification["independent_audit_sha256"] == sha(run / "independent-audit.json"),
            "Verification declaration/audit binding mismatch")
    require(audit["kind"] == "market_parent_modulated_decoder_independent_arithmetic_audit_v1" and
            audit["checks"] > 0 and audit["quality_gate_tolerance"] == TOL and
            all(audit[key] == 0 for key in ("tensor_archives_deserialized", "new_encoder_calls", "new_encoder_updates", "head_or_scaler_fits")),
            "Independent scalar-only arithmetic audit contract missing")
    for name, digest in audit["input_sha256"].items():
        source = (ROOT / name).resolve()
        require(source.is_relative_to(run) and sha(source) == digest, "Independently audited run input changed: " + name)
    wrapper = run.parent / "execution-preservation"
    for filename, key in (("sources-before.json", "sources_before_sha256"),
                          ("historical-before.json", "historical_before_sha256"),
                          ("historical-after.json", "historical_after_sha256")):
        require(sha(wrapper / filename) == verification[key], "Wrapper preservation binding changed: " + filename)
    require(sha(out / "preflight.json") == verification["preflight_sha256"] and
            sha(out / "admission-review.json") == verification["admission_sha256"] and
            verification["one_new_training_trajectory"] is True and verification["broad_scan_repeated"] is False,
            "Admitted single-trajectory/preflight binding missing")
    summary, initial = load(run / "summary.json"), load(run / "initial-query.json")
    loss_summary, normalizer, raw_provenance = validate_saved_run(run, summary, initial)
    build_logs = sorted(out.glob("*build*.log"))
    producer_logs = sorted(out.glob("build-[0-9][0-9].log"))
    require(producer_logs, "Producer build log required")
    final_build_log = (args.producer_build_log or producer_logs[-1]).resolve()
    require(final_build_log in producer_logs and final_build_log.stat().st_size > 0 and
            "error:" not in final_build_log.read_text(), "A preserved successful final producer build is required")
    failed_logs = [path for path in producer_logs if "error:" in path.read_text()]
    failure_manifests = sorted(out.glob("build-*-source/hashes.json")) + sorted(out.glob("producer-build-*-sources/manifest.json"))
    for failed in failed_logs:
        attempt = failed.stem.rsplit("-", 1)[-1]
        require((out / f"producer-build-{attempt}-sources/manifest.json").is_file() or
                (out / f"build-{attempt}-source/hashes.json").is_file(),
                "Preserve the exact source capsule for each failed producer compile")
    capsule_roots = sorted(path for path in out.iterdir() if path.is_dir() and
                           (path.name.endswith("-sources") or path.name.endswith("-source")))
    failure_sources = []
    for directory in capsule_roots:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                require(path.suffix != ".pt" and "witness" not in path.name.lower(), "Raw tensors cannot enter build evidence")
                failure_sources.append(asset(path))
    engineering_manifests = sorted(out.glob("engineering-*-sources.json"))
    require(engineering_manifests, "Every engineering prebuild source capsule must be retained")
    for manifest in engineering_manifests:
        capture = load(manifest)
        directory = (ROOT / capture["source_root"]).resolve()
        require(directory in capsule_roots and capture["captured_before_compilation"] is True and
                capture["captured_before_gpu_invocation"] is True and capture["file_count"] == len(capture["files"]),
                "Engineering prebuild source capsule metadata changed")
        for item in capture["files"]:
            path = (directory / item["path"]).resolve()
            require(path.is_relative_to(directory) and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                    "Failed/passed engineering source snapshot changed")
    for manifest in out.glob("producer-build-*-sources/manifest.json"):
        for item in load(manifest):
            path = (manifest.parent / item["path"]).resolve()
            require(path.is_relative_to(manifest.parent) and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                    "Producer prebuild source snapshot changed")
    readout = load(run / "readouts/report.json")
    final_rows, final_coverage = validate_readouts(readout, run)
    initial_rows, initial_coverage = validate_readouts(load(REUSED / "readouts/report.json"), REUSED,
                                                     reused_initial=True, new_run=run)
    new_rows, full_coverage = initial_rows + final_rows, initial_coverage and final_coverage
    old_table = load(OLD_TABLE)
    rows = []
    for original in old_table["rows"]:
        row = dict(original)
        row.update(reused=True, original_report_path=row.get("report_path"), original_report_sha256=row.get("report_sha256"),
                   report_path=relative(OLD_TABLE), report_sha256=OLD_TABLE_SHA)
        rows.append(row)
    require(len(rows) == 80, "All 80 historical rows must remain")
    rows.extend(new_rows)
    lookup = {(row["method"], row["task"]): row for row in rows}
    original, strict, individual, coverage = decisions(lookup, full_coverage)
    market_pass = all(item["passed"] for item in original + strict + individual + [coverage])
    timing = load(run / "readout-timing.json")
    require(timing["kind"] == "saved_feature_readout_timing_v1" and timing["elapsed_ns"] > 0 and
            timing["elapsed_ns"] == timing["finished_epoch_ns"] - timing["started_epoch_ns"], "Readout timing scope/arithmetic mismatch")
    readout_seconds = timing["elapsed_ns"] / 1e9
    sources = {}
    for line in (run / "source-closure.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        member = Path(name)
        require(not member.is_absolute() and ".." not in member.parts and name not in sources and sha(ROOT / name) == digest,
                "Compiled source closure changed or duplicated")
        sources[name] = digest
    sources[relative(Path(__file__))] = sha(Path(__file__))
    auditor = ROOT / "code/scripts/audit-market-parent-modulated-decoder.py"
    require(auditor.is_file(), "Independent auditor source missing")
    sources[relative(auditor)] = sha(auditor)
    for name in ("code/scripts/execute-market-parent-modulated-decoder.py", "code/scripts/preflight-market-parent-modulated-decoder.ps1"):
        sources[name] = sha(ROOT / name)
    sources["code/scripts/append-encoder-search-ledger-v2.py"] = sha(ROOT / "code/scripts/append-encoder-search-ledger-v2.py")
    sources[relative(RAW_LOADER)] = sha(RAW_LOADER)
    reused_assets = [asset(REUSED / name) for name in ("bank-standardizer.pt", "bank-standardizer.json",
                     "independent-audit.json", "readouts/report.json", "features/initialized_v28-train.pt",
                     "features/initialized_v28-validation.pt", "summary.json", "verification.json")]
    shared_assets = []
    require(sha(SHARED / "shared-assets.sha256") == SHARED_SHA, "Shared bank ledger changed")
    for line in (SHARED / "shared-assets.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        member = Path(name)
        require(not member.is_absolute() and ".." not in member.parts and sha(SHARED / member) == digest, "Shared asset changed")
        shared_assets.append(asset(SHARED / member))
    run_assets = [asset(path) for path in sorted(run.rglob("*")) if path.is_file()]
    wrapper_assets = [asset(path) for path in sorted(wrapper.rglob("*")) if path.is_file()]
    prior_docs = [path for path in sorted(out.rglob("*")) if path.is_file()] if out.exists() else []
    for item in run_assets:
        source = ROOT / item["source"]
        if source.suffix in TEXT_SUFFIXES and "witness" not in source.name.lower():
            copy_exclusive(source, out / "capture/run" / source.relative_to(run), item["sha256"])
    for name, digest in sources.items():
        copy_exclusive(ROOT / name, out / "capture/sources" / name, digest)
    for item in wrapper_assets:
        source = ROOT / item["source"]
        require(source.suffix in TEXT_SUFFIXES, "Unexpected execution-wrapper artifact")
        copy_exclusive(source, out / "capture/execution-preservation" / source.relative_to(wrapper), item["sha256"])
    copy_exclusive(OLD_TABLE, out / "capture/controls/v28-table_rows.json", OLD_TABLE_SHA)
    for item in reused_assets:
        source = ROOT / item["source"]
        if source.suffix == ".json" and "witness" not in source.name.lower():
            copy_exclusive(source, out / "capture/reused-v28" / source.relative_to(REUSED), item["sha256"])
    for index, source in enumerate(dict.fromkeys(prior_docs + [admission])):
        require(source.suffix != ".pt" and "witness" not in source.name.lower(), "Raw tensor/witness cannot enter docs")
        copy_exclusive(source, out / "capture/prerequisites" / f"{index:03d}-{source.name}")
    for item in failure_sources:
        source = ROOT / item["source"]
        copy_exclusive(source, out / "capture/build-attempts" / source.relative_to(out), item["sha256"])
    build_attempts = dict(kind="market_parent_modulated_decoder_build_attempt_preservation_v1",failed_producer_compiles=len(failed_logs),
                          final_producer_build_log=asset(final_build_log),final_producer_binary_sha256=summary["binary_sha256"],
                          preserved_build_logs=[asset(path) for path in build_logs],
                          failed_source_manifests=[asset(path) for path in failure_manifests],
                          engineering_source_manifests=[asset(path) for path in engineering_manifests],
                          exact_attempt_source_capsule_assets=failure_sources,
                          engineering_admission=asset(engineering_receipt),quality_started_during_failed_compile=False)
    dump(out / "build-attempts.json", build_attempts)
    training_rows = []
    for old in old_table["training_rows"]:
        row = dict(old)
        row.update(reused=True, report_path=relative(OLD_TABLE), report_sha256=OLD_TABLE_SHA)
        training_rows.append(row)
    for label, updates, query, seconds in [(TAG + ".init", 0, initial, None),
                                            (TAG, 2048, summary["trained_query"], summary["training_seconds_including_checkpoints"])]:
        training_rows.append(dict(encoder=label, updates=updates, inherited_updates=2048, reused=False,
                                  train_error=query["train"]["mae_train_scaled"], validation_error=query["validation"]["mae_train_scaled"],
                                  gpu_training_seconds=seconds, batch_size=8, device="cuda:0", timing_includes_checkpoint_IO=True))
    tables = dict(kind="market_parent_modulated_decoder_results_v1", status="complete", dataset_id="MKT-HIST-2", instance=TAG,
                  rows=rows, training_rows=training_rows, original_criteria=original, strict_native36_criteria=strict,
                  individual_neural_trend_criteria=individual, coverage_criterion=coverage,
                  original_criteria_passed=sum(item["passed"] for item in original), original_criteria_total=9,
                  strict_criteria_passed=sum(item["passed"] for item in strict), strict_criteria_total=8,
                  full_market_gate_passed=market_pass, synthetic_transfer_measured=False, primary="RPB-v22", primary_changed=False,
                  comparison_absolute_arithmetic_tolerance=TOL, D36_actual_seeds=SEEDS, D36_head_parameters=dict(linear=148, neural=660),
                  training_loss=loss_summary, parent_stream_parity=True,
                  historical_task_rows_reused=80, initial_task_rows_reused=4, new_final_task_rows=4,
                  initial_scoring_reuse=asset(run / "initial-scoring-reuse.json"), new_initial_head_fits=0)
    lines = [f"# {TAG}: parent-modulated residual decoder results", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
             f"**The complete market gate {'passes' if market_pass else 'fails'}: original {tables['original_criteria_passed']}/9 and strict native36 floors {tables['strict_criteria_passed']}/8. RPB-v22 remains primary.**", "",
             "One new CUDA trajectory on RTX A2000 8GB/cuda:0: 2048 branch updates/B8 after 2048 inherited parent updates.",
             "Same 4096 TRAIN anchors, 112 TRAIN/32 known VALIDATION probes, input scaler, targets, masks and counter streams. TRAIN Jan 1–21; known VALIDATION Jan 22–28; Jan 29–31 unused.",
             "All 80 historical task rows are reused. Four new initial-control rows reuse the exact saved V28 initialized scoring after full36 value, row-ID and validity parity; zero initial head refits. Only the new final native36 banks receive one ridge pipeline/four target fits and three neural fits. No old encoder replay or control refit.", "",
             "V29 adds a bias-free trainable gain matrix A[27,32], initialized to literal zero. For detached exact parent L, change coordinates z and decoder D, the canonical waveform residual is R=z@D.T and its gain is G=1+L@A.T, broadcast over time. The prediction is exact frozen parent reconstruction plus the semantic-channel gather of R*G. New arithmetic is CUDA F32.",
             "The genuine raw-F64 symmetric-relative bank, increment F32 rounding/F64 moments, 270-coordinate layout and frozen V28 TRAIN standardizer are unchanged. Both views share one final visible mask. Parent and waveform targets retain historical raw-F32-before-scaler preprocessing.",
             "The saved normalizer was fit once in V28 on 4096 natural TRAIN windows with zero encoder calls. V29 reuses its exact archive, original V28 source identity and statistics JSON. New calibration fits and calibration encoder calls are zero; no new normalizer is relabeled as fitted V29 state.",
             "Waveform Huber1, protected parent32, learned4/native36, input units, loss, masks and fixed heads stay unchanged. No task labels, conditional map, auxiliary objective, learned bias, clipping, gain constraint or post-encoder PCA is added.",
             "Encoder tensors start from the same saved V24 native36 initialization, never trained V28 weights. Full initial36 equals saved V28 initialization exactly; the new gain and waveform decoder start at zero, giving unity gain and exact parent reconstruction. Initial/final parent32 remain exact.", "",
             "Fixed ridge1/neural16-tanh/Adam0.01/100 steps; three repetitions 2701/2802/2903; native36 heads 148/660 parameters.",
             "Exact uint64 seeds: " + ", ".join(map(str, SEEDS)) + ". Means retain all fits, without selecting repetitions or checkpoints.",
             "These are known-source chronological representation results, not forecasting, TEST or independent confirmation. R² is not classification accuracy.", ""]
    display = ["Raw data — no encoder (recent128)", "Raw data — no encoder (full256)", "RPB-v22.market-02.init", "RPB-v22.market-02",
               "RPB-v23.market-01.init", "RPB-v23.market-01", "RPB-v23.market-01.alt-01", "RPB-v24.market-01.init", "RPB-v24.market-01",
               NATIVE36 + ".init", NATIVE36, "RPB-v25.market-01.init", "RPB-v25.market-01",
               "RPB-v26.market-01.init", "RPB-v26.market-01", "RPB-v27.market-01.init", "RPB-v27.market-01",
               "RPB-v28.market-01.init", "RPB-v28.market-01", TAG + ".init", TAG]
    for task in TASKS:
        lines.extend(["## " + task.replace("_", " ").capitalize(), "", LEGEND + " · intact known VALIDATION.", ""])
        table(lines, ["Method", "Size", "Linear head R²", "Neural head R²", "Coverage %"],
              [(model, lookup[model, task]["size"], f"{lookup[model, task]['linear_r2']:.6f}",
                f"{lookup[model, task]['neural_mean_r2']:.6f}", f"{100 * lookup[model, task]['coverage']:.0f}") for model in display])
        lines.extend([LEGEND + " · intact known VALIDATION · supplemental physical errors, unit:" + lookup[TAG, task]["unit"] + ".", ""])
        table(lines, ["Method", "Linear MAE", "Linear RMSE", "Neural mean MAE", "Neural mean RMSE"],
              [(model,) + tuple(f"{lookup[model, task][metric]:.8g}" for metric in
                                ("linear_mae", "linear_rmse", "neural_mean_mae", "neural_mean_rmse")) for model in display])
        for model in (TAG + ".init", TAG):
            lines.append(model + (" reused initialized" if model.endswith(".init") else " new final") +
                         " neural R² by 2701/2802/2903: " + ", ".join(f"{value:.17g}" for value in lookup[model, task]["neural_r2_each"]) + ".")
        lines.append("")
    lines.extend(["## Training and calibration", "", LEGEND + " · unchanged fixed hidden TRAIN/known-VALIDATION queries.", ""])
    table(lines, ["Encoder", "Updates", "Train error ↓", "Validation error ↓", "GPU training seconds"],
          [(row["encoder"], row["updates"], f"{row['train_error']:.6f}", f"{row['validation_error']:.6f}",
            "—" if row["gpu_training_seconds"] is None else f"{row['gpu_training_seconds']:.3f}") for row in training_rows])
    lines.extend(["Errors are fixed-query TRAIN-CF-scaled MAE. New branch updates and inherited parent exposure remain separate.",
                  "New calibration fits/encoder calls: 0/0; fit time: 0 seconds. The reused V28 normalizer originally cost 2.929513154 seconds including input I/O; that historical cost is not charged as new fitting. Archive reuse/loading is included in orchestration time.",
                  "GPU training includes checkpoint/witness/loader-parity I/O and excludes initial/final captures and final head fitting.",
                  f"Fixed saved-feature fitting/scoring/artifact I/O: {readout_seconds:.6f}s; pure head-fit compute is unavailable. Capture timings remain in initial-query/summary JSON.",
                  f"Initialized/final feature extraction and archive I/O: {sum(initial[role]['feature_seconds_including_io'] for role in ('train', 'validation')):.6f}s/{sum(summary['trained_query'][role]['feature_seconds_including_io'] for role in ('train', 'validation')):.6f}s; fixed-query capture and I/O: {sum(initial[role]['query_seconds_including_io'] for role in ('train', 'validation')):.6f}s/{sum(summary['trained_query'][role]['query_seconds_including_io'] for role in ('train', 'validation')):.6f}s.",
                  f"Independent saved-scalar audit: {audit['elapsed_seconds']:.6f}s. Wrapper duration {verification['total_wrapper_seconds_including_build_io_and_head_fitting']:.6f}s also includes preservation hashing/build/I/O/head fitting and is not GPU training time.",
                  "Every loaded raw F64 batch is checked against the legacy F32 loader, with zero encoder calls for that comparison. Parent preprocessing also performs the historical CPU rounding/scaler path before CUDA. Calibration, training and feature/query durations therefore include this loader parity/preprocessing overhead; they are not a fair inference-speed comparison with older runs.",
                  f"All producer/engineering build logs and any failed source capsules are preserved. Failed producer compiles: {len(failed_logs)}; successful final log: {relative(final_build_log)}. Earlier V28 compile failure/source and all old engineering evidence remain intact.",
                  f"Waveform loss first/last:{loss_summary['first']:.8f}/{loss_summary['last']:.8f}; first64/last64 means:{loss_summary['first64_mean']:.8f}/{loss_summary['last64_mean']:.8f}.",
                  "Different training minibatches are descriptive; fixed-query errors retain the matched scoring population.",
                  "Total/active/protected parameters: 298289/15768/282521. The new gain adds 864 trainable parameters; native width stays 36. Frozen normalizer metadata is not trainable.",
                  f"Projection/decoder/gain nonzero-gradient counts: {summary['nonzero_change_gradient_updates']}/{summary['nonzero_decoder_gradient_updates']}/{summary['nonzero_gain_gradient_updates']}; weight-change counts: {summary['projection_weight_updates']}/{summary['decoder_weight_updates']}/{summary['gain_weight_updates']}.",
                  "The first decoder loss gradient is nonzero while projection/gain loss gradients are zero. Initially nonzero projection weights may move through AdamW decay; gain remains literal zero on that step. After the decoder opens, projection and gain gradients/updates are observed. Parent gradients/moments/decay are absent.", "",
                  "## Every predeclared condition", "", "Only absolute1e−12 arithmetic tolerance applies. Positivity checks require positive saved values.", ""])
    for title, items in [("Original nine", original), ("Strict native36 protection", strict), ("All three neural trend fits", individual), ("Coverage", [coverage])]:
        lines.extend(["### " + title, ""])
        for item in items:
            detail = ""
            if "candidate" in item:
                detail = f"; candidate {item['candidate']:.17g}; floor {item['floor']:.17g}; difference {item.get('difference_from_floor', item['difference']):+.17g}"
            lines.append(f"- {item['criterion']}: **{'PASS' if item['passed'] else 'FAIL'}**{detail}.")
        lines.append("")
    if market_pass:
        direction = "The full market gate passes. Next test this exact frozen checkpoint on the original compatible TEMPO-5 cohorts, both unchanged scaler policies and fixed heads. Keep its new branch normalizer frozen. Transfer is not measured here; fresh chronology follows a useful balanced result."
    else:
        failed = [item["criterion"] for item in original + strict + individual + [coverage] if not item["passed"]]
        direction = "The exact parent-modulated decoder mechanism stops. Failed conditions: " + ", ".join(failed) + ". Preserve all outcomes without gain, normalizer, head, rate, seed, width or budget rescue; no synthetic transfer to rescue this market failure."
    lines.extend(["## Preservation and next direction", "", direction,
                  "Keep native36 as the recorded balanced reference until a separately verified selection changes it. V25, v26 and all other per-task leaders remain visible.",
                  "The unchanged bank is invariant to positive multiplicative units, but the full encoder retains levels. The new generic decoder can combine detached parent context with changes; unbounded gains can cross zero, amplify gradients or overfit. Reconstruction changes establish no causal explanation of task generalization.",
                  f"Independent saved-arithmetic audit passed {audit['checks']} checks. Wrapper verification and source/input protection passed; the earlier full preservation scan is bound separately, not claimed repeated.",
                  "Local checkpoints, native features and optimizer/calibration archives remain ignored. Exact source/text captures and all full-precision physical errors/repetitions are retained beside this report.",
                  "The generated candidate-record.json is append-ready verified metadata. This reporter does not edit the live search ledger, primary selection, Git history or remotes.", ""])
    dump(out / "table_rows.json", tables)
    exclusive(report_path, ("\n".join(lines).rstrip() + "\n").encode())
    inventory = dict(kind="market_parent_modulated_decoder_assets_v1", run=relative(run), run_assets=run_assets,
                     wrapper_assets=wrapper_assets,
                     captured_sources=[asset(ROOT / name) for name in sources], shared_assets=shared_assets,
                     reused_table=asset(OLD_TABLE), admission=asset(admission), wrapper_verification=asset(run / "verification.json"),
                     independent_audit=asset(run / "independent-audit.json"), reused_v28_assets=reused_assets)
    dump(out / "asset_hashes.json", inventory)
    candidate = dict(record_id="2026-10-15-verified/" + TAG, model=TAG, protocol="market-parent-modulated-decoder-v1",
                     design="Unchanged raw-F64 relative bank and saved V28 normalizer; fresh V24 initialization and zero-start parent-modulated waveform residual decoder",
                     evidence="verified new market run; no automatic selection", run=relative(run), card_sha256=CARD,
                     native_width=36, new_branch_updates=2048, inherited_parent_updates=2048, batch_size=8,
                     input_scaler_identity=summary["scaler_identity"], normalizer_sha256=summary["bank_normalizer_sha256"],
                     normalizer_identity=summary["bank_normalizer_identity"],
                     bank_family=BANK_FAMILY,raw_contract=RAW_CONTRACT,increment_recipe=INCREMENT_RECIPE,
                     raw_loading_provenance=asset(run / "raw-loading-provenance.json"),
                     build_attempts=asset(out / "build-attempts.json"),
                     encoder_parameters=dict(total=298289, optimizer_updated=15768, protected_parent=282521, gain=864),
                     decoder_modulation=dict(gain_matrix_geometry=[27,32],gain_bias=False,parent_detached=True,
                                             literal_zero_initialized=True,native_width=36,arithmetic="float32",
                                             gain_formula="1 + L.detach() @ A.T; canonical C/F gain broadcast over H",
                                             residual_formula="semantic_channel_gather((z @ D.T) * gain)"),
                     calibration_reuse=asset(run / "calibration-reuse.json"),
                     reused_normalizer_metadata=asset(REUSED / "bank-standardizer.json"),
                     reused_normalizer_archive=asset(REUSED / "bank-standardizer.pt"),
                     reused_raw_calibration_audit=asset(REUSED / "independent-audit.json"),
                     initial_scoring_reuse=asset(run / "initial-scoring-reuse.json"),
                     initial_scoring_source_report=asset(REUSED / "readouts/report.json"),
                     initial_scoring_source_features=[dict(role=role,**asset(REUSED / "features" / f"initialized_v28-{role}.pt"))
                                                      for role in ("train","validation")],
                     new_initial_head_fits=0,initial_task_rows_reused=4,new_final_task_rows=4,
                     normalizer_source_identity=summary["bank_normalizer_source_identity"], final_task_rows=[row for row in rows if row["method"] == TAG],
                     original_conditions=original, strict_conditions=strict, individual_trend_conditions=individual, coverage_condition=coverage,
                     full_market_gate_passed=market_pass, synthetic_transfer_measured=False, primary_changed=False,
                     checkpoints=[item for item in run_assets if "/checkpoints/" in item["source"]],
                     report_sha256=sha(report_path), table_rows_sha256=sha(out / "table_rows.json"),
                     card_asset=asset(ROOT / "doc/MARKET_PARENT_MODULATED_DECODER_CARD.md"), report_asset=asset(report_path), reused_table=asset(OLD_TABLE),
                     head_recipe=dict(ridge_penalty=1,neural_hidden=16,neural_activation="tanh",neural_optimizer="Adam",neural_learning_rate=.01,
                                      neural_updates=100,outputs=4,repetitions=REPS,uint64_actual_seeds_decimal=list(map(str,SEEDS)),
                                      ridge_parameters=148,neural_parameters=660,feature_and_target_normalizers="TRAIN-only; original fixed evaluator",post_encoder_PCA=False),
                     policy="Append this distinct record only after controlling-task verification; preserve every old entry and live leader unless a verified win changes it.")
    dump(out / "candidate-record.json", candidate)
    completion = dict(kind="market_parent_modulated_decoder_completion_v1", protocol="market-parent-modulated-decoder-v1",date="2026-10-15", status="complete", instance=TAG,
                      run=relative(run), dataset_id="MKT-HIST-2", designed_complexity_level=4, encoder_trajectories=1,
                      new_branch_updates=2048, inherited_parent_updates=2048, batch_size=8, attempted_batches=summary["attempted_batches"],
                      old_encoder_quality_replays=0, old_head_refits=0, CPU_encoder_calls=0, neural_fits=3, ridge_pipeline_fits=1, ridge_target_fits=4,
                      new_initial_head_fits=0,initial_task_rows_reused=4,new_final_task_rows=4,historical_task_rows_reused=80,
                      initial_scoring_reuse_sha256=sha(run / "initial-scoring-reuse.json"),
                      initial_scoring_source_report_sha256=REUSED_READOUT_SHA,
                      initial_full36_values_valid_row_ids_exact=True,
                      gain_parameters=864,parameters=298289,optimizer_updated_parameters=15768,protected_parameters=282521,
                      original_criteria_passed=tables["original_criteria_passed"], original_criteria_total=9,
                      strict_criteria_passed=tables["strict_criteria_passed"], strict_criteria_total=8,
                      all_three_final_neural_trend_positive=all(item["passed"] for item in individual), full_coverage=full_coverage,
                      full_market_gate_passed=market_pass, stopped_without_rescue=not market_pass, synthetic_transfer_measured=False,
                      primary_changed=False, card_sha256=CARD, source_closure_sha256=summary["source_closure_sha256"], binary_sha256=summary["binary_sha256"],
                      fixed_evaluator_binary_sha256=FIXED, normalizer_sha256=summary["bank_normalizer_sha256"],
                      normalizer_identity=summary["bank_normalizer_identity"],
                      bank_family=BANK_FAMILY,raw_contract=RAW_CONTRACT,increment_recipe=INCREMENT_RECIPE,
                      raw_loader_source_sha256=sha(RAW_LOADER),genuine_F64_differs_from_F32_count=raw_provenance["genuine_F64_differs_from_F32_count"],
                      historical_parent_preprocessing_exact=True,new_bank_normalizer_fits=0,new_calibration_parent_CUDA_encoder_calls=0,
                      calibration_reuse_sha256=sha(run / "calibration-reuse.json"),
                      reused_calibration_original_seconds=2.929513154,reused_raw_calibration_audit_sha256=REUSED_AUDIT_SHA,
                      training_seconds_includes_raw_loader_audit=True,
                      build_attempts_sha256=sha(out / "build-attempts.json"),failed_producer_compiles_preserved=len(failed_logs),
                      final_producer_build_log_sha256=sha(final_build_log),
                      report_sha256=sha(report_path), table_rows_sha256=sha(out / "table_rows.json"), asset_hashes_sha256=sha(out / "asset_hashes.json"),
                      candidate_record_sha256=sha(out / "candidate-record.json"), reporter_sha256=sha(Path(__file__)),
                      independent_arithmetic_checks=audit["checks"], independent_audit_sha256=sha(run / "independent-audit.json"),
                      wrapper_verification_sha256=sha(run / "verification.json"),
                      calibration_seconds=summary["bank_normalizer_fit_seconds_including_io"], gpu_training_seconds_including_checkpoint_IO=summary["training_seconds_including_checkpoints"],
                      saved_feature_fitting_scoring_and_IO_seconds=readout_seconds, pure_head_fit_seconds=None,
                      reporting_and_hash_capture_seconds=time.perf_counter() - started, source_assets=run_assets,
                      captured_sources=inventory["captured_sources"], checkpoint_assets=candidate["checkpoints"])
    dump(out / "completion.json", completion)
    members = [dict(path=path.relative_to(out).as_posix(), sha256=sha(path), bytes=path.stat().st_size)
               for path in sorted(out.rglob("*")) if path.is_file()]
    dump(out / "durable_files.json", dict(kind="market_parent_modulated_decoder_durable_v1", report=asset(report_path), files=members,
                                         policy="Frozen outputs; later additive receipts are recorded separately."))
    require(not any(path.suffix == ".pt" or "witness" in path.name.lower() for path in out.rglob("*") if path.is_file()), "Tensor/witness exported into docs")
    print(json.dumps(dict(status="complete", run=relative(run), task_rows=len(rows), original_passed=tables["original_criteria_passed"],
                          strict_passed=tables["strict_criteria_passed"], full_market_gate_passed=market_pass, report=relative(report_path))))


if __name__ == "__main__":
    main()


