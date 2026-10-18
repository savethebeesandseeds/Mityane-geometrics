#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Report immutable saved v27 parent-conditional evidence using JSON/CSV and hashes only.

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
TAG = "RPB-v27.market-01"
NATIVE36 = "RPB-v24.market-01.alt-01"
CARD = "6e24e48bfdd05148f99446cf1421a713a7736d522f6c00159ba156e43b39c29b"
FIXED = "5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14"
PARENT_SHA = "e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c"
INIT_SHA = "2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108"
SHARED_SHA = "dbc208fb7c858909ff6f04e70b939adaab1c3e8c27f8e72f261a620cf1b55e10"
MANIFEST_SHA = "828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8"
OLD_TABLE = ROOT / "doc/results/market_conditioned_changes_v1/table_rows.json"
OLD_TABLE_SHA = "56da31fe5256dca8c59ca3816bb9ff32e2a07a9075e353bc2d830cece11d678f"
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
NORMALIZER_SHA = "15e11373b14fe438fe920ccf13a72b86aa4719ea1c17dee8422249db1c185535"
NORMALIZER_ID = "rpb-bank-standardizer-v1-fnv1a64-59b99cdeeeadc038"
CALIBRATION_BANK_SHA = "e375dbbfc29cbb6a0d70def2de42d83a68bc5d376a8cef6daf88edbfd7d73c08"
V26 = ROOT / "output/runs/market-conditioned-changes/run-RjTTZZ"


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


def validate_readouts(report, run):
    require(report["kind"] == "fixed_past_window_regression_v1" and report["dataset_id"] == "MKT-HIST-2",
            "fixed evaluator dataset/kind mismatch")
    require(report["card_sha256"] == CARD and report["input_manifest_sha256"] == MANIFEST_SHA,
            "readout card/input binding mismatch")
    require((report["training_rows"], report["validation_rows"]) == (112, 32), "readout split changed")
    heads = {"ridge_penalty": 1, "neural_hidden": 16, "neural_learning_rate": .01,
             "neural_updates": 100, "outputs": 4, "loss": "masked TRAIN-target-standardized MSE",
             "device": "CPU (frozen readouts only)"}
    require(all(report["heads"][key] == value for key, value in heads.items()), "fixed head recipe changed")
    parent_targets = load(PARENT / "readouts/report.json")["methods"][0]["repetitions"][0]["targets"]
    require([method["name"] for method in report["methods"]] == ["initialized_v27", "trained_v27"],
            "only the two new saved-bank methods may be fitted")
    coverage = True
    rows = []
    for method in report["methods"]:
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
            row = dict(dataset_id="MKT-HIST-2", task=task, recipe_id="market-parent-conditional-changes-v1",
                       designed_complexity_level=4, complexity_scale_max=5, evidence_stage="known-source development",
                       view="intact chronological VALIDATION", method=TAG + (".init" if method["name"].startswith("initialized") else ""),
                       source_method=method["name"], reused=False, size=36, linear_r2=ridge["r2"],
                       linear_mae=ridge["mae"], linear_rmse=ridge["rmse"], valid=ridge["valid"], total=32,
                       coverage=min(target[head]["coverage"] for target in targets for head in ("ridge", "neural")),
                       unit=targets[0]["unit"], report_path=relative(run / "readouts/report.json"),
                       report_sha256=sha(run / "readouts/report.json"), repetitions=[])
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
    expected = dict(kind="market_parent_conditional_changes_v1", protocol="market-parent-conditional-changes-v1",
                    dataset="MKT-HIST-2", instance=TAG, H=128, W=64, D=36,
                    level_width=32, change_width=4, parameters=297425, optimizer_updated_parameters=14904,
                    protected_parameters=282521, completed_updates=2048, attempted_batches=2048,
                    window_presentations=16384, parent_updates=2048, probe_train_rows=112, probe_validation_rows=32,
                    CPU_encoder_calls=0, cuda_learning_verified=True, protected_level_exact=True, protected_optimizer_states=0,
                    initial_prefix_matches_saved_trained_v22=True, final_prefix_matches_saved_trained_v22=True,
                    initial_weights_match_saved_v24_init=True, initial_all36_equality_to_v24_not_expected=True,
                    card_sha256=CARD, parent_inference_sha256=PARENT_SHA, shared_ledger_sha256=SHARED_SHA,
                    input_manifest_sha256=MANIFEST_SHA, scaler_identity="rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969",
                    bank_normalizer_TRAIN_anchors=4096, bank_normalizer_validation_anchors=0, bank_normalizer_encoder_calls=0,
                    bank_normalizer_sha256=NORMALIZER_SHA, bank_normalizer_identity=NORMALIZER_ID,
                    parent_calibration_TRAIN_anchors=4096, parent_calibration_CUDA_encoder_calls=512,
                    parent_calibration_CPU_encoder_calls=0, parent_calibration_encoder_updates=0,
                    parent_calibration_validation_anchors=0, new_bank_normalizer_fits=0,
                    bank_normalizer_fit_seconds_including_io=0, reused_calibration_bank_sha256=CALIBRATION_BANK_SHA,
                    conditional_map_coefficient_count=8910, conditional_map_coefficient_storage_bytes=71280,
                    conditional_map_frozen_tensor_storage_bytes=2571950)
    require(all(summary[key] == value for key, value in expected.items()), "Declared v27 run contract mismatch")
    for filename, key in [("study.conf", "config_sha256"), ("encoder.conf", "settings_sha256"),
                          ("prospective-card.md", "card_sha256"), ("source-closure.sha256", "source_closure_sha256")]:
        require(sha(run / filename) == summary[key], "Saved binding changed: " + filename)
    require(sha(Path("/opt/cuwacunu/build/market-parent-conditional/market_parent_conditional_changes")) == summary["binary_sha256"],
            "New producer binary changed")
    require(sha(PARENT / "checkpoints/inference-2048.pt") == PARENT_SHA and
            sha(OLD36 / "checkpoints/inference-0.pt") == INIT_SHA, "Exact preserved parent/initial checkpoint changed")
    names = ["inference-0.pt", "inference-2048.pt", "training-512.pt", "training-1024.pt", "training-2048.pt"]
    require(sorted(path.name for path in (run / "checkpoints").iterdir()) == sorted(names), "Five immutable checkpoints required")
    normalizer = load(run / "bank-standardizer.json")
    normalizer_contract = dict(kind="market_conditioned_changes_TRAIN_fit_v1", batch_size=8, bank_width=270,
                               channel_count=3, history_length=128, input_width=9, minimum_count=2, floor=1e-6)
    require(all(normalizer[key] == value for key, value in normalizer_contract.items()),
            "Frozen normalizer geometry/support/floor changed")
    require(normalizer["TRAIN_anchors"] == 4096 and normalizer["validation_anchors"] == normalizer["encoder_calls"] == 0,
            "TRAIN-only calibration population changed")
    require(normalizer["identity"] == summary["bank_normalizer_identity"] and
            normalizer["source_identity"] == summary["bank_normalizer_source_identity"] and
            normalizer["input_scaler_identity"] == summary["bank_normalizer_input_scaler_identity"] == summary["scaler_identity"] and
            normalizer["archive_sha256"] == summary["bank_normalizer_sha256"], "Normalizer source/input-scaler identity changed")
    require(all(len(normalizer[key]) == 270 for key in ("counts", "sum", "sumsq", "mean", "scale", "support")), "Normalizer width changed")
    require(sha(run / "bank-standardizer.pt") == normalizer["archive_sha256"], "Exact local bank normalizer archive changed")
    require(sha(V26 / "bank-standardizer.pt") == NORMALIZER_SHA and
            sha(run / "bank-standardizer.json") == sha(V26 / "bank-standardizer.json") and
            sha(V26 / "calibration-bank.pt") == CALIBRATION_BANK_SHA,
            "Exact reused v26 bank normalizer/calibration changed")
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
    parameters = load(run / "initial-parameter-parity.json")
    require(parameters == dict(saved_v24_initial_encoder_parameters_and_buffers_exact=True, zero_wave_decoder=True,
                               full_initial36_equality_not_expected=True), "Exact saved initialization parameter receipt missing")
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
              ("projection_weight_changed", "projection_weight_updates"),
              ("wave_decoder_weight_changed", "decoder_weight_updates")]
    for field, key in counts:
        count = sum(int(row[field]) for row in trace)
        require(0 < count <= 2048 and count == summary[key], "CUDA gradient/weight count mismatch")
    require(trace[0]["projection_gradient_nonzero"] == "0" and trace[0]["wave_decoder_gradient_nonzero"] == "1" and
            trace[0]["wave_decoder_weight_changed"] == "1" and all(trace[1][key] == "1" for key, _ in counts),
            "First decoder/second projection learning proof missing")
    losses = [float(row["loss"]) for row in trace]
    require(all(finite(value) and value >= 0 for value in losses), "Nonfinite waveform loss")
    reuse = load(run / "calibration-reuse.json")
    require(reuse["kind"] == "market_parent_conditional_calibration_reuse_v1" and
            reuse["new_bank_normalizer_fits"] == reuse["new_bank_encoder_calls"] == 0 and
            reuse["normalizer_source_sha256"] == NORMALIZER_SHA and reuse["bank_source_sha256"] == CALIBRATION_BANK_SHA and
            reuse["normalizer_identity"] == NORMALIZER_ID, "Frozen v26 calibration reuse receipt changed")
    require((ROOT / reuse["normalizer_source_archive"]).resolve() == V26 / "bank-standardizer.pt" and
            (ROOT / reuse["bank_source_archive"]).resolve() == V26 / "calibration-bank.pt", "Reused archive paths changed")
    mapping = load(run / "conditional-map-summary.json")
    map_contract = dict(kind="market_parent_conditional_map_summary_v1", TRAIN_anchors=4096, validation_anchors=0,
                        bank_width=270, parent_width=32, coefficient_count=8910, penalty=1,
                        intercept_penalized=False, minimum_count=2, parent_scale_floor=1e-6,
                        normalizer_identity=NORMALIZER_ID, input_scaler_identity=summary["scaler_identity"],
                        parent_identity=PARENT_SHA, floor=1e-6, ridge=1, minimum_support=2, frozen=True,
                        input_geometry=[3,128,9,32,0,1,2], sampling_interval=60000,
                        recipe_identity="parent-f32-f64-population-normalization;conditioned-bank-f32;coordinate-masked-f64-cholesky-ridge1-slopes-unpenalized-intercept-v1",
                        bank_layout="canonical-channel-feature-moment;global-mean,rms;quarter0-mean,rms;quarter1-mean,rms;quarter2-mean,rms;quarter3-mean,rms;quarter-normalized-by-global-rms;visible-adjacent-min2-v1")
    require(all(mapping[key] == value for key, value in map_contract.items()), "Prospective conditional-map contract mismatch")
    require(mapping["identity"] == summary["conditional_map_identity"] and
            mapping["source_identity"] == summary["conditional_map_source_identity"] and
            mapping["archive_sha256"] == summary["conditional_map_sha256"] == sha(run / "conditional-map.pt") and
            2 <= mapping["supported_parent_rows"] <= 4096 and 0 < mapping["supported_bank_coordinates"] <= 270,
            "Frozen conditional-map provenance/support mismatch")
    require(finite(summary["parent_calibration_seconds_including_io"]) and summary["parent_calibration_seconds_including_io"] > 0 and
            finite(summary["conditional_map_fit_seconds_including_io"]) and summary["conditional_map_fit_seconds_including_io"] > 0,
            "Separate calibration extraction/fit timings required")
    return dict(first=losses[0], last=losses[-1], first64_mean=statistics.mean(losses[:64]), last64_mean=statistics.mean(losses[-64:])), normalizer, mapping


def main():
    started = time.perf_counter()
    require(Path("/.dockerenv").is_file() and ROOT == Path("/cuwacunu"), "Use existing managed cuwacunu container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/results/market_parent_conditional_changes_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "doc/MARKET_PARENT_CONDITIONAL_CHANGE_RESULTS.md")
    parser.add_argument("--admission-log", type=Path)
    args = parser.parse_args()
    run, out, report_path = args.run.resolve(), args.output_dir.resolve(), args.report.resolve()
    admission = (args.admission_log or out / "engineering-01.log").resolve()
    require(run.is_dir() and run.parent == ROOT / "output/runs/market-parent-conditional-changes" and run.name.startswith("run-"), "New v27 run required")
    require(out.is_relative_to(ROOT / "doc/results") and report_path.is_relative_to(ROOT / "doc"), "Repository documentation outputs required")
    require(not any(path.exists() for path in (report_path, out / "completion.json", out / "table_rows.json", out / "asset_hashes.json",
                                              out / "durable_files.json", out / "candidate-record.json", out / "capture")), "Exclusive outputs already exist")
    require(sha(OLD_TABLE) == OLD_TABLE_SHA, "Frozen v26 historical table changed")
    require(sha(ROOT / "doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md") == CARD, "Prospective v27 card changed")
    require(sha(Path("/opt/cuwacunu/build/market-study/market_fixed_regression")) == FIXED, "Preserved fixed evaluator changed")
    require("PASS" in admission.read_text() and "RPB-v27" in admission.read_text(), "Artificial CUDA admission missing")
    verification, audit = load(run / "verification.json"), load(run / "independent-audit.json")
    require(verification["status"] == audit["status"] == "passed", "Passed wrapper verification and independent arithmetic audit required")
    require(verification["kind"] == "market_parent_conditional_changes_verification_v1" and
            verification["protocol"] == "market-parent-conditional-changes-v1" and verification["instance"] == TAG and
            verification["card_sha256"] == CARD and verification["independent_audit_sha256"] == sha(run / "independent-audit.json"),
            "Verification declaration/audit binding mismatch")
    require(audit["kind"] == "market_parent_conditional_changes_independent_arithmetic_audit_v1" and
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
    loss_summary, normalizer, mapping = validate_saved_run(run, summary, initial)
    readout = load(run / "readouts/report.json")
    new_rows, full_coverage = validate_readouts(readout, run)
    old_table = load(OLD_TABLE)
    rows = []
    for original in old_table["rows"]:
        row = dict(original)
        row.update(reused=True, original_report_path=row.get("report_path"), original_report_sha256=row.get("report_sha256"),
                   report_path=relative(OLD_TABLE), report_sha256=OLD_TABLE_SHA)
        rows.append(row)
    require(len(rows) == 64, "All64 historical rows must remain")
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
    auditor = ROOT / "code/scripts/audit-market-parent-conditional-changes.py"
    require(auditor.is_file(), "Independent auditor source missing")
    sources[relative(auditor)] = sha(auditor)
    for name in ("code/scripts/execute-market-parent-conditional-changes.py", "code/scripts/preflight-market-parent-conditional-changes.ps1"):
        sources[name] = sha(ROOT / name)
    sources["code/scripts/append-encoder-search-ledger.py"] = sha(ROOT / "code/scripts/append-encoder-search-ledger.py")
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
        if source.suffix in TEXT_SUFFIXES and "witness" not in source.name.lower() and source.name != "conditional-map.json":
            copy_exclusive(source, out / "capture/run" / source.relative_to(run), item["sha256"])
    for name, digest in sources.items():
        copy_exclusive(ROOT / name, out / "capture/sources" / name, digest)
    for item in wrapper_assets:
        source = ROOT / item["source"]
        require(source.suffix in TEXT_SUFFIXES, "Unexpected execution-wrapper artifact")
        copy_exclusive(source, out / "capture/execution-preservation" / source.relative_to(wrapper), item["sha256"])
    copy_exclusive(OLD_TABLE, out / "capture/controls/v26-table_rows.json", OLD_TABLE_SHA)
    for index, source in enumerate(dict.fromkeys(prior_docs + [admission])):
        require(source.suffix != ".pt" and "witness" not in source.name.lower(), "Raw tensor/witness cannot enter docs")
        copy_exclusive(source, out / "capture/prerequisites" / f"{index:03d}-{source.name}")
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
    tables = dict(kind="market_parent_conditional_change_results_v1", status="complete", dataset_id="MKT-HIST-2", instance=TAG,
                  rows=rows, training_rows=training_rows, original_criteria=original, strict_native36_criteria=strict,
                  individual_neural_trend_criteria=individual, coverage_criterion=coverage,
                  original_criteria_passed=sum(item["passed"] for item in original), original_criteria_total=9,
                  strict_criteria_passed=sum(item["passed"] for item in strict), strict_criteria_total=8,
                  full_market_gate_passed=market_pass, synthetic_transfer_measured=False, primary="RPB-v22", primary_changed=False,
                  comparison_absolute_arithmetic_tolerance=TOL, D36_actual_seeds=SEEDS, D36_head_parameters=dict(linear=148, neural=660),
                  training_loss=loss_summary, parent_stream_parity=True)
    lines = [f"# {TAG}: parent-conditional change-branch results", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
             f"**The complete market gate {'passes' if market_pass else 'fails'}: original {tables['original_criteria_passed']}/9 and strict native36 floors {tables['strict_criteria_passed']}/8. RPB-v22 remains primary.**", "",
             "One new CUDA trajectory on RTX A2000 8GB/cuda:0:2048 branch updates/B8 after2048 inherited parent updates.",
             "Same4096 TRAIN anchors,112 TRAIN/32 known VALIDATION probes, input scaler, targets, masks and counter streams. TRAIN Jan1–21; known VALIDATION Jan22–28; Jan29–31 unused.",
             "All64 historical task rows are reused. Only new initialized/final native36 banks receive two ridge pipelines/eight target fits and six neural fits; no old quality encoder replay/head refit.", "",
             "V27 subtracts the frozen TRAIN prediction of the conditioned generic change bank from parent32 before its learned projection.",
             "It reuses the exact v26 bank normalizer. The new unlabeled conditional map uses4096 natural-mask TRAIN windows and512 necessary frozen CUDA parent forwards, with zero calibration optimizer updates or held-out calibration inputs.",
             "Map fitting uses CPU F64 parent-normalized ridge slopes with unit penalty and an unpenalized intercept; runtime subtraction/division/map projection and bank correction use CUDA F32. Invalid/unsupported entries emit zero.",
             "It retains waveform Huber1, protected parent32, four learned change coordinates and the zero-start residual decoder. No auxiliary loss, task labels, residual restandardization, clipping or PCA.",
             "Encoder weights match the original native36 initialization. Complete initial36 changes through the conditional correction; parent prefix32 and initial reconstruction remain exact.", "",
             "Fixed ridge1/neural16-tanh/Adam0.01/100 steps; three repetitions2701/2802/2903; native36 heads148/660 parameters.",
             "Exact uint64 seeds:" + ", ".join(map(str, SEEDS)) + ". Means retain all fits, without selecting repetitions or checkpoints.",
             "These are known-source chronological representation results, not forecasting, TEST or independent confirmation. R² is not classification accuracy.", ""]
    display = ["Raw data — no encoder (recent128)", "Raw data — no encoder (full256)", "RPB-v22.market-02.init", "RPB-v22.market-02",
               "RPB-v23.market-01.init", "RPB-v23.market-01", "RPB-v23.market-01.alt-01", "RPB-v24.market-01.init", "RPB-v24.market-01",
               NATIVE36 + ".init", NATIVE36, "RPB-v25.market-01.init", "RPB-v25.market-01",
               "RPB-v26.market-01.init", "RPB-v26.market-01", TAG + ".init", TAG]
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
            lines.append(model + " neural R² by2701/2802/2903:" + ", ".join(f"{value:.17g}" for value in lookup[model, task]["neural_r2_each"]) + ".")
        lines.append("")
    lines.extend(["## Training and calibration", "", LEGEND + " · unchanged fixed hidden TRAIN/known-VALIDATION queries.", ""])
    table(lines, ["Encoder", "Updates", "Train error ↓", "Validation error ↓", "GPU training seconds"],
          [(row["encoder"], row["updates"], f"{row['train_error']:.6f}", f"{row['validation_error']:.6f}",
            "—" if row["gpu_training_seconds"] is None else f"{row['gpu_training_seconds']:.3f}") for row in training_rows])
    lines.extend(["Errors are fixed-query TRAIN-CF-scaled MAE. New branch updates and inherited parent exposure remain separate.",
                  f"New calibration parent extraction: {summary['parent_calibration_seconds_including_io']:.6f}s including I/O and512 frozen CUDA forwards; map fit: {summary['conditional_map_fit_seconds_including_io']:.6f}s including I/O. Bank standardizer reused, zero new fits. GPU training includes checkpoint I/O and excludes calibration and captures.",
                  f"Fixed saved-feature fitting/scoring/artifact I/O: {readout_seconds:.6f}s; pure head-fit compute is unavailable. Capture timings remain in initial-query/summary JSON.",
                  f"Initialized/final feature extraction and archive I/O: {sum(initial[role]['feature_seconds_including_io'] for role in ('train', 'validation')):.6f}s/{sum(summary['trained_query'][role]['feature_seconds_including_io'] for role in ('train', 'validation')):.6f}s; fixed-query capture and I/O: {sum(initial[role]['query_seconds_including_io'] for role in ('train', 'validation')):.6f}s/{sum(summary['trained_query'][role]['query_seconds_including_io'] for role in ('train', 'validation')):.6f}s.",
                  f"Independent saved-scalar audit: {audit['elapsed_seconds']:.6f}s. Wrapper duration {verification['total_wrapper_seconds_including_build_io_and_head_fitting']:.6f}s also includes preservation hashing/build/I/O/head fitting and is not GPU training time.",
                  f"Waveform loss first/last:{loss_summary['first']:.8f}/{loss_summary['last']:.8f}; first64/last64 means:{loss_summary['first64_mean']:.8f}/{loss_summary['last64_mean']:.8f}.",
                  "Different training minibatches are descriptive; fixed-query errors retain the matched scoring population.",
                  "Total/active/protected encoder parameters:297425/14904/282521. The frozen conditional map adds8910 coefficients (33×270),71,280 bytes in F64 fitting storage or35,640 bytes after F32 runtime conversion, plus frozen statistics. It adds no trainable parameters.",
                  f"All frozen conditional-map tensors occupy {summary['conditional_map_frozen_tensor_storage_bytes']} bytes. The full coefficient/statistic JSON and witnesses remain ignored local files; durable metadata contains their identities and hashes.",
                  f"Projection/decoder nonzero-gradient counts:{summary['nonzero_change_gradient_updates']}/{summary['nonzero_decoder_gradient_updates']}; weight-change counts:{summary['projection_weight_updates']}/{summary['decoder_weight_updates']}.", "",
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
        direction = "The full market gate passes. Next test this exact frozen checkpoint on the original compatible TEMPO-5 cohorts, both unchanged scaler policies and fixed heads. Keep the bank normalizer and conditional map frozen. Transfer is not measured here; fresh chronology follows a useful balanced result."
    else:
        failed = [item["criterion"] for item in original + strict + individual + [coverage] if not item["passed"]]
        direction = "The exact parent-conditional mechanism stops. Failed conditions:" + ", ".join(failed) + ". Preserve all outcomes without map-penalty, normalizer, head, rate, seed, width or budget rescue; no synthetic transfer to rescue this market failure."
    lines.extend(["## Preservation and next direction", "", direction,
                  "Keep native36 as the recorded balanced reference until a separately verified selection changes it. V25, v26 and other per-task leaders remain visible.",
                  "This is approximate residualization with a changed learning/readout prior, without an orthogonality or causal generalization guarantee. Under varying masks it need not preserve the same affine feature span; it may remove signals useful to the residual decoder.",
                  f"Independent saved-arithmetic audit passed {audit['checks']} checks. Wrapper verification and source/input protection passed; the earlier full preservation scan is bound separately, not claimed repeated.",
                  "Local checkpoints, native features and optimizer/calibration archives remain ignored. Exact source/text captures and all full-precision physical errors/repetitions are retained beside this report.",
                  "The generated candidate-record.json is append-ready verified metadata. This reporter does not edit the live search ledger, primary selection, Git history or remotes.", ""])
    dump(out / "table_rows.json", tables)
    exclusive(report_path, ("\n".join(lines).rstrip() + "\n").encode())
    inventory = dict(kind="market_parent_conditional_change_assets_v1", run=relative(run), run_assets=run_assets,
                     wrapper_assets=wrapper_assets,
                     captured_sources=[asset(ROOT / name) for name in sources], shared_assets=shared_assets,
                     reused_table=asset(OLD_TABLE), admission=asset(admission), wrapper_verification=asset(run / "verification.json"),
                     independent_audit=asset(run / "independent-audit.json"))
    dump(out / "asset_hashes.json", inventory)
    candidate = dict(record_id="2026-10-15-verified/" + TAG, model=TAG, protocol="market-parent-conditional-changes-v1",
                     design="Frozen TRAIN parent-conditional map of the unchanged conditioned adjacent-change bank",
                     evidence="verified new market run; no automatic selection", run=relative(run), card_sha256=CARD,
                     native_width=36, new_branch_updates=2048, inherited_parent_updates=2048, batch_size=8,
                     input_scaler_identity=summary["scaler_identity"], normalizer_sha256=summary["bank_normalizer_sha256"],
                     normalizer_identity=summary["bank_normalizer_identity"],
                     conditional_map_identity=summary["conditional_map_identity"], conditional_map_sha256=summary["conditional_map_sha256"],
                     conditional_map_source_identity=summary["conditional_map_source_identity"],
                     normalizer_source_identity=summary["bank_normalizer_source_identity"], final_task_rows=[row for row in rows if row["method"] == TAG],
                     original_conditions=original, strict_conditions=strict, individual_trend_conditions=individual, coverage_condition=coverage,
                     full_market_gate_passed=market_pass, synthetic_transfer_measured=False, primary_changed=False,
                     checkpoints=[item for item in run_assets if "/checkpoints/" in item["source"]],
                     report_sha256=sha(report_path), table_rows_sha256=sha(out / "table_rows.json"),
                     card_asset=asset(ROOT / "doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md"), report_asset=asset(report_path),
                     reused_table=asset(OLD_TABLE), conditional_map_summary=asset(run / "conditional-map-summary.json"),
                     head_recipe=dict(ridge_penalty=1, neural_hidden=16, neural_activation="tanh", neural_optimizer="Adam",
                                      neural_learning_rate=.01, neural_updates=100, outputs=4, repetitions=REPS,
                                      uint64_actual_seeds_decimal=list(map(str,SEEDS)),ridge_parameters=148,neural_parameters=660,
                                      feature_and_target_normalizers="TRAIN-only; original fixed evaluator",post_encoder_PCA=False),
                     policy="Append this distinct record only after controlling-task verification; preserve every old entry and live leader unless a verified win changes it.")
    dump(out / "candidate-record.json", candidate)
    completion = dict(kind="market_parent_conditional_change_completion_v1", protocol="market-parent-conditional-changes-v1",
                      date="2026-10-15", status="complete", instance=TAG,
                      run=relative(run), dataset_id="MKT-HIST-2", designed_complexity_level=4, encoder_trajectories=1,
                      new_branch_updates=2048, inherited_parent_updates=2048, batch_size=8, attempted_batches=summary["attempted_batches"],
                      old_encoder_quality_replays=0, old_head_refits=0, CPU_encoder_calls=0, neural_fits=6, ridge_pipeline_fits=2, ridge_target_fits=8,
                      original_criteria_passed=tables["original_criteria_passed"], original_criteria_total=9,
                      strict_criteria_passed=tables["strict_criteria_passed"], strict_criteria_total=8,
                      all_three_final_neural_trend_positive=all(item["passed"] for item in individual), full_coverage=full_coverage,
                      full_market_gate_passed=market_pass, stopped_without_rescue=not market_pass, synthetic_transfer_measured=False,
                      primary_changed=False, card_sha256=CARD, source_closure_sha256=summary["source_closure_sha256"], binary_sha256=summary["binary_sha256"],
                      fixed_evaluator_binary_sha256=FIXED, normalizer_sha256=summary["bank_normalizer_sha256"],
                      normalizer_identity=summary["bank_normalizer_identity"],
                      conditional_map_sha256=summary["conditional_map_sha256"], conditional_map_identity=summary["conditional_map_identity"],
                      conditional_map_source_identity=summary["conditional_map_source_identity"], conditional_map_coefficient_count=8910,
                      conditional_map_coefficient_storage_bytes=71280, conditional_map_frozen_tensor_storage_bytes=2571950,
                      report_sha256=sha(report_path), table_rows_sha256=sha(out / "table_rows.json"), asset_hashes_sha256=sha(out / "asset_hashes.json"),
                      candidate_record_sha256=sha(out / "candidate-record.json"), reporter_sha256=sha(Path(__file__)),
                      independent_arithmetic_checks=audit["checks"], independent_audit_sha256=sha(run / "independent-audit.json"),
                      wrapper_verification_sha256=sha(run / "verification.json"),
                      parent_calibration_seconds_including_io=summary["parent_calibration_seconds_including_io"],
                      conditional_map_fit_seconds_including_io=summary["conditional_map_fit_seconds_including_io"],
                      new_calibration_parent_CUDA_encoder_calls=512, new_calibration_encoder_updates=0, new_bank_normalizer_fits=0,
                      gpu_training_seconds_including_checkpoint_IO=summary["training_seconds_including_checkpoints"],
                      saved_feature_fitting_scoring_and_IO_seconds=readout_seconds, pure_head_fit_seconds=None,
                      reporting_and_hash_capture_seconds=time.perf_counter() - started, source_assets=run_assets,
                      captured_sources=inventory["captured_sources"], checkpoint_assets=candidate["checkpoints"])
    dump(out / "completion.json", completion)
    members = [dict(path=path.relative_to(out).as_posix(), sha256=sha(path), bytes=path.stat().st_size)
               for path in sorted(out.rglob("*")) if path.is_file()]
    dump(out / "durable_files.json", dict(kind="market_parent_conditional_change_durable_v1", report=asset(report_path), files=members,
                                         policy="Frozen outputs; later additive receipts are recorded separately."))
    require(not any(path.suffix == ".pt" or "witness" in path.name.lower() or path.name == "conditional-map.json"
                    for path in out.rglob("*") if path.is_file()), "Tensor/map-weight/witness exported into docs")
    print(json.dumps(dict(status="complete", run=relative(run), task_rows=len(rows), original_passed=tables["original_criteria_passed"],
                          strict_passed=tables["strict_criteria_passed"], full_market_gate_passed=market_pass, report=relative(report_path))))


if __name__ == "__main__":
    main()


