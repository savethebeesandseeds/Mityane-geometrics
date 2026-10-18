#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Report immutable saved v25 evidence using JSON/CSV and hashes only.

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
TAG = "RPB-v25.market-01"
NATIVE36 = "RPB-v24.market-01.alt-01"
CARD = "c67e5c37d56e5bc2801a9d32152589e3067ae11f8b18c8c804418b8cca807938"
FIXED = "5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14"
PARENT_SHA = "e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c"
INIT_SHA = "2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108"
SHARED_SHA = "dbc208fb7c858909ff6f04e70b939adaab1c3e8c27f8e72f261a620cf1b55e10"
MANIFEST_SHA = "828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8"
OLD_TABLE = ROOT / "doc/results/separate_change_native36_v1/table_rows.json"
OLD_TABLE_SHA = "1723aa555f22ec224f5ba44291e8ead8e5488d57c2441fa481dd83e3c3d1e1f8"
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
    require([method["name"] for method in report["methods"]] == ["initialized_v25", "trained_v25"],
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
            row = dict(dataset_id="MKT-HIST-2", task=task, recipe_id="market-change-energy-v1",
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


def validate_context(context):
    require(context["protocol"] == "market-change-energy-train-context-v1" and context["dataset_id"] == "MKT-HIST-2" and
            context["designed_complexity_level"] == 4 and context["TRAIN_rows"] == 112, "TRAIN context role/protocol mismatch")
    require(all(context[key] == 0 for key in ("new_encoder_calls", "new_encoder_updates", "new_head_fits", "held_out_tensor_inputs")),
            "TRAIN context must use saved TRAIN arithmetic without model calls, updates, refits or held-out tensors")
    require([method["method"] for method in context["methods"]] ==
            ["RPB-v22.market-02", NATIVE36, TAG + ".init", TAG], "TRAIN context methods changed")
    opened = []
    for entry in context["opened_inputs"]:
        path = Path(entry["path"])
        path = path if path.is_absolute() else ROOT / path
        require(path.is_relative_to(ROOT / "output") and path.suffix == ".pt" and
                sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"], "saved TRAIN context input binding changed")
        require("validation" not in path.name and "test" not in path.name.lower(), "held-out TRAIN-context archive forbidden")
        opened.append(asset(path))
    rows = []
    for method in context["methods"]:
        require(method["size"] == (32 if method["method"] == "RPB-v22.market-02" else 36), "TRAIN context width changed")
        require([view["context"] for view in method["contexts"]] == ["all112", "first56", "last56"], "TRAIN contexts changed")
        for view in method["contexts"]:
            count = 112 if view["context"] == "all112" else 56
            require(view["rows"] == count and [task["task"] for task in view["tasks"]] == TASKS,
                    "TRAIN context sizes/tasks changed")
            for task in view["tasks"]:
                ridge, neural = task["ridge"], task["neural_mean_repetition_scores"]
                require((ridge["valid"], ridge["total"], ridge["target_eligible"], ridge["coverage"]) == (count, count, count, 1),
                        "TRAIN context coverage incomplete")
                require([repetition["repetition"] for repetition in task["neural_each"]] == REPS,
                        "TRAIN context repetitions changed")
                require(all(finite(scores[metric]) for scores in (ridge, neural) for metric in ("r2", "mae", "rmse")),
                        "nonfinite TRAIN context score")
                rows.append(dict(method=method["method"], size=method["size"], task=task["task"], view=view["context"],
                                 total=count, valid=count, coverage=1, linear_r2=ridge["r2"], neural_mean_r2=neural["r2"],
                                 linear_mae=ridge["mae"], linear_rmse=ridge["rmse"], neural_mean_mae=neural["mae"],
                                 neural_mean_rmse=neural["rmse"], unit=task["unit"], neural_each=task["neural_each"],
                                 reused_model_and_fit=not method["method"].startswith(TAG), new_head_fits=0))
    return rows, opened


def validate_saved_run(run, summary, initial):
    expected = dict(kind="market_change_energy_v1", dataset="MKT-HIST-2", instance=TAG, H=128, W=64, D=36,
                    level_width=32, change_width=4, parameters=298505, optimizer_updated_parameters=15984,
                    protected_parameters=282521, completed_updates=2048, attempted_batches=2048,
                    window_presentations=16384, parent_updates=2048, probe_train_rows=112, probe_validation_rows=32,
                    CPU_encoder_calls=0, cuda_learning_verified=True, protected_level_exact=True, protected_optimizer_states=0,
                    initial_prefix_matches_saved_trained_v22=True, initial_all36_match_saved_v24_init=True,
                    final_prefix_matches_saved_trained_v22=True, card_sha256=CARD, parent_inference_sha256=PARENT_SHA,
                    shared_ledger_sha256=SHARED_SHA, input_manifest_sha256=MANIFEST_SHA,
                    scaler_identity="rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969", moment_normalizer_TRAIN_anchors=4096,
                    moment_normalizer_validation_anchors=0, moment_normalizer_encoder_calls=0)
    require(all(summary[key] == value for key, value in expected.items()), "v25 saved run contract mismatch")
    for filename, key in [("study.conf", "config_sha256"), ("encoder.conf", "settings_sha256"),
                          ("prospective-card.md", "card_sha256"), ("source-closure.sha256", "source_closure_sha256"),
                          ("moment-rms-scaler.pt", "moment_normalizer_sha256")]:
        require(sha(run / filename) == summary[key], "saved binding changed: " + filename)
    require(sha(Path("/opt/cuwacunu/build/market-energy/market_change_energy")) == summary["binary_sha256"],
            "new producer binary changed")
    require(sha(PARENT / "checkpoints/inference-2048.pt") == PARENT_SHA and
            sha(OLD36 / "checkpoints/inference-0.pt") == INIT_SHA, "matched local parent/init missing or changed")
    checkpoint_names = ["inference-0.pt", "inference-2048.pt", "training-512.pt", "training-1024.pt", "training-2048.pt"]
    require(sorted(path.name for path in (run / "checkpoints").iterdir()) == sorted(checkpoint_names),
            "required immutable checkpoint inventory changed")
    norm = load(run / "moment-rms-scaler.json")
    require(norm["TRAIN_anchors"] == 4096 and norm["validation_anchors"] == norm["encoder_calls"] == 0 and
            norm["source_identity"] == summary["moment_source_identity"] and norm["archive_sha256"] == summary["moment_normalizer_sha256"],
            "RMS calibration role/identity changed")
    require(all(len(norm[key]) == 270 for key in ("counts", "sumsq", "scales", "support")), "RMS calibration width")
    for count, squared, scale, supported in zip(norm["counts"], norm["sumsq"], norm["scales"], norm["support"]):
        require(type(count) is int and 0 <= count <= 4096 and finite(squared) and squared >= 0 and
                finite(scale) and scale >= 1e-6 and bool(supported) == (count > 0), "RMS sufficient statistics invalid")
        expected_scale = max(math.sqrt(squared / count), 1e-6) if count else 1.
        require(math.isclose(scale, expected_scale, rel_tol=1e-12, abs_tol=1e-15), "uncentered RMS arithmetic mismatch")
    before, after = load(run / "initialization-parity.json"), load(run / "final-prefix-parity.json")
    require(before == dict(saved_v22_train_and_validation_prefix32_equal=True,
                           saved_v24_initial_train_and_validation_all36_equal=True, old_encoder_replay_calls=0, old_head_refits=0),
            "full initialized36/prefix32 parity receipt changed")
    require(after == dict(saved_v22_train_and_validation_prefix32_equal=True, old_encoder_replay_calls=0, old_head_refits=0),
            "final parent-prefix parity receipt changed")
    parent = load(PARENT / "summary.json")
    for role, rows in (("train", 112), ("validation", 32)):
        for key in ("mae_train_scaled", "row_mae", "row_target_cells", "row_eligible_channels", "eligible_rows",
                    "total_rows", "target_cells", "target_history"):
            require(initial[role][key] == parent["trained_query"][role][key], "fixed initial waveform query differs from parent")
        require(summary["trained_query"][role]["eligible_rows"] == summary["trained_query"][role]["total_rows"] == rows,
                "final fixed-query coverage changed")
    with (run / "training.csv").open(newline="") as handle:
        training = list(csv.DictReader(handle))
    with (OLD36 / "training.csv").open(newline="") as handle:
        old_training = list(csv.DictReader(handle))
    require(len(training) == len(old_training) == 2048, "training stream length changed")
    stream_keys = ("attempt", "update", "eligible_examples", "target_cells", "row_seed", "mask_seed", "anchors")
    require(all(all(row[key] == old[key] for key in stream_keys) for row, old in zip(training, old_training)),
            "row/anchor/query/context counter streams differ")
    components = {}
    for key in ("loss", "wave_loss", "aux_loss", "signed_loss", "energy_loss"):
        values = [float(row[key]) for row in training]
        require(all(finite(value) for value in values), "nonfinite saved loss component")
        components[key] = dict(first=values[0], last=values[-1], first64_mean=statistics.mean(values[:64]),
                               last64_mean=statistics.mean(values[-64:]))
    for row in training:
        expected_loss = float(row["wave_loss"]) + .1 * float(row["aux_loss"])
        require(abs(float(row["loss"]) - expected_loss) <= 4e-7 * (1 + abs(expected_loss)),
                "saved F32 total loss composition mismatch")
        expected_aux = .5 * (float(row["signed_loss"]) + float(row["energy_loss"]))
        require(abs(float(row["aux_loss"]) - expected_aux) <= 4e-7 * (1 + abs(expected_aux)),
                "saved F32 auxiliary group balance mismatch")
    count_fields = [("projection_gradient_nonzero", "nonzero_change_gradient_updates"),
                    ("wave_decoder_gradient_nonzero", "nonzero_decoder_gradient_updates"),
                    ("moment_decoder_gradient_nonzero", "nonzero_moment_gradient_updates"),
                    ("projection_weight_changed", "projection_weight_updates"),
                    ("wave_decoder_weight_changed", "decoder_weight_updates"),
                    ("moment_decoder_weight_changed", "moment_weight_updates")]
    for field, key in count_fields:
        count = sum(int(row[field]) for row in training)
        require(0 < count <= 2048 and count == summary[key], "actual CUDA learning count mismatch: " + field)
    require(training[0]["projection_gradient_nonzero"] == "0" and
            all(training[0][key] == "1" for key in ("wave_decoder_gradient_nonzero", "moment_decoder_gradient_nonzero",
                                                    "wave_decoder_weight_changed", "moment_decoder_weight_changed")) and
            all(training[1][key] == "1" for key, _ in count_fields), "first-open/second-learn evidence mismatch")
    return components, norm


def main():
    started = time.perf_counter()
    require(Path("/.dockerenv").is_file() and ROOT == Path("/cuwacunu"), "use existing managed cuwacunu container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "doc/MARKET_CHANGE_ENERGY_RESULTS.md")
    parser.add_argument("--train-context", required=True, type=Path)
    parser.add_argument("--admission-log", type=Path)
    parser.add_argument("--peer-review", type=Path)
    args = parser.parse_args()
    run, out, report_path = args.run.resolve(), args.output_dir.resolve(), args.report.resolve()
    context_path = args.train_context.resolve()
    admission_path = (args.admission_log or out / "engineering-01.log").resolve()
    peer_path = (args.peer_review or out / "peer-review.json").resolve()
    require(run.is_dir() and run.parent == ROOT / "output/runs/market-change-energy" and run.name.startswith("run-"), "new v25 run required")
    require(out.is_relative_to(ROOT / "doc/results") and report_path.is_relative_to(ROOT / "doc"), "durable outputs belong in integration docs")
    destinations = [report_path, out / "table_rows.json", out / "completion.json", out / "asset_hashes.json",
                    out / "durable_files.json", out / "capture"]
    require(not any(path.exists() for path in destinations), "exclusive reporting destination already exists")
    require(sha(OLD_TABLE) == OLD_TABLE_SHA, "preserved native36 control table changed")
    require(sha(ROOT / "doc/MARKET_CHANGE_ENERGY_CARD.md") == CARD, "frozen prospective card changed")
    require(sha(Path("/opt/cuwacunu/build/market-study/market_fixed_regression")) == FIXED, "fixed evaluator changed")
    require("PASS RPB-v25 CUDA admission:" in admission_path.read_text(), "artificial CUDA admission missing")
    peer = load(peer_path)
    require(peer.get("status") in ("passed", "no_blocker_found") and peer.get("card_sha256") == CARD and
            peer.get("engineering", {}).get("result") == "passed", "prequality peer review receipt missing/pending")
    audit = load(run / "independent-audit.json")
    require(audit["kind"] == "market_change_energy_independent_arithmetic_audit_v1" and audit["status"] == "passed" and
            audit["quality_gate_tolerance"] == TOL and audit["new_encoder_calls"] == audit["new_encoder_updates"] ==
            audit["head_or_scaler_fits"] == 0, "independent saved-arithmetic audit missing/pending")
    context = load(context_path)
    train_context_rows, train_context_inputs = validate_context(context)
    summary, initial = load(run / "summary.json"), load(run / "initial-query.json")
    components, norm = validate_saved_run(run, summary, initial)
    readout = load(run / "readouts/report.json")
    added_rows, full_coverage = validate_readouts(readout, run)
    rows = []
    old_tables = load(OLD_TABLE)
    for original in old_tables["rows"]:
        row = dict(original)
        row.update(reused=True, original_report_path=row.get("report_path"), original_report_sha256=row.get("report_sha256"),
                   report_path=relative(OLD_TABLE), report_sha256=OLD_TABLE_SHA)
        rows.append(row)
    require(len(rows) == 48, "preserve all48 old task rows")
    rows.extend(added_rows)
    lookup = {(row["method"], row["task"]): row for row in rows}
    for task in TASKS:
        before, matched = lookup[NATIVE36 + ".init", task], lookup[TAG + ".init", task]
        for metric in ("linear_r2", "linear_mae", "linear_rmse", "neural_mean_r2", "neural_mean_mae", "neural_mean_rmse"):
            require(abs(before[metric] - matched[metric]) <= TOL, "matched initial bank fixed-head score changed: " + metric)
        require(before["neural_r2_each"] == matched["neural_r2_each"], "matched initial head repetition scores changed")
    original, strict, individual, coverage = decisions(lookup, full_coverage)
    market_pass = all(item["passed"] for item in original + strict + individual + [coverage])
    timing = load(run / "readout-timing.json")
    require(timing["kind"] == "saved_feature_readout_timing_v1" and timing["label"] == "fixed saved-feature fitting/scoring/I/O" and
            timing["elapsed_ns"] > 0 and timing["elapsed_ns"] == timing["finished_epoch_ns"] - timing["started_epoch_ns"], "readout timing arithmetic/scope")
    readout_seconds = timing["elapsed_ns"] / 1e9
    source_pins = {}
    for line in (run / "source-closure.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and sha(ROOT / path) == digest, "compiled source closure changed")
        require(name not in source_pins, "duplicate source closure member")
        source_pins[name] = digest
    source_pins[relative(Path(__file__))] = sha(Path(__file__))
    auditor_path = ROOT / "code/scripts/audit-market-change-energy.py"
    source_pins[relative(auditor_path)] = sha(auditor_path)
    for name in ("code/evaluation/market_change_energy_train_context_v1/main.cpp",
                 "code/evaluation/market_change_energy_train_context_v1/context.mk"):
        source_pins[name] = sha(ROOT / name)
    require(sha(SHARED / "shared-assets.sha256") == SHARED_SHA, "shared ledger changed")
    shared_assets = []
    for line in (SHARED / "shared-assets.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        member = Path(name)
        require(not member.is_absolute() and ".." not in member.parts and sha(SHARED / member) == digest, "shared source changed")
        shared_assets.append(asset(SHARED / member))
    run_assets = [asset(path) for path in sorted(run.rglob("*")) if path.is_file()]
    prior_docs = [path for path in sorted(out.rglob("*")) if path.is_file()] if out.exists() else []
    out.mkdir(parents=True, exist_ok=True)
    for item in run_assets:
        source = ROOT / item["source"]
        if source.suffix in TEXT_SUFFIXES:
            copy_exclusive(source, out / "capture/run" / source.relative_to(run), item["sha256"])
    for name, digest in source_pins.items():
        copy_exclusive(ROOT / name, out / "capture/sources" / name, digest)
    copy_exclusive(OLD_TABLE, out / "capture/controls/native36-table_rows.json", OLD_TABLE_SHA)
    copy_exclusive(context_path, out / "capture/train-context.json")
    for index, source in enumerate(dict.fromkeys(prior_docs + [admission_path, peer_path])):
        require(source.suffix != ".pt", "never export tensor archives")
        copy_exclusive(source, out / "capture/prerequisites" / f"{index:03d}-{source.name}")
    training_rows = []
    for previous in old_tables["training_rows"]:
        row = dict(previous)
        row.update(reused=True, report_path=relative(OLD_TABLE), report_sha256=OLD_TABLE_SHA)
        training_rows.append(row)
    for label, updates, query, seconds in [(TAG + ".init", 0, initial, None),
                                            (TAG, 2048, summary["trained_query"], summary["training_seconds_including_checkpoints"])]:
        training_rows.append(dict(encoder=label, updates=updates, inherited_updates=2048, reused=False,
                                  train_error=query["train"]["mae_train_scaled"], validation_error=query["validation"]["mae_train_scaled"],
                                  gpu_training_seconds=seconds, dataset_id="MKT-HIST-2", task="masked_reconstruction",
                                  designed_complexity_level=4, complexity_scale_max=5, error_metric="TRAIN-CF-scaled fixed-query MAE",
                                  batch_size=8, device="cuda:0", timing_includes_checkpoint_IO=True))
    tables = dict(kind="market_change_energy_results_v1", status="complete", dataset_id="MKT-HIST-2", instance=TAG,
                  rows=rows, training_rows=training_rows, original_criteria=original, strict_native36_criteria=strict,
                  individual_neural_trend_criteria=individual, coverage_criterion=coverage,
                  original_criteria_passed=sum(item["passed"] for item in original), original_criteria_total=9,
                  strict_criteria_passed=sum(item["passed"] for item in strict), strict_criteria_total=8,
                  full_market_gate_passed=market_pass, synthetic_transfer_authorized_by_gate=market_pass,
                  synthetic_transfer_measured=False, primary="RPB-v22", primary_changed=False,
                  comparison_absolute_arithmetic_tolerance=TOL, D36_actual_seeds=SEEDS,
                  D36_head_parameters=dict(linear=148, neural=660), training_loss_components=components,
                  parent_stream_parity=True, train_context_rows=train_context_rows,
                  train_context_path=relative(context_path), train_context_sha256=sha(context_path))
    lines = [f"# {TAG}: balanced change-energy results", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
             f"**The complete declared market gate {'passes' if market_pass else 'fails'}. Original conditions: {tables['original_criteria_passed']}/9; strict native36 floors: {tables['strict_criteria_passed']}/8. RPB-v22 remains primary.**", "",
             "One new CUDA trajectory, 2048 additional branch updates/B8 after2048 inherited parent updates. Same4096 TRAIN anchors,",
             "112 TRAIN/32 known-VALIDATION readout windows, original source/scaler/targets/query plans and counter streams.",
             "Execution uses cuda:0 on the existing NVIDIA RTX A2000 8GB in the managed cuwacunu container.",
             "TRAIN Jan1–21; known VALIDATION Jan22–28; Jan29–31 unused. All48 historical task rows are reused from the sealed native36 table.",
             "Only new initialized/final banks receive two ridge pipelines/eight target fits and six neural fits. No old encoder replay or old head refit.", "",
             "V25 keeps the entire trained v22 prefix32 and unchanged four-coordinate v24 export, adding a training-only zero-start4→270 moment decoder.",
             "It predicts the existing visible adjacent-change moments with fixed balanced signed/energy Huber loss: waveform+0.1×auxiliary.",
             "The auxiliary RMS normalizer is fit once without centering on full-natural TRAIN4096 only, using generic moments and zero encoder calls.",
             "Counts/scales/support/layout/source are saved. No task labels enter this objective. Native36 remains24 learned+12 fixed coordinates; no post-encoder PCA.", "",
             "Fixed ridge1 and neural16-tanh/Adam0.01/100-step recipes; repetitions2701/2802/2903, head parameters148/660.",
             "Exact width-derived uint64 seeds: " + ", ".join(map(str, SEEDS)) + ". Both new views share these settings.",
             "Neural entries average all three fits, with every repetition and physical error saved in JSON. R² is not classification accuracy.",
             "Three head repetitions on one chronological development split measure head variation; independent chronological uncertainty is not estimated.", ""]
    display = ["Raw data — no encoder (recent128)", "Raw data — no encoder (full256)", "RPB-v22.market-02.init", "RPB-v22.market-02",
               "RPB-v23.market-01.init", "RPB-v23.market-01", "RPB-v23.market-01.alt-01", "RPB-v24.market-01.init", "RPB-v24.market-01",
               NATIVE36 + ".init", NATIVE36, TAG + ".init", TAG]
    for task in TASKS:
        lines.extend(["## " + task.replace("_", " ").capitalize(), "", LEGEND + " · intact known VALIDATION.", ""])
        table(lines, ["Method", "Size", "Linear head R²", "Neural head R²", "Coverage %"],
              [(method, lookup[method, task]["size"], f"{lookup[method, task]['linear_r2']:.3f}",
                f"{lookup[method, task]['neural_mean_r2']:.3f}", f"{100 * lookup[method, task]['coverage']:.0f}") for method in display])
        lines.extend(["Supplemental physical errors, unit: " + lookup[TAG, task]["unit"] + ".", ""])
        table(lines, ["Method", "Linear MAE", "Linear RMSE", "Neural mean MAE", "Neural mean RMSE"],
              [(method,) + tuple(f"{lookup[method, task][metric]:.8g}" for metric in
                                 ("linear_mae", "linear_rmse", "neural_mean_mae", "neural_mean_rmse")) for method in display])
        for method in (TAG + ".init", TAG):
            lines.append(method + " neural R² by repetition2701/2802/2903: " +
                         ", ".join(f"{value:.17g}" for value in lookup[method, task]["neural_r2_each"]) + ".")
        lines.append("")
    lines.extend(["## Training and reconstruction", "", LEGEND + " · unchanged fixed hidden queries, TRAIN and known VALIDATION.", ""])
    table(lines, ["Encoder", "Updates", "Train error ↓", "Validation error ↓", "GPU training seconds"],
          [(row["encoder"], row["updates"], f"{row['train_error']:.6f}", f"{row['validation_error']:.6f}",
            "—" if row["gpu_training_seconds"] is None else f"{row['gpu_training_seconds']:.3f}") for row in training_rows])
    lines.extend(["Updates count new branch updates; inherited parent2048 is recorded separately. GPU training time includes checkpoint I/O and excludes normalization calibration,",
                  "initial/final feature/query capture and fixed readouts. Sampled16384 presentations are not guaranteed complete epochs.",
                  f"Calibration took {summary['moment_normalizer_fit_seconds_including_io']:.6f}s including input I/O; it used zero learned encoder calls.",
                  f"Fixed saved-feature fitting/scoring/artifact I/O took {readout_seconds:.6f}s; pure head-fit compute is unavailable.",
                  "Capture timings remain separate in initial-query.json and summary.json. Reporting/hash time is in completion.",
                  f"Independent saved arithmetic audit passed {audit['checks']} checks. Its float32 loss tolerance is {audit['loss_arithmetic_tolerance']}; quality gates separately use1e−12.",
                  f"Total/active/protected parameters:298505/15984/282521. Projection/wave-decoder/moment-decoder nonzero-gradient updates: {summary['nonzero_change_gradient_updates']}/{summary['nonzero_decoder_gradient_updates']}/{summary['nonzero_moment_gradient_updates']}.",
                  f"Corresponding weight-change counts:{summary['projection_weight_updates']}/{summary['decoder_weight_updates']}/{summary['moment_weight_updates']}.",
                  "Initial all36 banks match saved native36 initialization, and initial/final prefix32+validity+rows match the saved trained parent exactly.", "",
                  "### Loss components", "", LEGEND + " · sampled masked/context-deleted TRAIN minibatches.", "",
                  "Different training minibatches; fixed-query MAE above remains the matched reconstruction comparison.", ""])
    table(lines, ["Component", "First", "Last", "First64 mean", "Last64 mean"],
          [(key,) + tuple(f"{item[field]:.8f}" for field in ("first", "last", "first64_mean", "last64_mean")) for key, item in components.items()])
    lines.extend(["## Declared decisions", "", "All arithmetic comparisons use the predeclared absolute1e−12 tolerance. Positivity checks require a positive saved value.", ""])
    for title, items in [("Original nine conditions", original), ("Strict native36 protection", strict),
                         ("Every final neural trend repetition", individual), ("Coverage", [coverage])]:
        lines.extend(["### " + title, ""])
        for item in items:
            detail = ""
            if "candidate" in item:
                detail = f"; candidate {item['candidate']:.17g}; floor {item['floor']:.17g}; difference {item.get('difference_from_floor', item['difference']):+.17g}"
            lines.append(f"- {item['criterion']}: **{'PASS' if item['passed'] else 'FAIL'}**{detail}.")
        lines.append("")
    lines.extend(["## Saved TRAIN context", "", LEGEND + " · first56/last56 of the same saved112 TRAIN rows, descriptive only.", "",
                  "The supplied context receipt reads saved predictions and targets with zero encoder calls or fits. Its full metrics and provenance are captured beside this report.",
                  "These subdivisions share one training history and do not provide independent confirmation or a reason to tune this objective.", ""])
    # The independent context reader supplies these scores; the reporter does
    # not reopen its tensor inputs or implement a second prediction evaluator.
    for task in TASKS:
        for view in ("all112", "first56", "last56"):
            selected = [row for row in train_context_rows if row["view"] == view and row["task"] == task]
            lines.extend([f"{task}: {view}.", "", LEGEND + f" · saved {view} TRAIN predictions; no new fit.", ""])
            table(lines, ["Method", "Size", "Linear head R²", "Neural head R²", "Coverage %"],
                  [(row["method"], row["size"], f"{row['linear_r2']:.3f}", f"{row['neural_mean_r2']:.3f}",
                    f"{100 * row['coverage']:.0f}") for row in selected])
    lines.extend(["## Physical errors and next direction", "",
                  "Full-precision linear/mean-neural MAE and RMSE, every neural repetition and target normalization are in table_rows.json and captured readouts/report.json.",
                  "Units: volatility logreturn; trend logreturn/tick; activity log1p(base_volume); relation correlation.", ""])
    if market_pass:
        lines.extend(["The full market gate passes. Next evaluate this exact frozen final checkpoint on the predeclared compatible TEMPO-5 cohorts and both scaler policies.",
                      "That transfer is not measured by this report. Primary selection remains v22; fresh chronological confirmation follows a useful balanced candidate."])
    else:
        failed = [item["criterion"] for item in original + strict + individual + [coverage] if not item["passed"]]
        lines.extend(["The exact declared v25 objective stops. Failed conditions: " + ", ".join(failed) + ".",
                      "Do not tune its auxiliary coefficient, head, rate, width, seed or budget, or run synthetic transfer to rescue this market failure.",
                      "Keep native36 as the leading reference and v22 primary. Inspect the saved per-task and TRAIN context tradeoffs before declaring a separate mechanism."])
    lines.extend(["", "All native40/native36/v25 artifacts remain distinct and preserved. Exact parent information survives; fitting extra coordinates can still change generalization.",
                  "This is known-source historical representation development: no forecasting, TEST or fresh chronology. Better auxiliary loss alone does not meet the balanced encoder goal.",
                  "Local checkpoint/native/optimizer archives are hashed, never exported into docs or Git. Captured text, exact sources, admission/peer evidence and manifests remain beside this report.", ""])
    dump(out / "table_rows.json", tables)
    exclusive(report_path, ("\n".join(lines).rstrip() + "\n").encode())
    inventory = dict(kind="market_change_energy_asset_hashes_v1", run=relative(run), run_assets=run_assets,
                     captured_sources=[asset(ROOT / name) for name in source_pins], shared_assets=shared_assets,
                     reused_table=asset(OLD_TABLE), train_context=asset(context_path), train_context_input_archives=train_context_inputs,
                     train_context_local_assets=[asset(path) for path in sorted(context_path.parent.rglob("*")) if path.is_file()],
                     admission=asset(admission_path), peer_review=asset(peer_path))
    dump(out / "asset_hashes.json", inventory)
    completion = dict(kind="market_change_energy_completion_v1", date="2026-10-15", status="complete", instance=TAG,
                      run=relative(run), dataset_id="MKT-HIST-2", designed_complexity_level=4, encoder_trajectories=1,
                      new_branch_updates=2048, inherited_parent_updates=2048, batch_size=8, attempted_batches=summary["attempted_batches"],
                      old_encoder_quality_replays=0, old_head_refits=0, CPU_encoder_calls=0, neural_fits=6, ridge_pipeline_fits=2, ridge_target_fits=8,
                      original_criteria_passed=tables["original_criteria_passed"], original_criteria_total=9,
                      strict_criteria_passed=tables["strict_criteria_passed"], strict_criteria_total=8,
                      all_three_final_neural_trend_positive=all(item["passed"] for item in individual), full_coverage=full_coverage,
                      full_market_gate_passed=market_pass, synthetic_transfer_measured=False, primary_changed=False,
                      stopped_without_rescue=not market_pass, card_sha256=CARD, source_closure_sha256=summary["source_closure_sha256"],
                      binary_sha256=summary["binary_sha256"], fixed_evaluator_binary_sha256=FIXED,
                      report_sha256=sha(report_path), table_rows_sha256=sha(out / "table_rows.json"), asset_hashes_sha256=sha(out / "asset_hashes.json"),
                      reporter_sha256=sha(Path(__file__)), saved_evidence_reporting_seconds=time.perf_counter() - started,
                      calibration_seconds=summary["moment_normalizer_fit_seconds_including_io"],
                      gpu_training_seconds_including_checkpoint_IO=summary["training_seconds_including_checkpoints"],
                      saved_feature_fitting_scoring_and_io_seconds=readout_seconds, pure_head_fit_seconds=None,
                      independent_arithmetic_audit_status="passed", independent_arithmetic_checks=audit["checks"],
                      independent_arithmetic_audit_sha256=sha(run / "independent-audit.json"),
                      global_preservation_receipts="recorded separately by the controlling task",
                      checkpoint_assets=[item for item in run_assets if "/checkpoints/" in item["source"]],
                      source_assets=run_assets, captured_sources=inventory["captured_sources"])
    dump(out / "completion.json", completion)
    members = [dict(path=path.relative_to(out).as_posix(), sha256=sha(path), bytes=path.stat().st_size)
               for path in sorted(out.rglob("*")) if path.is_file()]
    dump(out / "durable_files.json", dict(kind="market_change_energy_durable_files_v1", report=asset(report_path), files=members,
                                         policy="Existing prerequisites and new captures preserved; later additive receipts are recorded separately."))
    require(not any(path.suffix == ".pt" for path in out.rglob("*")), "tensor archive must remain local ignored output")
    print(json.dumps(dict(status="complete", run=relative(run), original_passed=tables["original_criteria_passed"],
                          strict_passed=tables["strict_criteria_passed"], full_market_gate_passed=market_pass,
                          task_rows=len(rows), run_artifacts=len(run_assets), report=relative(report_path))))


if __name__ == "__main__":
    main()
