#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Publish saved native36 evidence; never execute a model or fit a head."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TAG = "RPB-v24.market-01.alt-01"
CARD = "0ce8ce1a523150f5519945f4e484886dd22c7bbf212ebc15e0c1047bd04b3ca9"
EVALUATOR = "5f4dbe583f276f666a14065038061c9f624fbd00349f588975f2562d9fe52d14"
PARENT_SHA = "e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c"
SHARED_SHA = "dbc208fb7c858909ff6f04e70b939adaab1c3e8c27f8e72f261a620cf1b55e10"
MANIFEST_SHA = "828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8"
OLD_TABLE_SHA = "28c63e1ec4b7f0f8fbd72aec3e37c4324c9c5c9fda54ff36d32d83dddc619656"
SEEDS = [13088411117524848790, 10018485049408248344, 9083751155097190455]
REPS = [2701, 2802, 2903]
TASKS = ["past_volatility", "past_trend", "past_activity", "past_relation"]
OLD_TABLE = ROOT / "doc/results/separate_change_v1/table_rows.json"
OUT = ROOT / "doc/results/separate_change_native36_v1"
REPORT = ROOT / "doc/MARKET_SEPARATE_CHANGE_NATIVE36_RESULTS.md"
PARENT = ROOT / "output/runs/market-capacity-study/run-base-gryiAV"
OLD_V24 = ROOT / "output/runs/market-separate-changes/run-Lr58TR"
LEGEND = "MKT-HIST-2 · BTC/ETH/SOL January 2024 minute history, recent 128 past-only targets · complexity 4/5"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def relative(path):
    return str(Path(path).relative_to(ROOT))


def exclusive(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def dump(path, value):
    exclusive(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def asset(path):
    return {"source": relative(path), "sha256": sha(path), "bytes": path.stat().st_size}


def copy_exclusive(source, destination, expected):
    require(sha(source) == expected, f"changed source before capture: {source}")
    exclusive(destination, source.read_bytes())
    require(sha(destination) == expected, f"captured hash mismatch: {destination}")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate_report(report, run):
    require(report["dataset_id"] == "MKT-HIST-2" and report["card_sha256"] == CARD,
            "readout dataset/card mismatch")
    require(report["input_manifest_sha256"] == MANIFEST_SHA, "readout input manifest")
    require((report["training_rows"], report["validation_rows"]) == (112, 32), "readout split")
    heads = report["heads"]
    expected = {"ridge_penalty": 1, "neural_hidden": 16, "neural_learning_rate": .01,
                "neural_updates": 100, "outputs": 4, "loss": "masked TRAIN-target-standardized MSE",
                "device": "CPU (frozen readouts only)"}
    require(all(heads[k] == value for k, value in expected.items()), "fixed head recipe changed")
    parent_targets = load(PARENT / "readouts/report.json")["methods"][0]["repetitions"][0]["targets"]
    require([m["name"] for m in report["methods"]] ==
            ["initialized_v24_native36", "trained_v24_native36"], "only two new readout methods permitted")
    for method in report["methods"]:
        require((method["size"], method["ridge_parameters"], method["neural_parameters"],
                 method["ridge_fits"], method["ridge_task_fits"], method["neural_fits"],
                 method["training_feature_rows"], method["feature_training_supported"]) ==
                (36, 148, 660, 1, 4, 3, 112, True), "head dimension, support or fit counts")
        for role in ("training", "validation"):
            suffix = "train" if role == "training" else "validation"
            require(Path(method[role + "_path"]).resolve() ==
                    run / "features" / f"{method['name']}-{suffix}.pt", "new-bank path binding")
        require([rep["repetition"] for rep in method["repetitions"]] == REPS, "fixed repetitions")
        require([rep["actual_seed"] for rep in method["repetitions"]] == SEEDS, "exact uint64 seeds")
        for index, rep in enumerate(method["repetitions"]):
            require(type(rep["actual_seed"]) is int and rep["neural_updates_completed"] == 100,
                    "unsigned seed precision or neural updates")
            require(rep["ridge_fit_reused"] == (index != 0), "one ridge pipeline per view")
            require([t["name"] for t in rep["targets"]] == TASKS, "four declared targets")
            for target, parent_target in zip(rep["targets"], parent_targets):
                require(target["train_target_eligible"] == 112 and target["train_target_supported"]
                        and target["method_training_supported"], "training support changed")
                require(all(target[key] == parent_target[key] for key in
                            ("name", "unit", "train_target_mean", "train_target_scale")), "fixed target identity/scaling changed")
                for head in ("ridge", "neural"):
                    metrics = target[head]
                    require((metrics["valid"], metrics["total"], metrics["target_eligible"], metrics["coverage"])
                            == (32, 32, 32, 1), "full scoring coverage required")
                    require(all(finite(metrics[k]) for k in ("r2", "mae", "rmse")), "nonfinite metric")


def new_rows(report, report_path):
    rows = []
    for method in report["methods"]:
        label = TAG + (".init" if method["name"].startswith("initialized") else "")
        for index, task in enumerate(TASKS):
            targets = [rep["targets"][index] for rep in method["repetitions"]]
            ridge = targets[0]["ridge"]
            require(all(target["ridge"] == ridge for target in targets), "ridge repeated values changed")
            row = dict(dataset_id="MKT-HIST-2", task=task, recipe_id="separate-changes-native36-v1",
                       designed_complexity_level=4, complexity_scale_max=5,
                       evidence_stage="known-source development", view="intact chronological VALIDATION",
                       method=label, source_method=method["name"], reused=False, size=36,
                       linear_r2=ridge["r2"], linear_mae=ridge["mae"], linear_rmse=ridge["rmse"],
                       unit=targets[0]["unit"], valid=32, total=32, coverage=1,
                       report_path=relative(report_path), report_sha256=sha(report_path))
            for metric in ("r2", "mae", "rmse"):
                row[f"neural_{metric}_each"] = [t["neural"][metric] for t in targets]
                row[f"neural_mean_{metric}"] = statistics.mean(row[f"neural_{metric}_each"])
            row["repetitions"] = [{"repetition": rep["repetition"], "actual_seed": rep["actual_seed"],
                                   "neural_updates_completed": rep["neural_updates_completed"],
                                   "target": rep["targets"][index]} for rep in method["repetitions"]]
            rows.append(row)
    return rows


def main():
    started = time.perf_counter()
    require(Path("/.dockerenv").is_file() and ROOT == Path("/cuwacunu"), "use existing cuwacunu container")
    require(len(sys.argv) == 2, "usage: report-market-native36.py NEW_RUN_DIRECTORY")
    run = Path(sys.argv[1]).resolve()
    require(run.is_dir() and run.parent == ROOT / "output/runs/market-separate-changes-native36",
            "native36 run directory required")
    require(not any(path.exists() for path in [REPORT, OUT / "table_rows.json", OUT / "completion.json",
                                              OUT / "run", OUT / "sources"]), "exclusive report outputs already exist")
    require(sha(OLD_TABLE) == OLD_TABLE_SHA, "preserved native40 result table changed")
    require(sha(ROOT / "doc/MARKET_SEPARATE_CHANGE_NATIVE36_CARD.md") == CARD, "declaration changed")
    require(sha(Path("/opt/cuwacunu/build/market-study/market_fixed_regression")) == EVALUATOR,
            "fixed evaluator executable changed")
    require("PASS separate native36:" in (OUT / "engineering.log").read_text(), "engineering admission missing")
    summary, initial = load(run / "summary.json"), load(run / "initial-query.json")
    expected = {"kind": "market_separated_changes_native36_v1", "dataset": "MKT-HIST-2", "instance": TAG,
                "H": 128, "W": 64, "D": 36, "level_width": 32, "change_width": 4, "parameters": 297425,
                "optimizer_updated_parameters": 14904, "protected_parameters": 282521,
                "completed_updates": 2048, "attempted_batches": 2048, "window_presentations": 16384,
                "parent_updates": 2048, "probe_train_rows": 112, "probe_validation_rows": 32,
                "CPU_encoder_calls": 0, "cuda_learning_verified": True, "protected_level_exact": True,
                "protected_optimizer_states": 0, "initial_prefix_matches_saved_trained_v22": True,
                "final_prefix_matches_saved_trained_v22": True, "card_sha256": CARD,
                "parent_inference_sha256": PARENT_SHA, "shared_ledger_sha256": SHARED_SHA,
                "input_manifest_sha256": MANIFEST_SHA,
                "scaler_identity": "rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969"}
    require(all(summary[k] == value for k, value in expected.items()), "run contract mismatch")
    require(summary["nonzero_decoder_gradient_updates"] == 2048 and
            summary["nonzero_change_gradient_updates"] == 2047, "actual CUDA learning counts")
    for receipt in ("initialization-parity.json", "final-prefix-parity.json"):
        parity = load(run / receipt)
        require(parity == {"saved_v22_train_and_validation_prefix32_equal": True,
                           "old_encoder_replay_calls": 0, "old_head_refits": 0}, "exact saved prefix receipt")
    for filename, key in [("study.conf", "config_sha256"), ("encoder.conf", "settings_sha256"),
                          ("prospective-card.md", "card_sha256"), ("source-closure.sha256", "source_closure_sha256")]:
        require(sha(run / filename) == summary[key], f"run binding changed: {filename}")
    require(sha(Path("/opt/cuwacunu/build/market-study/market_separate_changes_native36")) == summary["binary_sha256"],
            "producer executable changed")
    for name in ["inference-0.pt", "inference-2048.pt", "training-512.pt", "training-1024.pt", "training-2048.pt"]:
        require((run / "checkpoints" / name).is_file(), "required saved checkpoint missing")
    parent = load(PARENT / "summary.json")
    for role in ("train", "validation"):
        for key in ("mae_train_scaled", "row_mae", "row_target_cells", "row_eligible_channels", "eligible_rows",
                    "total_rows", "target_cells", "target_history"):
            require(initial[role][key] == parent["trained_query"][role][key], "initial residual/parent query parity")
        final = summary["trained_query"][role]
        require(final["eligible_rows"] == final["total_rows"] == (112 if role == "train" else 32), "query coverage")
    with (run / "training.csv").open(newline="") as handle:
        training = list(csv.DictReader(handle))
    with (PARENT / "training.csv").open(newline="") as handle:
        previous_training = list(csv.DictReader(handle))
    require(len(training) == len(previous_training) == 2048, "training row count")
    stream_keys = ("attempt", "update", "eligible_examples", "target_cells", "row_seed", "mask_seed", "anchors")
    require(all(all(row[k] == old[k] for k in stream_keys) for row, old in zip(training, previous_training)),
            "parent row/anchor/mask streams differ")
    losses = [float(row["loss"]) for row in training]
    require(all(map(math.isfinite, losses)), "nonfinite saved loss")
    loss_summary = dict(first=losses[0], last=losses[-1], first64_mean=statistics.mean(losses[:64]),
                        last64_mean=statistics.mean(losses[-64:]))
    report_path = run / "readouts/report.json"
    readout = load(report_path)
    validate_report(readout, run)
    timing = load(run / "readout-timing.json")
    require(timing["kind"] == "saved_feature_readout_timing_v1" and
            timing["label"] == "fixed saved-feature fitting/scoring/I/O", "readout timing scope")
    require(type(timing["elapsed_ns"]) is int and timing["elapsed_ns"] > 0 and
            timing["finished_epoch_ns"] - timing["started_epoch_ns"] == timing["elapsed_ns"], "readout timing arithmetic")
    readout_seconds = timing["elapsed_ns"] / 1e9
    rows = []
    for original in load(OLD_TABLE)["rows"]:
        row = dict(original)
        row.update(reused=True, original_report_path=row.get("report_path"), original_report_sha256=row.get("report_sha256"),
                   report_path=relative(OLD_TABLE), report_sha256=OLD_TABLE_SHA)
        rows.append(row)
    rows.extend(new_rows(readout, report_path))
    lookup = {(row["method"], row["task"]): row for row in rows}
    criteria = []
    trend = lookup[TAG, "past_trend"]
    for name, key in [("positive_linear_trend", "linear_r2"), ("positive_neural_mean_trend", "neural_mean_r2")]:
        criteria.append(dict(criterion=name, value=trend[key], passed=trend[key] > 0))
    for task in ("past_volatility", "past_activity", "past_relation"):
        for key in ("linear_r2", "neural_mean_r2"):
            old, new = lookup["RPB-v22.market-02", task][key], lookup[TAG, task][key]
            criteria.append(dict(criterion=f"{task}_{key}_retention", baseline=old, candidate=new,
                                 difference=new - old, maximum_allowed_loss=.05, passed=new - old >= -.05))
    reference = lookup["RPB-v23.market-01", "past_trend"]["linear_r2"]
    require(reference == .3871309972495327, "full precision v23 reference changed")
    criteria.append(dict(criterion="linear_trend_exceeds_saved_v23", baseline=reference,
                         candidate=trend["linear_r2"], difference=trend["linear_r2"] - reference,
                         passed=trend["linear_r2"] > reference))
    passed = sum(item["passed"] for item in criteria)
    source_pins = []
    for line in (run / "source-closure.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and sha(ROOT / path) == digest, "source closure changed")
        source_pins.append((ROOT / path, digest))
    source_pins.append((Path(__file__).resolve(), sha(Path(__file__))))
    require(len({str(path) for path, _ in source_pins}) == len(source_pins), "duplicate source closure")
    shared = ROOT / "output/runs/market-capacity-study/shared-MKT-HIST-2-v1"
    require(sha(shared / "shared-assets.sha256") == SHARED_SHA, "shared asset ledger changed")
    for line in (shared / "shared-assets.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and sha(shared / path) == digest, "shared asset changed")
    run_assets = [asset(path) for path in sorted(run.rglob("*")) if path.is_file()]
    text_suffixes = {".json", ".csv", ".log", ".conf", ".md", ".sha256"}
    for item in run_assets:
        source = ROOT / item["source"]
        if source.suffix in text_suffixes:
            copy_exclusive(source, OUT / "run" / source.relative_to(run), item["sha256"])
    for source, digest in source_pins:
        copy_exclusive(source, OUT / "sources" / source.relative_to(ROOT), digest)
    training_rows = []
    for path in (PARENT, ROOT / "output/runs/market-consecutive-changes/run-hAxXvh",
                 ROOT / "output/runs/market-frozen-changes/run-tVEpLV", OLD_V24):
        record = load(path / "summary.json")
        training_rows.append(dict(encoder=record["instance"], updates=2048,
                                  inherited_updates=record.get("parent_updates", 0), reused=True,
                                  train_error=record["trained_query"]["train"]["mae_train_scaled"],
                                  validation_error=record["trained_query"]["validation"]["mae_train_scaled"],
                                  gpu_training_seconds=record.get("training_seconds_including_checkpoints",
                                                                record.get("segment_training_seconds_including_checkpoints"))))
    training_rows.extend([dict(encoder=TAG + ".init", updates=0, inherited_updates=2048, reused=False,
                               train_error=initial["train"]["mae_train_scaled"],
                               validation_error=initial["validation"]["mae_train_scaled"], gpu_training_seconds=None),
                          dict(encoder=TAG, updates=2048, inherited_updates=2048, reused=False,
                               train_error=summary["trained_query"]["train"]["mae_train_scaled"],
                               validation_error=summary["trained_query"]["validation"]["mae_train_scaled"],
                               gpu_training_seconds=summary["training_seconds_including_checkpoints"])])
    for row in training_rows:
        row.update(dataset_id="MKT-HIST-2", task="masked_reconstruction",
                   recipe_id="separate-changes-native36-report-v1", designed_complexity_level=4,
                   complexity_scale_max=5, view="fixed hidden queries: TRAIN and known VALIDATION",
                   error_metric="TRAIN-CF-scaled MAE", batch_size=8, device="cuda:0",
                   timing_includes_checkpoint_IO=True)
    tables = dict(kind="separate_change_native36_results_v1", status="complete", dataset_id="MKT-HIST-2",
                  instance=TAG, rows=rows, training_rows=training_rows, criteria=criteria, criteria_passed=passed,
                  criteria_total=9, development_goal_met=passed == 9, primary="RPB-v22", primary_changed=False,
                  D36_actual_seeds=SEEDS, D36_head_parameters=dict(linear=148, neural=660),
                  training_loss=loss_summary, parent_stream_parity=True)
    initial_trend = lookup[TAG + ".init", "past_trend"]
    disposition = "passes" if passed == 9 else "fails"
    lines = [f"# {TAG}: native36 market results", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
             f"**The candidate {disposition} the balanced development goal ({passed}/9). RPB-v22 remains primary.**", "",
             "One new CUDA trajectory: 2048 branch updates/B8 after 2048 inherited parent updates. Same 4096 TRAIN anchors,",
             "112 TRAIN/32 known-VALIDATION nonoverlapping readout windows and three symbols from one month.",
             "TRAIN Jan 1–21; known VALIDATION Jan 22–28; Jan 29–31 unused. All historical rows are reused; no older encoder",
             "quality run or head fit was repeated. New initial/final banks receive two ridge pipelines/eight target fits",
             "and six neural fits. CPU reads saved embeddings; no CPU encoder execution.", "",
             "Complete trained v22 native32 is preserved exactly, now with four separate learned change coordinates and",
             "a zero-start residual decoder. Native36 has 24 learned + 12 fixed coordinates; initialized new banks already",
             "contain the trained parent. Input[B,3,128,9], output[B,36]; no PCA after encoding. Native40 remains preserved.", "",
             "Ridge penalty 1; neural 16 tanh units, Adam .01/100 updates, repetitions 2701/2802/2903. D36 has 148/660 head",
             "parameters versus 132/596 at D32 and 164/724 at D40. Exact D36 seeds are",
             ", ".join(str(seed) for seed in SEEDS) + "; both new views share them under the unchanged width-derived rule.",
             "TRAIN-only feature/target scaling and all nine prospective preferences are unchanged. Width, inherited",
             "exposure and the residual decoder limit equal-budget causal comparisons. Three head fits on one split",
             "measure head variation, without independent chronological-cohort uncertainty.", "",
             "R²=1 is perfect, 0 equals a scoring-population constant mean, negative is worse. Coverage is 32/32 throughout.",
             "Raw widths include masks; full256 has longer context. All old initialized views are retained below and in JSON;",
             "the frozen-v23 warm start equals the registered trained v22 parent and is retained separately in JSON.", ""]
    display_methods = ["Raw data — no encoder (recent128)", "Raw data — no encoder (full256)",
                       "RPB-v22.market-02.init", "RPB-v22.market-02", "RPB-v23.market-01.init", "RPB-v23.market-01",
                       "RPB-v23.market-01.alt-01", "RPB-v24.market-01.init", "RPB-v24.market-01", TAG + ".init", TAG]
    for task in TASKS:
        lines.extend(["## " + task.replace("_", " ").capitalize(), "", LEGEND + " · intact known VALIDATION.", "",
                      "| Method | Size | Linear head R² | Neural head R² | Coverage % |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for method in display_methods:
            row = lookup[method, task]
            lines.append(f"| {method} | {row['size']} | {row['linear_r2']:.3f} | {row['neural_mean_r2']:.3f} | {100 * row['coverage']:.0f} |")
        lines.append("")
    lines.extend(["## Training and reconstruction", "", LEGEND + " · fixed hidden queries, TRAIN and known VALIDATION.", "",
                  "| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for row in training_rows:
        seconds = "—" if row["gpu_training_seconds"] is None else f"{row['gpu_training_seconds']:.2f}"
        lines.append(f"| {row['encoder']} | {row['updates']} | {row['train_error']:.3f} | {row['validation_error']:.3f} | {seconds} |")
    lines.extend(["", "Errors are standardized fixed-query MAE. Updates count new branch updates; frozen-v23, native40 and native36",
                  "also inherit 2048 parent updates. GPU training time includes checkpoint I/O and excludes capture/readouts.",
                  f"Native36 attempted {summary['attempted_batches']} batches with no skips; 16384 sampled presentations are four nominal 4096-window passes, not guaranteed epochs.",
                  f"Masked Huber loss first/last minibatch: {losses[0]:.6f}/{losses[-1]:.6f}; first 64/last 64 means: {loss_summary['first64_mean']:.6f}/{loss_summary['last64_mean']:.6f}.",
                  "These batches differ; fixed-query errors compare saved row/mask plans. Same row/anchor/mask streams match the parent exactly.",
                  f"There are 297425 parameters: 14904 new parameters update and 282521 remain protected. Decoder/projection nonzero-gradient updates: {summary['nonzero_decoder_gradient_updates']}/{summary['nonzero_change_gradient_updates']}.",
                  f"Peak GPU allocator allocation/reservation: {summary['cuda_peak_allocated_bytes']/1048576:.2f}/{summary['cuda_peak_reserved_bytes']/1048576:.2f} MiB.", "",
                  "## Changes, criteria and next direction", "",
                  f"Trend linear R² changes {initial_trend['linear_r2']:.6f}→{trend['linear_r2']:.6f}; mean-neural R² changes {initial_trend['neural_mean_r2']:.6f}→{trend['neural_mean_r2']:.6f}.",
                  "These initial/final measurements share head dimensions/seeds. They separate learning from information already supplied by the trained parent and initialized new path.",
                  "Per-task activity/volatility/relation changes and every neural repetition remain in the detailed JSON; do not infer one overall encoder accuracy.", "",
                  f"The predeclared goal {disposition}: **{passed} of 9** conditions pass.", ""])
    for task in ("past_volatility", "past_activity", "past_relation"):
        before, after = lookup[TAG + ".init", task], lookup[TAG, task]
        lines.append(f"{task.replace('_', ' ').capitalize()} initial→final R²: linear {before['linear_r2']:.6f}→{after['linear_r2']:.6f}; neural mean {before['neural_mean_r2']:.6f}→{after['neural_mean_r2']:.6f}.")
    native40_trend = lookup["RPB-v24.market-01", "past_trend"]
    lines.extend(["", f"Preserved native40 trend R² is {native40_trend['linear_r2']:.6f}/{native40_trend['neural_mean_r2']:.6f} (linear/neural mean); native36 is {trend['linear_r2']:.6f}/{trend['neural_mean_r2']:.6f}. This is a descriptive cross-width comparison, with no added post-result preference.", ""])
    for item in criteria:
        value = item.get("candidate", item.get("value"))
        detail = f"; difference {item['difference']:+.6f}" if "difference" in item else ""
        lines.append(f"- {item['criterion']}: {'PASS' if item['passed'] else 'FAIL'}, value {value:.6f}{detail}.")
    if passed == 9:
        next_direction = "Next: separately declare fresh chronological confirmation of unchanged native36 before considering a primary/serving decision."
    else:
        failed = ", ".join(item["criterion"] for item in criteria if not item["passed"])
        next_direction = "Next: inspect the saved per-task tradeoffs (" + failed + ") before declaring another architecture. Keep this frozen recipe stopped; no head, seed, rate or budget rescue."
    lines.extend(["", next_direction, "Keep RPB-v22 primary, useful v23 native32 and original v24 native40 trend checkpoints, plus every native36 initial/final artifact.",
                  "This is known-source historical representation development: no forecasting, TEST, fresh synthetic quality or independent chronological confirmation.",
                  "Exact parent-prefix preservation retains information, but extra head inputs and changed width-dependent initialization can still change fitted-head scores.", "",
                  "## Evidence and physical-unit errors", "",
                  "[Machine-readable rows](results/separate_change_native36_v1/table_rows.json) contain full-precision R², physical-unit MAE/RMSE, all three neural repetitions, uint64 seeds and all nine decisions.",
                  "Volatility units are logreturn; trend logreturn/tick; activity log1p(base_volume); relation correlation.",
                  "[Completion](results/separate_change_native36_v1/completion.json) hashes every local run artifact. Nonbinary evidence and exact sources are copied beside it; weights, optimizer states and banks remain local and ignored.",
                  "Exact initial/final parent32+validity+row IDs, protected values/no optimizer moments and actual CUDA learning pass. The prequality artificial CUDA fixture proves masking/gap/poison safety and typed next-update resume; it adds no market trajectory.",
                  f"Fixed saved-feature fitting/scoring/artifact I/O took {readout_seconds:.6f} seconds, measured separately around the unchanged evaluator. It is not pure head-fitting compute time.",
                  "Feature/query capture timings are retained separately in captured initial-query/summary JSON. Report/audit time is recorded in completion.", ""])
    dump(OUT / "table_rows.json", tables)
    exclusive(REPORT, ("\n".join(lines) + "\n").encode())
    completion = dict(kind="separate_change_native36_completion_v1", date="2026-10-14", run=relative(run), instance=TAG,
                      dataset_id="MKT-HIST-2", designed_complexity_level=4, completed_branch_updates=2048,
                      inherited_parent_updates=2048, attempted_batches=2048, batch_size=8, encoder_trajectories=1,
                      neural_fits=6, ridge_pipeline_fits=2, ridge_target_fits=8, old_encoder_quality_replays=0,
                      old_readout_refits=0, CPU_encoder_calls=0, protected_level_exact=True, protected_optimizer_states=0,
                      optimizer_updated_parameters=14904, protected_parameters=282521,
                      initial_and_final_prefix32_match_saved_trained_v22=True, parent_stream_parity=True,
                      parent_inference_sha256=PARENT_SHA, card_sha256=CARD, fixed_evaluator_binary_sha256=EVALUATOR,
                      source_closure_sha256=summary["source_closure_sha256"], binary_sha256=summary["binary_sha256"],
                      reporter_sha256=sha(Path(__file__)), old_table_sha256=OLD_TABLE_SHA,
                      report_sha256=sha(REPORT), table_rows_sha256=sha(OUT / "table_rows.json"),
                      primary_changed=False, forecasting_measured=False, test_measured=False,
                      criteria_passed=passed, criteria_total=9, head_fit_seconds=None,
                      saved_feature_readout_elapsed_ns=timing["elapsed_ns"],
                      saved_feature_readout_seconds_including_scoring_and_io=readout_seconds,
                      head_timing_note="elapsed evaluator time includes fitting, scoring and artifact I/O; pure fit compute unavailable",
                      saved_evidence_audit_seconds=time.perf_counter()-started, source_assets=run_assets,
                      captured_sources=[asset(source) for source, _ in source_pins])
    dump(OUT / "completion.json", completion)
    print(json.dumps({"status": "complete", "run": relative(run), "criteria_passed": passed,
                      "criteria_total": 9, "table_rows": len(rows), "run_artifacts": len(run_assets)}))


if __name__ == "__main__":
    main()
