#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Report frozen saved-feature transfer; stdlib only, no model/head execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time


MASTERS = (941101, 941102, 941103)
MODES = ("synthetic_train", "market_scaler")
VIEWS = ("validation_intact", "validation_deleted")
HEADS = ("ridge", "tiny_secondary")
REPETITIONS = (2701, 2802, 2903)
SEEDS36 = ("13088411117524848790", "10018485049408248344", "9083751155097190455")
MODEL = "RPB-v24.market-01.alt-01"
REFERENCE = "RPB-v22.alt-01"
CARD_HASH = "b0552d60cd50582a3bff213a2778e867ceabe1e20e52cd48e02aec4c34bea82b"
INPUT_HASH = "fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb"
CHECKPOINT_HASH = "a3939b13df8b1704fc3621876cf3adc6efaddd98951147ff43ec6408978fa98c"
DEPS = {
    "include/embedding/shared/types.h": "3c7987b9df5d936e009da619474af7ec08630083721281efb5523492200e2371",
    "include/embedding/shared/data.h": "b9a7b80f1df93f541d05453ebc6e6abde5de1a2144f8d5e03ad5f32a068cb92e",
    "include/embedding/shared/evaluation.h": "63b7064fec06d3e33ccca7e32b20ed604c7e97550a79117ff9bbf6b93283a270",
    "include/embedding/shared/feature_harness.h": "0823c7a8e72c55e9907a4d89b076393d6356cb495122c23c78a3086ba9062657",
    "include/embedding/shared/fixed_feature_readouts.h": "aea837bf3129044e2921695c4770082f07ae2f4ec7508b1ffeef535dfd52ce79",
    "src/data.cpp": "2a5b1b19baa0f67fac9020d91cd2883a35f7448a3e2fedb3d430b5928800043c",
    "src/feature_harness.cpp": "8cb2aaa6c2b0d6582441fe51fe04445479e15824a44003cf59b97a8f8a67abbc",
    "src/fixed_feature_readouts.cpp": "1de2800e99c52a3df6b2bc168cd7b4c900888162e0056d1c681c2e74c3396b2c",
}
TEXT_SUFFIXES = {".json", ".log", ".md", ".mk", ".h", ".cpp", ".sha256", ".py", ".sh", ".conf", ".txt"}
LEGEND = ("Dataset: **TEMPO-5 · complexity 5/5** · slow lead/lag beneath fast rhythms, "
          "nine grouped features and structured distractors.")


def check(value, message):
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(contents)


def save_json(path, value):
    save_new(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def close(left, right, message):
    check(math.isclose(left, right, abs_tol=1e-12, rel_tol=0), message)


def seed(repetition, width):
    mask = (1 << 64) - 1
    value = (repetition + 0x9e3779b97f4a7c15 * width) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return str(value ^ (value >> 31))


def ledger(path):
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            check(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "bad SHA256 ledger")
            result.append((digest, name))
    return result


def validate_recipe(report):
    check(report["protocol"] == "fixed-feature-readouts-v1", "unexpected fixed-head protocol")
    recipe = report["recipe"]
    for field, expected in {"ridge_penalty": 1, "tiny_hidden": 16, "tiny_steps": 100,
                            "tiny_learning_rate": .01, "validation_fits": 0,
                            "encoder_calls": 0, "pca_fits": 0}.items():
        check(recipe[field] == expected, "head recipe changed: " + field)
    check(recipe["probe_seed_policy"] == "stream_seed(repetition,width);same_equal_width_methods", "probe seed rule changed")


def validate_method(method, width):
    check(method["size"] == width and method["status"] == "measured", "method width or measured status mismatch")
    check(not method["inputs_train_prepared"] and method["outer_train_normalizer_fits"] == 1,
          "outer TRAIN feature normalization changed")
    check(len(method["repetitions"]) == 3, "three fixed readout repetitions required")
    for index, rep in enumerate(method["repetitions"]):
        check(rep["id"] == "rep-" + str(REPETITIONS[index]) and rep["status"] == "measured", "fit repetition mismatch")
        check(rep["actual_probe_seed_decimal"] == seed(REPETITIONS[index], width), "width-dependent seed mismatch")
        check(rep["ridge_parameters"] == 2 * width + 2 and rep["neural_parameters"] == 16 * (width + 3) + 2,
              "dimension-dependent binary head parameter count mismatch")
        for view in ("training",) + VIEWS:
            value = rep[view]
            rows, sources = (512, 256) if view == "training" else (256, 128)
            population = value["population"]
            check(population["total_rows"] == rows and population["total_source_groups"] == sources,
                  "declared readout population mismatch")
            for head in HEADS:
                score = value[head]
                check(score["total"] == rows and score["valid"] == population["valid_rows"], "score population mismatch")
                check(0 <= score["correct"] <= score["valid"] <= rows, "invalid correct/valid count")
                close(score["coverage"], score["valid"] / rows, "coverage arithmetic mismatch")
                close(score["accuracy"], score["correct"] / score["valid"], "accuracy arithmetic mismatch")
            if view != "training":
                for head, key in (("ridge", "ridge_grouped_interval"), ("tiny_secondary", "tiny_grouped_interval")):
                    interval = value[key]
                    check(interval["replicates"] == 1000 and interval["confidence"] == .95,
                          "saved confidence interval recipe changed")
                    close(interval["estimate"], value[head]["accuracy"], "confidence interval estimate mismatch")
                    check(0 <= interval["lower"] <= interval["upper"] <= 1, "invalid saved confidence interval")


def aggregate(methods):
    result = {}
    for view in VIEWS:
        result[view] = {}
        for head in HEADS:
            per_master = [statistics.mean(rep[view][head]["accuracy"] for rep in method["repetitions"]) for method in methods]
            result[view][head] = {"mean": statistics.mean(per_master), "master_means": per_master,
                                  "minimum": min(per_master), "maximum": max(per_master)}
        coverage = [statistics.mean(rep[view]["population"]["coverage"] for rep in method["repetitions"]) for method in methods]
        result[view]["coverage"] = {"mean": statistics.mean(coverage), "master_means": coverage,
                                    "minimum": min(coverage), "maximum": max(coverage)}
    return result


def pct(value):
    return f"{100 * value:.2f}"


def quality_table(mode, view, old, new):
    primary = "Primary input policy: exact saved synthetic TRAIN scaler" if mode == "synthetic_train" else "Secondary input policy: unchanged checkpoint market scaler"
    view_text = "intact VALIDATION" if view == "validation_intact" else "VALIDATION with exact saved additional 30% coordinate erasure"
    text = [LEGEND + " " + view_text + ".", "", primary + ". Equal three-cohort means; each neural score averages three fixed fits.", "",
            "| Method | Size | Linear head % | Neural head % | Coverage % |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for model, width, data in ((REFERENCE, 32, old), (MODEL, 36, new)):
        row = data[view]
        text.append(f"| {model} | {width} | {pct(row['ridge']['mean'])} | {pct(row['tiny_secondary']['mean'])} | {pct(row['coverage']['mean'])} |")
    return "\n".join(text)


def repetitions_table(title, methods):
    text = [title, "", LEGEND + " Known intact/deleted VALIDATION; all saved fixed fits.", "",
            "| Master | Fit | View | Linear % [95% interval] | Neural % [95% interval] | Coverage % |",
            "| ---: | ---: | --- | ---: | ---: | ---: |"]
    for master, method in zip(MASTERS, methods):
        for rep in method["repetitions"]:
            for view in VIEWS:
                row = rep[view]
                linear, neural = row["ridge_grouped_interval"], row["tiny_grouped_interval"]
                text.append(f"| {master} | {rep['id'][4:]} | {'intact' if view == VIEWS[0] else 'deleted'} | "
                            f"{pct(row['ridge']['accuracy'])} [{pct(linear['lower'])}, {pct(linear['upper'])}] | "
                            f"{pct(row['tiny_secondary']['accuracy'])} [{pct(neural['lower'])}, {pct(neural['upper'])}] | "
                            f"{pct(row['population']['coverage'])} |")
    return "\n".join(text)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--extract-binary", type=Path, default=Path("/opt/cuwacunu/build/market-transfer/market_synthetic_transfer_extract"))
    parser.add_argument("--read-binary", type=Path, default=Path("/opt/cuwacunu/build/market-transfer/market_synthetic_transfer_read"))
    args = parser.parse_args()
    check(Path("/.dockerenv").exists(), "Run this project reporter inside the existing managed development container")
    project = Path(__file__).resolve().parents[2]
    run = args.runroot.resolve(strict=True)
    inputs = (args.inputs or project / "output/inputs/market-synthetic-transfer-v1").resolve(strict=True)
    durable = project / "doc/results/market_synthetic_transfer_v1"
    report_path = project / "doc/MARKET_SYNTHETIC_TRANSFER_RESULTS.md"
    check(not any(path.exists() for path in (report_path, durable / "table_rows.json",
          durable / "completion.json", durable / "asset_hashes.json", durable / "receipts",
          durable / "sources")), "report outputs must be new exclusive paths")
    card = project / "doc/MARKET_SYNTHETIC_TRANSFER_CARD.md"
    check(sha(card) == CARD_HASH and sha(inputs / "manifest.json") == INPUT_HASH, "prospective card or staged input manifest changed")
    manifest = load(inputs / "manifest.json")
    check(tuple(manifest["masters"]) == MASTERS, "input cohorts changed")
    input_assets = []
    for member in manifest["members"]:
        file = inputs / member["path"]
        check(file.is_file() and file.stat().st_size == member["bytes"] and sha(file) == member["sha256"],
              "staged input changed: " + member["path"])
        check(member["original_manifest_verified"] or member["path"] == "controls/v22-summary.json",
              "unverified staged historical artifact")
        input_assets.append({"path": str(file), "sha256": member["sha256"], "bytes": member["bytes"]})

    execution = load(run / "verification.json")
    check(execution["status"] == "passed" and all(execution[key] == 0 for key in
          ("encoder_updates", "old_encoder_calls", "old_head_fits")), "combined execution audit failed")
    combined = load(run / "provenance-before.json")
    check(sha(run / "provenance-before.json") == execution["combined_provenance_sha256"]
          and len(combined) == execution["files_unchanged"], "combined provenance binding")
    for name, info in combined.items():
        file = Path(name)
        check(file.is_file() and file.stat().st_size == info["bytes"] and sha(file) == info["sha256"],
              "combined source/input/binary changed: " + name)
    feature_pins = load(run / "features-before-readout.json")
    check(len(feature_pins) == execution["saved_feature_files_unchanged"], "saved feature file count")
    for name, info in feature_pins.items():
        file = Path(name)
        check(file.is_file() and file.stat().st_size == info["bytes"] and sha(file) == info["sha256"],
              "saved feature changed after readout: " + name)
    features = load(run / "features/summary.json")
    check(features["checkpoint_sha256"] == CHECKPOINT_HASH and features["card_sha256"] == CARD_HASH,
          "frozen checkpoint or card identity mismatch")
    for key in ("encoder_updates", "decoder_calls", "head_calls", "CPU_encoder_calls", "scaler_refits"):
        check(features[key] == 0, "forbidden extractor work: " + key)
    for key in ("all_parameters_and_buffers_bitwise_unchanged", "all_parameter_gradients_undefined",
                "masked_sentinel_input_invariance", "source_hashes_unchanged"):
        check(features[key] is True, "failed frozen extraction witness: " + key)
    check(features["labels_passed_to_encoder"] is False and features["resampling"] is False,
          "extractor observation contract changed")
    check(features["batch_size"] == 8 and features["native_width"] == 36,
          "actual CUDA extraction geometry mismatch")
    check(features["total_CUDA_encoder_calls"] == 768 and features["total_rows"] == 6144,
          "unexpected encoder-call budget")
    check(len(features["runs"]) == 6, "exactly two modes across three cohorts required")
    combinations = set()
    for captured in features["runs"]:
        combination = (captured["mode"], captured["master"])
        check(combination in {(mode, master) for mode in MODES for master in MASTERS} and combination not in combinations,
              "unknown or duplicated extractor mode/cohort")
        combinations.add(combination)
        feature_path = run / "features" / captured["mode"] / ("master-" + str(captured["master"])) / "features.pt"
        check(sha(feature_path) == captured["features_sha256"], "saved embeddings changed after extraction")
        check([role["role"] for role in captured["roles"]] == ["training", "validation", "deleted"], "extraction role mismatch")
        for role in captured["roles"]:
            rows = 512 if role["role"] == "training" else 256
            check(role["rows"] == rows and role["CUDA_encoder_calls"] == rows // 8 and role["masked_cells"] > 0,
                  "actual masked CUDA extraction witness missing")

    before = run / "features/sources-before.sha256"
    after = run / "features/sources-after.sha256"
    check(before.read_bytes() == after.read_bytes(), "extraction source ledgers differ")
    pinned = ledger(before)
    proc_entries = 0
    source_records = []
    for digest, name in pinned:
        if name.startswith("/proc/") and name.endswith("/exe"):
            proc_entries += 1
            check(digest == features["binary_sha256"], "captured running executable digest mismatch")
            continue
        file = Path(name) if Path(name).is_absolute() else project / name
        check(file.is_file() and sha(file) == digest, "pinned input/source changed since extraction: " + name)
        source_records.append((file.resolve(), digest))
    check(proc_entries == 1 and sha(args.extract_binary) == features["binary_sha256"], "actual persistent extraction binary mismatch")
    source_records = list({file: digest for file, digest in source_records +
        [(Path(name).resolve(), info["sha256"]) for name, info in combined.items()]}.items())

    dependency_pairs = []
    deps_root = project / "code/evaluation/market_synthetic_transfer_v1/deps"
    for name, original_hash in DEPS.items():
        file = deps_root / name
        check(sha(file) == original_hash, "copied fixed-head arithmetic dependency changed: " + name)
        dependency_pairs.append({"integration_copy": str(file), "embedding_original_relative": "code/shared/" + name,
                                 "original_sha256_pinned_at_copy": original_hash, "copied_sha256": original_hash})

    reader = load(run / "readouts/reader-summary.json")
    for key in ("encoder_calls", "encoder_updates", "old_encoder_calls", "old_head_fits", "post_encoder_pca_fits"):
        check(reader[key] == 0, "forbidden readout work: " + key)
    check(reader["primary_input_mode"] == MODES[0] and reader["secondary_input_mode"] == MODES[1], "input policy selection changed")
    check(reader["model"] == MODEL and reader["native_width"] == 36 and reader["device"] == "cpu",
          "reader model, width or head device changed")
    check(reader["training_rows_per_cohort"] == 512 and reader["validation_rows_per_cohort"] == 256 and
          reader["training_sources_per_cohort"] == 256 and reader["validation_sources_per_cohort"] == 128,
          "reader declared split/source counts changed")
    old_summary = load(inputs / "controls/v22-summary.json")
    old_methods, new_methods = [], {mode: [] for mode in MODES}
    old_raw, new_raw = [], []
    for master in MASTERS:
        old_report = load(inputs / "controls" / ("master-" + str(master)) / "report.json")
        new_report = load(run / "readouts" / ("master-" + str(master)) / "report.json")
        for value in (old_report, new_report):
            validate_recipe(value)
            check(int(value["master_seed"]) == master, "readout cohort mismatch")
        old_method = next(method for method in old_report["methods"] if method["method"] == "native_512")
        validate_method(old_method, 32)
        old_methods.append(old_method)
        old_raw.append(old_report)
        new_raw.append(new_report)
        check({method["method"] for method in new_report["methods"]} == set(MODES) and not new_report["pairs"],
              "unexpected new methods or comparison fits")
        check(new_report["fit_counts"] == {"outer_train_normalizer_fits": 2, "ridge_fits": 6, "tiny_fits": 6,
                                            "validation_fits": 0, "encoder_calls": 0, "pca_fits": 0}, "readout fit count mismatch")
        for mode in MODES:
            method = next(method for method in new_report["methods"] if method["method"] == mode)
            validate_method(method, 36)
            check(tuple(rep["actual_probe_seed_decimal"] for rep in method["repetitions"]) == SEEDS36, "declared native36 seed mismatch")
            for rep in method["repetitions"]:
                fit_dir = run / "readouts" / ("master-" + str(master)) / mode / rep["id"]
                for name in ("fit.pt", "training-predictions.pt", "validation-intact-predictions.pt", "validation-deleted-predictions.pt"):
                    check((fit_dir / name).is_file(), "missing saved fit/prediction: " + str(fit_dir / name))
            new_methods[mode].append(method)
    old_scores = aggregate(old_methods)
    for view in VIEWS:
        for item in HEADS + ("coverage",):
            original = old_summary["combined_scores"]["candidate_full"][view][item]
            close(old_scores[view][item]["mean"], original["mean"], "reused old mean differs from saved summary")
            for actual, expected in zip(old_scores[view][item]["master_means"], original["master_means"]):
                close(actual, expected, "reused old cohort mean differs from saved summary")
    scores = {mode: aggregate(new_methods[mode]) for mode in MODES}
    retention = []
    for view in VIEWS:
        for head in HEADS:
            old_value, new_value = old_scores[view][head]["mean"], scores[MODES[0]][view][head]["mean"]
            retention.append({"view": view, "head": head, "old_mean": old_value, "new_mean": new_value,
                              "new_minus_old": new_value - old_value, "minimum_allowed_delta": -.05,
                              "pass": new_value - old_value >= -.05 - 1e-12})
    full_coverage = all(rep[view]["population"]["coverage"] == 1 for mode in MODES
                        for method in new_methods[mode] for rep in method["repetitions"] for view in VIEWS)
    comparable = all(condition["pass"] for condition in retention) and full_coverage
    disposition = "primary-transfer-comparable" if comparable else "primary-transfer-retention-fails"

    # Preserve all small original receipts and source snapshots. Binary tensor
    # assets stay in the ignored local run; every one is bound by SHA256 below.
    durable.mkdir(parents=True, exist_ok=True)
    def copy_text(source, target):
        if source.suffix.lower() not in TEXT_SUFFIXES and source.name != "Makefile":
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        check(not target.exists(), "snapshot destination exists")
        shutil.copyfile(source, target)
        check(sha(source) == sha(target), "snapshot copy changed bytes")
    for root, name in ((run, "run"), (inputs, "inputs")):
        for file in sorted(root.rglob("*")):
            check(not file.is_symlink(), "symlink in frozen evidence")
            if file.is_file():
                copy_text(file, durable / "receipts" / name / file.relative_to(root))
    sources = {file for file, _ in source_records if file.is_relative_to(project) and not file.is_relative_to(project / "output")}
    sources.update(file for file in (project / "code/evaluation/market_synthetic_transfer_v1").rglob("*") if file.is_file())
    sources.update((card, Path(__file__).resolve(), project / "AGENTS.md"))
    for file in sorted(sources):
        copy_text(file, durable / "sources" / file.relative_to(project))
    run_assets = [{"path": str(file.relative_to(run)), "sha256": sha(file), "bytes": file.stat().st_size}
                  for file in sorted(run.rglob("*")) if file.is_file()]
    save_json(durable / "asset_hashes.json", {"run_root": str(run), "run_assets": run_assets,
              "staged_input_assets": input_assets, "source_pins": [{"path": str(file), "sha256": digest} for file, digest in source_records],
              "dependency_copy_pairs": dependency_pairs, "extract_binary_sha256": sha(args.extract_binary),
              "read_binary_sha256": sha(args.read_binary), "card_sha256": CARD_HASH, "input_manifest_sha256": INPUT_HASH})
    panels = []
    for mode in MODES:
        for view in VIEWS:
            panels.append({"dataset_id": "TEMPO-5", "task": "slow timing", "recipe_id": "tempo-5-slow-fast-nine-feature-v1",
                           "designed_complexity_level": 5, "complexity_scale_max": 5, "view": view,
                           "input_policy": mode, "policy_role": "primary" if mode == MODES[0] else "secondary",
                           "head_fitting_dataset_id": "TEMPO-5", "scoring_dataset_id": "TEMPO-5",
                           "rows": [{"method": model, "size": width, "evidence": evidence,
                                     "encoder_fitting_dataset_id": fitting,
                                     "linear_head_accuracy": data[view]["ridge"]["mean"],
                                     "neural_head_accuracy": data[view]["tiny_secondary"]["mean"],
                                     "coverage": data[view]["coverage"]["mean"]}
                                    for model, width, evidence, fitting, data in
                                    ((REFERENCE, 32, "reused_saved_scores", "TEMPO-5", old_scores),
                                     (MODEL, 36, "new_frozen_checkpoint_transfer", "MKT-HIST-2", scores[mode]))]})
    save_json(durable / "table_rows.json", {"protocol": "market-synthetic-transfer-v1", "status": "complete", "disposition": disposition,
              "panels": panels, "reference": old_scores, "new_scores": scores, "retention_conditions": retention,
              "full_coverage_all_modes": full_coverage, "all_saved_repetitions_and_intervals":
              {"new": new_raw, "reused_reference": old_raw}, "encoder_updates": 0})

    intact, deleted = scores[MODES[0]][VIEWS[0]], scores[MODES[0]][VIEWS[1]]
    lead = ("The frozen market-trained encoder **does transfer well on TEMPO-5** under the declared retention criterion."
            if comparable else "The frozen market-trained encoder **does not transfer well enough on TEMPO-5 to match the specialized synthetic reference** under the declared criterion.")
    content = ["# Frozen market encoder on hard synthetic signals", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
               lead, "", f"With the primary synthetic TRAIN input normalization, intact linear/neural accuracy is **{pct(intact['ridge']['mean'])}%/{pct(intact['tiny_secondary']['mean'])}%**; "
               f"with saved additional 30% erasure it is **{pct(deleted['ridge']['mean'])}%/{pct(deleted['tiny_secondary']['mean'])}%**. "
               f"{sum(condition['pass'] for condition in retention)}/4 retention comparisons pass. Coverage across both declared preprocessing policies is {'100%' if full_coverage else 'below 100%' }.", "",
               f"This is a **new frozen-checkpoint transfer test** of {MODEL}, the exact native36 market-trained instance. "
               f"The {REFERENCE} native32 comparison is **reused synthetic-trained evidence**, not a replay or new encoder training. "
               "The market checkpoint is identical across all three known synthetic cohorts; they are three data cohorts, not three new encoder trajectories.", "",
               "Each cohort reuses 512 TRAIN examples from 256 paired source groups and 256 VALIDATION examples from 128 disjoint groups. "
               "C3/H128/F9 observations, intact/deleted masks and saved TRAIN scalers are unchanged. CUDA:0 float32 inference uses batch8. "
               "Native36 reaches the fixed heads directly, with no post-encoder PCA. Ridge penalty1 and neural16-tanh/Adam.01/100 steps use repetitions2701/2802/2903. "
               "The width-derived seed policy is unchanged; native36 has74/626 binary classifier parameters versus66/562 at native32. "
               "Each TRAIN fit scores both validation views unchanged:18 new ridge and18 new neural fits; zero old-control fits.", ""]
    for mode in MODES:
        for view in VIEWS:
            content.extend((quality_table(mode, view, old_scores, scores[mode]), ""))
    content.extend(("## Declared retention decisions", "", LEGEND + " Primary synthetic_train policy; intact and deleted VALIDATION.", "",
                    "| View | Head | Change versus synthetic v22 (percentage points) | At least −5 points |",
                    "| --- | --- | ---: | --- |"))
    for condition in retention:
        content.append(f"| {'intact' if condition['view'] == VIEWS[0] else 'deleted'} | {'linear' if condition['head'] == 'ridge' else 'neural'} | "
                       f"{100 * condition['new_minus_old']:+.2f} | {'pass' if condition['pass'] else 'fail'} |")
    content.extend(("", "The secondary market_scaler panels describe the unchanged market input pipeline on synthetic units. "
                    "They were declared before scoring and cannot replace the primary view or rescue a failed retention condition.", "",
                    "## What changed and verification", "", "Only the input source and external input-scaler policy changed. No encoder/decoder training occurred: **0 new encoder updates**, "
                    "after the checkpoint's inherited2048 parent updates plus2048 change-branch updates. No synthetic target reached the encoder. "
                    "The trained weights/buffers remain bitwise exact, all parameter gradients remain undefined, masks/values/native36 exports are verified on CUDA, "
                    "and hidden sentinels leave normalized encoder inputs exact. No decoder calls, CPU encoder calls, TEST access, resampling, new data or mask generation occurred.", "",
                    f"CUDA frozen extraction took **{features['inference_seconds']:.3f} seconds** including input normalization, device transfers, mask checks and state-independent inference capture, "
                    "excluding separate saved-file serialization. "
                    f"Fixed CPU head fitting/scoring and its output I/O took **{reader['head_fitting_and_scoring_seconds_including_io']:.3f} seconds**. "
                    "Reporting/hash-audit time is recorded separately in completion.json. There is no new encoder training error or training-time table.", "",
                    "Checkpoint SHA256: `" + CHECKPOINT_HASH + "`. Prospective card SHA256: `" + CARD_HASH + "`. "
                    "The exact staged-data manifest, checkpoint/settings/card/source pre/post ledgers, running extraction binary identity and eight byte-identical fixed-head dependency pairs passed. "
                    "Saved embeddings, fitted heads and prediction tensors remain local and ignored; every run asset has an availability/hash record in "
                    "[asset_hashes.json](results/market_synthetic_transfer_v1/asset_hashes.json). "
                    "All nonbinary receipts and source snapshots are preserved under the same evidence directory.", "",
                    "## Cohorts, repetitions and uncertainty", "", "The following are all three fixed readout fits on each known cohort, including the unchanged saved old-control fits. "
                    "Intervals are the original95% source-group percentile bootstrap with1000 replicates, conditional on each fitted readout. "
                    "They describe within-cohort sampling, not independent model training or unseen-domain uncertainty. "
                    "Per-cohort means and all full-precision populations, scores, seeds and intervals are also retained in "
                    "[table_rows.json](results/market_synthetic_transfer_v1/table_rows.json).", "",
                    repetitions_table("Primary synthetic_train — " + MODEL, new_methods[MODES[0]]), "",
                    repetitions_table("Secondary market_scaler — " + MODEL, new_methods[MODES[1]]), "",
                    repetitions_table("Reused synthetic-trained reference — " + REFERENCE, old_methods), "",
                    "## Next direction", ""))
    content.append("Keep the frozen market checkpoint and use this direct transfer evidence when choosing representations. "
                   + ("The next useful check is a separately declared transfer test on another compatible hard signal family; this result establishes timing transfer on TEMPO-5 only."
                      if comparable else "The next useful step is a saved-TRAIN-only diagnosis of which native coordinate groups retain the slow timing relation under each fixed input policy, with no encoder updates or head refits.")
                   + " Historical TEMPO-4/AMP-2 use different C/H/F shapes and remain unmeasured for this checkpoint. "
                   "No automatic promotion follows; RPB-v22 remains primary and all historical artifacts remain preserved.")
    save_new(report_path, "\n".join(content) + "\n")
    completion = {"protocol": "market-synthetic-transfer-v1", "status": "complete", "disposition": disposition,
                  "model_tag": MODEL, "masters": list(MASTERS), "new_encoder_updates": 0,
                  "inherited_parent_updates": 2048, "inherited_change_branch_updates": 2048,
                  "CUDA_encoder_calls": features["total_CUDA_encoder_calls"], "CPU_encoder_calls": 0,
                  "new_ridge_fits": 18, "new_neural_fits": 18, "old_control_head_fits": 0,
                  "primary_retention_conditions_passed": sum(value["pass"] for value in retention),
                  "primary_retention_conditions_total": 4, "full_coverage_all_modes": full_coverage,
                  "inference_seconds": features["inference_seconds"],
                  "head_fitting_and_scoring_seconds_including_io": reader["head_fitting_and_scoring_seconds_including_io"],
                  "reporting_and_hash_audit_seconds": time.monotonic() - started,
                  "report_sha256": sha(report_path), "table_rows_sha256": sha(durable / "table_rows.json"),
                  "asset_hashes_sha256": sha(durable / "asset_hashes.json"), "card_sha256": CARD_HASH,
                  "input_manifest_sha256": INPUT_HASH, "checkpoint_sha256": CHECKPOINT_HASH}
    save_json(durable / "completion.json", completion)
    receipts = [{"path": str(file.relative_to(durable)), "sha256": sha(file), "bytes": file.stat().st_size}
                for file in sorted(durable.rglob("*")) if file.is_file()]
    save_json(durable / "durable_files.json", receipts)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
