#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Append one verified v26 record; preserve historical records and primary selection.

Run once, after the v26 reporter, in the existing cuwacunu container. This stage
records a failed market mechanism or a market pass awaiting separate transfer.
It never promotes the balanced reference. Mirror the resulting two documents
locally to embedding through the controlling task's host orchestration.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TAG = "RPB-v26.market-01"
PROTOCOL = "market-conditioned-changes-v1"
CARD_SHA = "66de5fbe159230c82c04bd45a0d48cdf0694e101a082319da2f8bf2bd8b10e84"
OLD_TABLE_SHA = "7d11f577beb61b924a9b896fe3f5dea9c464b779fc34bde19f99ef80751109de"
TASKS = ["past_volatility", "past_trend", "past_activity", "past_relation"]
HEADS = ["linear_r2", "neural_mean_r2"]
OLD_MODELS = ["RPB-v22.market-02", "RPB-v23.market-01", "RPB-v23.market-01.alt-01",
              "RPB-v24.market-01", "RPB-v24.market-01.alt-01", "RPB-v25.market-01"]
CORE = [model for model in OLD_MODELS if model != "RPB-v23.market-01.alt-01"] + [TAG]
LEGEND = "MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def asset(path):
    return dict(repository="integration", path=relative(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def verify_asset(item):
    name = item.get("path", item.get("source"))
    require(isinstance(name, str), "Missing asset path")
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
            "Evidence asset changed: " + name)
    return path


def exclusive(path, content):
    with path.open("xb") as stream:
        stream.write(content)


def table(lines, headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines.extend(["| " + " | ".join(map(cell, headers)) + " |",
                  "| " + " | ".join("---" if j == 0 else "---:" for j in range(len(headers))) + " |"])
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    lines.append("")


def maxima(methods, rows, scope):
    result = []
    for task in TASKS:
        for head in HEADS:
            best = max(rows[model, task][head] for model in methods)
            winners = [model for model in methods if rows[model, task][head] == best]
            result.append(dict(task=task, head=head, method=winners[0], tied_methods=winners,
                               size=rows[winners[0], task]["size"], r2=best,
                               coverage=rows[winners[0], task]["coverage"], scope=scope))
    return result


def pareto(methods, rows):
    values = {model: [rows[model, task][head] for task in TASKS for head in HEADS] for model in methods}
    return [model for model in methods if not any(
        other != model and all(a >= b for a, b in zip(values[other], values[model])) and
        any(a > b for a, b in zip(values[other], values[model])) for other in methods)]


def markdown(ledger, market_pass, before_json, before_md):
    entries = {entry["model"]: entry for entry in ledger["candidate_entries"]}
    rows = {(entry["model"], row["task"]): row for entry in entries.values() for row in entry["final_task_rows"]}
    v26 = entries[TAG]
    outcome = ("passes the full market gate; compatible frozen TEMPO-5 transfer remains pending"
               if market_pass else "fails the full market gate and its exact mechanism is stopped")
    lines = ["# Encoder search ledger", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
             f"{TAG} {outcome}. RPB-v22 remains primary and RPB-v24.market-01.alt-01 remains the balanced research reference.",
             "All six earlier candidate records are preserved semantically unchanged. V25 and every task/head reference remain available; no universal winner is inferred from selected task maxima.", "",
             "This ledger reuses verified saved measurements. Updating it calls no encoder, deserializes no tensor, fits no head/scaler and performs no Git commit, push or publication.",
             "The JSON and this Markdown are mirrored identically into both repositories. Every evidence path resolves from C:/Work/cuwacunu/cuwacunu_dev; local ignored checkpoints are absent from a Git clone.", "",
             "## Matched final checkpoints", "", LEGEND + " · intact known chronological VALIDATION.",
             "Same C3/H128/F9/P8/W64, original TRAIN CF scaler,4096 legal TRAIN anchors and112 TRAIN/32 known-VALIDATION probes. TRAIN Jan1–21; VALIDATION Jan22–28; Jan29–31 unused.",
             "Core scope: matched v22 parent, fresh v23, native40, native36, v25 and v26. Native widths, width-derived head seeds/costs and inherited exposure differ; this is descriptive development evidence.",
             "V26 applies frozen TRAIN coordinate means/population SDs to the unchanged generic change bank before projection; parent32 and waveform objective remain protected. No PCA or added task labels.", ""]
    for task in TASKS:
        lines.extend(["### " + task.replace("_", " "), "", LEGEND + " · " + task.replace("_", " ") + " · intact known VALIDATION.", ""])
        table(lines, ["Method", "Size", "Linear head R²", "Neural head R²", "Coverage %"],
              [(model, rows[model, task]["size"], f"{rows[model, task]['linear_r2']:.6f}",
                f"{rows[model, task]['neural_mean_r2']:.6f}", f"{100 * rows[model, task]['coverage']:.0f}") for model in CORE])
    lines.extend(["## Own frozen decisions and exposure", "", LEGEND + " · each card keeps its own declared criteria; no retroactive rescore.", ""])
    decision_rows = []
    for model in OLD_MODELS + [TAG]:
        entry, score = entries[model], entries[model]["own_original_criteria"]
        count = "No nine-condition score" if score.get("total") is None else f"{score['passed']}/{score['total']}"
        budget = entry["encoder_budget"]
        decision_rows.append((model, entry["input_geometry"]["native_width"], count,
                              f"{budget['new_updates']}/{budget['inherited_parent_updates']}", entry["status"]))
    table(lines, ["Method", "Native size", "Own conditions", "New/inherited updates", "Status"], decision_rows)
    own = v26["own_original_criteria"]
    lines.extend([f"V26 strict native36 protection: {own['strict_passed']}/8; all three final neural trend fits positive: {own['all_three_trend_neural_positive']}; full32/32 coverage: {own['full_coverage']}.",
                  "Every trajectory uses B8 on RTX A2000 8GB/cuda:0. Three neural repetitions on one encoder/split are head variation, not independent market cohorts. Old cards retain parent N/A, v23 eight-condition counts and native40/native36/v25 outcomes.", "",
                  "## Reconstruction and training cost", "", LEGEND + " · fixed hidden TRAIN/known-VALIDATION query MAE.", ""])
    training = [entries[model]["encoder_budget"]["training_and_reconstruction"] for model in OLD_MODELS + [TAG]]
    table(lines, ["Encoder", "Updates", "Train error ↓", "Validation error ↓", "GPU training seconds"],
          [(row["encoder"], row["updates"], f"{row['train_error']:.6f}", f"{row['validation_error']:.6f}",
            "—" if row["gpu_training_seconds"] is None else f"{row['gpu_training_seconds']:.3f}") for row in training])
    lines.extend(["Errors are original fixed-query TRAIN-CF-scaled MAE. GPU training includes checkpoint I/O; inherited parent exposure remains separate. Calibration, extraction, fixed saved-head fitting/scoring/I/O and independent audit are separately timed in the linked v26 result.", "",
                  "## Fixed head costs and task references", "", LEGEND + " · TRAIN-only feature/target normalization; unchanged ridge1 and neural16-tanh/Adam0.01/100 updates; repetitions2701/2802/2903.", ""])
    heads = {}
    for model in OLD_MODELS + [TAG]:
        entry = entries[model]
        heads[entry["input_geometry"]["native_width"]] = entry["head_recipe"]
    table(lines, ["Native size", "Ridge parameters", "Neural parameters", "Exact uint64 seeds for2701/2802/2903"],
          [(width, recipe["ridge_parameters"], recipe["neural_parameters"], ", ".join(recipe["uint64_actual_seeds_decimal"]))
           for width, recipe in sorted(heads.items())])
    lines.extend([LEGEND + " · core six final checkpoints, intact known VALIDATION · descriptive task/head maxima.", ""])
    table(lines, ["Task", "Head", "Method", "R²"],
          [(item["task"], item["head"], ", ".join(item["tied_methods"]), f"{item['r2']:.17g}")
           for item in ledger["leader_views"]["current_core_per_task_head"]])
    lines.extend(["Core eight-axis Pareto frontier: " + ", ".join(ledger["leader_views"]["current_core_eight_axis_Pareto_front"]) + ". Exact saved comparisons are descriptive and confer no balanced selection.",
                  "The frozen-parent v23 alternative remains separate from the core six. Its activity linear R²0.7714374863251234 and all other scores remain preserved even if another record later exceeds them.", "",
                  LEGEND + " · all seven preserved final records, including frozen-v23 alternative · intact known VALIDATION.", ""])
    table(lines, ["Task", "Head", "Method", "R²"],
          [(item["task"], item["head"], ", ".join(item["tied_methods"]), f"{item['r2']:.17g}")
           for item in ledger["leader_views"]["current_all_recorded_per_task_head"]])
    lines.extend(["## Protection and next direction", "",
                  "Candidate records are append-only. Corrections receive linked records; earlier leader views and both preceding live ledger files are preserved. Per-task metrics, physical errors, every neural repetition, cards/reports and all checkpoint hashes remain in the JSON/evidence.",
                  "A balanced-pointer change needs all original9, strict native36 eight floors, all three trend repetitions positive, full coverage and separately verified compatible TEMPO-5 retention under both unchanged scaler policies. This market-only append stage changes no balanced or primary pointer.",
                  "V25 stronger scores remain informational references. No stopped recipe receives normalizer/head/rate/width/seed/budget rescue. No failed-market synthetic transfer; TEMPO-4/AMP-2 stay unmeasured for this checkpoint.",
                  ("Next measure the exact frozen v26 checkpoint on the declared TEMPO-5 cohorts with its new bank normalizer frozen; no transfer or new balanced selection is recorded yet."
                   if market_pass else "Stop the exact v26 conditioned-bank recipe, preserve its outcomes and declare any different justified mechanism prospectively. Continue the search with the retained balanced/per-task references."),
                  "Fresh chronology remains separate confirmation. No forecasting, TEST or universal-best claim follows.", "",
                  "## Evidence", "",
                  "- [Exact machine ledger](encoder_search_ledger.json)",
                  "- [V26 frozen result](C:/Work/cuwacunu/cuwacunu_dev/doc/MARKET_CONDITIONED_CHANGE_RESULTS.md)",
                  "- [V26 prospective card](C:/Work/cuwacunu/cuwacunu_dev/doc/MARKET_CONDITIONED_CHANGE_CARD.md)",
                  "- [V25 preserved result](C:/Work/cuwacunu/cuwacunu_dev/doc/MARKET_CHANGE_ENERGY_RESULTS.md)",
                  "- [Native36 balanced result](C:/Work/cuwacunu/cuwacunu_dev/doc/MARKET_SEPARATE_CHANGE_NATIVE36_RESULTS.md)",
                  "- [Native36 frozen TEMPO-5 transfer](C:/Work/cuwacunu/cuwacunu_dev/doc/MARKET_SYNTHETIC_TRANSFER_RESULTS.md)",
                  f"- Previous JSON: {before_json['path']} SHA256 {before_json['sha256']}",
                  f"- Previous Markdown: {before_md['path']} SHA256 {before_md['sha256']}", "",
                  "All35 candidate checkpoints were hash-checked and remain Git-ignored during this ledger update. No tensor was deserialized or exported. Existing preservation receipts bind historical scans; this ledger performs no broad scan. The unrelated roadmap remains untouched."])
    return ("\n".join(lines).rstrip() + "\n").encode()


def main():
    require(Path("/.dockerenv").is_file() and ROOT == Path("/cuwacunu"), "Use existing managed cuwacunu container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "doc/results/market_conditioned_changes_v1")
    parser.add_argument("--expected-ledger-sha256", required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    require(evidence == ROOT / "doc/results/market_conditioned_changes_v1", "Declared v26 evidence directory required")
    ledger_path, md_path = ROOT / "doc/encoder_search_ledger.json", ROOT / "doc/ENCODER_SEARCH_LEDGER.md"
    require(sha(ledger_path) == args.expected_ledger_sha256, "Live ledger changed since controlling-task review")
    before_bytes, md_bytes = ledger_path.read_bytes(), md_path.read_bytes()
    ledger = json.loads(before_bytes)
    old_entries = copy.deepcopy(ledger["candidate_entries"])
    snapshot = load(evidence / "search-ledger-before.json")
    require([entry["model"] for entry in old_entries] == OLD_MODELS and
            old_entries == snapshot["candidate_entries"], "All six historical candidate records must remain semantically exact")
    require(ledger["live"]["primary_architecture"] == "RPB-v22" and
            ledger["live"]["leading_balanced_research_reference"] == "RPB-v24.market-01.alt-01" and
            ledger["live"]["verified_new_nine_of_nine_winner"] is None, "Unexpected prior primary/balanced selection")
    require(sha(ROOT / "doc/MARKET_CONDITIONED_CHANGE_CARD.md") == CARD_SHA and
            sha(ROOT / "doc/results/market_change_energy_v1/table_rows.json") == OLD_TABLE_SHA, "Frozen declaration/control changed")
    receipt = evidence / "search-ledger-update.json"
    backups = [evidence / "ledger-update-before.json", evidence / "ledger-update-before.md"]
    staged = [evidence / "ledger-update-generated.json", evidence / "ledger-update-generated.md"]
    require(not any(path.exists() for path in [receipt] + backups + staged), "Preserve prior ledger update attempt")
    completion_path, candidate_path, table_path = [evidence / name for name in ("completion.json", "candidate-record.json", "table_rows.json")]
    completion, candidate, tables = load(completion_path), load(candidate_path), load(table_path)
    report_path = ROOT / "doc/MARKET_CONDITIONED_CHANGE_RESULTS.md"
    require(completion["status"] == tables["status"] == "complete" and completion["instance"] == candidate["model"] == tables["instance"] == TAG and
            completion["kind"] == "market_conditioned_change_completion_v1" and completion["card_sha256"] == candidate["card_sha256"] == CARD_SHA,
            "Verified v26 completion/candidate required")
    for path, key in ((table_path, "table_rows_sha256"), (candidate_path, "candidate_record_sha256"),
                      (report_path, "report_sha256"), (evidence / "asset_hashes.json", "asset_hashes_sha256")):
        require(sha(path) == completion[key], "Frozen reporter binding changed: " + key)
    inventory = load(evidence / "asset_hashes.json")
    for section in ("run_assets", "wrapper_assets", "captured_sources", "shared_assets"):
        for item in inventory[section]:
            verify_asset(item)
    durable = load(evidence / "durable_files.json")
    require(durable["report"]["sha256"] == completion["report_sha256"], "Durable report binding changed")
    for item in durable["files"]:
        path = (evidence / item["path"]).resolve()
        require(path.is_relative_to(evidence) and path.suffix != ".pt" and "witness" not in path.name.lower() and
                sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                "Durable reporter artifact changed: " + item["path"])
    require(candidate["report_sha256"] == completion["report_sha256"] and candidate["table_rows_sha256"] == completion["table_rows_sha256"],
            "Candidate report/table mismatch")
    run = (ROOT / completion["run"]).resolve()
    require(run.parent == ROOT / "output/runs/market-conditioned-changes" and candidate["run"] == completion["run"], "Declared run mismatch")
    verification, audit = load(run / "verification.json"), load(run / "independent-audit.json")
    require(verification["status"] == audit["status"] == "passed" and verification["protocol"] == PROTOCOL and
            verification["card_sha256"] == CARD_SHA and verification["instance"] == TAG and
            sha(run / "verification.json") == completion["wrapper_verification_sha256"] and
            sha(run / "independent-audit.json") == completion["independent_audit_sha256"], "Passed frozen wrapper/audit bindings required")
    require(len(tables["rows"]) == 64 and sum(not row["reused"] for row in tables["rows"]) == 8,
            "Preserve all56 old and8 new task rows")
    final = [row for row in tables["rows"] if row["method"] == TAG]
    require([row["task"] for row in final] == TASKS and candidate["final_task_rows"] == final and
            all(row["size"] == 36 and row["total"] == 32 and 0 < row["valid"] <= 32 and
                all(math.isfinite(row[key]) for key in HEADS) for row in final), "Exact four final task rows required")
    original, strict = tables["original_criteria"], tables["strict_native36_criteria"]
    individual, coverage = tables["individual_neural_trend_criteria"], tables["coverage_criterion"]
    require(len(original) == 9 and len(strict) == 8 and len(individual) == 3 and
            sum(item["passed"] for item in original) == completion["original_criteria_passed"] and
            sum(item["passed"] for item in strict) == completion["strict_criteria_passed"] and
            candidate["original_conditions"] == original and candidate["strict_conditions"] == strict and
            candidate["individual_trend_conditions"] == individual and candidate["coverage_condition"] == coverage,
            "Own frozen criteria must match verified report")
    market_pass = all(item["passed"] for item in original + strict + individual + [coverage])
    require(completion["full_market_gate_passed"] == candidate["full_market_gate_passed"] == tables["full_market_gate_passed"] == market_pass and
            completion["all_three_final_neural_trend_positive"] == all(item["passed"] for item in individual) and
            completion["full_coverage"] == coverage["passed"] and
            completion["synthetic_transfer_measured"] is False and completion["primary_changed"] is False,
            "Market-only disposition mismatch")
    checkpoints = []
    for item in candidate["checkpoints"]:
        path = verify_asset(item)
        require(path.parent == run / "checkpoints" and path.suffix == ".pt", "New checkpoint escaped declared leaf")
        checkpoints.append(dict(path=relative(path), sha256=item["sha256"], bytes=item["bytes"],
                                local_only=True, git_ignored=True, deserialized_for_ledger=False))
    require(sorted(Path(item["path"]).name for item in checkpoints) == sorted([
        "inference-0.pt", "inference-2048.pt", "training-512.pt", "training-1024.pt", "training-2048.pt"]), "Five preserved v26 checkpoints required")
    all_checkpoints = [item for entry in old_entries for item in entry["checkpoints"]] + checkpoints
    require(len(all_checkpoints) == 35, "All35 candidate checkpoints required")
    for item in all_checkpoints:
        verify_asset(item)
    ignored = subprocess.run(["git", "-c", "safe.directory=/cuwacunu", "-C", str(ROOT), "check-ignore", "--stdin"],
                             input="\n".join(item["path"] for item in all_checkpoints) + "\n", text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(ignored.returncode == 0 and set(ignored.stdout.splitlines()) == {item["path"] for item in all_checkpoints},
            "Every indexed checkpoint must remain Git ignored")
    training = [row for row in tables["training_rows"] if row["encoder"] == TAG]
    require(len(training) == 1, "Exact final reconstruction/cost row required")
    normalizer_path = run / "bank-standardizer.json"
    normalizer = load(normalizer_path)
    require(normalizer["identity"] == completion["normalizer_identity"] == candidate["normalizer_identity"] and
            normalizer["archive_sha256"] == completion["normalizer_sha256"] == candidate["normalizer_sha256"] and
            sha(run / "bank-standardizer.pt") == normalizer["archive_sha256"], "Normalizer content binding changed")
    entry = copy.deepcopy(candidate)
    entry.update(architecture_tag="RPB-v26", recorded_date="2026-10-15",
                 status="market gate passed; conditional TEMPO-5 pending; no balanced promotion" if market_pass else "stopped exact conditioned-bank mechanism; preserved task references",
                 description="Frozen TRAIN mean/population-SD conditioning of unchanged bank; protected parent32 plus learned4/native36; waveform objective only",
                 core_matched_scope=True, dataset_scope=ledger["matched_scope"]["id"],
                 input_geometry=dict(C=3,H=128,F=9,P=8,W=64,native_width=36,output="one native vector per window"),
                 shared_ledger_sha256="dbc208fb7c858909ff6f04e70b939adaab1c3e8c27f8e72f261a620cf1b55e10",
                 head_recipe=copy.deepcopy(old_entries[4]["head_recipe"]),
                 encoder_budget=dict(new_updates=2048,inherited_parent_updates=2048,batch_size=8,device="cuda:0",GPU="NVIDIA RTX A2000 8GB",
                                     quality_points=[0,2048],intermediate512_1024="preserved recovery state; never selected by quality",
                                     training_recipe="AdamW0.001/weight-decay0.0001;clip1;mask25%;context15%;waveform Huber1 only",
                                     training_and_reconstruction=training[0]),
                 card=asset(ROOT / "doc/MARKET_CONDITIONED_CHANGE_CARD.md"), report=asset(report_path), completion=asset(completion_path),
                 own_frozen_result_table=asset(table_path), comparison_rows_source=asset(table_path), verified_candidate_record=asset(candidate_path),
                 own_original_criteria=dict(source=asset(table_path),conditions=original,passed=sum(item["passed"] for item in original),total=9,
                                            rescore_under_later_card=False,note="Counts refer only to this instance own frozen prospective card.",
                                            strict_native36_conditions=strict,strict_passed=sum(item["passed"] for item in strict),strict_total=8,
                                            all_three_trend_neural_positive=all(item["passed"] for item in individual),full_coverage=coverage["passed"],
                                            full_market_gate_passed=market_pass),
                 checkpoints=checkpoints, calibration=dict(metadata=asset(normalizer_path),archive=asset(run / "bank-standardizer.pt"),
                                                         identity=normalizer["identity"],fitted_on="4096 natural-mask TRAIN anchors only",encoder_calls=0),
                 initial_task_rows=[row for row in tables["rows"] if row["method"] == TAG + ".init"],
                 balanced_promotion_eligible=False, conditional_transfer_status="pending after market pass" if market_pass else "not permitted after market failure")
    ledger["candidate_entries"].append(entry)
    lookup = {(candidate["model"], row["task"]): row for candidate in ledger["candidate_entries"] for row in candidate["final_task_rows"]}
    views = dict(current_evidence_source=asset(table_path), current_core_methods=CORE,
                 current_core_per_task_head=maxima(CORE, lookup, "core matched six final checkpoints; descriptive; width/exposure differ"),
                 current_all_recorded_per_task_head=maxima(OLD_MODELS + [TAG], lookup, "all seven final records including frozen-v23 alternative; descriptive"),
                 current_core_eight_axis_Pareto_front=pareto(CORE, lookup),
                 current_all_recorded_eight_axis_Pareto_front=pareto(OLD_MODELS + [TAG], lookup))
    before_live, before_views = copy.deepcopy(ledger["live"]), copy.deepcopy(ledger["leader_views"])
    ledger["leader_views"].update(views)
    ledger["matched_scope"]["core_final_methods"] = CORE
    ledger["live"]["active_work"] = ("Verified v26 market pass; compatible frozen TEMPO-5 transfer pending" if market_pass else
                                      "V26 exact mechanism stopped after verified market gate failure; declare next justified mechanism")
    ledger["live"]["active_work_is_new_encoder_candidate"] = False
    ledger["live"]["latest_measured_candidate"] = TAG
    ledger["live"]["latest_market_gate_passed"] = market_pass
    ledger["live"]["balanced_promotion_pending_transfer"] = TAG if market_pass else None
    # No primary, balanced reference, old-score claim or historical entry changes.
    require(ledger["candidate_entries"][:6] == old_entries and ledger["live"]["primary_architecture"] == before_live["primary_architecture"] and
            ledger["live"]["leading_balanced_research_reference"] == before_live["leading_balanced_research_reference"] and
            ledger["live"]["verified_new_nine_of_nine_winner"] is None, "Historical/selection preservation failed")
    ledger["preservation"]["checkpoints_indexed"] = 35
    ledger["preservation"]["v26_completion"] = asset(completion_path)
    ledger["preservation"]["v26_wrapper_verification"] = asset(run / "verification.json")
    ledger["preservation"]["v26_checkpoint_hashes_reverified_and_Git_ignored"] = True
    ledger["evidence_stage"] = "verified saved known-source chronological development; market-only ledger append"
    before_json = dict(repository="integration",path=relative(backups[0]),sha256=hashlib.sha256(before_bytes).hexdigest(),bytes=len(before_bytes))
    before_md = dict(repository="integration",path=relative(backups[1]),sha256=hashlib.sha256(md_bytes).hexdigest(),bytes=len(md_bytes))
    event = dict(event_id="2026-10-15-v26-market-only-append",date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 appended_model=TAG,market_gate_passed=market_pass,synthetic_transfer_measured=False,balanced_pointer_changed=False,primary_changed=False,
                 prior_live=before_live,prior_leader_views=before_views,current_views=views,prior_ledger=before_json,prior_markdown=before_md,
                 original_six_candidate_entries_canonical_sha256=canonical_sha(old_entries),new_completion=asset(completion_path),
                 policy="This single append stage cannot select a new balanced leader; a separate verified TEMPO-5 event is required.")
    ledger.setdefault("leader_history", []).append(event)
    ledger["last_index_update"] = dict(model=TAG,mode="market-only append",new_encoder_calls=0,new_encoder_updates=0,new_head_fits=0,
                                       new_scaler_fits=0,tensor_archives_deserialized=0,Git_commits_or_pushes=0,mirror_required=True)
    json_bytes = encoded(ledger)
    markdown_bytes = markdown(ledger, market_pass, before_json, before_md)
    # Exclusive receipts preserve the entire original documents and generated
    # replacement payload before either live file is replaced.
    for path, content in zip(backups + staged, [before_bytes, md_bytes, json_bytes, markdown_bytes]):
        exclusive(path, content)
    require(sha(ledger_path) == args.expected_ledger_sha256 and md_path.read_bytes() == md_bytes, "Concurrent live ledger edit")
    temporary = [ledger_path.with_name(ledger_path.name + ".v26-new"), md_path.with_name(md_path.name + ".v26-new")]
    for path, content in zip(temporary, [json_bytes, markdown_bytes]):
        exclusive(path, content)
    os.replace(temporary[0], ledger_path)
    os.replace(temporary[1], md_path)
    require(load(ledger_path)["candidate_entries"][:6] == old_entries and len(load(ledger_path)["candidate_entries"]) == 7,
            "Final append-only semantics failed")
    result = dict(kind="encoder_search_ledger_v26_market_append_v1",status="passed",protocol=PROTOCOL,
                  before_json=before_json,before_markdown=before_md,after_json=asset(ledger_path),after_markdown=asset(md_path),
                  original_six_entries_semantically_unchanged=True,original_six_entries_canonical_sha256=canonical_sha(old_entries),
                  candidate_entries=7,checkpoints_available_hash_matched_and_ignored=35,appended_model=TAG,
                  full_market_gate_passed=market_pass,synthetic_transfer_measured=False,balanced_pointer_changed=False,primary_changed=False,
                  source=asset(Path(__file__)),new_encoder_calls=0,new_encoder_updates=0,new_head_fits=0,new_scaler_fits=0,tensors_deserialized=0,
                  commit_or_push=False,embedding_mirror="pending controlling task host copy and byte-equality verification")
    exclusive(receipt, encoded(result))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
