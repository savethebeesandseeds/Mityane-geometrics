#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Append one verified market candidate without promoting primary/balanced pointers.

This saved-JSON/hash-only helper runs in the existing cuwacunu container after a
candidate reporter. Earlier candidate entries and leader history stay intact.
Only the controlling task performs the identical local embedding mirror.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["past_volatility", "past_trend", "past_activity", "past_relation"]
HEADS = ["linear_r2", "neural_mean_r2"]
LEGEND = "MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only targets · complexity4/5"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(value):
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
    require(isinstance(name, str), "Asset path required")
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
            "Frozen evidence changed: " + name)
    return path


def exclusive(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)


def maxima(methods, rows, scope):
    result = []
    for task in TASKS:
        for head in HEADS:
            best = max(rows[model, task][head] for model in methods)
            winners = [model for model in methods if rows[model, task][head] == best]
            row = rows[winners[0], task]
            result.append(dict(task=task, head=head, method=winners[0], tied_methods=winners,
                               size=row["size"], r2=best, coverage=row["coverage"], scope=scope))
    return result


def pareto(methods, rows):
    scores = {model: [rows[model, task][head] for task in TASKS for head in HEADS] for model in methods}
    return [model for model in methods if not any(other != model and
            all(a >= b for a, b in zip(scores[other], scores[model])) and
            any(a > b for a, b in zip(scores[other], scores[model])) for other in methods)]


def table(lines, headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines.extend(["| " + " | ".join(map(cell, headers)) + " |",
                  "| " + " | ".join("---" if j == 0 else "---:" for j in range(len(headers))) + " |"])
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    lines.append("")


def make_markdown(ledger, model, old_count, market_pass, before_json, before_md):
    entries = {entry["model"]: entry for entry in ledger["candidate_entries"]}
    methods, core = list(entries), ledger["matched_scope"]["core_final_methods"]
    lookup = {(entry["model"], row["task"]): row for entry in entries.values() for row in entry["final_task_rows"]}
    latest = entries[model]
    result = ("passes the complete market gate; separately verified frozen TEMPO-5 transfer remains pending" if market_pass else
              "fails the complete market gate; its exact mechanism is stopped")
    lines = ["# Encoder search ledger", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
             f"{model} {result}. RPB-v22 remains primary; RPB-v24.market-01.alt-01 remains the balanced research reference.",
             f"All{old_count} earlier candidate entries are preserved semantically unchanged. Every checkpoint, failed condition and useful per-task reference remains available.",
             "This ledger reuses verified saved measurements. It executes no encoder, deserializes no tensor, fits no head/scaler and performs no Git commit, push or publication.", "",
             "This Markdown and the exact JSON are identical local mirrors in both repositories. Evidence paths resolve from C:/Work/cuwacunu/cuwacunu_dev. Checkpoints and frozen calibration weights remain local ignored files.", "",
             "## Matched final checkpoints", "", LEGEND + " · intact known chronological VALIDATION.",
             "Same C3/H128/F9/P8/W64, original TRAIN CF scaler,4096 TRAIN anchors,112 TRAIN/32 known-VALIDATION probes. TRAIN Jan1–21; VALIDATION Jan22–28; Jan29–31 unused.",
             "Core methods: " + ", ".join(core) + ". Native widths, width-derived head seeds/costs and inherited exposure differ; this comparison describes known development evidence.",
             "Latest mechanism: " + latest["description"] + ".", ""]
    for task in TASKS:
        lines.extend(["### " + task.replace("_", " "), "", LEGEND + " · " + task.replace("_", " ") + " · intact known VALIDATION.", ""])
        table(lines, ["Method", "Size", "Linear head R²", "Neural head R²", "Coverage %"],
              [(name, lookup[name, task]["size"], f"{lookup[name, task]['linear_r2']:.6f}",
                f"{lookup[name, task]['neural_mean_r2']:.6f}", f"{100*lookup[name, task]['coverage']:.0f}") for name in core])
    lines.extend(["## Own frozen decisions and exposure", "", LEGEND + " · counts belong to each own card; older recipes are never rescored under a later gate.", ""])
    decision_rows = []
    for name, entry in entries.items():
        score, budget = entry["own_original_criteria"], entry["encoder_budget"]
        count = "No nine-condition score" if score.get("total") is None else f"{score['passed']}/{score['total']}"
        decision_rows.append((name, entry["input_geometry"]["native_width"], count,
                              f"{budget['new_updates']}/{budget['inherited_parent_updates']}", entry["status"]))
    table(lines, ["Method", "Native size", "Own conditions", "New/inherited updates", "Status"], decision_rows)
    score = latest["own_original_criteria"]
    lines.extend([f"Latest strict native36 floors:{score['strict_passed']}/8; all three final neural trend fits positive:{score['all_three_trend_neural_positive']}; full32/32 coverage:{score['full_coverage']}.",
                  "Old parent N/A, fresh/frozen-v23 eight-condition decisions, native40/native36/v25/v26 outcomes and per-task leaders remain intact. Three fixed neural repetitions on one checkpoint are head variation, not independent market cohorts.", "",
                  "## Reconstruction and training cost", "", LEGEND + " · unchanged fixed hidden TRAIN/known-VALIDATION query MAE.", ""])
    training = [entry["encoder_budget"]["training_and_reconstruction"] for entry in entries.values()]
    table(lines, ["Encoder", "Updates", "Train error ↓", "Validation error ↓", "GPU training seconds"],
          [(row["encoder"], row["updates"], f"{row['train_error']:.6f}", f"{row['validation_error']:.6f}",
            "—" if row["gpu_training_seconds"] is None else f"{row['gpu_training_seconds']:.3f}") for row in training])
    lines.extend(["All trajectories use B8 on RTX A2000 8GB/cuda:0. Errors are original fixed-query TRAIN-CF-scaled MAE. GPU training includes checkpoint I/O and excludes calibration/extraction/head fitting; inherited exposure is separate.",
                  "The latest result separately reports necessary frozen calibration forwards, map fitting, reused statistics, extraction, fixed saved-head fitting/scoring/I/O and audit durations. Frozen calibration coefficients are storage, not trainable encoder parameters.", "",
                  "## Fixed head costs and task leaders", "", LEGEND + " · TRAIN-only normalizers; ridge1; neural16-tanh/Adam0.01/100 steps; repetitions2701/2802/2903.", ""])
    recipes = {}
    for entry in entries.values():
        recipes[entry["input_geometry"]["native_width"]] = entry["head_recipe"]
    table(lines, ["Native size", "Ridge parameters", "Neural parameters", "Exact uint64 seeds for2701/2802/2903"],
          [(width, recipe["ridge_parameters"], recipe["neural_parameters"], ", ".join(recipe["uint64_actual_seeds_decimal"]))
           for width, recipe in sorted(recipes.items())])
    for title, key in [(f"Core{len(core)} final checkpoints", "current_core_per_task_head"),
                       (f"All{len(methods)} preserved final records including frozen-v23 alternative", "current_all_recorded_per_task_head")]:
        lines.extend([LEGEND + " · " + title + " · intact known VALIDATION · descriptive task/head maxima.", ""])
        table(lines, ["Task", "Head", "Method", "R²"],
              [(item["task"], item["head"], ", ".join(item["tied_methods"]), f"{item['r2']:.17g}") for item in ledger["leader_views"][key]])
    lines.extend(["Core eight-axis Pareto frontier: " + ", ".join(ledger["leader_views"]["current_core_eight_axis_Pareto_front"]) + ". Exact saved scalar comparisons describe tradeoffs and do not select a universal winner.",
                  "The frozen-parent v23 alternative stays outside the core scope and inside the broader maxima. V25 and v26 partial leaders remain visible even when a future balanced candidate supersedes them.", "",
                  "## Protection and next direction", "",
                  "Candidate records are append-only. Corrections append linked records. Preserve all previous leader views, exact own criteria, physical errors, every neural repetition, cards/reports, initialization and recovery checkpoints.",
                  "Changing the balanced pointer requires all original9 conditions, strict native36 eight floors, all three trend repetitions positive, full coverage and separately verified compatible TEMPO-5 retention under both unchanged input-scaler policies. This market-only append changes no balanced or primary pointer.",
                  "Only the declared absolute1e-12 arithmetic tolerance applies to quality. Per-task maxima do not silently become new mandatory floors. No stopped map/normalizer/loss/head/rate/width/seed/budget rescue or failed-market transfer.",
                  ("Next measure the exact frozen checkpoint on the declared compatible TEMPO-5 cohorts with every new calibration state frozen. Transfer is pending; no new balanced selection is recorded."
                   if market_pass else "Stop the exact latest mechanism, preserve all outcomes and declare any different justified mechanism prospectively before quality. Continue searching with the retained balanced and per-task references."),
                  "Fresh chronology, TEMPO-4 and AMP-2 remain separate; no forecasting, TEST or universal-best claim follows.", "",
                  "## Evidence", "", "- [Exact machine ledger](encoder_search_ledger.json)",
                  f"- Latest report: {latest['report']['path']} SHA256 {latest['report']['sha256']}",
                  f"- Latest card: {latest['card']['path']} SHA256 {latest['card']['sha256']}",
                  f"- Previous JSON: {before_json['path']} SHA256 {before_json['sha256']}",
                  f"- Previous Markdown: {before_md['path']} SHA256 {before_md['sha256']}", "",
                  f"All{ledger['preservation']['checkpoints_indexed']} candidate checkpoints were byte-hash checked and remain Git-ignored. This update performs no broad historical scan, deserialization or weight export. Earlier preservation receipts stay bound. The unrelated roadmap remains untouched."])
    return ("\n".join(lines).rstrip()+"\n").encode()


def main():
    require(Path("/.dockerenv").is_file() and ROOT == Path("/cuwacunu"), "Use existing managed cuwacunu container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--expected-prior-entries", type=int, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()
    require(evidence.parent == ROOT / "doc/results" and args.expected_prior_entries >= 7, "Declared new market evidence directory required")
    ledger_path, md_path = ROOT / "doc/encoder_search_ledger.json", ROOT / "doc/ENCODER_SEARCH_LEDGER.md"
    require(sha(ledger_path) == args.expected_ledger_sha256, "Live ledger changed since controlling-task review")
    before_bytes, md_bytes = ledger_path.read_bytes(), md_path.read_bytes()
    ledger = json.loads(before_bytes)
    old_entries = copy.deepcopy(ledger["candidate_entries"])
    snapshot = load(evidence / "search-ledger-before.json")
    require(len(old_entries) == args.expected_prior_entries and old_entries == snapshot["candidate_entries"], "Historical candidate entries changed")
    require(ledger["live"]["primary_architecture"] == "RPB-v22" and
            ledger["live"]["leading_balanced_research_reference"] == "RPB-v24.market-01.alt-01" and
            ledger["live"]["verified_new_nine_of_nine_winner"] is None, "Separate selection event required before using another primary/balanced baseline")
    completion_path, candidate_path, table_path = [evidence/name for name in ["completion.json", "candidate-record.json", "table_rows.json"]]
    completion, candidate, tables = load(completion_path), load(candidate_path), load(table_path)
    model, protocol = candidate["model"], candidate["protocol"]
    require(re.fullmatch(r"RPB-v[0-9]+\.market-[0-9]+(?:\.alt-[0-9]+)?", model) and model not in [entry["model"] for entry in old_entries], "Distinct declared market instance required")
    require(completion["status"] == tables["status"] == "complete" and completion["instance"] == tables["instance"] == model and
            completion["card_sha256"] == candidate["card_sha256"], "Verified candidate completion required")
    report_path, card_path = verify_asset(candidate["report_asset"]), verify_asset(candidate["card_asset"])
    old_table = verify_asset(candidate["reused_table"])
    require(sha(card_path) == candidate["card_sha256"], "Prospective card mismatch")
    for path, key in [(table_path,"table_rows_sha256"),(candidate_path,"candidate_record_sha256"),
                      (report_path,"report_sha256"),(evidence/"asset_hashes.json","asset_hashes_sha256")]:
        require(sha(path) == completion[key], "Frozen reporter hash binding: "+key)
    require(candidate["report_sha256"] == completion["report_sha256"] and candidate["table_rows_sha256"] == completion["table_rows_sha256"], "Candidate report/table binding")
    inventory, durable = load(evidence/"asset_hashes.json"), load(evidence/"durable_files.json")
    for section in ["run_assets","wrapper_assets","captured_sources","shared_assets"]:
        for item in inventory[section]:
            verify_asset(item)
    for item in durable["files"]:
        path=(evidence/item["path"]).resolve()
        require(path.is_relative_to(evidence) and path.suffix!=".pt" and "witness" not in path.name.lower() and path.name!="conditional-map.json" and
                sha(path)==item["sha256"] and path.stat().st_size==item["bytes"], "Durable artifact changed: "+item["path"])
    require(sha(report_path)==durable["report"]["sha256"], "Durable report mismatch")
    run=(ROOT/completion["run"]).resolve()
    require(run.is_relative_to(ROOT/"output/runs") and run.name.startswith("run-") and candidate["run"]==completion["run"], "Declared run required")
    verification, audit = load(run/"verification.json"), load(run/"independent-audit.json")
    require(verification["status"]==audit["status"]=="passed" and verification["protocol"]==protocol and
            verification["card_sha256"]==candidate["card_sha256"] and verification["instance"]==model and
            sha(run/"verification.json")==completion["wrapper_verification_sha256"] and
            sha(run/"independent-audit.json")==completion["independent_audit_sha256"], "Verified wrapper/audit bindings required")
    old_rows=load(old_table)["rows"]
    require(len(tables["rows"])==len(old_rows)+8 and sum(not row["reused"] for row in tables["rows"])==8, "Preserve every old task row and eight new rows")
    provenance={"reused","report_path","report_sha256","original_report_path","original_report_sha256"}
    for old,new in zip(old_rows,tables["rows"][:len(old_rows)]):
        require({key:value for key,value in old.items() if key not in provenance}==
                {key:value for key,value in new.items() if key not in provenance}, "Historical task values changed")
    final=[row for row in tables["rows"] if row["method"]==model]
    require([row["task"] for row in final]==TASKS and candidate["final_task_rows"]==final and
            all(row["size"]==36 and row["total"]==32 and 0<row["valid"]<=32 and all(math.isfinite(row[key]) for key in HEADS) for row in final), "Four complete native36 final task rows required")
    original, strict=tables["original_criteria"], tables["strict_native36_criteria"]
    individual, coverage=tables["individual_neural_trend_criteria"], tables["coverage_criterion"]
    require(len(original)==9 and len(strict)==8 and len(individual)==3 and
            sum(item["passed"] for item in original)==completion["original_criteria_passed"] and
            sum(item["passed"] for item in strict)==completion["strict_criteria_passed"] and candidate["original_conditions"]==original and
            candidate["strict_conditions"]==strict and candidate["individual_trend_conditions"]==individual and candidate["coverage_condition"]==coverage,
            "Exact own frozen conditions required")
    market_pass=all(item["passed"] for item in original+strict+individual+[coverage])
    require(completion["full_market_gate_passed"]==candidate["full_market_gate_passed"]==tables["full_market_gate_passed"]==market_pass and
            completion["all_three_final_neural_trend_positive"]==all(item["passed"] for item in individual) and completion["full_coverage"]==coverage["passed"] and
            completion["synthetic_transfer_measured"] is False and completion["primary_changed"] is False, "Market-only disposition required")
    checkpoints=[]
    for item in candidate["checkpoints"]:
        path=verify_asset(item)
        require(path.parent==run/"checkpoints" and path.suffix==".pt", "Candidate checkpoint escaped run")
        checkpoints.append(dict(path=relative(path),sha256=item["sha256"],bytes=item["bytes"],local_only=True,git_ignored=True,deserialized_for_ledger=False))
    require(sorted(Path(item["path"]).name for item in checkpoints)==sorted(["inference-0.pt","inference-2048.pt","training-512.pt","training-1024.pt","training-2048.pt"]), "Five immutable checkpoints required")
    all_checkpoints=[item for entry in old_entries for item in entry["checkpoints"]]+checkpoints
    require(len(all_checkpoints)==5*(len(old_entries)+1), "Every historical checkpoint required")
    for item in all_checkpoints:
        verify_asset(item)
    ignored=subprocess.run(["git","-c","safe.directory=/cuwacunu","-C",str(ROOT),"check-ignore","--stdin"],
                           input="\n".join(item["path"] for item in all_checkpoints)+"\n",text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    require(ignored.returncode==0 and set(ignored.stdout.splitlines())=={item["path"] for item in all_checkpoints}, "Every indexed checkpoint must remain Git ignored")
    training=[row for row in tables["training_rows"] if row["encoder"]==model]
    require(len(training)==1, "One final reconstruction/cost row required")
    calibration=dict(new_parent_CUDA_calibration_calls=completion.get("new_calibration_parent_CUDA_encoder_calls",0),
                     calibration_encoder_updates=0,new_bank_normalizer_fits=completion.get("new_bank_normalizer_fits",0))
    if "normalizer_identity" in candidate:
        parent_normalizer=load(run/"bank-standardizer.json")
        require(parent_normalizer["identity"]==completion["normalizer_identity"]==candidate["normalizer_identity"] and
                parent_normalizer["archive_sha256"]==completion["normalizer_sha256"]==candidate["normalizer_sha256"]==sha(run/"bank-standardizer.pt"), "Normalizer content binding required")
        calibration.update(bank_metadata=asset(run/"bank-standardizer.json"),bank_archive=asset(run/"bank-standardizer.pt"))
    if "conditional_map_summary" in candidate:
        compact_map=verify_asset(candidate["conditional_map_summary"])
        map_summary=load(compact_map)
        require(map_summary["identity"]==completion["conditional_map_identity"]==candidate["conditional_map_identity"] and
                map_summary["archive_sha256"]==completion["conditional_map_sha256"]==candidate["conditional_map_sha256"]==sha(run/"conditional-map.pt"), "Conditional-map content binding required")
        calibration.update(map_metadata=asset(compact_map),map_archive=asset(run/"conditional-map.pt"),
                           map_identity=map_summary["identity"],map_coefficient_count=map_summary["coefficient_count"])
    recipe=next(entry["head_recipe"] for entry in old_entries if entry["model"]=="RPB-v24.market-01.alt-01")
    if "head_recipe" in candidate:
        require(candidate["head_recipe"]==recipe, "Exact width-dependent fixed-head metadata mismatch")
    status="market gate passed; separate conditional transfer pending; no balanced promotion" if market_pass else "stopped exact declared mechanism; preserved task references"
    entry=copy.deepcopy(candidate)
    entry.update(architecture_tag=model.split(".")[0],recorded_date=completion["date"],status=status,description=candidate["design"],
                 core_matched_scope=True,dataset_scope=ledger["matched_scope"]["id"],
                 input_geometry=dict(C=3,H=128,F=9,P=8,W=64,native_width=36,output="one native vector per window"),
                 shared_ledger_sha256="dbc208fb7c858909ff6f04e70b939adaab1c3e8c27f8e72f261a620cf1b55e10",head_recipe=copy.deepcopy(recipe),
                 encoder_budget=dict(new_updates=completion["new_branch_updates"],inherited_parent_updates=completion["inherited_parent_updates"],batch_size=8,
                                     device="cuda:0",GPU="NVIDIA RTX A2000 8GB",quality_points=[0,2048],
                                     intermediate512_1024="preserved recovery state; never selected by quality",training_and_reconstruction=training[0]),
                 card=asset(card_path),report=asset(report_path),completion=asset(completion_path),own_frozen_result_table=asset(table_path),
                 comparison_rows_source=asset(table_path),verified_candidate_record=asset(candidate_path),
                 own_original_criteria=dict(source=asset(table_path),conditions=original,passed=sum(item["passed"] for item in original),total=9,
                                            rescore_under_later_card=False,note="Counts refer only to this instance own frozen prospective card.",
                                            strict_native36_conditions=strict,strict_passed=sum(item["passed"] for item in strict),strict_total=8,
                                            all_three_trend_neural_positive=all(item["passed"] for item in individual),full_coverage=coverage["passed"],full_market_gate_passed=market_pass),
                 checkpoints=checkpoints,calibration=calibration,
                 initial_task_rows=[row for row in tables["rows"] if row["method"]==model+".init"],balanced_promotion_eligible=False,
                 conditional_transfer_status="pending after full market pass" if market_pass else "not permitted after market failure")
    previous_live, previous_views=copy.deepcopy(ledger["live"]),copy.deepcopy(ledger["leader_views"])
    previous_core=list(ledger["matched_scope"]["core_final_methods"])
    ledger["candidate_entries"].append(entry)
    core=previous_core+[model]
    methods=[entry["model"] for entry in ledger["candidate_entries"]]
    rows={(entry["model"],row["task"]):row for entry in ledger["candidate_entries"] for row in entry["final_task_rows"]}
    views=dict(current_evidence_source=asset(table_path),current_core_methods=core,
               current_core_per_task_head=maxima(core,rows,f"core matched{len(core)} final checkpoints; descriptive; width/exposure differ"),
               current_all_recorded_per_task_head=maxima(methods,rows,f"all{len(methods)} final records including frozen-v23 alternative; descriptive"),
               current_core_eight_axis_Pareto_front=pareto(core,rows),current_all_recorded_eight_axis_Pareto_front=pareto(methods,rows))
    ledger["leader_views"].update(views)
    ledger["matched_scope"]["core_final_methods"]=core
    ledger["live"].update(active_work=(f"Verified {model} market pass; frozen compatible transfer pending" if market_pass else
                                      f"{model} exact mechanism stopped after verified market failure; declare next justified mechanism"),
                          active_work_is_new_encoder_candidate=False,latest_measured_candidate=model,latest_market_gate_passed=market_pass,
                          balanced_promotion_pending_transfer=model if market_pass else None)
    require(ledger["candidate_entries"][:-1]==old_entries and ledger["live"]["primary_architecture"]==previous_live["primary_architecture"] and
            ledger["live"]["leading_balanced_research_reference"]==previous_live["leading_balanced_research_reference"] and
            ledger["live"]["verified_new_nine_of_nine_winner"] is None, "Append-only/selection preservation failed")
    ledger["preservation"].update(checkpoints_indexed=len(all_checkpoints),latest_completion=asset(completion_path),
                                  latest_wrapper_verification=asset(run/"verification.json"),latest_checkpoint_hashes_reverified_and_Git_ignored=True)
    ledger["evidence_stage"]="verified saved known-source chronological development; market-only ledger append"
    backups=[evidence/"ledger-update-before.json",evidence/"ledger-update-before.md"]
    staged=[evidence/"ledger-update-generated.json",evidence/"ledger-update-generated.md"]
    receipt=evidence/"search-ledger-update.json"
    require(not any(path.exists() for path in backups+staged+[receipt]), "Preserve every prior ledger update attempt")
    before_json=dict(repository="integration",path=relative(backups[0]),sha256=hashlib.sha256(before_bytes).hexdigest(),bytes=len(before_bytes))
    before_md=dict(repository="integration",path=relative(backups[1]),sha256=hashlib.sha256(md_bytes).hexdigest(),bytes=len(md_bytes))
    ledger.setdefault("leader_history",[]).append(dict(event_id=completion["date"]+"-market-only-append/"+model,
        date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),appended_model=model,market_gate_passed=market_pass,
        synthetic_transfer_measured=False,balanced_pointer_changed=False,primary_changed=False,prior_live=previous_live,prior_leader_views=previous_views,
        current_views=views,prior_ledger=before_json,prior_markdown=before_md,historical_candidate_entries_canonical_sha256=canonical_sha(old_entries),
        new_completion=asset(completion_path),policy="A separate verified compatible-transfer selection event is required before changing the balanced pointer."))
    ledger["last_index_update"]=dict(model=model,mode="market-only append",new_encoder_calls=0,new_encoder_updates=0,new_head_fits=0,
                                     new_scaler_fits=0,tensor_archives_deserialized=0,Git_commits_or_pushes=0,mirror_required=True)
    json_bytes=encode(ledger)
    markdown_bytes=make_markdown(ledger,model,len(old_entries),market_pass,before_json,before_md)
    for path,content in zip(backups+staged,[before_bytes,md_bytes,json_bytes,markdown_bytes]):
        exclusive(path,content)
    require(sha(ledger_path)==args.expected_ledger_sha256 and md_path.read_bytes()==md_bytes,"Concurrent live ledger edit")
    temporary=[ledger_path.with_name(ledger_path.name+".new-candidate"),md_path.with_name(md_path.name+".new-candidate")]
    for path,content in zip(temporary,[json_bytes,markdown_bytes]):
        exclusive(path,content)
    os.replace(temporary[0],ledger_path)
    os.replace(temporary[1],md_path)
    require(load(ledger_path)["candidate_entries"][:-1]==old_entries and len(load(ledger_path)["candidate_entries"])==len(old_entries)+1,"Final append semantics failed")
    result=dict(kind="encoder_search_ledger_market_append_v1",status="passed",protocol=protocol,appended_model=model,
                before_json=before_json,before_markdown=before_md,after_json=asset(ledger_path),after_markdown=asset(md_path),
                historical_entries_semantically_unchanged=len(old_entries),historical_entries_canonical_sha256=canonical_sha(old_entries),
                candidate_entries=len(old_entries)+1,checkpoints_available_hash_matched_and_ignored=len(all_checkpoints),
                full_market_gate_passed=market_pass,synthetic_transfer_measured=False,balanced_pointer_changed=False,primary_changed=False,
                source=asset(Path(__file__)),new_encoder_calls=0,new_encoder_updates=0,new_head_fits=0,new_scaler_fits=0,tensors_deserialized=0,
                commit_or_push=False,embedding_mirror="pending controlling-task host copy and byte-equality verification")
    exclusive(receipt,encode(result))
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
