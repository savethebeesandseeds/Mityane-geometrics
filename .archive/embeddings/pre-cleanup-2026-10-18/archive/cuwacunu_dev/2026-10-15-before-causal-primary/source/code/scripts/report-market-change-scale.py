#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Capture saved TRAIN scale-diagnosis summaries; no tensor loader or fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "market-change-scale-diagnosis-v1"
CARD = "848c5dbf3b6dbce0450c4287c3bd30d0d0b7978ea5571a925d86f492e11d0260"
ADDENDUM = "19a3dc4455a79d40ebd7e6086be1ad44d2d077a26763d2f3826557c3b789fa8c"
MANIFEST = "241625e6815fa32175f082e4ffc73d460eec9b050de3d2f750649186b10488e3"
PREFLIGHT = "178ecb1e7a34eba32ec7db59e214779ecb939c30b9af2d20e9fd41397da94873"
NEURAL_SOURCE = ROOT / "doc/results/market_change_energy_v1/table_rows.json"
NEURAL_SOURCE_SHA = "7d11f577beb61b924a9b896fe3f5dea9c464b779fc34bde19f99ef80751109de"
MODELS = ["RPB-v22.market-02", "RPB-v24.market-01.alt-01", "RPB-v25.market-01"]
TASKS = ["past_volatility", "past_trend", "past_activity", "past_relation"]
CONTEXTS = [("all112", 112), ("first56", 56), ("last56", 56)]
LEGEND = "MKT-HIST-2 · BTC/ETH/SOL January2024 minute history, recent128 past-only tasks · complexity4/5"
TOL = 1e-10


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def invalid(value):
        raise RuntimeError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def record(path):
    return dict(path=relative(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def exclusive(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def dump(path, value):
    exclusive(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def copy_exclusive(source, destination, expected=None):
    expected = expected or sha(source)
    require(sha(source) == expected, "Source changed before capture: " + str(source))
    exclusive(destination, Path(source).read_bytes())
    require(sha(destination) == expected, "Captured source changed: " + str(destination))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(actual, expected, label):
    require(finite(actual) and finite(expected) and abs(actual - expected) / (1 + abs(expected)) <= TOL, label)


def fmt(value, digits=6):
    return "undefined" if value is None else f"{value:.{digits}g}"


def table(lines, headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines.extend(["| " + " | ".join(map(cell, headers)) + " |",
                  "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"])
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    lines.append("")


def verify_record(path, expected):
    require(Path(path).is_file() and sha(path) == expected["sha256"] and Path(path).stat().st_size == expected["bytes"],
            "Protected saved file mismatch: " + str(path))


def quality_rows(diagnosis, frozen):
    require([method["model"] for method in diagnosis["methods"]] == MODELS, "Declared three saved methods changed")
    old = {(row["method"], row["task"], row["view"]): row for row in frozen["train_context_rows"]}
    rows = []
    for method in diagnosis["methods"]:
        width = 32 if method["model"] == MODELS[0] else 36
        require(method["width"] == width, "Saved method width mismatch")
        require([view["context"] for view in method["contexts"]] == [name for name, _ in CONTEXTS], "TRAIN contexts changed")
        for view in method["contexts"]:
            count = dict(CONTEXTS)[view["context"]]
            require([target["task"] for target in view["tasks"]] == TASKS, "Four TRAIN tasks required")
            for target in view["tasks"]:
                linear = target["linear"]
                require((linear["total"], linear["valid"], linear["coverage"]) == (count, count, 1), "Full TRAIN scoring support required")
                previous = old[method["model"], target["task"], view["context"]]
                for metric in ("r2", "mae", "rmse"):
                    near(linear[metric], previous["linear_" + metric], "Saved ridge score reproduction changed")
                rows.append(dict(dataset_id="MKT-HIST-2", designed_complexity_level=4, complexity_scale_max=5,
                                 task=target["task"], method=method["model"], size=width, view=view["context"],
                                 evidence_stage="new saved-TRAIN arithmetic; no new encoder quality", valid=count, total=count,
                                 coverage=1, linear_r2=linear["r2"], linear_mae=linear["mae"], linear_rmse=linear["rmse"],
                                 linear_head_reused=True, linear_scores_recomputed_from_existing_saved_ridge=True,
                                 neural_mean_r2=previous["neural_mean_r2"], neural_mean_mae=previous["neural_mean_mae"],
                                 neural_mean_rmse=previous["neural_mean_rmse"], neural_each=previous["neural_each"],
                                 neural_reused_metadata_only=True, neural_source_path=relative(NEURAL_SOURCE),
                                 neural_source_sha256=NEURAL_SOURCE_SHA, unit=previous["unit"],
                                 components=target["components"], associations=target["associations"]))
    require(len(rows) == 36, "Exactly36 task/context/method rows required")
    return rows


def summary_stats(moment, context):
    return next(item for item in moment["contexts"] if item["context"] == context)


def contribution_text(value):
    if "std" in value and "rms" in value:
        return "SD " + fmt(value["std"]) + "; RMS " + fmt(value["rms"])
    return fmt(value.get("mean")) + " [" + fmt(value.get("min")) + ", " + fmt(value.get("max")) + "]"


def main():
    started = time.perf_counter()
    require(Path("/.dockerenv").is_file() and ROOT == Path("/cuwacunu"), "Use existing managed cuwacunu container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/results/market_change_scale_diagnosis_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "doc/MARKET_CHANGE_SCALE_DIAGNOSIS_RESULTS.md")
    args = parser.parse_args()
    run, out, report = args.run.resolve(), args.output_dir.resolve(), args.report.resolve()
    require(run.is_dir() and run.parent == ROOT / "output/runs/market-change-scale-diagnosis" and run.name.startswith("run-"), "New saved diagnosis run required")
    require(out.is_relative_to(ROOT / "doc/results") and report.is_relative_to(ROOT / "doc"), "Durable reports belong in repository docs")
    require(not any(path.exists() for path in (report, out / "completion.json", out / "table_rows.json",
                                              out / "asset_hashes.json", out / "durable_files.json", out / "capture")),
            "Exclusive report outputs already exist; preserve them")
    pinned = {ROOT / "doc/MARKET_CHANGE_SCALE_DIAGNOSIS_CARD.md": CARD,
              ROOT / "doc/MARKET_CHANGE_SCALE_DIAGNOSIS_ADDENDUM.md": ADDENDUM,
              out / "input-manifest.json": MANIFEST, out / "preflight.json": PREFLIGHT,
              NEURAL_SOURCE: NEURAL_SOURCE_SHA}
    for path, expected in pinned.items():
        require(sha(path) == expected, "Frozen declaration/source changed: " + str(path))
    manifest = load(out / "input-manifest.json")
    require(manifest["protocol"] == PROTOCOL and manifest["training_rows"] == 112 and len(manifest["members"]) == 12,
            "Exact TRAIN manifest required")
    input_records = {}
    for member in manifest["members"]:
        path = ROOT / member["path"]
        require(path.is_relative_to(ROOT / "output/runs") and path.suffix == ".pt" and
                not any(word in member["path"].lower() for word in ("validation", "test", "checkpoints")), "Non-TRAIN input forbidden")
        verify_record(path, member)
        input_records[member["path"]] = record(path)
    require(len(input_records) == 12, "Duplicate TRAIN input")
    verification, audit = load(run / "verification.json"), load(run / "independent-audit.json")
    require(verification["status"] == audit["status"] == "passed" and verification["protocol"] == audit["protocol"] == PROTOCOL,
            "Passed wrapper verification and independent audit required")
    require(all(verification[key] == 0 for key in ("encoder_calls", "encoder_updates", "head_fits", "scaler_fits", "held_out_tensor_inputs")) and
            verification["diagnostic_input_archives"] == 12 and verification["training_rows"] == 112, "Forbidden diagnostic work or wrong role")
    require(audit["input_archive_count"] == 12 and audit["normalized_absolute_tolerance"] == TOL and
            all(audit[key] == 0 for key in ("new_encoder_calls", "new_encoder_updates", "head_or_scaler_fits", "tensor_archives_deserialized_by_this_audit")),
            "Independent audit scope mismatch")
    require(audit["maximum_normalized_error"] <= TOL, "Independent numerical bound failed")
    for key, expected in (("card_sha256", CARD), ("addendum_sha256", ADDENDUM), ("input_manifest_sha256", MANIFEST)):
        require(verification[key] == audit[key] == expected, "Declaration binding mismatch: " + key)
    require(sha(run / "independent-audit.json") == verification["independent_audit_sha256"] and
            sha(run / "provenance-before.json") == verification["provenance_sha256"], "Saved audit/provenance hash mismatch")
    diagnosis_path = run / "reader/diagnosis.json"
    diagnosis = load(diagnosis_path)
    require(diagnosis["protocol"] == PROTOCOL and diagnosis["admission_checks"]["status"] == "passed" and
            diagnosis["admission_checks"]["opened_archives"] == 12, "Reader admission incomplete")
    for field in ("exact_prefix32", "exact_row_group_target_support", "allnormalizers_reproduced", "allridge_predictions_reproduced"):
        require(diagnosis["admission_checks"][field] is True, "Reader admission failed: " + field)
    require(diagnosis["pins"]["card_sha256"] == CARD and diagnosis["pins"]["addendum_sha256"] == ADDENDUM and
            diagnosis["pins"]["input_manifest_sha256"] == MANIFEST, "Reader declaration mismatch")
    opened = load(run / "reader/opened-inputs.json")
    require(isinstance(opened, list) and len(opened) == 12, "Actual loader input ledger required")
    opened_paths = set()
    for entry in opened:
        path = Path(entry["path"])
        name = relative(path if path.is_absolute() else ROOT / path)
        require(name in input_records and name not in opened_paths and entry["sha256"] == input_records[name]["sha256"] and
                entry["bytes"] == input_records[name]["bytes"], "Actual loader differs from declared TRAIN allowlist")
        opened_paths.add(name)
    require(opened_paths == set(input_records), "Actual TRAIN loader ledger incomplete")
    provenance = load(run / "provenance-before.json")
    source_records, binaries = {}, []
    for name, expected in provenance.items():
        path = Path(name)
        verify_record(path, expected)
        if path.is_relative_to(ROOT):
            source_records[relative(path)] = record(path)
        else:
            binaries.append(dict(path=str(path), **expected))
    source_records[relative(Path(__file__))] = record(Path(__file__))
    require(any(item["sha256"] == verification["binary_sha256"] for item in binaries), "Reader binary provenance missing")
    frozen = load(NEURAL_SOURCE)
    rows = quality_rows(diagnosis, frozen)
    lookup = {(row["method"], row["task"], row["view"]): row for row in rows}
    assets = [record(path) for path in sorted(run.rglob("*")) if path.is_file()]
    prior_docs = [path for path in sorted(out.rglob("*")) if path.is_file()] if out.exists() else []
    run_text = ["verification.json", "independent-audit.json", "provenance-before.json", "input-manifest.json",
                "historical-preservation-before.json", "historical-preservation-after.json", "reader.log", "audit.log",
                "reader/diagnosis.json", "reader/opened-inputs.json"]
    for name in run_text:
        copy_exclusive(run / name, out / "capture/run" / name)
    for name, entry in source_records.items():
        require("witness" not in name.lower() and not name.endswith(".pt"), "Raw arrays/archive forbidden in durable capture")
        copy_exclusive(ROOT / name, out / "capture/sources" / name, entry["sha256"])
    for index, path in enumerate(prior_docs):
        require("witness" not in path.name.lower() and path.suffix != ".pt", "Raw witness cannot enter durable docs")
        copy_exclusive(path, out / "capture/prerequisites" / f"{index:03d}-{path.name}")
    copy_exclusive(NEURAL_SOURCE, out / "capture/reused-neural-context-source.json", NEURAL_SOURCE_SHA)
    table_rows = dict(protocol=PROTOCOL, status="complete", dataset_id="MKT-HIST-2", designed_complexity_level=4,
                      complexity_scale_max=5, evidence_stage="saved fitted-TRAIN diagnosis only", rows=rows,
                      new_encoder_calls=0, new_encoder_updates=0, new_head_fits=0, new_scaler_fits=0,
                      new_neural_scoring=False, neural_evidence="reused frozen JSON metadata only; no neural tensor access",
                      neural_source=record(NEURAL_SOURCE), primary="RPB-v22", primary_changed=False,
                      new_validation_quality=False, diagnosis_source=record(diagnosis_path))
    lines = ["# Saved TRAIN change-scale diagnosis", "", "Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.", "",
             "New arithmetic from twelve saved TRAIN archives; zero encoder calls/updates, head/scaler fits or held-out tensor inputs.",
             "RPB-v22 remains primary, native36 remains the balanced8/9 reference, and the v25 objective remains stopped after its7/9 result.",
             "This diagnosis measures associations on the same fitted112 TRAIN windows. It cannot establish why known-VALIDATION volatility failed or select a new winner.", "",
             LEGEND + ". TRAIN Jan1–21, contexts all112/first56/last56. The two halves share the fitted heads and source period; they are descriptive, without independent uncertainty.",
             "CPU saved-array arithmetic runs in the existing GPU-enabled cuwacunu container. No encoder checkpoint, CSV/cache, query plan or new window is opened.", "",
             "Linear columns below reproduce the existing ridge2701 predictions using their saved normalizers/weights; no solve or optimizer runs.",
             "Neural columns reuse the three-repetition means from the frozen v25 TRAIN-context JSON, hash " + NEURAL_SOURCE_SHA + ".",
             "No neural parameters or predictions are inspected/scored by this diagnosis. Fixed head recipes and native32/native36 widths remain unchanged; no PCA.", ""]
    for context, count in CONTEXTS:
        for task in TASKS:
            lines.extend(["## " + task.replace("_", " ").capitalize() + " · " + context, "",
                          LEGEND + f" · {count} saved fitted-TRAIN rows, {context}; neural means reused from frozen metadata.", ""])
            table(lines, ["Method", "Size", "Linear head R²", "Neural head R²", "Coverage %"],
                  [(model, lookup[model, task, context]["size"], f"{lookup[model, task, context]['linear_r2']:.6f}",
                    f"{lookup[model, task, context]['neural_mean_r2']:.6f}", "100") for model in MODELS])
    lines.extend(["## Signed drift and centered variation", "", LEGEND + " · global close increments, each fixed TRAIN context.", "",
                  "Drift share is mean²/empirical-RMS² on identical visible adjacent pairs. True zero-energy ratios are undefined; supported counts remain explicit.",
                  "The existing bank floors small positive RMS; the unfloored identity RMS²=variance+mean² is audited separately.", ""])
    drift_rows = []
    for family in ("scaled", "symmetric", "close_raw", "close_proportional", "close_log"):
        selected = [item for item in diagnosis["moment_summaries"] if item["family"] == family and item["feature"] == 3 and item["partition"] == 0]
        for item in selected:
            for context, _ in CONTEXTS:
                values = summary_stats(item, context)["metrics"]["drift_share"]
                drift_rows.append((family + "/c" + str(item["channel"]), context, fmt(values["median"]), fmt(values["max"]), values["count"]))
    table(lines, ["Close increment family/channel", "TRAIN context", "Median drift share", "Maximum drift share", "Supported rows"], drift_rows)
    lines.extend(["These values quantify the removable signed-drift energy. They do not establish centering as a remedy for the encoder's held-out score.",
                  "All27 channel/feature global/quarter count, mean, RMS, centered SD, drift and floor summaries remain in captured diagnosis.json.", "",
                  "## Price level, absolute and relative variation", "", LEGEND + " · equal-channel aggregate; all declared association pairs, fixed TRAIN contexts.", "",
                  "Pearson r uses supported finite rows. The aggregate requires all three channels and averages them equally. Undefined correlations remain undefined.", ""])
    association_rows = [item for item in diagnosis["metric_associations"] if item["left"].startswith("aggregate.") and item["right"].startswith("aggregate.")]
    pairs = list(dict.fromkeys((item["left"], item["right"]) for item in association_rows))
    require(len(pairs) == 18 and len(association_rows) == 54, "All18 declared aggregate association pairs and three TRAIN contexts required")
    associations = {(item["left"], item["right"], item["context"]): item for item in association_rows}
    table(lines, ["Association", "all112 r", "first56 r", "last56 r", "all112 support"],
          [(left.removeprefix("aggregate.") + " / " + right.removeprefix("aggregate."),) +
           tuple(fmt(associations[left, right, context]["pearson"]) for context, _ in CONTEXTS) +
           (associations[left, right, "all112"]["count"],) for left, right in pairs])
    lines.extend(["Positive-log associations use only rows where both quantities are strictly positive, then correlate their logarithms; each cell reports r and its population count.",
                  LEGEND + " · positive-only log/log associations in the same fixed TRAIN contexts.", ""])
    table(lines, ["Association", "all112 log r (n)", "first56 log r (n)", "last56 log r (n)"],
          [(left.removeprefix("aggregate.") + " / " + right.removeprefix("aggregate."),) +
           tuple(fmt(associations[left, right, context]["positive_log_pearson"]) + " (" + str(associations[left, right, context]["positive_log_count"]) + ")"
                 for context, _ in CONTEXTS) for left, right in pairs])
    lines.extend(["Approximate raw levels are recovered with the saved scaler and observed support, without claiming bit-exact raw-source recovery.",
                  "Proportional/log close increments use identical positive adjacent-endpoint support. Symmetric fractional increments are bounded and handle both-zero endpoints as valid zero.",
                  "Close mean/median level distributions, channel-specific and quarter statistics, target reconstruction differences and floor effects remain captured in JSON.", "",
                  "## Bank magnitude and variation", "", LEGEND + " · actual270-coordinate bank on the same fixed TRAIN contexts.", "",
                  "Uncentered squared magnitude and across-row centered variance have different denominators. Shares below keep them separate; close slots overlap the global/quarter groups.", ""])
    bank_rows = []
    for view in diagnosis["bank_balance"]:
        for group in view["groups"]:
            if group["group"] in ("global_slots", "quarter_slots", "close_slots"):
                raw, centered = 100 * group["raw_squared_share"], 100 * group["centered_variance_share"]
                bank_rows.append((view["context"], group["group"], fmt(raw), fmt(centered), fmt(raw - centered)))
    table(lines, ["TRAIN context", "Bank group", "Raw squared share %", "Centered variance share %", "Difference pp"], bank_rows)
    lines.extend(["Large constant offsets can dominate uncentered magnitude while contributing little across-row variation. Uncentered L2 dominance alone does not establish missing information in centered task-head inputs.",
                  "Every coordinate and feature/channel group is preserved in the detailed summary; no normalizer, projection or encoder export is fitted here.", "",
                  "## Existing ridge prefix and suffix contributions", "", LEGEND + " · all112 fitted-TRAIN rows, physical prediction units.", "",
                  "These are components of the original fitted predictions. Prefix shift compares the candidate's fitted prefix term with the parent predictor; the old parent information itself remains exact.",
                  "Cells show population SD/RMS when available, otherwise mean[min,max]. Correlations and every task/context component remain in JSON.", ""])
    for task in TASKS:
        lines.extend([task + ". " + LEGEND + " · all112 TRAIN; unit " + lookup[MODELS[0], task, "all112"]["unit"] + ".", ""])
        table(lines, ["Method", "Prefix shift", "Suffix contribution", "Prediction delta", "Parent residual"],
              [(model,) + tuple(contribution_text(lookup[model, task, "all112"]["components"][name]) for name in
                                 ("prefix_shift", "suffix", "prediction_delta", "parent_residual")) for model in MODELS])
    lines.extend(["## Audit, costs and next comparison", "",
                  f"Reader admission passed {diagnosis['admission_checks']['checks']} checks; independent scalar audit passed {audit['checks']} checks.",
                  f"Maximum independent normalized error: {audit['maximum_normalized_error']:.17g}, limit1e−10. These numerical checks do not change encoder quality gates (1e−12 arithmetic tolerance).",
                  f"Reader wall time including I/O: {verification['reader_seconds_including_io']:.6f}s; reader+independent audit wall time including I/O: {verification['reader_and_audit_seconds_including_io']:.6f}s.",
                  "GPU training and head-fitting times are not applicable; neither operation runs. Reporting/hash capture time is recorded separately in completion.",
                  "The twelve actual opened inputs, card/addendum, source/binary provenance and preexisting integration artifacts passed the wrapper's before/after checks.",
                  "The earlier complete two-repository77,831-file preservation scan and14-checkpoint/54-pair preflight remain bound; every historical source/checkpoint is preserved.", "",
                  "Next use these TRAIN associations to specify one separate generic scale/variation-bank comparison, keeping exact initialization, native dimensions, fixed heads, budget and native36 protection gates declared before quality.",
                  "Compare the proposed statistic directly with the existing raw-difference bank definition; assess signed drift separately from price-level/relative-scale effects and centered coordinate variation.",
                  "No new mechanism is implemented or admitted by this report. The v25 objective remains stopped; no coefficient, seed, head, rate or budget rescue follows.",
                  "A successful new candidate must retain the recorded market and conditional TEMPO-5 gains and then receive separate chronological confirmation. No causal explanation or new best encoder is established here.", ""])
    dump(out / "table_rows.json", table_rows)
    exclusive(report, ("\n".join(lines).rstrip() + "\n").encode())
    inventory = dict(protocol=PROTOCOL, run=relative(run), inputs=list(input_records.values()), run_assets=assets,
                     sources=list(source_records.values()), binary_provenance=binaries, reused_neural_source=record(NEURAL_SOURCE),
                     local_witness_policy="Raw arrays and numerical witnesses are hashed only; they remain in ignored local output and are not durable captures.")
    dump(out / "asset_hashes.json", inventory)
    completion = dict(protocol=PROTOCOL, status="complete", date="2026-10-15", disposition="audited-saved-TRAIN-diagnosis-only",
                      run=relative(run), dataset_id="MKT-HIST-2", training_rows=112, diagnostic_input_archives=12,
                      new_encoder_calls=0, new_encoder_updates=0, new_head_fits=0, new_scaler_fits=0, held_out_tensor_inputs=0,
                      new_neural_scoring=False, neural_scores_reused_from_frozen_metadata=True, primary_changed=False,
                      new_validation_quality=False, card_sha256=CARD, addendum_sha256=ADDENDUM, input_manifest_sha256=MANIFEST,
                      preflight_sha256=PREFLIGHT, report_sha256=sha(report), table_rows_sha256=sha(out / "table_rows.json"),
                      asset_hashes_sha256=sha(out / "asset_hashes.json"), reporter_sha256=sha(Path(__file__)),
                      diagnosis_sha256=sha(diagnosis_path), independent_audit_sha256=sha(run / "independent-audit.json"),
                      wrapper_verification_sha256=sha(run / "verification.json"), independent_arithmetic_checks=audit["checks"],
                      independent_maximum_normalized_error=audit["maximum_normalized_error"],
                      reader_wall_seconds_including_io=verification["reader_seconds_including_io"],
                      reader_audit_wall_seconds_including_io=verification["reader_and_audit_seconds_including_io"],
                      reporting_and_hash_capture_seconds=time.perf_counter() - started, sources=list(source_records.values()),
                      source_assets=assets, raw_array_or_witness_exports=0, tensor_archive_exports=0)
    dump(out / "completion.json", completion)
    members = [dict(path=path.relative_to(out).as_posix(), sha256=sha(path), bytes=path.stat().st_size)
               for path in sorted(out.rglob("*")) if path.is_file()]
    dump(out / "durable_files.json", dict(protocol=PROTOCOL, report=record(report), files=members,
                                         policy="Immutable captured report and evidence; later additive receipts remain separate."))
    require(not any(path.suffix == ".pt" or "witness" in path.name.lower() for path in out.rglob("*") if path.is_file()),
            "Raw tensor/witness file must remain in ignored local output")
    for name, expected in input_records.items():
        verify_record(ROOT / name, expected)
    for name, expected in source_records.items():
        verify_record(ROOT / name, expected)
    print(json.dumps(dict(status="complete", protocol=PROTOCOL, run=relative(run), task_rows=len(rows),
                          report=relative(report), independent_checks=audit["checks"], new_encoder_calls=0, new_head_fits=0)))


if __name__ == "__main__":
    main()
