#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Artificial two-update study-to-utility workflow; no market quality measurement."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

execution_log = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, succeeds=True):
    result = subprocess.run([str(part) for part in command], text=True, capture_output=True)
    if execution_log is not None:
        with execution_log.open("a") as stream:
            stream.write(f"COMMAND {list(map(str, command))!r}\nEXIT {result.returncode}\n"
                         f"STDOUT\n{result.stdout}\nSTDERR\n{result.stderr}\n")
    if (result.returncode == 0) != succeeds:
        raise RuntimeError(f"Unexpected exit {result.returncode}: {command}\n{result.stdout}\n{result.stderr}")
    return result


def checked_manifest(archive):
    manifest = json.loads(Path(str(archive) + ".manifest.json").read_text())
    assert manifest["archive_sha256"] == digest(archive), "raw saved bytes match sidecar"
    assert len(manifest["components"]) == 3, "joint archive owns exactly three components"
    return manifest


def fields_digest(domain, fields):
    result = hashlib.sha256()
    for value in ["cuwacunu_artifact_record_v1", domain, str(len(fields)), *fields]:
        raw = value.encode("utf-8")
        result.update(len(raw).to_bytes(8, "little"))
        result.update(raw)
    return result.hexdigest()


def augment_issuance_source_closure(fixture):
    """Extend this new artificial fixture without changing its generator bytes."""
    root = Path.cwd().resolve()
    tree = root / "code/latent_market/forecast_issuance/versions/v1"
    assert tree.is_dir(), "version-owned issuance source tree exists"
    manifest = fixture / "source-closure.sha256"
    previous_hash = digest(manifest)
    pins = {}
    for line in manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        sha, name = line.split(None, 1)
        source = Path(name)
        if not source.is_absolute():
            source = manifest.parent / source
        assert digest(source) == sha, "original fixture source pins remain exact"
        pins[source.resolve()] = sha
    for source in tree.rglob("*"):
        if source.is_file() and not source.is_symlink():
            pins[source.resolve()] = digest(source)
    assert any(source.is_relative_to(tree) for source in pins)
    manifest.write_text("# Synthetic source superset extended for forecast issuance; no production closure claim.\n"
                        + "".join(f"{sha}  {source.as_posix()}\n" for source, sha in sorted(pins.items())))
    config = fixture / "study.conf"
    text, count = re.subn(r"(\bSOURCE_CLOSURE_SHA256\s*=\s*)" + previous_hash + r"(\s*;)",
                          lambda match: match.group(1) + digest(manifest) + match.group(2), config.read_text())
    assert count == 1, "the generated study pins its augmented source closure exactly once"
    config.write_text(text)
    # Keep the generator's local file inventory internally consistent too.
    inventory = fixture / "artifacts.sha256"
    updated = []
    for line in inventory.read_text().splitlines():
        if not line or line.startswith("#"):
            updated.append(line)
            continue
        sha, name = line.split(None, 1)
        path = Path(name)
        if not path.is_absolute():
            path = fixture / path
        if path.resolve() in {manifest.resolve(), config.resolve()}:
            sha = digest(path)
        else:
            assert digest(path) == sha, "all other generated fixture files remain unchanged"
        updated.append(f"{sha}  {name}")
    inventory.write_text("\n".join(updated) + "\n")


def integer_setting(config, name):
    return int(re.search(r"\b" + name + r"\s*=\s*([0-9]+)\s*;", config.read_text()).group(1))


def boolean(value):
    assert value in {"0", "1", "false", "true"}, "explicit serialized Boolean"
    return value in {"1", "true"}


def sequence(value):
    if value.startswith("["):
        return [str(item) for item in json.loads(value)]
    return re.split(r"[;|\s]+", value) if value else []


def checked_issuance(fixture, study):
    publication = json.loads((study / "state-issuance.json").read_text())
    assert all(isinstance(value, str) for value in publication.values()), "publication uses exact string bindings"
    assert publication["schema"] == "forecast-issuance-publication-v1"
    assert publication["policy_sha256"] == "18fcaeab478e0b66dfe2fca132cc4c5d8ae3de368eb4a74d7428fdf9bfef2805"
    assert re.fullmatch("[0-9a-f]{64}", publication["contract_sha256"])
    assert publication["prediction_sha256"] == digest(study / "state-predictions.pt")
    assert publication["ledger_sha256"] == digest(study / "state-issuance.csv")
    assert publication["timeline_sha256"] == digest(fixture / "timeline.bin")
    assert publication["pairs_sha256"] == digest(fixture / "pairs.bin")
    checkpoint = study / "inference.pt"
    expected_coordinate = "synthetic-latent-market-backbone-v1:Target State Representation Network:" + digest(checkpoint)
    assert publication["target_coordinate_id"] == expected_coordinate
    study_receipt = dict(line.split(" ", 1) for line in (study / "study.txt").read_text().splitlines()[1:])
    assert publication["bundle_ref"] == study_receipt["bundle_ref"]
    encoder_sidecar = fixture / "fixture-checkpoint.txt.manifest.json"
    encoder = json.loads(encoder_sidecar.read_text())["components"][0]
    inference_sidecar = Path(str(checkpoint) + ".manifest.json")
    predictive = checked_manifest(checkpoint)["components"]
    selected = [(encoder["module_ref"]["component"], digest(encoder_sidecar), encoder["model_ref"])]
    selected += [(item["module_ref"]["component"], digest(inference_sidecar), item["model_ref"]) for item in predictive]
    fields = [field for component, sidecar, model in sorted(selected) for field in (sidecar, component, model)]
    assert publication["bundle_ref"] == fields_digest("BundleRef", fields), "publication binds all four exact compatible parents"
    base, interval = 1_800_000_000_000, 60_000
    fit = integer_setting(fixture / "study.conf", "FIT_CUTOFF_MS")
    assert int(publication["encoder_fit_ms"]) == base
    assert all(int(publication[key]) == fit for key in ["online_fit_ms", "target_fit_ms", "dynamics_fit_ms"])
    assert int(publication["history_steps"]) == 4 and int(publication["spacing_ticks"]) == 1
    assert int(publication["state_width"]) == 4
    origins = [*range(5, 120), *range(123, 179)]
    with (study / "state-issuance.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert int(publication["origin_count"]) == len(origins) == 171
    assert int(publication["prediction_count"]) == len(rows) == 342
    indexed = {(row["pair_id"], int(row["horizon_ticks"])): row for row in rows}
    expected = {(f"synthetic-origin-{tick}", horizon) for tick in origins for horizon in [1, 4]}
    assert len(indexed) == len(rows) and set(indexed) == expected, "every original origin/horizon is retained exactly once"
    for tick in origins:
        for horizon in [1, 4]:
            row = indexed[(f"synthetic-origin-{tick}", horizon)]
            origin = base + tick * interval
            assert int(row["origin_ms"]) == origin and int(row["decision_ms"]) == origin + 5
            assert row["target_coordinate_id"] == expected_coordinate and row["bundle_ref"] == publication["bundle_ref"]
            assert all(row[key] == publication[key] for key in ["encoder_fit_ms", "online_fit_ms", "target_fit_ms", "dynamics_fit_ms"])
            endpoints = [base + step * interval for step in range(tick - 3, tick + 1)]
            assert list(map(int, sequence(row["input_endpoints_ms"]))) == endpoints
            available = list(map(int, sequence(row["input_available_ms"])))
            assert len(available) == len(endpoints) == 4
            assert sequence(row["input_window_ids"]) == [f"synthetic-latent-market-backbone-v1:window:{endpoint}" for endpoint in endpoints]
            if boolean(row["valid"]):
                assert int(row["issuance_reasons"]) == 0 and int(row["original_input_reasons"]) == 0
                assert all(int(row[key]) < origin for key in ["encoder_fit_ms", "online_fit_ms", "target_fit_ms", "dynamics_fit_ms"])
                assert available == [endpoint + 5 for endpoint in endpoints]
                assert max(available) <= int(row["decision_ms"])
            else:
                assert int(row["issuance_reasons"]) != 0, "invalid associations retain issuance reasons"
            assert boolean(row["valid"]) == (126 <= tick <= 178), "strict timing and full history support determine issuance"
    for horizon in [1, 4]:
        delayed = indexed[("synthetic-origin-119", horizon)]
        assert int(delayed["decision_ms"]) == fit and int(delayed["origin_ms"]) < fit
        assert not boolean(delayed["valid"]) and int(delayed["issuance_reasons"]) & 0b1110 == 0b1110
    for tick in [176, 177, 178]:
        late = indexed[(f"synthetic-origin-{tick}", 4)]
        assert not boolean(late["target_eligible"]) and int(late["target_ledger_reasons"]) != 0
        assert boolean(late["valid"]) and int(late["issuance_reasons"]) == 0, "unavailable future targets never prevent legal issuance"
    return publication, rows


def main():
    global execution_log
    assert Path("/.dockerenv").exists(), "Use the managed development container"
    build = Path(sys.argv[1]).resolve(strict=True)
    parent = Path(tempfile.mkdtemp(prefix="snapshot-pipeline-", dir=os.environ["TMPDIR"]))
    execution_log = parent / "pipeline.log"
    fixture, first, resumed = parent / "fixture", parent / "first", parent / "resumed"
    run([build / "latent_market_v1_pipeline_fixture", fixture])
    augment_issuance_source_closure(fixture)
    config = fixture / "study.conf"
    run([build / "latent_market_v1_study", config, first])
    for name in ["training-1.pt", "training-2.pt", "training.pt", "inference.pt"]:
        manifest = checked_manifest(first / name)
        assert manifest["archive_kind"] == ("inference" if name == "inference.pt" else "training")
        if name != "inference.pt":
            assert len(manifest["resume_identity_sha256"]) == 64
    resume_sidecar = Path(str(first / "training-1.pt") + ".manifest.json")
    run([build / "latent_market_v1_study", config, resumed, first / "training-1.pt", digest(resume_sidecar)])
    expected = checked_manifest(first / "inference.pt")
    actual = checked_manifest(resumed / "inference.pt")
    assert expected["components"] == actual["components"], "resumed inference source/state/config/parents match"
    receipt = (resumed / "study.txt").read_text()
    assert "resume_archive_sha256 " + digest(first / "training-1.pt") in receipt
    assert "bundle_ref " in receipt
    first_issuance, first_rows = checked_issuance(fixture, first)
    resumed_issuance, resumed_rows = checked_issuance(fixture, resumed)
    for left, right in zip(first_rows, resumed_rows):
        assert {key: value for key, value in left.items() if key not in {"bundle_ref", "target_coordinate_id"}} == {
            key: value for key, value in right.items() if key not in {"bundle_ref", "target_coordinate_id"}}, "resume preserves chronological association metadata"
    if digest(first / "inference.pt") == digest(resumed / "inference.pt"):
        assert first_issuance["target_coordinate_id"] == resumed_issuance["target_coordinate_id"]
        assert first_issuance["bundle_ref"] == resumed_issuance["bundle_ref"]

    # Fresh study coordinates must be accepted by utility without archive edits.
    utility_config = parent / "utility.conf"
    run([sys.executable, Path(__file__).with_name("make_utility_fixture.py"),
         fixture, first, utility_config])
    utility_output = parent / "utility"
    run([build / "latent_market_v1_utility", utility_config, utility_output])
    for name in ["readouts.pt", "predictions.pt", "scores.csv", "utility.txt"]:
        assert (utility_output / name).is_file(), "utility publishes every synthetic output"
    assert "encoder_calls 0\nnetwork_updates 0\nTEST_opened false" in (utility_output / "utility.txt").read_text()

    # Re-pin deliberately corrupted copies so raw archive hashes do not mask the
    # coordinate check. Retired tokens and another checkpoint's coordinates fail
    # before a utility output is created; source/historical archives stay intact.
    original_state = first / "state-bank.pt"
    original_state_hash = digest(original_state)
    for mode in ["retired-token", "stale-checkpoint"]:
        wrong_state = parent / (mode + ".pt")
        run([build / "latent_market_v1_coordinate_fixture", original_state, wrong_state, mode])
        wrong_config = parent / (mode + ".conf")
        wrong_config.write_text(utility_config.read_text()
                               .replace(str(original_state), str(wrong_state))
                               .replace(original_state_hash, digest(wrong_state)))
        rejected_output = parent / (mode + "-must-not-exist")
        rejection = run([build / "latent_market_v1_utility", wrong_config, rejected_output], succeeds=False)
        assert "State target coordinate identity mismatch" in rejection.stderr
        assert not rejected_output.exists(), "bad coordinates expose no risk readouts"
    assert digest(original_state) == original_state_hash, "original state archive remains unchanged"

    # Changing only the selected parent cannot relabel already captured arrays.
    config_text = config.read_text()
    encoder_ref = json.loads((fixture / "fixture-checkpoint.txt.manifest.json").read_text())["components"][0]["model_ref"]
    stale = fixture / "stale-parent.conf"
    stale.write_text(config_text.replace("ENCODER_MODEL_REF = " + encoder_ref,
                                         "ENCODER_MODEL_REF = " + "0" * 64))
    rejected_leaf = parent / "must-not-exist"
    rejection = run([build / "latent_market_v1_study", stale, rejected_leaf], succeeds=False)
    assert "original Encoder ModelRef" in rejection.stderr
    assert not rejected_leaf.exists(), "lineage failure exposes no output or trained snapshot"
    # A trusted resume pin is mandatory, and corruption fails before continuation.
    run([build / "latent_market_v1_study", config, parent / "missing-pin", first / "training-1.pt"], succeeds=False)
    run([build / "latent_market_v1_study", config, parent / "wrong-pin", first / "training-1.pt", "0" * 64], succeeds=False)
    print("PASS snapshot pipeline: fresh study-to-utility handoff, complete chronological issuance with strict fit cutoffs and target-blind late-target forecasts, retired/wrong-checkpoint coordinate rejection, all checkpoint sidecars, exact resumed component identities, capture lineage rejection and resume pins; synthetic engineering only")
    print("Evidence preserved at " + str(parent))
    with execution_log.open("a") as stream:
        stream.write("PASS synthetic study-to-utility pipeline, complete chronological issuance, coordinate rejections, capture lineage and exact resume\n")


if __name__ == "__main__":
    main()
