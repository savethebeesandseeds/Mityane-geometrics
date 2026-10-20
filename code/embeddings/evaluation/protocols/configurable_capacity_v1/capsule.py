#!/usr/bin/env python3
"""Independent saved-information tensor association audit.

Maintained capsule metadata commands use capsule.mjs. This module retains the
scientific bind mode and the original contract functions as reference source.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

MASTERS = (941101, 941102, 941103)
PROTOCOL = "configurable-capacity-v1"
BINARIES = ("embedding_configurable_capacity", "embedding_configurable_capacity_data",
            "embedding_configurable_capacity_readout",
            "code/protocols/configurable_capacity_v1/generator_test")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    # Admission/integrity checks must also run under PYTHONOPTIMIZE or python -O.
    if not condition:
        raise ValueError(message)


def new(path, text):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def direct(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            f"Direct absolute path required: {path}")


def document(path):
    direct(path)
    require(path.is_file() and path.stat().st_nlink == 1, f"Single-link regular receipt required: {path}")
    return json.loads(path.read_text())


def verify_admission(root):
    admission = document(root / "admission.json")
    require(admission == {"protocol": PROTOCOL, "status": "admitted", "quality_encoder_runs": 0},
            "Completed engineering admission required before quality generation")
    engineering = document(root / "runner-engineering" / "complete.json")
    require(engineering == {"status": "passed", "protocol": PROTOCOL, "master": 941000,
                            "profiles": ["A", "B", "C"], "updates_each": 2, "CPU_encoder_calls": 0},
            "Actual two-update CUDA admission required for all three profiles")
    receipt = document(root / "freeze" / "receipt.json")
    require(receipt["protocol"] == PROTOCOL
            and receipt["source_id"] == digest(root / "freeze" / "sources.sha256")
            and receipt["card_sha256"] == digest(root / "freeze" / "card.md")
            and receipt["config_sha256"] == digest(root / "freeze" / "config.conf"),
            "Admission source/card/settings receipt differs")


def freeze(root, build):
    direct(root)
    direct(build)
    out = root / "freeze"
    out.mkdir()
    sources = subprocess.check_output(
        ["make", "-s", "print-configurable-capacity-sources", f"BUILD_DIR={build}"], text=True)
    paths = sources.splitlines()
    require(paths and paths == sorted(set(paths)) and all(Path(p).is_file() for p in paths),
            "Closed sorted source list required")
    new(out / "source-list.txt", sources)
    manifest = "".join(f"{digest(Path(p))}  {p}\n" for p in paths)
    new(out / "sources.sha256", manifest)
    source_id = digest(out / "sources.sha256")
    require(subprocess.check_output([str(build / BINARIES[0]), "--source-id"], text=True).strip() == source_id,
            "Compiled producer does not match captured sources")
    expected = {p: digest(Path(p)) for p in paths}
    require(manifest == "".join(f"{expected[p]}  {p}\n" for p in paths),
            "Source changed during manifest capture")
    for p in paths:
        destination = out / "source" / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
        require(digest(destination) == expected[p] == digest(Path(p)),
                f"Source changed during snapshot: {p}")
    for name, source in (
        ("card.md", "code/embeddings/evaluation/cards/configurable_capacity_v1.md"),
        ("config.conf", "code/embeddings/encoders/raw_patch_bottleneck_mae/config/configurable_spectral.conf")):
        shutil.copyfile(source, out / name)
    new(out / "binaries.sha256", "".join(f"{digest(build / p)}  {build / p}\n" for p in BINARIES))
    receipt = {"protocol": "configurable-capacity-v1", "source_id": source_id,
               "card_sha256": digest(out / "card.md"), "config_sha256": digest(out / "config.conf"),
               "build_directory": str(build), "source_files": len(paths), "git_head":
               subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    new(out / "receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def _information_signature(path):
    direct(path)
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1,
            f"Single-link regular information role required: {path}")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mode,
            value.st_mtime_ns, value.st_ctime_ns)


def _admit_information_files(data):
    # Scientific input boundary: admit all nine observation/witness pairs before
    # the first codec load. No payload is opened by this metadata-only preflight.
    paths = [data / f"master-{master}/{view}{suffix}.pt"
             for master in MASTERS
             for view in ("training", "validation", "validation-deleted")
             for suffix in ("", "-information")]
    admitted, identities = {}, set()
    for path in paths:
        signature = _information_signature(path)
        require(signature[:2] not in identities, "Information roles must not alias")
        identities.add(signature[:2])
        admitted[path] = signature
    require(len(admitted) == 18, "Complete eighteen-role information matrix")
    return admitted


def bind(root):
    data = root / "quality-data"
    information_path = data / "information-report.json"
    information_signature = _information_signature(information_path)
    information = document(information_path)
    require(information["passed"] is True and len(information["masters"]) == 3,
            "All three information cohorts must pass")
    require([m["master"] for m in information["masters"]] == list(MASTERS),
            "Information master order differs")
    for m in information["masters"]:
        require(m["history"] == 128 and m["passed"] is True and len(m["views"]) == 3,
                "Information history/population differs")
        require([v["view"] for v in m["views"]] == ["training", "validation", "validation-deleted"],
                "Exactly TRAIN, intact and deleted information views required")
        for v in m["views"]:
            expected = 512 if v["view"] == "training" else 256
            require(v["total"] == v["supported"] == v["correct"] == expected and v["passed"] is True,
                    "Every declared information row must be supported and correct")
    files = [f"master-{m}/{v}.pt" for m in MASTERS
             for v in ("training", "validation", "validation-deleted")]
    require(len(files) == 9, "Nine observation roles required")
    admitted = _admit_information_files(data)
    def unchanged(path):
        require(_information_signature(path) == admitted[path],
                f"Information role changed after admission: {path}")
    codec_path = Path("/mityane/.archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py")
    require(digest(codec_path) == "4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d",
            "Pinned saved archive codec differs")
    spec = importlib.util.spec_from_file_location("capacity_saved_codec", codec_path)
    codec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = codec
    spec.loader.exec_module(codec)
    evidence = [information_path]
    for p in files:
        witness = data / p.replace(".pt", "-information.pt")
        unchanged(data / p)
        unchanged(witness)
        original, decisions = codec.load(data / p), codec.load(witness)
        unchanged(data / p)
        unchanged(witness)
        rows = 512 if p.endswith("/training.pt") else 256
        require(decisions["valid"]["dtype"] == "BoolStorage" and decisions["valid"]["shape"] == [rows],
                "Information validity schema differs")
        require(all(decisions["valid"]["values"]), "Unsupported information decision")
        for key in ("labels_scoring_only", "source_ids"):
            require(original[key]["dtype"] == decisions[key]["dtype"] == "LongStorage"
                    and original[key]["shape"] == decisions[key]["shape"] == [rows],
                    f"Information label/source schema differs: {key}")
            require(list(original[key]["values"]) == list(decisions[key]["values"]),
                    f"Information role association differs: {key}")
        require(list(decisions["predictions"]["values"]) == list(original["labels_scoring_only"]["values"]),
                "Information prediction differs from scoring-only label")
        require(all(n == 119 for n in decisions["frequency_pairs_evaluated"]["values"]),
                "Closed information frequency search differs")
        diag = list(decisions["diagnostics"]["values"])
        require(decisions["diagnostics"]["shape"] == [rows, 4] and all(math.isfinite(v) for v in diag),
                "Finite information diagnostics required")
        for r in range(rows):
            slow, fast, residual, margin = diag[4*r:4*r+4]
            require(any(abs(slow - 1/n) < 1e-15 for n in range(32, 49))
                    and any(abs(fast - 1/n) < 1e-15 for n in range(8, 15))
                    and 0 <= residual <= .0025 and .12 <= abs(margin) <= .60,
                    "Information period/residual/phase gate differs")
        evidence.append(witness)
    for path in admitted:
        unchanged(path)
    hashes = {}
    for path in admitted:
        unchanged(path)
        hashes[path] = digest(path)
        unchanged(path)
    require(_information_signature(information_path) == information_signature,
            "Information report changed during validation")
    hashes[information_path] = digest(information_path)
    require(_information_signature(information_path) == information_signature,
            "Information report changed during manifest hashing")
    new(root / "freeze" / "inputs.sha256", "".join(f"{hashes[data / p]}  {p}\n" for p in sorted(files)))
    new(root / "freeze" / "information.sha256",
        "".join(f"{hashes[p]}  {p}\n" for p in evidence))
    print("All nine fixed role archives bound; observed-only information gate passed.")


def finish(root):
    require(not (root / "artifacts.sha256").exists() and not (root / "complete.json").exists(),
            "Completed capsule cannot be finalized again")
    receipt = document(root / "freeze" / "receipt.json")
    bindings = {"source_fingerprint": receipt["source_id"], "card_sha": receipt["card_sha256"],
                "config_sha": receipt["config_sha256"],
                "input_manifest_sha": digest(root / "freeze" / "inputs.sha256")}
    for profile in "ABC":
        for master in MASTERS:
            p = root / "profiles" / profile / f"master-{master}"
            result = document(p / "complete.json")
            require(result["status"] == "complete" and result["updates"] == 512
                    and result["protocol"] == PROTOCOL and result["profile"] == profile
                    and result["master"] == master
                    and result["report_sha256"] == digest(p / "report.json")
                    and all(result[key] == value for key, value in bindings.items()),
                    "Completed producer counter/report/bindings differ")
            for point in (0, 512):
                for name in ("checkpoint.pt", "features.pt", "reconstruction-training.pt", "reconstruction-validation.pt"):
                    path = p / f"point-{point}" / name
                    direct(path)
                    require(path.is_file() and path.stat().st_nlink == 1,
                            f"Single-link retained producer archive required: {path}")
    audit = document(root / "readouts" / "complete.json")
    require(audit["status"] == "complete" and audit["protocol"] == PROTOCOL
            and audit["saved_arithmetic_fits_replayed"] == 81
            and audit["report_sha256"] == digest(root / "readouts" / "report.json"),
            "Completed saved-reader counter/report differs")
    summary = document(root / "summary.json")
    require(summary["protocol"] == PROTOCOL and summary["saved_arithmetic_fits_replayed"] == 81
            and summary["source_fingerprint"] == bindings["source_fingerprint"]
            and summary["card_sha"] == bindings["card_sha"]
            and summary["input_manifest_sha"] == bindings["input_manifest_sha"]
            and (root / "report.md").is_file(), "Completed publisher/report bindings differ")
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    # Content-address the entire retained capsule, including source captures and inputs.
    new(root / "artifacts.sha256", "".join(f"{digest(p)}  {p.relative_to(root)}\n" for p in paths))
    new(root / "complete.json", json.dumps({"protocol": "configurable-capacity-v1", "status": "complete",
        "encoder_trajectories": 9, "quality_masters": list(MASTERS), "completed_updates_per_run": 512,
        "saved_head_fits": 81, "artifacts_manifest_sha256": digest(root / "artifacts.sha256")}, indent=2) + "\n")
    print(f"CAPACITY_COMPLETE {root}")


def main():
    require(Path("/.dockerenv").exists(), "Run inside the managed development container.")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("bind",), help="metadata commands migrated to capsule.mjs")
    p.add_argument("root", type=Path)
    a = p.parse_args()
    direct(a.root)
    bind(a.root)


if __name__ == "__main__":
    main()
