#!/usr/bin/env python3
"""Stdlib-only capsule receipts; no model, classifier, fitting or tensor runtime."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

MASTERS = (941101, 941102, 941103)
BINARIES = ("embedding_configurable_capacity", "embedding_configurable_capacity_data",
            "embedding_configurable_capacity_readout",
            "code/protocols/configurable_capacity_v1/generator_test")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def new(path, text):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def direct(path):
    assert path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(), path


def freeze(root, build):
    direct(root)
    direct(build)
    out = root / "freeze"
    out.mkdir()
    sources = subprocess.check_output(
        ["make", "-s", "print-configurable-capacity-sources", f"BUILD_DIR={build}"], text=True)
    paths = sources.splitlines()
    assert paths and paths == sorted(set(paths)) and all(Path(p).is_file() for p in paths)
    new(out / "source-list.txt", sources)
    manifest = "".join(f"{digest(Path(p))}  {p}\n" for p in paths)
    new(out / "sources.sha256", manifest)
    source_id = digest(out / "sources.sha256")
    assert subprocess.check_output([str(build / BINARIES[0]), "--source-id"], text=True).strip() == source_id
    for p in paths:
        destination = out / "source" / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
        assert digest(destination) == digest(Path(p))
    for name, source in (
        ("card.md", "code/evaluation/cards/configurable_capacity_v1.md"),
        ("config.conf", "code/encoders/raw_patch_bottleneck_mae/config/configurable_spectral.conf")):
        shutil.copyfile(source, out / name)
    new(out / "binaries.sha256", "".join(f"{digest(build / p)}  {build / p}\n" for p in BINARIES))
    receipt = {"protocol": "configurable-capacity-v1", "source_id": source_id,
               "card_sha256": digest(out / "card.md"), "config_sha256": digest(out / "config.conf"),
               "build_directory": str(build), "source_files": len(paths), "git_head":
               subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    new(out / "receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def bind(root):
    data = root / "quality-data"
    information = json.loads((data / "information-report.json").read_text())
    assert information["passed"] and len(information["masters"]) == 3
    assert [m["master"] for m in information["masters"]] == list(MASTERS)
    for m in information["masters"]:
        assert m["history"] == 128 and m["passed"] and len(m["views"]) == 3
        for v in m["views"]:
            expected = 512 if v["view"] == "training" else 256
            assert v["total"] == v["supported"] == v["correct"] == expected and v["passed"]
    files = [f"master-{m}/{v}.pt" for m in MASTERS
             for v in ("training", "validation", "validation-deleted")]
    assert len(files) == 9
    for p in files:
        direct(data / p)
        assert (data / p).stat().st_nlink == 1
    codec_path = Path("/embedding/code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py")
    assert digest(codec_path) == "4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d"
    spec = importlib.util.spec_from_file_location("capacity_saved_codec", codec_path)
    codec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = codec
    spec.loader.exec_module(codec)
    evidence = [data / "information-report.json"]
    for p in files:
        witness = data / p.replace(".pt", "-information.pt")
        direct(witness)
        original, decisions = codec.load(data / p), codec.load(witness)
        rows = 512 if p.endswith("/training.pt") else 256
        assert decisions["valid"]["dtype"] == "BoolStorage" and decisions["valid"]["shape"] == [rows]
        assert all(decisions["valid"]["values"])
        for key in ("labels_scoring_only", "source_ids"):
            assert original[key]["dtype"] == decisions[key]["dtype"] == "LongStorage"
            assert original[key]["shape"] == decisions[key]["shape"] == [rows]
            assert list(original[key]["values"]) == list(decisions[key]["values"])
        assert list(decisions["predictions"]["values"]) == list(original["labels_scoring_only"]["values"])
        assert all(n == 119 for n in decisions["frequency_pairs_evaluated"]["values"])
        diag = list(decisions["diagnostics"]["values"])
        assert decisions["diagnostics"]["shape"] == [rows, 4] and all(math.isfinite(v) for v in diag)
        for r in range(rows):
            slow, fast, residual, margin = diag[4*r:4*r+4]
            assert any(abs(slow - 1/n) < 1e-15 for n in range(32, 49))
            assert any(abs(fast - 1/n) < 1e-15 for n in range(8, 15))
            assert 0 <= residual <= .0025 and .12 <= abs(margin) <= .60
        evidence.append(witness)
    new(root / "freeze" / "inputs.sha256", "".join(f"{digest(data / p)}  {p}\n" for p in sorted(files)))
    new(root / "freeze" / "information.sha256",
        "".join(f"{digest(p)}  {p}\n" for p in evidence))
    print("All nine fixed role archives bound; observed-only information gate passed.")


def finish(root):
    for profile in "ABC":
        for master in MASTERS:
            p = root / "profiles" / profile / f"master-{master}"
            result = json.loads((p / "complete.json").read_text())
            assert result["status"] == "complete" and result["updates"] == 512
            for point in (0, 512):
                for name in ("checkpoint.pt", "features.pt", "reconstruction-training.pt", "reconstruction-validation.pt"):
                    assert (p / f"point-{point}" / name).is_file(), name
    audit = json.loads((root / "readouts" / "complete.json").read_text())
    assert audit["status"] == "complete" and audit["saved_arithmetic_fits_replayed"] == 81
    assert (root / "report.md").is_file() and (root / "summary.json").is_file()
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    # Content-address the entire retained capsule, including source captures and inputs.
    new(root / "artifacts.sha256", "".join(f"{digest(p)}  {p.relative_to(root)}\n" for p in paths))
    new(root / "complete.json", json.dumps({"protocol": "configurable-capacity-v1", "status": "complete",
        "encoder_trajectories": 9, "quality_masters": list(MASTERS), "completed_updates_per_run": 512,
        "saved_head_fits": 81, "artifacts_manifest_sha256": digest(root / "artifacts.sha256")}, indent=2) + "\n")
    print(f"CAPACITY_COMPLETE {root}")


def main():
    assert Path("/.dockerenv").exists(), "Run inside the managed development container."
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "bind", "finish"))
    p.add_argument("root", type=Path)
    p.add_argument("--build", type=Path)
    a = p.parse_args()
    direct(a.root)
    if a.mode == "freeze":
        freeze(a.root, a.build)
    elif a.mode == "bind":
        bind(a.root)
    else:
        finish(a.root)


if __name__ == "__main__":
    main()
