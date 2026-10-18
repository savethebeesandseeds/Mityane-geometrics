#!/usr/bin/env python3
"""Bind an explicit TRAIN-only diagnostic input set without discovering payloads.

--role-plan reads pinned JSON/hash metadata only. --freeze, after a supplied
frozen-card check, hashes exactly 195 declared TRAIN files without decoding them.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

PROTOCOL = "training-objective-diagnostic-v1"
MASTERS = (4404, 5505, 6606, 7707, 8808)
TRAINING_NAMESPACE = "native-development-v1/lag_sign"
PARENTS = (
    {"id": "v4", "tag": "RPB-v4", "policy_id": "",
     "capsule": "output/runs/rpb-context-replication/context-replication-JjNEUc", "cohort_prefix": "reference",
     "inventory_sha256": "13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9",
     "orchestration_source_id": "9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828",
     "training_producer_source_id": "2fff50c48605ee21dbbfd28d6989bdcf6b5cd057785c2acb19419571af48b895",
     "core_writer_source_id": "587f2423758c3e70c2c7665d9e7b59bfe10af4c7f9aaba23f5318b3c55b13bca"},
    {"id": "v7", "tag": "RPB-v7", "policy_id": "rpb-training-context-deletion-015-v1",
     "capsule": "output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2", "cohort_prefix": "results/candidate-development",
     "inventory_sha256": "d4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821",
     "orchestration_source_id": "f22cc8d5f2ac8af3e6a6056c6ad2693ffcc82cf39e8744c32fb13038710c5311",
     "training_producer_source_id": "d8ed2771d1615beabccbead081847813b2562b7b1f9b5347d4825c075ccc27a7",
     "core_writer_source_id": "e379b8f8abb102c256fb843466b4fbfbecb6fdc21298b862fd0b6dfeb47b853e"},
    {"id": "v8", "tag": "RPB-v8", "policy_id": "rpb-training-context-balanced-030-v1",
     "capsule": "output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV", "cohort_prefix": "results/candidate-development",
     "inventory_sha256": "e0cdfbc729c8d989e16bb5e8515b5c5d73b7b445bdf91d2d938d88e09c969ac7",
     "orchestration_source_id": "78ab6ffdb0a3edf141e0ad81e58fe10f36e276d4b7d8af1288859004b5f0f38c",
     "training_producer_source_id": "3532deae194828f2640ff4fadacbcd94bbccfb8af6eaf37182cb8137ae84cf41",
     "core_writer_source_id": "073ec84b7c781cf48a578e58a4c9602c6b765c9c23f6065f27338780be8a1835"},
)
PARENT_METADATA = (
    ("parent-inputs.json", "220767022994e5f7bab0497057034593de4cc47e3bf9d1f64de479e8de0fcd6a"),
    ("parent-inputs.sha256", "d65ff6deacac126c29011ae3097be9bd1e2b4d520249119a61b38e9f95423b61"),
)
COHORT_FIELDS = (
    ("controlled_training_path", "controlled-training.pt", "TRAIN-observations-labels-source-order"),
    ("checkpoint_path", "milestone-512/checkpoint.pt", "frozen-checkpoint"),
    ("checkpoint_audit_path", "milestone-512/checkpoint.pt.audit.pt", "TRAIN-producer-audit"),
    ("checkpoint_scaler_path", "milestone-512/checkpoint.pt.scaler.pt", "TRAIN-fitted-model-scaler"),
    ("checkpoint_raw_training_path", "milestone-512/checkpoint.pt.training-raw.pt", "checkpoint-TRAIN-observations"),
    ("native_training_features_path", "milestone-512/native-training.pt", "TRAIN-native-export"),
    ("training_reconstruction_path", "milestone-512/training-reconstruction.pt", "TRAIN-fixed-query"),
)
TSV_COLUMNS = (
    "input_id", "tag", "master_seed", "budget", "expected_policy_id", "training_namespace",
    "orchestration_source_id", "training_producer_source_id", "core_writer_source_id",
) + tuple(field[0] for field in COHORT_FIELDS) + tuple(
    value for rep in (1, 2, 3) for value in (f"rep_{rep}_native_fit_path", f"rep_{rep}_training_predictions_path"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_relative(path):
    result = Path(path)
    require(isinstance(path, str) and not result.is_absolute() and path == result.as_posix() and
            ".." not in result.parts and not any(char in path for char in "\t\r\n"), "safe exact relative role path")
    require(not {"validation", "test", "testing", "stress"}.intersection(re.split("[^a-z0-9]+", path.lower())),
            "held-out role path forbidden")
    return result


def role_plan(repo):
    metadata, indices = [], {}
    for parent in PARENTS:
        path = repo / parent["capsule"] / "artifact-integrity.json"
        data = path.read_bytes()
        require(sha(data) == parent["inventory_sha256"], "pinned inventory metadata " + parent["id"])
        inventory = json.loads(data)
        entries = {}
        for item in inventory["files"]:
            relative = item["path"]
            require(relative not in entries and isinstance(item["bytes"], int) and item["bytes"] >= 0 and
                    re.fullmatch("[0-9a-f]{64}", item["sha256"]), "unambiguous inventory record")
            entries[relative] = item
        require(inventory["file_count"] == len(entries), "inventory declared count")
        indices[parent["id"]] = entries
        metadata.append({"path": str(path), "bytes": len(data), "sha256": sha(data), "role": "pinned-inventory-metadata"})
    base = repo / PARENTS[2]["capsule"]
    parent_rows = None
    for filename, expected in PARENT_METADATA:
        path = base / filename
        data = path.read_bytes()
        require(sha(data) == expected, "pinned V8 parent boundary metadata " + filename)
        metadata.append({"path": str(path), "bytes": len(data), "sha256": sha(data), "role": "prior-explicit-role-metadata"})
        if filename.endswith(".json"):
            parent_rows = {(item["parent_id"], item["path"]): item for item in json.loads(data)["inputs"]}
    require(len(parent_rows) == 326, "prior role metadata count; not a payload allowlist")
    inputs, instances = [], []

    def add(instance, physical_parent, relative, role, field):
        safe_relative(relative)
        item = indices[physical_parent["id"]].get(relative)
        require(item is not None, "exact metadata role exists: " + physical_parent["id"] + "/" + relative)
        if physical_parent["id"] != "v8":
            bound = parent_rows.get((physical_parent["id"], relative))
            require(bound is not None and (bound["bytes"], bound["sha256"]) == (item["bytes"], item["sha256"]),
                    "original selected role agrees with V8 prior boundary")
        absolute = str(repo / physical_parent["capsule"] / relative)
        require(not any(char in absolute for char in "\t\r\n"), "single-line absolute binding")
        instance[field] = absolute
        inputs.append({"input_id": instance["input_id"], "parent_id": physical_parent["id"],
                       "path": relative, "manifest_path": physical_parent["id"] + "/" + relative,
                       "absolute_path": absolute, "kind": role, "api_field": field,
                       "tag": instance["tag"], "master_seed": instance["master_seed"], "budget": 512, "split": "TRAIN",
                       "expected_policy_id": instance["expected_policy_id"], "bytes": item["bytes"], "sha256": item["sha256"],
                       "asset_writer_orchestration_source_id": physical_parent["orchestration_source_id"],
                       "asset_parent_inventory_sha256": physical_parent["inventory_sha256"],
                       "original_orchestration_source_id": instance["orchestration_source_id"],
                       "original_training_producer_source_id": instance["training_producer_source_id"],
                       "original_core_writer_source_id": instance["core_writer_source_id"]})

    for master in MASTERS:
        for parent in PARENTS:
            instance = {"input_id": f"seed-{master}-{parent['tag']}-updates-512", "tag": parent["tag"],
                        "master_seed": master, "budget": 512, "expected_policy_id": parent["policy_id"],
                        "training_namespace": TRAINING_NAMESPACE,
                        "orchestration_source_id": parent["orchestration_source_id"],
                        "training_producer_source_id": parent["training_producer_source_id"], "core_writer_source_id": parent["core_writer_source_id"]}
            prefix = parent["cohort_prefix"] + f"/seed-{master}-lag_sign/"
            for field, suffix, role in COHORT_FIELDS:
                add(instance, parent, prefix + suffix, role, field)
            for rep in (1, 2, 3):
                prefix = f"results/readouts/{instance['input_id']}/rep-{rep}/"
                add(instance, PARENTS[2], prefix + "native-fit.pt", "TRAIN-fitted-native-heads", f"rep_{rep}_native_fit_path")
                add(instance, PARENTS[2], prefix + "native-training-predictions.pt", "TRAIN-native-head-predictions", f"rep_{rep}_training_predictions_path")
            instances.append(instance)
    require(len(inputs) == 195 and len({item["absolute_path"] for item in inputs}) == 195 and len(instances) == 15,
            "exact195 physical roles and15 instances")
    return {"format_version": 1, "protocol": PROTOCOL, "stage": "TRAIN-only-development-diagnostic",
            "metadata_only": True, "payload_bytes_read": 0, "payloads_decoded": 0,
            "parents": list(PARENTS), "metadata_sources": metadata, "inputs": inputs, "instances": instances,
            "instances_tsv_columns": list(TSV_COLUMNS), "input_count": 195, "instance_count": 15,
            "metadata_declared_bytes": sum(item["bytes"] for item in inputs),
            "typed_cohort_contract": {"rows": 256, "source_pairs": 128, "shape": [256, 3, 32, 3],
                "raw_dtype": "float64", "observation_mask_dtype": "bool", "source_ids_key": "source_ids_json",
                "source_ids_encoding": "UTF8 uint8 JSON array in controlled_training_path; validate actual IDs against TRAIN audit fit_source_manifest",
                "labels_key": "labels_scoring_only", "labels_dtype": "int64", "label_access": "TRAIN scoring/grouping only; not encoder fitting",
                "channel_ids": [0, 1, 2], "feature_units": "unitless,unitless,unitless", "training_namespace": TRAINING_NAMESPACE,
                "native_width": 32, "native_post_encoder_pca": False, "readout_repetitions": [2701, 2802, 2903]},
            "scope": "explicit TRAIN-only files; no point0, VAL manifest/payload, raw/PCA fits, TEST/stress or directory discovery"}


def freeze(plan, args):
    require(args.output and args.frozen_card and re.fullmatch("[0-9a-f]{64}", args.frozen_card_sha256 or ""),
            "freeze requires new output and exact frozen-card path/SHA")
    card = args.frozen_card.resolve(strict=True)
    require(file_sha(card) == args.frozen_card_sha256, "card verified before first allowed payload hash")
    output = args.output.resolve()
    require(not output.exists(), "exclusive input output directory required")
    require(output.is_relative_to(args.repo_root / "output/runs/rpb-training-objective-diagnostic"),
            "new diagnostic capsule input directory required")
    for parent in PARENTS:
        require(not output.is_relative_to(args.repo_root / parent["capsule"]), "never write into historical parent")
    for item in plan["inputs"]:
        path = Path(item["absolute_path"])
        require(path.resolve(strict=True) == path and path.is_file() and path.stat().st_size == item["bytes"],
                "exact physical role, size and no symlink alias: " + item["manifest_path"])
        require(file_sha(path) == item["sha256"], "actual allowed TRAIN bytes match frozen inventory: " + item["manifest_path"])
    for item in plan["metadata_sources"]:
        require(file_sha(Path(item["path"])) == item["sha256"], "metadata unchanged during freeze")
    require(file_sha(card) == args.frozen_card_sha256, "frozen card unchanged")
    output.mkdir(parents=True, exist_ok=False)
    plan_bytes = json_bytes(plan)
    manifest = "".join(item["sha256"] + "  " + item["absolute_path"] + "\n" for item in sorted(plan["inputs"], key=lambda item: item["absolute_path"])).encode("utf-8")
    rows = ["\t".join(TSV_COLUMNS)] + ["\t".join(str(instance[column]) for column in TSV_COLUMNS) for instance in plan["instances"]]
    bound = dict(plan, metadata_only=False, actual_allowed_files_hashed=195, payloads_decoded=0,
                 frozen_card_path=str(card), frozen_card_sha256=args.frozen_card_sha256,
                 input_role_plan_sha256=sha(plan_bytes), manifest_sha256=sha(manifest),
                 instances_tsv_sha256=sha(("\n".join(rows) + "\n").encode("utf-8")))
    for name, data in (("input-role-plan.json", plan_bytes), ("inputs.json", json_bytes(bound)),
                       ("inputs.sha256", manifest), ("instances.tsv", ("\n".join(rows) + "\n").encode("utf-8"))):
        with (output / name).open("xb") as stream:
            stream.write(data)
    return {"status": "frozen", "protocol": PROTOCOL, "output": str(output), "inputs": 195, "instances": 15,
            "actual_allowed_files_hashed": 195, "payloads_decoded": 0,
            "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)} for path in sorted(output.iterdir())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--role-plan", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path("/embedding"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen-card", type=Path)
    parser.add_argument("--frozen-card-sha256")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve(strict=True)
    plan = role_plan(args.repo_root)
    result = plan if args.role_plan else freeze(plan, args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
