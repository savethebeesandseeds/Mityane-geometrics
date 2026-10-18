#!/usr/bin/env python3
"""Negative admission/receipt tests. No tensor runtime, model, head or quality data."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import capsule


class ReceiptFixture:
    def __init__(self, root):
        self.root = root
        (root / "freeze").mkdir()
        (root / "runner-engineering").mkdir()
        (root / "readouts").mkdir()
        for name in ("sources.sha256", "card.md", "config.conf", "inputs.sha256"):
            (root / "freeze" / name).write_text("receipt fixture\n")
        self.receipt = {"protocol": capsule.PROTOCOL,
                        "source_id": capsule.digest(root / "freeze" / "sources.sha256"),
                        "card_sha256": capsule.digest(root / "freeze" / "card.md"),
                        "config_sha256": capsule.digest(root / "freeze" / "config.conf")}
        self.write(root / "freeze" / "receipt.json", self.receipt)
        self.write(root / "admission.json", {"protocol": capsule.PROTOCOL,
                                             "status": "admitted", "quality_encoder_runs": 0})
        self.write(root / "runner-engineering" / "complete.json",
                   {"status": "passed", "protocol": capsule.PROTOCOL, "master": 941000,
                    "profiles": ["A", "B", "C"], "updates_each": 2, "CPU_encoder_calls": 0})
        self.bindings = {"source_fingerprint": self.receipt["source_id"],
                         "card_sha": self.receipt["card_sha256"],
                         "config_sha": self.receipt["config_sha256"],
                         "input_manifest_sha": capsule.digest(root / "freeze" / "inputs.sha256")}
        for profile in "ABC":
            for master in capsule.MASTERS:
                run = root / "profiles" / profile / f"master-{master}"
                run.mkdir(parents=True)
                self.write(run / "report.json", {"fixture": "counter/binding checks only"})
                for point in (0, 512):
                    (run / f"point-{point}").mkdir()
                    for name in ("checkpoint.pt", "features.pt", "reconstruction-training.pt",
                                 "reconstruction-validation.pt"):
                        # Opaque placeholders: tests verify state gating, not tensor payloads.
                        (run / f"point-{point}" / name).write_bytes(b"receipt fixture")
                self.write(run / "complete.json",
                           dict(self.bindings, status="complete", protocol=capsule.PROTOCOL,
                                profile=profile, master=master, updates=512,
                                report_sha256=capsule.digest(run / "report.json")))
        self.write(root / "readouts" / "report.json", {"fixture": "reader receipt"})
        self.write(root / "readouts" / "complete.json",
                   {"status": "complete", "protocol": capsule.PROTOCOL,
                    "saved_arithmetic_fits_replayed": 81,
                    "report_sha256": capsule.digest(root / "readouts" / "report.json")})
        self.write(root / "summary.json", dict(self.bindings, protocol=capsule.PROTOCOL,
                                               saved_arithmetic_fits_replayed=81))
        (root / "report.md").write_text("Receipt test fixture, not measured results.\n")

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + "\n")


class CapsuleRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="capacity-receipt-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.fixture = ReceiptFixture(self.root)

    def reject_finish(self, message):
        with self.assertRaisesRegex(ValueError, message):
            capsule.finish(self.root)
        self.assertFalse((self.root / "artifacts.sha256").exists())
        self.assertFalse((self.root / "complete.json").exists())

    def test_failed_admission_cannot_be_treated_as_admitted(self):
        capsule.verify_admission(self.root)
        self.fixture.write(self.root / "admission.json",
                           {"protocol": capsule.PROTOCOL, "status": "failed", "quality_encoder_runs": 0})
        with self.assertRaisesRegex(ValueError, "engineering admission"):
            capsule.verify_admission(self.root)

    def test_admission_requires_actual_all_profile_cuda_engineering(self):
        p = self.root / "runner-engineering" / "complete.json"
        value = json.loads(p.read_text())
        value["profiles"] = ["A", "B"]
        self.fixture.write(p, value)
        with self.assertRaisesRegex(ValueError, "all three profiles"):
            capsule.verify_admission(self.root)

    def test_modified_frozen_settings_reject_admission(self):
        (self.root / "freeze" / "config.conf").write_text("changed settings\n")
        with self.assertRaisesRegex(ValueError, "receipt differs"):
            capsule.verify_admission(self.root)

    def test_duplicate_information_view_is_not_a_complete_population(self):
        (self.root / "quality-data").mkdir()
        views = [{"view": role, "total": rows, "supported": rows,
                  "correct": rows, "passed": True}
                 for role, rows in (("training", 512), ("validation", 256), ("validation", 256))]
        information = {"passed": True, "masters": [{"master": master, "history": 128,
                                                     "passed": True, "views": copy.deepcopy(views)}
                                                    for master in capsule.MASTERS]}
        self.fixture.write(self.root / "quality-data" / "information-report.json", information)
        with self.assertRaisesRegex(ValueError, "information views"):
            capsule.bind(self.root)
        self.assertFalse((self.root / "freeze" / "information.sha256").exists())

    def test_incomplete_run_cannot_publish_complete_even_with_optimized_python(self):
        p = self.root / "profiles" / "A" / "master-941101" / "complete.json"
        value = json.loads(p.read_text())
        value["updates"] = 511
        self.fixture.write(p, value)
        self.reject_finish("producer counter/report/bindings")

    def test_changed_producer_report_rejects_completion(self):
        (self.root / "profiles" / "A" / "master-941101" / "report.json").write_text("changed\n")
        self.reject_finish("producer counter/report/bindings")

    def test_wrong_cohort_receipt_rejects_completion(self):
        p = self.root / "profiles" / "B" / "master-941102" / "complete.json"
        value = json.loads(p.read_text())
        value["master"] = 941101
        self.fixture.write(p, value)
        self.reject_finish("producer counter/report/bindings")

    def test_changed_reader_report_rejects_completion(self):
        (self.root / "readouts" / "report.json").write_text("changed\n")
        self.reject_finish("saved-reader counter/report")

    def test_wrong_publisher_input_binding_rejects_completion(self):
        p = self.root / "summary.json"
        value = json.loads(p.read_text())
        value["input_manifest_sha"] = "0" * 64
        self.fixture.write(p, value)
        self.reject_finish("publisher/report bindings")

    def test_existing_completion_is_preserved_without_hashing_again(self):
        completed = self.root / "complete.json"
        completed.write_text("preserved receipt\n")
        with patch.object(capsule, "digest", side_effect=RuntimeError("must not hash")):
            with self.assertRaisesRegex(ValueError, "finalized again"):
                capsule.finish(self.root)
        self.assertEqual(completed.read_text(), "preserved receipt\n")

    def test_valid_receipts_finalize_once_with_bound_inventory(self):
        with redirect_stdout(io.StringIO()):
            capsule.finish(self.root)
        completed = self.root / "complete.json"
        original = completed.read_bytes()
        value = json.loads(original)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["encoder_trajectories"], 9)
        self.assertEqual(value["completed_updates_per_run"], 512)
        self.assertEqual(value["saved_head_fits"], 81)
        self.assertEqual(value["artifacts_manifest_sha256"], capsule.digest(self.root / "artifacts.sha256"))
        with self.assertRaisesRegex(ValueError, "finalized again"):
            capsule.finish(self.root)
        self.assertEqual(completed.read_bytes(), original)


if __name__ == "__main__":
    capsule.require(Path("/.dockerenv").exists(), "Run tests inside the managed container.")
    unittest.main()
