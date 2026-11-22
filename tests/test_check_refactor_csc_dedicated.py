#!/usr/bin/env python3
"""Dedicated v4 evidence accepts only its separate raw schedule and identity."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_refactor_csc_dedicated as dedicated  # noqa: E402
import collect_refactor_csc_dedicated as collector  # noqa: E402
from test_check_refactor_csc_pair import make_bundle, rehash  # noqa: E402


class DedicatedEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        self.collection_id = "1234"
        self.command = ["build/bench_refactor_csc", *collector.COMMAND]
        for replica in (1, 2, 3):
            old = make_bundle(self.root, replica, self.commit)
            target = self.root / f"sprint225-v4-pair-1234-1-{replica}"
            shutil.move(str(old), str(target))
            run_path = target / "run.json"
            run = json.loads(run_path.read_text())
            for name in ("workflow_run_id", "workflow_run_attempt", "workflow_job_key"):
                del run[name]
            run.update({
                "policy_version": "s225-v4", "collection_id": "1234",
                "collection_attempt": "1", "execution_mode": "dedicated_offline",
                "operator_attestation": {
                    "operator": "independent operator", "exclusive_window_acknowledged": True,
                },
                "argv": self.command,
            })
            run["runner"]["runner_image"] = "dedicated-macos-intel"
            run["runner"]["runner_label"] = "isolated-intel-mac"
            run_path.write_text(json.dumps(run))
            receipt_path = target / "decision.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["policy_version"] = "s225-v4"
            receipt_path.write_text(json.dumps(receipt))
            table = target / "trials.tsv"
            with table.open(newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                rows = list(reader)
                fields = reader.fieldnames
            for row in rows:
                row["argv"] = json.dumps(self.command)
            with table.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t",
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            rehash(target)

    def decision(self) -> tuple[dict, int]:
        return dedicated.decide(self.root, self.commit, ROOT, self.collection_id)

    def test_three_matching_raw_replicas_are_provisional(self) -> None:
        report, code = self.decision()
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "provisional_pass")
        self.assertEqual(report["independent_review"], "pending")

    def test_wrong_workload_is_rejected(self) -> None:
        path = self.root / "sprint225-v4-pair-1234-1-2"
        run_path = path / "run.json"
        run = json.loads(run_path.read_text())
        run["argv"][-1] = "1000"
        run_path.write_text(json.dumps(run))
        rehash(path)
        report, code = self.decision()
        self.assertEqual(code, 2)
        self.assertIn("workload", report["detail"])

    def test_different_host_is_rejected(self) -> None:
        path = self.root / "sprint225-v4-pair-1234-1-3"
        run_path = path / "run.json"
        run = json.loads(run_path.read_text())
        run["runner"]["runner_label"] = "another-mac"
        run_path.write_text(json.dumps(run))
        rehash(path)
        report, code = self.decision()
        self.assertEqual(code, 2)
        self.assertIn("host", report["detail"])

    def test_overlapping_replica_windows_are_rejected(self) -> None:
        path = self.root / "sprint225-v4-pair-1234-1-1"
        run_path = path / "run.json"
        run = json.loads(run_path.read_text())
        run["ended_at_utc"] = (datetime.fromisoformat(run["ended_at_utc"]) +
                               timedelta(seconds=1)).isoformat()
        run_path.write_text(json.dumps(run))
        rehash(path)
        report, code = self.decision()
        self.assertEqual(code, 2)
        self.assertIn("overlap", report["detail"])

    def test_missing_attestation_is_rejected(self) -> None:
        path = self.root / "sprint225-v4-pair-1234-1-1"
        run_path = path / "run.json"
        run = json.loads(run_path.read_text())
        del run["operator_attestation"]
        run_path.write_text(json.dumps(run))
        rehash(path)
        self.assertEqual(self.decision()[1], 2)

    def test_missing_replica_is_rejected(self) -> None:
        shutil.rmtree(self.root / "sprint225-v4-pair-1234-1-3")
        report, code = self.decision()
        self.assertEqual(code, 2)
        self.assertIn("missing raw pair", report["detail"])

    def test_original_zip_bundle_is_accepted(self) -> None:
        path = self.root / "sprint225-v4-pair-1234-1-2"
        with zipfile.ZipFile(path.with_suffix(".zip"), "w") as archive:
            for file in path.rglob("*"):
                if file.is_file():
                    archive.write(file, file.relative_to(path).as_posix())
        shutil.rmtree(path)
        self.assertEqual(self.decision()[1], 0)

    def test_tampered_raw_stream_is_rejected(self) -> None:
        path = self.root / "sprint225-v4-pair-1234-1-2"
        with (path / "trials/0006-candidate.stdout").open("ab") as stream:
            stream.write(b"tampered\n")
        report, code = self.decision()
        self.assertEqual(code, 2)
        self.assertIn("hash mismatch", report["detail"])

    def test_host_preflight_requires_exclusive_attestation(self) -> None:
        with self.assertRaisesRegex(ValueError, "attestation"):
            collector.require_dedicated_host("mac", "operator", False)

    def test_build_subprocess_obeys_deadline_and_preserves_log(self) -> None:
        log = self.root / "build.log"
        start = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "deadline expired"):
            collector.run_process(
                [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
                self.root, start + 0.1, log,
            )
        self.assertLess(time.monotonic() - start, 3)
        self.assertIn(b"started", log.read_bytes())

    def test_failed_collection_writes_blocked_decision_and_archive(self) -> None:
        output = self.root / "failed-collection"
        argv = [
            "collect_refactor_csc_dedicated.py", "--output-root", str(output),
            "--host-id", "isolated-intel-mac", "--operator", "operator",
            "--exclusive-window-attested",
        ]
        with (mock.patch.object(sys, "argv", argv),
              mock.patch.object(collector, "require_dedicated_host", return_value={}),
              mock.patch.object(collector, "git", return_value=self.commit),
              mock.patch.object(collector, "_collect_one",
                                side_effect=TimeoutError("build deadline expired"))):
            self.assertEqual(collector.main(), 2)
        decision = json.loads((output / "sprint225-v4-decision.json").read_text())
        self.assertEqual(decision["status"], "blocked")
        self.assertIn("deadline", decision["detail"])
        raw = output / f"sprint225-v4-pair-{decision['collection_id']}-1-1"
        self.assertTrue((raw / "failure.json").is_file())
        self.assertTrue((raw / "hashes.sha256").is_file())
        self.assertTrue(raw.with_suffix(".zip").is_file())
        self.assertTrue((output / "archive-sha256.txt").is_file())


if __name__ == "__main__":
    unittest.main()
