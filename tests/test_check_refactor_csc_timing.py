#!/usr/bin/env python3
"""Decision and evidence tests for the one-cell Sprint 225 timing validator."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_refactor_csc_timing as gate  # noqa: E402


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def trial_stdout(value: str) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(gate.CSV_FIELDS)
    row = {name: "0" for name in gate.CSV_FIELDS}
    row.update({"benchmark": "bench_refactor_csc", "category": "proof", "matrix": "nos4.mtx",
                "scenario": "chol_spd", "n": "100", "nnz": "594", "refactor_csc_ms": value})
    writer.writerow([row[name] for name in gate.CSV_FIELDS])
    return buffer.getvalue().encode()


def bundle(path: Path, role: str, attempt: int, values: list[str], *, cpu: str = "Apple M1 (Virtual)",
           source: str = "a" * 40) -> dict:
    now = datetime.now(timezone.utc) - timedelta(hours=2)
    stamp = now.isoformat()
    run = {
        "schema_version": 2, "policy_version": gate.POLICY, "role": role,
        "dry_run": False, "source_commit": source, "source_dirty": False,
        "workflow_run_id": "1234", "workflow_run_attempt": str(attempt),
        "workflow_job_key": "collect-baseline" if role == "baseline" else "candidate-timing",
        "runner": {"runner_image": "macos26", "runner_image_version": "20261102.0473.1",
                   "os": "macOS-26.7.2-arm64", "architecture": "arm64", "cpu_model": cpu,
                   "compiler_path": "/usr/bin/cc", "compiler_version": "Apple clang 21",
                   "runner_label": f"ephemeral-{attempt}"},
        "build_recipe": "make build/bench_refactor_csc SPARSE_IDX_BITS=32",
        "build_mode": "serial", "index_width_bits": 32,
        "thread_environment": {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"},
        "backend_environment": {"SPARSE_LDLT_DENSE_BACKEND": "unset"},
        "executable_sha256": "1" * 64, "library_sha256": "2" * 64,
        "benchmark_source_sha256": "3" * 64, "makefile_sha256": "4" * 64,
        "fixture_sha256": gate.FIXTURE_SHA256, "argv": gate.ARGV,
        "timeout_seconds": 30, "warmup_count": 5, "measured_count": 20,
        "metric": "refactor_csc_ms", "metric_units": "ms",
        "started_at_utc": stamp, "ended_at_utc": stamp,
    }
    receipt = {"policy_version": gate.POLICY, "role": role, "dry_run": False,
               "status": "pending_baseline_review" if role == "baseline" else "pending_timing_review",
               "trial_count": 25, "all_trials_valid": True, "baseline_accepted": False,
               "freshness_result": "separate"}
    members = {"run.json": encoded(run), "decision.json": encoded(receipt)}
    records = []
    for number, value in enumerate(["0.100"] * 5 + values, 1):
        stdout = trial_stdout(value)
        stderr = b""
        members[f"trials/{number:04d}.stdout"] = stdout
        members[f"trials/{number:04d}.stderr"] = stderr
        records.append({"ordinal": number, "phase": "warmup" if number <= 5 else "measured",
                        "argv": json.dumps(gate.ARGV), "started_at_utc": stamp,
                        "ended_at_utc": stamp, "outer_elapsed_ns": 1000,
                        "exit_code": 0, "timed_out": False, "refactor_csc_ms": value,
                        "validity": "valid", "stdout_sha256": gate.digest(stdout),
                        "stderr_sha256": gate.digest(stderr)})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=gate.TRIAL_FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    members["trials.tsv"] = output.getvalue().encode()
    members["hashes.sha256"] = "".join(
        f"{gate.digest(data)}  {name}\n" for name, data in sorted(members.items())
    ).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    return run


def accepted_baseline(root: Path) -> tuple[Path, dict]:
    baseline_id = "synthetic-reviewed"
    directory = root / gate.POLICY / baseline_id
    directory.mkdir(parents=True)
    jobs = []
    first = None
    for attempt in range(1, 4):
        filename = f"job-{attempt}.zip"
        run = bundle(directory / filename, "baseline", attempt, ["0.100"] * 20)
        if first is None:
            first = run
        jobs.append({"bundle_file": filename,
                     "bundle_sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                     "run_id": "1234", "attempt": str(attempt), "job_key": "collect-baseline",
                     "median_ms": "0.100", "mad_ratio": "0", "p90_ratio": "1"})
    assert first is not None
    now = datetime.now(timezone.utc)
    manifest = {"schema_version": 1, "policy_version": gate.POLICY,
                "status": "accepted", "target_id": gate.TARGET,
                "baseline_id": baseline_id, "approved_by": "synthetic-test-reviewer",
                "source_commit": "a" * 40,
                "accepted_at_utc": (now - timedelta(hours=1)).isoformat(),
                "expires_at_utc": (now + timedelta(days=29)).isoformat(),
                "fingerprint": gate.fingerprint(first), "jobs": jobs,
                "between_job_spread": "0", "reference_median_ms": "0.100"}
    payload = encoded(manifest)
    (directory / "baseline.json").write_bytes(payload)
    root.mkdir(exist_ok=True)
    (root / "active.json").write_bytes(encoded({
        "status": "accepted", "policy_version": gate.POLICY,
        "baseline_id": baseline_id, "baseline_manifest_sha256": gate.digest(payload),
    }))
    return directory, manifest


def rehash_directory(directory: Path) -> None:
    members = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "hashes.sha256")
    (directory / "hashes.sha256").write_text("".join(
        f"{gate.digest(path.read_bytes())}  {path.relative_to(directory).as_posix()}\n"
        for path in members
    ), encoding="ascii")


class TimingDecisionTests(unittest.TestCase):
    def test_real_branch_without_accepted_baseline_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report, code = gate.decision(root / "missing", root / "candidate.zip")
            self.assertEqual((report["primary_rule"], report["status"], code),
                             ("S225_BASELINE_MISSING", "blocked", 2))
            self.assertIsNone(report["reference_median_ms"])
            preflight, preflight_code = gate.decision(root / "missing", None)
            self.assertEqual((preflight["primary_rule"], preflight_code),
                             ("S225_BASELINE_MISSING", 2))

    def test_preflight_ready_is_not_a_timing_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            report, code = gate.decision(root, None)
            self.assertEqual((report["status"], report["primary_rule"], code),
                             ("ready", "S225_BASELINE_READY", 0))
            self.assertIsNone(report["candidate_median_ms"])

    def test_preflight_cli_publishes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "build/timing-decision.json"
            process = subprocess.run([
                sys.executable, str(ROOT / "scripts/check_refactor_csc_timing.py"),
                "--baseline-root", str(root / "absent"), "--preflight", "--output", str(output),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(output.read_text())["primary_rule"], "S225_BASELINE_MISSING")

    def test_synthetic_pass_and_regression_have_exact_decimal_ceiling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            candidate = root / "candidate.zip"
            bundle(candidate, "candidate", 1, ["0.110"] * 20, source="b" * 40)
            report, code = gate.decision(root, candidate, "b" * 40)
            self.assertEqual((report["primary_rule"], report["status"], code),
                             ("S225_TIMING_PASS", "pass", 0))
            self.assertEqual((report, code), gate.decision(root, candidate, "b" * 40))
            self.assertEqual(report["candidate_median_ms"], "0.110")
            self.assertEqual(Decimal(report["allowed_ceiling_ms"]), Decimal("0.12500"))
            self.assertEqual(report["warmup_count"], 5)
            self.assertEqual(report["measured_count"], 20)
            self.assertEqual(len(report["baseline_job_medians_ms"]), 3)
            self.assertEqual(len(report["candidate_hashes_sha256"]), 64)
            bundle(root / "regression.zip", "candidate", 2, ["0.130"] * 20, source="b" * 40)
            report, code = gate.decision(root, root / "regression.zip", "b" * 40)
            self.assertEqual((report["primary_rule"], report["status"], code),
                             ("S225_TIMING_REGRESSION", "regression", 1))

    def test_dispersion_precedes_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            candidate = root / "noisy.zip"
            bundle(candidate, "candidate", 1, ["0.100"] * 10 + ["0.200"] * 10)
            report, code = gate.decision(root, candidate)
            self.assertEqual((report["primary_rule"], report["status"], code),
                             ("S225_CANDIDATE_DISPERSION", "unstable", 1))

    def test_runner_mismatch_and_corrupt_raw_stream_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            other_cpu = root / "wrong-cpu.zip"
            bundle(other_cpu, "candidate", 1, ["0.100"] * 20, cpu="Apple M2 Pro (Virtual)")
            report, code = gate.decision(root, other_cpu)
            self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_IDENTITY", 2))
            candidate_dir = root / "candidate-dir"
            candidate_dir.mkdir()
            zip_path = root / "candidate.zip"
            bundle(zip_path, "candidate", 1, ["0.100"] * 20)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(candidate_dir)
            (candidate_dir / "trials/0006.stdout").write_bytes(b"changed")
            report, code = gate.decision(root, candidate_dir)
            self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_INVALID", 2))

    def test_warmup_timeout_fails_even_with_rehashed_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            zip_path = root / "candidate.zip"
            bundle(zip_path, "candidate", 1, ["0.100"] * 20)
            directory = root / "candidate-dir"
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(directory)
            tsv = directory / "trials.tsv"
            rows = list(csv.DictReader(io.StringIO(tsv.read_text()), delimiter="\t"))
            rows[0]["outer_elapsed_ns"] = "30000000001"
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=gate.TRIAL_FIELDS,
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            tsv.write_text(output.getvalue())
            rehash_directory(directory)
            report, code = gate.decision(root, directory)
            self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_INVALID", 2))

    def test_noncanonical_or_unbounded_metric_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            for number, metric in enumerate(("1e999999", "30000.001"), 1):
                candidate = root / f"candidate-{number}.zip"
                bundle(candidate, "candidate", number, [metric] * 20)
                report, code = gate.decision(root, candidate)
                self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_INVALID", 2))

    def test_failed_trial_is_reported_as_invalid_timing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            zip_path = root / "candidate.zip"
            bundle(zip_path, "candidate", 1, ["0.100"] * 20)
            directory = root / "candidate-dir"
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(directory)
            receipt = json.loads((directory / "decision.json").read_text())
            receipt["all_trials_valid"] = False
            (directory / "decision.json").write_bytes(encoded(receipt))
            tsv = directory / "trials.tsv"
            rows = list(csv.DictReader(io.StringIO(tsv.read_text()), delimiter="\t"))
            rows[0]["exit_code"] = "1"
            rows[0]["validity"] = "process_failed"
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=gate.TRIAL_FIELDS,
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            tsv.write_text(output.getvalue())
            rehash_directory(directory)
            report, code = gate.decision(root, directory)
            self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_INVALID", 2))
            self.assertIn("trial 1", report["detail"])

    def test_baseline_provenance_is_checked_before_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, manifest = accepted_baseline(root)
            manifest["approved_by"] = "   "
            payload = encoded(manifest)
            (directory / "baseline.json").write_bytes(payload)
            pointer = json.loads((root / "active.json").read_text())
            pointer["baseline_manifest_sha256"] = gate.digest(payload)
            (root / "active.json").write_bytes(encoded(pointer))
            report, code = gate.decision(root, root / "missing-candidate.zip")
            self.assertEqual((report["primary_rule"], code), ("S225_BASELINE_INVALID", 2))

    def test_rejected_group_cannot_become_active_by_pointer_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, manifest = accepted_baseline(root)
            rejected = ROOT / "docs/planning/EPIC_20/SPRINT_225/artifacts/day5-rejected-baseline/attempt-1.zip"
            (directory / "job-1.zip").write_bytes(rejected.read_bytes())
            manifest["jobs"][0]["bundle_sha256"] = gate.digest(rejected.read_bytes())
            payload = encoded(manifest)
            (directory / "baseline.json").write_bytes(payload)
            pointer = json.loads((root / "active.json").read_text())
            pointer["baseline_manifest_sha256"] = gate.digest(payload)
            (root / "active.json").write_bytes(encoded(pointer))
            report, code = gate.decision(root, root / "missing-candidate.zip")
            self.assertEqual((report["primary_rule"], code), ("S225_BASELINE_INVALID", 2))


if __name__ == "__main__":
    unittest.main()
