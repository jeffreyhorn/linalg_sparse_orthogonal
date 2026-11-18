#!/usr/bin/env python3
"""Decision and evidence tests for the one-cell Sprint 225 timing validator."""

from __future__ import annotations

import csv
import copy
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
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_refactor_csc_timing as gate  # noqa: E402
import collect_refactor_csc_timing as collector  # noqa: E402


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
           source: str = "a" * 40, file_hashes: dict[str, str] | None = None) -> dict:
    now = datetime.now(timezone.utc) - timedelta(hours=3 if role == "baseline" else 1)
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
        "effective_build": {
            "cc": "cc", "sysroot": "/SyntheticSDK",
            "cflags": "-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 -isysroot /SyntheticSDK",
            "ldflags": "-lm",
            "sparse_cflags": "-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 -isysroot /SyntheticSDK -DSPARSE_IDX_BITS=32",
        },
        "build_mode": "serial", "index_width_bits": 32,
        "thread_environment": {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"},
        "backend_environment": {
            "SPARSE_LDLT_DENSE_BACKEND": "unset", "SPARSE_CHOL_DENSE_BACKEND": "unset",
            "SPARSE_SUPERNODAL_POSTORDER": "unset", "SPARSE_ND_SUPERNODAL_POSTORDER": "unset",
        },
        "executable_sha256": "1" * 64, "library_sha256": "2" * 64,
        "benchmark_source_sha256": "3" * 64, "makefile_sha256": "4" * 64,
        "fixture_sha256": gate.FIXTURE_SHA256, "argv": gate.ARGV,
        "timeout_seconds": 30, "warmup_count": 5, "measured_count": 20,
        "metric": "refactor_csc_ms", "metric_units": "ms",
        "started_at_utc": stamp, "ended_at_utc": stamp,
    }
    if file_hashes is not None:
        run.update(file_hashes)
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


def accepted_baseline(root: Path, file_hashes: dict[str, str] | None = None) -> tuple[Path, dict]:
    baseline_id = "synthetic-reviewed"
    directory = root / gate.POLICY / baseline_id
    directory.mkdir(parents=True)
    jobs = []
    first = None
    for attempt in range(1, 4):
        filename = f"job-{attempt}.zip"
        run = bundle(directory / filename, "baseline", attempt, ["0.100"] * 20,
                     file_hashes=file_hashes)
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
                "accepted_at_utc": (now - timedelta(hours=2)).isoformat(),
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


def mutate_candidate(root: Path, change) -> Path:
    zip_path = root / "candidate.zip"
    bundle(zip_path, "candidate", 1, ["0.100"] * 20, source="b" * 40)
    directory = root / "candidate-dir"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(directory)
    run = json.loads((directory / "run.json").read_text())
    change(run)
    (directory / "run.json").write_bytes(encoded(run))
    rehash_directory(directory)
    return directory


def synthetic_checkout(root: Path) -> dict[str, str]:
    files = {
        "fixture_sha256": ("tests/data/suitesparse/nos4.mtx",
                           (ROOT / "tests/data/suitesparse/nos4.mtx").read_bytes()),
        "benchmark_source_sha256": ("benchmarks/bench_refactor_csc.c", b"synthetic benchmark source"),
        "makefile_sha256": ("Makefile", b"synthetic pinned build recipe"),
        "executable_sha256": ("build/bench_refactor_csc", b"synthetic candidate executable"),
        "library_sha256": ("build/libsparse_lu_ortho.a", b"synthetic candidate archive"),
    }
    result = {}
    for field, (name, data) in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result[field] = gate.digest(data)
    (root / "build/.sparse-idx-bits").write_text("32\n")
    return result


def candidate_directory(root: Path) -> Path:
    zip_path = root / "candidate.zip"
    bundle(zip_path, "candidate", 1, ["0.100"] * 20, source="b" * 40)
    directory = root / "candidate-dir"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(directory)
    return directory


def write_trial_rows(directory: Path, rows: list[dict[str, str]]) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=gate.TRIAL_FIELDS,
                            delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (directory / "trials.tsv").write_text(output.getvalue())


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
            self.assertEqual((report, code), gate.decision(
                root, candidate, "b" * 40,
                evaluation_time=datetime.fromisoformat(report["evaluated_at_utc"])))
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

    def test_candidate_workload_and_host_mutations_have_identity_diagnostics(self):
        changes = (
            ("fixture-digest", lambda run: run.__setitem__("fixture_sha256", "f" * 64), "fixture digest mismatch"),
            ("benchmark-source", lambda run: run.__setitem__("benchmark_source_sha256", "f" * 64), "fingerprint"),
            ("build-recipe", lambda run: run.__setitem__("makefile_sha256", "f" * 64), "fingerprint"),
            ("command", lambda run: run.__setitem__("argv", ["other-benchmark", *gate.ARGV[1:]]), "command"),
            ("repeat-count", lambda run: run["argv"].__setitem__(3, "2"), "command"),
            ("backend", lambda run: run["backend_environment"].__setitem__("SPARSE_LDLT_DENSE_BACKEND", "dense"), "backend"),
            ("build-mode", lambda run: run.__setitem__("build_mode", "parallel"), "build mode"),
            ("runner-cpu", lambda run: run["runner"].__setitem__("cpu_model", "other CPU"), "fingerprint"),
            ("compiler", lambda run: run["runner"].__setitem__("compiler_version", "other compiler"), "fingerprint"),
            ("unknown-host", lambda run: run["runner"].__setitem__("runner_image", "unknown"), "sentinel"),
        )
        for name, change, detail in changes:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                accepted_baseline(root)
                candidate = mutate_candidate(root, change)
                report, code = gate.decision(root, candidate, "b" * 40)
                self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_IDENTITY", 2))
                self.assertIn(detail, report["detail"])

    def test_candidate_checkout_bytes_and_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkout = root / "checkout"
            hashes = synthetic_checkout(checkout)
            baseline = root / "baselines"
            accepted_baseline(baseline, hashes)
            candidate = root / "candidate.zip"
            bundle(candidate, "candidate", 1, ["0.100"] * 20,
                   source="b" * 40, file_hashes=hashes)
            report, code = gate.decision(baseline, candidate, "b" * 40, checkout)
            self.assertEqual((report["primary_rule"], code), ("S225_TIMING_PASS", 0))
            for name in ("tests/data/suitesparse/nos4.mtx", "build/bench_refactor_csc",
                         "build/libsparse_lu_ortho.a",
                         "benchmarks/bench_refactor_csc.c", "Makefile"):
                with self.subTest(file=name):
                    path = checkout / name
                    original = path.read_bytes()
                    path.write_bytes(original + b"changed")
                    report, code = gate.decision(baseline, candidate, "b" * 40, checkout)
                    self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_IDENTITY", 2))
                    self.assertIn(name, report["detail"])
                    path.write_bytes(original)
            report, code = gate.decision(baseline, candidate, "c" * 40, checkout)
            self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_IDENTITY", 2))
            self.assertIn("commit", report["detail"])
            (checkout / "build/.sparse-idx-bits").write_text("64\n")
            report, code = gate.decision(baseline, candidate, "b" * 40, checkout)
            self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_IDENTITY", 2))
            self.assertIn("width stamp", report["detail"])

    def test_missing_expired_superseded_and_mismatched_baselines(self):
        cases = ("missing-pointer", "missing-manifest", "missing-bundle",
                 "expired", "superseded", "malformed-schema", "version-mismatch",
                 "pointer-digest-mismatch", "source-mismatch")
        for name in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                directory, manifest = accepted_baseline(root)
                pointer_path = root / "active.json"
                pointer = json.loads(pointer_path.read_text())
                if name == "missing-pointer":
                    pointer_path.unlink()
                elif name == "missing-manifest":
                    (directory / "baseline.json").unlink()
                elif name == "missing-bundle":
                    (directory / "job-1.zip").unlink()
                else:
                    if name == "expired":
                        manifest["expires_at_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
                    elif name == "superseded":
                        manifest["status"] = "superseded"
                    elif name == "malformed-schema":
                        manifest["schema_version"] = 99
                    elif name == "version-mismatch":
                        manifest["policy_version"] = "s225-v2"
                    elif name == "source-mismatch":
                        manifest["source_commit"] = "c" * 40
                    payload = encoded(manifest)
                    (directory / "baseline.json").write_bytes(payload)
                    pointer["baseline_manifest_sha256"] = (
                        "0" * 64 if name == "pointer-digest-mismatch" else gate.digest(payload)
                    )
                    pointer_path.write_bytes(encoded(pointer))
                report, code = gate.decision(root, None)
                expected = "S225_BASELINE_MISSING" if name.startswith("missing-") else "S225_BASELINE_INVALID"
                self.assertEqual((report["primary_rule"], code), (expected, 2))

    def test_weak_sample_mutations_keep_available_raw_streams(self):
        cases = (
            "single-sample", "incomplete-measured", "missing-warmup",
            "timeout", "malformed-time", "altered-hash",
        )
        for name in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                accepted_baseline(root)
                candidate = candidate_directory(root)
                rows = list(csv.DictReader(io.StringIO((candidate / "trials.tsv").read_text()), delimiter="\t"))
                if name == "single-sample":
                    write_trial_rows(candidate, rows[:6])
                    receipt = json.loads((candidate / "decision.json").read_text())
                    receipt["trial_count"] = 6
                    (candidate / "decision.json").write_bytes(encoded(receipt))
                    rehash_directory(candidate)
                elif name == "incomplete-measured":
                    write_trial_rows(candidate, rows[:-1])
                    rehash_directory(candidate)
                elif name == "missing-warmup":
                    (candidate / "trials/0001.stdout").unlink()
                elif name == "timeout":
                    rows[0]["timed_out"] = "True"
                    rows[0]["validity"] = "timeout"
                    write_trial_rows(candidate, rows)
                    receipt = json.loads((candidate / "decision.json").read_text())
                    receipt["all_trials_valid"] = False
                    (candidate / "decision.json").write_bytes(encoded(receipt))
                    rehash_directory(candidate)
                elif name == "malformed-time":
                    rows[0]["started_at_utc"] = "not-a-time"
                    write_trial_rows(candidate, rows)
                    rehash_directory(candidate)
                else:
                    (candidate / "trials/0006.stdout").write_bytes(b"altered raw output")
                raw_before = {path.relative_to(candidate).as_posix(): gate.digest(path.read_bytes())
                              for path in candidate.rglob("*") if path.is_file()}
                report, code = gate.decision(root, candidate, "b" * 40)
                expected = "S225_CANDIDATE_MISSING" if name == "missing-warmup" else "S225_CANDIDATE_INVALID"
                self.assertEqual((report["primary_rule"], code), (expected, 2))
                raw_after = {path.relative_to(candidate).as_posix(): gate.digest(path.read_bytes())
                             for path in candidate.rglob("*") if path.is_file()}
                self.assertEqual(raw_before, raw_after)
                if name != "missing-warmup":
                    self.assertEqual(len([p for p in raw_after if p.endswith(".stdout")]), 25)

    def test_exact_threshold_boundary_and_noisy_candidate_keep_raw_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            for name, samples, rule, code in (
                ("boundary", ["0.125"] * 20, "S225_TIMING_PASS", 0),
                ("over", ["0.126"] * 20, "S225_TIMING_REGRESSION", 1),
                ("noisy", ["0.100"] * 10 + ["0.200"] * 10, "S225_CANDIDATE_DISPERSION", 1),
            ):
                with self.subTest(name=name):
                    candidate = root / f"{name}.zip"
                    bundle(candidate, "candidate", 1, samples, source="b" * 40)
                    before = gate.digest(candidate.read_bytes())
                    report, actual_code = gate.decision(root, candidate, "b" * 40)
                    self.assertEqual((report["primary_rule"], actual_code), (rule, code))
                    self.assertEqual(gate.digest(candidate.read_bytes()), before)
                    self.assertEqual(report["measured_count"], 20)

    def test_report_mutations_cannot_infer_portable_or_freshness_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            candidate = root / "candidate.zip"
            bundle(candidate, "candidate", 1, ["0.100"] * 20, source="b" * 40)
            report, code = gate.decision(root, candidate, "b" * 40)
            self.assertEqual(code, 0)
            self.assertEqual(gate.verify_report_against_evidence(report, root, candidate, "b" * 40), 0)
            self.assertEqual(report["claim_scope"], gate.CLAIM_SCOPE)
            self.assertEqual(report["non_claims"], gate.NON_CLAIMS)
            for name, changes in (
                ("freshness", {"freshness_result": "pass"}),
                ("portable-speed", {"claim_scope": "portable_speed_superiority"}),
                ("parity", {"non_claims": []}),
                ("missing-raw-hash", {"candidate_hashes_sha256": None}),
                ("missing-hosted-run", {"candidate_run_id": None}),
            ):
                with self.subTest(name=name):
                    altered = copy.deepcopy(report)
                    altered.update(changes)
                    with self.assertRaises(gate.EvidenceError) as captured:
                        gate.verify_report_against_evidence(altered, root, candidate, "b" * 40)
                    self.assertEqual(captured.exception.rule, "S225_REPORT_INVALID")
            with self.assertRaises(gate.EvidenceError) as captured:
                gate.verify_report_against_evidence(report, root, root / "missing-candidate.zip", "b" * 40)
            self.assertEqual(captured.exception.rule, "S225_REPORT_INVALID")
            report_path = root / "timing-decision.json"
            report_path.write_bytes(encoded(report))
            command = [sys.executable, str(ROOT / "scripts/verify_refactor_csc_timing_report.py"),
                       "--baseline-root", str(root), "--candidate", str(candidate),
                       "--report", str(report_path), "--expected-candidate-commit", "b" * 40]
            self.assertEqual(subprocess.run(command, capture_output=True, check=False).returncode, 0)
            command[command.index(str(candidate))] = str(root / "missing-candidate.zip")
            self.assertEqual(subprocess.run(command, capture_output=True, check=False).returncode, 2)


class ReviewRegressionTests(unittest.TestCase):
    def test_build_step_receipt_is_parsed_without_dropping_sdk_or_flags(self):
        values = {
            "cc": "cc", "sysroot": "/SyntheticSDK",
            "cflags": "-std=c11 -O2 -isysroot /SyntheticSDK",
            "ldflags": "-lm", "sparse_cflags": "-std=c11 -O2 -isysroot /SyntheticSDK -DSPARSE_IDX_BITS=32",
        }
        receipt = "".join(f"{key}\t{value}\n" for key, value in values.items())
        self.assertEqual(collector.parse_build_config(receipt), values)
        with self.assertRaises(ValueError):
            collector.parse_build_config(receipt + "cc\tother\n")

    def test_inherited_build_and_symbolic_overrides_are_rejected(self):
        for name in ("SPARSE_MUTEX", "CFLAGS", "SDKROOT", "MAKEFLAGS",
                     "SPARSE_SUPERNODAL_POSTORDER", "SPARSE_ND_SUPERNODAL_POSTORDER"):
            with self.subTest(name=name), mock.patch.dict(collector.os.environ, {name: ""}, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    collector.check_policy_environment(False)

    def test_effective_build_settings_are_captured_and_locked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted_baseline(root)
            for name, change in (
                ("mutex-flags", lambda build: build.__setitem__("cflags", build["cflags"] + " -DSPARSE_MUTEX")),
                ("sdk", lambda build: build.update({
                    "sysroot": "/OtherSDK",
                    "cflags": build["cflags"].replace("/SyntheticSDK", "/OtherSDK"),
                    "sparse_cflags": build["sparse_cflags"].replace("/SyntheticSDK", "/OtherSDK"),
                })),
            ):
                with self.subTest(name=name):
                    candidate = mutate_candidate(root, lambda run: change(run["effective_build"]))
                    report, code = gate.decision(root, candidate)
                    self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_IDENTITY", 2))
                    (root / "candidate-dir").rename(root / f"checked-{name}")

    def test_candidate_collection_must_fit_baseline_window_and_clock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, manifest = accepted_baseline(root)
            candidate = candidate_directory(root)
            original = json.loads((candidate / "run.json").read_text())
            for name, instant in (
                ("after-expiry", datetime.fromisoformat(manifest["expires_at_utc"]) + timedelta(seconds=1)),
                ("before-acceptance", datetime.fromisoformat(manifest["accepted_at_utc"]) - timedelta(seconds=1)),
                ("future", datetime.now(timezone.utc) + timedelta(hours=1)),
            ):
                with self.subTest(name=name):
                    run = copy.deepcopy(original)
                    run["started_at_utc"] = run["ended_at_utc"] = instant.isoformat()
                    (candidate / "run.json").write_bytes(encoded(run))
                    rows = list(csv.DictReader(io.StringIO((candidate / "trials.tsv").read_text()), delimiter="\t"))
                    for row in rows:
                        row["started_at_utc"] = row["ended_at_utc"] = instant.isoformat()
                    write_trial_rows(candidate, rows)
                    rehash_directory(candidate)
                    report, code = gate.decision(root, candidate)
                    self.assertEqual((report["primary_rule"], code), ("S225_CANDIDATE_IDENTITY", 2))

    def test_historical_report_recomputes_after_expiry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory, manifest = accepted_baseline(root)
            manifest["expires_at_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
            payload = encoded(manifest)
            (directory / "baseline.json").write_bytes(payload)
            pointer = json.loads((root / "active.json").read_text())
            pointer["baseline_manifest_sha256"] = gate.digest(payload)
            (root / "active.json").write_bytes(encoded(pointer))
            candidate = root / "candidate.zip"
            bundle(candidate, "candidate", 1, ["0.100"] * 20)
            report, code = gate.decision(root, candidate)
            self.assertEqual((report["primary_rule"], code), ("S225_TIMING_PASS", 0))
            real_datetime = datetime

            class LaterDatetime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return real_datetime.now(tz) + timedelta(minutes=2)

            with mock.patch.object(gate, "datetime", LaterDatetime):
                self.assertEqual(gate.verify_report_against_evidence(report, root, candidate), 0)
                preflight, preflight_code = gate.decision(root, None)
                self.assertEqual((preflight["primary_rule"], preflight_code),
                                 ("S225_BASELINE_INVALID", 2))


if __name__ == "__main__":
    unittest.main()
