#!/usr/bin/env python3
"""Independent paired timing decision and adversarial evidence checks."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_refactor_csc_pair as pair  # noqa: E402
import check_refactor_csc_timing as common  # noqa: E402
import collect_refactor_csc_pair as collector  # noqa: E402


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def metric_csv(value: str) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(common.CSV_FIELDS)
    row = {name: "0" for name in common.CSV_FIELDS}
    row.update({"benchmark": "bench_refactor_csc", "category": "proof",
                "matrix": "nos4.mtx", "scenario": "chol_spd", "n": "100",
                "nnz": "594", "refactor_csc_ms": value})
    writer.writerow([row[name] for name in common.CSV_FIELDS])
    return buffer.getvalue().encode()


def rehash(directory: Path) -> None:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "hashes.sha256")
    (directory / "hashes.sha256").write_text("".join(
        f"{common.digest(path.read_bytes())}  {path.relative_to(directory).as_posix()}\n"
        for path in files
    ), encoding="ascii")


def make_bundle(root: Path, replica: int, candidate_commit: str,
                candidate_values: list[str] | None = None,
                policy: str = pair.POLICY) -> Path:
    directory = root / f"sprint225-pair-1234-1-{replica}"
    (directory / "trials").mkdir(parents=True)
    (directory / "build-baseline.log").write_bytes(b"synthetic baseline build\n")
    (directory / "build-candidate.log").write_bytes(b"synthetic candidate build\n")
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    config = {"cc": "cc", "sysroot": "/SyntheticSDK", "ldflags": "-lm"}
    config["cflags"] = "-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 -isysroot /SyntheticSDK"
    config["sparse_cflags"] = config["cflags"] + " -DSPARSE_IDX_BITS=32"

    def side(commit: str, binary: str) -> dict:
        return {"source_commit": commit, "source_dirty": False,
                "effective_build": config, "executable_sha256": binary * 64,
                "library_sha256": "3" * 64,
                "benchmark_source_sha256": pair.source_sha(commit, "benchmarks/bench_refactor_csc.c", ROOT),
                "makefile_sha256": pair.source_sha(commit, "Makefile", ROOT),
                "fixture_sha256": common.FIXTURE_SHA256}

    argv = ["build/bench_refactor_csc", *collector.HOSTED_COMMANDS[policy]]
    run = {"schema_version": 1, "policy_version": policy,
           "baseline_commit": pair.BASELINE_COMMIT, "candidate_commit": candidate_commit,
           "baseline": side(pair.BASELINE_COMMIT, "1"),
           "candidate": side(candidate_commit, "2"),
           "workflow_run_id": "1234", "workflow_run_attempt": "1",
           "workflow_job_key": "collect-pair", "replica": replica,
           "runner": {"runner_image": "macos15", "runner_image_version": "20261119.1",
                      "os": "macOS-15.7.9-x86_64", "architecture": "x86_64",
                      "cpu_model": "Intel i7", "compiler_path": "/usr/bin/cc",
                      "compiler_version": "Apple clang 17", "runner_label": f"ephemeral-{replica}"},
           "argv": argv, "warmup_pairs": 5, "measured_pairs": 20,
           "timeout_seconds_per_side": 30, "metric": "refactor_csc_ms", "metric_units": "ms",
           "thread_environment": {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"},
           "started_at_utc": stamp, "ended_at_utc": stamp}
    (directory / "run.json").write_bytes(encoded(run))
    (directory / "decision.json").write_bytes(encoded({
        "policy_version": policy, "status": "pending_pair_review",
        "all_trials_valid": True, "trial_count": 50,
        "baseline_accepted": False, "freshness_result": "separate",
    }))
    values = candidate_values or ["0.100"] * 20
    rows = []
    for number in range(1, 26):
        order = ("baseline", "candidate") if number % 2 else ("candidate", "baseline")
        for side_name in order:
            value = "0.100" if side_name == "baseline" or number <= 5 else values[number - 6]
            stdout = metric_csv(value)
            stem = directory / "trials" / f"{number:04d}-{side_name}"
            stem.with_suffix(".stdout").write_bytes(stdout)
            stem.with_suffix(".stderr").write_bytes(b"")
            rows.append({"pair": number, "phase": "warmup" if number <= 5 else "measured",
                         "side": side_name, "argv": json.dumps(argv),
                         "started_at_utc": stamp, "ended_at_utc": stamp,
                         "outer_elapsed_ns": 1000, "exit_code": 0, "timed_out": False,
                         "refactor_csc_ms": value, "validity": "valid",
                         "stdout_sha256": common.digest(stdout),
                         "stderr_sha256": common.digest(b"")})
    with (directory / "trials.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair.FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    rehash(directory)
    return directory


class PairDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        self.directories = [make_bundle(self.root, replica, self.commit) for replica in (1, 2, 3)]

    def decide(self):
        return pair.decide(self.root, self.commit, ROOT, "1234", "1")

    def test_three_valid_pairs_pass(self):
        report, code = self.decide()
        self.assertEqual((report["primary_rule"], code), ("S225_PAIR_PASS", 0))
        self.assertEqual(report["reference_median_ratio"], "1")
        self.assertEqual(len(report["jobs"]), 3)

    def test_v5_longer_workload_is_separate_from_v3(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directories = [
                make_bundle(root, replica, self.commit, policy="s225-v5")
                for replica in (1, 2, 3)
            ]
            report, code = pair.decide(root, self.commit, ROOT, "1234", "1", "s225-v5")
            self.assertEqual((report["policy_version"], code), ("s225-v5", 0))
            self.assertEqual(pair.decide(root, self.commit, ROOT, "1234", "1")[1], 2)
            self.change_run(directories[0], "argv", pair.EXPECTED_ARGV)
            rehash(directories[0])
            self.assertEqual(pair.decide(root, self.commit, ROOT, "1234", "1", "s225-v5")[1], 2)

    def test_zip_and_extracted_directory_bind_the_same_raw_members(self):
        directory = self.directories[0]
        archive_path = self.root / "pair.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(directory).as_posix())
        members_dir, digest_dir = pair.read_bundle(directory)
        members_zip, digest_zip = pair.read_bundle(archive_path)
        self.assertEqual(members_dir, members_zip)
        self.assertEqual(digest_dir, digest_zip)
        result_dir, _ = pair.check_pair(directory, 1, self.commit, ROOT, "1234", "1")
        result_zip, _ = pair.check_pair(archive_path, 1, self.commit, ROOT, "1234", "1")
        self.assertEqual(result_dir, result_zip)

    def test_hosted_build_command_stops_before_job_timeout(self):
        self.assertLess(collector.COLLECTION_LIMIT_SECONDS, 20 * 60)
        log = self.root / "timed-build.log"
        start = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "deadline expired"):
            collector.run_command(
                [sys.executable, "-c", "import time; print('build-started', flush=True); time.sleep(10)"],
                self.root, start + 0.1, log,
            )
        self.assertLess(time.monotonic() - start, 3)
        self.assertIn(b"build-started", log.read_bytes())

    def test_hosted_output_cannot_be_removed_by_candidate_make_clean(self):
        with self.assertRaisesRegex(ValueError, "outside the candidate build tree"):
            collector.collect(ROOT / "build" / "sprint225-pair", 1)
        self.assertEqual(subprocess.run(
            ["git", "check-ignore", "-q", "sprint225-pair-raw/trials.tsv"],
            cwd=ROOT, check=False,
        ).returncode, 0)

    def test_hosted_failure_keeps_blocked_receipt_for_upload(self):
        output = self.root / "failed-pair"
        def fail_after_start(path, replica, policy):
            path.mkdir()
            (path / "build-baseline.log").write_bytes(b"compiler failed\n")
            raise TimeoutError("hosted collection deadline expired")
        with mock.patch.object(collector, "_collect", side_effect=fail_after_start):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                collector.collect(output, 1)
        self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "blocked")
        self.assertIn("build-baseline.log", (output / "hashes.sha256").read_text())

    def test_directory_total_limit_rejects_before_opening_file(self):
        directory = self.directories[0]
        for number in range(1, 12):
            (directory / "trials" / f"{number:04d}-baseline.stdout").write_bytes(
                b"x" * (common.MAX_FILE - 1)
            )
        with mock.patch.object(Path, "open", side_effect=AssertionError("read before size check")):
            with self.assertRaisesRegex(common.EvidenceError, "oversized raw pair"):
                pair.read_bundle(directory)

    def test_unexpected_members_reject_before_reading(self):
        directory = self.directories[0]
        (directory / "surprise.bin").write_bytes(b"unexpected")
        with mock.patch.object(Path, "open", side_effect=AssertionError("unexpected file was read")):
            with self.assertRaisesRegex(common.EvidenceError, "unexpected or unsafe member"):
                pair.read_bundle(directory)
        archive_path = self.root / "unexpected.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(directory).as_posix())
        with mock.patch.object(zipfile.ZipFile, "read",
                               side_effect=AssertionError("unexpected ZIP was read")):
            with self.assertRaisesRegex(common.EvidenceError, "unexpected or unsafe ZIP member"):
                pair.read_bundle(archive_path)

    def test_regression_and_dispersion_are_distinct_failures(self):
        for name, values, rule in (
            ("regression", ["0.130"] * 20, "S225_PAIR_REGRESSION"),
            ("dispersion", ["0.100"] * 10 + ["0.200"] * 10, "S225_PAIR_DISPERSION"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for replica in (1, 2, 3):
                    make_bundle(root, replica, self.commit, values)
                report, code = pair.decide(root, self.commit, ROOT, "1234", "1")
                self.assertEqual((report["primary_rule"], code), (rule, 1))

    def test_mutated_raw_evidence_fails_even_with_rehashed_manifest(self):
        for name, mutate in (
            ("wrong-source", lambda d: self.change_run(d, "candidate_commit", "f" * 40)),
            ("wrong-run", lambda d: self.change_run(d, "workflow_run_id", "9999")),
            ("wrong-order", lambda d: self.change_trial(d, 0, "side", "candidate")),
            ("wrong-argv", lambda d: self.change_trial(d, 0, "argv", "[]")),
            ("wrong-metric", lambda d: self.change_trial(d, 0, "refactor_csc_ms", "0.101")),
            ("extra-member", lambda d: (d / "surprise.txt").write_text("extra")),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                directories = [make_bundle(root, replica, self.commit) for replica in (1, 2, 3)]
                mutate(directories[0])
                rehash(directories[0])
                report, code = pair.decide(root, self.commit, ROOT, "1234", "1")
                self.assertEqual((report["primary_rule"], code), ("S225_PAIR_INVALID", 2))

    def test_missing_replica_and_runner_mismatch_fail(self):
        self.assertEqual(self.decide()[1], 0)
        directory = self.directories[2]
        shutil.rmtree(directory)
        self.assertEqual(self.decide()[0]["primary_rule"], "S225_PAIR_MISSING")
        make_bundle(self.root, 3, self.commit)
        self.change_run(self.directories[1], "runner_image_version", "different", nested="runner")
        rehash(self.directories[1])
        self.assertEqual(self.decide()[0]["primary_rule"], "S225_PAIR_INVALID")

    def test_changed_committed_benchmark_harness_is_rejected(self):
        worktree = self.root / "changed-harness"
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
                       cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            harness = worktree / "benchmarks/bench_refactor_csc.c"
            with harness.open("a", encoding="utf-8") as stream:
                stream.write("\n/* Different timing harness for this negative fixture. */\n")
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "commit", "-am", "Change benchmark harness"], cwd=worktree, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            changed_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                     cwd=worktree, text=True).strip()
            bundle_root = self.root / "changed-bundle"
            directory = make_bundle(bundle_root, 1, changed_commit)
            with self.assertRaisesRegex(common.EvidenceError,
                                        "baseline and candidate benchmark source differ"):
                pair.check_pair(directory, 1, changed_commit, ROOT, "1234", "1")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                           cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @staticmethod
    def change_run(directory: Path, field: str, value: str, nested: str | None = None):
        path = directory / "run.json"
        run = json.loads(path.read_bytes())
        (run[nested] if nested else run)[field] = value
        path.write_bytes(encoded(run))

    @staticmethod
    def change_trial(directory: Path, row_index: int, field: str, value: str):
        path = directory / "trials.tsv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        rows[row_index][field] = value
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=pair.FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


class CollectorDiagnosticsTests(unittest.TestCase):
    def test_failed_build_prints_captured_diagnostics(self):
        failure = subprocess.CalledProcessError(
            2, ["make", "build/bench_refactor_csc"],
            output=b"compiler: missing symbol\n", stderr=b"linker: failed\n",
        )
        output = io.StringIO()
        with mock.patch.object(collector, "collect", side_effect=failure), \
             mock.patch.object(sys, "argv", ["collect_refactor_csc_pair.py", "--output-dir", "unused", "--replica", "1"]), \
             contextlib.redirect_stderr(output):
            self.assertEqual(collector.main(), 2)
        self.assertIn("compiler: missing symbol", output.getvalue())
        self.assertIn("linker: failed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
