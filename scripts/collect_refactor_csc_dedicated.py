#!/usr/bin/env python3
"""Collect the proposed Sprint 225 v4 pairs on one isolated Intel Mac."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from collect_refactor_csc_pair import (
    BASELINE_COMMIT, FIELDS, MEASURED_PAIRS, TIMEOUT_SECONDS, WARMUP_PAIRS,
    capture, git, now, side_receipt, write_json,
)
from collect_refactor_csc_timing import (
    FIXTURE, FIXTURE_SHA256, ROOT, check_policy_environment, host_metadata,
    parse_build_config, parse_metric, sha256,
)


POLICY = "s225-v4"
COMMAND = ["tests/data/suitesparse/nos4.mtx", "--repeat", "10000"]
RUNNER_IMAGE = "dedicated-macos-intel"
JOB_LIMIT_SECONDS = 20 * 60


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("replica exceeded the 20-minute deadline")
    return seconds


def run_process(command: list[str], cwd: Path, deadline: float,
                log_path: Path | None = None) -> bytes:
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=remaining(deadline))
    except (subprocess.TimeoutExpired, TimeoutError):
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
    if log_path is not None:
        with log_path.open("ab") as stream:
            stream.write(b"$ " + " ".join(command).encode() + b"\n")
            stream.write(output)
    if timed_out:
        raise TimeoutError(f"replica deadline expired during {' '.join(command)}")
    if process.returncode != 0:
        raise ValueError(f"command failed ({process.returncode}): {' '.join(command)}")
    return output


def build_side(root: Path, deadline: float, log_path: Path) -> dict[str, str]:
    run_process(["make", "clean"], root, deadline, log_path)
    config = parse_build_config(run_process(
        ["make", "--no-print-directory", "-s", "sprint225-build-config",
         "SPARSE_IDX_BITS=32"], root, deadline, log_path,
    ).decode("utf-8"))
    run_process(["make", "build/bench_refactor_csc", "SPARSE_IDX_BITS=32"],
                root, deadline, log_path)
    return config


def require_dedicated_host(host_id: str, operator: str, attested: bool) -> dict[str, str]:
    if not attested or not host_id.strip() or not operator.strip():
        raise ValueError("operator, host ID, and exclusive-window attestation are required")
    if sys.platform != "darwin" or platform.machine() != "x86_64":
        raise ValueError("v4 collection requires a dedicated Intel macOS host")
    if os.environ.get("GITHUB_ACTIONS"):
        raise ValueError("v4 collection must run offline, outside GitHub Actions")
    check_policy_environment(False)
    if git(ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("candidate checkout must be clean")
    if sha256((ROOT / FIXTURE).read_bytes()) != FIXTURE_SHA256:
        raise ValueError("candidate fixture differs from locked nos4 bytes")
    host = host_metadata()
    host["runner_label"] = host_id.strip()
    host["runner_image"] = RUNNER_IMAGE
    host["runner_image_version"] = platform.mac_ver()[0]
    if not all(host.get(key) for key in (
        "runner_image_version", "os", "architecture", "cpu_model",
        "compiler_path", "compiler_version",
    )):
        raise ValueError("dedicated host identity is incomplete")
    return host


def _collect_one(output: Path, replica: int, collection_id: str, candidate_commit: str,
                 host: dict[str, str], operator: str) -> None:
    deadline = time.monotonic() + JOB_LIMIT_SECONDS
    started_at = now()
    with tempfile.TemporaryDirectory(prefix="s225-v4-") as temp:
        baseline_root = Path(temp) / "baseline"
        candidate_root = Path(temp) / "candidate"
        try:
            for target, commit in ((baseline_root, BASELINE_COMMIT),
                                   (candidate_root, candidate_commit)):
                run_process(["git", "worktree", "add", "--detach", str(target), commit],
                            ROOT, deadline)
            baseline_config = build_side(
                baseline_root, deadline, output / "build-baseline.log",
            )
            candidate_config = build_side(
                candidate_root, deadline, output / "build-candidate.log",
            )
            if baseline_config != candidate_config:
                raise ValueError("baseline and candidate effective build settings differ")
            baseline = side_receipt(baseline_root, baseline_config)
            candidate = side_receipt(candidate_root, candidate_config)
            if baseline["source_dirty"] or candidate["source_dirty"]:
                raise ValueError("build changed tracked source files")
            if baseline["fixture_sha256"] != candidate["fixture_sha256"]:
                raise ValueError("baseline and candidate fixture bytes differ")
            if baseline["benchmark_source_sha256"] != candidate["benchmark_source_sha256"]:
                raise ValueError("baseline and candidate benchmark source bytes differ")
            (output / "trials").mkdir()
            run: dict[str, object] = {
                "schema_version": 1, "policy_version": POLICY,
                "baseline": baseline, "candidate": candidate,
                "baseline_commit": BASELINE_COMMIT, "candidate_commit": candidate_commit,
                "collection_id": collection_id, "collection_attempt": "1",
                "execution_mode": "dedicated_offline", "replica": replica,
                "runner": host, "operator_attestation": {
                    "operator": operator, "exclusive_window_acknowledged": True,
                },
                "argv": ["build/bench_refactor_csc", *COMMAND],
                "warmup_pairs": WARMUP_PAIRS, "measured_pairs": MEASURED_PAIRS,
                "timeout_seconds_per_side": TIMEOUT_SECONDS,
                "metric": "refactor_csc_ms", "metric_units": "ms",
                "thread_environment": {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"},
                "started_at_utc": started_at,
            }
            rows = []
            with (output / "trials.tsv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for number in range(1, WARMUP_PAIRS + MEASURED_PAIRS + 1):
                    order = ("baseline", "candidate") if number % 2 else ("candidate", "baseline")
                    for side in order:
                        root = baseline_root if side == "baseline" else candidate_root
                        start = now()
                        t0 = time.monotonic_ns()
                        timeout = min(TIMEOUT_SECONDS, remaining(deadline))
                        stdout, stderr, status, timed_out = capture(root, COMMAND, timeout)
                        elapsed = time.monotonic_ns() - t0
                        end = now()
                        stem = f"{number:04d}-{side}"
                        (output / "trials" / f"{stem}.stdout").write_bytes(stdout)
                        (output / "trials" / f"{stem}.stderr").write_bytes(stderr)
                        metric, validity = parse_metric(stdout) if status == 0 and not timed_out else ("", "process_failed")
                        if timed_out:
                            validity = "timeout"
                        row = {
                            "pair": number, "phase": "warmup" if number <= WARMUP_PAIRS else "measured",
                            "side": side, "argv": json.dumps(run["argv"]),
                            "started_at_utc": start, "ended_at_utc": end,
                            "outer_elapsed_ns": elapsed, "exit_code": status,
                            "timed_out": timed_out, "refactor_csc_ms": metric,
                            "validity": validity, "stdout_sha256": sha256(stdout),
                            "stderr_sha256": sha256(stderr),
                        }
                        writer.writerow(row)
                        stream.flush()
                        rows.append(row)
                        if timed_out:
                            raise TimeoutError(f"replica {replica} trial exceeded its deadline")
                        remaining(deadline)
            run["ended_at_utc"] = now()
            write_json(output / "run.json", run)
            valid = all(row["validity"] == "valid" for row in rows)
            write_json(output / "decision.json", {
                "policy_version": POLICY, "status": "pending_pair_review",
                "all_trials_valid": valid, "trial_count": len(rows),
                "baseline_accepted": False, "freshness_result": "separate",
            })
            write_manifest(output)
            remaining(deadline)
            if not valid:
                raise ValueError(f"replica {replica} has invalid trials")
        finally:
            for target in (candidate_root, baseline_root):
                subprocess.run(["git", "worktree", "remove", "--force", str(target)],
                               cwd=ROOT, check=False, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=30)


def write_manifest(output: Path) -> None:
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "hashes.sha256")
    (output / "hashes.sha256").write_text("".join(
        f"{sha256(path.read_bytes())}  {path.relative_to(output).as_posix()}\n"
        for path in paths
    ), encoding="ascii")


def collect_one(output: Path, replica: int, collection_id: str, candidate_commit: str,
                host: dict[str, str], operator: str) -> None:
    output.mkdir(parents=True)
    try:
        _collect_one(output, replica, collection_id, candidate_commit, host, operator)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            TimeoutError, ValueError) as exc:
        write_json(output / "failure.json", {
            "policy_version": POLICY, "replica": replica,
            "status": "blocked", "reason": str(exc), "ended_at_utc": now(),
        })
        write_manifest(output)
        raise


def archive_evidence(root: Path, collection_id: str) -> None:
    digest_lines = []
    for replica in (1, 2, 3):
        name = f"sprint225-v4-pair-{collection_id}-1-{replica}"
        directory = root / name
        if not directory.is_dir():
            continue
        archive_path = root / f"{name}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(directory).as_posix())
        digest_lines.append(f"{sha256(archive_path.read_bytes())}  {archive_path.name}\n")
    decision = root / "sprint225-v4-decision.json"
    archive_path = root / "sprint225-v4-decision.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(decision, decision.name)
    digest_lines.append(f"{sha256(archive_path.read_bytes())}  {archive_path.name}\n")
    (root / "archive-sha256.txt").write_text("".join(digest_lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--exclusive-window-attested", action="store_true")
    args = parser.parse_args()
    try:
        host = require_dedicated_host(args.host_id, args.operator,
                                      args.exclusive_window_attested)
        root = args.output_root.resolve()
        if root.exists():
            raise ValueError("output root must be new")
        collection_id = str(time.time_ns())
        candidate_commit = git(ROOT, "rev-parse", "HEAD")
        root.mkdir(parents=True)
        for replica in (1, 2, 3):
            collect_one(root / f"sprint225-v4-pair-{collection_id}-1-{replica}",
                        replica, collection_id, candidate_commit, host, args.operator.strip())
        from check_refactor_csc_dedicated import decide

        report, code = decide(root, candidate_commit, ROOT, collection_id)
        write_json(root / "sprint225-v4-decision.json", report)
        archive_evidence(root, collection_id)
        print(f"s225-v4: {report['primary_rule']}: {report['detail']}; evidence={root}")
        return code
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            TimeoutError, ValueError) as exc:
        if "root" in locals() and root.is_dir() and "collection_id" in locals():
            write_json(root / "sprint225-v4-decision.json", {
                "schema_version": 1, "policy_version": POLICY,
                "baseline_commit": BASELINE_COMMIT,
                "candidate_commit": candidate_commit,
                "collection_id": collection_id,
                "status": "blocked", "primary_rule": "S225_V4_COLLECTION_FAILED",
                "detail": str(exc), "independent_review": "pending",
                "evaluated_at_utc": now(),
            })
            archive_evidence(root, collection_id)
        print(f"s225-v4: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            for stream in (exc.stdout, exc.stderr):
                if stream:
                    print(stream.decode("utf-8", "replace") if isinstance(stream, bytes) else stream,
                          file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
