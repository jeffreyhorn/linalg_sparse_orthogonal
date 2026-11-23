#!/usr/bin/env python3
"""Collect one source-pinned same-machine Sprint 225 paired timing job."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from collect_refactor_csc_timing import (
    FIXTURE, FIXTURE_SHA256, ROOT, check_policy_environment,
    host_metadata, parse_build_config, parse_metric, sha256,
)


POLICY = "s225-v3"
BASELINE_COMMIT = "c42d6715fad7e895bae4941a006df3563f1afc38"
COMMAND = ["tests/data/suitesparse/nos4.mtx", "--repeat", "1000"]
HOSTED_COMMANDS = {
    POLICY: COMMAND,
    "s225-v5": ["tests/data/suitesparse/nos4.mtx", "--repeat", "10000"],
}
WARMUP_PAIRS = 5
MEASURED_PAIRS = 20
TIMEOUT_SECONDS = 30
# Leave time inside the hosted 20-minute job for cleanup and artifact upload.
COLLECTION_LIMIT_SECONDS = 16 * 60
FIELDS = (
    "pair", "phase", "side", "argv", "started_at_utc", "ended_at_utc",
    "outer_elapsed_ns", "exit_code", "timed_out", "refactor_csc_ms",
    "validity", "stdout_sha256", "stderr_sha256",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("hosted pair exceeded the 16-minute collection deadline")
    return seconds


def run_command(command: list[str], root: Path, deadline: float,
                log_path: Path | None = None) -> bytes:
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
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
        raise TimeoutError(f"hosted pair deadline expired during {' '.join(command)}")
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=output)
    return output


def build(root: Path, deadline: float, log_path: Path) -> dict[str, str]:
    run_command(["make", "clean"], root, deadline, log_path)
    settings = run_command(
        ["make", "--no-print-directory", "-s", "sprint225-build-config", "SPARSE_IDX_BITS=32"],
        root, deadline, log_path,
    ).decode("utf-8")
    config = parse_build_config(settings)
    run_command(["make", "build/bench_refactor_csc", "SPARSE_IDX_BITS=32"],
                root, deadline, log_path)
    return config


def capture(root: Path, command: list[str] | None = None,
            timeout: float = TIMEOUT_SECONDS) -> tuple[bytes, bytes, int, bool]:
    process = subprocess.Popen(
        [str(root / "build/bench_refactor_csc"), *(COMMAND if command is None else command)], cwd=root,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, process.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode, True


def side_receipt(root: Path, config: dict[str, str]) -> dict[str, object]:
    return {
        "source_commit": git(root, "rev-parse", "HEAD"),
        "source_dirty": bool(git(root, "status", "--porcelain", "--untracked-files=all")),
        "effective_build": config,
        "executable_sha256": sha256((root / "build/bench_refactor_csc").read_bytes()),
        "library_sha256": sha256((root / "build/libsparse_lu_ortho.a").read_bytes()),
        "benchmark_source_sha256": sha256((root / "benchmarks/bench_refactor_csc.c").read_bytes()),
        "makefile_sha256": sha256((root / "Makefile").read_bytes()),
        "fixture_sha256": sha256((root / FIXTURE).read_bytes()),
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _collect(output: Path, replica: int, policy: str = POLICY) -> int:
    if output.exists() or replica not in (1, 2, 3):
        raise ValueError("output must be new and replica must be 1, 2, or 3")
    if policy not in HOSTED_COMMANDS:
        raise ValueError("unsupported hosted timing policy")
    if output.resolve().is_relative_to((ROOT / "build").resolve()):
        raise ValueError("output must be outside the candidate build tree; make clean removes build/")
    check_policy_environment(False)
    if sys.platform != "darwin" or git(ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("paired collection requires a clean hosted macOS checkout")
    if not os.environ.get("GITHUB_RUN_ID") or not os.environ.get("GITHUB_RUN_ATTEMPT"):
        raise ValueError("paired collection requires hosted run identity")
    if sha256((ROOT / FIXTURE).read_bytes()) != FIXTURE_SHA256:
        raise ValueError("candidate fixture differs from locked nos4 bytes")
    host = host_metadata()
    if not all(host.get(key) for key in (
        "runner_image", "runner_image_version", "os", "architecture", "cpu_model",
        "compiler_path", "compiler_version",
    )):
        raise ValueError("incomplete hosted runner identity")
    candidate_commit = git(ROOT, "rev-parse", "HEAD")
    deadline = time.monotonic() + COLLECTION_LIMIT_SECONDS
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="s225-paired-") as temp:
        baseline_root = Path(temp) / "baseline"
        try:
            run_command(["git", "worktree", "add", "--detach", str(baseline_root),
                         BASELINE_COMMIT], ROOT, deadline)
            if git(baseline_root, "rev-parse", "HEAD") != BASELINE_COMMIT:
                raise ValueError("baseline worktree commit mismatch")
            baseline_config = build(baseline_root, deadline, output / "build-baseline.log")
            candidate_config = build(ROOT, deadline, output / "build-candidate.log")
            if baseline_config != candidate_config:
                raise ValueError("baseline and candidate effective build settings differ")
            baseline = side_receipt(baseline_root, baseline_config)
            candidate = side_receipt(ROOT, candidate_config)
            if baseline["source_dirty"] or candidate["source_dirty"]:
                raise ValueError("build changed tracked source files")
            if baseline["fixture_sha256"] != candidate["fixture_sha256"]:
                raise ValueError("baseline and candidate fixture bytes differ")
            if baseline["benchmark_source_sha256"] != candidate["benchmark_source_sha256"]:
                raise ValueError("baseline and candidate benchmark source bytes differ")
            (output / "trials").mkdir()
            run: dict[str, object] = {
                "schema_version": 1, "policy_version": policy,
                "baseline": baseline, "candidate": candidate,
                "baseline_commit": BASELINE_COMMIT, "candidate_commit": candidate_commit,
                "workflow_run_id": os.environ["GITHUB_RUN_ID"],
                "workflow_run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                "workflow_job_key": os.environ.get("GITHUB_JOB", ""),
                "replica": replica, "runner": host,
                "argv": ["build/bench_refactor_csc", *HOSTED_COMMANDS[policy]],
                "warmup_pairs": WARMUP_PAIRS, "measured_pairs": MEASURED_PAIRS,
                "timeout_seconds_per_side": TIMEOUT_SECONDS,
                "metric": "refactor_csc_ms", "metric_units": "ms",
                "thread_environment": {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"},
                "started_at_utc": now(),
            }
            rows = []
            with (output / "trials.tsv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for pair in range(1, WARMUP_PAIRS + MEASURED_PAIRS + 1):
                    order = ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")
                    for side in order:
                        root = baseline_root if side == "baseline" else ROOT
                        start = now()
                        t0 = time.monotonic_ns()
                        stdout, stderr, status, timed_out = capture(
                            root, HOSTED_COMMANDS[policy],
                            timeout=min(TIMEOUT_SECONDS, remaining(deadline)),
                        )
                        elapsed = time.monotonic_ns() - t0
                        end = now()
                        stem = f"{pair:04d}-{side}"
                        (output / "trials" / f"{stem}.stdout").write_bytes(stdout)
                        (output / "trials" / f"{stem}.stderr").write_bytes(stderr)
                        metric, validity = parse_metric(stdout) if status == 0 and not timed_out else ("", "process_failed")
                        if timed_out:
                            validity = "timeout"
                        row = {
                            "pair": pair, "phase": "warmup" if pair <= WARMUP_PAIRS else "measured",
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
                            raise TimeoutError(f"hosted pair {replica} trial timed out")
                        remaining(deadline)
            run["ended_at_utc"] = now()
            write_json(output / "run.json", run)
            valid = all(row["validity"] == "valid" for row in rows)
            write_json(output / "decision.json", {
                "policy_version": policy, "status": "pending_pair_review",
                "all_trials_valid": valid, "trial_count": len(rows),
                "baseline_accepted": False, "freshness_result": "separate",
            })
            remaining(deadline)
            write_manifest(output)
            print(f"collected {len(rows)} paired side trials; valid={valid}; replica={replica}")
            return 0 if valid else 1
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline_root)],
                           cwd=ROOT, check=False, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=30)


def write_manifest(output: Path) -> None:
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name != "hashes.sha256")
    (output / "hashes.sha256").write_text("".join(
        f"{sha256(path.read_bytes())}  {path.relative_to(output).as_posix()}\n"
        for path in paths
    ), encoding="ascii")


def collect(output: Path, replica: int, policy: str = POLICY) -> int:
    try:
        return _collect(output, replica, policy)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            TimeoutError, ValueError) as exc:
        if output.is_dir():
            write_json(output / "failure.json", {
                "policy_version": policy, "replica": replica,
                "status": "blocked", "reason": str(exc), "ended_at_utc": now(),
            })
            write_manifest(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--policy", choices=tuple(HOSTED_COMMANDS), default=POLICY)
    args = parser.parse_args()
    try:
        return collect(args.output_dir, args.replica, args.policy)
    except subprocess.CalledProcessError as exc:
        print(f"refactor-csc-pair: {exc}", file=sys.stderr)
        for stream in (exc.stdout, exc.stderr):
            if stream:
                detail = stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else stream
                print(detail, file=sys.stderr)
        return 2
    except (OSError, subprocess.TimeoutExpired, TimeoutError, ValueError) as exc:
        print(f"refactor-csc-pair: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
