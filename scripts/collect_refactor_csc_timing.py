#!/usr/bin/env python3
"""Capture one Sprint 225 refactor CSC trial job without making a gate decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path("tests/data/suitesparse/nos4.mtx")
FIXTURE_SHA256 = "6450e9e3d921764802075128a053b0c13debf4462152bf4fdaa059c29e7e3a80"
POLICY_VERSION = "s225-v1"
COMMAND = ["tests/data/suitesparse/nos4.mtx", "--repeat", "1"]
TIMEOUT_SECONDS = 30
FORBIDDEN_OVERRIDES = (
    "CC", "CFLAGS", "CPPFLAGS", "LDFLAGS", "SPARSE_MUTEX", "SPARSE_OPENMP",
    "SPARSE_IDX_BITS", "SYSROOT", "SDKROOT", "MAKEFLAGS", "MFLAGS",
    "GNUMAKEFLAGS", "MAKEOVERRIDES", "SPARSE_LDLT_DENSE_BACKEND",
    "SPARSE_CHOL_DENSE_BACKEND", "SPARSE_SUPERNODAL_POSTORDER",
    "SPARSE_ND_SUPERNODAL_POSTORDER",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def capture_process(argv: list[str], env: dict[str, str]) -> tuple[bytes, bytes, int | None, bool]:
    process = subprocess.Popen(
        argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        return stdout, stderr, process.returncode, False
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode, True


def parse_metric(stdout: bytes) -> tuple[str, str]:
    try:
        contents = stdout.decode("utf-8", errors="strict")
        rows = list(csv.DictReader(io.StringIO(contents)))
    except (UnicodeDecodeError, csv.Error):
        return "", "invalid_csv"
    if len(rows) != 1 or not rows[0] or None in rows[0]:
        return "", "wrong_row_count_or_shape"
    row = rows[0]
    if any(row.get(key) != value for key, value in (
        ("benchmark", "bench_refactor_csc"), ("category", "proof"),
        ("matrix", "nos4.mtx"), ("scenario", "chol_spd"), ("n", "100"),
    )):
        return "", "wrong_selected_row"
    from decimal import Decimal, InvalidOperation

    try:
        value = Decimal(row["refactor_csc_ms"])
    except (KeyError, InvalidOperation, TypeError):
        return "", "invalid_metric"
    if not value.is_finite() or value <= 0:
        return "", "invalid_metric"
    return str(value), "valid"


def host_metadata() -> dict[str, str]:
    compiler = shutil.which("cc") or ""
    version = ""
    if compiler:
        version = subprocess.check_output([compiler, "--version"], text=True).strip()
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {
        "runner_label": os.environ.get("RUNNER_NAME", ""),
        "runner_image": os.environ.get("ImageOS", ""),
        "runner_image_version": os.environ.get("ImageVersion", ""),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": cpu,
        "compiler_path": compiler,
        "compiler_version": version,
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_build_config(output: str) -> dict[str, str]:
    fields = {}
    for line in output.splitlines():
        if "\t" not in line:
            raise ValueError(f"unexpected Make build setting: {line}")
        key, value = line.split("\t", 1)
        if key in fields:
            raise ValueError(f"duplicate Make build setting: {key}")
        fields[key] = value
    if set(fields) != {"cc", "sysroot", "cflags", "ldflags", "sparse_cflags"}:
        raise ValueError("incomplete effective Make build settings")
    return fields


def effective_build_config() -> dict[str, str]:
    output = subprocess.check_output(
        ["make", "--no-print-directory", "-s", "sprint225-build-config", "SPARSE_IDX_BITS=32"],
        cwd=ROOT, text=True,
    )
    return parse_build_config(output)


def check_policy_environment(dry_run: bool) -> None:
    overrides = [name for name in FORBIDDEN_OVERRIDES if name in os.environ]
    if overrides and not dry_run:
        raise ValueError(f"build or workload overrides outside locked policy: {', '.join(overrides)}")
    if any(os.environ.get(name) for name in (
        "SPARSE_LDLT_DENSE_BACKEND", "SPARSE_CHOL_DENSE_BACKEND", "SPARSE_OPENMP",
        "SPARSE_SUPERNODAL_POSTORDER", "SPARSE_ND_SUPERNODAL_POSTORDER",
    )):
        raise ValueError("backend or symbolic workload override is outside locked policy")


def collect(output: Path, dry_run: bool, role: str = "baseline",
            build_config_path: Path | None = None) -> int:
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    fixture = ROOT / FIXTURE
    if sha256(fixture.read_bytes()) != FIXTURE_SHA256:
        raise ValueError("nos4 fixture SHA-256 differs from locked policy")
    binary = ROOT / "build/bench_refactor_csc"
    library = ROOT / "build/libsparse_lu_ortho.a"
    if not binary.is_file() or not library.is_file():
        raise ValueError("build benchmark and library before collecting trials")
    check_policy_environment(dry_run)
    current_build = effective_build_config()
    if build_config_path is None:
        if not dry_run:
            raise ValueError("official collection requires a build-step configuration receipt")
        build_config = current_build
    else:
        build_config = parse_build_config(build_config_path.read_text(encoding="utf-8"))
        if build_config != current_build:
            raise ValueError("build-step and collection Make settings differ")
    host = host_metadata()
    dirty = bool(git_value("status", "--porcelain", "--untracked-files=all"))
    if not dry_run:
        if sys.platform != "darwin" or dirty:
            raise ValueError("official collection requires clean macOS checkout")
        if not all(host[key] for key in (
            "runner_image", "runner_image_version", "cpu_model", "compiler_path", "compiler_version"
        )):
            raise ValueError("official collection requires complete runner/compiler identity")
        if not os.environ.get("GITHUB_RUN_ID") or not os.environ.get("GITHUB_RUN_ATTEMPT"):
            raise ValueError("official collection requires hosted run identity")

    warmups, measured = (1, 1) if dry_run else (5, 20)
    output.mkdir(parents=True)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    run: dict[str, object] = {
        "schema_version": 2, "policy_version": POLICY_VERSION,
        "role": role, "dry_run": dry_run,
        "source_commit": git_value("rev-parse", "HEAD"), "source_dirty": dirty,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "workflow_job_key": os.environ.get("GITHUB_JOB", ""),
        "runner": host, "build_recipe": "make build/bench_refactor_csc SPARSE_IDX_BITS=32",
        "effective_build": build_config,
        "build_mode": "serial", "index_width_bits": 32,
        "thread_environment": {"OMP_NUM_THREADS": "1", "SPARSE_OPENMP": "unset"},
        "backend_environment": {
            "SPARSE_LDLT_DENSE_BACKEND": "unset", "SPARSE_CHOL_DENSE_BACKEND": "unset",
            "SPARSE_SUPERNODAL_POSTORDER": "unset",
            "SPARSE_ND_SUPERNODAL_POSTORDER": "unset",
        },
        "executable_sha256": sha256(binary.read_bytes()),
        "library_sha256": sha256(library.read_bytes()),
        "benchmark_source_sha256": sha256((ROOT / "benchmarks/bench_refactor_csc.c").read_bytes()),
        "makefile_sha256": sha256((ROOT / "Makefile").read_bytes()),
        "fixture_sha256": FIXTURE_SHA256,
        "argv": ["build/bench_refactor_csc", *COMMAND],
        "timeout_seconds": TIMEOUT_SECONDS,
        "warmup_count": warmups, "measured_count": measured,
        "metric": "refactor_csc_ms", "metric_units": "ms",
        "started_at_utc": utc_now(),
    }
    records: list[dict[str, object]] = []
    columns = (
        "ordinal", "phase", "argv", "started_at_utc", "ended_at_utc",
        "outer_elapsed_ns", "exit_code", "timed_out", "refactor_csc_ms",
        "validity", "stdout_sha256", "stderr_sha256",
    )
    with (output / "trials.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        stream.flush()
        for ordinal in range(1, warmups + measured + 1):
            phase = "warmup" if ordinal <= warmups else "measured"
            started = utc_now()
            t0 = time.monotonic_ns()
            stdout, stderr, exit_code, timed_out = capture_process([str(binary), *COMMAND], env)
            elapsed_ns = time.monotonic_ns() - t0
            ended = utc_now()
            stem = f"{ordinal:04d}"
            (trials_dir / f"{stem}.stdout").write_bytes(stdout)
            (trials_dir / f"{stem}.stderr").write_bytes(stderr)
            metric, reason = parse_metric(stdout) if exit_code == 0 and not timed_out else ("", "process_failed")
            if timed_out:
                reason = "timeout"
            row: dict[str, object] = {
                "ordinal": ordinal, "phase": phase,
                "argv": json.dumps(["build/bench_refactor_csc", *COMMAND]),
                "started_at_utc": started, "ended_at_utc": ended,
                "outer_elapsed_ns": elapsed_ns, "exit_code": exit_code,
                "timed_out": timed_out, "refactor_csc_ms": metric, "validity": reason,
                "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr),
            }
            writer.writerow(row)
            stream.flush()
            records.append(row)
            write_json(output / "progress.json", {"attempted": ordinal, "expected": warmups + measured})

    run["ended_at_utc"] = utc_now()
    write_json(output / "run.json", run)
    valid = all(row["validity"] == "valid" for row in records)
    write_json(output / "decision.json", {
        "policy_version": POLICY_VERSION, "role": role, "dry_run": dry_run,
        "status": "dry_run_only" if dry_run else (
            "pending_baseline_review" if role == "baseline" else "pending_timing_review"
        ),
        "trial_count": len(records), "all_trials_valid": valid,
        "freshness_result": "separate", "baseline_accepted": False,
    })
    (output / "progress.json").unlink()
    paths = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "hashes.sha256").write_text(
        "".join(f"{sha256(path.read_bytes())}  {path.relative_to(output).as_posix()}\n" for path in paths),
        encoding="ascii",
    )
    print(f"collected {len(records)} trials in {output}; valid={valid}; dry_run={dry_run}")
    return 0 if valid else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="one warmup and one measured trial; never baseline evidence")
    parser.add_argument("--role", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--build-config", type=Path,
                        help="Make-expanded settings captured before the pinned benchmark build")
    args = parser.parse_args()
    try:
        return collect(args.output_dir, args.dry_run, args.role, args.build_config)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"refactor-csc-collector: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
