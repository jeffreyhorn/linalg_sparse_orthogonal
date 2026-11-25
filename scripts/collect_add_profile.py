#!/usr/bin/env python3
"""Collect the frozen Sprint 227 paired sparse-add profile."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


FIXTURES = ("wide-row", "varied-rows", "narrow-rows")
VARIANTS = ("baseline", "candidate")
PER_COMMAND_SECONDS = 30.0
COLLECTION_SECONDS = 15.0 * 60.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_process(argv: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise RuntimeError(f"command timed out after {timeout:.3f}s: {argv!r}")
    return process.returncode, stdout, stderr


def remaining_seconds(start: float) -> float:
    remaining = COLLECTION_SECONDS - (time.monotonic() - start)
    if remaining <= 0.0:
        raise RuntimeError("15-minute collection cap exceeded")
    return min(PER_COMMAND_SECONDS, remaining)


def validate_smoke(stdout: bytes, fixture: str) -> None:
    rows = list(csv.DictReader(stdout.decode("utf-8").splitlines()))
    if len(rows) != 1 or rows[0].get("fixture") != fixture:
        raise RuntimeError(f"warmup did not emit one matching fixture row: {fixture}")


def measured_run(
    executable: Path,
    fixture: str,
    variant: str,
    round_number: int,
    order_in_pair: int,
    raw: Path,
    collection_start: float,
) -> None:
    stem = raw / f"{fixture}-{round_number}-{variant}"
    for extension in ("stdout", "stderr", "time", "status", "meta"):
        if stem.with_suffix(f".{extension}").exists():
            raise RuntimeError(f"refusing to overwrite {stem}.{extension}")

    timing_path = stem.with_suffix(".time")
    argv = ["/usr/bin/time", "-l", "-o", str(timing_path), str(executable), fixture]
    started_at = utc_now()
    started = time.monotonic()
    status, stdout, stderr = run_process(argv, remaining_seconds(collection_start))
    ended_at = utc_now()
    duration = time.monotonic() - started

    stem.with_suffix(".stdout").write_bytes(stdout)
    stem.with_suffix(".stderr").write_bytes(stderr)
    stem.with_suffix(".status").write_text(f"{status}\n", encoding="utf-8")
    metadata = {
        "fixture": fixture,
        "round": round_number,
        "variant": variant,
        "order_in_pair": order_in_pair,
        "argv": argv,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_seconds": duration,
        "per_command_cap_seconds": PER_COMMAND_SECONDS,
        "collection_cap_seconds": COLLECTION_SECONDS,
    }
    stem.with_suffix(".meta").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != 0:
        raise RuntimeError(f"measured command failed with status {status}: {argv!r}")
    validate_smoke(stdout, fixture)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()

    executables = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    for variant, executable in executables.items():
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"{variant} executable is not runnable: {executable}")
    if args.raw.exists():
        if any(args.raw.iterdir()):
            raise RuntimeError(f"raw directory is not empty: {args.raw}")
    else:
        args.raw.mkdir(parents=True)

    collection_start = time.monotonic()
    print(f"collection started: {utc_now()}")
    for fixture in FIXTURES:
        for variant in VARIANTS:
            for warmup in range(1, 3):
                argv = [str(executables[variant]), fixture]
                status, stdout, stderr = run_process(argv, remaining_seconds(collection_start))
                if status != 0 or stderr or not stdout:
                    raise RuntimeError(
                        f"warmup failed: fixture={fixture} variant={variant} "
                        f"warmup={warmup} status={status} stderr={stderr!r}"
                    )
                validate_smoke(stdout, fixture)

        for round_number in range(1, 11):
            order = VARIANTS if round_number % 2 == 1 else tuple(reversed(VARIANTS))
            for order_in_pair, variant in enumerate(order, start=1):
                measured_run(
                    executables[variant],
                    fixture,
                    variant,
                    round_number,
                    order_in_pair,
                    args.raw,
                    collection_start,
                )
    elapsed = time.monotonic() - collection_start
    print(f"collection completed: {utc_now()} ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()
