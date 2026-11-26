#!/usr/bin/env python3
"""Collect the frozen Sprint 228 low-rank memory packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


WORKLOADS = (
    "measure-square-512-r16",
    "measure-tall-1024x256-r8",
    "measure-wide-256x1024-r24",
    "measure-zero-512-r16",
)
ROUTES = ("default", "dense")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(root: Path) -> tuple[str, list[dict[str, str]]]:
    paths = [
        root / "Makefile",
        root / "CMakeLists.txt",
        root / "scripts" / "run_sprint228_memory_packet.py",
    ]
    for directory in (root / "include", root / "src"):
        paths.extend(path for path in directory.iterdir() if path.suffix in {".c", ".h"})
    paths.append(root / "benchmarks" / "bench_svd_lowrank_memory.c")
    records = []
    combined = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        records.append({"path": relative, "sha256": digest})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return combined.hexdigest(), records


def run_text(command: list[str], root: Path) -> str:
    return subprocess.run(
        command, cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout.strip()


def collect_one(
    binary: Path,
    root: Path,
    output: Path,
    workload: str,
    route: str,
    phase: str,
    pair: int,
    order: int,
    timeout_seconds: float,
) -> dict[str, object]:
    stem = f"{phase}-{workload}-{pair:02d}-{order}-{route}"
    stdout_path = output / f"{stem}.stdout"
    stderr_path = output / f"{stem}.stderr"
    receipt_path = output / f"{stem}.receipt.json"
    started = time.monotonic_ns()
    timed_out = False
    returncode = -1
    stdout = b""
    stderr = b""
    try:
        completed = subprocess.run(
            [str(binary), workload, route],
            cwd=root,
            env={**os.environ, "OMP_NUM_THREADS": "1", "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
    elapsed = time.monotonic_ns() - started
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)

    parsed: dict[str, object] = {}
    if not timed_out and returncode == 0:
        lines = stdout.decode("utf-8").strip().splitlines()
        if len(lines) == 1:
            parsed = json.loads(lines[0])
    receipt: dict[str, object] = {
        "phase": phase,
        "workload": workload,
        "route": route,
        "pair": pair,
        "order": order,
        "command": [str(binary), workload, route],
        "timeout_seconds": timeout_seconds,
        "collector_elapsed_ns": elapsed,
        "timed_out": timed_out,
        "returncode": returncode,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "result": parsed,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def validate(samples: list[dict[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    ratios: dict[str, list[float]] = {name: [] for name in WORKLOADS}
    by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for receipt in samples:
        result = receipt.get("result")
        if receipt["timed_out"] or receipt["returncode"] != 0 or not isinstance(result, dict):
            errors.append(f"failed sample: {receipt['workload']} {receipt['pair']} {receipt['route']}")
            continue
        if not result.get("correct"):
            errors.append(f"correctness failure: {receipt['workload']} {receipt['pair']} {receipt['route']}")
        workload = str(receipt["workload"])
        route = str(receipt["route"])
        expected = 0 if workload == "measure-zero-512-r16" or route == "default" else 2_097_152
        if result.get("reconstruction_scratch_peak_bytes") != expected:
            errors.append(f"scratch mismatch: {workload} {receipt['pair']} {route}")
        by_key[(workload, int(receipt["pair"]), route)] = result

    for workload in WORKLOADS:
        for pair in range(10):
            default = by_key.get((workload, pair, "default"))
            dense = by_key.get((workload, pair, "dense"))
            if not default or not dense:
                continue
            for field in (
                "checksum",
                "nnz",
                "bidiag_work_peak_bytes",
                "retained_factor_bytes",
                "sparse_shell_map_bytes",
                "sparse_node_slab_bytes",
            ):
                if default.get(field) != dense.get(field):
                    errors.append(f"route accounting mismatch: {workload} pair {pair} field {field}")
            dense_ns = int(dense["wall_ns"])
            if dense_ns <= 0:
                errors.append(f"invalid dense timing: {workload} pair {pair}")
            else:
                ratios[workload].append(int(default["wall_ns"]) / dense_ns)

    timing = {}
    for workload, values in ratios.items():
        if not values:
            timing[workload] = {"median_ratio": None, "max_ratio": None, "pass": False}
            continue
        median_ratio = statistics.median(values)
        max_ratio = max(values)
        nonzero = workload != "measure-zero-512-r16"
        timing[workload] = {
            "median_ratio": median_ratio,
            "max_ratio": max_ratio,
            "pass": (not nonzero) or (median_ratio <= 1.25 and max_ratio <= 1.50),
        }
    memory_pass = not errors
    timing_pass = all(
        details["pass"] for name, details in timing.items() if name != "measure-zero-512-r16"
    )
    return {"errors": errors, "memory_pass": memory_pass, "timing_pass": timing_pass, "timing": timing}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/bench_svd_lowrank_memory"))
    parser.add_argument("--library", type=Path, default=Path("build/libsparse_lu_ortho.a"))
    parser.add_argument("--output", type=Path, default=Path("build/sprint228-day11-memory"))
    parser.add_argument("--allow-dirty-development", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = (root / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    library = (root / args.library).resolve() if not args.library.is_absolute() else args.library
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)

    git_head = run_text(["git", "rev-parse", "HEAD"], root)
    git_dirty = bool(run_text(["git", "status", "--porcelain", "--untracked-files=all"], root))
    if git_dirty and not args.allow_dirty_development:
        raise SystemExit("sprint228-memory: dirty worktree rejected (use development override only)")
    source_sha, source_files = source_snapshot(root)
    metadata = {
        "schema": "s228-memory-packet-v1",
        "collector_command": ["python3", "scripts/run_sprint228_memory_packet.py"]
        + (["--allow-dirty-development"] if args.allow_dirty_development else []),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "development_override": args.allow_dirty_development,
        "source_snapshot_sha256": source_sha,
        "source_files": source_files,
        "binary": str(binary.relative_to(root)),
        "binary_sha256": sha256_file(binary),
        "library": str(library.relative_to(root)),
        "library_sha256": sha256_file(library),
        "compiler": run_text([os.environ.get("CC", "cc"), "--version"], root).splitlines()[0],
        "flags": "-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 SPARSE_IDX_BITS=32",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "omp_num_threads": "1",
        "locale": "C",
        "power_state_note": "not controlled; RSS contextual and exact category counters primary",
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    collection_start = time.monotonic()
    for workload in WORKLOADS:
        for route in ROUTES:
            for warmup in range(2):
                collect_one(binary, root, output, workload, route, "warmup", warmup, warmup, 30.0)

    measured = []
    for workload in WORKLOADS:
        for pair in range(10):
            order = ROUTES if pair % 2 == 0 else tuple(reversed(ROUTES))
            for position, route in enumerate(order):
                remaining = 900.0 - (time.monotonic() - collection_start)
                if remaining <= 0:
                    raise SystemExit("sprint228-memory: 15-minute collection cap exceeded")
                measured.append(
                    collect_one(
                        binary, root, output, workload, route, "measured", pair, position,
                        min(30.0, remaining),
                    )
                )

    ending_sha, _ = source_snapshot(root)
    validation = validate(measured)
    summary = {
        "schema": "s228-memory-summary-v1",
        "source_snapshot_sha256": source_sha,
        "source_unchanged_during_collection": ending_sha == source_sha,
        "git_dirty": git_dirty,
        "development_override": args.allow_dirty_development,
        "warmup_count": 16,
        "measured_count": len(measured),
        "collection_elapsed_seconds": time.monotonic() - collection_start,
        **validation,
        "protocol_accepted": (
            not git_dirty
            and ending_sha == source_sha
            and validation["memory_pass"]
            and validation["timing_pass"]
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = []
    for path in sorted(output.iterdir()):
        if path.name == "manifest.sha256" or not path.is_file():
            continue
        manifest.append(f"{sha256_file(path)}  {path.name}")
    (output / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation["memory_pass"] and ending_sha == source_sha else 1


if __name__ == "__main__":
    raise SystemExit(main())
