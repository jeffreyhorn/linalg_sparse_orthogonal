#!/usr/bin/env python3
"""Collect the frozen Sprint 228 low-rank memory packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
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
COLLECTION_CAP_SECONDS = 900.0
INVOCATION_TIMEOUT_SECONDS = 30.0


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


def display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def load_build_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"sprint228-memory: verified build metadata missing: {path}")
    metadata: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t", 1)
        if len(fields) != 2 or not fields[0] or not fields[1] or fields[0] in metadata:
            raise SystemExit(
                f"sprint228-memory: invalid build metadata at {path}:{line_number}"
            )
        metadata[fields[0]] = fields[1]
    required = {
        "schema",
        "build_system",
        "cc",
        "compiler",
        "cflags",
        "ldflags",
        "sparse_cflags",
        "index_bits",
    }
    if set(metadata) != required or metadata["schema"] != "s228-memory-build-v1":
        raise SystemExit(f"sprint228-memory: unsupported build metadata schema: {path}")
    if metadata["build_system"] != "make" or metadata["index_bits"] not in {"32", "64"}:
        raise SystemExit(f"sprint228-memory: unverifiable build configuration: {path}")
    return metadata


def prepare_output_directory(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"sprint228-memory: output already exists; refusing overwrite: {path}")
    path.mkdir(parents=True)


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

    parsed: dict[str, object] | None = None
    result_error: str | None = None
    if not timed_out and returncode == 0:
        try:
            lines = stdout.decode("utf-8").strip().splitlines()
            if len(lines) != 1:
                result_error = f"expected one JSON line, found {len(lines)}"
            else:
                decoded = json.loads(lines[0])
                if isinstance(decoded, dict):
                    parsed = decoded
                else:
                    result_error = "probe JSON result is not an object"
        except UnicodeDecodeError as exc:
            result_error = f"probe stdout is not valid UTF-8 at byte {exc.start}"
        except json.JSONDecodeError as exc:
            result_error = f"probe stdout is not valid JSON at character {exc.pos}"
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
        "result_error": result_error,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def validate_receipt(
    receipt: dict[str, object], errors: list[str], expected_index_bits: int
) -> dict[str, object] | None:
    phase = str(receipt.get("phase", "unknown"))
    workload = str(receipt.get("workload", "unknown"))
    route = str(receipt.get("route", "unknown"))
    pair = receipt.get("pair", "unknown")
    result = receipt.get("result")
    if receipt.get("timed_out") or receipt.get("returncode") != 0 or not isinstance(result, dict):
        errors.append(f"failed {phase} sample: {workload} {pair} {route}")
        return None
    if result.get("schema") != "s228-memory-v1" or result.get("workload") != workload or result.get(
        "route"
    ) != route:
        errors.append(f"identity mismatch: {phase} {workload} {pair} {route}")
    if result.get("index_bits") != expected_index_bits:
        errors.append(f"index-width mismatch: {phase} {workload} {pair} {route}")
    if not result.get("correct"):
        errors.append(f"correctness failure: {phase} {workload} {pair} {route}")
    expected_bytes = (
        0 if workload == "measure-zero-512-r16" or route == "default" else 2_097_152
    )
    expected_allocations = 1 if expected_bytes else 0
    if (
        result.get("reconstruction_scratch_peak_bytes") != expected_bytes
        or result.get("reconstruction_scratch_allocations") != expected_allocations
    ):
        errors.append(f"scratch mismatch: {phase} {workload} {pair} {route}")
    return result


def validate(
    warmups: list[dict[str, object]],
    samples: list[dict[str, object]],
    expected_index_bits: int,
) -> dict[str, object]:
    errors: list[str] = []
    ratios: dict[str, list[float]] = {name: [] for name in WORKLOADS}
    by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for receipt in warmups:
        validate_receipt(receipt, errors, expected_index_bits)
    for receipt in samples:
        result = validate_receipt(receipt, errors, expected_index_bits)
        if result is None:
            continue
        workload = str(receipt["workload"])
        route = str(receipt["route"])
        by_key[(workload, int(receipt["pair"]), route)] = result

    for workload in WORKLOADS:
        for pair in range(10):
            default = by_key.get((workload, pair, "default"))
            dense = by_key.get((workload, pair, "dense"))
            if not default or not dense:
                errors.append(f"missing measured pair: {workload} pair {pair}")
                continue
            for field in (
                "checksum",
                "nnz",
                "bidiag_work_peak_bytes",
                "retained_factor_bytes",
                "sparse_shell_map_bytes",
                "sparse_node_slab_bytes",
            ):
                default_value = default.get(field)
                dense_value = dense.get(field)
                if default_value is None:
                    errors.append(
                        f"missing default accounting: {workload} pair {pair} field {field}"
                    )
                if dense_value is None:
                    errors.append(
                        f"missing dense accounting: {workload} pair {pair} field {field}"
                    )
                if (
                    default_value is not None
                    and dense_value is not None
                    and default_value != dense_value
                ):
                    errors.append(f"route accounting mismatch: {workload} pair {pair} field {field}")
            try:
                dense_ns = int(dense["wall_ns"])
                default_ns = int(default["wall_ns"])
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"invalid timing: {workload} pair {pair}")
                continue
            if dense_ns <= 0:
                errors.append(f"invalid dense timing: {workload} pair {pair}")
            elif default_ns <= 0:
                errors.append(f"invalid default timing: {workload} pair {pair}")
            else:
                ratios[workload].append(default_ns / dense_ns)

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


def collect_samples(
    binary: Path,
    root: Path,
    output: Path,
    collection_start: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], bool]:
    warmups: list[dict[str, object]] = []
    measured: list[dict[str, object]] = []
    collection_cap_exceeded = False

    warmup_plan = (
        (workload, route, warmup)
        for workload in WORKLOADS
        for route in ROUTES
        for warmup in range(2)
    )
    for workload, route, warmup in warmup_plan:
        remaining = COLLECTION_CAP_SECONDS - (time.monotonic() - collection_start)
        if remaining <= 0:
            collection_cap_exceeded = True
            break
        warmups.append(
            collect_one(
                binary,
                root,
                output,
                workload,
                route,
                "warmup",
                warmup,
                warmup,
                min(INVOCATION_TIMEOUT_SECONDS, remaining),
            )
        )

    if not collection_cap_exceeded:
        measured_plan = (
            (workload, pair, position, route)
            for workload in WORKLOADS
            for pair in range(10)
            for position, route in enumerate(
                ROUTES if pair % 2 == 0 else tuple(reversed(ROUTES))
            )
        )
        for workload, pair, position, route in measured_plan:
            remaining = COLLECTION_CAP_SECONDS - (time.monotonic() - collection_start)
            if remaining <= 0:
                collection_cap_exceeded = True
                break
            measured.append(
                collect_one(
                    binary,
                    root,
                    output,
                    workload,
                    route,
                    "measured",
                    pair,
                    position,
                    min(INVOCATION_TIMEOUT_SECONDS, remaining),
                )
            )

    return warmups, measured, collection_cap_exceeded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/bench_svd_lowrank_memory"))
    parser.add_argument("--library", type=Path, default=Path("build/libsparse_lu_ortho.a"))
    parser.add_argument("--output", type=Path, default=Path("build/sprint228-day11-memory"))
    parser.add_argument("--build-metadata", type=Path)
    parser.add_argument("--allow-dirty-development", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = (root / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    library = (root / args.library).resolve() if not args.library.is_absolute() else args.library
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    build_metadata_path = args.build_metadata or Path(str(binary) + ".build.tsv")
    build_metadata_path = (
        (root / build_metadata_path).resolve()
        if not build_metadata_path.is_absolute()
        else build_metadata_path
    )
    git_head = run_text(["git", "rev-parse", "HEAD"], root)
    git_dirty = bool(run_text(["git", "status", "--porcelain", "--untracked-files=all"], root))
    if git_dirty and not args.allow_dirty_development:
        raise SystemExit("sprint228-memory: dirty worktree rejected (use development override only)")
    build_metadata = load_build_metadata(build_metadata_path)
    index_bits = int(build_metadata["index_bits"])
    source_sha, source_files = source_snapshot(root)
    binary_sha = sha256_file(binary)
    library_sha = sha256_file(library)
    prepare_output_directory(output)
    frozen_binary = output / "executed-binary"
    shutil.copy2(binary, frozen_binary)
    retained_build_metadata = output / "build-metadata.tsv"
    shutil.copy2(build_metadata_path, retained_build_metadata)
    metadata = {
        "schema": "s228-memory-packet-v1",
        "collector_command": ["python3", "scripts/run_sprint228_memory_packet.py"]
        + (["--allow-dirty-development"] if args.allow_dirty_development else []),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "development_override": args.allow_dirty_development,
        "source_snapshot_sha256": source_sha,
        "source_files": source_files,
        "binary": display_path(binary, root),
        "binary_sha256": binary_sha,
        "execution_binary": frozen_binary.name,
        "execution_binary_sha256": sha256_file(frozen_binary),
        "library": display_path(library, root),
        "library_sha256": library_sha,
        "build_metadata": retained_build_metadata.name,
        "build_metadata_sha256": sha256_file(retained_build_metadata),
        "build_system": build_metadata["build_system"],
        "cc": build_metadata["cc"],
        "compiler": build_metadata["compiler"],
        "cflags": build_metadata["cflags"],
        "ldflags": build_metadata["ldflags"],
        "flags": build_metadata["sparse_cflags"],
        "index_bits": index_bits,
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
    warmups, measured, collection_cap_exceeded = collect_samples(
        frozen_binary, root, output, collection_start
    )

    ending_sha, _ = source_snapshot(root)
    binary_unchanged = sha256_file(binary) == binary_sha
    library_unchanged = sha256_file(library) == library_sha
    execution_binary_unchanged = sha256_file(frozen_binary) == binary_sha
    validation = validate(warmups, measured, index_bits)
    if collection_cap_exceeded:
        validation["errors"] = [
            *validation["errors"],
            "15-minute collection cap exceeded before all samples completed",
        ]
        validation["memory_pass"] = False
        validation["timing_pass"] = False
    identity_errors = []
    if not binary_unchanged:
        identity_errors.append("selected binary changed during collection")
    if not library_unchanged:
        identity_errors.append("selected library changed during collection")
    if not execution_binary_unchanged:
        identity_errors.append("retained execution binary changed during collection")
    if identity_errors:
        validation["errors"] = [*validation["errors"], *identity_errors]
        validation["memory_pass"] = False
    summary = {
        "schema": "s228-memory-summary-v1",
        "source_snapshot_sha256": source_sha,
        "source_unchanged_during_collection": ending_sha == source_sha,
        "binary_unchanged_during_collection": binary_unchanged,
        "library_unchanged_during_collection": library_unchanged,
        "execution_binary_unchanged_during_collection": execution_binary_unchanged,
        "git_dirty": git_dirty,
        "development_override": args.allow_dirty_development,
        "collection_cap_exceeded": collection_cap_exceeded,
        "expected_warmup_count": 16,
        "expected_measured_count": 80,
        "warmup_count": len(warmups),
        "measured_count": len(measured),
        "collection_elapsed_seconds": time.monotonic() - collection_start,
        **validation,
        "protocol_accepted": (
            not git_dirty
            and ending_sha == source_sha
            and binary_unchanged
            and library_unchanged
            and execution_binary_unchanged
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
    return (
        0
        if validation["memory_pass"]
        and ending_sha == source_sha
        and binary_unchanged
        and library_unchanged
        and execution_binary_unchanged
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
