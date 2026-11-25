#!/usr/bin/env python3
"""Validate Sprint 227 raw add samples and write a per-process CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


FIXTURES = {
    "wide-row": ("s227-add-wide-r1-c4096-v1", 1, 4096, 2048, 2048, 4096, 4096),
    "varied-rows": ("s227-add-varied-r64-c1024-v1", 64, 1024, 4608, 4608, 9216, 256),
    "narrow-rows": ("s227-add-narrow-r1024-c64-v1", 1024, 64, 4096, 4096, 8192, 8),
}
VARIANTS = ("baseline", "candidate")
EXTENSIONS = ("stdout", "stderr", "time", "status", "meta")
BENCH_COLUMNS = (
    "fixture",
    "workload_id",
    "rows",
    "cols",
    "a_nnz",
    "b_nnz",
    "c_nnz",
    "max_output_row_width",
    "elapsed_ms",
)
OUTPUT_COLUMNS = (
    "fixture",
    "workload_id",
    "round",
    "variant",
    "order_in_pair",
    "elapsed_ms",
    "rows",
    "cols",
    "a_nnz",
    "b_nnz",
    "c_nnz",
    "max_output_row_width",
    "max_rss_bytes",
    "peak_footprint_bytes",
    "exit_status",
)


def time_field(timing: str, label: str, sample: Path) -> int:
    matches = re.findall(rf"^\s*(\d+)\s+{re.escape(label)}\s*$", timing, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{sample}: expected exactly one {label!r} line")
    value = int(matches[0])
    if value <= 0:
        raise ValueError(f"{sample}: nonpositive {label}")
    return value


def read_sample(raw: Path, fixture: str, round_number: int, variant: str) -> dict[str, object]:
    sample = raw / f"{fixture}-{round_number}-{variant}"
    status = sample.with_suffix(".status").read_text(encoding="utf-8").strip()
    if status != "0":
        raise ValueError(f"{sample}: exit status {status!r}")
    if sample.with_suffix(".stderr").read_bytes():
        raise ValueError(f"{sample}: benchmark stderr is not empty")

    with sample.with_suffix(".stdout").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != BENCH_COLUMNS:
            raise ValueError(f"{sample}: unexpected benchmark columns")
        rows = list(reader)
    if len(rows) != 1 or rows[0]["fixture"] != fixture:
        raise ValueError(f"{sample}: expected one matching fixture row")
    bench = rows[0]
    expected = FIXTURES[fixture]
    actual = (bench["workload_id"],) + tuple(int(bench[name]) for name in BENCH_COLUMNS[2:8])
    if actual != expected:
        raise ValueError(f"{sample}: workload identity or counts changed: {actual!r}")
    elapsed_ms = float(bench["elapsed_ms"])
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
        raise ValueError(f"{sample}: invalid elapsed time")

    metadata = json.loads(sample.with_suffix(".meta").read_text(encoding="utf-8"))
    expected_order = 1 if (round_number % 2 == 1) == (variant == "baseline") else 2
    if (
        metadata.get("fixture") != fixture
        or metadata.get("round") != round_number
        or metadata.get("variant") != variant
        or metadata.get("order_in_pair") != expected_order
        or metadata.get("per_command_cap_seconds") != 30.0
        or metadata.get("collection_cap_seconds") != 900.0
    ):
        raise ValueError(f"{sample}: metadata does not match frozen protocol")
    argv = metadata.get("argv")
    if not isinstance(argv, list) or len(argv) != 6 or argv[:3] != ["/usr/bin/time", "-l", "-o"]:
        raise ValueError(f"{sample}: unexpected invocation argv")
    if argv[-1] != fixture or Path(argv[-2]).name != f"bench_add_{variant}":
        raise ValueError(f"{sample}: invocation executable or fixture mismatch")

    timing = sample.with_suffix(".time").read_text(encoding="utf-8")
    return {
        "fixture": fixture,
        "workload_id": bench["workload_id"],
        "round": round_number,
        "variant": variant,
        "order_in_pair": expected_order,
        "elapsed_ms": bench["elapsed_ms"],
        **{name: bench[name] for name in BENCH_COLUMNS[2:8]},
        "max_rss_bytes": time_field(timing, "maximum resident set size", sample),
        "peak_footprint_bytes": time_field(timing, "peak memory footprint", sample),
        "exit_status": status,
    }


def describe(values: list[float]) -> str:
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (
        f"median={statistics.median(values):.6f}, min={min(values):.6f}, "
        f"max={max(values):.6f}, IQR={q3 - q1:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    expected_files = {
        f"{fixture}-{round_number}-{variant}.{extension}"
        for fixture in FIXTURES
        for round_number in range(1, 11)
        for variant in VARIANTS
        for extension in EXTENSIONS
    }
    actual_files = {path.name for path in args.raw.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            f"raw bundle mismatch: missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    samples = [
        read_sample(args.raw, fixture, round_number, variant)
        for fixture in FIXTURES
        for round_number in range(1, 11)
        for variant in VARIANTS
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)

    print(f"validated {len(samples)} successful samples from {len(actual_files)} raw files")
    for fixture in FIXTURES:
        selected = {
            variant: [
                sample
                for sample in samples
                if sample["fixture"] == fixture and sample["variant"] == variant
            ]
            for variant in VARIANTS
        }
        for variant in VARIANTS:
            elapsed = [float(sample["elapsed_ms"]) for sample in selected[variant]]
            rss = [float(sample["max_rss_bytes"]) for sample in selected[variant]]
            footprint = [float(sample["peak_footprint_bytes"]) for sample in selected[variant]]
            print(f"{fixture} {variant} ms: {describe(elapsed)}")
            print(f"{fixture} {variant} RSS bytes: {describe(rss)}")
            print(f"{fixture} {variant} footprint bytes: {describe(footprint)}")
        baseline_ms = [float(sample["elapsed_ms"]) for sample in selected["baseline"]]
        candidate_ms = [float(sample["elapsed_ms"]) for sample in selected["candidate"]]
        paired_ratios = [before / after for before, after in zip(baseline_ms, candidate_ms)]
        print(
            f"{fixture} median-of-medians ratio baseline/candidate: "
            f"{statistics.median(baseline_ms) / statistics.median(candidate_ms):.3f}; "
            f"median paired ratio: {statistics.median(paired_ratios):.3f}; "
            f"paired ratios: {','.join(f'{value:.6f}' for value in paired_ratios)}"
        )


if __name__ == "__main__":
    main()
