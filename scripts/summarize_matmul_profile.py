#!/usr/bin/env python3
"""Validate Sprint 218 raw matmul samples and write a per-process CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


FIXTURES = {
    "banded": (256, 256, 256, 256, 766, 766),
    "wide-row": (1, 1024, 1024, 1024, 1024, 1024),
    "rectangular": (64, 32, 96, 2048, 3072, 6144),
}
VARIANTS = ("baseline", "candidate")
BENCH_COLUMNS = (
    "fixture",
    "a_rows",
    "a_cols",
    "b_cols",
    "a_nnz",
    "b_nnz",
    "c_nnz",
    "zero_drops",
    "elapsed_ms",
)
OUTPUT_COLUMNS = (
    "fixture",
    "round",
    "variant",
    "order_in_pair",
    "elapsed_ms",
    "a_rows",
    "a_cols",
    "b_cols",
    "a_nnz",
    "b_nnz",
    "c_nnz",
    "zero_drops",
    "max_rss_bytes",
    "peak_footprint_bytes",
    "exit_status",
)


def time_field(stderr: str, label: str, sample: Path) -> int:
    matches = re.findall(rf"^\s*(\d+)\s+{re.escape(label)}\s*$", stderr, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{sample}: expected exactly one {label!r} line")
    value = int(matches[0])
    if value <= 0:
        raise ValueError(f"{sample}: nonpositive {label}")
    return value


def read_sample(raw: Path, fixture: str, round_number: int, variant: str) -> dict[str, object]:
    sample = raw / f"{fixture}-{round_number}-{variant}"
    status = sample.with_suffix(".status").read_text().strip()
    if status != "0":
        raise ValueError(f"{sample}: exit status {status!r}")

    with sample.with_suffix(".stdout").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != BENCH_COLUMNS:
            raise ValueError(f"{sample}: unexpected benchmark columns")
        rows = list(reader)
    if len(rows) != 1 or rows[0]["fixture"] != fixture:
        raise ValueError(f"{sample}: expected one matching fixture row")
    bench = rows[0]
    dimensions = tuple(int(bench[name]) for name in BENCH_COLUMNS[1:7])
    if dimensions != FIXTURES[fixture] or int(bench["zero_drops"]) != 0:
        raise ValueError(f"{sample}: fixture dimensions or counts changed")
    elapsed_ms = float(bench["elapsed_ms"])
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError(f"{sample}: invalid elapsed time")

    stderr = sample.with_suffix(".stderr").read_text()
    order = 1 if (round_number % 2 == 1) == (variant == "baseline") else 2
    return {
        "fixture": fixture,
        "round": round_number,
        "variant": variant,
        "order_in_pair": order,
        "elapsed_ms": bench["elapsed_ms"],
        **{name: bench[name] for name in BENCH_COLUMNS[1:8]},
        "max_rss_bytes": time_field(stderr, "maximum resident set size", sample),
        "peak_footprint_bytes": time_field(stderr, "peak memory footprint", sample),
        "exit_status": status,
    }


def describe(values: list[float]) -> str:
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (
        f"median={statistics.median(values):.6f}, "
        f"min={min(values):.6f}, max={max(values):.6f}, "
        f"IQR={q3 - q1:.6f}"
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
        for extension in ("stdout", "stderr", "status")
    }
    actual_files = {path.name for path in args.raw.iterdir() if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"raw bundle mismatch: missing={missing}, extra={extra}")

    samples = [
        read_sample(args.raw, fixture, round_number, variant)
        for fixture in FIXTURES
        for round_number in range(1, 11)
        for variant in VARIANTS
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)

    print(f"validated {len(samples)} successful samples from {len(actual_files)} raw files")
    for fixture in FIXTURES:
        by_variant = {
            variant: [
                sample
                for sample in samples
                if sample["fixture"] == fixture and sample["variant"] == variant
            ]
            for variant in VARIANTS
        }
        for variant in VARIANTS:
            selected = by_variant[variant]
            elapsed = [float(sample["elapsed_ms"]) for sample in selected]
            rss = [float(sample["max_rss_bytes"]) for sample in selected]
            footprint = [float(sample["peak_footprint_bytes"]) for sample in selected]
            print(f"{fixture} {variant} ms: {describe(elapsed)}")
            print(f"{fixture} {variant} RSS bytes: {describe(rss)}")
            print(f"{fixture} {variant} footprint bytes: {describe(footprint)}")
        baseline = by_variant["baseline"]
        candidate = by_variant["candidate"]
        baseline_ms = [float(sample["elapsed_ms"]) for sample in baseline]
        candidate_ms = [float(sample["elapsed_ms"]) for sample in candidate]
        paired_delta = [
            float(before["elapsed_ms"]) - float(after["elapsed_ms"])
            for before, after in zip(baseline, candidate)
        ]
        print(
            f"{fixture} median ratio baseline/candidate: "
            f"{statistics.median(baseline_ms) / statistics.median(candidate_ms):.3f}; "
            f"median paired delta ms: {statistics.median(paired_delta):.6f}"
        )


if __name__ == "__main__":
    main()
