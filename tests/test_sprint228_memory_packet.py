#!/usr/bin/env python3
"""Regression tests for the Sprint 228 evidence collector."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_sprint228_memory_packet", ROOT / "scripts" / "run_sprint228_memory_packet.py"
)
assert SPEC and SPEC.loader
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def result(workload: str, route: str, *, index_bits: int = 32) -> dict[str, object]:
    scratch = 0 if workload == "measure-zero-512-r16" or route == "default" else 2_097_152
    return {
        "schema": "s228-memory-v1",
        "workload": workload,
        "route": route,
        "index_bits": index_bits,
        "correct": True,
        "checksum": "fixture",
        "nnz": 1,
        "bidiag_work_peak_bytes": 10,
        "retained_factor_bytes": 20,
        "reconstruction_scratch_peak_bytes": scratch,
        "reconstruction_scratch_allocations": 1 if scratch else 0,
        "sparse_shell_map_bytes": 30,
        "sparse_node_slab_bytes": 40,
        "wall_ns": 100,
    }


def receipt(
    workload: str,
    route: str,
    phase: str,
    pair: int,
    *,
    index_bits: int = 32,
) -> dict[str, object]:
    return {
        "phase": phase,
        "workload": workload,
        "route": route,
        "pair": pair,
        "timed_out": False,
        "returncode": 0,
        "result": result(workload, route, index_bits=index_bits),
    }


def valid_packet() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    warmups = [
        receipt(workload, route, "warmup", warmup)
        for workload in COLLECTOR.WORKLOADS
        for route in COLLECTOR.ROUTES
        for warmup in range(2)
    ]
    measured = [
        receipt(workload, route, "measured", pair)
        for workload in COLLECTOR.WORKLOADS
        for pair in range(10)
        for route in COLLECTOR.ROUTES
    ]
    return warmups, measured


class Sprint228CollectorTests(unittest.TestCase):
    def test_malformed_probe_output_is_retained_and_rejected(self) -> None:
        malformed_outputs = (b"\xff", b"{", b"{}\n{}\n", b"[]\n")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for index, stdout in enumerate(malformed_outputs):
                with self.subTest(stdout=stdout):
                    completed = subprocess.CompletedProcess(
                        args=["probe"], returncode=0, stdout=stdout, stderr=b"diagnostic"
                    )
                    with mock.patch.object(COLLECTOR.subprocess, "run", return_value=completed):
                        sample = COLLECTOR.collect_one(
                            Path("probe"),
                            ROOT,
                            output,
                            COLLECTOR.WORKLOADS[0],
                            COLLECTOR.ROUTES[0],
                            "malformed",
                            index,
                            0,
                            1.0,
                        )
                    self.assertIsNone(sample["result"])
                    self.assertTrue(sample["result_error"])
                    self.assertTrue(any(output.glob(f"malformed-*-{index:02d}-*.receipt.json")))
                    errors: list[str] = []
                    validation = COLLECTOR.validate_receipt(sample, errors, 32)
                    self.assertIsNone(validation)
                    self.assertEqual(len(errors), 1)
                    self.assertIn("failed malformed sample", errors[0])

    def test_make_metadata_tracks_archive_configuration_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            archive = build / "libsparse_lu_ortho.a"
            obj = build / "sparse_types.o"
            sidecar = build / "bench_svd_lowrank_memory.build.tsv"
            common = [
                "make",
                "--no-print-directory",
                "-C",
                str(ROOT),
                f"BUILDDIR={build}",
                f"LIB_OBJS={obj}",
            ]

            first = subprocess.run(
                [*common, "CFLAGS=-std=c11 -O0", str(archive)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertIn("-O0", first.stdout)
            stamp = build / ".sparse-build-config.tsv"
            self.assertIn("cflags\t-std=c11 -O0", stamp.read_text(encoding="utf-8"))

            second = subprocess.run(
                [*common, "CFLAGS=-std=c11 -O2", str(archive), str(sidecar)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertIn("-O2", second.stdout)
            self.assertIn("src/sparse_types.c", second.stdout)
            self.assertEqual(
                sidecar.read_bytes(),
                stamp.read_bytes(),
                "probe metadata must be the archive's shared build receipt",
            )

    def test_existing_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            output.mkdir()
            with self.assertRaisesRegex(SystemExit, "refusing overwrite"):
                COLLECTOR.prepare_output_directory(output)

    def test_new_output_directory_is_created_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            COLLECTOR.prepare_output_directory(output)
            self.assertTrue(output.is_dir())
            with self.assertRaises(SystemExit):
                COLLECTOR.prepare_output_directory(output)

    def test_build_metadata_is_exact_and_artifact_bound(self) -> None:
        contents = (
            "schema\ts228-memory-build-v1\n"
            "build_system\tmake\n"
            "cc\tcc\n"
            "compiler\tExample C 1.0\n"
            "cflags\t-O2\n"
            "ldflags\t-lm\n"
            "sparse_cflags\t-O2 -DSPARSE_IDX_BITS=64\n"
            "index_bits\t64\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.build.tsv"
            path.write_text(contents, encoding="utf-8")
            metadata = COLLECTOR.load_build_metadata(path)
            self.assertEqual(metadata["compiler"], "Example C 1.0")
            self.assertEqual(metadata["index_bits"], "64")

    def test_build_metadata_rejects_unverified_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.build.tsv"
            path.write_text(
                "schema\ts228-memory-build-v1\n"
                "build_system\tmake\n"
                "cc\tcc\n"
                "compiler\tExample C 1.0\n"
                "cflags\t-O2\n"
                "ldflags\t-lm\n"
                "sparse_cflags\t-O2\n"
                "index_bits\tunknown\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "unverifiable"):
                COLLECTOR.load_build_metadata(path)

    def test_valid_warmups_and_measurements_pass(self) -> None:
        warmups, measured = valid_packet()
        validation = COLLECTOR.validate(warmups, measured, 32)
        self.assertTrue(validation["memory_pass"])
        self.assertTrue(validation["timing_pass"])
        self.assertEqual(validation["errors"], [])

    def test_failed_warmup_invalidates_packet_without_entering_ratios(self) -> None:
        warmups, measured = valid_packet()
        warmups[0]["timed_out"] = True
        warmups[0]["result"] = {}
        validation = COLLECTOR.validate(warmups, measured, 32)
        self.assertFalse(validation["memory_pass"])
        self.assertTrue(validation["timing_pass"])
        self.assertIn("failed warmup sample", validation["errors"][0])

    def test_nonpositive_timing_from_either_route_is_rejected(self) -> None:
        for route in COLLECTOR.ROUTES:
            with self.subTest(route=route):
                warmups, measured = valid_packet()
                sample = next(item for item in measured if item["route"] == route)
                sample["result"]["wall_ns"] = 0
                validation = COLLECTOR.validate(warmups, measured, 32)
                self.assertFalse(validation["memory_pass"])
                self.assertTrue(
                    any(f"invalid {route} timing" in error for error in validation["errors"])
                )

    def test_overflowing_timing_is_rejected_without_raising(self) -> None:
        warmups, measured = valid_packet()
        measured[0]["result"]["wall_ns"] = float("inf")
        validation = COLLECTOR.validate(warmups, measured, 32)
        self.assertFalse(validation["memory_pass"])
        self.assertTrue(any("invalid timing" in error for error in validation["errors"]))

    def test_missing_or_null_route_accounting_is_rejected(self) -> None:
        fields = (
            "checksum",
            "nnz",
            "bidiag_work_peak_bytes",
            "retained_factor_bytes",
            "sparse_shell_map_bytes",
            "sparse_node_slab_bytes",
        )
        for mode in ("missing", "null"):
            with self.subTest(mode=mode):
                warmups, measured = valid_packet()
                samples = [
                    item
                    for item in measured
                    if item["workload"] == COLLECTOR.WORKLOADS[0] and item["pair"] == 0
                ]
                for sample in samples:
                    for field in fields:
                        if mode == "missing":
                            del sample["result"][field]
                        else:
                            sample["result"][field] = None

                validation = COLLECTOR.validate(warmups, measured, 32)
                self.assertFalse(validation["memory_pass"])
                for route in COLLECTOR.ROUTES:
                    for field in fields:
                        self.assertIn(
                            f"missing {route} accounting: {COLLECTOR.WORKLOADS[0]} "
                            f"pair 0 field {field}",
                            validation["errors"],
                        )

    def test_result_width_must_match_build_metadata(self) -> None:
        warmups, measured = valid_packet()
        measured[0]["result"]["index_bits"] = 64
        validation = COLLECTOR.validate(warmups, measured, 32)
        self.assertFalse(validation["memory_pass"])
        self.assertTrue(any("index-width mismatch" in error for error in validation["errors"]))

    def test_collection_cap_finalizes_failed_packet_with_actual_counts(self) -> None:
        metadata = (
            "schema\ts228-memory-build-v1\n"
            "build_system\tmake\n"
            "cc\tcc\n"
            "compiler\tExample C 1.0\n"
            "cflags\t-O2\n"
            "ldflags\t-lm\n"
            "sparse_cflags\t-O2 -DSPARSE_IDX_BITS=32\n"
            "index_bits\t32\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            binary = temporary / "probe"
            library = temporary / "library.a"
            build_metadata = temporary / "probe.build.tsv"
            output = temporary / "packet"
            binary.write_bytes(b"probe")
            library.write_bytes(b"library")
            build_metadata.write_text(metadata, encoding="utf-8")
            arguments = [
                "run_sprint228_memory_packet.py",
                "--binary",
                str(binary),
                "--library",
                str(library),
                "--build-metadata",
                str(build_metadata),
                "--output",
                str(output),
                "--allow-dirty-development",
            ]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                COLLECTOR, "collect_samples", return_value=([], [], True)
            ):
                self.assertEqual(COLLECTOR.main(), 1)

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["collection_cap_exceeded"])
            self.assertEqual(summary["warmup_count"], 0)
            self.assertEqual(summary["measured_count"], 0)
            self.assertFalse(summary["memory_pass"])
            self.assertFalse(summary["timing_pass"])
            self.assertFalse(summary["protocol_accepted"])
            self.assertTrue(
                any("15-minute collection cap exceeded" in error for error in summary["errors"])
            )
            manifest = (output / "manifest.sha256").read_text(encoding="utf-8")
            self.assertIn("  summary.json\n", manifest)


if __name__ == "__main__":
    unittest.main()
