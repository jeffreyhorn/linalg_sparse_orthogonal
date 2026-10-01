#!/usr/bin/env python3
"""Focused behavior regression for extracted LDLT CSC native parity tests."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SELECTED_TESTS = [
    "test_native_1x1_diagonal_matches_wrapper",
    "test_native_1x1_tridiagonal_matches_wrapper",
    "test_native_1x1_mixed_indefinite_matches_wrapper",
    "test_native_1x1_with_swap_matches_wrapper",
    "test_native_1x1_tridiag_large_matches_wrapper",
    "test_native_detects_near_zero_1x1_pivot",
    "test_native_1x1_identity_matches_wrapper",
    "test_native_2x2_forced_matches_wrapper",
    "test_native_2x2_nonadjacent_partner_matches_wrapper",
    "test_native_mixed_pivots_matches_wrapper",
    "test_native_mixed_pivots_larger_matches_wrapper",
    "test_native_2x2_solve_matches_linked_list",
    "test_native_2x2_inertia_matches_wrapper",
]

BASELINE_SUMMARY = {
    "Tests run": 100,
    "Tests failed": 0,
    "Tests skipped": 0,
    "Assertions": 3556,
}


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require_success(result: subprocess.CompletedProcess[str], command: str) -> None:
    if result.returncode != 0:
        raise AssertionError(f"{command} failed\n{result.stdout}{result.stderr}")


def output_excerpt(output: str, limit: int = 4000) -> str:
    if len(output) <= limit:
        return output
    return output[:limit] + "\n... output truncated ...\n" + output[-limit:]


def parse_summary(output: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for key in BASELINE_SUMMARY:
        match = re.search(rf"^{re.escape(key)}:\s+(\d+)$", output, re.MULTILINE)
        if not match:
            raise AssertionError(
                f"missing summary field {key!r}\n{output_excerpt(output)}"
            )
        summary[key] = int(match.group(1))
    return summary


def assert_selected_tests_passed_in_order(output: str) -> None:
    previous = -1
    for test_name in SELECTED_TESTS:
        marker = f"[PASS] {test_name}"
        index = output.find(marker)
        if index == -1:
            raise AssertionError(
                "missing selected native parity pass marker: "
                f"{marker}\n{output_excerpt(output)}"
            )
        if index <= previous:
            raise AssertionError(
                "selected native parity pass marker out of order: "
                f"{marker}\n{output_excerpt(output)}"
            )
        previous = index


def main() -> None:
    build = run_command(["make", "build/test_ldlt_csc"])
    require_success(build, "make build/test_ldlt_csc")

    result = run_command(["./build/test_ldlt_csc"])
    require_success(result, "./build/test_ldlt_csc")
    output = result.stdout + result.stderr

    assert_selected_tests_passed_in_order(output)
    summary = parse_summary(output)
    if summary != BASELINE_SUMMARY:
        raise AssertionError(
            f"unexpected test_ldlt_csc summary: {summary!r}\n{output_excerpt(output)}"
        )
    if "ALL TESTS PASSED" not in output:
        raise AssertionError(f"missing ALL TESTS PASSED marker\n{output_excerpt(output)}")


if __name__ == "__main__":
    main()
