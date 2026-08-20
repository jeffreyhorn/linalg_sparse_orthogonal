#!/usr/bin/env python3
"""Focused tests for the bounded LU external dense-reference helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lu_external_dense_reference import (
    build_lu_nonsym_square_5,
    build_rhs,
    dense_solve,
    fixture_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "tests" / "lu_external_dense_reference.py"

EXPECTED_MATRIX = [
    [4.0, -1.0, 0.0, 2.0, 0.5],
    [1.5, 5.0, -2.0, 0.0, 1.0],
    [0.0, 2.0, 6.0, -1.0, 0.0],
    [3.0, 0.0, 1.0, 7.0, -2.0],
    [-1.0, 0.5, 0.0, 2.0, 8.0],
]
EXPECTED_SOLUTION = [1.0, 2.0, 3.0, 4.0, 5.0]
EXPECTED_RHS = [12.5, 10.5, 18.0, 24.0, 48.0]


def assert_vector_close(actual: list[float], expected: list[float], tol: float) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: {len(actual)} != {len(expected)}")
    for index, (lhs, rhs) in enumerate(zip(actual, expected)):
        if abs(lhs - rhs) > tol:
            raise AssertionError(f"value {index} mismatch: {lhs} != {rhs} within {tol}")


def test_lu_nonsym_square_5_fixture_matrix_and_rhs() -> None:
    matrix = build_lu_nonsym_square_5()
    assert matrix == EXPECTED_MATRIX
    assert fixture_matrix("lu_nonsym_square_5") == EXPECTED_MATRIX

    x_true, rhs = build_rhs(matrix)
    assert x_true == EXPECTED_SOLUTION
    assert rhs == EXPECTED_RHS


def test_lu_nonsym_square_5_dense_solution_matches_fixture_contract() -> None:
    matrix = build_lu_nonsym_square_5()
    _, rhs = build_rhs(matrix)
    solution = dense_solve(matrix, rhs)
    assert_vector_close(solution, EXPECTED_SOLUTION, 1e-12)


def test_lu_nonsym_square_5_cli_contract() -> None:
    result = subprocess.run(
        [sys.executable, str(HELPER), "lu_nonsym_square_5"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines[0] == "OK 5"
    values = [float(line) for line in lines[1:]]
    assert_vector_close(values, EXPECTED_SOLUTION, 1e-12)


def test_unknown_fixture_fails_without_pass_evidence() -> None:
    result = subprocess.run(
        [sys.executable, str(HELPER), "not_a_fixture"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    assert "ERROR unknown fixture not_a_fixture" in result.stdout


def main() -> int:
    test_lu_nonsym_square_5_fixture_matrix_and_rhs()
    test_lu_nonsym_square_5_dense_solution_matches_fixture_contract()
    test_lu_nonsym_square_5_cli_contract()
    test_unknown_fixture_fails_without_pass_evidence()
    print("test-lu-external-dense-reference: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
