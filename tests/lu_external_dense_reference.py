#!/usr/bin/env python3
"""Dense reference solves for bounded external LU differential tests."""

from __future__ import annotations

import sys
from typing import List, Tuple


def build_lu_nonsym_square_5() -> List[List[float]]:
    return [
        [4.0, -1.0, 0.0, 2.0, 0.5],
        [1.5, 5.0, -2.0, 0.0, 1.0],
        [0.0, 2.0, 6.0, -1.0, 0.0],
        [3.0, 0.0, 1.0, 7.0, -2.0],
        [-1.0, 0.5, 0.0, 2.0, 8.0],
    ]


def build_lu_singular_square_4() -> List[List[float]]:
    return [
        [1.0, 2.0, -1.0, 0.0],
        [2.0, 4.0, -2.0, 0.0],
        [0.0, 1.0, 3.0, 1.0],
        [1.0, 0.0, 0.5, -1.0],
    ]


def fixture_matrix(name: str) -> List[List[float]]:
    if name == "lu_nonsym_square_5":
        return build_lu_nonsym_square_5()
    if name == "lu_singular_square_4":
        return build_lu_singular_square_4()
    raise ValueError(f"unknown fixture {name}")


def matvec(a: List[List[float]], x: List[float]) -> List[float]:
    return [sum(value * x[j] for j, value in enumerate(row)) for row in a]


def build_rhs(a: List[List[float]]) -> Tuple[List[float], List[float]]:
    x_true = [float(i + 1) for i in range(len(a))]
    return x_true, matvec(a, x_true)


def dense_solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    work = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        pivot_value = work[pivot][col]
        if abs(pivot_value) < 1e-14:
            raise ValueError("matrix is singular to dense reference tolerance")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]

        for row in range(col + 1, n):
            factor = work[row][col] / work[col][col]
            if factor == 0.0:
                continue
            work[row][col] = 0.0
            for k in range(col + 1, n + 1):
                work[row][k] -= factor * work[col][k]

    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        rhs = work[i][n]
        for j in range(i + 1, n):
            rhs -= work[i][j] * x[j]
        x[i] = rhs / work[i][i]
    return x


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("ERROR expected one fixture key")
        return 1

    try:
        dense = fixture_matrix(argv[1])
        _, b = build_rhs(dense)
        x = dense_solve(dense, b)
    except Exception as exc:  # pragma: no cover - exercised from C harness
        print(f"ERROR {exc}")
        return 1

    print(f"OK {len(x)}")
    for value in x:
        print(f"{value:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
