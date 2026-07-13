#!/usr/bin/env python3
"""Dense least-squares references for bounded external QR tests."""

from __future__ import annotations

import math
import sys
from typing import List, Tuple


def build_qr_overdetermined_incompatible_4x2() -> Tuple[List[List[float]], List[float]]:
    a = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
    ]
    x_exact = [2.0, -1.0]
    orthogonal_residual = [-1.0, -1.0, 1.0, 0.0]
    b = []
    for row, residual in zip(a, orthogonal_residual):
        b.append(sum(row[j] * x_exact[j] for j in range(2)) + residual)
    return a, b


def fixture_system(name: str) -> Tuple[List[List[float]], List[float]]:
    if name == "qr_overdetermined_incompatible_4x2":
        return build_qr_overdetermined_incompatible_4x2()
    raise ValueError(f"unknown fixture {name}")


def gram_ata(a: List[List[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(a[0])
    ata = [[0.0 for _ in range(cols)] for _ in range(cols)]
    for i in range(cols):
        for j in range(i, cols):
            value = sum(a[r][i] * a[r][j] for r in range(rows))
            ata[i][j] = value
            ata[j][i] = value
    return ata


def gram_atb(a: List[List[float]], b: List[float]) -> List[float]:
    rows = len(a)
    cols = len(a[0])
    return [sum(a[r][j] * b[r] for r in range(rows)) for j in range(cols)]


def solve_2x2(a: List[List[float]], b: List[float]) -> List[float]:
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) < 1e-14:
        raise ValueError("singular normal-equation reference system")
    return [
        (b[0] * a[1][1] - a[0][1] * b[1]) / det,
        (a[0][0] * b[1] - b[0] * a[1][0]) / det,
    ]


def residual_norm(a: List[List[float]], b: List[float], x: List[float]) -> float:
    accum = 0.0
    for row, target in zip(a, b):
        diff = sum(row[j] * x[j] for j in range(len(x))) - target
        accum += diff * diff
    return math.sqrt(accum)


def least_squares_reference(name: str) -> List[float]:
    a, b = fixture_system(name)
    x = solve_2x2(gram_ata(a), gram_atb(a, b))
    return [x[0], x[1], residual_norm(a, b, x)]


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("ERROR expected one fixture key")
        return 1

    try:
        values = least_squares_reference(argv[1])
    except Exception as exc:  # pragma: no cover - exercised from C harness
        print(f"ERROR {exc}")
        return 1

    print(f"OK {len(values)}")
    for value in values:
        print(f"{value:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
