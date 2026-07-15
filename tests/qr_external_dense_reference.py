#!/usr/bin/env python3
"""Dense least-squares references for bounded external QR tests."""

from __future__ import annotations

from fractions import Fraction
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


def build_qr_overdetermined_compatible_5x3() -> Tuple[List[List[float]], List[float]]:
    a = [
        [1.0, 0.0, 2.0],
        [0.0, 1.0, -1.0],
        [2.0, -1.0, 0.0],
        [1.0, 1.0, 1.0],
        [3.0, 0.0, -2.0],
    ]
    x_exact = [1.0, -2.0, 0.5]
    b = []
    for row in a:
        b.append(sum(row[j] * x_exact[j] for j in range(3)))
    return a, b


def build_qr_rankdef_duplicate_5x4_rank_only() -> List[List[float]]:
    return [
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, -1.0, 1.0],
        [2.0, -1.0, 0.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [3.0, 0.0, -2.0, 0.0],
    ]


def build_qr_underdetermined_minnorm_2x4() -> Tuple[List[List[float]], List[float]]:
    a = [
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    b = [1.0, 1.0]
    return a, b


def fixture_system(name: str) -> Tuple[List[List[float]], List[float]]:
    if name == "qr_overdetermined_incompatible_4x2":
        return build_qr_overdetermined_incompatible_4x2()
    if name == "qr_overdetermined_compatible_5x3":
        return build_qr_overdetermined_compatible_5x3()
    if name == "qr_underdetermined_minnorm_2x4":
        return build_qr_underdetermined_minnorm_2x4()
    raise ValueError(f"unknown fixture {name}")


def rank_fixture_matrix(name: str) -> List[List[float]]:
    if name == "qr_rankdef_duplicate_5x4_rank_only":
        return build_qr_rankdef_duplicate_5x4_rank_only()
    raise ValueError(f"unknown rank fixture {name}")


def matrix_rank(a: List[List[float]]) -> int:
    work = [[Fraction(value).limit_denominator() for value in row] for row in a]
    rows = len(work)
    cols = len(work[0])
    rank = 0

    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if work[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            work[rank], work[pivot] = work[pivot], work[rank]

        pivot_value = work[rank][col]
        for j in range(col, cols):
            work[rank][j] /= pivot_value

        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][col]
            for j in range(col, cols):
                work[row][j] -= factor * work[rank][j]
        rank += 1
        if rank == rows:
            break

    return rank


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


def solve_dense(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    work = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) < 1e-14:
            raise ValueError("singular normal-equation reference system")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]

        pivot_value = work[col][col]
        for j in range(col, n + 1):
            work[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = work[row][col]
            for j in range(col, n + 1):
                work[row][j] -= factor * work[col][j]

    return [work[i][n] for i in range(n)]


def residual_norm(a: List[List[float]], b: List[float], x: List[float]) -> float:
    accum = 0.0
    for row, target in zip(a, b):
        diff = sum(row[j] * x[j] for j in range(len(x))) - target
        accum += diff * diff
    return math.sqrt(accum)


def least_squares_reference(name: str) -> List[float]:
    a, b = fixture_system(name)
    x = solve_dense(gram_ata(a), gram_atb(a, b))
    return x + [residual_norm(a, b, x)]


def rank_reference(name: str) -> List[float]:
    a = rank_fixture_matrix(name)
    return [float(matrix_rank(a))]


def minnorm_reference(name: str) -> List[float]:
    a, b = fixture_system(name)
    aat = [[sum(row_i[k] * row_j[k] for k in range(len(row_i))) for row_j in a] for row_i in a]
    y = solve_dense(aat, b)
    x = [sum(a[row][col] * y[row] for row in range(len(a))) for col in range(len(a[0]))]
    return x + [residual_norm(a, b, x), math.sqrt(sum(value * value for value in x))]


def economy_projector_reference(name: str) -> List[float]:
    if name != "qr_economy_projector_5x3":
        raise ValueError(f"unknown economy projector fixture {name}")

    a, _ = build_qr_overdetermined_compatible_5x3()
    rows = len(a)
    cols = len(a[0])
    ata = gram_ata(a)
    projector: List[float] = []
    for row_j in range(rows):
        weights = solve_dense(ata, a[row_j])
        for row_i in range(rows):
            projector.append(sum(a[row_i][col] * weights[col] for col in range(cols)))

    return [float(rows), float(cols), float(cols), float(cols)] + projector


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("ERROR expected one fixture key")
        return 1

    try:
        if argv[1] == "qr_rankdef_duplicate_5x4_rank_only":
            values = rank_reference(argv[1])
        elif argv[1] == "qr_underdetermined_minnorm_2x4":
            values = minnorm_reference(argv[1])
        elif argv[1] == "qr_economy_projector_5x3":
            values = economy_projector_reference(argv[1])
        else:
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
