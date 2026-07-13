#!/usr/bin/env python3
"""Dense singular-value references for bounded external SVD tests."""

from __future__ import annotations

import math
import sys
from typing import List


def build_svd_rect_fullrank_6x4() -> List[List[float]]:
    return [
        [3.0, -1.0, 0.0, 2.0],
        [0.0, 4.0, 1.0, -1.0],
        [2.0, 0.0, 3.0, 0.5],
        [5.0, 3.0, 4.0, 1.5],
        [-1.0, 5.0, 4.0, -0.5],
        [3.0, 4.0, 7.0, 2.5],
    ]


def fixture_matrix(name: str) -> List[List[float]]:
    if name == "svd_rect_fullrank_6x4":
        return build_svd_rect_fullrank_6x4()
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


def jacobi_symmetric_eigenvalues(a: List[List[float]]) -> List[float]:
    n = len(a)
    work = [row[:] for row in a]
    max_sweeps = 100
    tol = 1e-14

    for _ in range(max_sweeps):
        p = 0
        q = 1
        max_offdiag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                offdiag = abs(work[i][j])
                if offdiag > max_offdiag:
                    max_offdiag = offdiag
                    p = i
                    q = j
        if max_offdiag < tol:
            break

        app = work[p][p]
        aqq = work[q][q]
        apq = work[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t_sign = 1.0 if tau >= 0.0 else -1.0
        t = t_sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(n):
            if k == p or k == q:
                continue
            aik = work[k][p]
            akq = work[k][q]
            work[k][p] = c * aik - s * akq
            work[p][k] = work[k][p]
            work[k][q] = s * aik + c * akq
            work[q][k] = work[k][q]

        work[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        work[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        work[p][q] = 0.0
        work[q][p] = 0.0
    else:
        raise ValueError("Jacobi reference did not converge")

    return [work[i][i] for i in range(n)]


def singular_values(a: List[List[float]]) -> List[float]:
    evals = jacobi_symmetric_eigenvalues(gram_ata(a))
    sigmas = []
    for value in evals:
        if value < 0.0 and abs(value) < 1e-10:
            value = 0.0
        if value < 0.0:
            raise ValueError("negative eigenvalue from A^T A")
        sigmas.append(math.sqrt(value))
    sigmas.sort(reverse=True)
    return sigmas


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("ERROR expected one fixture key")
        return 1

    try:
        sigma = singular_values(fixture_matrix(argv[1]))
    except Exception as exc:  # pragma: no cover - exercised from C harness
        print(f"ERROR {exc}")
        return 1

    print(f"OK {len(sigma)}")
    for value in sigma:
        print(f"{value:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
