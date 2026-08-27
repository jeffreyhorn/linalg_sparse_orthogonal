#!/usr/bin/env python3
"""Dense SPD reference solve for bounded external Cholesky differential tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple


def mm_load_dense(path: str) -> List[List[float]]:
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        if len(header) != 5 or header[0] != "%%MatrixMarket" or header[1] != "matrix":
            raise ValueError("invalid Matrix Market header")
        if header[2] != "coordinate":
            raise ValueError("only coordinate matrices supported")
        field = header[3].lower()
        symmetry = header[4].lower()
        if field not in ("real", "integer"):
            raise ValueError(f"unsupported field {field}")

        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        nrows, ncols, nnz = (int(part) for part in line.split())
        if nrows != ncols:
            raise ValueError("matrix must be square")

        dense = [[0.0 for _ in range(ncols)] for _ in range(nrows)]
        for _ in range(nnz):
            line = fh.readline()
            while line.startswith("%"):
                line = fh.readline()
            i_s, j_s, v_s = line.split()
            i = int(i_s) - 1
            j = int(j_s) - 1
            v = float(v_s)
            dense[i][j] = v
            if symmetry == "symmetric" and i != j:
                dense[j][i] = v
            elif symmetry == "skew-symmetric" and i != j:
                dense[j][i] = -v
            elif symmetry == "hermitian":
                raise ValueError("hermitian matrices are unsupported in this helper")

        return dense


def build_cholesky_spd_tridiag_5() -> List[List[float]]:
    return [
        [4.0, -1.0, 0.0, 0.0, 0.0],
        [-1.0, 4.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 4.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, 4.0, -1.0],
        [0.0, 0.0, 0.0, -1.0, 4.0],
    ]


def fixture_matrix(name: str) -> List[List[float]]:
    if name == "cholesky_spd_tridiag_5":
        return build_cholesky_spd_tridiag_5()
    path = Path(name)
    if path.suffix == ".mtx" or path.parent != Path("."):
        return mm_load_dense(name)
    raise ValueError(f"unknown fixture {name}")


def dense_cholesky(a: List[List[float]]) -> List[List[float]]:
    n = len(a)
    l = [[0.0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(j, n):
            s = a[i][j]
            for k in range(j):
                s -= l[i][k] * l[j][k]
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not SPD")
                l[j][j] = math.sqrt(s)
            else:
                l[i][j] = s / l[j][j]
    return l


def forward_substitute(l: List[List[float]], b: List[float]) -> List[float]:
    n = len(l)
    y = [0.0 for _ in range(n)]
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= l[i][k] * y[k]
        y[i] = s / l[i][i]
    return y


def backward_substitute(l: List[List[float]], y: List[float]) -> List[float]:
    n = len(l)
    x = [0.0 for _ in range(n)]
    for ii in range(n):
        i = n - 1 - ii
        s = y[i]
        for k in range(i + 1, n):
            s -= l[k][i] * x[k]
        x[i] = s / l[i][i]
    return x


def matvec(a: List[List[float]], x: List[float]) -> List[float]:
    out = [0.0 for _ in range(len(a))]
    for i, row in enumerate(a):
        acc = 0.0
        for j, value in enumerate(row):
            acc += value * x[j]
        out[i] = acc
    return out


def build_rhs(a: List[List[float]]) -> Tuple[List[float], List[float]]:
    x_true = [float(i + 1) for i in range(len(a))]
    return x_true, matvec(a, x_true)


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("ERROR expected one matrix path argument")
        return 1

    try:
        dense = fixture_matrix(argv[1])
        _, b = build_rhs(dense)
        l = dense_cholesky(dense)
        y = forward_substitute(l, b)
        x = backward_substitute(l, y)
    except FileNotFoundError:
        print("SKIP matrix file not found")
        return 0
    except Exception as exc:  # pragma: no cover - exercised from C harness
        print(f"ERROR {exc}")
        return 1

    print(f"OK {len(x)}")
    for value in x:
        print(f"{value:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
