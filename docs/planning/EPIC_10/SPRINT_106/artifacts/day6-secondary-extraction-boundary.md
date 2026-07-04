# Sprint 106 Day 6 - Secondary Extraction Boundary

## Purpose

Day 6 selects the next source extraction seam after the LDLT CSC
row-adjacency split. The selected seam must be narrow, private, reviewable,
and testable before implementation starts on Day 7.

## Inputs

- `docs/planning/EPIC_10/SPRINT_106/PLAN.md`
- `docs/planning/EPIC_10/SPRINT_106/artifacts/day2-extraction-target-rerank.md`
- current source sizes for LU CSR, QR, eigensolver, iterative, SVD, and LDLT
- representative source inspection of:
  - `src/sparse_qr.c`
  - `src/sparse_lu_csr.c`
  - `src/sparse_eigs.c`
  - `src/sparse_iterative.c`

## Current Secondary Candidate Snapshot

| candidate | current lines | nearby proof owner | Day 6 assessment |
|---|---:|---|---|
| `src/sparse_lu_csr.c` | 1,665 | `tests/test_lu_csr.c` | largest secondary source, but candidate helpers are embedded in CSR elimination and dense LU behavior |
| `src/sparse_qr.c` | 1,563 | `tests/test_qr.c` | best Day 7 source split: cohesive private Householder and column utility helpers |
| `src/sparse_eigs.c` | 1,538 | `tests/test_eigs.c` | valuable, but seams are close to backend dispatch, shift-invert, refinement, and Sprint 103 evidence behavior |
| `src/sparse_iterative.c` | 1,495 | `tests/test_iterative.c` | valuable, but seams are public-handle/workspace/stagnation related and need a separate boundary |
| `src/sparse_svd.c` | 1,319 | `tests/test_svd.c` | lower source pressure than QR/LU; proof fixture cleanup may be higher value |
| `src/sparse_ldlt.c` | 1,535 | `tests/test_ldlt.c` | defer until after the CSC follow-through settles linked-list/backend ownership expectations |

## Selected Day 7 Seam

Select the QR Householder and sparse-column helper seam.

Planned private files:

- `src/sparse_qr_internal.h`
- `src/sparse_qr_householder.c`

Planned responsibilities:

| helper | current owner | planned owner | rationale |
|---|---|---|---|
| `s29_qr_now_s(...)` | `src/sparse_qr.c` | `src/sparse_qr_householder.c` | private progress timing utility used by QR factorization only |
| `householder_compute(...)` | `src/sparse_qr.c` | `src/sparse_qr_householder.c` | pure Householder kernel |
| `householder_apply(...)` | `src/sparse_qr.c` | `src/sparse_qr_householder.c` | pure Householder application kernel used by factorization and Q application |
| `sparse_extract_column(...)` | `src/sparse_qr.c` | `src/sparse_qr_householder.c` | sparse-mode QR column utility |
| `householder_apply_to_column(...)` | `src/sparse_qr.c` | `src/sparse_qr_householder.c` | sparse-mode QR helper built on the shared apply kernel |

Recommended names for the moved helpers:

- `s29_qr_now_s`
- `sparse_qr_householder_compute`
- `sparse_qr_householder_apply`
- `sparse_qr_extract_column`
- `sparse_qr_householder_apply_to_column`

## Dependency Map

The new private source owner should include:

- `sparse_qr_internal.h`
- `sparse_matrix_internal.h`
- `<math.h>`
- `<string.h>`
- `<time.h>`

The private header should include:

- `sparse_qr.h`
- `sparse_matrix.h`

`src/sparse_qr.c` should include the new private header and stop declaring the
extracted helpers locally.

No public header change is planned.

## Build Follow-Through

Day 7 must update all library source membership surfaces:

- `build-metadata/library_sources.txt`
- `Makefile`
- `CMakeLists.txt`

The source-list checker should remain exact:

```sh
python3 scripts/check_library_sources.py
```

Expected source-count outcome after Day 7: 44 library sources if only
`src/sparse_qr_householder.c` is added to the existing 43-source list.

## Focused Validation Plan

Minimum focused validation after the Day 7 source split:

```sh
python3 scripts/check_library_sources.py
make build/test_qr build/test_colamd build/test_sprint6_integration
./build/test_qr
./build/test_colamd
./build/test_sprint6_integration
```

Because Day 7 will touch `.c` and likely `.h` files, the required full gate is:

```sh
make format && make lint && make test
```

## Skipped Candidates

| candidate | reason skipped for Day 7 | residual queue |
|---|---|---|
| LU CSR grow/validate split | smaller helper group and closer coupling to elimination internals | revisit after QR split if Day 8 can safely extract a second seam |
| LU CSR dense LU helpers | public-ish utility functions already exposed through `sparse_lu_csr.h`; extraction would need stronger API-owner review | defer |
| eigensolver shift-invert/refinement | high value but close to backend dispatch and external comparison claims | defer to a dedicated eigensolver boundary |
| iterative handle/workspace helpers | visible seam, but touches public handle lifecycle and block-solver defaults | defer to a dedicated iterative boundary |
| SVD source split | less obvious low-risk source seam than QR; test fixture pressure is higher than source pressure | defer |
| linked-list LDLT | related to direct solver work but not the next cleanest post-CSC seam | defer |

## Day 6 Decision

Day 7 should extract QR Householder and sparse-column helpers into a private QR
source owner. The seam is cohesive, private, low API risk, and covered by
focused QR, COLAMD+QR, and Sprint 6 QR integration tests.
