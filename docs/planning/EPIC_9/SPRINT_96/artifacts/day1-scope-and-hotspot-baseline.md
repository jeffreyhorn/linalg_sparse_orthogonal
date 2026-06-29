# Sprint 96 Day 1: Scope And Hotspot Baseline

## Purpose

Day 1 opens Sprint 96 by refreshing the large-source and giant-test baseline
from the live merged tree. The goal is not to rank or implement cleanup yet.
Day 1 defines the candidate surfaces and validation expectations that Day 2 can
rank.

## Sprint 96 Scope

Sprint 96 implements the Epic 9 maintainability phase centered on:

- large mixed-role source owners
- dense giant-test proof owners
- bounded source extraction design
- one direct-family source cleanup batch
- one solver/algorithm source cleanup batch
- one giant-test architecture batch
- internal comment/rationale cleanup on touched files
- full validation and closeout

Non-goals for Day 1:

- no source movement
- no test splitting
- no benchmark command rename
- no generated API edit
- no broad rewrite of `docs/algorithm.md`
- no attempt to resolve every sprint-named historical test owner

## Live Source Hotspot Candidates

Top current source and internal-header owners:

| Rank | File | Lines | Day 1 classification |
|---:|---|---:|---|
| 1 | `src/sparse_ldlt_csc.c` | 2760 | direct-family implementation hotspot |
| 2 | `src/sparse_iterative.c` | 1854 | solver-family implementation hotspot |
| 3 | `src/sparse_lu_csr.c` | 1665 | direct/CSR implementation hotspot |
| 4 | `src/sparse_qr.c` | 1563 | solver/algorithm implementation hotspot |
| 5 | `src/sparse_ldlt.c` | 1535 | direct-family implementation hotspot |
| 6 | `src/sparse_eigs.c` | 1534 | solver/algorithm implementation hotspot |
| 7 | `src/sparse_matrix.c` | 1355 | shared matrix-shell implementation hotspot |
| 8 | `src/sparse_svd.c` | 1319 | solver/algorithm implementation hotspot |
| 9 | `src/sparse_chol_csc.c` | 1279 | direct-family implementation hotspot |
| 10 | `src/sparse_lu.c` | 1042 | direct-family implementation hotspot |
| 11 | `src/sparse_chol_csc_internal.h` | 1017 | direct-family internal-header hotspot |
| 12 | `src/sparse_dense.c` | 955 | shared dense-helper hotspot |
| 13 | `src/sparse_ldlt_csc_internal.h` | 928 | direct-family internal-header hotspot |
| 14 | `src/sparse_eigs_thick_restart.c` | 915 | eigensolver implementation hotspot |
| 15 | `src/sparse_graph_internal.h` | 894 | graph/reorder internal-header hotspot |

Initial reading:

- `src/sparse_ldlt_csc.c` remains the largest direct-family candidate and is
  still the first named source in the Sprint 96 project plan.
- `src/sparse_iterative.c` remains the first solver-family candidate named in
  the project plan.
- QR, eigensolver, SVD, matrix-shell, and direct-family adjacent files remain
  live candidates for Day 2 ranking, but Day 1 does not pick the landing batch.

## Live Giant-Test Candidates

Top current proof-owner tests:

| Rank | File | Lines | Day 1 classification |
|---:|---|---:|---|
| 1 | `tests/test_chol_csc.c` | 5029 | giant direct-family proof owner |
| 2 | `tests/test_ldlt_csc.c` | 3680 | giant direct-family proof owner |
| 3 | `tests/test_integration.c` | 3421 | giant shared lifecycle/integration proof owner |
| 4 | `tests/test_qr.c` | 3234 | giant solver-family proof owner |
| 5 | `tests/test_ldlt.c` | 2977 | large direct-family proof owner |
| 6 | `tests/test_etree.c` | 2962 | large direct-family/algorithm proof owner |
| 7 | `tests/test_graph.c` | 2925 | large graph/reorder proof owner |
| 8 | `tests/test_iterative.c` | 2841 | large iterative proof owner |
| 9 | `tests/test_svd.c` | 2766 | large SVD proof owner |
| 10 | `tests/test_reorder_nd.c` | 2340 | large reorder/ND proof owner |
| 11 | `tests/test_ilu.c` | 1974 | large preconditioner proof owner |
| 12 | `tests/test_colamd.c` | 1957 | large ordering proof owner |
| 13 | `tests/test_lu_csr.c` | 1899 | large direct/CSR proof owner |
| 14 | `tests/test_minres.c` | 1588 | solver proof owner |
| 15 | `tests/test_bicgstab.c` | 1586 | solver proof owner |

Initial reading:

- `tests/test_chol_csc.c` remains the densest proof-owner concentration.
- `tests/test_ldlt_csc.c` and `tests/test_integration.c` are the next
  strongest direct-family and shared-lifecycle proof surfaces.
- Sprint 95 renamed the selected direct CSC proof owners; Day 2 should not
  rerank stale `test_sprint18/19/20` filenames.

## Benchmark, Header, And Support Candidates

These surfaces are important, but Day 1 classifies them as supporting surfaces
unless a source or proof cleanup changes ownership:

| Surface | Current Day 1 role |
|---|---|
| `benchmarks/bench_eigs.c` | benchmark-side large owner; does not own correctness proof |
| `benchmarks/bench_main.c` | broad harness; possible support surface if owner changes |
| `benchmarks/bench_refactor_csc.c` | direct repeated-run measurement support |
| `benchmarks/bench_ldlt_csc.c` | direct LDL^T measurement support |
| `include/sparse_iterative.h` | public API contract surface; not a first extraction target |
| `include/sparse_eigs.h` | public API contract surface; not a first extraction target |
| `include/sparse_matrix.h` | public API contract surface; not a first extraction target |
| `docs/algorithm.md` | large chronology/rationale follow-up; not Day 1 source work |
| `docs/maintainer_guide.md` | policy interpretation surface |
| `README.md`, `INSTALL.md`, `benchmarks/README.md` | support surfaces cleaned by Sprint 95 |

## Surface Separation

Implementation hotspots:

- source and internal-header files under `src/`
- only selected `include/*.h` files if a public API contract truly changes

Proof-owner hotspots:

- large `tests/test_*.c` files
- Makefile and CMake registrations if files split or move

Benchmark support surfaces:

- benchmark drivers and benchmark README
- useful for measurement and observability, not correctness ownership

Documentation/support surfaces:

- README, install, tutorial, maintainer guide, algorithm reference, examples,
  and planning artifacts
- only move when source/test cleanup changes current owner interpretation

Historical surfaces:

- `docs/planning/**`
- old sprint chronology in planning artifacts remains historical by design

## Validation Expectations

Use this validation split during Sprint 96:

| Change type | Minimum validation expectation |
|---|---|
| Planning/docs-only artifacts | `git diff --check` and whitespace/link sanity as appropriate |
| `.c` or `.h` source/comment changes | `make format && make lint && make test` |
| test file split/rename | `make format && make lint && make test`, plus stale-reference scans |
| Makefile/CMake registration changes | full quality chain and registration/reference scans |
| benchmark command or CLI changes | full quality chain plus targeted benchmark help/smoke checks |
| generated API docs | do not hand-edit; update source comments first |

## Day 1 Result

Sprint 96 starts from a current live baseline. The likely first implementation
center remains direct-family source ownership, with `src/sparse_ldlt_csc.c` as
the largest candidate. The likely second implementation lane remains a
solver/algorithm source owner such as `src/sparse_iterative.c`, while giant-test
work should follow a design step rather than a broad split.
