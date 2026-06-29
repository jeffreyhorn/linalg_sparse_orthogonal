# Sprint 96 Day 2: Hotspot Rerank

## Purpose

Day 2 ranks the live Sprint 96 maintainability candidates from Day 1 by review
cost, ownership ambiguity, extraction risk, and validation blast radius. The
goal is to choose bounded fix-now work for the sprint while separating broad
cleanup that should remain residual.

## Ranking Method

Candidates were weighted by:

- file size and local function density
- number of mixed responsibilities in one owner
- coupling to public API, build, and generated documentation surfaces
- proof-owner density and test registration size
- expected validation blast radius
- fit with the Sprint 96 project-plan sequence

The rerank favors work that can reduce current review cost without forcing a
public API redesign, broad benchmark rename, or generated documentation churn.

## Measured Implementation Signals

| File | Lines | Function-like entries | Rerank signal |
|---|---:|---:|---|
| `src/sparse_ldlt_csc.c` | 2760 | 51 | largest mixed direct-family owner |
| `src/sparse_iterative.c` | 1854 | 37 | largest iterative solver owner |
| `src/sparse_lu_csr.c` | 1665 | 15 | large direct/CSR owner with narrower entry count |
| `src/sparse_qr.c` | 1563 | 19 | solver/algorithm owner with giant proof file |
| `src/sparse_ldlt.c` | 1535 | 20 | public LDLT owner, adjacent to CSC direct work |
| `src/sparse_eigs.c` | 1534 | 35 | solver owner with restart and handle coupling |
| `src/sparse_matrix.c` | 1355 | 39 | shared matrix shell; high cross-cut risk |
| `src/sparse_svd.c` | 1319 | 14 | solver owner with lower entry density |
| `src/sparse_chol_csc.c` | 1279 | 31 | direct-family owner already paired with dense proof |

## Measured Proof-Owner Signals

| File | Lines | Test/register entries | Rerank signal |
|---|---:|---:|---|
| `tests/test_chol_csc.c` | 5029 | 304 | densest current proof owner |
| `tests/test_ldlt_csc.c` | 3680 | 193 | direct CSC proof concentration |
| `tests/test_integration.c` | 3421 | 112 | shared lifecycle and progress/cancel proof owner |
| `tests/test_qr.c` | 3234 | 147 | giant QR proof owner |
| `tests/test_ldlt.c` | 2977 | 177 | public LDLT proof owner |
| `tests/test_etree.c` | 2962 | 195 | direct algorithm proof owner |
| `tests/test_graph.c` | 2925 | 123 | graph/reorder proof owner |
| `tests/test_iterative.c` | 2841 | 158 | iterative solver proof owner |
| `tests/test_svd.c` | 2766 | 172 | SVD proof owner |
| `tests/test_reorder_nd.c` | 2340 | 71 | nested-dissection proof owner |

## Ranked Implementation Hotspots

### 1. `src/sparse_ldlt_csc.c`

This is the strongest Sprint 96 implementation candidate. It is the largest
source file, has the highest measured function-like entry count, and mixes
several review domains:

- dense backend selection and Accelerate/external probe logic
- allocation, row adjacency, and supernode detection helpers
- conversion, writeback, and validation helpers
- linked-list wrapper and bridge logic
- native Bunch-Kaufman workspace and factorization helpers
- solve and supernodal paths

Review-cost rating: very high.

Ownership ambiguity: high, because backend dispatch, conversion, symbolic
helpers, native numeric work, and solve paths all live in one owner.

Extraction risk: medium-high. The file has strong internal coupling, but most
candidate boundaries are still internal and do not require a public API change.

Validation risk: high. Any implementation change must run the full quality
chain and should be read against the CSC LDLT and direct-family proof owners.

Sprint 96 role: fix-now direct-family source cleanup candidate for Days 3-6.

### 2. `src/sparse_iterative.c`

This is the strongest solver-family source candidate. It is the second-largest
implementation owner and mixes:

- reusable handle and workspace helpers
- convergence and stagnation tracking
- CG default, matrix-free, and handle paths
- GMRES adapter, matrix-free, internal, and handle paths
- block CG, block GMRES, and MINRES wrappers
- BiCGStab and block/matrix-free paths

Review-cost rating: high.

Ownership ambiguity: high, because handle lifecycle, solver kernels,
matrix-free adapters, block wrappers, and convergence policy share one file.

Extraction risk: medium. The owner is public-API adjacent through
`include/sparse_iterative.h`, but a bounded internal helper cleanup can avoid
API movement.

Validation risk: high. Changes should run the full quality chain and receive
targeted attention from iterative, MINRES, BiCGStab, and block-solver tests.

Sprint 96 role: fix-now solver-family cleanup candidate for Days 7-9, pending
Day 7 final scope selection.

### 3. `src/sparse_qr.c`

QR is a strong alternate solver/algorithm candidate. It is large, paired with a
giant proof owner, and likely carries dense review cost. It ranks below
`src/sparse_iterative.c` because Sprint 96 explicitly names iterative work and
because QR cleanup could overlap more directly with the already-large
`tests/test_qr.c` proof surface.

Sprint 96 role: alternate solver-family candidate if iterative cleanup proves
too risky or too broad after Day 7 inspection.

### 4. `src/sparse_eigs.c`

The eigensolver owner has high function density and likely shares responsibility
with restart helpers and public handle semantics. It is a valid residual
maintainability target, but it is less aligned with the direct plus iterative
Sprint 96 sequence.

Sprint 96 role: residual or alternate solver-family candidate.

### 5. `src/sparse_lu_csr.c`, `src/sparse_ldlt.c`, `src/sparse_chol_csc.c`

These direct-family owners remain large, but they rank below
`src/sparse_ldlt_csc.c` for the fix-now direct lane. They should be inspected if
the selected LDLT CSC cleanup exposes shared helper ownership, but broad
cleanup across multiple direct owners would exceed the sprint's bounded-review
goal.

Sprint 96 role: residual direct-family candidates and context for source
extraction design.

### 6. `src/sparse_matrix.c`, `src/sparse_svd.c`, and internal headers

`src/sparse_matrix.c` is shared enough that cleanup could create a wide
validation blast radius. `src/sparse_svd.c` is large but lower priority than the
iterative/QR/eigs lane for this sprint. Internal headers such as
`src/sparse_chol_csc_internal.h`, `src/sparse_ldlt_csc_internal.h`, and
`src/sparse_graph_internal.h` should only move when a selected source cleanup
requires ownership clarification.

Sprint 96 role: residual, dependency-aware cleanup only.

## Ranked Proof-Owner Hotspots

### 1. `tests/test_chol_csc.c`

This is the strongest giant-test architecture candidate. It is the largest test
file and has the highest measured test/register density. It includes proof
clusters for allocation, growth, conversion, permutation, prediction,
validation, workspace behavior, elimination, solve, factorization, supernode
detection, postorder behavior, dense backend behavior, writeback, dispatch, and
external dense reference checks.

Review-cost rating: very high.

Ownership ambiguity: very high, because many different proof responsibilities
share one registration surface.

Extraction risk: medium-high. A registration-preserving split or helper
ownership cleanup is possible, but a broad proof rewrite would be too risky.

Validation risk: high. Any split or owner movement requires the full quality
chain and stale-reference scans.

Sprint 96 role: provisional Day 10-11 giant-test cleanup candidate.

### 2. `tests/test_ldlt_csc.c`

This is the strongest adjacent direct CSC proof candidate. It may become
fix-now if Days 3-6 touch LDLT CSC behavior enough that proof ownership should
move with it. Otherwise, touching both the largest direct source and this large
proof owner in the same sprint could create unnecessary review scope.

Sprint 96 role: adjacent proof context; possible targeted cleanup only.

### 3. `tests/test_integration.c`

This file is a high-value proof owner but has broad lifecycle and
progress/cancel coverage across multiple subsystems. It ranks below focused
direct proof owners because splitting it would affect more shared validation
surface.

Sprint 96 role: residual, unless selected source cleanup creates an explicit
integration-owner gap.

### 4. `tests/test_qr.c`

This file becomes a stronger fix-now proof candidate only if the solver-family
cleanup chooses QR over iterative work. Otherwise, it remains a residual giant
test candidate.

Sprint 96 role: alternate solver-proof cleanup candidate.

### 5. `tests/test_iterative.c` and other large proof owners

`tests/test_iterative.c`, `tests/test_svd.c`, `tests/test_ldlt.c`,
`tests/test_etree.c`, `tests/test_graph.c`, and `tests/test_reorder_nd.c` all
remain large proof-owner candidates. They should be kept out of the fix-now
queue unless the selected source cleanup exposes a narrow, owner-specific proof
boundary.

Sprint 96 role: residual proof-owner backlog.

## Fix-Now Queue

### Day 3: Source Extraction Design

- Map exact extraction boundaries in `src/sparse_ldlt_csc.c`.
- Select the direct-family cleanup batch for Days 4-6.
- Confirm the solver-family cleanup target for Days 7-9, with
  `src/sparse_iterative.c` as the default.
- Define validation commands and stale-reference scans before editing code.

### Days 4-6: Direct-Family Source Cleanup

- Implement one bounded cleanup in `src/sparse_ldlt_csc.c`.
- Prefer an internal boundary such as backend/probe ownership, conversion and
  writeback helpers, or native-workspace helpers over a public API change.
- Keep any internal-header edits narrowly tied to the selected extraction.

### Days 7-9: Solver-Family Source Cleanup

- Use `src/sparse_iterative.c` as the default solver-family source target.
- Prefer internal helper separation around handle/workspace policy,
  convergence/stagnation policy, or block-wrapper ownership.
- Keep `include/sparse_iterative.h` stable unless Day 7 finds a clear contract
  defect.

### Days 10-11: Giant-Test Architecture Cleanup

- Use `tests/test_chol_csc.c` as the provisional proof-owner target.
- Prefer a registration-preserving split or helper-owner cleanup over changing
  assertions broadly.
- Reconsider `tests/test_ldlt_csc.c` only if Days 4-6 directly change LDLT CSC
  proof ownership.

### Day 12: Comment And Rationale Cleanup

- Clean stale comments in the files touched by Days 4-11.
- Avoid broad documentation rewrites and generated API documentation edits.

### Days 13-14: Validation And Closeout

- Run the full quality chain for any `.c` or `.h` changes.
- Add targeted stale-reference scans for any split, rename, or registration
  movement.
- Capture closeout notes and residual maintainability backlog.

## Residual Maintainability Queue

The following work should remain outside the immediate fix-now scope unless a
selected cleanup makes it unavoidable:

- full decomposition of all direct-family source owners
- broad rewrites of `src/sparse_lu_csr.c`, `src/sparse_qr.c`,
  `src/sparse_eigs.c`, `src/sparse_svd.c`, or `src/sparse_matrix.c`
- broad public-header redesign
- benchmark command or harness renames
- generated API documentation edits
- broad `docs/algorithm.md` chronology modernization
- simultaneous splits of multiple giant proof owners
- cleanup of historical sprint planning names outside active product-facing
  surfaces

## Validation Notes

Day 2 changed planning artifacts only, so no source quality chain is required.
The minimum Day 2 hygiene checks are:

- `git diff --check`
- trailing-whitespace scan for `docs/planning/EPIC_9/SPRINT_96`

Any later `.c` or `.h` change must run `make format && make lint && make test`
before the day is considered complete.

## Day 2 Result

The live hotspot map now has a bounded fix-now order. `src/sparse_ldlt_csc.c`
is the direct-family source target, `src/sparse_iterative.c` is the default
solver-family source target, and `tests/test_chol_csc.c` is the provisional
giant-test architecture target. Broader source rewrites, public API movement,
benchmark changes, generated documentation changes, and multi-test splits stay
in the residual queue.
