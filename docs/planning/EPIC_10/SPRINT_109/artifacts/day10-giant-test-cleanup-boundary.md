# Day 10 Giant-Test Cleanup Boundary

## Purpose

Day 10 re-inventories the remaining large-test cleanup candidates across the
residual proof-owner tests and selects one bounded cleanup family for the next
implementation pass.

Day 10 moves no code.

## Large-Test Snapshot

Current residual giant-test sizes:

| File | Lines | Primary Proof Surface |
|---|---:|---|
| `tests/test_ldlt_csc.c` | 3896 | CSC LDLT factorization, solve, external dense reference, and supernodal proof lanes. |
| `tests/test_qr.c` | 3210 | QR factorization, least-squares, rank, reorder, sparse mode, refinement, and fixture-backed solve proof lanes. |
| `tests/test_iterative.c` | 2849 | CG, BiCGSTAB, GMRES, MINRES, preconditioner, restart, and convergence proof lanes. |
| `tests/test_svd.c` | 2890 | Dense-reference SVD, low-rank, partial SVD, full U/V, ordering, and reconstruction proof lanes. |

These files remain proof-owner territory. Cleanup is allowed only when it hides
repeated setup without hiding solver calls, expected values, tolerances, or
assertions that explain the tested contract.

## Duplicate-Work Exclusion List

The following families are explicitly excluded because Sprint 108 or earlier
Sprint 109 work already addressed them or documented their no-move contracts:

| File | Excluded Family | Reason |
|---|---|---|
| `tests/test_ldlt_csc.c` | row-adjacency exact-set helper and duplicate-entry assertion refinement | Already covered by Sprint 108 residual cleanup. |
| `tests/test_ldlt_csc.c` | `assert_s20_solve_residual_below` | Already extracted as the Sprint 108 LDLT solve-residual proof helper. |
| `tests/test_qr.c` | small 4x3 banded, duplicate-column, and near-duplicate fixture builders | Already covered by Sprint 108 QR fixture cleanup. |
| `tests/test_qr.c` | `make_qr_tall_diagonal_dominant` fixture builder | Already present; repeated tall-matrix construction is not a new cleanup target. |
| `tests/test_iterative.c` | sequential RHS helper, solver helper headers, and iterative handle helper headers | Already present and still the appropriate shared test-helper surface. |
| `tests/test_iterative.c` | diagonal-preconditioner fixture follow-through | Already completed in the prior sprint family. |
| `tests/test_svd.c` | diagonal fixture, rank-1 row-progression fixture, and full U/V 16x8 fixture | Already covered by Sprint 108 SVD validation cleanup. |
| `src/sparse_eigs.c` / `src/sparse_eigs_dense_internal.c` | dense Jacobi extraction | Already completed and validated on Days 4 and 5. |

## Candidate Inventory

| Candidate | File | Proof Clarity | Review Size | Validation Cost | Failure Localization | Decision |
|---|---|---|---|---|---|---|
| QR exact-solution RHS setup helper | `tests/test_qr.c` | High: helper can allocate/fill `x_exact` and `b = A*x_exact` only. | Small: one static helper plus repeated call-site replacement. | Moderate: focused `test_qr`, then full gate if code changes. | High: failures remain in the same QR tests with inline labels and tolerances. | Selected. |
| QR sequential RHS fill helper for least-squares/refinement smoke | `tests/test_qr.c` | Medium: setup-only, but less valuable than exact-RHS construction. | Small. | Moderate. | High. | Defer behind exact-RHS helper. |
| LDLT CSC external dense-reference oracle cleanup | `tests/test_ldlt_csc.c` | Medium/low: Python oracle, Windows skip, dense solve, and LDLT factorization are coupled. | Large. | High. | Medium. | Defer to a dedicated oracle-lane review. |
| Iterative exact-RHS allocation helper | `tests/test_iterative.c` | Medium: repeated pattern exists, but spans many solver families and convergence assumptions. | Large unless tightly sliced. | High. | Medium: assertion meanings vary by solver. | Defer until a smaller per-solver family is selected. |
| SVD full-mode proof-loop cleanup | `tests/test_svd.c` | Low for this sprint: loops expose storage-layout and orthogonality proof. | Medium/large. | High. | Medium. | Defer; assertions should remain inline. |

## Selected Cleanup Batch

Selected for the next code-change day:

```text
tests/test_qr.c exact-solution RHS setup helper
```

Scope:

- introduce one local static helper in `tests/test_qr.c` that allocates an
  `x_exact` vector, allocates a matching RHS vector, fills
  `x_exact[i] = i + 1`, and computes `b = A*x_exact`;
- replace repeated exact-RHS setup in QR solve and refinement tests where the
  helper does not obscure the tested solver behavior;
- keep matrix load/factorization, solve/refine calls, residual labels,
  tolerances, rank assertions, and solution-comparison assertions at the call
  sites.

Initial call-site candidates include:

- `test_qr_solve_nos4`;
- `test_qr_bcsstk04`;
- `test_qr_west0067`;
- `test_qr_vs_lu`;
- `test_qr_tall_synthetic`;
- `test_qr_refine_nos4`;
- reorder/AMD solve smoke where the helper can simplify setup without hiding
  the reorder proof.

Explicit exclusions from the selected batch:

- hand-authored tiny RHS arrays where the literal values explain the proof;
- overdetermined least-squares RHS vectors that are intentionally outside the
  column space;
- rank-deficient proof assertions and residual thresholds;
- dense-vs-sparse QR comparison assertions;
- any new compiled helper target or shared test-helper header.

## Proof-Visibility Rules

The selected helper may hide only allocation, sequential exact-solution fill,
and `sparse_matvec(A, x_exact, b)`.

The helper must not hide:

- `sparse_qr_factor`, `sparse_qr_factor_opts`, `sparse_qr_solve`, or
  `sparse_qr_refine` calls;
- expected rank values;
- residual labels and tolerances;
- QR-vs-LU solution comparison loops;
- reconstruction assertions;
- intentional least-squares or rank-deficient RHS values;
- cleanup of solver-owned QR structures.

The helper should remain `static` in `tests/test_qr.c`; no public header,
private production header, CTest registration, or build source list should
change.

## Focused Validation Plan

If the selected batch is implemented:

```sh
make build/test_qr
./build/test_qr
```

Because the implementation will modify a `.c` file, the branch must then run:

```sh
make format && make lint && make test
git diff --check
```

No extra CTest target or helper library is expected.

## Completion Criteria Status

- Residual giant-test cleanup candidates were inventoried.
- Sprint 108 and earlier Sprint 109 duplicate work was excluded.
- One bounded cleanup family was selected.
- The selected family requires no new compiled helper target.
- Proof assertions and solver calls remain visible at call sites.
- Focused and full validation scope is known before edits begin.
