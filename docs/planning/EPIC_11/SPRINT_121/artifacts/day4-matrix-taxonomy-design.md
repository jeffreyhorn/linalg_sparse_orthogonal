# Sprint 121 Day 4: Matrix Taxonomy Design

## Purpose

Define a shared deterministic matrix taxonomy for Sprint 121 SVD, QR, rank,
pseudoinverse, least-squares, low-rank, and expected-failure evidence. The
taxonomy combines the Day 2 SVD audit and Day 3 QR audit into fixture classes
that can drive helper extraction and focused proof without hiding tolerance,
rank, residual, reconstruction, orthogonality, or non-claim semantics.

This is a design artifact only. No C source, header, build, CMake, workflow, or
test membership changes are made by Day 4.

## Taxonomy Principles

- Builders construct deterministic matrices and known RHS/vector data; tests
  own the assertions.
- Fixture names must expose shape, rank, conditioning, sparsity, and behavior
  intent.
- Metadata must record expected rank, tolerance ownership, residual target,
  reconstruction target, singular-value or R-diagonal shape, and non-claims.
- SVD and QR may share fixture builders only when the fixture semantics are
  solver-neutral. Solver-specific checks remain in SVD or QR helper families.
- Partial SVD comparisons against this library's full SVD are internal
  regression oracles, not external dense-library parity.
- QR-vs-LU and QR-vs-SVD-pseudoinverse comparisons are bounded cross-solver
  checks, not broad direct-solver or LAPACK/SciPy parity claims.

## Fixture Metadata Schema

Every new or extracted fixture builder should have a metadata record with these
fields:

| Field | Required value |
|---|---|
| Fixture key | Stable kebab-case key, e.g. `svd-diag-threshold-6x6`. |
| Builder name | Proposed helper name or current scenario-local owner. |
| Matrix family | Diagonal, outer-product, duplicate-column, near-dependent, tridiagonal, Hilbert-like, SuiteSparse, generated sparse, or expected-failure. |
| Dimensions | `m`, `n`, and `k = min(m,n)` when relevant. |
| Sparsity pattern | Diagonal, banded, structured dense, sparse random-like deterministic, SuiteSparse, zero row/column, or duplicate column. |
| Shape class | Square, tall, wide, single-row, single-column, empty, or 1x1. |
| Rank model | Full rank, exact rank deficient, numerical threshold rank, rank unknown but bounded, or not applicable. |
| Expected rank | Exact value or explicit reason rank is fixture-dependent. |
| Conditioning/scaling | Well-conditioned, ill-conditioned, near-singular, threshold-sensitive, scaled, or not applicable. |
| Singular-value shape | Exact spectrum, separated spectrum, repeated values, zeros, tail-bound, internal-reference, or not applicable. |
| R-diagonal shape | Pivoted descending, zero/near-zero diagonal, rank threshold, or not applicable. |
| RHS/source vector | None, generated exact RHS, analytical incompatible RHS, SuiteSparse submatrix RHS, or explicit RHS. |
| Expected residual | Absolute/relative target and owner, or explicit nonzero residual expectation. |
| Reconstruction target | Max-absolute or relative-Frobenius target and owner. |
| Orthogonality target | U/Vt/Q target and orientation/stride owner. |
| Expected failure / skip | Unsupported, expected error, optional skip, or none. |
| Reference boundary | Exact analytical, internal full SVD, QR-vs-LU, QR-vs-pinv, SuiteSparse smoke, or none. |
| Non-claim notes | Statement that the fixture does not broaden external parity, platform, performance, or state-of-the-art claims. |

## Shared Fixture Taxonomy

| Fixture class | Primary consumers | Expected behavior | Tolerance owner | Non-claim note |
|---|---|---|---|---|
| `diag-exact-spectrum` | SVD, rank, condition number, QR diagonal | Exact singular values or R diagonal values; full rank unless zeros are named | Scenario test chooses `1e-10`/`1e-12` exact-value tolerance | Does not imply dense-library parity beyond analytical diagonal truth. |
| `diag-threshold-rank` | SVD rank, QR rank, condition number | Rank changes under explicit tolerances; threshold values documented | Rank test owns default and explicit tolerance semantics | Does not claim a universal numerical-rank policy. |
| `diag-repeated-spectrum` | SVD, partial SVD | Repeated singular values remain nonnegative and ordered within accepted equality | SVD test owns ordering and equality slack | Does not claim unique singular vectors in repeated subspaces. |
| `outer-product-rank-k` | SVD, QR, low-rank, nullspace | Exact low rank with known algebraic rank and optional known tail | Rank/low-rank test owns Frobenius and reconstruction targets | Does not claim large low-rank performance. |
| `duplicate-column-rank-def` | SVD rank, QR rank/nullspace, QR solve | Structural rank deficiency and known nullspace relationship | Rank/nullspace test owns `A*v` residual and rank threshold | Does not claim all rank-deficient patterns are covered. |
| `dependent-row-rank-def` | QR minimum-norm, QR nullspace | Dependent rows, zero rows, and consistent RHS behavior | Minimum-norm or nullspace test owns residual/norm target | Does not cover inconsistent rank-deficient least squares unless explicitly named. |
| `near-dependent-threshold` | SVD rank, QR rank, reconstruction | Near duplicate / near singular data with threshold-sensitive behavior | Scenario test owns explicit threshold and reconstruction slack | Does not claim robust rank detection for arbitrary noisy data. |
| `rectangular-tall-compatible` | QR solve, SVD full/partial, pseudoinverse | Tall matrix with generated RHS or exact reconstruction expected | Test owns near-zero residual and shape-specific reconstruction | Does not imply all tall least-squares cases are compatible. |
| `rectangular-tall-incompatible` | QR least squares, refinement | Tall matrix with nonzero least-squares residual expected | Test owns analytical or measured residual floor | Does not claim normal-equation or LAPACK least-squares parity. |
| `rectangular-wide-minnorm` | QR minnorm, SVD pseudoinverse | Wide underdetermined matrix with minimum-norm solution | Minimum-norm test owns norm comparison and residual | QR-vs-pinv is bounded cross-solver evidence only. |
| `rectangular-wide-basic` | SVD full/partial, QR factorization | Wide shape exercises storage and solve/factor boundaries | Solver test owns shape, rank, reconstruction, and vector layout | Does not imply minimum-norm behavior unless class says so. |
| `bidiagonal-explicit` | SVD low-level | Known bidiagonal SVD or Golub-Kahan extraction behavior | Bidiagonal test owns exact singular value and vector checks | Low-level SVD owner, not general matrix proof. |
| `hilbert-like-dense` | Partial SVD | Internal top-k comparison against full SVD with looser tolerance | Partial-SVD test owns relative top-k window | Internal regression oracle, not external dense truth. |
| `tridiagonal-band` | Low-rank, QR, refinement | Banded deterministic structure with known generated RHS or SVD tail | Scenario test owns residual or Frobenius tail target | Does not claim benchmark or performance result. |
| `suite-sparse-smoke` | SVD, QR, low-rank, minnorm | Bounded real-world smoke fixture, optionally skipped if unavailable | Scenario test owns matrix-specific tolerance and skip behavior | Smoke evidence only, not broad SuiteSparse coverage. |
| `mode-equivalence` | QR economy/sparse mode, low-rank env-on/off | Two implementation modes agree on rank, residual, output, or reconstruction | Mode test owns comparison metric and accepted slack | Does not claim one mode is faster or generally preferable. |
| `expected-api-error` | SVD, QR, low-rank, pinv, minnorm | Null, bad arg, factored/reordered matrix, invalid rank, or invalid tolerance returns expected error | Contract test owns exact error code | Not numerical evidence. |

## SVD-Specific Taxonomy Classes

| Fixture key | Expected behavior | Metadata requirements | Candidate owner |
|---|---|---|---|
| `svd-diag-exact` | Singular values equal absolute diagonal entries and remain descending | `sigma_expected[]`, rank, reconstruction tolerance, orthogonality target if UV requested | Days 6 and 10 helper/fixture expansion |
| `svd-diag-threshold` | Rank changes at explicit thresholds while reconstruction remains bounded | default rank, explicit-tolerance ranks, near-zero sigma values | Day 8 rank expansion |
| `svd-rank-def-duplicate-columns` | SVD rank and QR rank agree on duplicate-column fixture | duplicate relationship, expected rank, cross-solver non-claim | Day 8 cross-rank proof |
| `svd-lowrank-outer-product` | Dense/sparse low-rank output matches known rank or dense baseline | rank_k, tail sigma, Frobenius target, drop_tol | Day 10 low-rank expansion |
| `svd-partial-internal-reference` | Partial top-k values/vectors compare to this library's full SVD | full-SVD reference label, top-k tolerance, vector residual target | Day 10 partial-SVD expansion |
| `svd-pinv-moore-penrose` | Pseudoinverse satisfies named Moore-Penrose identity set | identity list, shape, rank, tolerance | Day 9 pseudoinverse expansion |
| `svd-condition-number` | Finite/infinite condition behavior follows sigma threshold | sigma_max/min, threshold, expected finite/infinite state | Day 8 rank/condition proof |

## QR-Specific Taxonomy Classes

| Fixture key | Expected behavior | Metadata requirements | Candidate owner |
|---|---|---|---|
| `qr-square-exact` | QR solve matches generated RHS and reconstruction is near exact | RHS generator, residual target, reconstruction target | Day 7 helper extraction |
| `qr-overdetermined-compatible` | Tall generated-RHS solve has near-zero residual | RHS generator, residual target, rank, shape | Day 9 least-squares expansion |
| `qr-overdetermined-incompatible` | Tall least-squares fixture has explicit nonzero residual expectation | analytical residual or bounded measured residual, rank, target norm | Day 9 least-squares expansion |
| `qr-underdetermined-minnorm` | Wide solve has minimum-norm solution with `A*x ~= b` | alternate solution or pinv reference, norm target, residual target | Day 9 minnorm expansion |
| `qr-rank-def-duplicate-column` | Rank and nullspace are owned by duplicate-column relationship | duplicate scale, expected rank, nullspace residual | Day 8 rank-deficient expansion |
| `qr-rank-def-dependent-row` | Dependent rows and zero rows preserve expected rank and consistency | row dependency, RHS consistency, residual target | Day 8-9 rank/minnorm expansion |
| `qr-near-rank-def-threshold` | Explicit rank tolerance changes rank or preserves monotonic behavior | perturbation, tolerance values, expected rank relation | Day 8 rank threshold expansion |
| `qr-economy-mode` | Economy mode preserves solve, Q shape, rank, and orthogonality expectations | full/economy comparison metric, Q dimensions, residual target | Day 7 helper extraction |
| `qr-sparse-mode` | Sparse mode agrees with dense mode on bounded metrics | dense/sparse labels, max-diff, rank equality, reconstruction | Day 7 helper extraction |
| `qr-reordered` | AMD/COLAMD/none reordering preserves solve/reconstruction and records fill | reorder mode, fill metric, solve residual | Deferred unless Day 9 touches reorder-sensitive solve proof |

## Expected-Failure Classes

| Class | Applies to | Expected behavior | Required metadata |
|---|---|---|---|
| `null-input-error` | SVD, QR, pinv, low-rank, minnorm, refinement | Return `SPARSE_ERR_NULL` or documented null-safe behavior | API, argument position, expected code |
| `bad-rank-or-k-error` | partial SVD, low-rank | Reject `k <= 0`, `k > min(m,n)`, or invalid rank | invalid value, dimensions, expected code |
| `bad-tolerance-error` | SVD, QR diagnostics when applicable | Reject negative tolerance where API promises rejection | tolerance value, API, expected code |
| `factored-or-permuted-matrix-error` | SVD, QR, low-rank, pinv, minnorm | Reject non-original row/column state | mutation source, expected `SPARSE_ERR_BADARG` |
| `optional-suite-sparse-skip` | SuiteSparse smoke fixtures | Skip if fixture is unavailable or explicitly too slow | fixture path, skip message, non-claim note |
| `unsupported-full-mode-error` | partial SVD | Reject unsupported full-vector partial SVD mode | option combination, expected code |
| `expected-incompatible-residual` | QR least squares | Nonzero residual is expected and bounded | RHS class, residual metric, acceptance target |

## Helper Placement Plan

| Helper family | Preferred placement | Allowed responsibility | Must remain scenario-local |
|---|---|---|---|
| Matrix builders | New narrow test-only helper header selected on Day 5, likely `tests/test_svd_qr_fixture_helpers.h` if source movement is justified | Build deterministic diagonal, duplicate-column, near-dependent, rectangular, and generated sparse matrices | Tolerance, expected rank, expected residual, performance or parity wording |
| Metadata constants | Same helper header or scenario-local static constants | Name fixture key, dimensions, expected rank, and analytical values | Pass/fail assertions and skip policies |
| SVD reconstruction / orthogonality | SVD-specific helper header or scenario-local static helpers | Compute residual/orthogonality with explicit leading dimensions | Economy/full interpretation, partial-SVD looser tolerance |
| QR reconstruction / residual | QR-specific helper header or existing QR test-local helper split | Compute `A*P = Q*R`, exact RHS, and residual measurements | Absolute vs relative residual threshold, compatible vs incompatible LS meaning |
| Rank/nullspace checks | Solver-specific helper families | Compute rank, nullspace residual, or threshold diagnostics | Default threshold policy and expected rank claims |
| Minimum-norm helpers | QR helper only after Day 7-9 owner decision | Measure norm, residual, and optional pinv comparison | Test placement and QR-vs-pinv non-claim wording |
| SuiteSparse fixture wrappers | Scenario-local by default | Load fixture and record skip reason | Broad corpus, platform, or performance claims |

## Implementation Sequencing Notes

1. Day 5 should rank helper extraction candidates using this taxonomy and keep
   source-list, CMake, CTest count, rollback, and validation impact explicit.
2. Day 6 should extract only SVD helpers whose tolerances stay caller-owned:
   reconstruction, orthogonality, rank, low-rank residual, and pseudoinverse
   identity measurement.
3. Day 7 should extract QR helpers for reconstruction, residual measurement,
   exact-RHS construction, duplicate-column builders, and minimum-norm
   measurements only if ownership remains clear.
4. Day 8 should add rank-deficient and near-dependent fixtures using
   `diag-threshold-rank`, `duplicate-column-rank-def`,
   `dependent-row-rank-def`, and `near-dependent-threshold` classes.
5. Day 9 should add rectangular least-squares and pseudoinverse evidence using
   separate compatible, incompatible, and minimum-norm classes.
6. Day 10 should add low-rank and partial-SVD evidence with internal-reference
   and non-claim language explicitly recorded.
7. Day 11-12 dense-reference or external-process pilot work should reference
   fixture taxonomy keys and record the reference trust boundary separately
   from local solver regression evidence.

## Validation Notes

This was a documentation-only design. Required validation is limited to
`git diff --check` and a focused trailing-whitespace scan over
`docs/planning/EPIC_11/SPRINT_121`.

## Completion Criteria Status

| Criterion | Status |
|---|---|
| Item 2 matrix taxonomy design is complete | Complete: shared, SVD-specific, QR-specific, and expected-failure classes are defined. |
| Each fixture class has explicit expected behavior and tolerance ownership | Complete: every taxonomy table names expected behavior and owner boundaries. |
| No taxonomy entry claims broad LAPACK/SciPy parity | Complete: non-claim notes preserve bounded analytical, internal, cross-solver, and SuiteSparse-smoke trust boundaries. |

## Non-Claims

This taxonomy does not claim broad LAPACK, SciPy, SuiteSparse, PETSc, Trilinos,
Eigen, platform, packaging, benchmark, or state-of-the-art parity. It defines
bounded deterministic fixture classes and metadata for Sprint 121 proof-owner
cleanup and expansion.
