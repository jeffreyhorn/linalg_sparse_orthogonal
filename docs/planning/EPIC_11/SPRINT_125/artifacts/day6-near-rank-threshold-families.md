# Sprint 125 Day 6 Near-Rank-Deficient Threshold Families

## Purpose

Day 6 defines the near-rank-deficient QR threshold-family policy needed before
Sprint 125 can accept or defer threshold evidence on Day 7.  The policy keeps
fixture-local threshold checks separate from global QR rank-policy claims and
from residual, nullspace, minimum-norm, pseudoinverse, economy, sparse-mode,
reorder, backend, corpus, and performance evidence.

This is a design artifact only.  It does not change C source, headers, Python
helpers, build targets, CMake, CTest, CI, public APIs, or public solver
selection wording.

## Inputs Reviewed

| Input | Day 6 Use |
| --- | --- |
| Sprint 125 Day 4 and Day 5 artifacts | Require pinned rank/nullity thresholds before any nullspace/subspace threshold evidence. |
| Sprint 124 Day 2 QR rank policy | Provides the rule that threshold evidence must be fixture-local and must not imply a global QR rank policy. |
| `include/sparse_qr.h` | Documents `sparse_qr_rank()` and `sparse_qr_rank_info()` relative tolerance semantics. |
| `src/sparse_qr.c` | Implements post-factorization rank as `abs(R(i,i)) > tol * abs(R(0,0))`, with `tol <= 0` using `eps * max(m,n)`. |
| `tests/test_qr.c` | Owns deterministic QR rank, nullspace, diagonal threshold, rank diagnostics, economy, sparse-mode, and reorder evidence. |
| `tests/qr_external_dense_reference.py` | Current standard-library helper owner for bounded external QR evidence. |

## Current Threshold Evidence Inventory

| Evidence | Owner | Current Behavior | Day 6 Interpretation |
| --- | --- | --- | --- |
| `test_rank_explicit_tol` | `tests/test_qr.c` | Verifies that a looser explicit tolerance does not increase rank on a tiny diagonal fixture. | Directional tolerance smoke only; not enough for external threshold evidence. |
| `test_qr_rank_diagonal_threshold_fixture` | `tests/test_qr.c` | Uses diagonal values `[1, 1e-8, 1e-12, 0]` and expects ranks 3, 2, and 1 at tolerances `1e-14`, `1e-10`, and `1e-6`. | Best Day 7 candidate because expected ranks are mechanically tied to relative diagonal buckets. |
| `sparse_qr_rank_info()` | `src/sparse_qr.c` | Publishes rank, `k`, `r_max`, `r_min`, `condest`, and `near_deficient`. | Useful diagnostics; do not treat `near_deficient` as the threshold proof itself. |
| Factorization-time `qr.rank` | `src/sparse_qr.c` | Uses an internal `1e-14 * max(m,n) * abs(R(0,0))` style threshold while factoring. | Diagnostic only for Day 7 threshold fixtures unless the fixture explicitly claims factorization-time rank. |
| Nullspace threshold use | `sparse_qr_nullspace()` | Uses `sparse_qr_rank(qr, tol)` to decide nullity. | Day 7 may compute nullity only if the threshold rank claim is accepted first. |

## Threshold Semantics

`sparse_qr_rank(qr, tol)` and `sparse_qr_rank_info(qr, tol)` use relative
threshold semantics:

- if `tol > 0`, the absolute threshold is `tol * abs(R(0,0))`;
- if `tol <= 0`, the absolute threshold is
  `eps * max(m,n) * abs(R(0,0))`;
- rank is the count of leading `R` diagonal entries whose absolute values are
  strictly greater than the absolute threshold;
- expected rank changes must be tied to named fixture values and named
  thresholds, not inferred from residual success or solver success.

## Candidate Matrix Families

| Candidate family | Example fixture | Expected ranks | Trust value | Day 6 disposition |
| --- | --- | --- | --- | --- |
| Diagonal bucket ladder | `qr_rank_threshold_diag4_family` with diagonal `[1, 1e-8, 1e-12, 0]` | rank 3 at `1e-14`, rank 2 at `1e-10`, rank 1 at `1e-6` | High. Exact diagonal `R` behavior makes threshold semantics easy to diagnose. | Preferred Day 7 candidate. |
| Scaled diagonal bucket ladder | scale the diagonal ladder by `1e-6` and `1e6` | Same ranks at the same relative tolerances | Moderate. Proves relative scale handling if Day 7 has time. | Candidate only after the unscaled ladder is accepted. |
| Perturbed duplicate-column family | duplicate-column 5x4 plus `epsilon` perturbation in one duplicate column | rank changes when `epsilon / abs(R(0,0))` crosses the named tolerance | Moderate, but QR pivoting and roundoff make diagnostics harder. | Defer unless diagonal ladder evidence is already complete. |
| Dependent-row near-threshold family | dependent-row fixture with a tiny row perturbation | rank changes across named tolerances | Moderate, but row-space perturbation mixes rank and residual interpretation. | Defer to a future threshold/nullspace owner. |
| Wide near-threshold family | wide 3x5 or 4x6 fixture with tiny independent column | rank/nullity changes across named tolerances | Useful for nullity but higher policy risk. | Defer until Day 7 or later only after rank threshold evidence is accepted. |
| SuiteSparse near-threshold corpus | optional Matrix Market slice with observed small `R` diagonals | Broad but unstable across data/platforms. | Defer to Day 8-9 corpus policy. |

## Preferred Day 7 Fixture Protocol

If Day 7 accepts the diagonal ladder, use a bounded helper/test protocol with
these semantics:

| Field | Requirement |
| --- | --- |
| Fixture key | Prefer a new explicit key such as `qr_rank_threshold_diag4_family`; do not overload `qr_rankdef_duplicate_5x4_rank_only`. |
| Matrix | 4x4 diagonal `[1, 1e-8, 1e-12, 0]`. |
| Thresholds | `1e-14`, `1e-10`, `1e-6`. |
| Expected ranks | `3`, `2`, `1`. |
| Optional metadata | `rows`, `cols`, `k`, and diagonal magnitudes. |
| Output protocol | Either `OK 6` for threshold/rank pairs or a richer named-count protocol. |
| Test owner | `tests/test_qr.c`, because threshold rank is QR factorization/rank diagnostic evidence. |
| Helper owner | `tests/qr_external_dense_reference.py`, using standard-library arithmetic only. |
| Required comparison | Compare product ranks from `sparse_qr_rank(qr, threshold)` against expected ranks. |
| Diagnostics | Print fixture key, threshold, expected rank, product rank, `R` diagonal, and absolute threshold. |

## Threshold and Scale Policy

| Policy Point | Rule |
| --- | --- |
| Fixture-local thresholds | Every accepted threshold fixture must list exact threshold values and expected ranks. |
| Relative scale | Relative thresholds may be checked with scaled variants only if expected ranks stay unchanged and diagnostics include scale. |
| Default threshold | `tol <= 0` behavior may be diagnostic, but Day 7 should not promote it as an external default-rank claim unless explicitly accepted. |
| Strict comparison | Expected ranks must account for the implementation's strict `abs(R(i,i)) > abs_tol` rule. |
| Zero tail | Exact zero diagonal entries must be treated as below any non-negative threshold; do not conflate this with tiny positive tails. |
| Perturbation size | Perturbations must be separated by at least two orders of magnitude from the adjacent accepted thresholds to avoid roundoff ambiguity. |
| Nullity | Nullity may be asserted as `n - rank` only after the fixture's rank at that threshold is accepted. |

## Stability and Failure Diagnostics

Any accepted Day 7 threshold test should report enough information to decide
whether a failure is a fixture problem, reference-helper problem, or product QR
rank behavior problem:

- fixture key;
- matrix family and scale;
- threshold;
- expected rank;
- product rank;
- `R` diagonal magnitudes in factorization order;
- absolute threshold used by the product code;
- optional `sparse_qr_rank_info()` fields: `rank`, `r_max`, `r_min`,
  `condest`, and `near_deficient`;
- helper status: `OK`, `SKIP`, or `ERROR`.

## Non-Global Interpretation Rules

Accepted Day 7 threshold evidence may claim only that the named fixture family
produces the expected rank at the named relative thresholds.

It must not claim:

- a global QR rank threshold policy;
- equivalence to LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos,
  Eigen, ARPACK, or any dense-library threshold policy;
- broad rank-deficient QR parity;
- residual correctness;
- raw nullspace basis equality;
- minimum-norm or pseudoinverse behavior;
- economy-mode, sparse-mode, reorder, backend, platform, corpus, performance,
  package, ABI, or public API behavior.

## Day 7 Implementation Checklist

1. Start with the diagonal bucket ladder.
2. Keep threshold evidence in `tests/test_qr.c`.
3. Add helper output only if it improves trust beyond the existing
   deterministic test; otherwise explicitly defer external threshold evidence.
4. If helper-backed evidence is accepted, keep the protocol value-only and
   rank-only.
5. Print threshold diagnostics for each expected rank.
6. Run focused helper and QR tests, then the full C quality gate if `.c` or
   `.h` files change.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 4 has explicit fixture-family rules. | Complete | See candidate matrix families and preferred Day 7 protocol. |
| No threshold evidence creates a global rank-policy claim. | Complete | See non-global interpretation rules. |
| Day 7 can implement or defer without rediscovering semantics. | Complete | See threshold/scale policy, diagnostics, and implementation checklist. |

