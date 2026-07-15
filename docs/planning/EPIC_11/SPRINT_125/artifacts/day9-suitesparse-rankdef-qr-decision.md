# Sprint 125 Day 9 SuiteSparse Rank-Deficient QR Decision

## Decision

Explicitly deferred SuiteSparse rank-deficient QR evidence.

The checked-in SuiteSparse corpus does not currently provide a documented,
small rank-deficient QR candidate with pinned expected rank, threshold,
nullity, residual semantics, and support-tier behavior. The default QR
SuiteSparse matrices currently exercised by `tests/test_qr_solve.c` are
full-rank controls under the product QR implementation:

| Matrix | Focused run diagnostic | Current interpretation |
| --- | --- | --- |
| `nos4.mtx` | `rank=100` | Full-rank QR solve control. |
| `bcsstk04.mtx` | `rank=132` | Full-rank QR reconstruction and solve control. |
| `west0067.mtx` | `rank=67` | Full-rank QR solve control. |

Day 9 therefore does not add a SuiteSparse rank-deficient QR test. Adding one
without a pinned expected-rank contract would weaken the Day 8 corpus policy
and risk relabeling existing full-rank controls as rank-deficient evidence.

## Reviewed Candidate Paths

| Candidate path | Disposition | Reason |
| --- | --- | --- |
| Reuse `nos4.mtx` | Rejected | Current QR solve run reports full rank 100; no rank-deficient expectation is documented. |
| Reuse `bcsstk04.mtx` | Rejected | Current QR solve test asserts `qr.rank == n` and focused run reports full rank 132. |
| Reuse `west0067.mtx` | Rejected | Current focused run reports full rank 67; no rank-deficient expectation is documented. |
| Add `steam1.mtx` QR rank evidence | Deferred | It is a default checked-in SuiteSparse matrix, but it has no current QR rank-deficient owner or pinned expected-rank metadata. |
| Promote `fs_541_1.mtx` or `orsirr_1.mtx` | Deferred | Existing large SuiteSparse lane is gated by `SPARSE_TEST_LARGE=1`; no expected-rank contract is pinned. |
| Promote heavier report-only matrices | Deferred | `bcsstk14`, `Kuu`, `s3rmt3m3`, `Pres_Poisson`, `bloweybq`, and `tuma1` are too large or too unsupported for default rank-deficient QR evidence without a separate corpus-taxonomy decision. |

## Deferral Contract

A future sprint may promote SuiteSparse rank-deficient QR evidence only after
all of the following are available:

- a named matrix path and support tier
- expected rank, nullity, or threshold/rank pairs from an independent source
- explicit threshold semantics and absolute threshold diagnostics
- load, factorization, rank, `R` diagonal, residual, and reconstruction
  diagnostics as applicable
- default versus optional-large skip behavior
- focused validation command for the exact executable path
- full `make format && make lint && make test` gate if `.c` or `.h` files
  change

If the selected matrix is optional-large or report-only, the future owner must
prove skip behavior separately from numerical failure behavior. Missing
optional data may skip; accepted numerical disagreement must fail.

## Evidence Preserved

Existing SuiteSparse QR controls remain valid as controls, not as
rank-deficient evidence:

- `nos4.mtx` continues to cover QR solve, QR-vs-LU, refine, economy, and
  reorder/fillin paths.
- `bcsstk04.mtx` continues to cover QR reconstruction, QR solve, and
  sparse-mode comparison as a full-rank control.
- `west0067.mtx` continues to cover QR solve and sparse-mode comparison as a
  full-rank control.

The bounded Sprint 125 non-corpus rank-deficient QR evidence remains the source
of rank-deficient behavior proof for this sprint:

- `qr_rankdef_duplicate_5x4_residual_only`
- `qr_rankdef_duplicate_5x4_nullspace_projector`
- `qr_rank_threshold_diag4_family`

## Non-Claims Preserved

- No SuiteSparse rank-deficient QR evidence is accepted in Day 9.
- No broad SuiteSparse corpus correctness claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK, or backend
  parity claim.
- No portable performance, scalability, memory, platform, package, ABI, public
  API, or CI support claim.
- No global QR rank-threshold policy.
- No broad rank-deficient QR behavior claim.
- No raw nullspace or Q-basis orientation claim.
- No minimum-norm or pseudoinverse behavior claim.

## Validation

Focused diagnostic run passed:

```text
./build/test_qr_solve
```

The focused run passed 18 tests, 0 failures, 0 skips, and 1089 assertions. It
reported the current default SuiteSparse QR control ranks:

```text
nos4 QR solve: rank=100
bcsstk04: rank=132
west0067: rank=67
```

Day 9 changed documentation only. Required documentation validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 5 is complete or explicitly deferred. | Complete by deferral | No accepted SuiteSparse rank-deficient candidate satisfies the Day 8 gate. |
| Optional-corpus behavior is explicit and reproducible. | Complete | Future promotion contract separates default, optional-large, and report-only behavior. |
| Support-tier wording remains bounded by evidence. | Complete | Existing SuiteSparse QR matrices remain controls; no new corpus claim is added. |
