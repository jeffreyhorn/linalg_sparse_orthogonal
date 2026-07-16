# Sprint 126 Day 10 SuiteSparse Minimum-Norm Corpus Gate

## Decision

Day 10 defines the SuiteSparse minimum-norm corpus gate for Sprint 126 and does
not add new evidence today.

Sprint 125 already accepted one bounded default checked-in SuiteSparse
minimum-norm smoke: the first 30 rows of `west0067.mtx`, producing a 30 x 67
underdetermined system with `b = A * ones`, residual below `1e-8`, positive
solution norm, and `||x_min|| <= ||ones|| + 1e-8`. Sprint 126 must not relabel
that smoke as broad SuiteSparse, optional-large, rank-deficient, platform, or
performance evidence.

Day 11 may accept additional SuiteSparse minimum-norm evidence only if it
satisfies the metadata, diagnostics, tolerance, support-tier, and skip rules
below. Otherwise Day 11 must explicitly defer Project Plan Item 6's
SuiteSparse corpus portion.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_126/PLAN.md` Day 10 | Defines the SuiteSparse minimum-norm corpus gate deliverables. |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 126 Item 6 | Requires optional-large SuiteSparse, rank-deficient SuiteSparse, and larger underdetermined minimum-norm evidence to have pinned residual, norm, rank, nullity, corpus metadata, and exact-value ownership where justified. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day10-minnorm-behavior-owner-map.md` | Defines owner-local minimum-norm behavior lanes and helper boundaries. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day11-core-minnorm-evidence.md` | Records accepted core COLAMD, fallback, rank-deficient, refinement, and zero-row minimum-norm lanes. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day12-oracle-corpus-minnorm-decision.md` | Records the accepted `west0067` 30 x 67 submatrix smoke and deferred optional-large/rank-deficient corpus lanes. |
| `tests/test_colamd.c` | Owns current minimum-norm behavior, QR-vs-SVD cross-check, and SuiteSparse submatrix smoke. |
| `tests/test_qr_solve.c` | Owns the focused exact external 2x4 minimum-norm QR solve fixture. |
| `tests/test_svd.c` | Owns SVD pseudoinverse behavior used only as bounded cross-check evidence. |
| `tests/data/suitesparse/*.mtx` | Checked-in corpus and support-tier inventory. |

## Existing Evidence Baseline

| Lane | Owner | Status | Sprint 126 interpretation |
| --- | --- | --- | --- |
| Exact 2x4 QR minimum-norm | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Complete | Exact small-fixture anchor only; not SuiteSparse evidence. |
| Core owner-local minimum-norm | `tests/test_colamd.c` | Complete | COLAMD, fallback, rank-deficient, refinement, and zero-row fixture evidence only. |
| QR-vs-SVD minimum-norm cross-check | `tests/test_colamd.c`, `tests/test_svd.c` | Complete | One bounded 2x4 cross-check; SVD is not a global QR oracle. |
| `west0067` minimum-norm submatrix smoke | `tests/test_colamd.c` | Complete | One default checked-in 30 x 67 corpus smoke; not optional-large, not rank-deficient SuiteSparse corpus proof. |

## Candidate Inventory

| Candidate | Shape/source | Support tier | Day 11 disposition |
| --- | --- | --- | --- |
| Repeat `west0067` first 30 rows | 30 x 67 underdetermined submatrix | Default checked-in | Reject as duplicate unless Day 11 adds a distinct pinned metric not already covered. |
| New `west0067` row window or wider extraction | Derived underdetermined submatrix | Default checked-in | Candidate only with extraction rule, rank/nullity, residual target, norm bound, and duplicate-fence against the existing first-30-row smoke. |
| `steam1.mtx` submatrix | Derived underdetermined submatrix | Default checked-in | Candidate only with extraction rule, rank/nullity, residual/norm metadata, and focused runtime diagnostics. |
| `fs_541_1.mtx` submatrix | Derived underdetermined submatrix | Optional large, `SPARSE_TEST_LARGE=1` | Candidate only after optional-data present/missing behavior, runtime budget, rank/nullity, residual/norm metadata, and skip diagnostics are pinned. |
| `orsirr_1.mtx` submatrix | Derived underdetermined submatrix | Optional large, `SPARSE_TEST_LARGE=1` | Candidate only after optional-data present/missing behavior, runtime budget, rank/nullity, residual/norm metadata, and skip diagnostics are pinned. |
| Report-only matrices (`bcsstk14`, `s3rmt3m3`, `Kuu`, `bloweybq`, `Pres_Poisson`, `tuma1`) | Derived submatrices or full corpus paths | Report-only for Day 11 | Defer unless a future sprint promotes support tier and pins rank/nullity plus residual/norm metadata. |
| Rank-deficient SuiteSparse minimum-norm corpus | Any checked-in or optional matrix/submatrix with rank deficiency | Depends on selected matrix | Candidate only after independent rank/nullity metadata, threshold semantics, residual target, norm rule, and failure interpretation are pinned. |

## Metadata Protocol

Accepted SuiteSparse minimum-norm evidence must record all of the following
before implementation:

1. Matrix path, extraction rule, resulting shape, nnz, and support tier.
2. RHS construction and whether a named feasible vector is known.
3. Expected rank, nullity, or explicit threshold/rank metadata when rank or
   rank deficiency is part of the claim.
4. Residual metric and tolerance.
5. Solution norm metric and comparison target: exact expected norm,
   comparison against a named feasible vector, or bounded QR-vs-SVD
   cross-check. Product output alone is not an expected norm source.
6. Claim owner: SuiteSparse submatrix smoke, optional-large smoke,
   rank-deficient corpus, QR-vs-SVD cross-check, or exact-value
   underdetermined fixture.
7. Diagnostics printed before assertions.
8. Skip behavior for missing optional data and platform limitations.
9. Focused validation command and full quality-gate requirement if code or
   headers change.

If any required field is missing, Day 11 must defer the candidate instead of
weakening the claim.

## Diagnostics And Tolerance Rules

| Quantity | Required diagnostic | Acceptance rule |
| --- | --- | --- |
| Corpus identity | Matrix key, path, support tier, extraction rule, shape, nnz | Must match artifact metadata. |
| Load and gate state | Load status, optional gate value, skip reason if skipped | Default missing data fails; optional missing data may skip only with named gate and owner. |
| Rank/nullity | Expected rank/nullity and threshold when claimed; observed rank if measured | Required for rank-deficient claims; optional for pure smoke lanes if rank is not claimed. |
| Residual | Absolute max residual or relative residual with scale rule | Must have fixture-local tolerance before assertions. |
| Norm | `||x_min||` and comparison target | Must compare against exact value, named feasible vector, or bounded cross-check target. |
| Failure interpretation | Numeric disagreement, load failure, solve failure, skip | Numeric disagreement after acceptance fails; skip is allowed only for explicitly optional missing data or proven platform blockers. |

Default tolerance starting points:

| Lane | Residual tolerance | Norm tolerance | Notes |
| --- | ---: | ---: | --- |
| Default checked-in submatrix smoke | `1e-8` absolute max residual | `1e-8` against named feasible-vector bound | Matches current `west0067` smoke unless Day 11 proves a tighter fixture-local bound. |
| Optional-large submatrix smoke | Candidate-specific | Candidate-specific | Must include runtime and skip-path proof. |
| Rank-deficient SuiteSparse minimum-norm | Candidate-specific | Candidate-specific | Must pin rank/nullity and threshold before registration. |
| QR-vs-SVD corpus cross-check | Candidate-specific | Candidate-specific | Must preserve SVD as bounded cross-check, not oracle. |

## Support-Tier And Skip Policy

| Tier | Policy |
| --- | --- |
| Default checked-in | Missing data is a failure. Accepted numerical disagreement is a failure. |
| Optional large | Use `SPARSE_TEST_LARGE=1` unless Day 11 defines a narrower QR minimum-norm gate. Missing data may skip only when the skip message names the matrix, gate, and owner. |
| Report-only | Not eligible for default Day 11 evidence without future support-tier promotion. |
| Platform | Pure C minimum-norm tests should not inherit Python-helper Windows skips. Any platform skip must name the concrete blocker. |

## Day 11 Candidate Decision

Day 11 may proceed only in one of these ways:

| Path | Requirements | Outcome |
| --- | --- | --- |
| Accept one bounded candidate | Satisfy the metadata protocol, support-tier policy, diagnostics, tolerance rules, duplicate fence, and validation commands before implementation. | Add one owner-local test or metadata-backed assertion with focused and full validation. |
| Explicitly defer all candidates | Show the missing metadata or support-tier blocker for each candidate. | Preserve the existing `west0067` smoke and publish future-owner promotion gates. |

The preferred Day 11 default is deferral unless an independent extraction,
rank/nullity, residual, and norm contract is available for a non-duplicate
candidate.

## Non-Claims

Day 10 does not prove:

- new SuiteSparse minimum-norm evidence;
- optional-large SuiteSparse minimum-norm behavior;
- rank-deficient SuiteSparse minimum-norm behavior;
- broad QR minimum-norm optimality;
- SVD pseudoinverse as a global QR oracle;
- broad SuiteSparse corpus correctness, platform support, or performance;
- LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK, vendor-backend,
  dense-library, ecosystem, or external package parity;
- COLAMD, fallback, refinement, nullspace, Q-basis, economy, sparse-mode,
  reorder, backend, package, ABI, public API, CI, CMake, CTest, scalability,
  memory, or state-of-the-art behavior.

## Validation

Day 10 changed documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_126
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 6 has clear SuiteSparse acceptance criteria. | Complete | Metadata protocol, diagnostics/tolerance rules, support-tier policy, and Day 11 candidate decision define the gate. |
| Optional-large behavior is documented before implementation. | Complete | Optional-large candidates require gate, missing-data skip proof, runtime budget, rank/nullity, residual/norm metadata, and diagnostics. |
| No broad minimum-norm or external parity claim is introduced. | Complete | Existing evidence is fenced and non-claims are explicit. |
