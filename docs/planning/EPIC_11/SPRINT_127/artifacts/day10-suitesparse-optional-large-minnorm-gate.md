# Sprint 127 Day 10 SuiteSparse Optional-Large Minimum-Norm Gate

## Decision

Day 10 defines the Sprint 127 SuiteSparse and optional-large minimum-norm gate.
It does not add new corpus evidence today.

The accepted SuiteSparse minimum-norm corpus baseline is still the bounded
Sprint 125 `west0067.mtx` first-30-row smoke: a 30 x 67 underdetermined system
with `b = A * ones`, residual below `1e-8`, positive solution norm, and
`||x_min|| <= ||ones|| + 1e-8`. Sprint 127 must not relabel that smoke as broad
SuiteSparse, optional-large, rank-deficient SuiteSparse, platform, performance,
or QR-vs-SVD corpus evidence.

Day 11 may accept additional SuiteSparse or optional-large minimum-norm
evidence only if the selected candidate satisfies the metadata, diagnostics,
tolerance, support-tier, skip, runtime, and duplicate-fence rules below.
Otherwise Day 11 must explicitly defer Project Plan Item 6's corpus expansion.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_127/PLAN.md` Day 10 | Defines the SuiteSparse and optional-large minimum-norm gate deliverables. |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 127 Item 6 | Requires corpus and optional-large QR/minimum-norm work to remain behind extraction, metric, support-tier, and non-claim gates. |
| Sprint 125 Day 10-12 artifacts | Define minimum-norm owners, the bounded QR-vs-SVD cross-check, and the accepted `west0067` 30 x 67 smoke. |
| Sprint 126 Day 10-11 artifacts | Define the previous SuiteSparse minimum-norm metadata protocol and explicit deferral. |
| Sprint 126 Day 12-13 artifacts | Define the larger exact 5 x 10 minimum-norm evidence lane and preserve QR-vs-SVD cross-check boundaries. |
| Sprint 127 Day 8-9 artifacts | Reconfirm that checked-in SuiteSparse rank-deficient QR corpus evidence remains deferred without independent rank/nullity metadata. |
| `tests/test_colamd.c` | Owns current owner-local minimum-norm lanes, QR-vs-SVD cross-check, and `west0067` submatrix smoke. |
| `tests/test_qr_solve.c` | Owns the exact external 2 x 4 QR minimum-norm solve anchor. |
| `tests/test_svd.c` | Owns SVD pseudoinverse behavior used only for bounded cross-checks. |
| `tests/data/suitesparse/*.mtx` | Provides checked-in corpus data and optional/report-only candidate names. |
| `docs/maintainer_guide.md` | Records current bounded QR evidence and non-claims. |

## Existing Evidence Baseline

| Lane | Owner | Status | Sprint 127 interpretation |
| --- | --- | --- | --- |
| Exact 2 x 4 QR minimum-norm | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Complete | Small exact QR solve anchor only; not SuiteSparse evidence. |
| Core owner-local minimum-norm | `tests/test_colamd.c` | Complete | COLAMD, fallback, rank-deficient, refinement, zero-row, and related owner-local fixture evidence only. |
| Exact 5 x 10 minimum-norm | `tests/test_colamd.c`, `tests/qr_external_dense_reference.py` | Complete | Larger underdetermined exact-value lane only; not corpus evidence. |
| QR-vs-SVD minimum-norm cross-check | `tests/test_colamd.c`, `tests/test_svd.c` | Complete | Bounded small-fixture cross-check; SVD is not a global QR oracle. |
| `west0067` minimum-norm submatrix smoke | `tests/test_colamd.c` | Complete | One default checked-in 30 x 67 corpus smoke; not optional-large, rank-deficient, or broad SuiteSparse proof. |

## Candidate Inventory

| Candidate | Shape/source | Support tier | Day 11 disposition |
| --- | --- | --- | --- |
| Repeat `west0067` first 30 rows | 30 x 67 underdetermined submatrix | Default checked-in | Reject as duplicate unless Day 11 adds a distinct pinned metric not already covered. |
| New `west0067` row window or wider extraction | Derived underdetermined submatrix | Default checked-in | Candidate only with extraction rule, shape, nnz, rank/nullity if claimed, residual target, norm bound, and duplicate fence. |
| `steam1.mtx` submatrix | Derived underdetermined submatrix | Default checked-in | Candidate only with extraction metadata, expected rank/nullity or explicit non-rank claim, residual/norm target, and focused runtime diagnostics. |
| `fs_541_1.mtx` submatrix | Derived underdetermined submatrix | Optional large, `SPARSE_TEST_LARGE=1` | Candidate only after optional-data present/missing proof, runtime budget, extraction metadata, rank/nullity, residual/norm target, and skip diagnostics are pinned. |
| `orsirr_1.mtx` submatrix | Derived underdetermined submatrix | Optional large, `SPARSE_TEST_LARGE=1` | Candidate only after optional-data present/missing proof, runtime budget, extraction metadata, rank/nullity, residual/norm target, and skip diagnostics are pinned. |
| Report-only matrices (`bcsstk14`, `s3rmt3m3`, `Kuu`, `bloweybq`, `Pres_Poisson`, `tuma1`) | Derived submatrices or full corpus paths | Report-only | Defer unless a future sprint promotes support tier and pins rank/nullity plus residual/norm metadata. |
| Rank-deficient SuiteSparse minimum-norm corpus | Any checked-in or optional matrix/submatrix with rank deficiency | Depends on selected matrix | Candidate only after independent rank/nullity metadata, threshold semantics, residual target, norm rule, and failure interpretation are pinned. |
| SuiteSparse QR-vs-SVD minimum-norm corpus cross-check | Any bounded corpus extraction | Depends on selected matrix | Candidate only after QR and SVD roles, tolerances, runtime, and non-oracle wording are pinned. |

## Metadata Protocol

Accepted SuiteSparse or optional-large minimum-norm evidence must record all of
the following before implementation:

1. Matrix path, extraction rule, resulting shape, nnz, and support tier.
2. RHS construction and either a named feasible vector or independent expected
   solution metadata.
3. Expected rank/nullity or explicit threshold/rank metadata when rank
   deficiency is part of the claim.
4. Residual metric, tolerance, and scaling rule.
5. Solution norm metric and comparison target: exact expected norm, named
   feasible-vector bound, or bounded QR-vs-SVD cross-check target. Product
   output alone is not an expected norm source.
6. Claim owner: SuiteSparse submatrix smoke, optional-large smoke,
   rank-deficient corpus, QR-vs-SVD cross-check, or exact-value
   underdetermined fixture.
7. Diagnostics printed before assertions.
8. Skip behavior for missing optional data and platform limitations.
9. Runtime budget and default-suite eligibility.
10. Focused validation command and full quality-gate requirement if code or
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
| Runtime | Focused executable time and default/optional gate | Must fit the declared tier before default registration. |
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

## Runtime Expectations

Default checked-in evidence must fit the normal focused owner executable and
must not make `make test` meaningfully slower. Optional-large candidates must
remain outside the default suite unless a future owner proves checked-in data,
stable runtime, deterministic skip behavior, and cross-platform suitability.
Report-only candidates are inventory only until promoted by a later sprint.

## Day 11 Candidate Decision

Day 11 may proceed only in one of these ways:

| Path | Requirements | Outcome |
| --- | --- | --- |
| Accept one bounded candidate | Satisfy the metadata protocol, support-tier policy, diagnostics, tolerance rules, duplicate fence, runtime rule, and validation commands before implementation. | Add one owner-local test or metadata-backed assertion with focused and full validation. |
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
find docs/planning/EPIC_11/SPRINT_127 -type f -name '*.md' -print0 | \
  xargs -0 awk '(/[ \t]$/){print FILENAME ":" FNR ": trailing whitespace"; bad=1} END{exit bad}'
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No corpus minimum-norm candidate proceeds without extraction and metric rules. | Complete | Metadata protocol, diagnostics/tolerance rules, and candidate table gate Day 11. |
| Optional-large work has explicit missing-data and runtime behavior. | Complete | Optional-large candidates require `SPARSE_TEST_LARGE=1`, skip proof, support-tier notes, and runtime budget before acceptance. |
| Broad minimum-norm, SuiteSparse, and platform claims remain fenced. | Complete | Existing evidence is bounded and non-claims remain explicit. |
