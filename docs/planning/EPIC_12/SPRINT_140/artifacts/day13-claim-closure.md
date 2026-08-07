# Day 13 Claim Closure

## Closed Claim

Sprint 140 closes the selected residual for fixture-local partial-SVD behavior
on `partial_svd_clustered_repeated_diag8x6_k3_v1`.

The closed claim is:

> For the generated 8x6 clustered/repeated diagonal fixture with `k = 3`, the
> maintained local partial-SVD lane verifies top-3 singular values, left and
> right top-k subspace projectors, triplet residuals, orthogonality,
> default-budget success, tight-budget fail-closed behavior, and no partial
> `sigma`, `U`, or `Vt` arrays on tight-budget failure.

This is a fixture-local correctness and fail-closed claim. It is not a broad
partial-SVD correctness, convergence-rate, performance, platform, package, ABI,
external-library parity, partial-result, or state-of-the-art claim.

## Deliverable Summary

| Surface | Deliverable | Evidence |
| --- | --- | --- |
| Residual selection | Selected the clustered/repeated top-k subspace and budget residual for complete closure. | Day 1 and Day 2 artifacts. |
| Fixture design | Defined `partial_svd_clustered_repeated_diag8x6_k3_v1`, generator parameters, expected rows, comparison semantics, and claim boundary. | Day 3 and Day 4 artifacts. |
| Corpus fixture | Added fixture and generator manifest rows plus expected-result TSV rows for values, subspaces, residuals, orthogonality, and budget diagnostics. | Day 5 artifact and corpus TSV changes. |
| Oracle semantics | Added opt-in partial-SVD generated-reference rows and parser support for value, subspace, residual, status, and diagnostic comparisons. | Day 6 and Day 7 artifacts plus `scripts/run_corpus_oracle.py`. |
| Compiled proof owner | Added `tests/test_svd_partial_corpus.c` and registered it in Make/CMake. | Day 8 and Day 9 artifacts plus build-file changes. |
| Helper cleanup | Moved reusable residual/projector helpers into a focused shared helper header. | Day 10 artifact plus `tests/test_svd_partial_shared_helpers.h`. |
| Public wording | Published earned fixture-local wording across README, SVD API, corpus, tutorial, cookbook, solver-selection, algorithm, examples, and maintainer docs. | Day 11 artifact and documentation changes. |
| Validation | Ran corpus, oracle, focused test, source-list, CMake parity, full quality gate, and hygiene checks. | Day 12 artifact. |

## Validation-To-Claim Traceability

| Claim element | Proof surface | Validation evidence |
| --- | --- | --- |
| Generated 8x6 clustered/repeated diagonal fixture | `tests/corpus/manifests/fixtures.tsv`, `tests/corpus/manifests/generators.tsv`, and `scripts/validate_corpus_schema.py` | `python3 scripts/validate_corpus_schema.py` passed. |
| Top-3 singular values | `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv`, `scripts/run_corpus_oracle.py`, and `tests/test_svd_partial_corpus.c` | Oracle generation passed; focused test target passed 5 tests and 107 assertions. |
| Repeated-spectrum subspace comparison without raw vector identity | `tests/test_svd_partial_shared_helpers.h` and `tests/test_svd_partial_corpus.c` | Focused projector tests passed; docs retain raw singular-vector identity as a non-claim. |
| Triplet residuals and orthogonality | `tests/test_svd_partial_shared_helpers.h` and `tests/test_svd_partial_corpus.c` | Focused residual and orthogonality tests passed. |
| Default-budget success | `tests/test_svd_partial_corpus.c` and oracle expected status row | Focused default-success test passed; opt-in oracle row generated. |
| Tight-budget fail-closed behavior | `src/sparse_svd_partial.c`, `tests/test_svd_partial_corpus.c`, and oracle expected status/diagnostic rows | Focused tight-budget test passed and verifies no partial arrays on failure. |
| Build ownership | `Makefile` and `CMakeLists.txt` | `make source-list-check` passed; `make quality-review-cmake-compile` registered 59 CMake tests matching 59 Makefile tests. |
| Full changed-code quality gate | All changed `.c` and `.h` surfaces | `make format && make lint && make test` passed. |
| Claim wording hygiene | README, API docs, corpus docs, maintainer guide, and Sprint 140 artifacts | `git diff --check`, targeted whitespace scan, TSV width check, and markdown link check passed. |

## Remaining SVD And Partial-SVD Non-Claims

Sprint 140 intentionally leaves these non-claims visible:

- broad SVD or partial-SVD correctness;
- raw singular-vector identity for repeated or clustered singular values;
- broad repeated-spectrum behavior beyond the named fixture;
- broad rectangular, nonsymmetric, rank-deficient, null-space, pseudoinverse,
  or minimum-norm behavior;
- sparse-output or drop-tolerance optimality;
- convergence-rate behavior or portable iteration counts;
- partial-result guarantees on non-convergence;
- external-library parity with LAPACK, NumPy, SciPy, SuiteSparse, ARPACK, or
  other third-party solvers;
- platform parity or reviewed hosted-platform promotion;
- package, install, shared-library, or ABI claims;
- performance claims; and
- state-of-the-art status.

## Sprint 141 Report-Index Handoff

Sprint 141 should start from the following report-index requirements instead
of rediscovering them:

1. Normalize report-index rows so generated-reference, solver-backed, skip,
   stale, and unsupported rows have explicit status semantics.
2. Add freshness metadata for commit, branch, command, platform, compiler,
   configuration, generator hashes, expected-row hashes, and support tier.
3. Make stale-report detection a first-class failure class, especially when a
   partial-SVD row is cited from an older source commit or mismatched generated
   fixture hash.
4. Keep generated oracle and report files under ignored `build/` paths unless a
   future sprint deliberately promotes a reviewed artifact.
5. Prevent optional-data, skip, defer, or generated-reference rows from being
   interpreted as hosted-platform or solver-backed pass evidence.
6. Preserve fixture-local wording for
   `partial_svd_clustered_repeated_diag8x6_k3_v1` until reviewed evidence
   expands the claim boundary.

## Closeout Readiness

Day 13 is ready for Day 14 closeout because the claim, evidence, validation,
remaining non-claims, and Sprint 141 report-index handoff are now captured in
one artifact. Day 14 should inventory the full Sprint 140 artifact set and cite
the latest passing Day 12 validation command set unless additional source or
test changes require rerunning the full gate.
