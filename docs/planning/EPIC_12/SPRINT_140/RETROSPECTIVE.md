# Sprint 140 Retrospective

**Sprint:** 140 - Partial-SVD Edge-Case & Convergence Residual Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-140`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 140 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Re-read Sprint 138 corpus architecture, Sprint 139 QR closure handoff,
      current SVD/partial-SVD tests, corpus manifests, oracle scripts, public
      SVD docs, and maintainer guidance.
- [x] Re-audited partial-SVD residual candidates and selected one bounded
      priority residual:
      `partial_svd_clustered_repeated_diag8x6_k3_v1`.
- [x] Added deterministic generated corpus fixture metadata for an 8 by 6
      clustered/repeated diagonal matrix with `k = 3`.
- [x] Added expected-result rows for top-3 singular values, left and right
      top-k subspace projectors, triplet residuals, orthogonality,
      default-budget success, tight-budget non-convergence, and no partial
      arrays on tight-budget failure.
- [x] Extended corpus schema validation with the new partial-SVD generator and
      data-driven known-generator parameter checks.
- [x] Extended `scripts/run_corpus_oracle.py` with opt-in partial-SVD
      generated-reference rows behind `--include-partial-svd`.
- [x] Added comparison support for `value`, `subspace_distance`,
      `residual_norm`, `status`, and `diagnostic` rows while preserving QR
      rank/nullity/residual behavior.
- [x] Kept generated-reference rows distinct from solver-backed proof evidence:
      the partial-SVD oracle rows remain `support_tier=local_only` and
      generated outputs remain under ignored `build/` paths.
- [x] Added the focused C proof owner `tests/test_svd_partial_corpus.c`.
- [x] Registered `test_svd_partial_corpus` in both `Makefile` and
      `CMakeLists.txt`.
- [x] Added `tests/test_svd_partial_shared_helpers.h` and moved reusable
      partial-SVD residual/projector helper logic behind that focused helper
      surface.
- [x] Updated README, public SVD API comments, corpus docs, tutorial,
      cookbook, solver-selection, algorithm, examples, and maintainer docs
      with earned fixture-local partial-SVD wording.
- [x] Published claim closure, remaining partial-SVD non-claims, and Sprint 141
      report-index freshness/staleness handoff requirements.
- [x] Ran Sprint 140 focused and full validation:
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 scripts/run_corpus_oracle.py --include-partial-svd`;
  - `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus`;
  - `make source-list-check`;
  - `make quality-review-cmake-compile`;
  - `python3 -m py_compile scripts/validate_corpus_schema.py
    scripts/run_corpus_oracle.py`;
  - TSV width checks;
  - generated-artifact ignored/untracked checks;
  - `git diff --check`;
  - trailing-whitespace scans;
  - focused Markdown relative-link validation;
  - `make format && make lint && make test`.
- [x] The required full C quality gate passed because Sprint 140 modified
      `.c` and `.h` files.

## What Went Well

1. **The sprint closed one partial-SVD residual completely.** The selected
   generated fixture now has corpus metadata, expected rows, oracle semantics,
   a compiled proof owner, public wording, validation evidence, and explicit
   non-claims.

2. **The comparison semantics matched the math.** Projector/subspace checks
   replaced raw singular-vector identity for the clustered/repeated leading
   singular-value block, allowing valid basis rotations while still catching
   wrong subspaces.

3. **The convergence-budget boundary is now explicit.** The focused proof owner
   verifies default-budget success and tight-budget fail-closed behavior,
   including no visible partial `sigma`, `U`, or `Vt` arrays on failure.

4. **The proof owner stayed narrow.** `tests/test_svd_partial_corpus.c` owns
   the corpus-backed closure without turning the broad `tests/test_svd.c`
   surface into another mixed-purpose fixture registry.

5. **Documentation moved from planned wording to earned wording.** Public and
   maintainer docs now describe exactly what the Sprint 140 fixture proves and
   retain broad SVD/partial-SVD non-claims.

## What Didn't Go Well

1. **Generated-reference rows remain easy to over-interpret.** The oracle path
   is useful for corpus/report semantics, but its partial-SVD rows are not
   solver-backed hosted-platform evidence. Sprint 141 needs stronger report
   index status/freshness normalization.

2. **The fixture touches several ownership surfaces.** Closing one residual
   required coordinated changes across schema validation, manifests, expected
   rows, oracle scripts, tests, helper headers, build registration, and docs.
   The Day 12 and Day 14 validation artifacts are essential for avoiding drift.

3. **The sprint deliberately left adjacent partial-SVD gaps open.** Broad
   repeated-spectrum behavior, rank-deficient null-space behavior,
   sparse-output low-rank optimality, convergence-rate claims, and
   external-library parity remain deferred.

4. **The report freshness model is still mostly interpretive.** The sprint
   documents stale-report signals and support-tier boundaries, but automatic
   freshness enforcement is Sprint 141 work.

## Final Metrics

### Validation

| Metric | Sprint 140 close state |
| --- | --- |
| tracked `.c`/`.h` changes | yes: partial-SVD implementation, public header, focused test owner, and helper headers |
| `python3 scripts/validate_corpus_schema.py` | passed |
| `python3 scripts/run_corpus_oracle.py --include-partial-svd` | passed |
| focused Make `test_svd_partial_corpus` | passed: 5 tests, 0 failures, 0 skips, 107 assertions |
| CMake registration/parity | passed: 59 CMake tests matched 59 Makefile tests |
| source-list parity | passed: 49 library sources |
| full C quality gate | passed: `make format && make lint && make test` |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |
| focused Markdown relative-link validation | passed |
| generated corpus/report files tracked | no |
| generated corpus/report files ignored | yes |

### Artifact Package

| Metric | Sprint 140 close state |
| --- | ---: |
| daily artifacts under `SPRINT_140/artifacts/` | 14 |
| final retrospective files | 1 |
| focused partial-SVD corpus test files added | 1 |
| partial-SVD shared helper headers added | 1 |
| build-system surfaces changed | 2 |
| public/maintainer documentation surfaces changed | 9 |
| corpus manifest files changed | 2 |
| expected-result TSV files added | 1 |
| oracle/schema script surfaces changed | 2 |
| source-controlled generated oracle/report files | 0 |

## Closed Claim

Sprint 140 closes this claim:

For the maintained generated 8 by 6 clustered/repeated diagonal partial-SVD
corpus fixture `partial_svd_clustered_repeated_diag8x6_k3_v1`, the project
partial-SVD implementation verifies top-3 singular values, left and right
top-k subspace projectors, triplet residuals, orthogonality, default-budget
success, tight-budget fail-closed behavior, and no partial `sigma`, `U`, or
`Vt` arrays on tight-budget failure.

This claim is supported by:

- `tests/test_svd_partial_corpus.c`;
- `tests/test_svd_partial_shared_helpers.h`;
- `tests/corpus/expected/partial_svd_clustered_repeated_diag8x6_k3_v1.tsv`;
- `tests/corpus/manifests/fixtures.tsv`;
- `tests/corpus/manifests/generators.tsv`;
- `scripts/validate_corpus_schema.py`;
- `scripts/run_corpus_oracle.py --include-partial-svd`;
- `include/sparse_svd.h`;
- `tests/corpus/README.md`;
- `docs/maintainer_guide.md`;
- Day 12 and Day 14 validation evidence.

## Sprint 141 Readiness

Sprint 141 should use Sprint 140's report-index handoff as its starting point:

| Handoff field | Sprint 141 requirement |
| --- | --- |
| status normalization | Give generated-reference, solver-backed, skip, stale, and unsupported rows explicit status semantics. |
| freshness metadata | Record commit, branch, command, platform, compiler, configuration, generator hashes, expected-row hashes, and support tier. |
| stale-report detection | Treat stale source commits or mismatched fixture/generated hashes as first-class failures. |
| generated-output policy | Keep generated oracle/report artifacts under ignored `build/` paths unless a future sprint deliberately promotes reviewed artifacts. |
| support-tier interpretation | Prevent optional-data, skip, defer, and generated-reference rows from being interpreted as hosted-platform or solver-backed pass evidence. |
| claim boundary | Preserve fixture-local wording for `partial_svd_clustered_repeated_diag8x6_k3_v1` until reviewed evidence expands the boundary. |

## Residual Deferred Debt

Most important carry-forward work:

- Sprint 141 report freshness normalization and stale-report diagnostics;
- broad partial-SVD correctness across arbitrary spectra and shapes;
- broad repeated-spectrum behavior beyond the named fixture;
- broad rectangular, nonsymmetric, rank-deficient, null-space, pseudoinverse,
  and minimum-norm behavior;
- sparse-output and drop-tolerance low-rank optimality;
- convergence-rate or portable iteration-count evidence;
- partial-result guarantees after non-convergence;
- external-library parity with LAPACK, NumPy, SciPy, SuiteSparse, ARPACK, or
  other third-party solvers;
- hosted-platform promotion for partial-SVD corpus rows;
- package, install, shared-library, or ABI claims;
- performance claims;
- state-of-the-art claims.

Still consciously constrained rather than silently solved:

- no broad SVD or partial-SVD correctness claim;
- no raw singular-vector identity claim for repeated or clustered singular
  values;
- no broad external-library parity claim;
- no broad corpus completeness claim;
- no hosted platform parity claim;
- no package, ABI, shared-library, loader, or package-manager support claim;
- no portable performance claim;
- no state-of-the-art claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-partial-svd-residual-intake.md](./artifacts/day1-partial-svd-residual-intake.md)
- [day2-partial-svd-residual-reaudit.md](./artifacts/day2-partial-svd-residual-reaudit.md)
- [day3-partial-svd-closure-design.md](./artifacts/day3-partial-svd-closure-design.md)
- [day4-partial-svd-fixture-batch-design.md](./artifacts/day4-partial-svd-fixture-batch-design.md)
- [day5-partial-svd-fixture-implementation.md](./artifacts/day5-partial-svd-fixture-implementation.md)
- [day6-partial-svd-comparison-semantics.md](./artifacts/day6-partial-svd-comparison-semantics.md)
- [day7-partial-svd-oracle-implementation.md](./artifacts/day7-partial-svd-oracle-implementation.md)
- [day8-partial-svd-convergence-proof-design.md](./artifacts/day8-partial-svd-convergence-proof-design.md)
- [day9-partial-svd-proof-implementation.md](./artifacts/day9-partial-svd-proof-implementation.md)
- [day10-proof-owner-cleanup.md](./artifacts/day10-proof-owner-cleanup.md)
- [day11-documentation-update.md](./artifacts/day11-documentation-update.md)
- [day12-validation-pass.md](./artifacts/day12-validation-pass.md)
- [day13-claim-closure.md](./artifacts/day13-claim-closure.md)
- [day14-closeout-validation-summary.md](./artifacts/day14-closeout-validation-summary.md)

## Closeout

Sprint 140 is complete. It closes the selected partial-SVD edge-case and
convergence residual with a fixture-local corpus-backed evidence lane,
opt-in generated-reference oracle/report rows, a focused compiled proof owner,
updated public and maintainer claim wording, final validation evidence,
explicit remaining SVD/partial-SVD non-claims, and a Sprint 141 report-index
handoff. It does not promote generated reports into source control or widen
public claims beyond the named fixture-local partial-SVD behavior.
