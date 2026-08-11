# Sprint 151 Retrospective

**Sprint:** 151 - Partial-SVD Maintained Corpus Family Expansion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-151`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 151 day-by-day plan, working notes, artifact directory,
      closeout artifact, and retrospective.
- [x] Audited existing partial-SVD source, tests, helpers, corpus metadata,
      expected-result rows, oracle/report generation, and documentation claim
      boundaries.
- [x] Selected a bounded partial-SVD fixture-family expansion:
      rank-deficient rectangular range projectors, sparse low-rank output, and
      non-repeated fail-closed convergence behavior.
- [x] Deferred optional external dense-reference vector-residual fixtures
      because optional data, provenance, Windows skip behavior, and broad
      parity wording would widen the sprint surface.
- [x] Deferred additional repeated-spectrum fixtures beyond the Sprint 140
      seed because the existing clustered/repeated fixture already carries the
      strongest current repeated-spectrum local evidence.
- [x] Defined subspace-safe comparison semantics for singular values,
      projectors, residuals, orthogonality, sparse-output checks, and
      fail-closed behavior without raw singular-vector identity claims.
- [x] Added source-controlled fixture rows and generator rows for three new
      maintained partial-SVD fixtures.
- [x] Added source-controlled expected-result TSV files for rank, singular
      values, projector distances, residuals, orthogonality, sparse-output
      shape/values, dense error, sparse-vs-dense consistency, fail-closed
      status, and no-partial-array diagnostics.
- [x] Extended deterministic corpus generator validation for the selected
      partial-SVD fixtures.
- [x] Extended `tests/test_svd_partial_corpus.c` as the focused partial-SVD
      corpus proof owner.
- [x] Extended generated-local partial-SVD oracle/report generation and
      normalized report-index tests for the selected rows.
- [x] Updated README, algorithm docs, cookbook guidance, solver-selection docs,
      maintainer guidance, corpus docs, expected-result docs, and oracle-schema
      docs to name the bounded partial-SVD family and non-claims.
- [x] Ran integrated schema, focused partial-SVD proof-owner, affected SVD,
      oracle/report, report-index, stale-claim, whitespace, and full C quality
      gates.
- [x] Prepared the Sprint 152 generated report freshness handoff.

## What Went Well

1. **The partial-SVD corpus expanded without becoming a broad SVD claim.**
   Sprint 151 moved from the single Sprint 140 fixture to a four-fixture
   maintained partial-SVD corpus family, but the evidence remains explicitly
   fixture-local and local-only.

2. **Family selection stayed small enough to close completely.** Days 1-5
   separated intake, candidate scoring, family selection, comparison-contract
   design, and metadata design before rows or proof owners were added.

3. **The comparison contract avoided brittle vector identity.** The sprint
   used sorted singular values, projector distances, residual norms,
   orthogonality checks, sparse-output diagnostics, and fail-closed status
   rows instead of raw singular-vector equality, sign parity, basis orientation,
   phase, or arbitrary basis ordering.

4. **The proof owner stayed focused.** `tests/test_svd_partial_corpus.c` now
   owns the maintained partial-SVD corpus proof surface with fixture-keyed
   diagnostics and direct assertions for the selected expected rows.

5. **Report integration matched the evidence boundary.** Generated-local
   partial-SVD oracle rows are normalized and counted, but they remain
   `local_only` and do not become hosted CI, package, ABI, performance, or
   release-artifact proof.

6. **Documentation moved with the corpus.** Current docs now describe the
   four-fixture partial-SVD family and `26` generated-local partial-SVD oracle
   rows instead of the old Sprint 140-only shape.

## What Didn't Go Well

1. **Generated-row freshness remains advisory.** The freshness check passes,
   but generated-local oracle rows still emit expected
   `generated_present_unchecked` warnings because strict generated-row
   comparison has not yet been promoted.

2. **External dense-reference fixtures stayed outside the maintained corpus.**
   Existing owner-local tests are useful, but promoting optional external-data
   fixtures would require clearer provenance, platform behavior, and
   parity-wording rules.

3. **Repeated-spectrum expansion was intentionally deferred.** The Sprint 140
   clustered/repeated fixture remains the strongest repeated-spectrum seed.
   More repeated fixtures should be added only when they carry distinct claim
   value rather than duplicating the seed.

4. **Sparse-output evidence is deliberately narrow.** The sprint proves
   deterministic selected sparse low-rank output behavior for one fixture at
   `drop_tol=0`. It does not prove broad sparse-output optimality or
   drop-tolerance policy.

5. **The review surface is coordinated and non-trivial.** The sprint touched
   C tests, Python generator/schema/report scripts, corpus metadata, expected
   rows, and multiple docs. The integrated validation passed, but reviewers
   should evaluate the changes as one corpus-product update.

## Final Metrics

### Validation

| Metric | Sprint 151 close state |
| --- | --- |
| tracked `.c` changes | yes: `tests/test_svd_partial_corpus.c` |
| tracked `.h` changes | no |
| full C quality gate required | yes |
| focused partial-SVD proof owner | passed: 10 tests, 0 failed, 0 skipped, 247 assertions |
| affected broader SVD proof owner | passed: 114 tests, 0 failed, 0 skipped, 2067 assertions |
| corpus schema validation | passed |
| partial-SVD oracle generation | passed: `python3 scripts/run_corpus_oracle.py --include-partial-svd` |
| total generated oracle rows | 29 |
| generated-local partial-SVD oracle rows | 26 |
| report-index normalization | passed: `105` rows ok |
| oracle freshness check | passed: freshness ok for `31` rows with expected generated-local advisory warnings |
| report-index unit test | passed |
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| `git diff --check` | passed |
| active-doc stale wording search | passed; historical planning artifact references remain as baseline/search text |

### Artifact Package

| Metric | Sprint 151 close state |
| --- | ---: |
| daily artifacts under `SPRINT_151/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source files changed | 1 |
| script files changed | 2 |
| public/support docs changed | 8 |
| corpus manifest files changed | 2 |
| corpus schema docs changed | 1 |
| new expected-result TSV files | 3 |
| selected maintained partial-SVD fixtures | 4 |
| new maintained partial-SVD fixtures | 3 |
| generated-local partial-SVD oracle rows | 26 |

## Closed Claim

Sprint 151 closes this local claim:

The maintained partial-SVD corpus has been expanded from the Sprint 140 seed
fixture to a bounded four-fixture family covering selected clustered/repeated,
rank-deficient rectangular, sparse low-rank output, and non-repeated
fail-closed convergence behaviors, with source-controlled metadata, expected
rows, focused proof-owner tests, generated-local oracle/report rows,
normalized report-index checks, and aligned documentation.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-partial-svd-intake.md](./artifacts/day1-partial-svd-intake.md);
- [day2-partial-svd-family-candidate-audit.md](./artifacts/day2-partial-svd-family-candidate-audit.md);
- [day3-family-selection-claim-scope.md](./artifacts/day3-family-selection-claim-scope.md);
- [day4-comparison-contract-design.md](./artifacts/day4-comparison-contract-design.md);
- [day5-metadata-design.md](./artifacts/day5-metadata-design.md);
- [day6-metadata-batch.md](./artifacts/day6-metadata-batch.md);
- [day7-oracle-data-implementation.md](./artifacts/day7-oracle-data-implementation.md);
- [day8-proof-owner-test-design.md](./artifacts/day8-proof-owner-test-design.md);
- [day9-proof-owner-test-implementation.md](./artifacts/day9-proof-owner-test-implementation.md);
- [day10-report-integration-design.md](./artifacts/day10-report-integration-design.md);
- [day11-report-integration-implementation.md](./artifacts/day11-report-integration-implementation.md);
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-and-sprint-152-handoff.md](./artifacts/day14-closeout-and-sprint-152-handoff.md).

## Next-Sprint Readiness

Sprint 152 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| Generated family selection | Decide which generated families carry current claims and which remain advisory. |
| Partial-SVD oracle rows | Treat the `26` generated-local partial-SVD rows as local-only until freshness policy changes. |
| Strict generated freshness | Decide whether partial-SVD generated oracle rows should become required via `--require-generated`, strict freshness, hosted CI, or local-only checks. |
| Report metadata | Stabilize command, commit, branch, platform, compiler, configuration, support tier, artifact path, row count, and failure-message fields before promotion. |
| Documentation | Keep generated-local evidence wording bounded unless Sprint 152 promotes a stronger report-freshness policy. |
| Non-claims | Preserve no hosted CI, package, ABI, performance, external-library parity, or state-of-the-art inference from generated-local rows. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 151 close:

- strict generated-row freshness comparison for generated-local oracle rows;
- external dense-reference partial-SVD corpus promotion;
- additional repeated-spectrum partial-SVD fixture families;
- broad partial-SVD correctness;
- raw singular-vector identity, sign, orientation, phase, or basis-order
  parity;
- broad sparse-output/drop-tolerance optimality;
- convergence-rate guarantees or portable iteration-count guarantees;
- external-library parity;
- hosted CI proof for generated-local oracle rows;
- platform, package, ABI, performance, installed-consumer, or
  state-of-the-art proof for partial-SVD behavior.

Still consciously constrained rather than silently solved:

- generated-local `build/` oracle/report rows are not release artifacts;
- source-controlled fixture rows remain local-only corpus evidence;
- selected sparse-output checks do not imply broad sparse-matrix compression
  behavior;
- selected fail-closed behavior does not imply broad convergence policy;
- numerical corpus proof does not imply packaging/platform support.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-partial-svd-intake.md](./artifacts/day1-partial-svd-intake.md)
- [day2-partial-svd-family-candidate-audit.md](./artifacts/day2-partial-svd-family-candidate-audit.md)
- [day3-family-selection-claim-scope.md](./artifacts/day3-family-selection-claim-scope.md)
- [day4-comparison-contract-design.md](./artifacts/day4-comparison-contract-design.md)
- [day5-metadata-design.md](./artifacts/day5-metadata-design.md)
- [day6-metadata-batch.md](./artifacts/day6-metadata-batch.md)
- [day7-oracle-data-implementation.md](./artifacts/day7-oracle-data-implementation.md)
- [day8-proof-owner-test-design.md](./artifacts/day8-proof-owner-test-design.md)
- [day9-proof-owner-test-implementation.md](./artifacts/day9-proof-owner-test-implementation.md)
- [day10-report-integration-design.md](./artifacts/day10-report-integration-design.md)
- [day11-report-integration-implementation.md](./artifacts/day11-report-integration-implementation.md)
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md)
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md)
- [day14-closeout-and-sprint-152-handoff.md](./artifacts/day14-closeout-and-sprint-152-handoff.md)
