# Sprint 150 Retrospective

**Sprint:** 150 - QR Maintained Corpus Family Expansion
**Duration:** 14 days (Days 1-14 landed on branch `sprint-150`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 150 day-by-day plan, working notes, artifact directory,
      closeout artifact, and retrospective.
- [x] Audited existing QR proof surfaces, corpus rows, oracle/report behavior,
      and documentation claim boundaries.
- [x] Selected bounded QR fixture families for complete closure:
      rank-deficient rectangular QR and underdetermined minimum-norm QR.
- [x] Deferred reorder/COLAMD QR corpus promotion because its evidence mixes
      residual/status, permutation, fill, optional SuiteSparse, and
      performance-adjacent semantics.
- [x] Added source-controlled fixture rows and generator rows for five new QR
      maintained corpus fixtures.
- [x] Added source-controlled expected-result TSV files for the new rank,
      nullity, nullspace residual, projector/subspace, minimum-norm status,
      residual, solution norm, and exact-value rows.
- [x] Extended deterministic corpus generator validation for the selected QR
      fixtures.
- [x] Defined QR oracle semantics without raw QR basis or raw nullspace basis
      identity claims.
- [x] Extended `tests/test_qr_corpus.c` as the focused QR corpus proof owner.
- [x] Extended local QR oracle/report generation and reset stale generated
      oracle/report outputs before each run.
- [x] Updated README, algorithm docs, cookbook guidance, maintainer guidance,
      corpus docs, and oracle-schema docs to name the bounded QR family and
      non-claims.
- [x] Ran integrated schema, focused QR proof-owner, oracle/report,
      report-index, stale-claim, whitespace, and full C quality gates.
- [x] Prepared the Sprint 151 partial-SVD maintained corpus handoff.

## What Went Well

1. **The QR claim expanded without becoming broad.** Sprint 150 moved from the
   single Sprint 139 QR fixture to a six-fixture maintained QR corpus family,
   but the claim stayed fixture-local and avoided broad QR correctness,
   raw-basis parity, rank-policy, platform, package, ABI, performance, and
   state-of-the-art claims.

2. **Family selection happened before implementation.** Days 1-4 separated
   intake, candidate scoring, family selection, and metadata design before
   source-controlled rows were added. That kept the final family small enough
   to close completely.

3. **The proof owner stayed focused.** Sprint 150 extended
   `tests/test_qr_corpus.c` rather than growing the largest monolithic QR test
   file. The focused lane now proves 14 QR corpus tests with fixture-key
   diagnostics.

4. **Generated-local report behavior became more reliable.** Day 11 fixed the
   oracle generator so stale ignored `build/` oracle/report files are removed
   before the current QR run writes new output. That prevents stale partial-SVD
   or duplicate QR rows from contaminating report-index normalization.

5. **Documentation moved with the evidence.** README, algorithm docs,
   cookbook guidance, maintainer guidance, corpus docs, and oracle-schema docs
   now describe the same six-fixture family, `23` solver-backed QR rows, and
   explicit non-claims.

## What Didn't Go Well

1. **Reorder/COLAMD QR did not fit the sprint closure model.** The path has
   useful evidence, but it mixes residual, status, permutation, fill, optional
   SuiteSparse, and performance-adjacent semantics. Deferring it was the right
   decision, but it means one originally considered QR family remains outside
   the maintained corpus.

2. **Generated-row freshness is still advisory.** The freshness gate confirms
   generated-local rows are present, but strict generated-row comparison is
   still pending. The warnings are expected and documented, but they remain a
   real follow-up if generated reports become stronger release evidence.

3. **Historical artifacts still contain old QR row counts.** Current docs are
   aligned to `solver_qr_row_count=23`, but Day 1 and validation artifacts
   intentionally mention the old Sprint 139 count as historical baseline or
   stale-search text. Reviewers need to distinguish historical evidence from
   current guidance.

4. **Exact-value minimum-norm rows are intentionally narrow.** The sprint
   proved selected deterministic values for selected fixtures. It did not
   promote broad minimum-norm, least-squares, or pseudoinverse behavior.

5. **The sprint touched code, corpus, docs, and generator scripts together.**
   The integrated validation passed, but the review surface is wider than a
   pure documentation sprint and should be checked as a coordinated corpus
   product change.

## Final Metrics

### Validation

| Metric | Sprint 150 close state |
| --- | --- |
| tracked `.c` changes | yes: `tests/test_qr_corpus.c` |
| tracked `.h` changes | no |
| full C quality gate required | yes |
| focused QR proof owner | passed: 14 tests, 0 failed, 0 skipped, 258 assertions |
| corpus schema validation | passed |
| QR oracle generation | passed: `python3 scripts/run_corpus_oracle.py --include-solver-qr` |
| report-index normalization | passed: `78` rows ok |
| oracle freshness check | passed: freshness ok for `28` rows with expected generated-local advisory warnings |
| Python script compile | passed |
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| `git diff --check` | passed |
| targeted trailing-whitespace scan | passed |
| stale current-doc QR count search | passed; only historical artifact references remain |

### Artifact Package

| Metric | Sprint 150 close state |
| --- | ---: |
| daily artifacts under `SPRINT_150/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| source files changed | 1 |
| script files changed | 2 |
| public/support docs changed | 6 |
| corpus manifest files changed | 2 |
| new expected-result TSV files | 5 |
| selected maintained QR fixtures | 6 |
| generated-local QR oracle rows | 26 |
| generated-local solver-backed QR rows | 23 |

## Closed Claim

Sprint 150 closes this local claim:

The maintained QR corpus has been expanded from the Sprint 139 seed fixture to
a bounded six-fixture family covering selected rank-deficient rectangular and
underdetermined minimum-norm QR behaviors, with source-controlled metadata,
expected rows, focused proof-owner tests, generated-local oracle/report rows,
normalized report-index checks, and aligned documentation.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-qr-intake.md](./artifacts/day1-qr-intake.md);
- [day2-qr-family-candidate-audit.md](./artifacts/day2-qr-family-candidate-audit.md);
- [day3-family-selection-claim-scope.md](./artifacts/day3-family-selection-claim-scope.md);
- [day4-fixture-metadata-design.md](./artifacts/day4-fixture-metadata-design.md);
- [day5-fixture-metadata-batch.md](./artifacts/day5-fixture-metadata-batch.md);
- [day6-oracle-semantics-design.md](./artifacts/day6-oracle-semantics-design.md);
- [day7-oracle-data-implementation.md](./artifacts/day7-oracle-data-implementation.md);
- [day8-proof-owner-test-design.md](./artifacts/day8-proof-owner-test-design.md);
- [day9-proof-owner-implementation.md](./artifacts/day9-proof-owner-implementation.md);
- [day10-report-integration-design.md](./artifacts/day10-report-integration-design.md);
- [day11-report-integration-implementation.md](./artifacts/day11-report-integration-implementation.md);
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md).

## Next-Sprint Readiness

Sprint 151 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| Partial-SVD family selection | Select a small set of families that can be completely closed, not a broad partial expansion. |
| Sprint 140 seed fixture | Treat the Sprint 140 fixture-local closure as the partial-SVD seed, equivalent to how Sprint 150 used the Sprint 139 QR seed. |
| Comparison contract | Define singular-value, projector, vector residual, ordering, tolerance, sparse-output, and convergence semantics before adding rows. |
| Raw vector identity | Keep raw singular-vector identity, sign, basis orientation, and ordering claims out of scope unless separately proven. |
| Proof ownership | Add focused partial-SVD corpus tests rather than expanding broad monolithic SVD lanes. |
| Report generation | Reuse the Day 11 stale generated-output cleanup pattern before writing partial-SVD oracle/report rows. |
| Documentation | Align SVD docs, solver-selection docs, cookbook/tutorial references, and maintainer guidance with bounded fixture-local evidence. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 150 close:

- reorder/COLAMD QR corpus promotion;
- strict generated-row freshness comparison for generated-local oracle rows;
- broad rank-threshold policy;
- broad QR correctness;
- broad rank-deficient solve behavior;
- broad minimum-norm or least-squares behavior;
- SVD-pseudoinverse global-oracle behavior;
- external-library parity;
- platform, package, ABI, performance, installed-consumer, or
  state-of-the-art proof for QR behavior.

Still consciously constrained rather than silently solved:

- no raw QR basis or raw nullspace basis identity claim;
- no sign, orientation, scale, or column-order parity claim;
- no broad correctness claim from fixture-local corpus rows;
- no release-artifact claim from generated-local `build/` reports;
- no packaging/platform claim from numerical corpus proof;
- no performance or solver-optimality claim from residual/status evidence.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-qr-intake.md](./artifacts/day1-qr-intake.md)
- [day2-qr-family-candidate-audit.md](./artifacts/day2-qr-family-candidate-audit.md)
- [day3-family-selection-claim-scope.md](./artifacts/day3-family-selection-claim-scope.md)
- [day4-fixture-metadata-design.md](./artifacts/day4-fixture-metadata-design.md)
- [day5-fixture-metadata-batch.md](./artifacts/day5-fixture-metadata-batch.md)
- [day6-oracle-semantics-design.md](./artifacts/day6-oracle-semantics-design.md)
- [day7-oracle-data-implementation.md](./artifacts/day7-oracle-data-implementation.md)
- [day8-proof-owner-test-design.md](./artifacts/day8-proof-owner-test-design.md)
- [day9-proof-owner-implementation.md](./artifacts/day9-proof-owner-implementation.md)
- [day10-report-integration-design.md](./artifacts/day10-report-integration-design.md)
- [day11-report-integration-implementation.md](./artifacts/day11-report-integration-implementation.md)
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md)
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md)
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md)
