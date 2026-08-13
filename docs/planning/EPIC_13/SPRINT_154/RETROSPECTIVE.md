# Sprint 154 Retrospective

**Sprint:** 154 - External Comparison Harness And First Narrow Study
**Duration:** 14 days (Days 1-14 landed on branch `sprint-154`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 154 day-by-day plan, working notes, artifact directory,
      closeout artifact, Sprint 155 handoff, and retrospective.
- [x] Re-established the comparison boundary from Sprints 150-153, including
      QR/partial-SVD corpus evidence, selected oracle freshness, and
      static-first package/ABI non-claims.
- [x] Audited QR and partial-SVD comparison candidates and selected exactly one
      narrow target: `qr_underdetermined_minnorm_2x4`.
- [x] Defined dependency policy for the first study: source-controlled
      external-process dense reference helper required; optional NumPy/SciPy
      package baselines deferred.
- [x] Designed comparison output schema, selected row ids, provenance fields,
      status semantics, stale-output behavior, and non-claim boundaries.
- [x] Implemented `scripts/run_external_comparison.py` with project-side QR
      minimum-norm probing, baseline execution/parsing, comparison evaluation,
      generated study rows, summary output, manifest output, dependency
      diagnostics, and self-checks.
- [x] Added normalized report-index integration for
      `comparison/qr_minnorm`.
- [x] Added `make report-index-comparison-freshness` as the maintained local
      comparison freshness command.
- [x] Updated schema validation and report-index normalization so required
      comparison freshness fails closed on missing, duplicate, unexpected, or
      non-pass selected rows.
- [x] Updated README, maintainer guide, benchmark/report guidance, and solver
      selection docs to describe the comparison lane without overclaiming.
- [x] Published the first source-controlled narrow comparison study snapshot.
- [x] Ran focused comparison, schema, report-index, stale wording, and
      whitespace validation.
- [x] Prepared the Sprint 155 tutorial/header/API-reference handoff.

## What Went Well

1. **The comparison target stayed narrow.** The sprint resisted turning a new
   harness into a broad ecosystem parity story. It selected one QR
   minimum-norm fixture, froze accepted metrics, and carried non-claims through
   the implementation and docs.

2. **Dependency semantics stayed honest.** The selected baseline is a
   source-controlled dense reference helper executed out of process. Optional
   NumPy and SciPy package baselines are visible `defer` rows, not hidden
   prerequisites and not pass evidence.

3. **The harness is reproducible and auditable.** The generated artifacts
   record source commit, branch, worktree state, platform, compiler, project
   command, baseline command, Python executable/version, metric values,
   tolerances, status reasons, caveats, support tier, claim scope, and
   non-claims.

4. **Report-index integration became real rather than aspirational.** Sprint
   154 added a maintained `comparison/qr_minnorm` family, a generated-row
   loader, selected-row diagnostics, required freshness behavior, and a
   Makefile target that regenerates and checks the lane.

5. **The first study is reviewable without committing generated build output.**
   `build/comparison/qr_minnorm/` remains ignored local output, while
   `first-narrow-qr-minnorm-comparison-study.md` publishes a source-controlled
   snapshot with caveats and residuals.

6. **Documentation was aligned before closeout.** README, maintainer guide,
   report docs, and solver-selection docs now point to the maintained command
   and explicitly constrain interpretation to fixture-local evidence.

## What Didn't Go Well

1. **The study is intentionally tiny.** One QR minimum-norm fixture is a useful
   first maintained comparison lane, but it does not materially broaden
   ecosystem comparison depth.

2. **The baseline is not an external package.** The source-controlled helper is
   reproducible and dependency-light, but it does not compare against NumPy,
   SciPy, LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, or package-manager
   installs.

3. **Dirty-worktree provenance remains part of generated development output.**
   The generated study records `worktree_state=dirty` because the branch is
   actively under development. That is acceptable local provenance, but it is
   not release proof.

4. **Report freshness grew another specialized path.** The comparison family
   is properly bounded, but the report-index normalizer now has another
   selected-row policy that future maintainers must preserve.

5. **Partial-SVD comparison publication remains deferred.** Sprint 154 used
   partial-SVD corpus evidence as candidate input, but it did not publish a
   normalized external comparison family for partial-SVD.

## Final Metrics

### Validation

| Metric | Sprint 154 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required | no; focused comparison/report/docs gate used |
| comparison freshness | passed: `make report-index-comparison-freshness` |
| harness self-check | passed: `python3 scripts/run_external_comparison.py --self-check` |
| schema validation | passed: `python3 scripts/validate_corpus_schema.py` |
| combined report-index structure | passed: `85` rows ok for corpus/oracle/comparison |
| required comparison freshness | passed: freshness ok for `7` comparison rows |
| focused stale wording search | passed; active hits are non-claims or scoped boundaries |
| `git diff --check` | passed |

### Artifact Package

| Metric | Sprint 154 close state |
| --- | ---: |
| daily artifacts under `SPRINT_154/artifacts/` | 14 |
| published comparison study snapshots | 1 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| C source files changed | 0 |
| public header files changed | 0 |
| Python scripts added | 1 |
| Python scripts modified | 2 |
| Makefile targets added | 1 |
| report-family rows added | 1 |
| public/support docs changed | 4 |
| selected generated comparison rows | 6 |
| normalized comparison rows | 7 |

## Closed Claim

Sprint 154 closes this comparison claim:

The project now has one maintained local generated external-comparison lane for
`qr_underdetermined_minnorm_2x4`, where `sparse_qr_solve_minnorm` agrees with
the selected source-controlled dense reference helper on project status,
baseline status, residual norm, solution norm, solution values, and maximum
absolute project-vs-baseline solution delta under the recorded command,
commit, platform, compiler, and local-only support tier.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-comparison-boundary.md](./artifacts/day1-comparison-boundary.md);
- [day2-target-candidate-audit.md](./artifacts/day2-target-candidate-audit.md);
- [day3-comparison-target-selection.md](./artifacts/day3-comparison-target-selection.md);
- [day4-dependency-pinning-policy.md](./artifacts/day4-dependency-pinning-policy.md);
- [day5-comparison-output-schema-design.md](./artifacts/day5-comparison-output-schema-design.md);
- [day6-harness-architecture-design.md](./artifacts/day6-harness-architecture-design.md);
- [day7-harness-project-runner-scaffold.md](./artifacts/day7-harness-project-runner-scaffold.md);
- [day8-baseline-runner-implementation.md](./artifacts/day8-baseline-runner-implementation.md);
- [day9-comparison-logic-implementation.md](./artifacts/day9-comparison-logic-implementation.md);
- [day10-report-integration-design.md](./artifacts/day10-report-integration-design.md);
- [day11-report-integration-implementation.md](./artifacts/day11-report-integration-implementation.md);
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md);
- [day13-integrated-validation-and-study-publication.md](./artifacts/day13-integrated-validation-and-study-publication.md);
- [first-narrow-qr-minnorm-comparison-study.md](./artifacts/first-narrow-qr-minnorm-comparison-study.md);
- [day14-closeout-sprint155-handoff.md](./artifacts/day14-closeout-sprint155-handoff.md).

## Next-Sprint Readiness

Sprint 155 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| Tutorial wording | Mention comparison evidence only in advanced/report contexts, not as first-use proof. |
| QR minimum-norm docs | Keep `sparse_qr_solve_minnorm` evidence fixture-local unless new comparison lanes are added. |
| API reference cleanup | Preserve declarations and avoid adding broad external-library or performance claims. |
| Report-index docs | Use `make report-index-comparison-freshness` for the selected comparison lane. |
| Optional packages | Keep NumPy/SciPy as deferred package baselines, not pass evidence. |
| Header comments | Do not imply LAPACK, SuiteSparse, Eigen, hosted CI, package-manager, shared-library ABI, platform, or state-of-the-art proof. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 154 close:

- QR comparison beyond `qr_underdetermined_minnorm_2x4`;
- optional NumPy and SciPy package baselines;
- LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, and other ecosystem baselines;
- QR raw Q/R basis, sign/orientation/order, pivot-order, and rank-threshold
  comparison;
- broad rank-deficient, nullspace, economy-mode, sparse-mode, and reorder
  comparison;
- partial-SVD publication under the normalized `comparison` family;
- portable runtime or performance comparison;
- hosted CI comparison publication;
- package-manager, shared-library, loader, and ABI comparison lanes.

Still consciously constrained rather than silently solved:

- `comparison/qr_minnorm` is local-only, not hosted CI proof;
- `pass` rows support only the selected fixture-local statement;
- `skip`, `defer`, `fail`, and `error` rows do not count as proof;
- optional package absence cannot create pass evidence;
- source-controlled helper agreement is not NumPy/SciPy/LAPACK parity;
- generated `build/comparison/` outputs remain ignored local artifacts;
- dirty-worktree generated provenance is not release proof.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-comparison-boundary.md](./artifacts/day1-comparison-boundary.md)
- [day2-target-candidate-audit.md](./artifacts/day2-target-candidate-audit.md)
- [day3-comparison-target-selection.md](./artifacts/day3-comparison-target-selection.md)
- [day4-dependency-pinning-policy.md](./artifacts/day4-dependency-pinning-policy.md)
- [day5-comparison-output-schema-design.md](./artifacts/day5-comparison-output-schema-design.md)
- [day6-harness-architecture-design.md](./artifacts/day6-harness-architecture-design.md)
- [day7-harness-project-runner-scaffold.md](./artifacts/day7-harness-project-runner-scaffold.md)
- [day8-baseline-runner-implementation.md](./artifacts/day8-baseline-runner-implementation.md)
- [day9-comparison-logic-implementation.md](./artifacts/day9-comparison-logic-implementation.md)
- [day10-report-integration-design.md](./artifacts/day10-report-integration-design.md)
- [day11-report-integration-implementation.md](./artifacts/day11-report-integration-implementation.md)
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md)
- [day13-integrated-validation-and-study-publication.md](./artifacts/day13-integrated-validation-and-study-publication.md)
- [first-narrow-qr-minnorm-comparison-study.md](./artifacts/first-narrow-qr-minnorm-comparison-study.md)
- [day14-closeout-sprint155-handoff.md](./artifacts/day14-closeout-sprint155-handoff.md)
