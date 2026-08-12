# Sprint 152 Retrospective

**Sprint:** 152 - Generated Report Freshness Publication
**Duration:** 14 days (Days 1-14 landed on branch `sprint-152`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 152 day-by-day plan, working notes, artifact directory,
      closeout artifact, Sprint 153 handoff, and retrospective.
- [x] Audited generated report producers, report-family metadata, CI artifact
      boundaries, freshness states, and generated-local claim risks.
- [x] Selected the oracle generated-reference and solver-backed families as the
      only Sprint 152 generated freshness publication target.
- [x] Kept corpus fixture/generator/expected rows source-controlled rather than
      treating them as generated freshness proof.
- [x] Deferred benchmark, sentinel, guardrail, dead-code, coverage, package,
      CI, documentation, and runtime-backend generated freshness promotion
      unless a later sprint adds a narrower owned policy.
- [x] Added selected oracle policy checks for the combined QR plus partial-SVD
      command, row counts, solver-family counts, fixture-key coverage, stale
      commits, missing artifacts, failing rows, partial output, and missing
      selected families.
- [x] Added the maintained local target `make report-index-oracle-freshness`.
- [x] Strengthened focused report-index tests for required, strict, advisory,
      stale, missing, failing, partial, missing-solver-family, and
      missing-fixture-key behavior.
- [x] Updated report-family metadata and report-index schema documentation for
      the selected oracle freshness gate.
- [x] Updated README, algorithm docs, solver-selection docs, and maintainer
      guidance with selected local command wording and non-claims.
- [x] Ran final focused Python/report/schema/freshness checks and whitespace
      validation.
- [x] Prepared the Sprint 153 shared-library ABI product-decision handoff.

## What Went Well

1. **Generated freshness became an explicit product surface.** Sprint 152 moved
   the selected oracle rows from informal local output into a maintained local
   command with owner, row-count, fixture-key, solver-family, and non-claim
   policy.

2. **The sprint selected a small target and closed it.** Only
   `oracle/generated_reference` and `oracle/solver_backed` became selected
   generated freshness families. Everything else stayed advisory,
   source-controlled, hosted-external, optional, or deferred.

3. **The failure model became actionable.** Required oracle freshness now names
   missing artifacts, stale commits, failing comparison rows, partial selected
   output, missing solver families, missing fixture keys, and the regeneration
   command.

4. **Local evidence stayed local.** The new target regenerates ignored `build/`
   outputs but does not upload them, commit them, or treat them as hosted CI,
   package, ABI, performance, platform, release, or state-of-the-art proof.

5. **The documentation and manifest rows now agree.** Active guidance points at
   `make report-index-oracle-freshness`; QR-only and partial-SVD-only commands
   are documented as focused debugging variants that do not satisfy the
   selected combined policy.

6. **Sprint 153 gets a cleaner package/ABI boundary.** The handoff explicitly
   says generated oracle freshness is not shared-library, loader, package, or
   ABI evidence.

## What Didn't Go Well

1. **Row-level generated freshness is still advisory wording.** The selected
   aggregate policy passes, but generated-local rows still emit
   `generated_present_unchecked` warnings. Sprint 152 closed the aggregate
   selected policy, not full row-level strict comparison semantics.

2. **The report-family ecosystem is still uneven.** Benchmark, sentinel,
   guardrail, dead-code, and coverage rows remain useful but advisory or
   later-sprint owned. They should not be cited as claim-bearing generated
   freshness proof until they get the same level of command/path/owner policy.

3. **The focused target depends on local regeneration discipline.** The
   generated outputs are ignored by design, so maintainers must run
   `make report-index-oracle-freshness` when they need current local oracle
   evidence.

4. **Hosted CI evidence remains deliberately external.** CI package/platform
   logs are useful, but Sprint 152 did not create a hosted selected oracle lane
   or source-controlled generated artifact publication path.

5. **The review surface is cross-cutting.** The sprint touched Makefile,
   Python policy code, Python tests, metadata, schema docs, public docs, and
   planning artifacts. Reviewers should check the claim boundary as a coherent
   report-product change.

## Final Metrics

### Validation

| Metric | Sprint 152 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked `.h` changes | no |
| full C quality gate required | no; focused Python/report/docs gate used |
| selected oracle freshness target | passed: `make report-index-oracle-freshness` |
| corpus schema validation | passed |
| report-index unit test | passed |
| normalized corpus/oracle index | passed: `128` rows ok |
| selected oracle generated rows | `52` total |
| QR solver-backed oracle rows | `23` |
| partial-SVD solver-backed oracle rows | `26` |
| generated-reference oracle rows | `3` |
| strict selected oracle freshness | passed with `0` freshness errors and `52` expected row-level warnings |
| advisory/source-controlled freshness checks | passed with `11` advisory/source-controlled rows |
| stale active-doc wording search | passed; QR-only and partial-SVD-only references are intentional debug variants |
| `git diff --check` | passed |
| generated Python cache cleanup | complete |

### Artifact Package

| Metric | Sprint 152 close state |
| --- | ---: |
| daily artifacts under `SPRINT_152/artifacts/` | 14 |
| Sprint 153 handoff artifacts | 1 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| Makefile targets added | 1 |
| Python policy files changed | 1 |
| Python test files changed | 1 |
| public/support docs changed | 4 |
| report-family manifest files changed | 1 |
| report schema docs changed | 1 |
| selected generated freshness families | 2 |
| generated local selected oracle rows | 52 |

## Closed Claim

Sprint 152 closes this local claim:

The selected generated oracle report family now has a maintained local freshness
publication gate, `make report-index-oracle-freshness`, covering the combined
QR plus partial-SVD oracle output with selected row-count, solver-family,
fixture-key, stale-commit, missing-output, failing-row, partial-output, and
documentation policy checks.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-generated-report-baseline.md](./artifacts/day1-generated-report-baseline.md);
- [day2-generated-family-candidate-audit.md](./artifacts/day2-generated-family-candidate-audit.md);
- [day3-generated-family-selection.md](./artifacts/day3-generated-family-selection.md);
- [day4-freshness-policy-design.md](./artifacts/day4-freshness-policy-design.md);
- [day5-generator-stabilization-design.md](./artifacts/day5-generator-stabilization-design.md);
- [day6-generator-stabilization-implementation.md](./artifacts/day6-generator-stabilization-implementation.md);
- [day7-freshness-gate-design.md](./artifacts/day7-freshness-gate-design.md);
- [day8-freshness-gate-implementation.md](./artifacts/day8-freshness-gate-implementation.md);
- [day9-ci-artifact-policy.md](./artifacts/day9-ci-artifact-policy.md);
- [day10-ci-artifact-implementation.md](./artifacts/day10-ci-artifact-implementation.md);
- [day11-documentation-alignment.md](./artifacts/day11-documentation-alignment.md);
- [day12-integrated-regeneration-validation.md](./artifacts/day12-integrated-regeneration-validation.md);
- [day13-quality-gate-residual-review.md](./artifacts/day13-quality-gate-residual-review.md);
- [day14-closeout-summary.md](./artifacts/day14-closeout-summary.md);
- [sprint153-abi-package-handoff.md](./artifacts/sprint153-abi-package-handoff.md).

## Next-Sprint Readiness

Sprint 153 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| Shared-library ABI decision | Treat Sprint 152 oracle freshness as fixture-local generated evidence only. |
| Package rows | Use package/static-install rows as source-controlled proof-owner metadata, not generated freshness proof. |
| CI rows | Treat hosted CI logs as external evidence, not local generated artifacts. |
| Selected oracle target | Keep `make report-index-oracle-freshness` available as local solver evidence but do not cite it as ABI, loader, or package proof. |
| Public symbol surface | Audit headers, structs, macros, symbols, allocator behavior, callbacks, and version metadata independently. |
| Platform loader proof | Require separate Linux, macOS, and Windows loader evidence before claiming shared-library support. |
| Static-first fallback | If shared support is deferred, document exact blockers and maintained rejection tests. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 152 close:

- row-level strict generated freshness comparison semantics beyond the selected
  aggregate policy;
- benchmark generated freshness publication;
- sentinel hard-gate publication policy;
- sentinel advisory measurement publication policy;
- large-matrix guardrail generated freshness policy;
- dead-code generated report freshness policy;
- coverage generated report freshness policy;
- hosted selected oracle CI lane;
- generated artifact upload/retention policy for selected oracle output;
- release artifact proof for generated reports;
- package, ABI, loader, platform, performance, external-library parity, and
  state-of-the-art claims.

Still consciously constrained rather than silently solved:

- generated `build/` outputs remain ignored local artifacts;
- missing generated rows do not become pass evidence;
- optional/deferred/advisory rows do not count as freshness proof;
- source-controlled report-family rows remain governed by Git review and
  schema validation;
- hosted workflow logs are external evidence and not source-controlled
  freshness artifacts.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-generated-report-baseline.md](./artifacts/day1-generated-report-baseline.md)
- [day2-generated-family-candidate-audit.md](./artifacts/day2-generated-family-candidate-audit.md)
- [day3-generated-family-selection.md](./artifacts/day3-generated-family-selection.md)
- [day4-freshness-policy-design.md](./artifacts/day4-freshness-policy-design.md)
- [day5-generator-stabilization-design.md](./artifacts/day5-generator-stabilization-design.md)
- [day6-generator-stabilization-implementation.md](./artifacts/day6-generator-stabilization-implementation.md)
- [day7-freshness-gate-design.md](./artifacts/day7-freshness-gate-design.md)
- [day8-freshness-gate-implementation.md](./artifacts/day8-freshness-gate-implementation.md)
- [day9-ci-artifact-policy.md](./artifacts/day9-ci-artifact-policy.md)
- [day10-ci-artifact-implementation.md](./artifacts/day10-ci-artifact-implementation.md)
- [day11-documentation-alignment.md](./artifacts/day11-documentation-alignment.md)
- [day12-integrated-regeneration-validation.md](./artifacts/day12-integrated-regeneration-validation.md)
- [day13-quality-gate-residual-review.md](./artifacts/day13-quality-gate-residual-review.md)
- [day14-closeout-summary.md](./artifacts/day14-closeout-summary.md)
- [sprint153-abi-package-handoff.md](./artifacts/sprint153-abi-package-handoff.md)

