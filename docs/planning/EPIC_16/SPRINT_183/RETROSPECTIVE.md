# Sprint 183 Retrospective

**Sprint:** 183 - Additional Bounded External Comparison Family
**Duration:** 14 days (Days 1-14 landed on branch `sprint-183`)
**Status:** Complete

## Source Artifact Note

Sprint 183 was executed from the Epic 16 project-plan section for Sprint 183
and lives under `docs/planning/EPIC_16/SPRINT_183/` with its plan, working
notes, daily artifacts, closeout artifact, and retrospective in one package.

## Definition Of Done Checklist

- [x] Created Sprint 183 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited the inherited selected comparison authority, existing runner
      surface, manifest rows, generated artifact conventions, freshness target,
      and workflow guard invariants.
- [x] Inventoried candidate external comparison families and selected exactly
      one additional family: Cholesky SPD tridiagonal solve.
- [x] Defined the fixture, RHS, expected solution, expected norm, tolerances,
      six selected rows, required files, helper behavior, dependency policy,
      and non-claims for `cholesky_spd_tridiag_5`.
- [x] Implemented key-based Cholesky dense-reference helper support and focused
      helper tests.
- [x] Extended the external comparison runner with `cholesky-spd-tridiag-5`,
      Cholesky project probe generation, baseline dispatch, dependency rows,
      metadata, and focused runner tests.
- [x] Added report-family and selected-target manifest metadata for the new
      comparison family.
- [x] Integrated the new family into `make report-index-comparison-freshness`,
      Linux/macOS hosted selected comparison summaries, and fail-closed upload
      allowlists.
- [x] Hardened selected comparison workflow tests for Cholesky target presence,
      missing upload paths, broad upload rejection, and retained Windows
      non-promotion.
- [x] Aligned README, solver-selection docs, maintainer guide, corpus README,
      and report-index schema docs with the bounded Cholesky comparison claim.
- [x] Preserved non-claims for broad Cholesky correctness, broad SPD coverage,
      reordering parity, CSC-vs-linked-list parity, fill superiority,
      external-library parity, Windows report freshness, package/ABI,
      performance, release, and state-of-the-art evidence.
- [x] Ran focused helper, runner, manifest, workflow, normalizer, schema,
      freshness, deferral, C fixture, formatting, lint, full test, generated
      artifact, cache, and whitespace validation.
- [x] Confirmed generated build/report artifacts remained unstaged and no
      tracked C/header diffs remained after formatting.

## What Went Well

1. **The sprint added one bounded family without schema churn.** The Cholesky
   target reused the existing solve-shaped selected comparison row pattern:
   project status, baseline status, residual norm, solution norm, solution
   values, and project-vs-baseline max absolute delta.

2. **The fixture contract stayed simple and deterministic.** The 5x5 SPD
   tridiagonal fixture has exact entries, exact RHS, exact solution, clear
   `1e-10` tolerances, and a source-controlled dense Cholesky helper.

3. **The selected-target manifest remained authoritative.** Target key,
   expected rows, expected row IDs, required files, workflow metadata, support
   tier, claim scope, non-claims, and owner all live in
   `selected_report_targets.tsv`.

4. **Workflow promotion stayed fail-closed.** Linux and macOS hosted selected
   comparison lanes upload exact Cholesky files, reject broad
   `build/comparison/**` uploads, and have guard tests for missing Cholesky
   artifact paths.

5. **Windows was not accidentally promoted.** Sprint 183 kept the Sprint 182
   Windows report freshness deferral intact: no selected target lists
   `windows`, and Windows workflow guards still reject selected freshness
   commands and selected comparison artifact names.

6. **Documentation matched implementation.** README, solver-selection docs,
   maintainer guide, corpus README, and report-index schema docs describe the
   same fixture-local Cholesky comparison and the same non-claims.

7. **The final validation pass was broad enough.** Day 12 ran focused Python
   checks, selected freshness, package deferral guards, the actual Cholesky C
   test binary, `make format`, `make lint`, and full `make test`.

## What Didn't Go Well

1. **The runner needed a temporary metadata bypass during implementation.**
   Day 8 temporarily allowed the Cholesky target to run before the
   report-family row existed. Day 9 removed the bypass, but this pattern needs
   disciplined follow-through.

2. **A guessed Makefile test target was invalid.** `make test_cholesky` does
   not exist. The correct focused path is `build/test_cholesky`, or full
   `make test`.

3. **Full lint remains expensive.** `make lint` runs clang-tidy and cppcheck
   across a large source and test surface. It passed, but Day 12 required a
   long-running validation window.

4. **Hosted proof still waits for PR CI.** Local validation passed, but the
   reviewed Linux/macOS selected comparison lanes only become hosted evidence
   after the branch is pushed and CI runs.

5. **The active selected comparison list changed in several docs at once.**
   Moving from four to five selected comparison families required careful
   wording updates across README, solver-selection, maintainer, corpus, schema,
   workflow, and planning records.

## Final Metrics

### Validation

| Metric | Sprint 183 close state |
| --- | --- |
| Cholesky helper test | passed: `python3 tests/test_chol_external_dense_reference.py` |
| external comparison runner test | passed: `python3 tests/test_run_external_comparison.py` |
| selected comparison workflow guard | passed: `python3 tests/test_selected_comparison_workflow.py` |
| selected target manifest diagnostics | passed: `python3 tests/test_selected_report_targets_manifest.py` |
| report-index regression tests | passed: `python3 tests/test_normalize_report_index.py` |
| corpus schema validation | passed: `python3 scripts/validate_corpus_schema.py` |
| focused Cholesky C test | passed: `build/test_cholesky` |
| selected comparison freshness | passed: `make report-index-comparison-freshness` |
| static package deferral guard | passed |
| package-manager deferral guard | passed |
| formatting | passed: `make format` |
| lint | passed: `make lint` |
| full test suite | passed: `make test` |
| final generated artifact check | passed: no staged `build/comparison` or `build/report-index` output |
| final cache check | passed: no `__pycache__` directories |
| final `git diff --check` | passed |
| tracked C/header diffs after formatting | none |

### Changed Surface

| Metric | Sprint 183 close state |
| --- | ---: |
| selected comparison families added | 1 |
| selected comparison families active | 5 |
| new selected Cholesky rows | 6 |
| new selected Cholesky required files | 6 |
| workflow files changed | 2 |
| Makefile targets changed | 1 |
| runner scripts changed | 1 |
| helper scripts changed | 1 |
| Python tests added | 1 |
| Python tests changed | 3 |
| manifest files changed | 2 |
| public/maintainer docs changed | 3 |
| corpus/report-index docs changed | 2 |
| daily artifacts | 14 |
| retrospective files | 1 |
| project-plan items completed | 6 |
| generated build/report artifacts staged | 0 |
| tracked C source/header files changed | 0 |

### Claim Governance

| Metric | Sprint 183 close state |
| --- | ---: |
| broad Cholesky correctness claims added | 0 |
| broad SPD coverage claims added | 0 |
| Cholesky reordering parity claims added | 0 |
| CSC-vs-linked-list parity claims added | 0 |
| fill superiority claims added | 0 |
| external-library ecosystem parity claims added | 0 |
| Windows report freshness promotions added | 0 |
| package-manager support claims added | 0 |
| shared-library ABI claims added | 0 |
| performance superiority claims added | 0 |
| release readiness claims added | 0 |
| state-of-the-art claims added | 0 |

## Closed Claim

Sprint 183 closes this Epic 16 selected comparison claim:

For the selected `cholesky_spd_tridiag_5` fixture, the project one-shot
Cholesky factor/solve path and the selected source-controlled dense Cholesky
reference helper both succeed and agree within the selected residual,
solution-norm, solution-value, and project-vs-baseline max-delta tolerances.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-comparison-family-intake.md](./artifacts/day1-comparison-family-intake.md);
- [day2-existing-comparison-surface-audit.md](./artifacts/day2-existing-comparison-surface-audit.md);
- [day3-candidate-family-inventory.md](./artifacts/day3-candidate-family-inventory.md);
- [day4-family-selection.md](./artifacts/day4-family-selection.md);
- [day5-fixture-and-metric-contract.md](./artifacts/day5-fixture-and-metric-contract.md);
- [day6-helper-and-fixture-implementation.md](./artifacts/day6-helper-and-fixture-implementation.md);
- [day7-runner-extension-design.md](./artifacts/day7-runner-extension-design.md);
- [day8-runner-implementation.md](./artifacts/day8-runner-implementation.md);
- [day9-report-integration.md](./artifacts/day9-report-integration.md);
- [day10-freshness-gate-and-workflow-guard.md](./artifacts/day10-freshness-gate-and-workflow-guard.md);
- [day11-documentation-alignment.md](./artifacts/day11-documentation-alignment.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-claim-review-and-hardening.md](./artifacts/day13-claim-review-and-hardening.md);
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md).

No broad Cholesky correctness, broad SPD coverage, broad reordering coverage,
CSC-vs-linked-list parity, factor-layout identity, fill superiority,
NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity, external-library ecosystem parity,
Windows report freshness, package-manager proof, shared-library ABI proof,
performance superiority, release proof, or state-of-the-art claim was added.

## Sprint 184 Readiness

Sprint 184 should start from the next Epic 16 project-plan section. If it adds
another comparison family, use this Sprint 183 handoff:

| Future need | Sprint 183 handoff |
| --- | --- |
| Candidate selection | Start from selected target manifest authority and reject broad parity, package, platform, performance, or backend claims up front. |
| Fixture contract | Define exact matrix, RHS, expected values, row IDs, tolerances, helper behavior, required files, and non-claims before runner changes. |
| Helper path | Prefer source-controlled deterministic helpers; optional NumPy/SciPy rows should remain defer context, not pass evidence. |
| Runner extension | Reuse existing solve-shaped comparison machinery where possible; add new row shapes only with manifest and normalizer support. |
| Report integration | Add report-family and selected-target manifest rows before enforcing metadata checks. |
| Workflow promotion | Add exact Linux/macOS selected upload paths and guard tests; avoid broad `build/comparison/**` uploads. |
| Windows | Keep Windows report freshness deferred unless a future sprint promotes a reviewed Windows-safe generator, artifact scope, manifest row, workflow guard, and docs together. |
| Validation | Run focused helper/runner tests, manifest and normalizer tests, workflow guards, selected freshness, relevant C tests, deferral guards if wording touches package/ABI or package-manager surfaces, `make format`, `make lint`, `make test`, and `git diff --check`. |

Strong future candidates remain LDLT KKT solve, iterative SPD/nonsymmetric
solve, eigensolver bounded comparison, backend telemetry, and performance
comparison. Each should be treated as a separate selected family with its own
fixture contract and claim review.
