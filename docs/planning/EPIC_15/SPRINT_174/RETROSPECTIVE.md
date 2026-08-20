# Sprint 174 Retrospective

**Sprint:** 174 - Additional Bounded External Comparison Family
**Duration:** 14 days (Days 1-14 landed on branch `sprint-174`)
**Status:** Complete

## Source Artifact Note

Sprint 174 was executed from the active Epic 15 project-plan section for
Sprint 174 and lives under `docs/planning/EPIC_15/SPRINT_174/` with its plan,
working notes, daily artifacts, closeout artifact, and retrospective in one
package. The original sprint prompt referenced an older Epic 12 project-plan
path; `WORKING_NOTES.md` records that mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 174 plan, working notes, artifact directory, daily
      artifacts, closeout artifact, and retrospective.
- [x] Audited the existing selected comparison surface for QR minnorm, QR
      compatible least-squares, and partial-SVD diagonal top-k reports.
- [x] Inventoried candidate external-comparison families and selected one
      additional bounded family for complete closure.
- [x] Selected linked-list LU on `lu_nonsym_square_5` as the Sprint 174
      comparison family.
- [x] Added a focused dense-reference helper test for
      `tests/lu_external_dense_reference.py`.
- [x] Extended `scripts/run_external_comparison.py` with target
      `lu-nonsym-square-5`, project probe generation, baseline helper
      execution, residual calculation, and six selected generated rows.
- [x] Added the LU comparison artifact to selected report-index enforcement,
      report-family manifest ownership, and
      `make report-index-comparison-freshness`.
- [x] Updated README, maintainer guide, solver-selection docs, corpus docs,
      schema docs, and benchmark/report-index handoff docs with bounded LU
      comparison wording and non-claims.
- [x] Ran focused helper, runner, report-index, freshness, claim-scan,
      package/ABI deferral, and diff-hygiene checks.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required for Sprint 174 edits.

## What Went Well

1. **The comparison-family expansion stayed bounded.** Sprint 174 added one
   complete external comparison family instead of broadening several families
   only partially.

2. **The selected LU fixture reused existing evidence.** The sprint built on
   the maintained `lu_nonsym_square_5` dense-reference helper and existing C
   fixture semantics rather than introducing a new numerical case.

3. **The runner extension reused the existing report model.** The LU target
   fits the current non-partial-SVD `study.tsv` shape with project status,
   baseline status, residual norm, solution norm, solution values, and
   project-vs-baseline max absolute delta rows.

4. **Freshness became executable from the standard command.** The existing
   `make report-index-comparison-freshness` target now regenerates and checks
   QR minnorm, QR compatible least-squares, partial-SVD diagonal top-k, and
   LU nonsymmetric square-solve comparison reports.

5. **Report ownership is source-controlled.** The LU comparison has selected
   row IDs, selected artifact enforcement, a report-family manifest row,
   focused tests, and maintainer documentation that all agree on the same
   fixture and output path.

6. **The public claim is narrow enough to defend.** The new evidence supports
   only a fixture-local linked-list LU square-solve comparison against the
   selected dense helper, not broad LU correctness or external-library parity.

7. **Validation covered the changed surface.** Final checks exercised the LU
   helper, comparison runner, generated report freshness, normalized report
   index, stale wording scans, package-manager deferral, static package/ABI
   deferral, and diff hygiene.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 174 plan belongs to Epic 15. The sprint handled this by
   recording the mismatch and proceeding from
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.

2. **The comparison is still local-only.** Sprint 174 did not add hosted
   publication, CI artifact publication, or release evidence for the new LU
   report.

3. **The LU evidence is deliberately fixture-local.** The selected case is
   useful, but it does not prove broad LU behavior, pivoting superiority,
   singular-case reporting, LU CSR parity, or nonsymmetric ecosystem parity.

4. **The runner has more target-specific branching.** Adding LU required
   target-specific project-probe and baseline code in
   `scripts/run_external_comparison.py`; future families may need a clearer
   abstraction once more solve modes are added.

5. **Generated outputs remain ignored build artifacts.** This keeps the repo
   clean, but reviewers must run the freshness command to inspect regenerated
   comparison reports locally.

6. **Claim scans still need human interpretation.** Broad-support terms appear
   correctly in non-claims and bounded evidence statements, so scan output
   must be reviewed rather than treated as zero-match-only.

## Final Metrics

### Validation

| Metric | Sprint 174 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required by changed files | no |
| selected comparison freshness target | passed: `make report-index-comparison-freshness` |
| LU dense-reference helper test | passed: `python3 tests/test_lu_external_dense_reference.py` |
| comparison runner test | passed: `python3 tests/test_run_external_comparison.py` |
| report-index normalization test | passed: `python3 tests/test_normalize_report_index.py` |
| comparison runner self-check | passed: `python3 scripts/run_external_comparison.py --self-check` |
| direct comparison freshness check | passed: `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` |
| selected comparison freshness rows | passed: `normalize-report-index: freshness ok (32 rows)` |
| stale selected-comparison wording scan | passed by inspection; no maintained-doc stale matches |
| broad-claim/non-claim scan | passed by inspection; matches were fixture-local evidence or explicit non-claims |
| package-manager deferral guard | passed |
| static package/shared ABI deferral guard | passed |
| final `git diff --check` | passed |

### Changed Surface

| Metric | Sprint 174 close state |
| --- | ---: |
| C source files changed | 0 |
| public header files changed | 0 |
| Python helper tests added | 1 |
| comparison runner targets added | 1 |
| selected generated comparison row IDs added | 6 |
| selected generated comparison artifacts added | 1 |
| report-family manifest rows added | 1 |
| Make freshness target command additions | 1 |
| public/maintainer docs changed | 6 |
| runner/normalizer tests changed | 2 |
| daily artifacts under `SPRINT_174/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |

### Claim Governance

| Metric | Sprint 174 close state |
| --- | ---: |
| new bounded external comparison families added | 1 |
| selected LU comparison targets | 1 |
| selected LU generated rows | 6 |
| selected comparison contract rows total | 4 |
| selected generated comparison rows total | 28 |
| selected comparison rows checked by freshness | 32 |
| hosted comparison publication claims added | 0 |
| broad LU correctness claims added | 0 |
| broad nonsymmetric solve parity claims added | 0 |
| LU CSR parity claims added | 0 |
| sparse-direct solver parity claims added | 0 |
| external-library ecosystem parity claims added | 0 |
| package-manager support claims added | 0 |
| shared-library ABI support claims added | 0 |
| broad platform portability claims added | 0 |
| performance superiority claims added | 0 |
| state-of-the-art sparse linear algebra claims added | 0 |

## Closed Claim

Sprint 174 closes this Epic 15 comparison-publication claim:

The project now has a fourth selected local generated comparison family:
linked-list LU square solve on the `lu_nonsym_square_5` fixture. Maintainers
can run `make report-index-comparison-freshness` to regenerate the selected
QR minnorm, QR compatible least-squares, partial-SVD diagonal top-k, and LU
nonsymmetric square-solve reports, then check the selected comparison rows for
freshness against the current source commit.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-comparison-intake.md](./artifacts/day1-comparison-intake.md);
- [day2-candidate-family-inventory.md](./artifacts/day2-candidate-family-inventory.md);
- [day3-family-selection.md](./artifacts/day3-family-selection.md);
- [day4-fixture-design.md](./artifacts/day4-fixture-design.md);
- [day5-comparator-output-design.md](./artifacts/day5-comparator-output-design.md);
- [day6-fixture-implementation.md](./artifacts/day6-fixture-implementation.md);
- [day7-harness-extension-design.md](./artifacts/day7-harness-extension-design.md);
- [day8-harness-implementation.md](./artifacts/day8-harness-implementation.md);
- [day9-report-integration.md](./artifacts/day9-report-integration.md);
- [day10-freshness-gate.md](./artifacts/day10-freshness-gate.md);
- [day11-claim-documentation.md](./artifacts/day11-claim-documentation.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-integrated-claim-review.md](./artifacts/day13-integrated-claim-review.md);
- [day14-sprint-closeout.md](./artifacts/day14-sprint-closeout.md).

No broad LU correctness, broad nonsymmetric solve correctness, LU CSR parity,
sparse-direct solver parity, pivoting superiority, factor-layout identity,
NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity, hosted comparison publication,
release evidence, package-manager support, shared-library ABI support,
runtime-loader behavior, broad platform portability proof, performance
superiority, or state-of-the-art sparse linear algebra claim was added.

## Sprint 175 Readiness

| Future need | Sprint 174 handoff |
| --- | --- |
| Selected comparison freshness | Run `make report-index-comparison-freshness` before relying on selected local generated comparison rows. |
| LU comparison scope | Treat `lu-nonsym-square-5` as fixture-local linked-list LU square-solve evidence only. |
| Generated comparison output staging | Keep `build/comparison/*` generated outputs ignored and local unless a later sprint explicitly promotes hosted or artifact publication. |
| Additional direct-solver comparisons | Prefer one complete family at a time with fixture, helper, report rows, manifest ownership, freshness, docs, and non-claims. |
| Hosted comparison publication | Define URL ownership, retention, freshness semantics, CI behavior, and support wording before implementation. |
| Broad LU or LU CSR claims | Require broader fixture corpus, public API scope, solver-family coverage, external comparator policy, and claim gates before promotion. |
| Package-manager wording changes | Run `bash scripts/package_manager_deferral_check.sh`. |
| Static package/shared ABI wording changes | Run `bash scripts/static_package_deferral_check.sh`. |
