# Sprint 160 Retrospective

**Sprint:** 160 - QR Comparison Family Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-160`)
**Status:** Complete

## Source Artifact Note

Sprint 160 was planned from the Epic 14 project-plan section for Sprint 160
and lives under `docs/planning/EPIC_14/SPRINT_160/` with its plan, working
notes, artifacts, and retrospective in one package. The original sprint prompt
referenced an older Epic 12 path; `WORKING_NOTES.md` records that path mismatch
for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 160 plan, working notes, daily artifacts, closeout
      artifact, and retrospective.
- [x] Selected one bounded QR comparison expansion target:
      `qr_overdetermined_compatible_5x3`.
- [x] Added descriptor-backed `qr-compatible-ls` generation to
      `scripts/run_external_comparison.py`.
- [x] Preserved the existing `qr-minnorm` comparison family behavior.
- [x] Added source-controlled report metadata for `comparison/qr_compatible_ls`.
- [x] Expanded selected comparison freshness from 6 generated rows to 12
      generated rows across two selected QR comparison families.
- [x] Updated `make report-index-comparison-freshness` to regenerate both
      selected targets before strict freshness normalization.
- [x] Added focused CLI regression coverage for the comparison runner.
- [x] Tightened selected-comparison diagnostics so missing, mismatched, or
      non-pass selected rows name both selected study artifacts.
- [x] Preserved skip/defer semantics so optional NumPy/SciPy rows remain
      non-proof context.
- [x] Aligned README, maintainer guide, solver-selection docs, and corpus docs
      with the two-family selected comparison surface.
- [x] Preserved non-claims for broad QR parity, raw basis identity,
      sign/orientation identity, external-library parity, platform, package,
      ABI, performance, release, and state-of-the-art evidence.
- [x] Published the Sprint 161 partial-SVD comparison handoff.
- [x] Ran and recorded the final targeted validation set.

## What Went Well

1. **The comparison expansion stayed bounded.** Sprint 160 added exactly one
   new QR family, `qr-compatible-ls`, and kept it tied to the maintained
   `qr_overdetermined_compatible_5x3` fixture.

2. **The descriptor model avoided parallel one-off paths.** Refactoring the
   runner around target descriptors let `qr-minnorm` and `qr-compatible-ls`
   share row generation, validation, summary, and manifest behavior.

3. **Focused tests matched the changed behavior.** `tests/test_run_external_comparison.py`
   protects CLI target dispatch, generated files, row IDs, metadata, support
   tier, and optional dependency context without duplicating QR solver tests.

4. **Normalizer tests stayed the row-state owner.** Existing and updated
   `tests/test_normalize_report_index.py` coverage handles complete, missing,
   unexpected, duplicate, stale, fail, and defer selected comparison rows.

5. **Diagnostics now scale to multiple artifacts.** Selected comparison
   row-set, non-pass, and missing-family errors name both selected study files
   rather than pointing reviewers only to `qr_minnorm`.

6. **Docs caught up to the evidence surface.** Public and maintainer docs now
   describe selected QR comparison freshness as two-family evidence, not
   minimum-norm-only evidence.

7. **Sprint 161 has a concrete handoff.** The partial-SVD comparison handoff
   starts with a low-risk source-controlled target and carries forward the
   descriptor, metadata, focused-test, normalizer, and non-claim pattern.

## What Didn't Go Well

1. **The prompt path was stale.** The request referenced Epic 12 even though
   Sprint 160 belongs to Epic 14. The sprint recorded this and proceeded from
   the current Epic 14 plan.

2. **Report integration happened earlier than the original day split.** Day 6
   had to close the selected-row consistency gap immediately after adding the
   second comparison family metadata, because strict freshness saw the new rows
   before Day 10.

3. **Single-artifact diagnostics lingered.** Day 13 found one remaining
   missing-family diagnostic that still pointed only at
   `build/comparison/qr_minnorm/study.tsv`. It was fixed and covered.

4. **Historical artifacts naturally mention earlier one-family behavior.**
   Day 1 and design artifacts preserve their original context, so current
   present-tense claims should be read from Day 14 closeout, docs, tests, and
   implementation.

5. **Hosted artifact naming may need a future pass.** Sprint 160 local
   freshness is two-family, but hosted artifact naming from prior sprints may
   still be QR-minnorm-oriented until a later hosted-publication pass updates
   it.

## Final Metrics

### Validation

| Metric | Sprint 160 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required | no |
| Python syntax compile | passed |
| runner self-check | passed |
| focused runner tests | passed: `python3 tests/test_run_external_comparison.py` |
| focused normalizer tests | passed: `python3 tests/test_normalize_report_index.py` |
| corpus schema validation | passed: `python3 scripts/validate_corpus_schema.py` |
| selected comparison freshness | passed: `make report-index-comparison-freshness` |
| final `git diff --check` | passed |
| trailing-whitespace scans | passed |

### Selected Comparison Surface

| Metric | Sprint 160 close state |
| --- | ---: |
| selected comparison targets | 2 |
| selected generated comparison rows | 12 |
| source-controlled comparison contract rows | 2 |
| selected comparison artifact groups | 2 |
| selected optional dependency pass rows | 0 |
| selected optional dependency defer rows | 4 generated rows across both targets |
| broad external-library parity claims | 0 |
| broad QR parity claims | 0 |

### Artifact Package

| Metric | Sprint 160 close state |
| --- | ---: |
| daily artifacts | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| Python scripts changed | 2 |
| focused Python test files added | 1 |
| normalizer test files changed | 1 |
| Makefile targets changed | 1 |
| report metadata files changed | 1 |
| public/maintainer docs changed | 4 |
| generated build/report artifacts committed | 0 |

## Closed Claim

Sprint 160 closes this QR comparison-family claim:

The project now has two selected fixture-local QR comparison families,
`qr-minnorm` for `qr_underdetermined_minnorm_2x4` and `qr-compatible-ls` for
`qr_overdetermined_compatible_5x3`, generated by the descriptor-backed external
comparison runner, represented in source-controlled report metadata, enforced
by selected comparison freshness, covered by focused runner and normalizer
tests, and documented with explicit local-only support tiers and non-claims.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-target-selection.md](./artifacts/day2-target-selection.md);
- [day3-metric-contract.md](./artifacts/day3-metric-contract.md);
- [day4-harness-design.md](./artifacts/day4-harness-design.md);
- [day5-harness-implementation.md](./artifacts/day5-harness-implementation.md);
- [day6-corpus-integration.md](./artifacts/day6-corpus-integration.md);
- [day7-test-design.md](./artifacts/day7-test-design.md);
- [day8-focused-tests.md](./artifacts/day8-focused-tests.md);
- [day9-report-design.md](./artifacts/day9-report-design.md);
- [day10-report-integration.md](./artifacts/day10-report-integration.md);
- [day11-docs-alignment.md](./artifacts/day11-docs-alignment.md);
- [day12-local-validation.md](./artifacts/day12-local-validation.md);
- [day13-evidence-review.md](./artifacts/day13-evidence-review.md);
- [day14-closeout.md](./artifacts/day14-closeout.md).

## Sprint 161 Readiness

Sprint 161 should begin from these settled Sprint 160 boundaries:

| Starting item | Required posture |
| --- | --- |
| Comparison runner | Descriptor-backed targets exist and should be reused. |
| Selected row semantics | Row IDs, metrics, tolerance, support tier, artifact path, and non-claims are defined before implementation. |
| Report metadata | Add source-controlled `report_families.tsv` rows before interpreting generated rows. |
| Freshness gate | Regenerate selected output before strict normalization. |
| Runner tests | Cover target dispatch, generated files, row IDs, metadata, support tier, and optional dependency context. |
| Normalizer tests | Cover complete, missing, unexpected, duplicate, stale, fail, and defer selected rows. |
| C proof ownership | Do not add C tests unless solver implementation or fixture helper behavior changes. |

Recommended Sprint 161 first step:

Start with a low-risk partial-SVD target such as `partial_svd_diag6_k2`, define
subspace-safe row semantics before implementation, and avoid raw
singular-vector identity or broad partial-SVD parity claims.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 160 close:

- broad QR parity against LAPACK, NumPy, SciPy, SuiteSparse, Eigen, or any
  external-library ecosystem;
- raw QR basis identity, Q sign/orientation identity, and broad rank-threshold
  policy;
- broad rank-deficient QR solve behavior beyond maintained fixture-local
  evidence;
- optional NumPy/SciPy promoted comparison pass evidence;
- macOS and Windows selected comparison freshness parity;
- hosted artifact naming updates for the two-family QR comparison surface;
- broad partial-SVD comparison publication, deferred to Sprint 161;
- package-manager, shared-library, ABI, dynamic-loader, and install proof from
  report freshness;
- performance superiority, release evidence, and state-of-the-art sparse
  linear algebra claims.

Still consciously constrained rather than silently solved:

- generated comparison rows remain fixture-local and `local_only`;
- generated report files under `build/` remain ignored;
- source-controlled report-family metadata remains contract/context, not pass
  evidence by itself;
- optional dependency defers remain context and are never selected pass
  evidence;
- selected comparison freshness proves only the named fixtures and row set.
