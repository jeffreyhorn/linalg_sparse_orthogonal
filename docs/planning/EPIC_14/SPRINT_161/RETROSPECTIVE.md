# Sprint 161 Retrospective

**Sprint:** 161 - Partial-SVD Comparison Publication Closure
**Duration:** 14 days (Days 1-14 landed on branch `sprint-161`)
**Status:** Complete

## Source Artifact Note

Sprint 161 was planned from the Epic 14 project-plan section for Sprint 161
and lives under `docs/planning/EPIC_14/SPRINT_161/` with its plan, working
notes, artifacts, and retrospective in one package. The original sprint prompt
referenced an older Epic 12 project-plan path; `WORKING_NOTES.md` records that
path mismatch for traceability.

## Definition Of Done Checklist

- [x] Created Sprint 161 plan, working notes, daily artifacts, closeout
      artifact, and retrospective.
- [x] Selected one bounded partial-SVD comparison target:
      `partial_svd_diag6_k2`.
- [x] Defined subspace-safe row semantics that avoid raw singular-vector
      identity, sign/orientation identity, and repeated-spectrum ordering
      claims.
- [x] Added descriptor-backed `partial-svd-diag6-k2` generation to
      `scripts/run_external_comparison.py`.
- [x] Preserved the existing selected QR comparison families:
      `qr-minnorm` and `qr-compatible-ls`.
- [x] Added source-controlled report metadata for
      `comparison/partial_svd_diag6_k2`.
- [x] Expanded selected comparison freshness from 12 generated QR rows to 22
      generated rows across three selected comparison families.
- [x] Updated `make report-index-comparison-freshness` to regenerate the two
      selected QR targets and the selected partial-SVD target before strict
      freshness normalization.
- [x] Added focused runner coverage for the new target, generated split
      artifacts, row IDs, metadata, support tier, and optional dependency
      context.
- [x] Tightened normalizer coverage so complete, missing, unexpected,
      duplicate, stale, fail, skip, and defer selected-row states are covered
      for the expanded selected row set.
- [x] Preserved optional NumPy/SciPy dependency rows as `defer` context only.
- [x] Aligned README, maintainer guide, solver-selection docs, corpus docs,
      and report-index schema docs with the selected QR plus partial-SVD
      comparison freshness surface.
- [x] Preserved non-claims for broad SVD/partial-SVD correctness, raw
      singular-vector identity, vector sign/orientation identity,
      external-library parity, hosted/release/platform/package/ABI proof,
      performance, and state-of-the-art evidence.
- [x] Published the Sprint 162 Windows package parity handoff.
- [x] Ran and recorded the final targeted validation set.

## What Went Well

1. **The first partial-SVD comparison stayed narrow.** Sprint 161 selected
   `partial_svd_diag6_k2`, a deterministic diagonal top-k fixture, and avoided
   harder first targets such as tall, nonsymmetric, repeated-spectrum,
   rank-deficient, sparse-output, and fail-closed families.

2. **The QR comparison pattern transferred cleanly.** The Sprint 160
   descriptor-backed runner model let the partial-SVD target share command
   dispatch, split artifact generation, summary, manifest, metadata, and
   freshness behavior instead of adding a parallel one-off path.

3. **Metrics avoided raw vector identity.** The selected rows use status,
   singular-value, residual, orthogonality, and diagonal projector diagnostics,
   which are reviewable without claiming singular-vector sign, orientation, or
   ordering identity.

4. **Normalizer semantics scaled to a third comparison family.** The selected
   comparison row set now includes the ten partial-SVD rows and the two QR
   families, and diagnostics name all selected artifacts when freshness fails.

5. **Focused tests matched the changed surface.** `tests/test_run_external_comparison.py`
   owns target dispatch and generated row shape; `tests/test_normalize_report_index.py`
   owns selected freshness row-state behavior.

6. **Docs caught up before closeout.** README, maintainer guide,
   solver-selection docs, corpus docs, and schema docs now describe selected
   comparison freshness as QR plus partial-SVD and preserve local-only
   boundaries.

7. **The next sprint handoff is cleanly separated.** Sprint 162 starts from a
   Windows package parity product decision, not from solver comparison
   evidence.

## What Didn't Go Well

1. **The prompt path was stale again.** The request referenced Epic 12 while
   the active Sprint 161 plan lives under Epic 14. The sprint recorded the
   mismatch and proceeded from the current Epic 14 plan.

2. **Report publication required careful sequencing.** Source-controlled
   metadata, generated rows, normalizer selected-row IDs, Makefile freshness,
   tests, and docs all had to agree before any generated comparison could be
   treated as selected evidence.

3. **The row count became less obvious.** Selected comparison freshness now
   reports 25 normalized rows: three source-controlled contract rows and 22
   generated rows. The docs and artifacts had to spell that out to avoid
   reviewer confusion.

4. **Optional dependency wording remains subtle.** NumPy/SciPy rows appear in
   dependency artifacts, but they remain `defer` context and cannot be read as
   pass evidence.

5. **Generated local artifacts show dirty-worktree provenance.** That is
   acceptable for local validation and review context, but it reinforces why
   these rows cannot be release, hosted, or platform proof.

## Final Metrics

### Validation

| Metric | Sprint 161 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | no |
| full C quality gate required | no |
| selected comparison freshness | passed: `make report-index-comparison-freshness` |
| selected oracle freshness | passed: `make report-index-oracle-freshness` |
| combined normalized index check | passed: `153 rows ok` |
| corpus schema validation | passed: `python3 scripts/validate_corpus_schema.py` |
| focused normalizer tests | passed: `python3 tests/test_normalize_report_index.py` |
| focused runner tests | passed: `python3 tests/test_run_external_comparison.py` |
| Python syntax compile | passed |
| final `git diff --check` | passed |
| trailing-whitespace scans | passed |

### Selected Comparison Surface

| Metric | Sprint 161 close state |
| --- | ---: |
| selected comparison targets | 3 |
| selected QR comparison targets | 2 |
| selected partial-SVD comparison targets | 1 |
| selected generated comparison rows | 22 |
| source-controlled comparison contract rows | 3 |
| selected comparison artifact groups | 3 |
| selected optional dependency pass rows | 0 |
| optional NumPy/SciPy promoted pass evidence | 0 |
| broad external-library parity claims | 0 |
| broad partial-SVD correctness claims | 0 |
| raw singular-vector identity claims | 0 |

### Artifact Package

| Metric | Sprint 161 close state |
| --- | ---: |
| daily artifacts | 14 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| Python scripts changed | 2 |
| focused Python test files changed | 2 |
| Makefile targets changed | 1 |
| report metadata files changed | 1 |
| public/maintainer docs changed | 5 |
| generated build/report artifacts committed | 0 |

## Closed Claim

Sprint 161 closes this partial-SVD comparison publication claim:

The project now has one selected fixture-local partial-SVD comparison family,
`partial-svd-diag6-k2` for `partial_svd_diag6_k2`, generated by the
descriptor-backed external comparison runner, represented in
source-controlled report metadata, enforced by selected comparison freshness,
covered by focused runner and normalizer tests, and documented with explicit
local-only support tiers and non-claims.

The positive evidence is limited to one diagonal top-k fixture compared
against the source-controlled dense SVD reference helper. The generated rows
check project status, baseline status, two singular values, max singular-value
delta, residual norm, U/V orthogonality, and U/V diagonal projector
diagnostics.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md);
- [day2-target-selection.md](./artifacts/day2-target-selection.md);
- [day3-metric-contract.md](./artifacts/day3-metric-contract.md);
- [day4-harness-design.md](./artifacts/day4-harness-design.md);
- [day5-harness-implementation.md](./artifacts/day5-harness-implementation.md);
- [day6-expected-rows.md](./artifacts/day6-expected-rows.md);
- [day7-test-design.md](./artifacts/day7-test-design.md);
- [day8-focused-tests.md](./artifacts/day8-focused-tests.md);
- [day9-report-design.md](./artifacts/day9-report-design.md);
- [day10-report-integration.md](./artifacts/day10-report-integration.md);
- [day11-docs-alignment.md](./artifacts/day11-docs-alignment.md);
- [day12-validation.md](./artifacts/day12-validation.md);
- [day13-evidence-review.md](./artifacts/day13-evidence-review.md);
- [day14-closeout.md](./artifacts/day14-closeout.md).

## Sprint 162 Readiness

Sprint 162 should begin from these settled Sprint 161 boundaries:

| Starting item | Required posture |
| --- | --- |
| Comparison evidence | Do not reuse solver comparison evidence as package, platform, ABI, or release proof. |
| Windows CMake install proof | Treat separately from Windows `pkg-config` and Makefile parity. |
| Windows `pkg-config` | Decide whether to promote a selected provider-backed proof or retain an explicit non-claim. |
| Windows Makefile parity | Decide independently from CMake install and `pkg-config`. |
| Static-first package contract | Preserve existing static-first package metadata and exact-version CMake downstream proof. |
| Docs and CI wording | Update only for the selected Windows package product decision. |

Recommended Sprint 162 first step:

Audit current Windows CMake install/downstream proof against Linux/macOS Make
install and `pkg-config` proof, then choose one narrow Windows package parity
decision before editing CI or docs.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 161 close:

- broad partial-SVD correctness beyond maintained fixture-local corpus/oracle
  and selected comparison rows;
- raw singular-vector identity, vector sign/orientation identity, and
  repeated-spectrum ordering;
- tall, nonsymmetric, repeated-spectrum, rank-deficient, sparse-output, and
  fail-closed partial-SVD comparison families;
- optional NumPy/SciPy promoted comparison pass evidence;
- LAPACK, SuiteSparse, Eigen, or broader external-library ecosystem parity;
- macOS and Windows selected comparison freshness parity;
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
- Sprint 162 Windows package parity must be earned by package/install evidence,
  not by Sprint 161 solver-comparison evidence.
