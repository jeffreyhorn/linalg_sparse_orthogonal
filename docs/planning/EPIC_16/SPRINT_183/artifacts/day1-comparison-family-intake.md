# Sprint 183 Day 1: Comparison Family Intake

## Purpose

Establish Sprint 183 scope, inherited selected comparison authority, current
selected families, candidate evaluation criteria, and artifact structure before
selecting an additional bounded external comparison family.

## Project-Plan Scope

Sprint 183 implements the Epic 16 project-plan section "Sprint 183: Additional
Bounded External Comparison Family".

| Item | Day 1 baseline |
| --- | --- |
| 183.1 Family Selection | Start candidate selection from claim risk, user value, fixture stability, and comparator availability. |
| 183.2 Fixture and Metric Contract | Future days must define fixture identity, expected rows, tolerances, metrics, skip/defer behavior, and non-parity wording before implementation. |
| 183.3 Harness Extension | Future days must extend `scripts/run_external_comparison.py` and focused tests only for the selected family. |
| 183.4 Report Integration | Future days must generate, index, freshness-check, and manifest-register the selected comparison report. |
| 183.5 Documentation Alignment | Future days must update README, solver-selection docs, maintainer guide, corpus/report docs, and claim boundaries. |
| 183.6 Validation | Future days must run comparison generation, selected freshness checks, script tests, and relevant C tests. |

## Inherited Selected Comparison Surface

The selected target manifest currently owns four selected external comparison
families:

| Selected target | Fixture | Rows | Current evidence |
| --- | --- | ---: | --- |
| `qr-minnorm` | `qr_underdetermined_minnorm_2x4` | 6 | QR minimum-norm fixture-local project-vs-baseline rows against the selected dense QR helper. |
| `qr-compatible-ls` | `qr_overdetermined_compatible_5x3` | 6 | QR compatible least-squares fixture-local rows against the selected dense QR helper. |
| `partial-svd-diag6-k2` | `partial_svd_diag6_k2` | 10 | Partial-SVD diagonal top-k rows for singular values, residual, orthogonality, and projector diagnostics. |
| `lu-nonsym-square-5` | `lu_nonsym_square_5` | 6 | Linked-list LU nonsymmetric square solve rows against the selected dense LU helper. |

These rows are selected for Linux and macOS hosted report freshness. They are
not selected Windows report freshness evidence.

## Sprint 182 Handoff Boundary

Sprint 182 formally deferred Windows report freshness. Sprint 183 can add an
additional bounded comparison family without changing that decision. Any
Windows promotion would require a separate, deliberate implementation path:

- Windows-safe generator invocation without Makefile or Bash assumptions;
- CMake/MSVC project probe support;
- `.lib` linkage and `.exe` temporary executable handling;
- exact Python executable proof on hosted Windows;
- exact selected artifact upload scope;
- selected target manifest metadata for Windows workflow file, job, artifact,
  platform, support tier, claim scope, and non-claims;
- workflow guard allowlist and documentation updates.

Day 1 keeps that boundary out of scope for the ordinary additional-family
selection path.

## Candidate Evaluation Criteria

| Criterion | Selection question |
| --- | --- |
| User value | Does the family help users choose or trust a solver path they already need? |
| Fixture stability | Can a deterministic small fixture avoid flaky numerical or platform behavior? |
| Comparator availability | Is a source-controlled dense helper available or straightforward to add? |
| Implementation cost | Can the family fit the current runner model without broad refactoring? |
| Validation cost | Can focused script and relevant C tests cover the selected family reliably? |
| Claim risk | Can the evidence remain fixture-local and avoid broad parity or performance claims? |
| Manifest fit | Can selected row IDs, expected rows, required files, artifact patterns, and non-claims be exact? |
| Workflow fit | Can Linux/macOS selected freshness uploads remain exact and fail-closed? |

## Initial Candidate Boundaries

Candidate inventory should avoid:

- broad solver-family correctness claims;
- raw external-library parity claims;
- package-manager or shared-library support claims;
- performance superiority or state-of-the-art claims;
- optional dependency pass evidence;
- Windows report freshness promotion by implication;
- broad generated artifact uploads or unbounded report families.

## Day 2 Handoff

Day 2 should audit the existing comparison runner, selected manifest rows,
workflow guards, tests, and generated artifact conventions before Day 3
inventories candidate families.

## Validation

Day 1 changes planning artifacts only. Validation:

- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Notes |
| --- | --- | --- |
| Sprint 183 scope is tied to the Epic 16 project plan. | Complete | Project-plan items 183.1 through 183.6 are mapped. |
| Existing selected comparison families and non-claims are explicit. | Complete | The four inherited selected comparison families and Sprint 182 Windows boundary are recorded. |
| Candidate selection starts from shared criteria rather than preference. | Complete | Evaluation criteria cover value, fixture stability, comparator availability, cost, validation, claim risk, manifest fit, and workflow fit. |
