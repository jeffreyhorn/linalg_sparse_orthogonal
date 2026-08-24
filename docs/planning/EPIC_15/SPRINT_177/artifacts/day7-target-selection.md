# Sprint 177 Day 7: Closure Target Selection

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Requested sprint path:** `docs/planning/EPIC_15/SPRINT_177/`
**Status:** Complete

## Purpose

Select the exact Epic 16 closure targets for Sprints 178-186 using the Day 3
residual classification and Day 6 populated evidence/status matrix. This
selection favors complete, bounded closure over partial progress on broad
state-of-the-art, package, ABI, platform, or ecosystem claims.

## Selection Inputs

- Day 2 residual queue.
- Day 3 residual classification matrix.
- Day 4 repository surface inventory.
- Day 5 evidence/status matrix schema.
- Day 6 populated evidence/status matrix.
- `docs/planning/EPIC_16/PROJECT_PLAN.md` Sprint 178-186 plan.

## Selected-Gap Register

| Sprint | Selected target | Residuals | Matrix rows | Closure decision |
| --- | --- | --- | --- | --- |
| 178 | Allocation-failure proof batch 2 | S177-R01 | ESM-010 | Select one additional subsystem beyond iterative repeated-run handles and close it with deterministic cleanup proof, focused tests, a validation target, and scoped docs. |
| 179 | Generated API HTML publication decision | S177-R02 | ESM-005, ESM-011 | Close generated API HTML status by selecting hosted publication, retained artifact, committed output, or stronger local-only enforcement, then align navigation and freshness checks. |
| 180 | Package-manager provider decision | S177-R03 | ESM-002, ESM-003, ESM-004 | Select one provider proof path or record a stronger provider deferral with exact blockers, guard updates, and public non-claim wording. |
| 181 | Selected report target manifest | S177-R10 | ESM-006, ESM-007, ESM-009, ESM-013 | Centralize selected oracle, comparison, performance, artifact, expected-row, and support-tier metadata so workflow/report guards stop duplicating lists. |
| 182 | Windows report freshness decision | S177-R05 | ESM-008, ESM-012, ESM-013 | Promote one Windows-safe generated report freshness path or close Windows report freshness as a guarded product deferral. |
| 183 | Additional bounded external comparison family | S177-R07 | ESM-007, ESM-013 | Add one fully maintained fixture-local comparison family with metrics, indexed rows, freshness checks, manifest registration, and non-parity wording. |
| 184 | Public header coherence batch 3 | S177-R09 | ESM-005, ESM-011 | Normalize one high-impact public header family without declaration drift and align API reference/user docs. |
| 185 | Large test/source review-surface reduction | S177-R13 | ESM-014, ESM-013 | Reduce one large review surface by extracting helpers or proof-owner files while preserving behavior and build registration. |
| 186 | Final validation, claim calibration, and closeout | S177-R14 plus all selected rows | ESM-001 through ESM-014 | Reconcile completed evidence, recalibrate public claims, run integrated validation, and publish closeout/residual records. |

## Per-Target Closure Rationale

### Sprint 178: Allocation-Failure Proof Batch 2

Complete closure is feasible because the existing Sprint 176 proof establishes
private injection semantics, cleanup expectations, and a focused Make target.
The next sprint must select exactly one subsystem, define failure sites,
prove no stale state publication, prove retry behavior, and document that the
result is still not broad allocation-failure coverage.

### Sprint 179: Generated API HTML Publication Decision

Complete closure is feasible because the current state is already explicit:
generated API HTML is local-only, ignored, untracked, and validated by
`make api-docs-freshness`. The sprint can close the gap by either promoting a
publication mechanism or hardening the local-only product decision with
navigation and stale-output guards.

### Sprint 180: Package-Manager Provider Decision

Complete closure is feasible only if the target is a decision, not broad
provider support. The sprint may prove one static-first provider path if a
provider is viable, or it may close the gap with a stronger formal deferral,
blocker evidence, provider-recipe absence checks, and updated public wording.

### Sprint 181: Selected Report Target Manifest

Complete closure is feasible because the duplicated target-list problem has a
bounded set of selected oracle, comparison, and performance targets. A
source-controlled manifest can own selected targets, expected rows, artifacts,
support tiers, duplicate detection, and workflow guard expectations.

### Sprint 182: Windows Report Freshness Decision

Complete closure is feasible if limited to one Windows-safe freshness target
or an explicit guarded deferral. The sprint must not claim broad Windows
report parity; it should either add one reviewed CMake-first Windows freshness
lane or publish exact blockers and guards that prevent accidental claims.

### Sprint 183: Additional Bounded External Comparison Family

Complete closure is feasible because the comparison runner, reference-helper
pattern, report-index path, and selected freshness gates already exist. The
new work must add one family only, with source-controlled fixtures, metrics,
tolerances, expected rows, manifest registration, and docs that reject broad
external-library parity.

### Sprint 184: Public Header Coherence Batch 3

Complete closure is feasible because prior header batches established
declaration-preserving cleanup patterns. The sprint should pick one family,
capture declaration baselines, normalize contract wording, update examples or
docs, and run C/header plus docs validation if declarations or comments move.

### Sprint 185: Large Test/Source Review-Surface Reduction

Complete closure is feasible if exactly one cluster is selected. The best
candidates from Day 4 are large solver tests or report tooling where helper
extraction can reduce review cost without changing behavior. The sprint must
include source-list/build registration validation for any new files.

### Sprint 186: Final Validation And Claim Calibration

Complete closure is feasible because it is a reconciliation sprint, not a new
feature sprint. The sprint must update public claims only where evidence was
earned, retain non-claims for unclosed broad gaps, run integrated validation,
and publish Epic 16 closeout artifacts.

## Explicit Non-Goal Register

| Non-goal | Reason it stays rejected in Epic 16 |
| --- | --- |
| Broad state-of-the-art sparse linear algebra claim | Requires named competitors, versions, workloads, metrics, platforms, package provenance, and recurring evidence across many solver families. |
| Broad external-library parity | One additional comparison family is selected; LAPACK, NumPy, SciPy, SuiteSparse, MKL, Eigen, or other broad parity remains unearned. |
| Portable performance or performance superiority | Existing performance evidence is selected and methodology-bound; it is not a raw timing gate or superiority claim. |
| Shared-library support | Static-first-only package posture remains selected unless a separate full ABI/export/import project is funded. |
| Dynamic ABI compatibility | Exact package version metadata is not an ABI stability guarantee. |
| Runtime-loader behavior | No selected loader metadata, shared artifact, or installed shared consumer validation exists. |
| Broad package-manager support | Sprint 180 selects one provider decision or stronger deferral, not full provider ecosystem support. |
| Broad Windows parity | Windows remains CMake-first unless a selected target explicitly adds proof. No Windows Makefile or pkg-config execution parity is selected. |
| Broad generated-report parity | Selected report targets can be manifest-governed; all generated families and all platforms are not selected. |
| Broad allocation-failure safety | Sprint 178 adds one subsystem proof only; it does not cover every allocation path. |
| Whole-library API redesign or ABI freeze | Sprint 184 selects one public header family and preserves declarations unless validation says otherwise. |
| Whole-repository maintainability closure | Sprint 185 reduces one review surface only. |

## Target Dependencies

| Dependency | Effect |
| --- | --- |
| Sprint 181 before Sprint 182 and Sprint 183 | The selected-target manifest should own workflow/report metadata before Windows report status or a new comparison family changes selected targets. |
| Sprint 177 gates before Sprint 178-180 | Allocation, API, and package decisions need acceptance gates before implementation begins. |
| Sprint 179 before broad docs navigation updates | Generated API status determines what public docs can point to. |
| Sprint 180 before any provider wording promotion | Package-manager wording must remain deferred until a provider proof or stronger deferral is complete. |
| Sprint 186 after all implementation sprints | Final claim calibration must reflect actual completed evidence rather than planned evidence. |

## Acceptance Gate Handoff

Day 8 should produce detailed gates for:

- allocation-failure proof batch 2;
- generated API publication or local-only status;
- package-manager provider proof or deferral;
- selected report target manifest;
- Windows report freshness promotion or deferral.

Day 9 should produce detailed gates for:

- additional bounded comparison family;
- public header coherence;
- review-surface reduction;
- final claim calibration and closeout.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected targets match the Epic 16 project plan | Complete | Selected-gap register maps directly to Sprints 178-186. |
| Each selected target has a full-closure path | Complete | Per-target rationale defines bounded proof, decision, guard, or reconciliation closure. |
| Broad claims remain rejected unless funded | Complete | Explicit non-goal register rejects broad state-of-the-art, parity, ABI, platform, package, report, and performance claims. |
