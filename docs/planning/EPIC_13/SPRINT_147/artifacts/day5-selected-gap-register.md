# Sprint 147 Day 5 Selected Gap Register

## Purpose

Day 5 selects the Epic 13 gaps that are plausible to close completely across
Sprints 148-156. The selection starts from the Day 4 R1-R14 residual intake and
prefers complete closure of a smaller set of evidence-backed gaps over shallow
movement across every residual.

## Selection Criteria

| Criterion | Meaning |
| --- | --- |
| Product value | The gap affects user trust, platform usability, packaging, numerical credibility, or claim discipline. |
| Feasibility | The gap can plausibly close within one assigned sprint or a clearly sequenced set of sprints. |
| Evidence maturity | Prior Epic 12 artifacts, tests, docs, CI lanes, or report tooling already provide a usable starting point. |
| Closure risk | The gap has clear non-claims and stop conditions that prevent accidental broad support language. |

Scores are directional: `5` is strongest, `1` is weakest. Risk score means
lower closure risk, not higher danger.

## Candidate Ranking

| Rank | Residuals | Gap | Product Value | Feasibility | Evidence Maturity | Closure Risk | Decision |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | R2 | Windows staged test portability closure | 5 | 4 | 4 | 4 | Select for Sprint 148. |
| 2 | R5 | Broader maintained QR corpus family | 5 | 4 | 4 | 4 | Select for Sprint 150. |
| 3 | R6 | Broader maintained partial-SVD corpus family | 5 | 4 | 4 | 4 | Select for Sprint 151. |
| 4 | R3 | Windows reviewed install-validation parity decision | 5 | 4 | 3 | 4 | Select for Sprint 149 after R2 intake. |
| 5 | R7 | Selected generated report freshness publication | 4 | 4 | 4 | 4 | Select for Sprint 152 after QR/SVD families are known. |
| 6 | R4 | Shared-library ABI productization decision | 5 | 3 | 3 | 3 | Select for Sprint 153 as implementation or stronger deferral. |
| 7 | R12 | First narrow external comparison harness | 5 | 3 | 2 | 3 | Select for Sprint 154 after corpus expansion. |
| 8 | R8, R9 | Tutorial, header cleanup, and API reference coherence | 4 | 4 | 4 | 4 | Select for Sprint 155 after claim-bearing work lands. |
| 9 | R13 | State-of-the-art competitive decision | 5 | 2 | 1 | 2 | Select only as Sprint 156 claim recalibration; broad claim remains blocked. |
| 10 | R1 | Branch-specific hosted CI reconciliation | 3 | 4 | 3 | 4 | Duplicate final evidence task; fold into Sprint 156. |
| 11 | R10, R11 | Runtime/backend typed-control and sentinel expansion | 3 | 3 | 3 | 3 | Defer unless needed by Sprint 152 report freshness. |
| 12 | R14 | Package-manager distribution | 4 | 2 | 1 | 2 | Defer behind R4 product decision. |

## Selected Epic 13 Gaps

| Selected Gap | Residuals Covered | Sprint Owner | Closure Definition |
| --- | --- | --- | --- |
| Windows staged test portability | R2 | Sprint 148 | Promoted or intentionally replaced Windows-compatible coverage for staged pthread/POSIX test surfaces, with CMake/CI registration and support-tier docs aligned. |
| Windows install-validation parity decision | R3 | Sprint 149 | Reviewed Windows install-validation parity is explicitly promoted or rejected without claiming Windows Makefile or `pkg-config` parity. |
| QR maintained corpus expansion | R5 | Sprint 150 | Multiple bounded QR fixture families receive metadata, expected rows, focused proof owners, oracle/report integration, docs, and validation. |
| Partial-SVD maintained corpus expansion | R6 | Sprint 151 | Multiple bounded partial-SVD fixture families receive subspace-safe comparisons, metadata, expected rows, focused proof owners, oracle/report integration, docs, and validation. |
| Generated report freshness publication | R7 | Sprint 152 | Selected claim-bearing generated families receive stable generation commands and freshness gates while local generated rows remain local evidence. |
| Shared-library ABI product decision | R4 | Sprint 153 | The project either implements first supported shared-library proof or publishes a stronger tested static-first deferral with exact blockers. |
| External comparison harness and first narrow study | R12 | Sprint 154 | One narrow comparison target is implemented with pinned external dependencies, metrics, tolerances, report rows, caveats, and non-claims. |
| Adoption/API coherence closeout | R8, R9 | Sprint 155 | Tutorial, selected public headers, and API reference guidance align with earned claims and declaration-preservation checks. |
| Final claim recalibration and closeout | R1, R13 | Sprint 156 | Hosted evidence is reconciled, earned claims are recalibrated, residuals are published, and broad state-of-the-art remains rejected unless directly earned. |

## Non-Selected Residuals

| Residual | Disposition | Reason | Future Gate |
| --- | --- | --- | --- |
| R10: Runtime/backend typed-control promotion review | Deferred | Epic 13 already prioritizes Windows, corpus, report, ABI, comparison, adoption, and closeout. A typed control could widen ABI/API obligations and dilute closure. | Select one control with API design, ABI review, tests, docs, and package non-claim review. |
| R11: Additional runtime/backend sentinel rows | Deferred | Sentinel expansion overlaps R7, but standalone runtime sentinel growth is not required for the selected Epic 13 claims. | Add rows only when a selected runtime or report claim names fixtures, metrics, budgets, variance policy, freshness policy, and non-claims. |
| R14: Package-manager distribution | Deferred | Package-manager support depends on a stable shared/static product contract, release/versioning policy, recipe ownership, and update/uninstall proof. | Reopen after R4 decides shared-library/static-first posture and package metadata is stable. |

R1 is not non-selected; it is folded into Sprint 156 because branch/PR hosted
CI reconciliation is a final evidence activity rather than an independent
implementation gap. R13 is selected only as a final competitive decision gate,
not as a commitment to earn a broad state-of-the-art claim.

## Duplicate Fences

| Fence | Rule |
| --- | --- |
| Windows support wording | R2 staged test portability and R3 install-validation parity must remain separate. Promoting one does not imply the other. |
| Windows package parity | Sprint 149 may promote or reject reviewed Windows CMake install/downstream parity, but it must not imply Windows Makefile or Windows `pkg-config` parity. |
| ABI and distribution | R4 owns shared-library/static-first product posture. R14 package-manager recipes remain out of scope until that posture stabilizes. |
| QR corpus and comparison | Sprint 150 corpus proof must land before QR is used as an external comparison target. Fixture-local QR rows do not imply broad QR parity. |
| Partial-SVD corpus and comparison | Sprint 151 subspace-safe corpus proof must land before partial-SVD is used as an external comparison target. Fixture-local SVD rows do not imply broad SVD parity. |
| Generated freshness | Sprint 152 may require generated rows only for selected claim-bearing families. Benchmark, sentinel, coverage, dead-code, and guardrail rows remain advisory unless explicitly selected. |
| Adoption docs | Sprint 155 may reorganize tutorial and header docs only around earned evidence. It must not create new platform, package, solver, performance, or ABI claims. |
| Competitive claims | Sprint 156 may approve only narrow claims backed by Sprint 154 comparative evidence. Broad state-of-the-art remains rejected by default. |

## Sprint-To-Gap Map

| Sprint | Selected Gap | Dependencies | Primary Completion Signal |
| --- | --- | --- | --- |
| 148 | Windows staged test portability | Day 7 Windows evidence gate and Day 12 quality map. | Windows CMake lane intentionally registers/executes promoted coverage or records explicit rejected paths. |
| 149 | Windows install-validation parity decision | Sprint 148 platform outcome and package evidence gate. | Reviewed Windows install-validation parity is promoted or rejected with hosted proof/docs/report rows aligned. |
| 150 | QR maintained corpus expansion | Day 8 corpus evidence gate and Sprint 139 fixture-local closure. | New QR fixture families have source-controlled metadata, focused proof owners, oracle/report rows, and validation. |
| 151 | Partial-SVD maintained corpus expansion | Sprint 150 corpus lessons and Sprint 140 fixture-local closure. | New partial-SVD fixture families have subspace-safe proof owners, oracle/report rows, and validation. |
| 152 | Generated report freshness publication | Sprints 150-151 generated rows and Day 9 freshness gate. | Required generated families pass selected `--require-generated` freshness checks. |
| 153 | Shared-library ABI product decision | Sprint 149 package parity decision and static-first proof. | Shared support lands with proof or static-first deferral is stronger and tested. |
| 154 | External comparison harness and narrow study | Sprints 150-152 maintained fixtures/reports. | One comparison study names libraries, versions, fixtures, metrics, tolerances, caveats, and non-claims. |
| 155 | Tutorial/header/API coherence | Earned claims from Sprints 148-154. | Tutorial and selected headers align with the current support surface and declaration-preservation evidence. |
| 156 | Final validation, claim recalibration, residual publication | Sprints 148-155 complete or explicitly deferred. | Final evidence inventory, hosted CI reconciliation, claim audit, residual queue, and Epic retrospective are published. |

## Feasibility And Risk Notes

- The selected set is aggressive but coherent because each sprint has a single
  dominant closure target.
- Windows work is deliberately split into staged-test portability and
  install-validation parity to prevent one support-tier claim from masking the
  other.
- QR and partial-SVD expansion are selected before external comparison so the
  comparison harness uses maintained fixture families instead of ad hoc inputs.
- Generated freshness follows numerical corpus expansion so `--require-generated`
  gates are tied to real claim-bearing rows.
- ABI work is framed as a product decision, not a promise to implement shared
  libraries if blockers remain.
- Adoption work is late in the epic so it can reflect earned evidence rather
  than planned evidence.
- Broad state-of-the-art remains blocked unless the first narrow comparison
  study justifies a carefully bounded claim.

## Day 6 Handoff

Day 6 should convert selected gaps into candidate earned claims and explicit
non-claims. It should preserve these default positions:

- Windows parity claims require hosted Windows evidence.
- Broad QR and partial-SVD claims require multiple maintained fixture families,
  not single-fixture evidence.
- Generated report freshness is local unless a hosted artifact policy is added.
- Shared-library ABI support is not earned unless Sprint 153 implements and
  validates it.
- External-library parity and state-of-the-art claims remain rejected unless
  direct comparison evidence supports a narrow statement.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected gaps can plausibly close within Epic 13. | Complete | Selected gaps map one-to-one to Sprints 148-156 with prerequisite ordering. |
| Non-goals are explicit and defensible. | Complete | R10, R11, and R14 remain deferred; broad R13 claim remains blocked. |
| Sprint 148-156 sequencing follows dependencies. | Complete | Sprint-to-gap map orders Windows, corpus, freshness, ABI, comparison, adoption, and final closeout work by prerequisite evidence. |
