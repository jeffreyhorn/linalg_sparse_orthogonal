# Sprint 187 Day 6: Closure Target Selection

## Purpose

Select the exact complete-gap targets Epic 17 will attempt to close in
Sprints 188 through 195. This artifact turns the Day 5 ranked feasibility
shortlist into the binding planning input for the remaining Sprint 187
acceptance-gate work.

## Selection Principles

- Prefer complete closure of a bounded gap over partial progress on a broader
  gap.
- Promote support or capability claims only when validation evidence exists.
- Treat missing validation as a support-tier boundary, not as a documentation
  inconvenience.
- Preserve explicit non-claims for broad platform, ABI, package, performance,
  comparison, and state-of-the-art surfaces.
- Define each selected target so one 14-day sprint can finish it.

## Selected Closure Targets

| Sprint | Project-plan title | Selected closure target | Required close state | Why selected |
| --- | --- | --- | --- | --- |
| Sprint 188 | Homebrew Proof Completion | Close the local Homebrew formula proof gap. | The formula install/test/uninstall path passes locally or the license/formula blocker is formalized with guard coverage and claim wording that prevents package-manager overstatement. | Highest adoption value, concrete blocker, existing proof script, local `brew` availability, and strong support-tier impact. |
| Sprint 189 | PowerShell Validation Ownership | Own validation for PowerShell snippets and workflow-facing scripts. | Hosted Windows or equivalent owned validation parses/runs the selected PowerShell surface; local absence of `pwsh` is documented as a skip condition rather than hidden coverage. | Required before credible Windows report freshness promotion and bounded enough to close as validation ownership. |
| Sprint 190 | Windows Selected Report Freshness Decision | Decide and close one Windows report freshness lane. | Either one Windows-safe report freshness path is validated and documented, or the deferral is renewed with stronger blockers, guards, and revisit criteria. | Depends on Sprint 189 and closes the remaining Windows report ambiguity without claiming broad Windows parity. |
| Sprint 191 | Bounded External Comparison Family | Add one additional external comparison family. | The new family has fixed fixtures, reference source, metrics, tolerances, selected manifest entries, freshness checks, and support-tier documentation. | Adds state-of-the-art proof value while keeping broad external parity a non-goal. |
| Sprint 192 | Methodology-Bound Performance Evidence Lane | Close one bounded performance evidence lane. | A selected benchmark lane has methodology metadata, runtime/variance policy, freshness validation, artifact policy, and calibrated wording that avoids portable performance claims. | Converts performance from broad aspiration into evidence-backed, bounded methodology. |
| Sprint 193 | Selected Large Review-Surface Reduction | Reduce one large source or test review surface. | One selected cluster is split or helperized with behavior-preserving tests, source-list guards, and explicit no-behavior-change invariants. | Improves maintainability without destabilizing the whole storage or solver architecture. |
| Sprint 194 | Adoption And API Coherence Simplification | Align adoption docs, support tiers, diagnostics, examples, and API entry points. | User-facing setup and API guidance consistently reflect proven support tiers, unsupported surfaces, diagnostics, and examples. | High user value, low implementation risk, and necessary before final claim calibration. |
| Sprint 195 | Selected Reliability And Failure-Path Proof | Add deterministic reliability proof for one selected owner. | The selected owner has deterministic failure-path coverage for cleanup, stale outputs, retry behavior, and global-state cleanup where applicable. | Strong proof value, but scoped to one owner so it can close fully. |

## Complete Definition Of Done By Target

| Sprint | Done requires | Done excludes |
| --- | --- | --- |
| Sprint 188 | Passing proof command or explicit guarded blocker; package docs and claim wording updated to match; package guards prevent unsupported promotion. | Homebrew/core submission, bottles, Linuxbrew parity, public tap maintenance, vcpkg, Conan, pkgsrc, distro packages. |
| Sprint 189 | Owned PowerShell validation command or hosted workflow; documented local `pwsh` skip semantics; failure path visible in CI/logs. | Broad Windows Makefile parity, full Windows package-manager support, all shell dialects. |
| Sprint 190 | One selected Windows freshness path promoted or renewed deferral recorded; guards prevent stale unsupported reports from being treated as current. | All report generators on Windows, broad Windows report parity, non-Windows workflow redesign. |
| Sprint 191 | One bounded family integrated with selected manifest/freshness evidence; tolerances and unavailable dependency behavior documented. | Broad parity across all external libraries, performance superiority claims, every solver family. |
| Sprint 192 | Methodology, hardware/compiler/runtime metadata, freshness checks, and calibrated wording for one selected lane. | Cross-platform performance leadership, architecture-independent thresholds, broad benchmark suite expansion. |
| Sprint 193 | One selected cluster reduced with behavior-preserving guard tests and source-list coherence. | Core storage replacement, broad refactor of all large files, ABI or API redesign. |
| Sprint 194 | Support/readiness entry point, examples, diagnostics, install/readme guidance, and API docs align with evidence. | Hosted API publication, marketing rewrite, unsupported platform promotion. |
| Sprint 195 | One owner has deterministic failure, cleanup, retry, and stale-output proof with validation. | Exhaustive reliability campaign, all allocators, all solvers, broad concurrency proof. |

## Explicit Non-Goal Register

| Non-goal | Reason it is rejected for Epic 17 default scope | Future revisit trigger |
| --- | --- | --- |
| Unqualified state-of-the-art sparse linear algebra library claim | Epic 17 can improve evidence but cannot close package, ABI, Windows, performance, comparison, storage, and release maturity broadly enough for an unqualified claim. | Multiple independent comparison families, portable performance evidence, release/package maturity, ABI policy, and platform validation all exist. |
| Shared-library and dynamic ABI guarantees | Requires symbol visibility policy, versioning, installed shared consumers, loader metadata, and platform-specific packaging. | A future epic explicitly chooses ABI stability and dynamic distribution as product goals. |
| Broad Windows parity | Local environment lacks `pwsh`; current plan only selects PowerShell validation and one report freshness decision. | Hosted Windows validation covers build, test, docs, reports, packages, and examples consistently. |
| Broad package-manager distribution | Local Homebrew formula proof is the selected complete gap; broader distribution needs provider-specific policy and maintenance. | Homebrew/core, bottles, Linuxbrew, vcpkg, Conan, pkgsrc, or distro maintainership is selected explicitly. |
| Broad external library parity | One bounded family is feasible; broad parity would require many dependencies, fixtures, metrics, and support policies. | A future evidence epic selects multiple libraries and solver families with dependency ownership. |
| Portable performance leadership | One methodology-bound lane can be credible; portable leadership requires architecture/compiler/matrix-family breadth. | A controlled benchmark program spans representative platforms, compilers, dependencies, and matrix classes. |
| Core storage-model replacement | Too invasive for Epic 17 and likely to destabilize public semantics. | A dedicated architecture epic accepts migration, compatibility, and performance risk. |
| Hosted generated API publication | Lower value than package, Windows, comparison, performance, maintainability, adoption, and reliability closure in this epic. | Documentation distribution becomes a release blocker or project website work is selected. |

## Dependency And Ordering Decisions

| Order | Decision |
| --- | --- |
| Sprint 188 before Sprint 194 | Package support wording must reflect the Homebrew proof result before adoption docs are simplified. |
| Sprint 189 before Sprint 190 | PowerShell validation ownership is required before promoting or renewing Windows report freshness. |
| Sprint 191 before final calibration | Additional comparison evidence informs the final state-of-the-art boundary. |
| Sprint 192 before final calibration | Performance evidence determines whether any performance wording can be promoted. |
| Sprint 193 before Sprint 195 only if owners overlap | Maintainability extraction should land before reliability proof if Day 11 and Day 13 select the same owner surface. |
| Sprint 194 before Sprint 196 | Adoption/API coherence provides final claim-calibration inputs. |
| Sprint 195 before Sprint 196 | Reliability evidence must be included in final closeout wording and residuals. |

## Fallback Policy

Each selected sprint must finish with one of two accepted close states:

1. A promoted claim backed by passing validation and updated documentation.
2. A retained non-claim backed by explicit blockers, guard coverage, and revisit
   criteria.

The following outcomes are rejected for Epic 17 closure:

- unsupported claims promoted because implementation work landed without proof;
- stale reports treated as current evidence;
- platform/package/performance support implied by examples alone;
- broad capability language that outruns selected validation;
- partial refactors that increase review surface without a measurable guard.

## Inputs To Days 7 Through 14

- Day 7 should convert Sprint 188 into package and Homebrew acceptance gates.
- Day 8 should convert Sprints 189 and 190 into Windows acceptance gates.
- Day 9 should convert Sprint 191 into comparison acceptance gates.
- Day 9 should also convert Sprint 192 into performance acceptance gates.
- Day 10 should convert Sprints 193 and 195 into maintainability and
  reliability acceptance gates.
- Day 11 should convert Sprint 194 into adoption and API acceptance gates.
- Day 12 should consolidate validation requirements into a quality surface map.
- Day 13 should package Sprints 188 through 195 implementation handoffs.
- Day 14 should consolidate all gates into the Sprint 187 retrospective inputs.

## Validation

Day 6 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
