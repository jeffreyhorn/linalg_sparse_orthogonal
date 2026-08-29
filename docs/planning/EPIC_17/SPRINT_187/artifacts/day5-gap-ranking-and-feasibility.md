# Sprint 187 Day 5: Gap Ranking and Feasibility

## Purpose

Rank the reconciled Epic 17 closure candidates by user value, proof value,
state-of-the-art relevance, implementation risk, validation difficulty,
support-tier impact, environment dependency risk, and likelihood of complete
closure inside one 14-day sprint.

## Ranking Method

Scores use a 1-5 scale:

- 5 = strongest positive value or highest risk;
- 3 = meaningful but bounded;
- 1 = low value or low risk.

For feasibility, higher means more likely to close completely inside the
planned sprint. For implementation risk, validation difficulty, and
environment risk, higher means more risk.

## Ranked Candidate Matrix

| Rank | Candidate | Source IDs | User value | Proof value | State-of-the-art relevance | Implementation risk | Validation difficulty | Support-tier impact | Environment risk | Complete-closure feasibility | Target sprint | Rationale |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Homebrew proof completion | `E17-GAP-001`, `R186-PKG-LICENSE` | 5 | 5 | 3 | 2 | 3 | 5 | 2 | 5 | Sprint 188 | The blocker is concrete, the proof script and package guards already exist, and local `brew` is available. This is the cleanest complete product-proof closure. |
| 2 | PowerShell validation ownership | `E17-GAP-002`, `R186-WIN-PWSH` | 4 | 4 | 3 | 3 | 3 | 4 | 4 | 4 | Sprint 189 | Local `pwsh` is unavailable, but hosted Windows can own the proof. Closing validation ownership is prerequisite to any credible Windows report freshness promotion. |
| 3 | Windows report freshness decision | `E17-GAP-003`, `R186-WIN-REPORT-FRESHNESS` | 4 | 4 | 3 | 3 | 4 | 4 | 4 | 3 | Sprint 190 | Feasibility depends on Sprint 189. It can still close completely by selecting one Windows-safe lane or renewing the formal deferral with stronger guards. |
| 4 | Bounded external comparison family | `E17-GAP-004`, `R186-BROAD-COMPARISON` | 4 | 5 | 4 | 3 | 3 | 3 | 3 | 4 | Sprint 191 | The existing runner, selected manifest, and freshness flow make one new family feasible while broad parity stays a non-claim. |
| 5 | Methodology-bound performance lane | `E17-GAP-005` | 4 | 4 | 4 | 3 | 4 | 3 | 3 | 3 | Sprint 192 | Existing benchmark reports and sentinels provide a base, but hosted runtime, variance, threshold, and artifact policy make this harder than another comparison row. |
| 6 | Selected review-surface reduction | `E17-GAP-006`, `R186-REVIEW-SURFACE-NEXT`, narrowed `E17-GAP-011` | 3 | 4 | 2 | 4 | 4 | 2 | 2 | 4 | Sprint 193 | Prior helper-extraction and guard patterns exist. The work is feasible if limited to one cluster with explicit no-behavior-change invariants. |
| 7 | Adoption and API coherence | `E17-GAP-007`, `E17-GAP-008`, retained `R186-HOSTED-API` context | 5 | 3 | 2 | 2 | 3 | 4 | 2 | 5 | Sprint 194 | High user value and mostly documentation/example scope. The risk is oversimplifying support caveats, not implementation complexity. |
| 8 | Selected reliability proof | `E17-GAP-009`, narrowed `E17-GAP-013` | 4 | 5 | 3 | 4 | 4 | 3 | 2 | 3 | Sprint 195 | Strong proof value, but feasibility depends on selecting one owner with controllable allocation/failure hooks and clear cleanup invariants. |

## Candidate Dependencies

| Candidate | Depends on | Why |
| --- | --- | --- |
| Homebrew proof completion | License metadata or alternate formula strategy | The proof script cannot reach install/test/uninstall success without a valid license path. |
| PowerShell validation ownership | Hosted Windows availability or local `pwsh` | Local `pwsh` is absent, so the proof must be hosted-owned or explicitly skippable locally. |
| Windows report freshness decision | PowerShell validation ownership | Promotion requires a Windows-safe generator path and workflow validation; otherwise the sprint should renew deferral. |
| Bounded external comparison family | Selected fixture and dependency policy | The runner can support a new family only after exact fixture, reference, metric, tolerance, and optional dependency behavior are selected. |
| Methodology-bound performance lane | Selected benchmark and runtime budget | Hosted CI can carry only bounded runtime; variance/threshold policy must be settled before promotion. |
| Selected review-surface reduction | Single cluster selection | Broad cleanup is too risky; feasibility depends on choosing one owner and preserving behavior. |
| Adoption and API coherence | Prior package/Windows/evidence decisions | User-facing docs should reflect the package, Windows, comparison, and performance outcomes instead of preselecting them. |
| Selected reliability proof | Owner selection and failure hook availability | The work is complete only if one owner can support deterministic failure, cleanup, stale-output, and retry evidence. |

## Candidates Not Selected By Default

| Candidate | Status | Reason |
| --- | --- | --- |
| `R186-HOSTED-API` hosted generated API publication | Long-horizon retained decision | Epic 16 intentionally selected local-only generated HTML. It is lower value than package, Windows, comparison, performance, maintainability, adoption, and reliability closures unless Day 6 explicitly trades for it. |
| Broad linked-list storage replacement | Long-horizon architecture program | Replacing the core storage identity would exceed one epic and risks destabilizing many public workflows. |
| Broad numerical robustness campaign | Long-horizon evidence program | Better handled one bounded fixture/family at a time through comparison and reliability sprints. |
| Shared-library and dynamic ABI support | Retained non-claim | Requires ABI policy, symbol visibility, loader metadata, installed shared consumer proof, and platform-specific package work. |
| Broad Windows parity | Retained non-claim | Sprint 189-190 should close selected PowerShell/report freshness work, not Makefile, `pkg-config`, package-manager, or broad platform parity. |
| Broad package-manager distribution | Retained non-claim | Sprint 188 can prove local Homebrew formula behavior, but Homebrew/core, bottles, Linuxbrew, taps, vcpkg, Conan, pkgsrc, and distro packages stay out of scope. |
| Unqualified state-of-the-art claim | Retained non-claim | Epic 17 can improve credibility, but broad state-of-the-art positioning still needs multiple classes of external, performance, platform, package, ABI, and release evidence. |

## Shortlist For Day 6 Selection

Day 6 should select these complete-gap targets unless a new blocker appears:

1. Sprint 188: Homebrew proof completion.
2. Sprint 189: PowerShell validation ownership.
3. Sprint 190: Windows selected report freshness decision.
4. Sprint 191: Bounded external comparison family.
5. Sprint 192: Methodology-bound performance evidence lane.
6. Sprint 193: Selected large review-surface reduction.
7. Sprint 194: Adoption and API coherence simplification.
8. Sprint 195: Selected reliability and failure-path proof.

## Selection Risks and Fallbacks

| Risk | Fallback |
| --- | --- |
| License metadata decision cannot be made in Sprint 188. | Record a formal alternate formula strategy or retain package-manager support as a guarded non-claim with stronger proof-blocker evidence. |
| Hosted Windows cannot run the selected PowerShell validation. | Keep local `pwsh` absence as an environment residual and make Sprint 190 renew deferral instead of promotion. |
| Windows report freshness exceeds runtime or artifact complexity. | Close Sprint 190 as a reviewed renewed deferral with explicit blockers, guards, and revisit criteria. |
| New comparison dependency is unavailable or flaky. | Select a source-controlled reference helper or dependency-optional comparison that records unavailable dependency status without creating pass evidence. |
| Hosted performance lane is noisy or too slow. | Keep timing rows threshold-free but require methodology metadata and artifact freshness. |
| Selected review-surface cluster proves too coupled. | Select a smaller helper boundary inside the same cluster or defer broad extraction with a documented no-change decision. |
| Adoption simplification removes necessary caveats. | Keep support/readiness matrix as the compact entry point and link to owner-specific caveats instead of deleting them. |
| Reliability owner lacks controllable failure hooks. | Select a different owner with deterministic fail-at-count coverage potential. |

## Validation

Day 5 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
