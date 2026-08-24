# Sprint 177 Day 3: Residual Classification Matrix

## Purpose

Day 3 classifies the Day 2 residual queue by user value, claim risk,
implementation risk, testability, hosted/local evidence need, estimated
sprint cost, and closure quality. This creates an evidence-based shortlist
for later target selection without finalizing Sprint 178-186 scope before the
repository surface inventory and evidence/status matrix are complete.

## Classification Rubric

| Column | Scale | Meaning |
| --- | --- | --- |
| User value | High / Medium / Low | Expected adoption, maintainer, or user-confidence impact if closed. |
| Claim risk | High / Medium / Low | Risk that the current gap causes overclaiming, ambiguous support tiers, or misleading public wording. |
| Implementation risk | High / Medium / Low | Risk of invasive code, workflow, package, or documentation churn. |
| Testability | High / Medium / Low | Practicality of adding a deterministic local or hosted validation gate. |
| Evidence need | Hosted / Local / Hybrid / Decision | Whether closure requires hosted CI evidence, local validation, both, or a product decision/deferral. |
| Estimated sprint cost | 1 / 2 / 3+ | Approximate number of 14-day sprints needed for complete closure. |
| Closure quality | Complete / Narrow / Defer | Whether the gap can be fully closed in Epic 16, only narrowed, or should remain deferred. |

## Residual Classification Matrix

| ID | Residual | User value | Claim risk | Implementation risk | Testability | Evidence need | Est. sprint cost | Closure quality | Rationale |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| S177-R01 | Broader allocation-failure coverage beyond iterative repeated-run handles | High | High | Medium | High | Hybrid | 1 | Complete | One additional subsystem can be selected, instrumented, documented, and gated without claiming broad allocator reliability. |
| S177-R02 | Generated API HTML hosted publication or stronger local-only status | High | Medium | Medium | High | Decision / Hybrid | 1 | Complete | A publication or local-only decision has clear owner files and validation; it improves API discoverability or removes ambiguity. |
| S177-R03 | Package-manager provider support or stronger provider deferral | High | High | High | Medium | Decision / Hybrid | 1-2 | Narrow | One provider proof may fit one sprint if scoped tightly; full ecosystem support and upgrade behavior exceed one sprint. |
| S177-R04 | Shared-library and dynamic ABI product support | High | High | High | Medium | Hosted / Hybrid | 3+ | Defer | Actual shared-library ABI support requires symbol/export policy, loader behavior, ABI tests, package metadata, and cross-platform consumers. |
| S177-R05 | Windows generated report freshness | Medium | Medium | Medium | Medium | Hosted / Decision | 1 | Complete | One Windows-safe report lane or a stronger deferral can close the Windows report-freshness question for a selected family. |
| S177-R06 | Selected oracle freshness beyond Linux | Medium | Medium | Medium | Medium | Hosted | 1 | Narrow | macOS selected oracle freshness can be promoted if runtime is acceptable, but broad oracle/report parity remains out of scope. |
| S177-R07 | Additional bounded external comparison family | High | High | Medium | High | Hybrid | 1 | Complete | One fixture-local family with metrics, tolerances, report rows, and docs is a strong complete-gap closure. |
| S177-R08 | Portable or broader performance publication | Medium | High | High | Medium | Hosted / Hybrid | 2+ | Narrow | A second selected hosted row is feasible, but portable performance methodology and superiority claims remain broader than one sprint. |
| S177-R09 | Public-header coherence breadth | High | Medium | Medium | High | Local / Hybrid | 1 | Complete | One header family can be cleaned with declaration-preserving checks and docs alignment. |
| S177-R10 | Workflow and selected report target-list duplication | High | High | Medium | High | Local / Hosted-guarded | 1 | Complete | A canonical selected-target manifest can reduce recurring false-positive/false-negative workflow guard risk. |
| S177-R11 | Broad generated report hosting/freshness | Medium | High | High | Medium | Hosted / Hybrid | 3+ | Defer | Broad report-family hosting would require runtime/support policy for many generated families. |
| S177-R12 | Release packaging evidence and package-provider upgrade behavior | Medium | High | High | Low | Hosted / Release | 3+ | Defer | Release provenance, signing, upgrade, and registry behavior require a release/product program, not a baseline sprint. |
| S177-R13 | Large source/test review surface and maintainability drag | Medium | Medium | Medium | High | Local | 1 | Complete | One selected helper extraction can reduce review cost while preserving behavior with existing tests. |
| S177-R14 | Claim governance remains distributed | High | High | Low | High | Local | 1 | Complete | The evidence/status matrix can become the current claim/status authority and reduce distributed wording risk. |

## Complete-Closure Candidate Shortlist

These candidates appear suitable for complete closure in one 14-day sprint
each, subject to Day 4-6 evidence confirmation:

| Candidate | Residual IDs | Why it is a complete-closure candidate |
| --- | --- | --- |
| Evidence/status matrix and claim authority | S177-R14 | Low implementation risk, high claim-governance value, and directly required by Sprint 177. |
| Allocation-failure proof batch 2 | S177-R01 | Builds on the Sprint 176 proof pattern and can stay scoped to one subsystem. |
| Generated API HTML status decision | S177-R02 | Can end with either hosted publication or stronger local-only enforcement. |
| Selected report target manifest | S177-R10 | Directly addresses recurring workflow/report guard drift with a manifest-driven authority. |
| Windows report freshness decision | S177-R05 | Can close one selected Windows report path or publish a stronger guarded deferral. |
| Additional bounded comparison family | S177-R07 | Existing comparison harness makes one fixture-local family realistic. |
| Public-header coherence batch | S177-R09 | Prior declaration-preserving workflows make one more family feasible. |
| Large review-surface reduction | S177-R13 | One no-behavior-change extraction is bounded and testable. |

## Narrow-Or-Conditional Candidates

| Candidate | Residual IDs | Why it is conditional |
| --- | --- | --- |
| Package-manager provider path | S177-R03 | One provider may be feasible, but full provider availability and upgrade behavior are not. A decision or prototype proof should be scoped carefully. |
| Selected oracle freshness beyond Linux | S177-R06 | macOS selected oracle freshness may be feasible, but it competes with Windows report freshness and selected-target manifest work. |
| Second hosted performance row | S177-R08 | A single additional row can be added, but portable performance publication remains unearned. |

## Defer Candidates And Partial-Progress Traps

| Residual IDs | Defer reason |
| --- | --- |
| S177-R04 | Shared-library and dynamic ABI support is a multi-platform product and compatibility project. Selecting it without enough budget would produce only partial scaffolding. |
| S177-R11 | Broad generated report hosting/freshness across every family would require family-by-family runtime and support-tier ownership. |
| S177-R12 | Release packaging, provider upgrade behavior, provenance, and registry readiness exceed the current baseline/evidence epic shape. |
| Broad state-of-the-art and ecosystem parity non-goals | These require named competitors, versions, fixtures, metrics, platforms, package provenance, and recurring hosted evidence across many dimensions. |

## Dependency Notes

| Dependency | Impact |
| --- | --- |
| S177-R14 evidence/status matrix before final target selection | The matrix should be populated before finalizing claim-bearing Epic 16 selections. |
| S177-R10 selected report target manifest before S177-R05/S177-R07 expansion | Centralizing target metadata first reduces drift if Windows report freshness or another comparison family is added. |
| S177-R02 generated API status before S177-R09 header cleanup closeout | Header cleanup improves generated API inputs, but publication/local-only status determines final navigation wording. |
| S177-R01 allocation-failure proof before broad failure claims | A second proof can widen the evidence surface only to the selected subsystem, not the whole library. |
| S177-R03 package-provider decision depends on static-first boundaries | Provider work must preserve static-first package and no shared-library/dynamic ABI non-claims. |
| S177-R13 review-surface reduction depends on Day 4 file inventory | The exact cluster should not be selected before repository surface inventory identifies the best candidate. |

## Provisional Sprint-Mapping Implications

The Day 3 matrix supports the Epic 16 project-plan shape:

- Sprint 177 should continue baseline, matrix, gate, quality-map, and handoff
  work.
- Sprint 178 is well aligned with S177-R01 allocation-failure proof batch 2.
- Sprint 179 is well aligned with S177-R02 generated API status closure.
- Sprint 180 should treat S177-R03 as a provider decision/prototype or a
  stronger deferral, not broad package-manager support.
- Sprint 181 is strongly supported by S177-R10 selected report target
  manifest.
- Sprint 182 is supported by S177-R05 Windows report freshness decision.
- Sprint 183 is supported by S177-R07 additional bounded comparison family.
- Sprint 184 is supported by S177-R09 public-header coherence.
- Sprint 185 is supported by S177-R13 large review-surface reduction.
- Sprint 186 should reconcile S177-R14 and all completed evidence into final
  claim calibration.

## Day 3 Deliverables

- Residual classification matrix.
- Complete-closure candidate shortlist.
- Conditional/narrow candidate list.
- Defer and partial-progress trap list.
- Dependency notes.

## Completion Criteria Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every selected-candidate discussion has explicit closure reasoning | Complete | Shortlist and matrix rationale explain each candidate. |
| High-risk but non-closeable gaps are marked as deferred | Complete | Shared-library/dynamic ABI, broad report hosting, release packaging, and broad ecosystem claims are deferred. |
| Candidate selection is based on evidence rather than novelty | Complete | Classification uses Day 2 residuals, current blockers, testability, and closure quality. |

