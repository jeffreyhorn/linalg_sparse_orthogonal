# Day 1 Adoption QA Intake

## Purpose

Day 1 establishes the Sprint 116 adoption QA boundary before any content
changes. The sprint is an adoption-surface review sprint: it validates
references, scanability, audience fit, and evidence-bounded claims without
pulling implementation, package/platform parity, ABI, or helper-abstraction
work into scope.

## Inputs Reviewed

| Input | Reviewed for | Day 1 outcome |
|---|---|---|
| `docs/planning/EPIC_10/PROJECT_PLAN.md` Sprint 116 section | Sprint goal, prerequisites, items, estimates, and deliverables | Confirmed seven project-plan items and 56-hour total. |
| `docs/planning/EPIC_10/SPRINT_116/PLAN.md` Day 1 | Required Day 1 tasks and completion criteria | Confirmed working notes, artifact directory, inventory, exclusions, and owner map are required. |
| Sprint 111 retrospective | Adoption docs handoff and residual adoption debt | Treat completed adoption docs as audit surfaces, not rebuild targets. |
| Sprint 112 retrospective | Package/platform support truth | Use reviewed support truth as install and platform wording guardrail. |
| Sprint 113 retrospective | Behavior and proof-owner closeout | Avoid advertising unproven behavior ownership or internals. |
| Sprint 114 retrospective | Proof-owner, source-boundary, and non-package residuals | Preserve non-claims around source movement and helper abstraction. |
| Sprint 115 retrospective | Package/platform residual decisions | Preserve static-first, package-manager, ABI, Windows, macOS, and install-lane boundaries. |

## Adoption-Surface Inventory

| Surface | File | Sprint 116 QA owner |
|---|---|---|
| Project overview and first adoption path | `README.md` | Days 2-5, 10-13 |
| Install and support wording | `INSTALL.md` | Days 2-3, 10-13 |
| Tutorial workflow | `docs/tutorial.md` | Days 2-3, 12-13 |
| Solver recommendations | `docs/solver_selection.md` | Days 2-3, 10-13 |
| Matrix Market guidance | `docs/matrix_market.md` | Days 2-3, 12-13 |
| Algorithm reference | `docs/algorithm.md` | Days 8-9 |
| Benchmark usage and interpretation | `benchmarks/README.md` | Days 2-3, 6-7, 10-13 |
| Example entry points | `examples/README.md` | Days 2-3, 12-13 |

## Duplicate-Work Exclusion List

| Excluded work | Reason |
|---|---|
| New sparse solver implementation or behavior changes | Sprint 116 is documentation QA and claim-guardrail work only. |
| Package-manager recipes or support enablement | Sprint 115 deferred package-manager support until recipes and reviewed install/consumer proof exist. |
| Shared-library or dynamic ABI support | Sprint 115 preserved static-first support and deferred ABI productization. |
| Linux, macOS, or Windows install-lane promotion | Package/platform proof lanes are outside Sprint 116 unless already reviewed and documented. |
| Windows thread/fuzz/property parity | Sprint 115 kept these staged exclusions outside adoption claims. |
| Public API, install-header, or internal helper expansion | Adoption docs must not turn internal surfaces into public contracts. |
| Source movement or helper-abstraction cleanup | Sprint 114 non-package residuals stay out of this adoption QA sprint. |
| Rebuilding Sprint 111 docs from scratch | Sprint 111 completed the docs; Sprint 116 audits and adjusts only where QA requires it. |

## Sprint 115 Claim Guardrails

| Guardrail | Adoption wording implication |
|---|---|
| No reviewed Linux install CI lane beyond documented proof | Avoid language that implies full reviewed Linux install validation unless evidence exists. |
| No full reviewed macOS CMake install/export parity | Keep macOS install/export wording bounded and non-universal. |
| No Windows install-validation parity | Do not advertise Windows installed-package support as reviewed. |
| No Windows thread/fuzz/property parity | Do not imply all test lanes are Windows reviewed. |
| No shared-library package support | Keep package wording static-first unless future evidence changes the contract. |
| No dynamic ABI guarantee | Avoid ABI-stability or shared-object compatibility claims. |
| No package-manager support | Do not imply Homebrew, vcpkg, distro, or other package-manager availability. |
| No public API/install-header expansion | Keep adoption examples tied to reviewed headers and documented entry points. |

## Day-Level Owner Map

| Day | Owner focus | Project-plan item |
|---:|---|---|
| 1 | Intake, duplicate fence, claim guardrails, artifact map | Item 1 |
| 2 | External reference inventory | Item 1 |
| 3 | External reference network QA and focused fixes | Item 1 |
| 4 | README quality and CI-boundary review | Item 2 |
| 5 | README quality and CI-boundary follow-through | Item 2 |
| 6 | Benchmark scanability inventory | Item 3 |
| 7 | Benchmark scanability decision and cleanup | Item 3 |
| 8 | Algorithm reference positioning review | Item 4 |
| 9 | Algorithm reference positioning decision | Item 4 |
| 10 | Performance wording inventory | Item 5 |
| 11 | Evidence-bounded performance wording follow-through | Item 5 |
| 12 | Adoption non-claims checklist draft | Item 6 |
| 13 | Adoption non-claims follow-through | Item 6 |
| 14 | Documentation hygiene, validation, and handoff | Item 7 |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| All Sprint 116 project-plan items have day-level owners | Complete. |
| Sprint 115 package/platform decisions are available as claim guardrails | Complete. |
| No implementation or package/platform support work is pulled into Sprint 116 | Complete. |

## Validation Notes

- Day 1 changed Sprint 116 planning documentation only.
- No `.c` or `.h` files were modified.
- No code, workflow, Make/CMake, script, install, package, or ABI behavior was
  changed.
