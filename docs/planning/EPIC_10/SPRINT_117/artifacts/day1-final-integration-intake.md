# Sprint 117 Day 1 Final Integration Intake

## Purpose

Day 1 establishes the closeout frame for Sprint 117. It maps the Epic 10
evidence spine, identifies validation lanes by touched surface, assigns
day-level ownership for each Sprint 117 project-plan item, and preserves the
claim guardrails created by Sprint 100 and the residual decisions from Sprints
114 through 116.

## Inputs Reviewed

| Input | Closeout role |
|---|---|
| `docs/planning/EPIC_10/PROJECT_PLAN.md` Sprint 117 section | Source of the final integration, validation, comparison, cleanup, residual, and retrospective obligations. |
| `docs/planning/EPIC_10/SPRINT_117/PLAN.md` | Day-by-day execution plan for the closeout sprint. |
| `docs/planning/EPIC_10/SPRINT_100/artifacts/day6-state-of-the-art-target.md` | Final comparison target and disallowed broad-claim list. |
| `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-claim-non-goal-register.md` | Claim states, promotion rule, blocked/stretch list, and non-goals. |
| Sprint 100-116 retrospectives | Cross-sprint completion record and residual evidence spine. |
| Sprint 114 residual debt | Proof-owner, source-boundary, eigensolver, direct/iterative oracle, and SVD helper guardrails. |
| Sprint 115 residual debt | Package/platform, install, Windows, macOS, shared-library ABI, and package-manager guardrails. |
| Sprint 116 residual debt | Adoption-surface claim guardrails and future documentation/benchmark discoverability candidates. |

## Sprint 100 Claim Contract Intake

Sprint 100 defines Epic 10 as a productization and evidence epic, not as a
broad replacement claim for mature sparse linear algebra ecosystems. The final
Sprint 117 audit should preserve this target:

- product-grade, self-contained C sparse linear algebra maturity;
- compressed-first CSR/CSC workflows as the preferred product model;
- deeper external oracle evidence for priority solver families;
- clear package/platform support tiers;
- bounded backend/runtime and benchmark evidence;
- explicit maintainability progress where source/test ownership changed;
- calibrated public claims with remaining non-goals written down.

The following broad claims remain disallowed unless Sprint 117 finds direct
new evidence and validates the exact public wording:

- unqualified state-of-the-art sparse linear algebra replacement;
- SuiteSparse, PETSc, Trilinos, or vendor-backend parity;
- portable performance superiority;
- shared-library ABI stability;
- symmetric Linux/macOS/Windows reviewed parity;
- every-solver-family external validation;
- broad complex, mixed-precision, GPU, or distributed-memory maturity.

## Evidence Map

| Sprint 117 concern | Evidence sources | Day owner | Closeout decision needed |
|---|---|---:|---|
| End-state claims | Sprint 100 target/register, public docs, final sprint retrospectives, implementation and validation artifacts | Days 2-3 | Classify each claim as earned, partially earned, unsupported, deferred, or non-claim. |
| Full validation | Makefile, CMake, CTest, install tests, benchmark/report targets, documentation checks, workflows | Days 4-6 | Select and run the strongest reviewed and supplemental checks appropriate to touched surfaces. |
| Solver and comparison package | Sprint 102-105, Sprint 109, Sprint 113-114 artifacts, external oracle records, benchmark reports | Days 7-8 | Publish final comparison evidence without implying universal solver parity. |
| Unsupported claim cleanup | `README.md`, `INSTALL.md`, `docs/`, `examples/`, `benchmarks/README.md`, Sprint 116 non-claims | Day 8 | Remove, downgrade, or fence public/support wording that lacks evidence. |
| Residual queue | Sprint 100 register, Sprint 114-116 residual debt, Sprint 117 audit findings | Days 9-10 | Publish explicit post-Epic residuals and non-claims without duplicating completed work. |
| Sprint retrospective | Sprint 117 plan, working notes, artifacts, validation package | Days 11-12 | Close Sprint 117 with completed work, metrics, residuals, and validation results. |
| Epic retrospective | Epic 10 project plan, Sprint 100-117 artifacts and retrospectives | Days 13-14 | Close Epic 10 with earned claims, residuals, and post-epic handoff. |

## Validation-Lane Inventory

| Surface changed or audited | Candidate validation lane | Notes |
|---|---|---|
| Documentation-only edits | `git diff --check`; focused trailing-whitespace scan over touched docs | Sufficient for Day 1 because only planning docs are changed. |
| Public claim wording | Evidence-source cross-check against Sprint 100 target and Sprint 114-116 guardrails | Required before keeping, adding, or strengthening claims. |
| C source or public headers | `make format && make lint && make test` | Mandatory for any `.c` or `.h` modifications. |
| Make/CMake parity | `make quality-review-cmake` or focused CMake configure/build/CTest lane | Required if CMake, test registration, or reviewed CTest membership changes. |
| Install/export/package surface | `tests/test_install.sh`; `tests/test_cmake_install.sh`; relevant install target | Required if install guidance, package config, exported target, or package proof changes. |
| Source-list or build metadata | source-list parity check and focused compile/consumer proof | Required before moving proof owners or compile units. |
| Benchmark/report evidence | affected benchmark/report target and interpretation artifact | Keep local timing caveats and avoid portable superiority claims. |
| Platform support wording | reviewed-count and staged-exclusion check for Linux/macOS/Windows lanes | Required before modifying platform support tiers. |

## Day-Level Owner Map

| Project-plan item | Day ownership | Expected output |
|---|---|---|
| Item 1: End-State Claim Audit | Days 1-3 | Intake map, claim inventory, earned/downgrade/non-claim decision. |
| Item 2: Full Validation Pass | Days 4-6 | Validation design, execution log, final validation package. |
| Item 3: Final Comparison Package | Days 7-8 | Solver/reorder/benchmark/package comparison summary and evidence package. |
| Item 4: Unsupported Claim Cleanup | Day 8 | Public/support wording cleanup and cleanup evidence. |
| Item 5: Residual Queue and Non-Claims | Days 9-10 | Post-Epic residual queue and final non-claim register. |
| Item 6: Sprint 117 Retrospective | Days 11-12 | Sprint 117 retrospective and sprint-level validation summary. |
| Item 7: Epic 10 Retrospective | Days 13-14 | Epic 10 retrospective, earned-claim summary, and post-epic handoff. |

## Closeout Guardrails

Sprint 117 should not silently pull the following into scope:

- eigensolver source movement without exact old/new files, source-list and
  CMake updates, focused consumer proof, reviewed CTest evidence, and rollback
  instructions;
- direct/iterative oracle or broad SVD helper abstraction without proof that
  ownership can be shared without hiding solver-specific evidence;
- Linux install CI, macOS install/export parity, Windows install-validation
  parity, Windows thread/fuzz/property parity, or Windows Makefile parity;
- shared-library package support, dynamic ABI compatibility, package-manager
  support, or public install-header/API expansion;
- public Matrix I/O module or builder API claims;
- broad benchmark, performance, state-of-the-art, or platform parity claims.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| All Sprint 117 project-plan items have day-level owners. | Complete. |
| Sprint 100 target and Sprint 114-116 residual decisions are available as closeout guardrails. | Complete. |
| Unsupported implementation, package/platform, or claim work is not silently pulled into Sprint 117. | Complete. |
| Day 1 remains documentation-only. | Complete. |
