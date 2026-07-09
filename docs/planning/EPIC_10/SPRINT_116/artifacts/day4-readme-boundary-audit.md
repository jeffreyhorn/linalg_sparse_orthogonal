# Day 4 README Boundary Audit

## Purpose

Day 4 audits `README.md` for adoption quality, CI/support wording, install
claim boundaries, benchmark evidence boundaries, and places where the front
door risks becoming maintainer policy. Day 5 will apply only the compact
wording fixes identified here.

## Inputs Reviewed

| Input | Reviewed for |
|---|---|
| `README.md` | Adoption flow, CI wording, support-tier claims, install summary, benchmark/performance language, API overview boundaries. |
| `INSTALL.md` | Maintained install contract, static-first package surface, reviewed platform boundaries, package/ABI non-claims. |
| Sprint 115 retrospective residual debt | Package/platform guardrails: no reviewed Linux install CI lane, no full macOS install/export parity, no Windows install-validation parity, no shared-library/dynamic ABI guarantee, no package-manager support. |

## README Boundary Findings

| Area | README status | Day 5 decision |
|---|---|---|
| First-use adoption path | Good. The README starts with "Start Here", routes users to solver selection, install detail, examples, benchmarks, and maintainer guide by audience. | No edit required. |
| Maintainer-policy boundary | Good. The README explicitly keeps maintainer/quality policy in `docs/maintainer_guide.md` and historical evidence in `docs/planning/`. | No edit required. |
| CI/support-tier wording | Good. Linux is strongest reviewed truth; macOS is reviewed Apple Clang with supplemental Homebrew GCC/static install evidence; Windows is reviewed CMake subset and CMake-first consumer story. | No edit required. |
| Install summary | Mostly good. It points to `INSTALL.md`, names `pkg-config` and `find_package(Sparse)`, and states the maintained package surface is static. | One wording fix candidate: avoid "package-manager detail" in the build section. |
| Shared library and ABI claims | Good. README explicitly says shared-library packaging is deferred and the install contract is static archive surface. | No edit required. |
| Package-manager claims | Mostly good. README does not claim Homebrew/vcpkg/distro support, but the phrase "package-manager detail" can be misread. | Edit Day 5 to say "package/install-support detail" or similar. |
| Benchmark/performance wording | Good. README keeps benchmark details in `benchmarks/README.md` and says benchmark rows are branch-local measurement artifacts, not portable performance guarantees. | No edit required. |
| API overview size | Acceptable for now. The API section is long, but it is structured as a reference table and function group map rather than maintainer policy. | No Day 5 edit; future docs work can split if scanability suffers. |

## Unsupported-Claim Candidate Table

| Candidate | Location | Risk | Disposition |
|---|---|---|---|
| "package-manager detail" | `README.md` build section | Could imply package-manager support exists despite Sprint 115 deferral. | Day 5 edit candidate. |
| "supports find_package(Sparse)" | `README.md` CMake quick build snippet | Accurate for maintained static CMake package surface. | Keep. |
| "`pkg-config` or `find_package(Sparse)` against the maintained static package surface" | `README.md` installation section | Accurate and explicitly static. | Keep. |
| "Shared-library packaging is intentionally deferred" | `README.md` installation section | Correct non-claim. | Keep. |
| "make bench-fast remains the bounded PR-time runtime benchmark signal" | `README.md` quality section | Evidence-bounded and not a portable performance claim. | Keep. |
| "branch-local measurement artifacts, not portable performance guarantees" | `README.md` performance section | Correct guardrail. | Keep. |

## Compactness and Audience Notes

- The README remains adoption-facing because it gives users an immediate build
  path, a quick solve, workflow selection, and links to deeper docs.
- The quality and CI language is compact enough for a front door while still
  preserving support-tier boundaries.
- The benchmark/performance section avoids turning local measurements into
  broad performance claims.
- The long API overview is a scanability risk, but it is not a Day 5 boundary
  fix because it does not create an unsupported claim.

## Day 5 Edit Checklist

| Item | Edit decision | Rationale |
|---|---|---|
| Replace "package-manager detail" in the build section | Edit | Sprint 115 explicitly defers package-manager support; use wording that points to install/support details without implying supported package managers. |
| Leave CI/support-tier paragraph unchanged | No edit | It matches Sprint 115 truth and keeps reviewed/staged boundaries explicit. |
| Leave installation section static-package wording unchanged | No edit | It accurately describes `pkg-config`, `find_package(Sparse)`, and deferred shared-library packaging. |
| Leave benchmark/performance wording unchanged | No edit | It is already evidence-bounded and non-portable. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Day 5 has a concrete edit or no-edit decision | Complete; one wording edit candidate and several no-edit decisions recorded. |
| README package/platform claims match Sprint 115 truth | Complete, except for the single wording cleanup candidate. |
| README remains scoped to adoption, not maintainer policy | Complete. |

## Validation Notes

- Day 4 changed Sprint 116 planning documentation only.
- `README.md` and `INSTALL.md` were inspected but not edited.
- No `.c` or `.h` files were modified.
