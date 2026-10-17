# Sprint 214 Day 1: ABI And Release Evidence Intake

## Purpose

Establish the current static-first package baseline and identify the evidence
required for the Day 2 shared-proof versus design/deferral decision. This is an
intake record, not an ABI or release-readiness decision.

## Scope Trace

| Item | Day 1 input | Later acceptance owner |
| --- | --- | --- |
| 214.1 Scope decision | Sprint 170 static-first decision; Sprint 207 provider deferral; current CI and package contract. | Day 2 decision record. |
| 214.2 ABI policy | Sprint 170 header/symbol inventory and current public headers/build metadata. | Days 3-5 symbol, compatibility, loader, and platform policy. |
| 214.3 Build/install proof or guard | Make archive install; CMake explicit static target and shared configure rejection; static deferral guard. | Days 5-8 selected implementation plus regressions. |
| 214.4 Consumer evidence | Installed Make/`pkg-config`, CMake `find_package`, and Windows CMake static consumers. | Days 8-10 named-platform shared consumer or exact future tests. |
| 214.5 Release checklist | No source archive reproducibility, provenance, tag, artifact, and CI set is yet certified for release readiness. | Days 11-12 checklist with evidence owners and stop gates. |
| 214.6 Validation/closeout | Existing install scripts, deferral guards, docs checks, and reviewed quality targets. | Days 13-14 command-backed validation and residual queue. |

## Baseline Evidence

| Surface | Day 1 observation |
| --- | --- |
| Version | `VERSION` currently contains `2.2.0`; it is not an ABI epoch. |
| Product decision | Sprint 170 Day 9 explicitly chose static-first-only package and ABI posture. Shared libraries and dynamic ABI compatibility remain unsupported and deferred. |
| Package decision | Sprint 207 Day 5 selected continued public-provider deferral; the local Homebrew proof is developer-mode static source formula evidence only. |
| Make | `Makefile` builds `build/libsparse_lu_ortho.a`; install copies archive, checked-in public headers, generated version header, and `sparse.pc`. |
| CMake | `CMakeLists.txt` rejects `BUILD_SHARED_LIBS=ON`, declares `sparse_lu_ortho STATIC`, installs archive-only target metadata, and exports `Sparse::sparse_lu_ortho`. Its `ExactVersion` package check is not a dynamic ABI guarantee. |
| Installed consumers | `tests/test_install.sh` and `tests/test_cmake_install.sh` validate static install/consumption and reject shared artifacts or shared imported metadata. |
| Guard | `scripts/static_package_deferral_check.sh` requires shared configure rejection, static metadata, and public non-claims. It passed locally on Day 1. |
| CI | Linux and macOS workflows own reviewed static package-contract lanes. Windows workflow owns CMake static install/downstream validation but treats `sparse.pc` as metadata-only; no Windows Make or `pkg-config` execution parity. Workflow ownership is not a new hosted run result. |
| Documentation | `INSTALL.md#support-readiness-matrix` is the public support truth; README and maintainer guide route users to static-first install evidence and defer shared/ABI claims. |
| Residual | Epic 18 `E18-RQ-009`, Sprint 207 retrospective, and Sprint 213 retrospective preserve release/shared/ABI residuals. |

Sprint 170's header inventory is a useful starting point, not a present-day
export allowlist. Many installed headers expose concrete structs, callbacks,
and library-owned memory; `SPARSE_IDX_BITS` changes public types. The archive
has not been turned into a reviewed, hidden-by-default dynamic export surface.

## Current Claims And Non-Claims

| Current evidence permits | Evidence does not permit |
| --- | --- |
| Static archive build/install and installed Unix Make/`pkg-config` or CMake consumers under their reviewed lanes. | `.so`, `.dylib`, `.dll`, import-library, runtime-loader, SONAME, install-name/RPATH, or dynamic ABI support. |
| Windows CMake static install/downstream evidence and `sparse.pc` metadata inspection. | Windows Make install parity, Windows `pkg-config` execution, or broad cross-platform parity. |
| Exact CMake package-version matching. | Cross-version binary compatibility or a defined ABI epoch. |
| Developer-mode local Homebrew static source formula proof. | User-facing package-manager distribution, release packages, reproducible archives, or release readiness. |

Generated API HTML remains local-only under Sprint 213's policy and is not a
retained or published release artifact.

## Day 2 Decision Inputs

1. Can a single named platform/toolchain support a narrow shared target with
   explicit exported symbols, loader metadata, installed consumer execution,
   and CI proof without implying other-platform or ABI compatibility support?
2. If not, what exact design/deferral guards and future acceptance tests close
   the ambiguity while preserving the static-first contract?
3. Which exposed layouts, callbacks, allocation boundaries, index-width
   settings, and version semantics would belong to any selected dynamic ABI?
4. Which product owner approves the narrow claim or confirms continued
   deferral, and what evidence would stop implementation?

Day 1 selects neither path. The working-notes decision log preserves both
options until Day 2.

## Validation And Handoff

| Check | Day 1 result |
| --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed locally; all reported contract checks passed. |
| `bash scripts/package_manager_deferral_check.sh` | Interrupted during the local Homebrew proof subcheck when Homebrew started building CMake from source. Preliminary static/provider checks printed `ok`, but the overall guard did not finish and is not counted as passed. |
| Make/CMake install scripts, docs checks, full C quality gates | Not run for documentation-only intake; identified as later selected-path gates. |
| Whitespace checks (`git diff --check` plus new-file checks) | Passed for all three Sprint 214 Markdown files. |

Day 2 should use the evidence ledger and risk register in `WORKING_NOTES.md`
to choose one bounded path or escalate a missing product decision. It must not
equate a linkable shared file, package-version exactness, or a release checklist
with an earned dynamic ABI or release-readiness claim.
