# Sprint 143 Day 5 Package/ABI Product Decision

## Decision Record

| Field | Value |
| --- | --- |
| Decision ID | `epic12_sprint143_static_first_follow_through_v1` |
| Decision owner | Package/ABI owner |
| Decision date | 2026-08-08 |
| Selected path | Stricter static-first-only package contract |
| Non-selected path | Shared-library ABI support |
| Implementation window | Sprint 143 Days 6-14 |
| Promotion dependency | Sprint 144 owns macOS/Windows platform-lane promotion decisions. |

Sprint 143 will implement the stricter static-first-only path. The maintained
package surface is a static archive with installed public headers, CMake
install/export metadata, `pkg-config` metadata, downstream consumer proof,
exact-version checks, unsupported shared-artifact checks, and explicit
non-claims for shared libraries, dynamic ABI compatibility, runtime-loader
behavior, package-manager distribution, and broad platform parity.

Sprint 143 will not implement shared-library ABI support.

## Decision Inputs

| Input | Finding | Decision effect |
| --- | --- | --- |
| Sprint 137 gap selection | Sprint 137 selected static-first package/ABI follow-through and rejected shared-library packaging for Epic 12 implementation. | Favors static-first follow-through unless Sprint 143 audits find a low-risk shared path. |
| Sprint 142 handoff | Runtime/backend controls and sentinel rows must not become package, ABI, platform, or portable performance claims. | Requires conservative package wording and validation ownership. |
| Day 2 header/symbol audit | 19 installed headers, public structs/enums/callbacks, `SPARSE_IDX_BITS`, and 137 non-`sparse_` or internal-looking global symbols create a broad ABI-risk surface. | Shared ABI support would require export policy, visibility, allowlists, and ABI layout policy. |
| Day 3 metadata audit | Make/CMake/`pkg-config` install proof already supports static archive consumers; CMake rejects `BUILD_SHARED_LIBS=ON`; package rows are static-first proof-owner rows. | Static-first can be completed by strengthening existing proof. |
| Day 4 platform audit | Linux, macOS, and Windows all lack dynamic loader proof; macOS/Windows package lanes remain supplemental; Windows lacks DLL/export/import policy. | Shared support would exceed the remaining sprint budget and risk unsupported platform claims. |

## Decision Scorecard

Scores use `1` for weak/expensive/high-risk fit and `5` for strong/low-risk
fit inside the remaining Sprint 143 budget.

| Criterion | Shared-library ABI support | Static-first-only strengthening |
| --- | ---: | ---: |
| User value in this sprint | 3 | 4 |
| Proof feasibility | 1 | 5 |
| Platform risk | 1 | 4 |
| Implementation risk | 1 | 5 |
| Documentation burden | 2 | 4 |
| Claim risk | 1 | 5 |
| Fit with Sprint 137 selection | 1 | 5 |
| Fit with Sprint 142 handoff | 2 | 5 |
| Total | 12 | 37 |

The decisive factor is not that shared libraries lack value. The issue is that
the repository does not yet have the symbol visibility, ABI versioning, loader
proof, platform support-tier evidence, or package selector semantics required
to claim shared-library ABI support honestly inside the remaining sprint.

## Selected Implementation Path

Sprint 143 Days 6-14 should implement a static-first package contract that is
stricter, more observable, and harder to accidentally widen.

Required changes:

1. Strengthen unsupported shared-library guards.
2. Strengthen no-shared-artifact checks across package proof paths.
3. Clarify CMake install/export metadata so the installed target is explicitly
   static-only to downstream consumers.
4. Clarify `pkg-config` metadata comments and proof so it stays a static
   archive consumer contract.
5. Improve package test diagnostics for static archive, headers, version,
   unsupported artifact, and downstream consumer failures.
6. Align Linux, macOS, and Windows CI comments with the selected contract:
   Linux reviewed package baseline, macOS/Windows supplemental confidence
   unless Sprint 144 promotes them.
7. Update README, INSTALL, and maintainer guide wording so maintained
   static-first support and deferred shared ABI support are unambiguous.
8. Keep package report rows as proof-owner metadata unless a later day changes
   row semantics explicitly.

## Required Validation Checklist

| Surface | Required validation |
| --- | --- |
| Make install and `pkg-config` | `bash tests/test_install.sh` |
| CMake install/export | `bash tests/test_cmake_install.sh` |
| Unsupported shared-library guard | `bash scripts/static_package_deferral_check.sh` |
| Package report rows | `python3 scripts/normalize_report_index.py --family package --check` and `python3 scripts/normalize_report_index.py --family package --check-freshness` |
| Shell scripts changed | `bash -n` for changed shell scripts plus focused execution where feasible |
| CMake/Make/package metadata changed | Focused install/export proof and generated metadata inspection |
| CI workflows changed | Focused command/path/support-tier review |
| Public docs changed | Claim-boundary review for README, INSTALL, maintainer guide, and package docs |
| C or public headers changed | `make format && make lint && make test` after focused behavior checks |
| Planning artifacts only | `git diff --check` and trailing-whitespace scan |

## Non-Selected Path Deferral Ledger

| Deferred item | Reason deferred | Future promotion gate |
| --- | --- | --- |
| Shared-library build/install/export | Current CMake explicitly rejects `BUILD_SHARED_LIBS=ON`; adding shared support would require coordinated product design. | CMake shared target, install/export metadata, downstream shared consumers, and unsupported-platform boundaries. |
| Dynamic ABI compatibility | Public structs, enums, callbacks, `idx_t` width, and exported symbols lack a compatibility policy. | ABI epoch/version policy, layout policy, symbol allowlist, compatibility tests, and docs. |
| Export/import macro policy | No `SPARSE_API` or Windows `__declspec` policy exists. | Public macro design, hidden implementation visibility, MSVC import/export proof, and downstream consumer tests. |
| Loader compatibility | No Linux RPATH/RUNPATH, macOS install-name, or Windows DLL search-path proof exists. | Platform-specific loader tests and support-tier decisions. |
| Static/shared selectors | `pkg-config` and CMake package metadata currently describe one static imported target. | Deliberate target naming, selector semantics, coexistence tests, and docs. |
| Package-manager distribution | Local install/export proof does not define package-manager recipes or release mechanics. | Manager-specific recipes, install roots, upgrade/uninstall tests, and support docs. |
| macOS reviewed install/export parity | Current macOS package lanes are supplemental confidence paths. | Sprint 144 platform promotion decision with hosted evidence and failure ownership. |
| Windows reviewed install-validation parity | Current Windows install/downstream lane is supplemental CMake-first confidence only. | Sprint 144 promotion decision covering hosted evidence, exact static-first scope, and staged blockers. |

## Non-Claims To Preserve

- No shared-library packaging support.
- No dynamic ABI compatibility promise.
- No runtime-loader compatibility promise.
- No package-manager availability.
- No macOS or Windows reviewed package parity from Sprint 143 alone.
- No portable performance claim from package or sentinel rows.
- No state-of-the-art claim from package/ABI work.
- No runtime/backend control is promoted to public ABI by package wording.

## Implementation Stop Conditions

Stop and ask before proceeding if any remaining Sprint 143 change would:

1. Install or document `.so`, `.dylib`, or `.dll` artifacts.
2. Add `SPARSE_API`, export/import macros, `SOVERSION`, install-name,
   visibility, or dynamic ABI metadata.
3. Add CMake or `pkg-config` selectors that imply shared/static variant
   support.
4. Promote macOS or Windows package lanes from supplemental to reviewed parity.
5. Use package report freshness, runtime sentinel rows, or local benchmark rows
   as dynamic loader, platform, package-manager, or portable performance proof.
6. Change public C headers or implementation files without running the full C
   quality gate.

## Day 6 Implementation Input

Day 6 should design the static-first guard changes before editing build or
script surfaces. The highest-value first implementation target is the
unsupported-shared proof boundary: make the existing deferral guard, install
tests, and metadata comments describe the selected static-first decision as an
intentional maintained product contract rather than an absence of shared
support.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The sprint has one selected package path, not two partial paths. | Complete | Decision record selects stricter static-first-only support and rejects shared-library ABI implementation. |
| The selected path can be implemented and validated inside the remaining sprint budget. | Complete | Selected implementation list reuses existing install/export/guard proof and routes platform promotion to Sprint 144. |
| Deferred package/ABI claims are explicit and source-owned. | Complete | Deferral ledger names shared libraries, dynamic ABI, loader behavior, selectors, package managers, and platform parity with future gates. |
