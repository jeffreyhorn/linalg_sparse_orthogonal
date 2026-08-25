# Sprint 180 Working Notes

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Source Artifact Note

The Sprint 180 source section lives in
`docs/planning/EPIC_16/PROJECT_PLAN.md` under "Sprint 180:
Package-Manager Provider Decision". Sprint 180 artifacts in this directory
follow the Epic 16 scope.

## Sprint Goal

Prove one package-manager provider path or close the provider question with a
stronger formal deferral and guard update.

## Baseline Inputs

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- `docs/planning/EPIC_16/SPRINT_180/PLAN.md`
- `docs/planning/EPIC_15/SPRINT_171/artifacts/day5-package-manager-deferral.md`
- `scripts/package_manager_deferral_check.sh`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day6-populated-matrix.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day7-target-selection.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day8-gate-templates.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day10-quality-surface-map.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day11-claim-boundary-freeze.md`
- `docs/planning/EPIC_16/reviews/review-codex-2026-08-23.md`
- `docs/planning/EPIC_16/reviews/todo-codex-2026-08-23.md`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `sparse.pc.in`
- `cmake/SparseConfig.cmake.in`

## Starting Branch Snapshot

- Branch: `sprint-180`
- Starting commit: `88884f9960a895d7782d6d48fd369b38e7eeea9d`
- Recent base context:
  - `88884f99` Merge pull request #199 from `sprint-179`
  - `188747b5` Complete Sprint 179 generated API decision
  - `17754f05` Merge pull request #198 from `sprint-178`
  - `a7d58196` Complete Sprint 178 allocation-failure proof
  - `3907e754` Merge pull request #197 from `sprint-177`

## Sprint 180 Project-Plan Items

| Item | Name | Status | Notes |
| --- | --- | --- | --- |
| 180.1 | Provider Feasibility Audit | Complete | Day 1 defines provider evaluation criteria. Days 3-6 evaluate vcpkg, Homebrew, Conan, and pkgsrc. Day 7 compares the candidates. |
| 180.2 | Product Decision Record | Complete | Day 8 selects Homebrew local formula/tap proof as the Sprint 180 provider proof path. |
| 180.3 | Recipe or Deferral Artifact | Complete | Day 9 designs a Homebrew local formula-template prototype and proof-generated temporary formula flow. Day 10 adds the selected template and local-proof notes. |
| 180.4 | Proof Script | Complete | Day 11 designs the Homebrew local formula proof script. Day 12 implements it and records the current claim-safe missing-license stop condition. |
| 180.5 | Guard and Docs Update | Complete | Day 13 updates package-manager guard behavior and docs to record selected Homebrew local proof artifacts plus the current missing-license blocker. |
| 180.6 | Validation | Complete | Day 14 validates the selected Homebrew local proof boundary, guards, package/install checks, report-index ownership, generated-output hygiene, and whitespace state. |

## Current Package-Manager Baseline

| Surface | Current state |
| --- | --- |
| Product status | Package-manager support is formally deferred by Sprint 171. |
| Guard | `scripts/package_manager_deferral_check.sh` checks the Sprint 171 deferral record, provider recipe absence, provider-neutral package metadata, and public non-claim wording. |
| Supported package surface | Static-first source install and installed static consumer proof through Make, CMake, `pkg-config`, and named CI lanes. |
| Unsupported providers | vcpkg, Homebrew, Conan, pkgsrc, Debian, Fedora, system packages, registries, taps, and binary packages remain unsupported unless Sprint 180 changes the product decision. |
| Package metadata | `sparse.pc.in` and `cmake/SparseConfig.cmake.in` must remain provider-neutral unless a selected decision updates the guard and proof path. |
| Public docs | README, INSTALL, and maintainer guide distinguish source install from package-manager distribution and keep package-manager support as a non-claim. |

## Current Package Surface Inventory

Day 2 confirms the maintained package surface is static-first and
source-install oriented:

| Area | Current owner | Current role |
| --- | --- | --- |
| Make install | `Makefile` | Installs `libsparse_lu_ortho.a`, public headers, generated `sparse_version.h`, and generated `sparse.pc`; `uninstall` removes the static archive, installed headers, and `.pc` file. |
| CMake install/export | `CMakeLists.txt` | Installs the explicit static target archive, public headers, generated version header, `SparseTargets.cmake`, exact-version package metadata, and generated `sparse.pc`. |
| pkg-config template | `sparse.pc.in` | Describes static archive package metadata with include and link flags; contains no provider wording. |
| CMake package template | `cmake/SparseConfig.cmake.in` | Includes installed `SparseTargets.cmake` and checks required components; contains no provider wording. |
| Make/pkg-config proof | `tests/test_install.sh` | Proves Make install/uninstall, installed file shape, `pkg-config` resolution, exact version query, two downstream compile/link/run consumers, and cleanup. |
| CMake proof | `tests/test_cmake_install.sh` | Proves CMake configure/build/install, static imported-target metadata, `find_package(Sparse)`, exact-version behavior, mismatched-version rejection, installed consumer run, and static `.pc` metadata. |
| Static-first guard | `scripts/static_package_deferral_check.sh` | Protects the Sprint 170 static-first package contract and shared-library/dynamic ABI deferrals. |
| Package-manager guard | `scripts/package_manager_deferral_check.sh` | Protects Sprint 171 package-manager deferral, provider recipe absence, provider-neutral metadata, and public non-claims. |
| Package proof rows | `scripts/normalize_report_index.py` and `tests/corpus/manifests/report_families.tsv` | Identify source-controlled package proof owners; rows are ownership metadata, not proof that install commands were just run. |

## Package-Manager Claim Surface Inventory

| File | Current package-manager wording |
| --- | --- |
| README | States source install via Make or CMake is the supported path, installed metadata is static-archive scoped, and package-manager support is not currently provided. |
| INSTALL | Has a package-manager deferral support-split entry that lists vcpkg, Homebrew, Conan, pkgsrc, distro/system packages, provider registries, taps, recipes, and binary packages as unsupported. |
| Maintainer guide | Tells maintainers to run `scripts/package_manager_deferral_check.sh` when changing package-manager wording, provider recipe files, metadata templates, or provider support claims. |
| API reference | Points install support to `INSTALL.md` and keeps package-manager distribution outside API-reference claim scope. |
| Workflows | Linux and macOS package jobs prove static-first install/export surfaces only; Windows proves CMake install/downstream validation and metadata-only `sparse.pc` inspection. Workflow comments keep package-manager support as a non-claim. |

## Provider Feasibility Running Summary

| Provider | Status | Day | Summary |
| --- | --- | --- | --- |
| vcpkg | Eligible for local overlay proof; not registry-ready | Day 3 | Strong static-first and CMake install/export fit, plausible Windows user value, and manageable local overlay prototype shape. Blocked from registry-ready support by absent local vcpkg tool, no source archive/checksum policy, no standalone license file or SPDX-style license metadata, and no provider install/downstream/version/cleanup proof yet. |
| Homebrew | Eligible for local formula/tap proof; not core-ready or bottle-backed | Day 4 | Strong macOS user value and good CMake install/export fit. Blocked from Homebrew/core or bottle claims by absent formula/tap artifacts, no standalone license file, no immutable release URL plus SHA-256 policy, no formula `test do` proof, no `brew audit` evidence, and no bottle build/publish evidence. |
| Conan | Eligible for local recipe proof; not ConanCenter-ready or binary-hosted | Day 5 | Good CMake consumer fit and cross-platform user value, but higher recipe/modeling complexity than vcpkg or Homebrew. Blocked by absent local `conan`, no recipe or `test_package`, no profile/generator proof, no package ID/options policy, no standalone license file, and no remote/binary/package-revision evidence. |
| pkgsrc | Eligible only for local package skeleton proof; low-priority first-provider candidate | Day 6 | Broad Unix-oriented reach, but the weakest near-term fit for Sprint 180 because it requires pkgsrc bootstrap/tooling, package skeleton metadata, `distinfo`, `PLIST`, dependency/buildlink policy, package lint/build/install proof, and platform-boundary wording before any support claim. Blocked by absent local pkgsrc tools, no pkgsrc package artifacts, no standalone license file, no release/checksum policy, and no pkgsrc downstream/package cleanup proof. |
| Stronger deferral | Open | Days 7-8 | Remains valid if provider proof cost or claim risk exceeds the sprint. |

## Day 7 Provider Decision Matrix Summary

Day 7 compares the four audited providers against the shared Sprint 180
criteria. The recommendation is not yet the product decision; Day 8 must still
record exactly one selected path or formal renewed deferral.

| Rank | Candidate | Day 7 position | Rationale |
| --- | --- | --- | --- |
| 1 | Homebrew local formula/tap proof | Recommended first proof candidate | Strong static-first fit, local `brew` availability, existing macOS package proof prerequisites, bounded local formula shape, and lower immediate tooling risk than vcpkg, Conan, or pkgsrc. Scope must stay local formula/tap only. |
| 2 | vcpkg local overlay proof | Strong runner-up | Best C/C++ and Windows user value with strong CMake/static fit, but no local `vcpkg` tool is available and registry-readiness blockers are unresolved. |
| 3 | Conan local recipe proof | Defer as first provider | Good CMake fit, but package ID, profiles, generators, `package_info()`, cache isolation, and `test_package` make it higher complexity for the first provider proof. |
| 4 | pkgsrc local package skeleton proof | Reject as first provider | Broad Unix reach, but highest proof burden: bootstrap/tooling, package skeleton, `distinfo`, `PLIST`, dependency/buildlink policy, platform boundaries, and package cleanup evidence. |
| fallback | Stronger formal deferral | Keep as Day 8 fallback | Stronger deferral remains the correct Day 8 decision if Homebrew cannot be scoped to a local proof without over-claiming source/checksum, license, bottle, Homebrew/core, or Linuxbrew support. |

Day 7 recommends Homebrew local formula/tap proof as the strongest Day 8
decision candidate, with vcpkg retained as the principal rejected alternative.
The recommendation depends on keeping the claim boundary narrow: local source
formula/tap proof only, no Homebrew/core, no bottles, no Linuxbrew, no package
manager generalization, and no shared-library or dynamic ABI implication.

## Day 8 Product Decision Summary

Sprint 180 selects **Homebrew local formula/tap proof** as the package-manager
provider path for Days 9-14. This is an implementation decision, not a public
support claim.

| Field | Decision |
| --- | --- |
| Selected provider path | Homebrew local formula/tap proof. |
| Selected support level | Local source formula/tap proof only. |
| Explicitly unsupported | Homebrew/core, bottles, Linuxbrew, registry readiness, provider-hosted binaries, broad package-manager support, shared-library support, dynamic ABI support, static/shared selectors, and provider-managed upgrades. |
| Primary reason | It has the strongest immediate proofability: local `brew` availability, existing macOS static package proof prerequisites, and a bounded formula/test-script shape. |
| Main implementation condition | Days 9-13 must add a narrowly scoped prototype, proof behavior, docs wording, and guard update before any selected-provider wording is allowed outside planning artifacts. |
| Stop condition | If source/checksum, license metadata, formula location, proof-script, docs wording, or guard behavior cannot be bounded locally, the sprint must renew formal deferral instead of widening claims. |

The Sprint 171 package-manager deferral remains the active public posture until
the Homebrew local proof path, guard, and docs are implemented and validated.
Provider recipe/formula files outside planning remain prohibited until the
Days 9-13 work explicitly narrows the guard.

## Day 9 Artifact Design Summary

Day 9 designs the selected Homebrew provider material as a source-controlled
formula template plus a proof-generated temporary local formula. This avoids
adding a live `Formula/` tap path before proof-script and guard ownership are
implemented.

| Design field | Day 9 decision |
| --- | --- |
| Source-controlled artifact | `packaging/homebrew/sparse-lu-ortho.rb.in` formula template. |
| Future proof script | `scripts/homebrew_local_formula_proof.sh`. |
| Generated source input | Proof script creates a temporary source archive from the current checkout and injects its `file://` URL plus SHA-256 into a temporary formula. |
| Temporary formula path | Proof script creates a temp local tap/formula under its temp directory; no generated formula, tap, archive, cache, build tree, or installed prefix is committed. |
| Formula install path | Formula uses CMake source build and installs only the static archive package surface. |
| Test behavior | Formula `test do` builds and runs an installed CMake consumer and checks version/static-only installed metadata. |
| Guard treatment | Current guard should keep rejecting live `*/Formula/*` artifacts. Later narrowing may allow the exact template and proof script while continuing to reject all other provider files. |
| Public wording | No README, INSTALL, maintainer guide, package metadata, or workflow support wording changes on Day 9. |

## Day 10 Artifact Implementation Summary

Day 10 implements the selected Homebrew local-proof artifact as a template,
not as a live tap or rendered formula.

| Field | Day 10 implementation |
| --- | --- |
| Template | `packaging/homebrew/sparse-lu-ortho.rb.in` |
| Provider notes | `packaging/homebrew/README.md` |
| Template inputs | Placeholder URL, SHA-256, version, homepage, and license values rendered later by the proof script. |
| Install behavior | CMake source build with `SPARSE_OPENMP=OFF`, `SPARSE_MUTEX=OFF`, `CMAKE_INSTALL_LIBDIR=lib`, and static archive verification. |
| Test behavior | Downstream CMake consumer using `find_package(Sparse)` and `Sparse::sparse_lu_ortho`, with installed static metadata checks. |
| Generated-output posture | No committed rendered formula, tap, source archive, bottle, cache, build tree, install prefix, or logs. |
| Public wording | README, INSTALL, maintainer guide, package metadata, workflows, and guards remain unchanged on Day 10. |

## Day 11 Proof Script Design Summary

Day 11 designs `scripts/homebrew_local_formula_proof.sh` as a local-only proof
runner for the Day 10 template.

| Field | Day 11 design |
| --- | --- |
| Proof command | `bash scripts/homebrew_local_formula_proof.sh` |
| Provider tools | Requires local `brew`, `cmake`, Ruby, C compiler, `tar`, and SHA-256 tool. Missing `brew` exits claim-safely before any support wording can be inferred. |
| Render flow | Create temporary archive from checkout, compute SHA-256, render `packaging/homebrew/sparse-lu-ortho.rb.in` into a temp formula file, then run local `brew` checks. |
| Install proof | Install the rendered formula from source, using generated local archive input only. |
| Downstream proof | Run `brew test` so the formula `test do` builds and runs an installed CMake consumer. |
| Version proof | Inject `VERSION` into the formula and require exact `find_package(Sparse VERSION EXACT REQUIRED)` behavior in the formula test. |
| Cleanup proof | Uninstall the local formula and remove temp archive, formula, tap state, logs, and build output unless an explicit debug-preserve flag is used. |
| Failure posture | Every failure keeps Homebrew as a local proof target only; failures do not imply Homebrew/core, bottle, Linuxbrew, or public package-manager support. |

## Day 12 Proof Script Implementation Summary

Day 12 adds `scripts/homebrew_local_formula_proof.sh`.

| Field | Day 12 result |
| --- | --- |
| Proof script | Added and executable. |
| Implemented preflight | Tool checks, template placeholder checks, Ruby syntax check, temp-root setup, local source archive creation, SHA-256 computation, and cleanup trap. |
| Implemented install path | Formula rendering, `brew install --build-from-source`, static installed-file checks, `brew test`, uninstall, and cleanup are implemented for the path where accurate license metadata is available. |
| Current local outcome | Claim-safe unavailable exit before formula rendering/install because no standalone `LICENSE`, `COPYING`, or `NOTICE` file exists. |
| Exit status | `2` for the current missing-license stop condition. |
| Public posture | Homebrew proof remains unclaimed; Sprint 171 package-manager deferral remains the public posture until Day 13 reconciles guard/docs wording. |

## Day 13 Guard And Docs Summary

Day 13 reconciles public docs and guard behavior with the actual provider
state: selected Homebrew local proof artifacts exist, but provider support is
not claimed because the proof exits before install on missing standalone
license metadata.

| Surface | Day 13 update |
| --- | --- |
| `scripts/package_manager_deferral_check.sh` | Checks selected Homebrew proof template, notes, executable proof script, generated-output hygiene, proof-script exit behavior, provider-neutral metadata, and public non-claim wording. |
| `README.md` | States package-manager support remains unavailable and local Homebrew formula proof remains unclaimed due to missing standalone license metadata. |
| `INSTALL.md` | Records selected local Homebrew proof artifacts, current missing-license blocker, and unsupported Homebrew/core, bottle, Linuxbrew, unselected provider, tap, recipe, and binary-package claims. |
| `docs/maintainer_guide.md` | Documents the selected proof script and warns maintainers not to cite it as Homebrew support while blocked. |
| `sparse.pc.in` and `cmake/SparseConfig.cmake.in` | Unchanged and provider-neutral. |

## Day 14 Integrated Validation Summary

Day 14 validates the full Sprint 180 result. The selected provider path,
artifact shape, proof-script behavior, guard checks, public docs, package
metadata, install/export checks, generated-output hygiene, and whitespace state
are consistent.

| Surface | Day 14 result |
| --- | --- |
| Provider proof | `bash scripts/homebrew_local_formula_proof.sh` exits `2` claim-safely after creating a temporary source archive and before formula rendering because no standalone `LICENSE`, `COPYING`, or `NOTICE` file exists. |
| Provider guard | `bash scripts/package_manager_deferral_check.sh` passes with the selected Homebrew local proof boundary and public non-claims intact. |
| Static guard | `bash scripts/static_package_deferral_check.sh` passes, preserving the static-first package contract and shared-library/dynamic ABI deferrals. |
| Install proof | `bash tests/test_install.sh` passes with `23` Make install, pkg-config, downstream compile/run, version, and uninstall checks. |
| CMake proof | `bash tests/test_cmake_install.sh` passes with `27` CMake install/export, downstream `find_package`, exact-version, mismatch-version, pkg-config, and static metadata checks. |
| Report index | `python3 scripts/normalize_report_index.py --family package --check` passes with `7` package rows. |
| Syntax and hygiene | Shell syntax checks, formula-template Ruby syntax check, generated Homebrew output absence check, and `git diff --check` pass. |

Sprint 180 closes with one selected provider implementation path but no public
provider support claim. The Sprint 181 revisit gate is standalone license
metadata followed by a successful local Homebrew render/install/`brew test`/
uninstall/cleanup proof.

## Sprint 177 Acceptance Gate

Sprint 180 implements Sprint 177 Day 8 Gate 3:

| Field | Day 1 baseline |
| --- | --- |
| Target | Package-manager provider proof or deferral. |
| Owner files | `scripts/package_manager_deferral_check.sh`, package metadata templates, README, INSTALL, maintainer guide, Sprint 180 artifacts, and optional provider prototype files if selected. |
| Required evidence | One static-first provider path is proven with a local proof script, or a stronger formal deferral records exact blockers and fail-closed guards. |
| Required validation | Provider proof or deferral script, `bash scripts/package_manager_deferral_check.sh`, install checks if package metadata changes, `git diff --check`, and full C quality gates if C/header files change. |
| Pass definition | Exactly one provider decision is recorded; package-manager wording and metadata match it; unsupported providers remain absent or guarded; static-first and non-ABI boundaries remain intact. |
| Fail definition | Provider wording appears without proof, recipe files appear without a guard decision, static-first metadata weakens, or broad package-manager support is implied. |

## Provider Evaluation Criteria

| Criterion | Evaluation question |
| --- | --- |
| Static-first fit | Can the provider consume the maintained static archive install/export surface without implying shared-library, dynamic ABI, runtime-loader, or static/shared selector support? |
| CI feasibility | Can local or hosted checks run with bounded time, deterministic setup, and clear unavailable-tool failure behavior? |
| Recipe complexity | How many provider-specific files, patches, policies, metadata fields, and update steps are needed to keep the path maintainable? |
| User value | Does the provider materially reduce adoption friction for likely users without over-scoping platform or registry promises? |
| Proof completeness | Can the sprint prove install, downstream compile/link/run, version query, cleanup, and claim-safe failure behavior? |
| Maintenance cost | What ongoing version, checksum, source archive, dependency, registry, tap, or policy upkeep does the provider impose? |
| Claim risk | What unsupported package-manager, binary-package, upgrade, shared-library, ABI, platform, or registry-readiness claims could users infer? |

## Day 1 Decisions

- Treat `docs/planning/EPIC_16/PROJECT_PLAN.md` as the Sprint 180 source
  authority.
- Treat Sprint 171 package-manager deferral as inherited baseline evidence,
  not as the final Sprint 180 decision.
- Treat Sprint 177 Day 8 Gate 3 as the sprint acceptance gate.
- Evaluate vcpkg, Homebrew, Conan, and pkgsrc before selecting a provider or
  stronger deferral.
- Do not add provider recipes, formulas, manifests, ports, or package-manager
  support wording on Day 1.

## Open Risks

- A provider prototype may require tool installation, registry policies,
  archive/checksum workflow, or hosted CI behavior not represented in the
  current source tree.
- Local overlay or local tap proof may be useful but could be mistaken for
  registry readiness unless docs and guards are precise.
- Provider metadata could imply dynamic ABI compatibility, shared-library
  availability, or upgrade behavior if static-first boundaries are not kept
  adjacent to positive wording.
- Stronger deferral may be the correct product decision if no provider can
  satisfy proof completeness and maintenance-cost criteria inside the sprint.

## Daily Log

### Day 1 - Provider Decision Intake

Status: Complete

Completed:

- Re-read the Sprint 180 project-plan section.
- Reviewed the Sprint 171 package-manager deferral record and current
  package-manager deferral guard.
- Reviewed Sprint 177 evidence matrix row ESM-003, target selection notes,
  Gate 3, quality map, and claim-boundary freeze.
- Created Sprint 180 working notes and artifact directory structure.
- Defined provider evaluation criteria for static-first fit, CI feasibility,
  recipe complexity, user value, proof completeness, maintenance cost, and
  claim risk.
- Created the Day 1 provider-decision-intake artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `git diff --check`

### Day 2 - Current Package Surface Audit

Status: Complete

Completed:

- Inspected Make install/uninstall ownership, CMake install/export ownership,
  `sparse.pc.in`, and `cmake/SparseConfig.cmake.in`.
- Inspected `tests/test_install.sh` and `tests/test_cmake_install.sh` for
  install, downstream compile/link/run, version query, and cleanup proof.
- Inspected `scripts/static_package_deferral_check.sh` and
  `scripts/package_manager_deferral_check.sh`.
- Inventoried README, INSTALL, maintainer guide, API reference, workflow, and
  report-index package-manager claim surfaces.
- Recorded current unsupported-provider failure behavior and proof owners for
  later provider feasibility work.
- Created the Day 2 package-surface-audit artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `python3 scripts/normalize_report_index.py --family package --check`
- `git diff --check`

### Day 3 - vcpkg Feasibility Audit

Status: Complete

Completed:

- Reviewed current official vcpkg documentation for ports, packaging a
  library, overlay ports, triplet linkage variables, `vcpkg.json` metadata,
  and usage documentation.
- Compared vcpkg port requirements against the project CMake install/export,
  static target, exact-version metadata, package templates, and install proof
  scripts.
- Checked current local environment and repository state for vcpkg tooling,
  vcpkg artifacts, version source, and license metadata.
- Assessed static-only behavior, feature flags, dependency declaration,
  license metadata, versioning expectations, local and CI proof requirements,
  maintenance burden, user value, and claim risks.
- Recorded vcpkg as eligible for a local overlay proof but not registry-ready
  support without additional evidence.
- Created the Day 3 vcpkg-feasibility artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 4 - Homebrew Feasibility Audit

Status: Complete

Completed:

- Reviewed current official Homebrew documentation for formula authoring,
  adding software, bottles, taps, FAQ bottle behavior, and support tiers.
- Compared Homebrew formula requirements against the project CMake
  install/export layout, Make install path, static package metadata,
  macOS CI package proof, and package-manager deferral guard.
- Checked local environment and repository state for `brew`, formula/tap
  artifacts, version source, architecture, and license metadata.
- Assessed static-library fit, platform scope, bottle expectations, test block
  feasibility, versioning behavior, local and CI proof requirements, formula
  complexity, maintenance burden, user value, and claim risks.
- Recorded Homebrew as eligible for local formula/tap proof but not
  Homebrew/core-ready or bottle-backed support without additional evidence.
- Created the Day 4 homebrew-feasibility artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 5 - Conan Feasibility Audit

Status: Complete

Completed:

- Reviewed current official Conan 2 documentation for creating packages,
  `test_package`, `conan create`, CMakeDeps, package info, package type, and
  settings/options package ID behavior.
- Compared Conan recipe requirements against the project CMake package,
  static-first contract, install/export proof, exact-version package behavior,
  and package-manager deferral guard.
- Checked local environment and repository state for `conan`, Conan recipe
  files, `test_package` files, version source, and license metadata.
- Assessed package ID behavior, options, generators, dependency metadata,
  versioning, profile requirements, local and CI proof requirements, recipe
  complexity, maintenance burden, user value, and claim risks.
- Recorded Conan as eligible for local recipe proof but not ConanCenter-ready
  or binary-hosted support without additional evidence.
- Created the Day 5 conan-feasibility artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 6 - pkgsrc Feasibility Audit

Status: Complete

Completed:

- Reviewed current official pkgsrc and NetBSD pkgsrc guide documentation for
  package components, package creation, build phases, fixes, options, and
  buildlink dependency handling.
- Compared pkgsrc package requirements against the project Make and CMake
  install surfaces, static package metadata, install proof scripts, and
  package-manager deferral guard.
- Checked local environment and repository state for `bmake`, `pkg_info`,
  pkgsrc package files, version source, and license metadata.
- Assessed static-library fit, platform and bootstrap assumptions, metadata
  requirements, patch requirements, dependency/buildlink policy, PLIST and
  `distinfo` proof requirements, maintenance burden, user value, and claim
  risks.
- Recorded pkgsrc as eligible only for a local package skeleton proof and as a
  low-priority first-provider candidate for Sprint 180.
- Created the Day 6 pkgsrc-feasibility artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 7 - Provider Decision Matrix

Status: Complete

Completed:

- Built a comparison matrix for vcpkg, Homebrew, Conan, and pkgsrc using the
  Day 1 evaluation criteria.
- Compared static-first fit, CI feasibility, recipe complexity, maintenance
  cost, user value, proof completeness, and claim risk across all four
  candidates.
- Identified Homebrew local formula/tap proof as the strongest first-provider
  proof candidate, with vcpkg as the strongest runner-up.
- Recorded rejected-provider rationale for vcpkg, Conan, and pkgsrc using
  Days 3-6 evidence.
- Kept stronger formal deferral as the Day 8 fallback if Homebrew cannot be
  scoped safely around source/checksum, license, bottle, Homebrew/core,
  Linuxbrew, and shared-library claim risks.
- Created the Day 7 provider-decision-matrix artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 8 - Product Decision Record

Status: Complete

Completed:

- Wrote the package-manager provider product decision record.
- Selected Homebrew local formula/tap proof as the Sprint 180 implementation
  path.
- Recorded accepted evidence, rejected alternatives, exact blockers, revisit
  criteria, stop conditions, support wording boundaries, and implementation
  validation gates.
- Preserved Sprint 171 public deferral posture until Days 9-13 complete the
  selected prototype, proof script, guard narrowing, and docs wording.
- Created the Day 8 product-decision-record artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 9 - Artifact Design

Status: Complete

Completed:

- Designed the selected Homebrew prototype as a source-controlled formula
  template plus proof-generated temporary local formula/tap material.
- Defined artifact ownership, update frequency, relationship to package
  metadata, and generated-output hygiene.
- Defined expected success and failure behavior for local proof commands,
  including missing `brew`, source archive generation, SHA-256 injection,
  static-only install checks, downstream consumer checks, version checks,
  uninstall, and cleanup.
- Defined how the template and proof path avoid unsupported Homebrew/core,
  bottle, Linuxbrew, registry, binary-package, shared-library, dynamic ABI, and
  broad package-manager claims.
- Identified guard and docs references required for Days 10-13.
- Created the Day 9 artifact-design artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 10 - Artifact Implementation

Status: Complete

Completed:

- Added `packaging/homebrew/sparse-lu-ortho.rb.in` as the selected Homebrew
  local formula template.
- Added `packaging/homebrew/README.md` to preserve local-proof-only boundaries
  and generated-output hygiene.
- Connected the template to existing package metadata by consuming `VERSION`,
  installed `Sparse::sparse_lu_ortho`, `libsparse_lu_ortho.a`,
  `SparseConfig.cmake`, and `sparse.pc` after rendering.
- Kept the template static-first by disabling optional OpenMP and mutex
  behavior and failing on shared-library artifacts.
- Recorded focused implementation checks and residual risks in the Day 10
  artifact.

Validation:

- `test -f packaging/homebrew/sparse-lu-ortho.rb.in`
- `test -f packaging/homebrew/README.md`
- `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in`
- placeholder presence checks for URL, SHA-256, version, and license
- committed `*/Formula/*` absence check
- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 11 - Proof Script Design

Status: Complete

Completed:

- Inspected the Homebrew local formula template, provider notes, package
  manager guard, CMake install proof, and existing static package boundaries.
- Reviewed current official Homebrew documentation for formula testing, formula
  class/file expectations, local tap testing, and local formula behavior.
- Designed `scripts/homebrew_local_formula_proof.sh` command flow for tool
  checks, temporary source archive creation, SHA-256 injection, formula
  rendering, local install, `brew test`, version/static-only checks, uninstall,
  cleanup, and generated-output hygiene.
- Defined missing-tool, install failure, test failure, version mismatch,
  shared-artifact, cleanup, and license-metadata failure behavior.
- Defined local-versus-CI proof split and Day 12 implementation handoff.
- Created the Day 11 proof-script-design artifact.

Validation:

- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 12 - Proof Script Implementation

Status: Complete

Completed:

- Added executable `scripts/homebrew_local_formula_proof.sh`.
- Implemented option parsing, tool checks, template placeholder validation,
  Ruby syntax checks, temporary directory handling, cleanup trap, local source
  archive generation, SHA-256 calculation, formula rendering path, Homebrew
  install/test/uninstall path, static installed-file checks, unsupported
  provider wording checks, and shared-artifact rejection.
- Implemented claim-safe unavailable behavior for missing `brew`, missing
  prerequisites, stale template placeholders, and missing license metadata.
- Ran the proof command locally and recorded the current missing-license stop
  condition: no standalone `LICENSE`, `COPYING`, or `NOTICE` file exists, so
  the script exits `2` before formula rendering/install.
- Created the Day 12 proof-script-implementation artifact.

Validation:

- `test -x scripts/homebrew_local_formula_proof.sh`
- `bash -n scripts/homebrew_local_formula_proof.sh`
- `bash scripts/homebrew_local_formula_proof.sh --help`
- `bash scripts/homebrew_local_formula_proof.sh` exits `2` claim-safely on
  missing standalone license metadata
- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 13 - Guard And Docs Update

Status: Complete

Completed:

- Updated `scripts/package_manager_deferral_check.sh` from a pure deferral
  guard to a provider claim guard that still preserves the Sprint 171
  non-claim baseline while checking the selected Homebrew local proof boundary.
- Added guard checks for selected Homebrew template, provider notes,
  executable proof script, generated-output hygiene, provider-neutral package
  metadata, public non-claim wording, and claim-safe proof-script exit
  behavior.
- Updated README and INSTALL wording to describe selected local Homebrew proof
  artifacts and the current missing-license blocker without claiming Homebrew
  support.
- Updated the maintainer guide to document
  `scripts/homebrew_local_formula_proof.sh` and the current unclaimed proof
  posture.
- Created the Day 13 guard-and-docs-update artifact.

Validation:

- `bash scripts/homebrew_local_formula_proof.sh` exits `2` claim-safely on
  missing standalone license metadata
- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `git diff --check`

### Day 14 - Integrated Validation And Closeout

Status: Complete

Completed:

- Ran the selected Homebrew local formula proof and confirmed the current
  claim-safe unavailable behavior: exit status `2` before formula rendering
  because no standalone `LICENSE`, `COPYING`, or `NOTICE` file exists.
- Ran package-manager and static package guards.
- Ran shell syntax checks, formula-template Ruby syntax check, package
  report-index normalization, generated Homebrew output hygiene check, and
  whitespace review.
- Ran Make install validation and CMake install/export validation.
- Recorded Sprint 180 closeout evidence, residual risks, Sprint 181 handoff,
  and retrospective inputs in the Day 14 artifact.

Validation:

- `bash scripts/homebrew_local_formula_proof.sh` exits `2` claim-safely on
  missing standalone license metadata
- `bash scripts/package_manager_deferral_check.sh`
- `bash scripts/static_package_deferral_check.sh`
- `bash -n scripts/homebrew_local_formula_proof.sh`
- `bash -n scripts/package_manager_deferral_check.sh`
- `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in`
- `python3 scripts/normalize_report_index.py --family package --check`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- generated Homebrew output absence check under `packaging/homebrew`
- `git diff --check`
