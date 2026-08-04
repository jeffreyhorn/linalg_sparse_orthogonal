# Sprint 137 Day 10 - Package, ABI, Platform & Claim Templates

## Purpose

Day 10 defines decision and evidence templates for package/ABI work, platform
promotion, downstream proof, unsupported artifacts, and public claim gates.
These templates prepare Sprint 143 and Sprint 144 to execute without widening
package, ABI, platform, or public support claims by inference.

The Day 7 selected direction is static-first package/ABI follow-through in
Sprint 143 and one Windows CMake install/downstream platform lane in Sprint
144. Shared-library packaging, dynamic ABI compatibility, and package-manager
support remain future residuals unless a later approved plan changes scope.

## Package/ABI Decision Template

Sprint 143 must record an explicit product decision before changing package
behavior. The decision must choose exactly one primary path.

| Field | Required | Meaning |
| --- | --- | --- |
| `decision_id` | Yes | Stable identifier for the package/ABI decision. |
| `decision_owner` | Yes | Primary owner, normally Package/ABI owner. |
| `selected_path` | Yes | `static_first_follow_through`, `shared_library_abi`, or `defer_all_package_expansion`. |
| `decision_date` | Yes | Date the decision is recorded. |
| `input_evidence` | Yes | Prior install, CMake, pkg-config, static-deferral, CI, and support-tier evidence used. |
| `user_value` | Yes | User-facing problem the selected path solves. |
| `implementation_scope` | Yes | Concrete build, install, script, CI, documentation, and test changes in scope. |
| `out_of_scope` | Yes | Package and ABI behaviors intentionally not implemented. |
| `platform_scope` | Yes | Linux, macOS, Windows, supplemental, reviewed, staged, or local-only boundaries. |
| `downstream_proof_required` | Yes | Downstream Make, pkg-config, CMake, example, version, loader, or unsupported-artifact checks required. |
| `docs_required` | Yes | README, INSTALL, maintainer guide, CMake/package comments, and public non-claim updates required. |
| `validation_required` | Yes | Commands and hosted lanes required before merge. |
| `promotion_gate` | Yes | Evidence needed before any public package/ABI claim widens. |
| `residuals` | Yes | Deferred package/ABI work with owner and blocker. |

### Selected Static-First Decision Shape

```text
decision_id: epic12_static_first_follow_through_v1
decision_owner: Package/ABI owner
selected_path: static_first_follow_through
decision_date: TBD_BY_SPRINT_143
input_evidence: Make install/pkg-config; CMake install/export; static package deferral; package CI
user_value: keep the maintained static package reliable and unambiguous
implementation_scope: static install/export proof; optional static mode matrix; docs cleanup; unsupported shared-artifact checks
out_of_scope: shared libraries; dynamic ABI; package-manager recipes; runtime-loader compatibility
platform_scope: reviewed Linux package contract; supplemental macOS unless promoted later; Windows CMake lane handled by Sprint 144
downstream_proof_required: Make install; pkg-config compile/link/run; CMake find_package; exact-version check; unsupported shared-artifact checks
docs_required: README; INSTALL; maintainer guide; CMake/package comments; non-claims
validation_required: TBD_BY_SPRINT_143
promotion_gate: static archive install/export/downstream proof passes with no shared/ABI/package-manager wording
residuals: shared library ABI; package-manager support; loader compatibility
```

## Downstream Consumer Proof Template

Downstream proof must show that an installed package can be consumed from
outside the build tree. It must also show which package shape was not proven.

| Field | Required | Meaning |
| --- | --- | --- |
| `proof_id` | Yes | Stable identifier for the proof row or log. |
| `package_mode` | Yes | `static_default`, `static_mutex`, `static_openmp`, `shared`, or `unsupported`. |
| `consumer_type` | Yes | `make_pkg_config`, `cmake_find_package`, `manual_link`, `installed_example`, `version_constraint`, `loader`, or `unsupported_artifact`. |
| `install_prefix` | Yes | Install root used for the proof, normalized to avoid path-string ambiguity. |
| `library_artifact` | Yes | Expected installed library artifact path. |
| `header_artifacts` | Yes | Expected installed public header set or count. |
| `metadata_artifacts` | Yes | Expected `sparse.pc`, CMake config, CMake targets, or version files. |
| `command` | Yes | Exact command or script step used. |
| `source_commit` | Yes | Commit used for the proof. |
| `platform` | Yes | OS and architecture. |
| `compiler_or_generator` | Yes | Compiler, CMake generator, or toolchain. |
| `configuration` | Yes | Build type, library mode, optional modes, and relevant environment. |
| `expected_result` | Yes | Expected compile/link/run/status outcome. |
| `observed_result` | Yes | Observed outcome or placeholder until generated. |
| `status` | Yes | `pass`, `fail`, `skip`, `defer`, or `unsupported`. |
| `support_tier` | Yes | Reviewed, supplemental, staged, local-only, or unsupported tier. |
| `claim_scope` | Yes | Package claim supported by a pass. |
| `non_claims` | Yes | Unsupported package, ABI, loader, package-manager, or platform claims. |

## Required Proof Categories

| Category | Required for static-first path | Required for shared-library path | Notes |
| --- | --- | --- | --- |
| Make install | Yes | Yes if Make supports shared install | Must install expected headers, static archive, and metadata for static-first. |
| pkg-config cflags/libs | Yes | Yes | Must normalize paths and avoid extra whitespace or prefix ambiguity. |
| pkg-config exact version | Yes | Yes | Must prove exact package version metadata. |
| pkg-config downstream compile/link/run | Yes | Yes | Consumer must build outside the source tree. |
| CMake install/export | Yes | Yes | Must install target, config, targets, and version files. |
| CMake `find_package(Sparse)` downstream | Yes | Yes | Must compile, link, run, and check exact-version behavior. |
| Loader behavior | Not applicable | Yes | Required only for shared libraries; must include runtime search path behavior. |
| Symbol/export policy | Not applicable | Yes | Required only for dynamic ABI support. |
| ABI compatibility test | Not applicable | Yes | Required before any dynamic ABI claim. |
| Unsupported-artifact check | Yes | Yes | Static-first must prove no shared artifacts are installed; shared path must prove no stale static-only assumptions. |
| Package-manager recipe | No | No unless selected | Explicit future residual for Epic 12. |

## Unsupported-Artifact Checklist

The static-first path must fail if unsupported artifacts or metadata imply
unearned support.

| Check | Static-first expected result | Failure meaning |
| --- | --- | --- |
| Shared library files | No `.so`, `.dylib`, `.dll`, `.lib` import library, or shared CMake target is installed. | Static-first contract is ambiguous or broken. |
| Dynamic ABI metadata | No ABI epoch, soname, install-name, import/export macro claim, or compatibility statement appears in package metadata or docs. | Dynamic ABI claim is implied without proof. |
| Runtime loader notes | No loader-path, RPATH, PATH, install-name, or delay-load instruction is presented as maintained behavior. | Loader behavior claim is implied without proof. |
| Package-manager wording | No Homebrew, vcpkg, apt, yum, conda, Spack, NuGet, or system package support is claimed. | Package-manager support is implied without recipes and proof. |
| Static optional modes | Optional `SPARSE_MUTEX` or `SPARSE_OPENMP` rows state exact build/install/downstream proof and support tier. | Optional mode support is ambiguous. |
| Public docs alignment | README, INSTALL, maintainer guide, pkg-config, and CMake comments agree on static-first status. | Public package contract is inconsistent. |

## Platform Promotion Template

Sprint 144 must use this template before promoting any platform lane from
supplemental or staged status to reviewed status.

| Field | Required | Meaning |
| --- | --- | --- |
| `platform_decision_id` | Yes | Stable identifier for the platform promotion decision. |
| `selected_lane` | Yes | `linux_source_of_truth`, `macos_install_export`, `windows_cmake_install_downstream`, or `windows_staged_posix_tests`. |
| `current_tier` | Yes | Current support tier before work starts. |
| `target_tier` | Yes | Intended support tier after successful proof. |
| `lane_owner` | Yes | Platform owner and supporting package/test owners. |
| `hosted_ci_required` | Yes | Workflow/job names and required commands. |
| `local_checks_required` | Yes | Local checks feasible before relying on hosted CI. |
| `expected_counts` | Conditional | Expected CTest, header, artifact, or proof counts when relevant. |
| `failure_semantics` | Yes | What a failure means and who triages it. |
| `source_or_script_changes` | Conditional | Source, script, path, shell, CMake, PowerShell, or portability changes needed. |
| `package_integration` | Conditional | Package/report/freshness rows affected by platform promotion. |
| `docs_required` | Yes | README, INSTALL, maintainer guide, CI comments, support-tier wording. |
| `promotion_gate` | Yes | Exact evidence required before target tier is considered earned. |
| `fallback_decision` | Yes | What is recorded if hosted proof fails or blockers remain. |

### Selected Windows Lane Shape

```text
platform_decision_id: epic12_windows_cmake_install_downstream_v1
selected_lane: windows_cmake_install_downstream
current_tier: supplemental_windows
target_tier: reviewed_windows_package_lane
lane_owner: Platform owner
hosted_ci_required: Windows CMake install/downstream job with install, package config, exact-version, mismatch-version, example build, and example run proof
local_checks_required: CMake/package script review; docs link checks; any touched script syntax checks
expected_counts: installed headers; package files; no shared-library artifacts; CTest count if affected
failure_semantics: lane remains supplemental or rejected with blocker list
source_or_script_changes: PowerShell/CMake/path fixes only as needed
package_integration: Sprint 143 static-first package semantics must remain unchanged
docs_required: README; INSTALL; maintainer guide; workflow comments
promotion_gate: hosted Windows proof passes and support-tier docs match exact CMake-first scope
fallback_decision: publish blockers and keep Windows install/downstream supplemental
```

## Public Claim Gate Template

Every package, ABI, platform, and adoption claim update must pass a public
claim gate before merge.

| Field | Required | Meaning |
| --- | --- | --- |
| `claim_id` | Yes | Stable identifier for the claim. |
| `claim_text` | Yes | Exact public wording proposed. |
| `claim_surface` | Yes | README, INSTALL, maintainer guide, CMake package, pkg-config, workflow comment, example, tutorial, cookbook, or release note. |
| `evidence_rows` | Yes | Tests, CI jobs, report rows, package proofs, or generated artifacts backing the claim. |
| `validation_commands` | Yes | Commands or hosted lanes that must pass. |
| `support_tier` | Yes | Reviewed, supplemental, staged, local-only, optional, or unsupported boundary. |
| `docs_updated` | Yes | Surfaces updated together to avoid contradictory wording. |
| `non_claims` | Yes | Related claims still blocked. |
| `owner` | Yes | Workstream accountable for keeping the claim true. |
| `rollback_or_demote_condition` | Yes | Condition that requires reverting, demoting, or rewriting the claim. |

## Claim Gate Examples

| Proposed claim | Evidence required | Allowed wording | Blocked wording |
| --- | --- | --- | --- |
| Static-first package contract | Make install, pkg-config, CMake install/export, downstream consumers, exact-version checks, and unsupported shared-artifact checks pass. | Static library install/export is maintained for the specified support tier. | Shared ABI, dynamic loader, package-manager, or all-platform install support. |
| Optional static mode | Optional mode build/install/downstream proof passes with support-tier metadata. | Optional static mode is validated for the stated configuration and tier. | Optional mode parity across all platforms or loaders. |
| Windows CMake install/downstream reviewed lane | Hosted Windows job passes with expected files, example run, version checks, and docs alignment. | Windows CMake install/downstream lane is reviewed for the exact static-first scope. | General Windows parity, POSIX test promotion, Make/pkg-config parity, or package-manager support. |
| Unsupported shared-library path | Static-deferral guard rejects shared mode and docs explain deferral. | Shared-library support is intentionally unsupported/deferred. | Shared libraries are available, ABI compatible, or loader-safe. |

## Stop Conditions

- A package decision tries to implement shared-library support without symbol,
  visibility, ABI, loader, install/export, downstream, platform, and docs proof.
- A static-first change installs or documents shared artifacts, dynamic ABI, or
  package-manager support by implication.
- A downstream proof compiles only inside the source/build tree.
- A version proof checks a string but not exact package discovery behavior.
- A platform promotion relies on local evidence when the template requires
  hosted CI evidence.
- A Windows promotion expands beyond the CMake install/downstream lane into
  general Windows parity or staged POSIX/pthread tests.
- Public docs widen support tiers without validation commands and evidence
  rows.
- Any generated report or package log is treated as release proof or broad
  platform parity proof.

## Sprint 143-144 Handoff

Sprint 143 should:

1. Record the package/ABI product decision using the decision template.
2. Implement the selected static-first follow-through path.
3. Strengthen Make/pkg-config/CMake/downstream proof and optional static-mode
   boundaries.
4. Run unsupported-artifact checks and package docs alignment.
5. Publish residuals for shared libraries, dynamic ABI, loader behavior, and
   package-manager support.

Sprint 144 should:

1. Use Sprint 143 static-first semantics as an input.
2. Fill the platform promotion template for Windows CMake install/downstream.
3. Implement only the source/script/CI/docs changes required for that exact
   lane.
4. Promote the lane only after hosted proof and support-tier docs pass.
5. Otherwise publish blockers and keep the lane supplemental.

## Day 10 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 143 can decide package/ABI direction without missing proof categories. | Complete | Package/ABI decision template, required proof categories, downstream proof template, and unsupported-artifact checklist define the decision and proof surface. |
| Sprint 144 can evaluate platform promotion without inference. | Complete | Platform promotion template and selected Windows lane shape require current tier, target tier, hosted CI, expected counts, failure semantics, docs, and fallback decision. |
| Claim gates require docs and validation alongside implementation. | Complete | Public claim gate template, examples, and stop conditions require evidence rows, validation commands, support tier, docs updates, non-claims, owner, and rollback/demotion conditions. |
