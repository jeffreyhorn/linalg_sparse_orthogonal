# Sprint 171 Day 4: Recipe Or Deferral Artifact Design

## Purpose

Day 4 designs the source-controlled artifact for the Day 3 decision. Because
Sprint 171 selected formal package-manager deferral, the implementation will
not add a vcpkg port, Homebrew formula, Conan recipe, pkgsrc package, or distro
package metadata. Instead, Day 5 should add a formal deferral record that
states the unsupported provider boundary and the evidence required to revisit
it.

## Selected Artifact Shape

Day 5 should add:

`docs/planning/EPIC_15/SPRINT_171/artifacts/day5-package-manager-deferral.md`

The artifact should use this structure:

1. **Status**
   - Accepted for Sprint 171.
2. **Decision**
   - Package-manager support is formally deferred.
   - No provider is supported by current releases.
3. **Scope**
   - Source install remains maintained.
   - Package-manager distribution remains unsupported.
4. **Supported Claims**
   - Static archive source build/install.
   - Unix Make install/`pkg-config` proof.
   - Unix CMake install/export proof.
   - Linux/macOS static-first package CI lanes.
   - Windows CMake install/downstream validation.
   - Formal package-manager deferral.
5. **Unsupported Claims**
   - vcpkg, Homebrew, Conan, pkgsrc, Debian/Fedora/system packages, provider
     registries, binary packages, provider dependency/version/license/checksum
     policy, Windows Makefile parity, Windows `pkg-config` execution parity,
     shared-library support, dynamic ABI compatibility, runtime-loader
     behavior, broad platform parity, and state-of-the-art package status.
6. **Evidence Needed To Revisit**
   - Provider selection.
   - Source archive/checksum plan.
   - License and metadata plan.
   - Static/shared package policy.
   - Provider recipe or manifest.
   - Isolated install proof.
   - Downstream compile/link/run proof.
   - Version query proof.
   - Cleanup proof.
   - Documentation and guard updates.
7. **Consequences**
   - Public docs must route users to source install paths.
   - Guards must fail if provider support claims appear before provider proof.
   - Future provider work must name whether it is local overlay, tap/recipe,
     upstream registry-ready, or accepted upstream support.
8. **Validation**
   - Planning-only validation for Day 5 unless scripts/docs are also changed.

## Static-First Mapping

| Existing Surface | Deferral Artifact Mapping |
| --- | --- |
| `make install` | Remains source-install proof only. It must not be described as Homebrew, distro, pkgsrc, Conan, or vcpkg support. |
| `cmake --install` | Remains static CMake package install/export proof only. It may be used by a future provider but does not itself prove provider support. |
| `pkg-config` | Remains generated metadata for Unix-style installed static consumers. It is not package-manager distribution evidence. |
| Windows CMake install/downstream validation | Remains the reviewed Windows source-installed CMake consumer path. It does not prove Windows package-manager support. |
| `scripts/static_package_deferral_check.sh` | Remains the static package and shared-library ABI guard; Day 8/Day 9 can extend or pair it with package-manager deferral checks. |

## Version, Source, Metadata, And License Requirements

Any future provider support must define:

| Requirement | Minimum Evidence Needed Before Support |
| --- | --- |
| Version | Provider metadata matches `VERSION` and does not imply dynamic ABI compatibility. |
| Source | Source input is reproducible and described as checkout, archive, tap, overlay, or registry source. |
| Checksum | Immutable checksum policy exists for archive-based providers or is explicitly out of scope for local overlay proofs. |
| License | Provider metadata identifies the project license accurately. |
| Dependencies | Required and optional dependencies are explicit; optional OpenMP/mutex behavior is not enabled implicitly. |
| Static/shared policy | Static-first behavior is explicit; no shared-library, runtime-loader, or static/shared selector support is implied. |
| Downstream consumer | A provider-installed consumer compiles, links, runs, and reports expected version behavior. |
| Cleanup | Temporary installs, caches, build directories, and binary package outputs are removed or ignored. |

## Guard Entry Points

Day 8/Day 9 guard work should consider:

- adding package-manager deferral checks to `scripts/static_package_deferral_check.sh`;
- requiring the Day 5 deferral record and selected wording;
- checking README, INSTALL, and maintainer-guide package-manager wording;
- scanning package metadata templates for provider names or package-manager
  claims;
- checking no in-tree provider recipes are present unless a future provider
  decision selects them;
- preserving Sprint 170 shared-library ABI guards unchanged.

Provider-name scans should be targeted. Planning artifacts need to mention
candidate providers, so broad repository-wide zero-match checks would create
false positives.

## Expected Failure Modes

| Failure Mode | Handling |
| --- | --- |
| A provider recipe appears without a provider decision | Guard should fail and point to the missing provider decision/proof. |
| README or INSTALL says a provider is supported | Guard or claim scan should fail unless provider proof exists. |
| CMake or `pkg-config` metadata mentions package managers | Guard should fail because package metadata should describe installed static package semantics only. |
| Source archive/checksum is unresolved | Keep provider support deferred. |
| Provider tooling is unavailable locally | Deferral remains valid; future provider proof must define skip or hosted behavior. |
| Shared-library metadata appears during provider work | Stop and preserve Sprint 170 static-first-only package decision. |
| Generated package caches or binary outputs appear in status | Remove or ignore them before staging. |

## Rollback Criteria

Rollback or stop before proceeding if implementation:

- adds provider support wording without provider proof;
- commits provider cache, source archive, binary package, install-prefix, or
  build output;
- weakens static package deferral checks;
- introduces shared-library, dynamic ABI, runtime-loader, or static/shared
  selector claims;
- makes Windows package-manager or Windows `pkg-config` parity claims from
  CMake install evidence alone;
- cannot pass `git diff --check` after artifact creation.

## Day 4 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Recipe or deferral artifact design | Complete | Formal deferral record shape is specified for Day 5. |
| Version/source metadata requirements | Complete | Version, source, checksum, license, dependency, static/shared, consumer, and cleanup requirements are listed. |
| Static-first mapping notes | Complete | Current Make, CMake, `pkg-config`, Windows, and guard surfaces are mapped to deferral semantics. |
| Failure-mode list | Complete | Failure modes and handling are listed. |
| Day 4 artifact-design artifact | Complete | This file. |

## Validation

Day 4 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Implementation shape is clear before files are changed. | Complete | Day 5 deferral record sections and required content are specified. |
| Selected path preserves Sprint 170 package/ABI boundaries. | Complete | Static-first source install remains separate from provider support. |
| Deferral criteria are explicit if no provider is selected. | Complete | Evidence needed to revisit package-manager support is listed. |
