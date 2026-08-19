# Sprint 171 Day 5: Package-Manager Deferral Record

## Status

Accepted for Sprint 171.

## Decision

Package-manager support is formally deferred.

Current releases do not support a package-manager provider. No vcpkg,
Homebrew, Conan, pkgsrc, Debian, Fedora, system-package, registry, tap, or
binary-package support claim is promoted by Sprint 171.

## Scope

The maintained package surface remains the static-first source-install
surface:

- Unix Make install/uninstall plus `pkg-config` proof;
- Unix CMake install/export plus `find_package(Sparse)` proof;
- Linux and macOS reviewed static-first package CI lanes;
- Windows CMake install/downstream validation for the maintained static
  package surface;
- metadata-only Windows `sparse.pc` inspection.

Those surfaces are source-install and installed-consumer evidence. They are not
package-manager distribution evidence.

## Supported Claims

Current releases may claim:

- maintained static archive source build support;
- maintained static archive source install support;
- Unix-side Make install/uninstall plus `pkg-config` proof;
- Unix-side CMake install/export plus `find_package(Sparse)` proof;
- reviewed Linux static-first package-contract CI;
- reviewed macOS static-first Make install/`pkg-config` and CMake
  install/export proof;
- reviewed Windows CMake install/downstream validation for the static-first
  package surface;
- package-manager support is formally deferred.

These claims are limited to source install, installed static consumers, and
formal deferral.

## Unsupported Claims

Current releases must not claim:

- vcpkg support;
- Homebrew support;
- Conan support;
- pkgsrc support;
- Debian package support;
- Fedora/RPM package support;
- package-manager dependency resolution;
- provider-hosted binary packages;
- provider-managed version compatibility;
- provider-managed license, checksum, or source archive policy;
- provider registry readiness or upstream acceptance;
- tap readiness or bottle support;
- Windows package-manager support;
- Windows Makefile install parity;
- Windows `pkg-config` command execution parity;
- shared-library build/install support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- Linux SONAME support;
- macOS install-name/RPATH support;
- Windows DLL/import-library support;
- static/shared package selectors;
- broad platform parity;
- state-of-the-art package, distribution, install, or ABI status.

## Evidence Needed To Revisit

A future sprint may revisit package-manager support only after selecting one
provider and defining the proof level:

- local overlay proof only;
- local tap/formula proof only;
- registry-ready metadata;
- accepted upstream provider package.

Before any support claim, that sprint must provide:

| Evidence | Requirement |
| --- | --- |
| Provider selection | Exactly one provider selected with explicit platform scope. |
| Source input | Checkout, archive, tap, overlay, or registry source identified. |
| Checksum policy | Immutable checksum policy for archive-based providers, or explicit local-overlay exception. |
| License metadata | Provider metadata identifies the project license accurately. |
| Version metadata | Provider metadata matches `VERSION` and does not imply dynamic ABI compatibility. |
| Dependency policy | Required and optional dependencies are explicit; OpenMP and mutex behavior are not enabled implicitly. |
| Static/shared policy | Static-first behavior is selected; shared-library, runtime-loader, and static/shared selector support remain out of scope unless separately proven. |
| Recipe or manifest | Source-controlled provider recipe, formula, manifest, port, or package specification. |
| Isolated install proof | Provider installs into an isolated prefix or package tree. |
| Downstream consumer proof | Provider-installed package compiles, links, runs, and reports expected version behavior. |
| Cleanup proof | Temporary installs, caches, archives, package outputs, and build directories are cleaned or ignored. |
| Documentation | README, INSTALL, maintainer docs, and any provider docs state the exact support level. |
| Guard coverage | Automated or focused checks fail on unsupported provider claims and generated artifact staging. |

## Consequences

Immediate consequences:

- Do not add provider recipe files in Sprint 171 unless a later sprint changes
  the product decision.
- Public docs should route users to source install paths and describe
  package-manager support as deferred.
- Package metadata should stay provider-neutral and static archive scoped.
- Guards should fail if provider support claims appear without a provider
  proof decision.
- Future provider work must state whether it proves local overlay/tap
  behavior, registry readiness, or accepted upstream package support.

Deferred consequences:

- vcpkg remains the strongest future first-provider candidate because it can
  build on the static CMake install/export surface and has a plausible Windows
  story.
- Homebrew remains a plausible future macOS-first path once source
  archive/checksum and tap boundaries are selected.
- Conan, pkgsrc, Debian, Fedora, and other system package routes remain later
  candidates with higher policy and proof cost.

## Generated-Output Hygiene

No provider cache, package archive, source tarball, binary package, build tree,
install prefix, lockfile, or generated package-manager output should be
committed as part of the deferral decision.

## Day 5 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Provider recipe/proof artifact or deferral record | Complete | Formal package-manager deferral record is source-controlled in this file. |
| Focused artifact validation notes | Complete | Planning-only validation is listed below. |
| Generated-output hygiene check | Complete | Deferral record states generated package artifacts must not be committed. |
| Day 5 artifact-implementation artifact | Complete | This file. |

## Validation

Day 5 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Selected artifact exists in source control. | Complete | The formal package-manager deferral record exists. |
| Unsupported package-manager claims remain out of scope. | Complete | Provider and package-manager claims are explicitly unsupported. |
| No generated package/build outputs are added accidentally. | Complete | Day 5 adds only planning artifacts. |
