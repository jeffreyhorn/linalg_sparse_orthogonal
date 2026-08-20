# Sprint 171 Day 3: Provider Selection Decision

## Status

Accepted for Sprint 171.

## Decision

Sprint 171 selects **formal package-manager deferral** as the first
package-manager readiness path.

No package-manager provider is promoted to supported status in this sprint.
The project will instead add a source-controlled deferral record, guards, and
documentation that make package-manager support explicitly unsupported until a
future provider-specific proof exists.

## Rationale

Day 2 found no existing in-tree provider recipes and no provider-specific proof
surface. The current package evidence is strong for source install and static
package metadata, but it does not prove package-manager distribution.

Formal deferral is the right Sprint 171 selection because it:

- closes the current package-manager claim gap completely;
- avoids confusing Make install, CMake install/export, or `pkg-config`
  metadata with package-manager support;
- preserves the Sprint 170 static-first-only package and ABI decision;
- avoids source archive, checksum, registry, tap, bottle, profile, and
  distro-policy claims that are not yet validated;
- gives Sprint 172 and later work a guarded baseline if a provider proof is
  selected in a future sprint.

## Alternatives Considered

### vcpkg Overlay Proof

Rejected for Sprint 171 support promotion.

vcpkg is the strongest future first-provider candidate because it aligns with
the static CMake package surface and has a plausible Windows story. However,
promoting even overlay-level vcpkg support this sprint would require port
authoring, static triplet behavior, version/source metadata policy,
downstream-consumer proof, clear distinction from upstream registry support,
and guard/documentation work. That is feasible only as a narrow experiment and
still carries overclaiming risk.

### Homebrew Local Formula Proof

Rejected for Sprint 171 support promotion.

Homebrew is a plausible macOS-first provider, but credible support needs a
formula or tap boundary, source archive and checksum handling, static archive
install behavior, downstream consumer proof, and bottle/non-bottle wording.
Those requirements are larger than the safest package-manager claim closure.

### Conan Local Recipe Proof

Rejected for Sprint 171 support promotion.

Conan could serve CMake consumers, but the recipe, profiles, CMake generators,
package layout, options, and multi-platform semantics add more packaging
surface than a first provider decision should take on without dedicated proof.

### pkgsrc Or Linux Distro Packaging

Rejected for Sprint 171 support promotion.

pkgsrc, Debian, Fedora, and similar packaging paths require provider-specific
policy work, source archive and checksum metadata, static library packaging
policy, dependency metadata, and package layout proof. These remain later
candidates, not current support surfaces.

## Supported Claims After This Decision

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
- package-manager support is formally deferred and guarded by Sprint 171 work.

These are package/install and deferral claims. They are not provider support
claims.

## Unsupported Claims After This Decision

Current releases must not claim:

- vcpkg support;
- Homebrew support;
- Conan support;
- pkgsrc support;
- Debian, Fedora, or other distro package support;
- package-manager dependency resolution;
- provider-hosted binary packages;
- provider-managed version compatibility;
- provider-managed license, checksum, or source archive policy;
- provider registry readiness or upstream acceptance;
- Windows Makefile install parity;
- Windows `pkg-config` command execution parity;
- shared-library build/install support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- static/shared package selectors;
- broad platform parity;
- state-of-the-art package, distribution, install, or ABI status.

## Implementation Artifacts Needed

| Need | Planned Artifact |
| --- | --- |
| Formal deferral record | `docs/planning/EPIC_15/SPRINT_171/artifacts/day5-package-manager-deferral.md` or equivalent source-controlled decision artifact. |
| Deferral artifact design | Day 4 recipe-or-deferral artifact design. |
| Deferral validation script | A focused script or static package guard extension that proves provider claims remain unsupported. |
| Claim guard updates | Guard checks that distinguish source install, CMake/`pkg-config` install, and package-manager support. |
| User documentation | README, INSTALL, and maintainer-guide wording that directs users to source install while stating package-manager deferral. |
| Validation record | Day 12 and Day 13 artifacts recording deferral guard, install proof, claim scan, and staging hygiene results. |

## Future Provider Candidate

If a future sprint selects real provider support, vcpkg should be considered
first because it can drive the existing static CMake install/export surface and
has the best path to Windows CMake consumer proof. Any such future work must
state whether it proves an overlay port only, an upstream registry-ready port,
or a fully accepted provider package.

## Day 3 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Selected provider or deferral decision | Complete | Formal package-manager deferral is selected. |
| Supported-claim list | Complete | Current static source-install claims and the deferral claim are listed. |
| Unsupported-claim list | Complete | Provider, registry, binary package, Windows parity, shared-library, ABI, runtime-loader, broad platform, and state-of-the-art claims remain unsupported. |
| Implementation artifact list | Complete | Deferral record, guard, docs, and validation artifacts are listed. |
| Day 3 provider-selection artifact | Complete | This file. |

## Validation

Day 3 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Exactly one package-manager readiness path is selected. | Complete | Formal package-manager deferral is the selected path. |
| Claim boundaries are explicit before implementation. | Complete | Supported and unsupported claims are listed before guard/doc updates. |
| Unsupported providers remain non-claims. | Complete | No provider support claim is introduced. |
