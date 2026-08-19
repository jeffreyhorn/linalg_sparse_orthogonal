# Sprint 171 Day 2: Provider Candidate Inventory

## Purpose

Day 2 inventories package-manager provider candidates for Sprint 171. The goal
is not to select support yet; it is to make viable and non-viable paths visible
before Day 3 chooses one first-provider readiness path or formal deferral.

Sprint 171 starts from the Sprint 170 static-first package decision. Source
install, CMake package discovery, and `pkg-config` metadata remain maintained
static package surfaces, not package-manager distribution support.

## Current In-Tree Package Baseline

| Surface | Current Evidence | Package-Manager Meaning |
| --- | --- | --- |
| Make install | Installs `libsparse_lu_ortho.a`, public headers, generated `sparse_version.h`, and `sparse.pc`. | Useful source-install substrate, not provider support. |
| CMake install/export | Installs static target metadata under `lib/cmake/Sparse` and supports `find_package(Sparse)`. | Useful for providers that drive CMake builds. |
| `sparse.pc.in` | Generates static archive `pkg-config` metadata. | Useful metadata for Unix providers, not a provider recipe. |
| Windows CMake lane | Validates installed static `.lib`, CMake package files, and metadata-only `sparse.pc` inspection. | Useful Windows CMake evidence, not Windows package-manager parity. |
| Provider recipes | No vcpkg port, Homebrew formula, Conan recipe, pkgsrc package, distro spec, or package-manager manifest found in-tree. | No current package-manager support claim. |

## Provider Candidate Matrix

| Candidate | Expected Artifact | Platform Scope | Proof Needed | Readiness | Day 2 Assessment |
| --- | --- | --- | --- | --- | --- |
| vcpkg | `ports/sparse/portfile.cmake`, `vcpkg.json`, optional overlay-port docs | Windows, Linux, macOS through vcpkg overlay | configure/build/install static library; verify installed headers, CMake package, version metadata, no DLL/shared artifacts, downstream CMake consumer; optional feature policy for OpenMP/mutex | Medium | Strong candidate if the sprint accepts an overlay-port proof instead of upstream registry support. Best Windows story, but recipe semantics and port naming need care. |
| Homebrew | `Formula/sparse.rb` or documentation-only formula artifact | macOS, optional Linuxbrew | fetch local/source archive or checkout, build static archive with CMake or Make, install headers/package metadata, run installed consumer, verify no shared artifacts | Medium | Natural fit for macOS static package proof. Harder to make fully local without source archive/checksum handling or a tap boundary. |
| Conan | `conanfile.py` or `conanfile.txt` plus profile guidance | Linux, macOS, Windows depending profiles | create package layout, CMakeToolchain/CMakeDeps proof, package info for static library, downstream consumer, version and options checks | Low-Medium | Powerful but adds more packaging semantics than needed for first provider; proof cost is high for one sprint. |
| pkgsrc | package Makefile, PLIST, distinfo, patches | NetBSD/pkgsrc platforms and portable Unix | source archive/checksum, staged install, PLIST accuracy, buildlink metadata, installed consumer where available | Low | Broad Unix packaging value, but high maintenance and likely needs external tree conventions not already present. |
| Debian/Fedora/system packages | `debian/` packaging or RPM spec | Linux distro-specific | distro policy compliance, source tarball, license metadata, dependency policy, static library packaging policy, install and consumer proof | Low | Too broad for a first provider unless the sprint chooses one distro as a narrow proof. Static library policy varies by distro. |
| Documented deferral | formal package-manager deferral artifact plus guard/doc updates | All providers remain unsupported | decision record, public docs, maintainer docs, guard checks, install validation, targeted claim scans | High | Safest if Day 3 decides no provider can be credibly proven. Closes the claim gap by making provider support explicitly unsupported and guarded. |

## Proof Requirement Details

### Common Requirements For Any Selected Provider

- Provider decision record naming exactly one selected provider or formal
  deferral.
- Static-first-only package behavior unless a future decision changes the
  package product.
- Installed static archive and public headers.
- Installed version metadata that agrees with `VERSION`.
- Installed CMake package files or documented reason they are unavailable.
- Installed `sparse.pc` metadata where the provider supports Unix
  `pkg-config`.
- Downstream compile/link/run proof using the provider-installed package.
- Cleanup or isolated temporary install behavior.
- No staged shared-library artifacts unless a future shared-library product
  decision exists.
- No generated source archives, provider caches, install prefixes, or binary
  package outputs committed unintentionally.

### Version, Source, License, And Checksum Requirements

| Requirement | Provider Impact |
| --- | --- |
| Version | Provider metadata must derive from or match `VERSION`; exact CMake package-version semantics must not be described as ABI compatibility. |
| Source archive | Upstreamable provider recipes generally require a stable source archive URL; local overlay proofs can use the checkout but must not claim upstream registry readiness. |
| Checksum | Upstreamable formulas/ports/specs usually require immutable checksum metadata; local deferral or overlay proof can document this as unresolved. |
| License | Provider metadata must identify the project license accurately before claiming distribution readiness. |
| Dependencies | Current base package requires C compiler, CMake or Make, and math library; optional OpenMP/mutex behavior must not become implicit provider support. |
| Static/shared policy | Current package is static-first-only; provider metadata must not imply shared library, dynamic ABI, runtime-loader, or static/shared selector support. |

## Proof-Cost Estimate

| Candidate | Estimated Proof Cost | Cost Drivers |
| --- | ---: | --- |
| vcpkg overlay proof | 70-95 hours | port authoring, static triplet behavior, Windows/Linux/macOS differences, installed CMake consumer, version metadata, guard/docs. |
| Homebrew local formula proof | 65-90 hours | local formula/tap mechanics, source/checksum story, macOS install proof, bottle/non-bottle wording, guard/docs. |
| Conan local recipe proof | 90-120 hours | recipe layout, profiles, CMake generators, package info, multi-platform behavior, consumer package proof, guard/docs. |
| pkgsrc proof | 100-135 hours | external packaging conventions, PLIST/distinfo, source archive/checksum, platform policy, guard/docs. |
| Debian or Fedora proof | 95-135 hours | distro policy, source package layout, static library policy, metadata, platform-specific tooling, guard/docs. |
| Formal deferral | 45-65 hours | decision artifact, docs, guard checks, install proof preservation, claim scans. |

All candidates fit inside the Sprint 171 total budget only if scoped narrowly.
Only formal deferral has low enough proof cost to close the claim boundary
without provider tooling risk.

## Provider Risks

| Risk | Impact |
| --- | --- |
| Confusing overlay/local proof with upstream registry support | Could overstate package-manager readiness before registry review exists. |
| Source archive and checksum absence | Blocks credible upstream formula/spec/port claims. |
| Static-only package policy mismatch | Some providers may expect shared libraries, split packages, or explicit static options. |
| Windows provider semantics | vcpkg is plausible, but Windows package proof must not imply Windows Makefile or Windows `pkg-config` execution parity. |
| Optional OpenMP/mutex flags | Provider options could imply unsupported dependency/linkage behavior if not explicitly selected and tested. |
| License and metadata drift | Provider recipes require accurate license, version, homepage, and source metadata; stale values can become distribution defects. |
| Generated artifact staging | Provider experiments can produce archives, package caches, install trees, lockfiles, and build directories that must not be committed accidentally. |

## Day 3 Candidate Ranking

| Rank | Candidate | Reason |
| --- | --- | --- |
| 1 | Formal deferral | Most reliable way to close the package-manager claim gap if the sprint prioritizes enforceable non-claims over incomplete provider support. |
| 2 | vcpkg overlay proof | Best first provider if support is selected because it aligns with static CMake package metadata and Windows CMake validation. |
| 3 | Homebrew local formula proof | Good macOS-focused path, but source archive/checksum and tap/upstream boundaries need careful non-claim wording. |
| 4 | Conan local recipe proof | Useful for CMake consumers but heavier than needed for first-provider closure. |
| 5 | Debian/Fedora/pkgsrc | Valuable later, but too policy-heavy for a first provider unless sharply reduced to documentation-only exploration. |

## Day 2 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Provider candidate matrix | Complete | vcpkg, Homebrew, Conan, pkgsrc, system packages, and formal deferral are compared. |
| Proof-cost estimate | Complete | Candidate proof ranges are listed with cost drivers. |
| Platform/dependency requirement notes | Complete | Common proof, version, source, license, checksum, dependency, and static/shared requirements are listed. |
| Provider risk list | Complete | Main overclaiming, tooling, policy, and generated-output risks are recorded. |
| Day 2 provider-inventory artifact | Complete | This file. |

## Validation

Day 2 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Viable and non-viable provider paths are visible. | Complete | Candidate readiness and proof costs are listed. |
| Deferral remains available if no provider can be proven safely. | Complete | Formal deferral is ranked as the safest claim-boundary closure path. |
| Provider support is not inferred from source install support. | Complete | Current Make, CMake, and `pkg-config` surfaces are documented as source-install evidence only. |
