# Sprint 207 Day 6: Proof And Deferral Design

## Purpose

Design the selected continued-deferral implementation before editing package
scripts, guards, or documentation.

## Selected Implementation Path

Day 5 selected continued package-provider deferral with stronger guards. Day 6
therefore designs guard and documentation hardening around the existing local
proof instead of designing public provider promotion.

The retained proof command is:

```sh
HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh
```

## Proof State Semantics

| State | Exit | Required diagnostic meaning | Claim interpretation |
| --- | ---: | --- | --- |
| Pass | `0` | Local proof completed archive/checksum, temporary tap render, source install, installed static surface validation, downstream `brew test`, uninstall, and cleanup. | May cite developer-mode local static source formula proof only. |
| Unavailable | `2` | Local tool, host condition, or approved metadata is unavailable. | Package-manager support remains unclaimed. |
| Fail | Other nonzero | Project proof, formula, install, downstream test, or cleanup defect. | Fix defect before changing support wording. |

The proof command must not become evidence for:

- public Homebrew tap support;
- Homebrew/core readiness or acceptance;
- bottle support;
- Linuxbrew support;
- hosted binary packages;
- vcpkg, Conan, pkgsrc, distro/system package support;
- shared-library package support;
- dynamic ABI compatibility;
- broad package-manager distribution.

## Guard Implementation Design

### Package-Manager Guard

Primary owner: `scripts/package_manager_deferral_check.sh`

Required Day 7-9 hardening targets:

1. Keep existing Sprint 171 deferral record and Sprint 198 proof record checks.
2. Keep provider recipe absence checks for vcpkg, Conan, pkgsrc, distro
   package specs, RPM specs, and committed Homebrew formula/tap outputs.
3. Explicitly reject positive package-provider wording in public docs and
   maintainer docs, including:
   - package-manager distribution is supported or available;
   - Homebrew/core readiness is claimed;
   - public tap support is claimed;
   - bottle support is claimed;
   - Linuxbrew support is claimed;
   - vcpkg, Conan, pkgsrc, distro/system packages, or binary packages are
     claimed.
4. Require exact local-proof wording that says the Homebrew proof is
   developer-mode local static source formula proof only.
5. Require exact future-evidence wording for public tap/source formula and
   Homebrew/core readiness residuals.
6. Preserve generated-output hygiene checks under `packaging/homebrew`.

### Static Package Guard

Primary owner: `scripts/static_package_deferral_check.sh`

Required Day 7-9 hardening targets:

1. Keep `BUILD_SHARED_LIBS=ON` rejection evidence.
2. Keep static archive install contract checks.
3. Keep package metadata free of shared-library selectors, dynamic ABI
   wording, `Libs.private`, SONAME/DLL/dylib policy, and static/shared
   selection knobs.
4. Keep public docs from implying shared-library package support or dynamic
   ABI support through package-provider wording.

## Documentation Design

### User-Facing Wording

README and INSTALL should keep this model:

- use source install via Make or CMake for user-facing installs;
- package-manager distribution is not claimed;
- the Homebrew proof is developer-mode local static source formula proof only;
- no Homebrew/core, public tap, bottles, Linuxbrew, binary packages, vcpkg,
  Conan, pkgsrc, distro/system packages, shared-library package support, or
  dynamic ABI support is claimed.

### Maintainer Wording

`docs/maintainer_guide.md` and `packaging/homebrew/README.md` should explain:

- exact local proof command;
- proof pass/unavailable/fail interpretation;
- environment and cleanup expectations;
- future evidence required to reopen public tap/source formula support;
- future evidence required to reopen Homebrew/core readiness;
- guard commands that must pass before package wording changes.

### Planning Wording

Sprint 207 artifacts should retain:

- Day 5 selected decision;
- rejected option residuals;
- exact evidence required to revisit provider promotion;
- no broad package-manager support claim.

## Environment-Gate Design

| Environment condition | Desired behavior |
| --- | --- |
| Missing `brew` | Proof exits unavailable and says local Homebrew proof remains unclaimed. |
| Missing `cmake`, `ruby`, `tar`, checksum tool, or C compiler | Proof exits unavailable and keeps support unclaimed. |
| Missing standalone root license metadata | Proof exits unavailable before archive/render/install/test work. |
| Placeholder `SPARSE_HOMEBREW_LICENSE` | Proof exits unavailable before archive/render/install/test work. |
| Homebrew provider/platform refusal without developer mode | Treat as provider/environment limitation, not package support evidence. |
| Current-run temporary tap or formula remains after success/failure | Treat as cleanup defect. |
| Generated proof output appears under `packaging/homebrew` | Treat as guard failure. |

## Validation Mapping

| Planned change | Required validation |
| --- | --- |
| Package-manager guard hardening | `bash scripts/package_manager_deferral_check.sh` |
| Static package boundary wording or guard hardening | `bash scripts/static_package_deferral_check.sh` |
| Homebrew proof script or formula template edits | `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh`; `bash scripts/package_manager_deferral_check.sh` |
| README, INSTALL, Homebrew README, or maintainer guide edits | `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh` if static/shared wording is affected; `make docs-check` |
| Install metadata or CMake package metadata edits | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh` |
| C source or header edits | `make format && make lint && make test` |
| Planning-only edits | `git diff --check`; focused grep review for decision and non-claim consistency |

## Regression Targets For Days 7-9

| Regression target | Rationale |
| --- | --- |
| Reject public tap support wording without evidence. | Prevent local proof from becoming public provider support by wording drift. |
| Reject Homebrew/core readiness wording without evidence. | Prevent readiness/acceptance overclaim. |
| Reject bottle and Linuxbrew support wording. | These require separate provider evidence. |
| Reject other provider recipes or support wording. | One Homebrew proof must not imply vcpkg, Conan, pkgsrc, or distro support. |
| Require future evidence wording. | Make the deferral actionable rather than vague. |
| Require local-proof-only wording. | Preserve the exact Sprint 198/Sprint 207 boundary. |

## Implementation Boundary

Allowed implementation after this design:

- guard hardening;
- claim-safe documentation updates;
- residual and closeout planning evidence;
- focused validation and regression checks.

Disallowed implementation after this design:

- adding a user-facing public tap install claim;
- claiming Homebrew/core readiness;
- adding bottle/Linuxbrew support claims;
- adding other package-manager support claims;
- weakening static package or dynamic ABI non-claims.

## Day 6 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.3 has an implementation-ready design. | Complete. |
| Selected proof commands have clear pass, fail, and blocked states. | Complete. |
| Deferral or promotion cannot accidentally imply broader package support. | Complete; guard, docs, and validation mapping are defined for continued deferral. |

## Day 6 Outcome

Sprint 207 implementation should now proceed as guard and documentation
hardening for continued package-provider deferral. The existing local proof
remains valuable evidence, but Sprint 207 will not convert it into public tap,
Homebrew/core, bottle, Linuxbrew, binary package, shared-library, ABI, or broad
package-manager support.
