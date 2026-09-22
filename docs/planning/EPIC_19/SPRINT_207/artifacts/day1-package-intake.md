# Sprint 207 Day 1: Package Intake

## Purpose

Establish Sprint 207 scope, inherited package evidence, and package claim
boundaries before changing provider-facing package files, guards, or public
documentation.

## Sprint 207 Scope

Sprint 207 is the Epic 19 package distribution support decision sprint. Its
goal is to promote one exact package distribution path, or to close the
package distribution residual with stronger deferral guards and no
user-facing package claim.

The six project-plan items are:

| Item | Day 1 interpretation |
| --- | --- |
| 207.1 Provider Scope Decision | Decide between a public Homebrew tap/source formula, Homebrew/core readiness, or explicit continued deferral. |
| 207.2 Formula And Metadata Audit | Audit Homebrew formula inputs, root MIT metadata, source archive/checksum behavior, license wording, and installed static surface. |
| 207.3 Proof Path Implementation | Add or update the selected proof command, formula checks, environment gates, and cleanup behavior. |
| 207.4 Package Guard Alignment | Align package-manager and static-package guards with the selected tier and retained non-claims. |
| 207.5 User And Maintainer Docs | Update README, INSTALL, packaging docs, and maintainer guide with only earned provider status. |
| 207.6 Validation And Closeout | Run package proof, install tests, docs guards, static package guard, and the full C gate only if `.c` or `.h` files change. |

## Inherited Evidence

| Source | Current evidence |
| --- | --- |
| Sprint 198 retrospective | Sprint 198 closed with developer-mode local Homebrew static source formula proof completed; broad package/Homebrew support remains unclaimed. |
| Sprint 198 Day 14 closeout | `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT` completed archive/checksum, temporary tap rendering, source install, installed static package surface validation, downstream `brew test`, uninstall, cleanup, and proof exit `0` on macOS Intel x86_64 Tier 3 Homebrew. |
| `packaging/homebrew/README.md` | Homebrew material is proof-only and not a Homebrew/core formula, tap, bottle, Linuxbrew claim, or general package-manager support. |
| `scripts/homebrew_local_formula_proof.sh` | Local proof script validates approved license metadata, source archive contents, installed static package surface, downstream formula test behavior, cleanup, and retained non-claims. |
| `scripts/package_manager_deferral_check.sh` | Guard enforces Sprint 171 provider deferral, Sprint 198 local proof record, absence of unselected provider recipes, local Homebrew proof boundary, package metadata neutrality, and public non-claims. |
| `scripts/static_package_deferral_check.sh` | Guard enforces static-first package behavior and keeps shared-library packaging and dynamic ABI support deferred. |

## Current Package Surfaces

| Surface | Day 1 role |
| --- | --- |
| `packaging/homebrew/sparse-lu-ortho.rb.in` | Local proof formula template; not a committed provider formula. |
| `packaging/homebrew/README.md` | Homebrew local proof runbook and non-claim boundary. |
| `scripts/homebrew_local_formula_proof.sh` | Exact local proof command owner and cleanup owner. |
| `scripts/package_manager_deferral_check.sh` | Package-manager provider claim guard. |
| `scripts/static_package_deferral_check.sh` | Static package and shared-library/dynamic ABI deferral guard. |
| Root `LICENSE` | MIT metadata source used by local Homebrew proof. |
| README | User-facing support and package non-claim surface. |
| INSTALL | Install support matrix, package-manager deferral, and local proof status. |
| `docs/maintainer_guide.md` | Maintainer package proof, guard, and claim-boundary runbook. |
| `sparse.pc.in` and `cmake/SparseConfig.cmake.in` | Installed static package metadata that must stay provider-neutral unless a future support decision changes it. |

## Initial Claim Boundary

Sprint 207 starts from this bounded proof state:

- root MIT metadata exists;
- the local Homebrew proof can use `SPARSE_HOMEBREW_LICENSE=MIT`;
- the proof path is developer-mode and local;
- the formula is rendered into a temporary local tap;
- source archive, checksum, install, installed-surface validation, downstream
  `brew test`, uninstall, cleanup, and exit `0` are recorded in Sprint 198;
- installed package metadata is static archive oriented and provider-neutral;
- public and maintainer docs describe the proof as local static source formula
  evidence only.

Sprint 207 must not infer the following from that evidence:

- public Homebrew tap support;
- Homebrew/core readiness or acceptance;
- bottle support;
- Linuxbrew support;
- hosted binary package support;
- vcpkg, Conan, pkgsrc, distro/system package, or other package-manager
  support;
- broad package-manager distribution;
- shared-library package support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- package-manager platform parity;
- release readiness;
- state-of-the-art package ecosystem parity.

## Initial Validation Matrix

| Command | Purpose | Day 1 disposition |
| --- | --- | --- |
| `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT bash scripts/homebrew_local_formula_proof.sh` | Reproduce the local Homebrew static source formula proof when proof state is audited or changed. | Candidate command for Day 4 and later validation; not run on intake day. |
| `bash scripts/package_manager_deferral_check.sh` | Enforce package-manager non-claims and local Homebrew proof boundary. | Required after package guard or docs changes. |
| `bash scripts/static_package_deferral_check.sh` | Enforce static-first package contract and shared-library/dynamic ABI deferrals. | Required after package or install metadata changes. |
| `bash tests/test_install.sh` | Validate Make install/uninstall and downstream static package consumer behavior. | Required if install metadata changes. |
| `bash tests/test_cmake_install.sh` | Validate CMake install/downstream static package consumer behavior. | Required if CMake package metadata changes. |
| `make docs-check` | Validate documentation surfaces affected by package wording changes. | Candidate Day 12 validation. |
| `make format && make lint && make test` | Full C quality gate. | Required only if `.c` or `.h` files change. |
| `git diff --check` | Whitespace validation. | Required before closeout. |

## Initial Risk Register

| Risk | Day 1 mitigation |
| --- | --- |
| Local proof is mistaken for a user-facing Homebrew install path. | Keep proof-only wording and package non-claims in every decision artifact. |
| Provider promotion is selected without hosted or reproducible evidence. | Day 2 must define exact evidence needs before Day 5 provider decision. |
| Homebrew/core readiness is implied by a local formula template. | Treat the template as proof material until a later decision proves provider readiness. |
| Public tap, bottle, or Linuxbrew wording appears accidentally. | Keep package guard alignment as a required Day 9 task. |
| Static package metadata gains provider or shared-library wording. | Retain package metadata neutrality and static package guard checks. |
| Environment limitations are misclassified as project support. | Separate Homebrew tier, CLT/Xcode/compiler, and host platform evidence from provider claims. |

## Day 1 Completion Criteria

| Criterion | Status |
| --- | --- |
| Every Sprint 207 item has an initial evidence path or artifact category. | Complete in `WORKING_NOTES.md`. |
| Existing Homebrew and package metadata evidence is identified before edits. | Complete in this artifact and working notes. |
| Unsupported package, ABI, binary, platform, and release claims remain out of scope. | Complete via the initial claim-boundary and non-goal records. |

## Day 1 Outcome

Day 1 is complete. Sprint 207 starts from a narrow developer-mode local
Homebrew static source formula proof inherited from Sprint 198, with broad
package-manager distribution still unclaimed. The next sprint step is to
compare exact provider options before selecting a promotion or continued
deferral path.
