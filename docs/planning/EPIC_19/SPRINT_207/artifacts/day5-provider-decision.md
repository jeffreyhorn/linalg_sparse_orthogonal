# Sprint 207 Day 5: Provider Decision

## Purpose

Select exactly one Sprint 207 package distribution path or close the residual
with explicit continued deferral before implementing proof, guard, and
documentation changes.

## Decision

Sprint 207 selects **explicit continued deferral with stronger guards**.

This means Sprint 207 will not promote a public Homebrew tap/source formula,
Homebrew/core readiness, bottles, Linuxbrew, binary packages, other package
managers, shared-library package support, or broad package-manager
distribution. The sprint will instead make the current state harder to
overstate: developer-mode local Homebrew static source formula proof exists,
and user-facing package-manager distribution remains unclaimed.

## Evidence Applied

| Evidence source | Decision impact |
| --- | --- |
| Day 2 provider option matrix | Public tap/source formula remained the leading promotion candidate only if Day 3-5 evidence could prove stable provider ownership and non-local support. Continued deferral was the safe fallback. |
| Day 3 formula baseline | The current formula is named and scoped as local proof, uses placeholder rendering, points at a temporary `file://` archive, injects a local CMake bindir, and lacks public formula ownership or stable archive provenance. |
| Day 4 proof baseline | The local proof passed with `HOMEBREW_DEVELOPER=1 SPARSE_HOMEBREW_LICENSE=MIT`, but the evidence remains developer-mode local static source formula proof on an Intel macOS Homebrew host. |
| Current docs and guards | README, INSTALL, Homebrew README, maintainer guide, package-manager guard, and static package guard already preserve broad package non-claims. |

## Selected Path: Continued Deferral With Stronger Guards

### Why This Path Is Selected

- It is the only option fully supported by current evidence.
- It closes the ambiguity between local Homebrew proof and user-facing package
  distribution.
- It avoids overstating a public provider claim before stable archive,
  checksum, public formula ownership, and non-local formula proof exist.
- It preserves Sprint 198's useful local static proof while making unsupported
  provider paths explicit residuals.
- It is implementable within Sprint 207 without inventing release or provider
  evidence.

### Implementation Scope

Allowed changes after this decision:

- strengthen `scripts/package_manager_deferral_check.sh`;
- strengthen `scripts/static_package_deferral_check.sh` if package non-claim
  wording needs alignment;
- update README, INSTALL, `packaging/homebrew/README.md`, and
  `docs/maintainer_guide.md` with exact package status;
- add or update Sprint 207 residual and closeout artifacts;
- add focused regression coverage for forbidden provider wording;
- rerun local Homebrew proof, package guard, static package guard, docs checks,
  install checks when relevant, and C gate only if `.c` or `.h` files change.

Disallowed changes after this decision:

- publishing or documenting a user-facing Homebrew tap install path;
- claiming Homebrew/core readiness;
- claiming Homebrew/core acceptance;
- claiming bottles or Linuxbrew;
- adding vcpkg, Conan, pkgsrc, distro/system package, or binary package
  support claims;
- claiming shared-library packaging or dynamic ABI compatibility;
- claiming release readiness from local package proof.

## Rejected Path: Public Homebrew Tap / Source Formula

### Reason Rejected For Sprint 207

The current repository does not yet have the evidence needed to promote a
public tap/source formula:

- no source-controlled public tap formula;
- no documented public tap ownership model;
- no stable public source archive URL;
- no public checksum provenance;
- current formula class/name includes `Local`;
- current proof renders a temporary formula under a temporary local tap;
- current proof uses a temporary `file://` archive;
- current proof injects the local CMake bindir;
- current host proof is developer-mode local evidence, not provider support.

### Residual Evidence Required To Reopen

- Decide public tap ownership and repository layout.
- Add a provider formula separate from the local proof template, or explicitly
  document how one template safely serves both without overclaiming.
- Define stable source archive and SHA-256 provenance.
- Prove formula render/audit/install/test/uninstall on a supported macOS
  environment without relying on temporary local proof semantics.
- Keep Homebrew/core, bottles, Linuxbrew, binary packages, shared-library
  package support, dynamic ABI support, and other package managers as
  explicit non-claims.
- Update public and maintainer docs only after proof passes.

## Rejected Path: Homebrew/core Readiness

### Reason Rejected For Sprint 207

Homebrew/core readiness requires evidence beyond both Sprint 198 and Day 4:

- no Homebrew/core formula exists;
- no Homebrew/core audit-readiness artifact exists;
- no stable upstream release archive discipline is recorded;
- no upstream submission or maintenance owner is documented;
- current proof is developer-mode and local;
- current proof uses a temporary local tap and local archive;
- no bottle or Linuxbrew evidence exists;
- no public wording separates readiness from acceptance strongly enough yet.

### Residual Evidence Required To Reopen

- Complete all public tap/source formula residuals first.
- Add release/archive discipline suitable for provider review.
- Add Homebrew/core-style formula audit evidence.
- Document submission owner and maintenance expectations.
- Guard wording so readiness cannot be read as Homebrew/core acceptance,
  bottles, Linuxbrew, or binary package support.

## Claim Boundary After Decision

Earned and retained:

- root MIT license metadata;
- developer-mode local Homebrew static source formula proof;
- local source archive and checksum proof;
- temporary local tap render proof;
- source install proof;
- installed static archive/header/CMake/pkg-config surface validation;
- downstream `brew test`;
- uninstall and cleanup proof.

Still unclaimed:

- user-facing Homebrew install path;
- public Homebrew tap support;
- Homebrew/core readiness or acceptance;
- bottle support;
- Linuxbrew support;
- hosted binary packages;
- vcpkg, Conan, pkgsrc, distro/system package, or other package-manager
  support;
- broad package-manager distribution;
- shared-library package support;
- dynamic ABI compatibility;
- release readiness;
- package ecosystem state-of-the-art parity.

## Days 6-14 Direction

| Day range | Direction after Day 5 |
| --- | --- |
| Day 6 | Design the stronger deferral implementation: guard coverage, diagnostics, docs wording, and validation commands. |
| Days 7-8 | Implement guard and proof-boundary hardening for continued deferral, not provider promotion. |
| Day 9 | Align package-manager and static-package guards with the selected deferral tier. |
| Days 10-11 | Update public and maintainer docs to make the no-user-facing-package path explicit. |
| Days 12-14 | Validate proof, guards, docs, residuals, and closeout evidence. |

## Day 5 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.1 is complete with one selected decision path. | Complete; continued deferral with stronger guards is selected. |
| Unselected package paths remain non-claims with documented rationale. | Complete; public tap/source formula and Homebrew/core readiness residuals are recorded. |
| Implementation can proceed without ambiguity about provider scope. | Complete; Days 6-14 are scoped to stronger deferral, not provider promotion. |

## Day 5 Outcome

Sprint 207 will close the package distribution residual by strengthening the
deferral boundary instead of promoting a package provider. The current local
Homebrew proof remains valuable evidence, but it is not sufficient for a
public tap, Homebrew/core, bottle, Linuxbrew, binary package, shared-library,
ABI, or broad package-manager support claim.
