# Sprint 207 Day 2: Provider Scope Options

## Purpose

Compare exact package provider paths and define the evidence required before
Sprint 207 can promote a user-facing package distribution claim or deliberately
close the residual with stronger deferral guards.

## Starting Constraint

The repository currently has proof-local Homebrew material only:

- `packaging/homebrew/sparse-lu-ortho.rb.in` is a template rendered into a
  temporary local formula.
- `packaging/homebrew/README.md` says the directory is proof material, not a
  Homebrew/core formula, tap, bottle, Linuxbrew claim, or general
  package-manager support.
- `scripts/homebrew_local_formula_proof.sh` proves a developer-mode local
  static source formula path.
- `scripts/package_manager_deferral_check.sh` rejects unselected provider
  recipe artifacts and broad package claims.
- `scripts/static_package_deferral_check.sh` keeps shared-library packaging
  and dynamic ABI support deferred.

Sprint 207 therefore cannot start by assuming a public package provider is
already supported.

## Option 1: Public Homebrew Tap / Source Formula

### Description

Promote a bounded user-facing Homebrew tap/source formula tier. This would be
stronger than Sprint 198 local proof but narrower than Homebrew/core,
bottles, Linuxbrew, or broad package-manager support.

### Required Evidence

- Source-controlled public tap formula or documented public tap ownership
  model.
- Stable source archive URL or release artifact suitable for formula use.
- SHA-256 provenance for the selected archive.
- MIT license metadata carried into the formula.
- Formula render/audit/install/test/uninstall proof on a supported macOS
  environment or explicitly documented local environment.
- Installed static package surface validation: archive, headers, CMake package
  files, and `sparse.pc`.
- Cleanup behavior for taps, source archives, logs, Cellar entries, and
  temporary files.
- README, INSTALL, packaging, and maintainer wording that names only the
  earned public tap/source formula tier.
- Guard coverage retaining non-claims for Homebrew/core, bottles, Linuxbrew,
  binary packages, shared-library package support, dynamic ABI support, and
  all other package managers.

### Benefits

- Directly addresses the Epic 19 package distribution gap.
- Builds on Sprint 198 proof instead of starting from an unrelated provider.
- Provides a practical user-facing install story if proof can be made
  reproducible.
- Keeps Homebrew/core and bottle support separate.

### Risks

- Current formula template is explicitly local-proof material and uses local
  archive/tap behavior.
- User-facing tap support may require release/archive discipline not yet
  implemented.
- A public tap can be overread as Homebrew/core readiness unless docs and
  guards are precise.
- macOS Intel Tier 3 evidence from Sprint 198 may not be sufficient for a
  supported provider claim.

### Validation Requirements

| Validation | Required before promotion |
| --- | --- |
| Formula render/lint/audit | Yes, for the selected tap/source formula form. |
| Source archive checksum proof | Yes, using a stable archive source or explicitly scoped proof archive. |
| Install/test/uninstall proof | Yes, including downstream CMake consumer proof. |
| Cleanup proof | Yes, including generated taps, archives, logs, and Cellar state. |
| Package guards | Yes, with retained Homebrew/core/bottle/Linuxbrew and broad package non-claims. |
| Install tests | Required if install metadata changes. |
| Docs checks | Required after public wording changes. |

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| User value | 5 | Provides a concrete package path users can understand. |
| Evidence cost | 3 | Reuses Sprint 198 proof but needs public-provider archive and ownership evidence. |
| Maintenance burden | 3 | Requires formula ownership and update discipline. |
| Platform risk | 3 | Homebrew platform support tier and runner availability must be handled carefully. |
| Claim risk | 3 | Manageable if Homebrew/core, bottles, Linuxbrew, and broad support remain guarded. |
| Review surface | 3 | Formula, proof script, docs, and guards all need review. |

## Option 2: Homebrew/core Readiness

### Description

Promote readiness for Homebrew/core formula submission. This is stronger than
a project-owned tap/source formula and requires stricter release, audit,
metadata, and maintenance evidence.

### Required Evidence

- All public tap/source formula evidence.
- Stable upstream release archive and checksum policy.
- Formula structure compatible with Homebrew/core conventions.
- No local developer-mode dependency or temporary local tap assumption.
- Formula audit readiness and test block compatibility.
- Versioning and release process that can support provider updates.
- Maintainer ownership for upstream submission and follow-up.
- Documentation that distinguishes Homebrew/core readiness from acceptance,
  bottles, Linuxbrew, and binary package support.

### Benefits

- Highest user value among the considered Homebrew provider paths if accepted
  later.
- Forces release/archive discipline that would improve downstream packaging
  readiness.

### Risks

- The repository currently lacks a Homebrew/core formula artifact.
- Sprint 198 proof is developer-mode/local and not Homebrew/core evidence.
- Homebrew/core readiness can be overclaimed as Homebrew/core acceptance.
- Formula audit and upstream submission obligations may exceed Sprint 207's
  166-hour package decision scope.
- Bottles and Linuxbrew could be accidentally implied unless guarded.

### Validation Requirements

| Validation | Required before readiness claim |
| --- | --- |
| Stable release archive checksum proof | Required. |
| Homebrew/core-style formula audit | Required. |
| Non-developer-mode source install/test proof | Required. |
| Public docs and maintainer runbook | Required. |
| Homebrew/core acceptance wording guard | Required. |
| Bottle/Linuxbrew non-claim guard | Required. |

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| User value | 5 | Strongest potential provider signal. |
| Evidence cost | 5 | Requires release, archive, audit, and upstream-readiness work beyond local proof. |
| Maintenance burden | 5 | Requires ongoing update and submission discipline. |
| Platform risk | 4 | Must avoid overclaiming bottles, Linuxbrew, or unsupported Homebrew tiers. |
| Claim risk | 5 | Easy to confuse readiness, submission, acceptance, and bottle support. |
| Review surface | 5 | Touches release, packaging, docs, guards, and provider policy. |

## Option 3: Continued Deferral With Stronger Guards

### Description

Decide that Sprint 207 should not promote a user-facing package provider path.
Instead, close the residual by making deferral stronger, clearer, and more
enforceable.

### Required Evidence

- Updated package-manager guard coverage for public tap, Homebrew/core,
  bottles, Linuxbrew, vcpkg, Conan, pkgsrc, distro/system packages, binary
  packages, shared-library package support, and broad provider wording.
- Updated public docs that say no user-facing package-manager install path is
  claimed.
- Maintainer documentation explaining what evidence is required to reopen
  each provider tier.
- Residual queue entry listing exact future proof needs for public tap,
  Homebrew/core, bottles, Linuxbrew, and other providers.
- Validation showing package guard, static package guard, docs checks, and
  install checks still pass for the deferred state.

### Benefits

- Lowest claim risk.
- Prevents the existing local proof from being overstated.
- Can fully close the ambiguity around provider status even without promotion.
- Keeps package support truthful if release/archive/provider evidence is not
  ready.

### Risks

- Does not add a new user-facing install path.
- May leave the main package distribution gap functionally unresolved from a
  user's perspective.
- Requires clear residual wording so future sprints know exactly what
  evidence is missing.

### Validation Requirements

| Validation | Required before deferral closeout |
| --- | --- |
| `bash scripts/package_manager_deferral_check.sh` | Required. |
| `bash scripts/static_package_deferral_check.sh` | Required. |
| Provider recipe absence scan | Required through package guard or explicit check. |
| README/INSTALL/maintainer non-claim checks | Required. |
| `make docs-check` | Required if docs change. |
| Install tests | Required only if install/package metadata changes. |

### Day 2 Score

| Dimension | Score | Rationale |
| --- | ---: | --- |
| User value | 2 | Clarifies support but does not add an install path. |
| Evidence cost | 2 | Builds on existing guards and docs. |
| Maintenance burden | 1 | Low ongoing burden. |
| Platform risk | 1 | Does not claim provider platform support. |
| Claim risk | 1 | Strongest protection against overclaiming. |
| Review surface | 2 | Mostly guards and docs. |

## Comparative Summary

| Option | Promotion strength | Main blocker | Preliminary ranking |
| --- | --- | --- | --- |
| Public Homebrew tap/source formula | Bounded user-facing package tier | Needs stable public-provider ownership and proof beyond temporary local formula behavior. | First candidate if audit/proof can close evidence gaps safely. |
| Homebrew/core readiness | Strong provider-readiness tier | Needs release archive discipline, audit readiness, non-developer-mode proof, and submission/maintenance owner. | Not preferred for Sprint 207 unless Day 3-5 evidence is unexpectedly strong. |
| Continued deferral | No provider promotion | Does not add user-facing package support. | Safe fallback if promotion evidence remains insufficient. |

## Rejected-Path Risks To Carry Forward

| Risk | Applies to | Required mitigation |
| --- | --- | --- |
| Local proof is presented as public tap support. | Public tap/source formula | Separate temporary local proof from public tap formula evidence. |
| Readiness is confused with Homebrew/core acceptance. | Homebrew/core readiness | Use exact readiness wording and guard against acceptance/bottle claims. |
| Bottle or Linuxbrew wording appears without evidence. | Public tap/source formula, Homebrew/core readiness | Keep explicit non-claims and regression coverage. |
| Other package managers are implied by one Homebrew path. | All options | Guard vcpkg, Conan, pkgsrc, distro/system package, and binary package wording. |
| Static package support is confused with shared-library package support. | All options | Keep static-package guard and dynamic ABI non-claims. |
| Provider evidence depends on unsupported Homebrew tier. | Public tap/source formula, Homebrew/core readiness | Record environment tier and classify blockers before claim promotion. |

## Day 2 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.1 has concrete provider options before a decision is made. | Complete. |
| Every option has explicit validation requirements. | Complete. |
| Unsupported package claims cannot be inferred from undecided options. | Complete; all options retain non-claims until Day 5 decision and proof evidence. |

## Day 2 Outcome

Day 2 does not select the provider path. It establishes that the public
Homebrew tap/source formula path is the leading promotion candidate if later
audit and proof work can close the evidence gaps, while continued deferral is
the safe fallback. Homebrew/core readiness is treated as a high-risk path that
requires evidence beyond the current repository state.
