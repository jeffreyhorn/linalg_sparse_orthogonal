# Sprint 180 Day 8: Product Decision Record

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Status

Accepted for Sprint 180 implementation.

This record selects the provider proof path for the rest of Sprint 180. It
does not, by itself, make package-manager support a public claim. Public
support wording remains deferred until the selected prototype, proof script,
guard narrowing, documentation wording, and validation are complete.

## Decision

Sprint 180 selects **Homebrew local formula/tap proof** as the package-manager
provider path.

The selected proof level is:

- local formula or local tap proof only;
- source build only;
- static archive install only;
- macOS-local proof first;
- downstream compile/link/run from the provider-installed package;
- version, installed-file, uninstall, cleanup, and missing-tool behavior
  checks;
- no Homebrew/core, bottle, Linuxbrew, registry-ready, binary-package,
  provider-managed upgrade, shared-library, dynamic ABI, or broad
  package-manager support claim.

The selected path replaces the Day 7 recommendation with an authoritative Day
8 implementation decision. It does not remove the Sprint 171 public deferral
until later Sprint 180 implementation and validation gates pass.

## Accepted Evidence

| Evidence | Rationale |
| --- | --- |
| Homebrew has strong static-first fit. | The project already has Make and CMake install paths that install a static archive and reject shared-library support. |
| `brew` is locally available. | Day 4 found `/usr/local/bin/brew` and `Homebrew 6.0.19` on the current macOS host, reducing first-proof tooling uncertainty. |
| macOS static package proof already exists. | Existing macOS package lanes and local install tests prove the underlying static Make/CMake package surface that a formula can drive. |
| Formula proof can be bounded locally. | A local formula/tap proof can be scoped to source build, static install, formula `test do`, uninstall, and cleanup without claiming Homebrew/core or bottles. |
| vcpkg, Conan, and pkgsrc have higher first-proof uncertainty. | vcpkg lacks local tooling; Conan adds package ID/profile/generator complexity; pkgsrc lacks bootstrap/tooling and requires high package-policy proof. |

## Rejected Alternatives

| Alternative | Decision | Reason |
| --- | --- | --- |
| vcpkg local overlay proof | Rejected as first Sprint 180 provider path | Strong runner-up, but no local `vcpkg` executable is available. The proof would need bootstrap/setup policy before runtime cost and missing-tool behavior are known. Registry, binary, triplet linkage, Windows breadth, source/checksum, and license claim risks remain unresolved. |
| Conan local recipe proof | Rejected as first Sprint 180 provider path | A credible first proof must model package IDs, settings/options, profiles, generators, `package_info()`, cache isolation, and `test_package`; no local `conan` tool exists. |
| pkgsrc local package skeleton proof | Rejected as first Sprint 180 provider path | No local pkgsrc tooling or package skeleton exists, and proof requires bootstrap, `distinfo`, `PLIST`, package database queries, buildlink/options policy, platform boundaries, and cleanup evidence. |
| Renewed formal package-manager deferral | Rejected for Day 8, retained as stop condition | Homebrew has enough local proofability to justify implementation. Deferral must still be used if Days 9-13 cannot keep source/checksum, license, guard, proof, or wording boundaries narrow. |
| Homebrew/core or bottle support | Rejected | Core submission and bottles require release source, SHA-256, license metadata, audit/review/bottle evidence, and hosted binary-package policy not present in the repository. |
| Linuxbrew or broad Homebrew support | Rejected | Day 4 evidence is macOS-local. Linuxbrew and broad platform claims need separate platform proof. |

## Support-Tier Wording Boundary

After this decision, internal Sprint 180 planning artifacts may say:

- Sprint 180 selected Homebrew local formula/tap proof as the implementation
  path.
- The selected path is a local proof target, not public package-manager
  support.
- Homebrew/core, bottles, Linuxbrew, provider-hosted binary packages, registry
  readiness, and broad package-manager support remain unsupported.

Public docs may not claim Homebrew support until the Days 9-13 implementation
and validation work completes. Until then, README, INSTALL, maintainer guide,
package metadata, and workflows must preserve the Sprint 171 public deferral
posture.

If later Sprint 180 work succeeds, allowed public wording must stay limited to
the proven support level:

- local Homebrew formula/tap proof for source builds;
- static archive package only;
- exact proof command and host scope;
- no Homebrew/core, bottle, Linuxbrew, binary-package, registry, shared-library,
  dynamic ABI, or broad package-manager claim.

## Unsupported Claims

Sprint 180 must still reject wording or artifacts that imply:

- Homebrew/core acceptance;
- bottle build or published bottle availability;
- Linuxbrew support;
- general Homebrew support beyond the selected local proof;
- vcpkg, Conan, pkgsrc, Debian, Fedora, or system-package support;
- provider registry readiness or upstream acceptance;
- provider-hosted binary packages;
- provider-managed upgrades or dependency compatibility;
- shared-library builds, installs, or static/shared selectors;
- dynamic ABI compatibility, runtime-loader behavior, install-name/RPATH,
  SONAME, DLL/import-library, or broad platform parity;
- state-of-the-art status from package-manager evidence.

## Implementation Boundaries

Days 9-13 must keep the selected Homebrew path inside these boundaries:

| Area | Boundary |
| --- | --- |
| Provider scope | Only Homebrew local formula/tap proof may be implemented. Other providers remain guarded. |
| Artifact scope | No formula or tap artifact may be added until Day 9 designs its source-controlled location and guard treatment. |
| Build scope | Source build only through the existing static Make or CMake package surface. |
| Library scope | Static archive only; no shared libraries, runtime-loader behavior, or static/shared selector. |
| Platform scope | macOS-local proof only unless additional platform evidence is explicitly added later. |
| Source input | Local source archive or release archive policy must be explicit and must not imply distribution readiness. |
| License metadata | Formula metadata must not overstate license readiness. If accurate local formula metadata cannot be represented, stop and renew deferral. |
| Proof scope | Proof must cover tool availability, install/build, downstream compile/link/run, version or installed metadata query, static-only installed files, uninstall, cleanup, and missing-tool behavior. |
| Docs scope | Public docs must distinguish local proof from Homebrew/core, bottles, Linuxbrew, and broad package-manager support. |
| Guard scope | Guard must allow only the selected Homebrew proof artifacts in approved locations and keep every other provider fail-closed. |

## Stop Conditions

The selected Homebrew path must stop and become renewed formal deferral if any
of these conditions cannot be satisfied:

| Stop condition | Required response |
| --- | --- |
| No safe formula source input | Renew deferral rather than use ambiguous source/checksum behavior. |
| No safe license metadata | Renew deferral rather than add inaccurate provider metadata. |
| No narrow formula location | Renew deferral rather than permit broad `Formula/` or tap artifacts without guard ownership. |
| No bounded proof script | Renew deferral rather than add unverified formula files. |
| No static-only proof | Renew deferral rather than imply shared-library or dynamic ABI support. |
| No safe docs wording | Renew deferral rather than publish ambiguous Homebrew or package-manager claims. |
| No fail-closed guard update | Renew deferral rather than weaken provider-artifact protection. |

## Revisit Criteria

A future decision may broaden beyond local Homebrew proof only after separate
evidence exists:

| Future claim | Required evidence |
| --- | --- |
| Homebrew/core readiness | Immutable release source, SHA-256, accurate SPDX-compatible license metadata, accepted formula shape, audit/style evidence, and explicit no-bottle or bottle policy. |
| Bottle support | `brew bottle` or equivalent bottle build/publish evidence, bottle metadata, cleanup policy, and hosted-binary wording. |
| Linuxbrew support | Linuxbrew-specific build/install/test proof and platform-boundary docs. |
| vcpkg support | Selected vcpkg decision, local or hosted vcpkg tool proof, overlay/registry artifact, downstream proof, guard update, and docs wording. |
| Conan support | Selected Conan decision, recipe, package ID/profile policy, `test_package`, cache isolation, downstream proof, guard update, and docs wording. |
| pkgsrc support | Selected pkgsrc decision, bootstrap/tool proof, package skeleton, `distinfo`, `PLIST`, package query/downstream proof, guard update, and docs wording. |
| Broad package-manager support | Multiple provider decisions, provider-specific proof and guards, docs claim boundaries, and integrated validation. |

## Validation Gates For Days 9-14

The selected path passes Sprint 180 only if later work provides all selected
gates:

| Gate | Required evidence |
| --- | --- |
| Product decision | This record remains the authoritative selected provider decision. |
| Artifact design | Day 9 defines formula/tap artifact location, generated-output policy, and guard ownership. |
| Artifact implementation | Day 10 adds only the selected Homebrew prototype material or renews deferral if a stop condition is hit. |
| Proof script | Days 11-12 add and validate local proof behavior, including missing-tool handling. |
| Guard update | Day 13 narrows `scripts/package_manager_deferral_check.sh` to permit only selected Homebrew artifacts and reject other providers. |
| Docs update | Day 13 updates README, INSTALL, maintainer guide, and metadata wording only to the proven support level. |
| Static-first guard | `bash scripts/static_package_deferral_check.sh` continues to pass. |
| Package-manager guard | Updated or still-deferral `bash scripts/package_manager_deferral_check.sh` passes. |
| Install checks | Make/CMake install checks run if package metadata or install behavior changes. |
| Whitespace | `git diff --check` passes. |

## Pass/Fail Contract

The selected product decision passes when:

- exactly one provider path is selected: Homebrew local formula/tap proof;
- selected artifacts, proof scripts, docs, and guards match the local-only
  Homebrew proof boundary;
- unsupported providers remain absent or fail-closed;
- public wording does not imply Homebrew/core, bottles, Linuxbrew, registry
  readiness, binary packages, shared-library support, dynamic ABI support, or
  broad package-manager support;
- the proof script validates install, downstream compile/link/run, version or
  metadata behavior, uninstall, cleanup, and missing-tool handling.

The selected product decision fails if:

- more than one provider is selected;
- provider files appear before guard ownership is designed and implemented;
- Homebrew wording appears without proof;
- a local formula is treated as Homebrew/core, bottle, Linuxbrew, or registry
  evidence;
- static-first package metadata weakens or implies shared-library behavior;
- source/checksum or license metadata is inaccurate or ambiguous;
- unsupported provider artifacts are allowed by the guard.

## Day 8 Deliverables

- provider product decision record
- selected Homebrew local formula/tap proof path
- blocker and revisit-criteria list
- support-tier wording boundaries
- implementation and validation gate list
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day8-product-decision-record.md`

## Validation

Day 8 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, provider formula, or public user-facing docs were modified.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Exactly one provider decision is recorded. | Complete | Homebrew local formula/tap proof is selected above. |
| Rejected options have concrete rationale. | Complete | Rejected alternatives and stop conditions are listed above. |
| Downstream implementation work has clear boundaries. | Complete | Implementation boundaries, validation gates, and pass/fail contract are listed above. |
