# Sprint 180 Day 7: Provider Decision Matrix

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Compare vcpkg, Homebrew, Conan, and pkgsrc using the shared Sprint 180
provider criteria. Day 7 selects a recommendation for the Day 8 product
decision record, records rejected-provider rationale, preserves the stronger
deferral fallback, and keeps all package-manager support unclaimed until the
Day 8 decision and later proof work.

## Inputs

| Input | Role |
| --- | --- |
| `docs/planning/EPIC_16/SPRINT_180/artifacts/day1-provider-decision-intake.md` | Shared provider criteria and acceptance-gate baseline. |
| `docs/planning/EPIC_16/SPRINT_180/artifacts/day2-package-surface-audit.md` | Current package surface, proof owners, and claim inventory. |
| `docs/planning/EPIC_16/SPRINT_180/artifacts/day3-vcpkg-feasibility.md` | vcpkg feasibility evidence. |
| `docs/planning/EPIC_16/SPRINT_180/artifacts/day4-homebrew-feasibility.md` | Homebrew feasibility evidence. |
| `docs/planning/EPIC_16/SPRINT_180/artifacts/day5-conan-feasibility.md` | Conan feasibility evidence. |
| `docs/planning/EPIC_16/SPRINT_180/artifacts/day6-pkgsrc-feasibility.md` | pkgsrc feasibility evidence. |
| `scripts/package_manager_deferral_check.sh` | Current fail-closed guard for unselected provider artifacts and public non-claims. |
| `scripts/static_package_deferral_check.sh` | Static-first package contract guard. |

## Decision Criteria

Day 7 uses the Day 1 criteria:

| Criterion | Meaning for Day 7 |
| --- | --- |
| Static-first fit | Candidate must consume the maintained static archive install/export surface without implying shared-library, dynamic ABI, runtime-loader, or static/shared selector support. |
| CI feasibility | Candidate proof must be runnable or claim-safely unavailable with bounded setup, clear prerequisites, and deterministic cleanup. |
| Recipe complexity | Candidate should minimize provider-specific files, policy decisions, patches, update steps, and hidden package-manager semantics. |
| User value | Candidate should reduce adoption friction for likely users without creating broad provider or platform expectations. |
| Proof completeness | Candidate should be able to prove install, downstream compile/link/run, version behavior, cleanup, and failure behavior inside Sprint 180. |
| Maintenance cost | Candidate should have manageable ongoing version, checksum, source archive, dependency, registry, tap, or policy upkeep. |
| Claim risk | Candidate must avoid unearned package-manager, binary-package, upgrade, shared-library, ABI, platform, registry, tap, or upstream-acceptance claims. |

## Four-Provider Matrix

| Provider | Static-first fit | CI feasibility | Complexity | User value | Proof completeness | Maintenance cost | Claim risk | Day 7 position |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Homebrew | Strong for local formula using existing CMake or Make static install path. | Medium to high because `brew` is available locally and macOS package proof already exists. | Medium for a local formula/tap; high only for core or bottle support. | High for macOS users. | Medium; underlying install proof exists, formula/test/audit/uninstall proof is missing. | Medium for local tap, high for Homebrew/core or bottles. | High unless scoped away from core, bottles, Linuxbrew, shared libraries, and package-manager generalization. | Recommended first proof candidate. |
| vcpkg | Strong with existing CMake static target and possible static-linkage enforcement. | Medium; hosted setup is plausible, but no local `vcpkg` tool is available. | Medium for local overlay, high for registry readiness. | High for C/C++ and Windows users. | Medium; underlying CMake proof exists, provider install/downstream/version/cleanup proof is missing. | Medium to high depending on overlay versus registry scope. | High unless scoped away from registry, binaries, triplet shared linkage, and broad Windows claims. | Strong runner-up. |
| Conan | Good with CMake static package shape and `package_type = "static-library"` policy. | Medium; no local `conan`, and profile/cache behavior needs bounded setup. | Medium to high because recipe, settings/options, generators, `package_info()`, and `test_package` need policy. | Medium to high for Conan users and cross-platform C/C++ consumers. | Medium; no recipe, package ID evidence, profile proof, cache isolation, or test package exists. | High due to package IDs, profile compatibility, binary model, remotes, and revisions. | High because a recipe can imply remotes, binaries, profile compatibility, and dependency policy. | Defer as first provider. |
| pkgsrc | Conditional; can package static libraries but needs careful package and platform boundaries. | Low to medium; no local `bmake`, `pkg_info`, or pkgsrc tree is available. | High due to package skeleton, `distinfo`, `PLIST`, buildlink, patches, bootstrap, lint/build/install proof. | Medium for NetBSD, SmartOS, and cross-Unix users, but less aligned with current proof surface. | Low today; no package skeleton, package build/install, package query, downstream proof, or cleanup exists. | High due to checksums, PLIST drift, bootstrap variation, platform fixes, and buildlink policy. | High because skeletons can imply pkgsrc-current/pkgsrc-wip, binaries, broad Unix support, and package database behavior. | Reject as first provider. |

## Ranking

| Rank | Candidate | Why |
| --- | --- | --- |
| 1 | Homebrew local formula/tap proof | Best immediate proofability: local tool available, macOS package checks already exist, static CMake/Make install path fits, and the first proof can be scoped to local source formula behavior. |
| 2 | vcpkg local overlay proof | Best broader C/C++ and Windows user value, strong static CMake fit, and manageable overlay shape, but blocked by absent local tool and source/checksum/license policy. |
| 3 | Conan local recipe proof | Useful and technically viable, but too much first-provider complexity around package IDs, profiles, generators, cache isolation, and `test_package`. |
| 4 | pkgsrc local package skeleton proof | Technically possible, but weakest Sprint 180 first-provider choice due to bootstrap/tooling absence and high package-policy/platform proof burden. |

## Recommended Day 8 Decision Candidate

Day 7 recommends selecting **Homebrew local formula/tap proof** on Day 8.

The recommended scope is narrow:

- local formula or local tap proof only;
- source build only;
- static archive install only;
- downstream compile/link/run test through the installed package;
- version and installed-file checks;
- uninstall and temporary-artifact cleanup;
- claim-safe failure if `brew` is unavailable;
- documentation and guard wording that explicitly excludes Homebrew/core,
  bottles, Linuxbrew, registry readiness, binary packages, broad
  package-manager support, shared libraries, and dynamic ABI support.

This recommendation does not claim Homebrew support yet. It only identifies
the strongest candidate for the Day 8 product decision record and the Days
9-13 prototype, proof, guard, and docs work.

## Strongest Renewed Deferral Case

The strongest renewed deferral case is also Homebrew-shaped: if Day 8 cannot
scope a local formula/tap proof without implying distribution readiness, the
sprint should renew the formal package-manager deferral rather than choose a
weaker provider.

Exact blockers that would justify renewed deferral:

- no acceptable source archive and SHA-256 policy for formula input;
- no acceptable standalone license or copyright source for formula metadata;
- no acceptable wording that separates local formula proof from Homebrew/core,
  bottles, Linuxbrew, and package-manager support;
- no acceptable proof-script behavior for install, downstream compile/link/run,
  version checks, uninstall, cleanup, and missing-tool failure;
- no acceptable guard update that permits only the selected prototype while
  keeping all other provider artifacts fail-closed.

## Rejected-Provider Rationale

| Provider | Day 7 rationale |
| --- | --- |
| vcpkg | Strong runner-up, but not the first recommendation because no local `vcpkg` executable is present. The proof would need bootstrap/setup policy before runtime cost and missing-tool behavior are known. It also carries registry, binary, triplet linkage, Windows breadth, source/checksum, and license claim risks. |
| Conan | Deferred as first provider because the first proof must model package IDs, settings/options, profiles, generators, `package_info()`, cache isolation, and `test_package` behavior before making even a local recipe claim. No local `conan` tool exists. |
| pkgsrc | Rejected as first provider because no local pkgsrc tooling or package skeleton exists and credible proof requires bootstrap, `distinfo`, `PLIST`, package database queries, buildlink/options policy, platform boundaries, and cleanup evidence. |

## Unresolved Questions For Day 8

| Question | Required Day 8 answer |
| --- | --- |
| Product decision | Select Homebrew local formula/tap proof or renew formal deferral. |
| Source input | Decide whether a local proof may use a generated local source archive, or whether missing immutable release URL/SHA-256 policy forces deferral. |
| License metadata | Decide whether the current README purpose statement is enough for a local prototype, or whether lack of standalone license blocks provider material. |
| Formula location | If selected, choose an allowed source-controlled location that the guard can permit narrowly. |
| Proof boundary | Define the exact install, downstream compile/link/run, version, uninstall, cleanup, audit, and missing-tool checks required. |
| Support wording | Define the exact allowed wording for README, INSTALL, maintainer guide, and the product decision record. |
| Guard behavior | Define how `scripts/package_manager_deferral_check.sh` changes from blanket deferral to selected-provider-only allowance while keeping other providers fail-closed. |

## Day 7 Decision

Homebrew local formula/tap proof is the recommended Day 8 product decision
candidate. vcpkg remains the strongest rejected alternative. Conan and pkgsrc
should not be selected as the first provider path for Sprint 180 unless Day 8
intentionally rejects proofability as the primary criterion.

Stronger formal deferral remains valid if Day 8 determines that source,
checksum, license, guard, or wording blockers make even local Homebrew proof
too risky.

## Day 7 Deliverables

- four-provider comparison matrix
- recommended provider proof or deferral candidate
- rejected-provider rationale
- unresolved question list
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day7-provider-decision-matrix.md`

## Validation

Day 7 changed planning artifacts only. No `.c`, `.h`, package metadata,
workflow, guard, provider recipe/formula/skeleton, or public user-facing docs
were modified.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every provider candidate is compared on the same criteria. | Complete | Four-provider matrix above. |
| Recommendation is evidence-backed rather than preference-based. | Complete | Ranking, recommended Day 8 candidate, and rejected-provider rationale above. |
| Open questions are narrow enough for final decision work. | Complete | Day 8 unresolved-question table above. |
