# Sprint 180 Day 1: Provider Decision Intake

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Establish the Sprint 180 baseline before choosing a package-manager provider
proof path or renewing formal deferral. Day 1 records the source-plan
authority, inherited Sprint 171 deferral, Sprint 177 acceptance gate, artifact
layout, provider evaluation criteria, and protected non-claims.

## Source Authority

The active Sprint 180 project-plan section is:

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- section: `Sprint 180: Package-Manager Provider Decision`

The sprint artifact path is:

- `docs/planning/EPIC_16/SPRINT_180/`

## Starting Snapshot

| Field | Value |
| --- | --- |
| Branch | `sprint-180` |
| Starting commit | `88884f9960a895d7782d6d48fd369b38e7eeea9d` |
| Source project plan | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Sprint plan path | `docs/planning/EPIC_16/SPRINT_180/PLAN.md` |
| Working notes path | `docs/planning/EPIC_16/SPRINT_180/WORKING_NOTES.md` |
| Artifact directory | `docs/planning/EPIC_16/SPRINT_180/artifacts/` |

## Recent Prior PR Context

| Commit | Context |
| --- | --- |
| `88884f99` | Merged PR #199 from Sprint 179. |
| `188747b5` | Completed Sprint 179 generated API decision. |
| `17754f05` | Merged PR #198 from Sprint 178. |
| `a7d58196` | Completed Sprint 178 allocation-failure proof. |
| `3907e754` | Merged PR #197 from Sprint 177. |

## Sprint 180 Scope

Sprint 180 must close package-manager provider status by selecting and
enforcing exactly one of these outcomes:

| Outcome | Day 1 status |
| --- | --- |
| vcpkg provider proof | Open for feasibility audit and decision. |
| Homebrew provider proof | Open for feasibility audit and decision. |
| Conan provider proof | Open for feasibility audit and decision. |
| pkgsrc provider proof | Open for feasibility audit and decision. |
| Stronger formal deferral | Open for decision if provider proof cost, policy cost, or claim risk is too high. |

Sprint 180 must not broaden into full package-manager ecosystem support,
binary-package distribution, provider upgrade behavior, registry acceptance, or
release provenance.

## Inherited Sprint 171 Deferral Baseline

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_15/SPRINT_171/artifacts/day5-package-manager-deferral.md` | Package-manager support is formally deferred. No vcpkg, Homebrew, Conan, pkgsrc, Debian, Fedora, system-package, registry, tap, or binary-package claim is promoted. |
| Maintained package scope | The supported surface is static-first source install and installed static consumer proof through Make, CMake, `pkg-config`, and named reviewed CI lanes. |
| Evidence needed to revisit | A future sprint must select exactly one provider, identify source input, define checksum/license/version/dependency/static policy, add source-controlled provider material, prove isolated install, downstream consumer behavior, cleanup, documentation, and guard coverage. |
| Consequence for Sprint 180 | The Sprint 171 deferral remains the baseline until Sprint 180 records a replacement product decision or stronger deferral. |

## Current Guard Baseline

| Guard area | Current behavior |
| --- | --- |
| Deferral record | `scripts/package_manager_deferral_check.sh` requires the Sprint 171 deferral record and key deferral/revisit wording. |
| Provider recipe absence | The guard fails if unselected provider recipe artifacts such as `vcpkg.json`, `portfile.cmake`, `conanfile.py`, `Formula/`, `pkgsrc/`, Debian files, or spec files appear outside excluded trees. |
| Package metadata neutrality | The guard rejects provider/package-manager wording in `sparse.pc.in` and `cmake/SparseConfig.cmake.in`. |
| Public non-claims | The guard requires README, INSTALL, and maintainer guide wording that keeps package-manager support scoped as a non-claim. |

## Sprint 177 Acceptance Gate Baseline

Sprint 180 implements Sprint 177 Day 8 Gate 3:

| Field | Acceptance requirement |
| --- | --- |
| Target | Package-manager provider proof or deferral. |
| Owner files | `scripts/package_manager_deferral_check.sh`, package metadata templates, `INSTALL.md`, README, maintainer guide, provider proof or deferral artifacts, and optional provider prototype files if selected. |
| Required evidence | Prove one static-first provider path with a local proof script or publish a stronger formal deferral with exact blockers and fail-closed guards. |
| Validation commands | Provider proof or deferral script; `bash scripts/package_manager_deferral_check.sh`; install checks if package metadata changes; `git diff --check`; full quality gates if C/header files change. |
| Pass definition | Exactly one provider decision is recorded, wording and metadata match the decision, unsupported providers remain absent or guarded, and static-first/non-ABI boundaries remain intact. |
| Fail definition | Provider wording appears without proof, recipe files appear without a guard decision, static-first metadata weakens, or broad package-manager support is implied. |
| Claim boundary | One provider path is proven, or provider support is more strongly and explicitly deferred. |
| Protected non-claims | No broad package-manager ecosystem support, binary package availability, upgrade behavior, registry readiness, shared-library support, or dynamic ABI compatibility. |

## Provider Evaluation Criteria

| Criterion | Day 1 definition |
| --- | --- |
| Static-first fit | Provider path must consume the maintained static archive install/export surface without implying shared-library, dynamic ABI, runtime-loader, or static/shared selector support. |
| CI feasibility | Proof must be locally runnable or CI-feasible with bounded setup, bounded runtime, deterministic cleanup, and clear unavailable-tool failure behavior. |
| Recipe complexity | Recipe, formula, manifest, port, patch, metadata, and update surface should be small enough for maintainers to review and keep current. |
| User value | Provider path should reduce real adoption friction for likely users, not merely add a file that cannot be validated or maintained. |
| Proof completeness | The selected path should support install, downstream compile/link/run, version query, cleanup, and claim-safe failure proof where feasible. |
| Maintenance cost | Ongoing source archive, checksum, dependency, license, version, registry/tap policy, and provider-update burden must be explicit. |
| Claim risk | Evaluation must identify adjacent unsupported claims users could infer, including provider availability, binary packages, upgrade behavior, platform parity, shared-library support, and ABI compatibility. |

## Provider Candidate Starting Positions

| Candidate | Day 1 starting position |
| --- | --- |
| vcpkg | Plausible first-provider candidate because it can build on CMake install/export and may align with Windows static-first proof, but registry readiness and manifest/port proof are unearned. |
| Homebrew | Plausible macOS-first candidate because macOS package/install evidence exists, but formula, tap, source archive, checksum, bottle, and platform-scope boundaries are unearned. |
| Conan | Open candidate because it can model C/C++ package metadata, but recipe/package ID/options/profile complexity and support wording require audit. |
| pkgsrc | Open candidate, but likely higher policy and platform proof cost; must still be evaluated against the same criteria before rejection. |
| Stronger deferral | Valid outcome if no provider has enough proof completeness, maintainability, and claim-safety for Sprint 180. |

## Protected Non-Claims

Sprint 180 must not imply these claims unless the final provider decision and
validation explicitly earn them:

- broad package-manager support;
- vcpkg, Homebrew, Conan, pkgsrc, distro, registry, tap, or binary-package
  support beyond the selected path;
- provider-hosted binary packages;
- provider-managed dependency resolution or upgrade behavior;
- registry readiness, upstream acceptance, tap readiness, or bottle support;
- Windows package-manager support unless specifically selected and proven;
- shared-library build/install support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- static/shared package selectors;
- broad platform parity;
- release provenance, signing, or package-provider upgrade guarantees.

## Day 1 Decisions

- Treat the Epic 16 project plan as Sprint 180 source authority.
- Treat Sprint 171 package-manager deferral as the inherited baseline.
- Treat Sprint 177 Gate 3 as the Sprint 180 acceptance gate.
- Keep vcpkg, Homebrew, Conan, pkgsrc, and stronger deferral open until the
  feasibility audit and decision matrix are complete.
- Do not add provider recipe files or promote package-manager wording on Day 1.

## Day 1 Deliverables

- `docs/planning/EPIC_16/SPRINT_180/WORKING_NOTES.md`
- `docs/planning/EPIC_16/SPRINT_180/artifacts/`
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day1-provider-decision-intake.md`
- inherited guard and acceptance-gate notes
- provider evaluation criteria

## Validation

Day 1 changed planning artifacts only. No `.c`, `.h`, package metadata, guard,
or public user-facing docs were modified, so the full C quality gate and
install checks are not required for this day.

Validation commands:

```sh
bash scripts/package_manager_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Sprint 180 scope is tied to the Epic 16 project plan. | Complete | Source authority and scope sections above. |
| Inherited deferral and acceptance-gate requirements are explicit. | Complete | Sprint 171 deferral baseline and Sprint 177 Gate 3 sections above. |
| Provider comparison work starts from shared decision criteria. | Complete | Provider evaluation criteria table above. |
