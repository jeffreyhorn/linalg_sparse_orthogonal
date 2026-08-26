# Sprint 180 Day 13: Guard And Docs Update

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Reconcile package-manager guards and public docs with the actual Sprint 180
provider state: Homebrew local proof artifacts exist, but provider support is
not claimed because the proof exits before install on missing standalone
license metadata.

## Updated Surfaces

| Surface | Update |
| --- | --- |
| `scripts/package_manager_deferral_check.sh` | Now checks the selected Homebrew local proof boundary, generated-output hygiene, proof-script behavior, provider-neutral metadata, and public non-claims. |
| `README.md` | Notes that package-manager support remains unavailable and that local Homebrew proof remains unclaimed due to missing standalone license metadata. |
| `INSTALL.md` | Updates the support split with the selected local Homebrew proof artifacts and current missing-license blocker. |
| `docs/maintainer_guide.md` | Documents the selected proof script and warns maintainers not to cite it as Homebrew support while the proof is blocked. |
| `sparse.pc.in` | Unchanged and provider-neutral. |
| `cmake/SparseConfig.cmake.in` | Unchanged and provider-neutral. |

## Guard Behavior

The package-manager guard now requires:

- Sprint 171 deferral record still exists as the public non-claim baseline;
- unselected provider files remain absent;
- selected Homebrew proof template, notes, and executable proof script exist;
- generated Homebrew archives, logs, bottles, rendered formulae, and live
  `Formula/` paths are not committed;
- `scripts/homebrew_local_formula_proof.sh` exits either successfully for the
  local proof level or claim-safely with status `2`;
- current docs say local Homebrew formula proof remains unclaimed;
- package metadata templates contain no provider wording.

## Current Provider Status

| Field | Status |
| --- | --- |
| Selected path | Homebrew local formula proof. |
| Proof artifact | Present under `packaging/homebrew/`. |
| Proof script | Present at `scripts/homebrew_local_formula_proof.sh`. |
| Current proof result | Claim-safe unavailable, exit `2`, due to missing standalone license metadata. |
| Public support | Not claimed. |
| Unsupported claims | Homebrew/core, bottles, Linuxbrew, broad Homebrew support, broad package-manager support, binary packages, shared libraries, and dynamic ABI support. |

## Validation

Validation commands:

```sh
bash scripts/homebrew_local_formula_proof.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

`scripts/homebrew_local_formula_proof.sh` exits `2` claim-safely in the
current repository state. `scripts/package_manager_deferral_check.sh` treats
that as the expected unclaimed-provider posture.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Documentation matches the provider decision exactly. | Complete | README, INSTALL, and maintainer guide describe selected local Homebrew proof artifacts and the missing-license blocker without claiming support. |
| Guards reject unsupported package-manager claims. | Complete | Package-manager guard rejects unselected providers and generated Homebrew output while checking selected proof behavior. |
| Package metadata does not imply unearned provider support. | Complete | `sparse.pc.in` and `cmake/SparseConfig.cmake.in` remain unchanged and provider-neutral. |
