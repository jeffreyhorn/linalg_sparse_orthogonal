# Sprint 180 Day 12: Proof Script Implementation

**Sprint:** 180 - Package-Manager Provider Decision
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_180/`
**Status:** Complete

## Purpose

Implement the local Homebrew formula proof script designed on Day 11 and record
the locally observed proof behavior. Day 12 adds the script, verifies focused
preflight behavior, and records the current stop condition: accurate Homebrew
license metadata is unavailable because the repository has no standalone
license file.

## Implemented Script

| Path | Status | Role |
| --- | --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Added | Generates a temporary local source archive, renders the Homebrew formula template, and runs local Homebrew install/test/uninstall proof when prerequisites and license metadata are available. |

## Script Behavior

The script implements these stages:

1. Parse `--keep-temp` and `--help`.
2. Resolve the repository root, template, and `VERSION`.
3. Check local tools: `brew`, `cmake`, Ruby, C compiler, `tar`, and SHA-256
   tool.
4. Check required formula template placeholders and Ruby syntax.
5. Create a temporary proof root.
6. Create a temporary source archive from the current checkout's source,
   CMake/install metadata, examples, and any standalone license files.
7. Compute the archive SHA-256.
8. Stop safely if standalone license metadata is unavailable.
9. If license metadata is available, render a temporary local formula.
10. Install the rendered formula from source with Homebrew.
11. Check installed static archive, headers, CMake package files, and
    `sparse.pc`.
12. Reject shared-library artifacts and unsupported provider wording in
    installed package metadata.
13. Run `brew test` for the downstream CMake consumer.
14. Uninstall the local formula and clean temporary output.

## Current Local Result

The script is locally runnable and stops safely before invoking `brew install`
because the repository has no standalone `LICENSE`, `COPYING`, or `NOTICE`
file and no accurate `SPARSE_HOMEBREW_LICENSE` value can be used.

Observed command:

```sh
bash scripts/homebrew_local_formula_proof.sh
```

Observed result:

```text
homebrew-local-formula-proof: UNAVAILABLE: formula rendering blocked: no standalone LICENSE, COPYING, or NOTICE file exists for provider metadata
homebrew-local-formula-proof: local Homebrew proof remains unclaimed
```

Exit status: `2`.

This is the intended claim-safe failure path. It does not claim Homebrew
support, Homebrew/core readiness, bottle support, Linuxbrew support, or broad
package-manager support.

## Focused Checks

| Check | Result |
| --- | --- |
| `test -x scripts/homebrew_local_formula_proof.sh` | Passed |
| `bash -n scripts/homebrew_local_formula_proof.sh` | Passed |
| `bash scripts/homebrew_local_formula_proof.sh --help` | Passed |
| `bash scripts/homebrew_local_formula_proof.sh` | Claim-safe unavailable, exit `2`, due to missing standalone license metadata. |
| `bash scripts/package_manager_deferral_check.sh` | Passed |
| `bash scripts/static_package_deferral_check.sh` | Passed |
| `git diff --check` | Passed |

## Unsupported Proof Gaps

| Gap | Status | Follow-up |
| --- | --- | --- |
| Formula install proof | Blocked | Needs accurate standalone license metadata before rendering and `brew install`. |
| `brew test` downstream consumer proof | Blocked | Runs only after formula install proof is allowed. |
| Uninstall after installed formula | Not reached | Cleanup trap is implemented; installed-formula cleanup needs a successful install path to validate. |
| Day 13 guard/docs alignment | Pending | Guard and public docs still preserve Sprint 171 deferral until Day 13 updates them to the exact observed proof level. |

## Claim Boundary

Day 12 proves that the script fails safely when the selected provider proof is
not fully available. The current supported public posture remains Sprint 171
package-manager deferral.

The script and artifacts must not be described as:

- Homebrew/core support;
- bottle support;
- Linuxbrew support;
- broad Homebrew support;
- broad package-manager support;
- provider-hosted binary packages;
- shared-library support;
- dynamic ABI support.

## Day 12 Deliverables

- provider proof script
- install/downstream/version/cleanup proof wiring
- claim-safe failure messages
- focused proof validation summary
- Day 12 implementation notes
- `docs/planning/EPIC_16/SPRINT_180/artifacts/day12-proof-script-implementation.md`

## Validation

Day 12 added a shell proof script and planning notes. It did not modify `.c`,
`.h`, package metadata templates, workflows, guards, or public user-facing
package-manager docs.

Validation commands:

```sh
test -x scripts/homebrew_local_formula_proof.sh
bash -n scripts/homebrew_local_formula_proof.sh
bash scripts/homebrew_local_formula_proof.sh --help
bash scripts/homebrew_local_formula_proof.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Proof script exists and is locally runnable where feasible. | Complete | `scripts/homebrew_local_formula_proof.sh` exists and runs through preflight/archive/checksum before stopping safely on license metadata. |
| Unavailable-provider behavior fails safely without implying support. | Complete | Missing license metadata exits `2` with local proof unclaimed. |
| Cleanup behavior is validated or the remaining gap is documented. | Complete | Temp cleanup trap is implemented; installed-formula cleanup remains a blocked-path gap because install is not reached. |
