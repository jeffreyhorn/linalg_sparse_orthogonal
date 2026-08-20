# Sprint 171 Day 9: Package Claim Guard Implementation

## Purpose

Day 9 implements the package claim guard updates designed on Day 8. Sprint 171
selected formal package-manager deferral, so the implementation promotes the
package-manager deferral guard into the normalized package proof-owner surface
and aligns the user/maintainer documentation that lists package proof owners.

## Implemented Guard Updates

| Surface | Change | Claim Boundary Protected |
| --- | --- | --- |
| `scripts/normalize_report_index.py` | Added `package_manager_deferral` as a package proof-owner row. | Package evidence now includes the source-controlled guard for package-manager provider non-claims. |
| `INSTALL.md` | Added `scripts/package_manager_deferral_check.sh` to the normalized package rows proof-owner list. | User-facing package proof documentation now separates package-manager deferral from source install proof. |
| `docs/maintainer_guide.md` | Added package-manager deferral guard ownership and normalized package row coverage. | Maintainers have a direct owner for provider recipe absence, package metadata neutrality, and public non-claim wording. |

## Existing Guard Behavior Preserved

Day 9 did not duplicate package-manager checks into
`scripts/static_package_deferral_check.sh`. That script remains the owner for
the Sprint 170 static-first package posture, shared-library deferral, dynamic
ABI non-claims, and `BUILD_SHARED_LIBS=ON` rejection.

`scripts/package_manager_deferral_check.sh` remains the focused owner for the
Sprint 171 package-manager decision:

- it requires the Sprint 171 Day 5 deferral record;
- it checks unsupported provider wording and revisit evidence;
- it fails on unselected provider recipe artifacts outside planning/archive
  locations;
- it checks package metadata templates for provider neutrality;
- it checks public docs for package-manager non-claim wording.

## Normalized Package Proof-Owner Row

The new package proof-owner row has:

| Field | Value |
| --- | --- |
| `proof_name` | `package_manager_deferral` |
| `path` | `scripts/package_manager_deferral_check.sh` |
| `command` | `bash scripts/package_manager_deferral_check.sh` |
| `scope` | Package-manager deferral guardrail for provider support non-claims. |
| `configuration` | `package_surface=static_first;artifact_kind=source_controlled` |
| `freshness_status` | `source_controlled` when the script exists |

This row proves source-controlled ownership and scope. It does not claim that a
provider package was built, submitted, installed, or accepted by a registry.

## Unsupported Claims Still Blocked

The implemented guard surface continues to reject or avoid:

- vcpkg, Homebrew, Conan, pkgsrc, Debian/Fedora/system package, provider
  registry, tap, and binary-package support claims;
- package metadata wording that implies provider or package-manager
  distribution support;
- shared-library packaging, dynamic ABI compatibility, runtime-loader
  compatibility, and static/shared selector claims;
- Windows Makefile parity or Windows `pkg-config` command execution parity
  inferred from Windows CMake package proof.

## Focused Validation

Day 9 validation commands:

```sh
bash -n scripts/package_manager_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
```

Expected result:

- package-manager deferral guard passes;
- static-first/shared-library ABI deferral guard still passes;
- normalized package proof-owner rows include
  `package_package_manager_deferral_v1`;
- package freshness remains source-controlled;
- diff hygiene passes.

## Day 9 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Implemented package claim guards | Complete | The package-manager deferral guard is now part of normalized package proof ownership. |
| Selected deferral checks | Complete | Existing executable deferral checks remain the source of truth. |
| Focused guard validation log | Complete | Validation command list and expected outcomes are recorded above. |
| Day 9 claim-guard artifact | Complete | This file. |

## Validation

Day 9 changed Python, Markdown, and planning artifacts. No `.c` or `.h` files
were modified, so the full C quality gate is not required for this day.

Validation command:

```sh
bash -n scripts/package_manager_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Unsupported package-manager claims fail mechanically where feasible. | Complete | The package-manager deferral guard enforces deferral wording, metadata neutrality, public non-claims, and provider recipe absence. |
| Existing static-first package guards still pass. | Complete | Sprint 170 static/shared ABI guard behavior remains separate and unchanged. |
| Package-manager docs can be updated from a guarded boundary. | Complete | INSTALL and maintainer guide now list the package-manager deferral guard as a package proof owner. |
