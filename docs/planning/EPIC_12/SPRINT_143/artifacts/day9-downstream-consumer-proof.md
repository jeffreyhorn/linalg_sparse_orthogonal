# Sprint 143 Day 9 Downstream Consumer Proof

## Purpose

Strengthen the selected static-first downstream consumer proof for Make,
`pkg-config`, CMake, exact-version constraints, and unsupported artifacts. No
loader checks are added because Sprint 143 selected static-first-only support
and explicitly deferred shared-library ABI support.

## Changes Implemented

| Surface | Change | Reason |
| --- | --- | --- |
| `tests/test_install.sh` basic `pkg-config` consumer | Runtime output must now include version text, encoded version text, `nnz: 1`, and `OK`. | Proves the installed headers and static archive are used for more than a trivial executable exit. |
| `tests/test_install.sh` maintained example via `pkg-config` | Runtime output must include library version text, solution output, and `OK`. | Keeps the maintained example proof deterministic instead of accepting any output with `OK`. |
| `tests/test_cmake_install.sh` installed CMake example | Runtime output must include library version text, solution output, and `OK`. | Strengthens installed `find_package(Sparse)` consumer proof without changing the example. |
| `tests/test_cmake_install.sh` exact-version consumer | Exact-version `find_package(Sparse VERSION EXACT REQUIRED)` consumer is now built and run after configure succeeds. | Converts exact-version proof from configure-only to configure/build/run downstream proof. |

## Unsupported Artifact And Loader Boundary

Unsupported artifacts remain checked explicitly:

- Make install proof rejects `.so`, `.so.*`, `.dylib`, and `.dll` artifacts.
- CMake install proof rejects `.so`, `.so.*`, `.dylib`, and `.dll` artifacts.
- CMake package metadata proof rejects shared/module imported targets and
  shared imported locations.
- `pkg-config` proof rejects `Libs.private` and unsupported package/ABI
  wording under the current self-contained static contract.

No loader/runtime dynamic-link checks were added. The selected package path is
static-first-only, so loader behavior remains a deferred shared-library
residual rather than a pass condition.

## Deterministic Skip/Fail Behavior

- Missing `pkg-config` remains a failure for `tests/test_install.sh` because
  that script owns the Unix-side `pkg-config` downstream proof.
- Exact-version mismatch proof skips only when no lower same-major version can
  be constructed from `VERSION`.
- Exact-version consumer build/run is a failure if configure succeeds but the
  downstream executable cannot build or run.

## Focused Validation

Focused checks run for this batch:

```sh
bash -n tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh
bash scripts/static_package_deferral_check.sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_12/SPRINT_143 tests/test_install.sh tests/test_cmake_install.sh
```

Results:

| Check | Result |
| --- | --- |
| Shell syntax checks | Passed |
| `scripts/static_package_deferral_check.sh` | Passed |
| `tests/test_install.sh` | Passed: 23 passed, 0 failed |
| `tests/test_cmake_install.sh` | Passed: 26 passed, 0 failed, 0 skipped |
| Package report index check | Passed: 6 rows |
| Package report freshness check | Passed: 6 source-controlled advisory rows |

## Day 10 Input

Day 10 should align CI and package report wording with this proof shape:

1. Linux remains the reviewed package-contract lane.
2. macOS and Windows package install/downstream lanes remain supplemental.
3. Package report rows remain source-controlled proof-owner metadata.
4. CI wording should not imply loader, shared ABI, package-manager, or platform
   parity support from these static downstream proofs.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected package path has executable downstream proof. | Complete | Make `pkg-config`, maintained example, CMake installed example, and exact-version CMake consumer all build/run where applicable. |
| Unsupported artifacts are checked explicitly. | Complete | Install scripts and CMake metadata proof reject shared artifacts and shared imported metadata. |
| Proof scripts do not overclaim platform or package-manager support. | Complete | No loader checks were added; package-manager and shared ABI wording remains rejected. |
