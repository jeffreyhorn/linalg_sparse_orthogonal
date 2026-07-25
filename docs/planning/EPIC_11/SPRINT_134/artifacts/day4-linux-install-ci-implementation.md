# Sprint 134 Day 4 - Linux Install CI Implementation

## Purpose

Day 4 implements the Day 3 decision to promote a bounded Linux reviewed
static-first package-contract lane. The new lane validates the maintained
static archive install/export surface without widening shared-library,
dynamic ABI, package-manager, runtime-loader, macOS install/export, Windows
install-validation, or Windows Makefile claims.

## Implemented Changes

| File | Change |
| --- | --- |
| `.github/workflows/ci.yml` | Added `package-contract`, a separate Linux reviewed static-first package contract job. |
| `README.md` | Updated the CI summary to include the reviewed Linux static-first package contract. |
| `INSTALL.md` | Updated platform and install-validation wording to distinguish Linux reviewed package-contract proof from narrower macOS/Windows tiers. |
| `docs/maintainer_guide.md` | Recorded the Linux reviewed package-contract lane and corrected the Windows reviewed CTest count to 54. |

No C source or public header files changed.

## Linux Package Contract Job

The new Linux CI job runs on `ubuntu-latest`:

```yaml
package-contract:
  name: Linux reviewed static-first package contract
```

The job installs package proof tooling and then runs:

1. `bash tests/test_install.sh`
2. `bash tests/test_cmake_install.sh`
3. `bash scripts/static_package_deferral_check.sh`

The job is intentionally separate from the existing Linux reviewed
compile-quality, reviewed CMake parity, reviewed dead-code, and supplemental
runtime jobs so failures remain easy to classify.

## Support Boundary

| Claim | Day 4 status |
| --- | --- |
| Linux static-first package contract | Promoted to reviewed Linux CI. |
| Linux shared-library packaging | Still deferred and unsupported. |
| Linux dynamic ABI compatibility | Still deferred and unsupported. |
| Linux package-manager support | Still deferred and unsupported. |
| macOS CMake install/export parity | Unchanged; still pending Days 5-7. |
| Windows install validation | Unchanged; still pending Days 8-9. |
| Windows Makefile parity | Unchanged non-claim. |

## Validation

Day 4 local validation:

| Command | Result |
| --- | --- |
| `bash -n scripts/static_package_deferral_check.sh` | Pass. |
| `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/ci.yml")'` | Pass. |
| `git diff --name-only -- '*.c' '*.h'` | No C/header changes. |
| `bash tests/test_install.sh` | Pass: 22 checks, 0 failures. |
| `bash tests/test_cmake_install.sh` | Pass: 21 checks, 0 failures, 0 skips. |
| `bash scripts/static_package_deferral_check.sh` | Pass. |
| `git diff --check` | Pass. |
| Focused whitespace scan | Pass. |

## Residual Linux CI Queue

| Residual | Status |
| --- | --- |
| Runtime observation in GitHub CI | Deferred to PR/CI observation after the new lane runs on the hosted runner. |
| Package-manager validation | Deferred non-claim. |
| Optional `SPARSE_MUTEX`/`SPARSE_OPENMP` package matrix | Deferred; current lane validates the default package contract. |
| Shared-library and dynamic ABI validation | Deferred by Sprint 133 product decision. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected Linux decision is reflected in code/docs. | Complete | Workflow and support docs name the Linux reviewed static-first package-contract lane. |
| Workflow-equivalent local evidence exists for touched Linux surfaces. | Complete | The promoted package proof commands passed locally with syntax, YAML, diff, and whitespace hygiene. |
| Support wording still distinguishes reviewed, supplemental, and local proof. | Complete | README, INSTALL, maintainer guide, and workflow comments keep macOS/Windows narrower and preserve package non-claims. |
