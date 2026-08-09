# Sprint 144 Day 10 Selected-Lane Validation

## Purpose

Run a focused validation pass for the selected macOS reviewed static-first
install/export lane after Days 5-9 updated workflow, report, and documentation
support-tier surfaces.

## Touched Surface Summary

| Surface | Current role |
| --- | --- |
| `.github/workflows/macos-ci.yml` | Defines reviewed macOS static-first Make install/`pkg-config` and CMake install/export proof jobs. |
| `tests/corpus/manifests/report_families.tsv` | Makes the promoted macOS CI lane discoverable as source-controlled CI lane metadata. |
| `README.md` | Public CI summary now names reviewed macOS static-first install/export proof. |
| `INSTALL.md` | User-facing install/support interpretation now names reviewed macOS static-first install/export proof. |
| `docs/maintainer_guide.md` | Maintainer support-tier guidance now names reviewed macOS static-first install/export proof and preserved non-claims. |

## Validation Results

| Check | Result | Interpretation |
| --- | --- | --- |
| `bash tests/test_install.sh` | Passed: 23 passed, 0 failed | Local Make install/`pkg-config` proof still validates the static archive package path. |
| `bash tests/test_cmake_install.sh` | Passed: 26 passed, 0 failed, 0 skipped | Local CMake install/export and downstream consumer proof still validates the static package path. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static-first shared/ABI/package-selector deferral guard still passes. |
| Ruby YAML parse for Linux, macOS, and Windows workflows | Passed | Workflow syntax is locally parseable. |
| stale macOS supplemental install/export wording scan | Passed | Current selected-lane docs/workflow no longer describe macOS package proof as supplemental. |
| package report normalization | Passed: 6 rows | Package proof-owner rows remain valid. |
| package report freshness | Passed: 6 source-controlled advisory rows | Package rows remain advisory source-controlled proof-owner metadata. |
| CI report normalization | Passed: 1 row | CI lane-definition contract row remains valid. |
| CI report freshness | Passed: 1 source-controlled advisory row | CI row remains hosted-log external evidence, not local pass evidence. |

## Failure Classification

No failures occurred.

| Failure class | Day 10 status |
| --- | --- |
| Implementation defect | None found. |
| Environment constraint | Hosted `macos-latest` execution remains external to local validation. |
| Explicit non-claim | Shared-library packaging, dynamic ABI compatibility, runtime-loader compatibility, package-manager support, static/shared selectors, Windows parity, and broader macOS platform parity remain non-claims. |

## Hosted Evidence Boundary

Day 10 local validation proves that the selected proof scripts, report
semantics, workflow syntax, and documentation wording are coherent in this
checkout.

The reviewed macOS platform claim is fully earned only when GitHub Actions runs
these promoted jobs on `macos-latest`:

- `macOS reviewed static-first install and pkg-config proof`;
- `macOS reviewed static-first CMake install/export proof`.

## Skipped Checks

| Check | Reason |
| --- | --- |
| Hosted GitHub Actions macOS job execution | Not available from local checkout; PR CI is the proof owner. |
| `make format && make lint && make test` | No `.c` or `.h` files changed during Day 10. |
| Windows package promotion checks | Backup lane was not activated. |
| macOS CTest count checks | Selected lane owns install/export script proof, not CTest count governance. |

## Selected-Lane Evidence Summary

The selected macOS lane is ready for CI-hosted proof:

- package commands are unchanged and passed locally;
- static-first deferral guard passed locally;
- workflow syntax passed locally;
- docs and workflow wording agree on reviewed macOS static-first
  install/export proof;
- package and CI report rows remain advisory metadata with clear non-claims.

## Day 11 Handoff

Day 11 should run a cross-platform non-regression review:

1. Confirm Linux source-of-truth wording and package proof ownership remain
   unchanged.
2. Confirm Windows reviewed CMake subset, supplemental install/downstream
   confidence, and staged exclusions remain unchanged.
3. Re-scan docs/workflows for unsupported package/platform claims.
4. Re-check report semantics if any additional wording changes are needed.

## Day 10 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected-lane proof passes locally where feasible. | Complete | Install, CMake install/export, static deferral, workflow syntax, docs scans, and report checks passed. |
| Unresolved checks are tied to explicit environment or source blockers. | Complete | Only hosted `macos-latest` execution remains external to local validation. |
| Working notes contain enough evidence for closeout and review. | Complete | Working notes record command results, hosted boundary, and no C/header changes. |
