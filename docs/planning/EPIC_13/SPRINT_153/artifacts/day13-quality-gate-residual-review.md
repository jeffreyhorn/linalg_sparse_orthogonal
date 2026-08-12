# Sprint 153 Day 13 Quality Gate And Residual Review

## Purpose

Day 13 records the final quality-gate decision and residual package/ABI debt
before Sprint 153 closeout. The selected product decision remains stronger
static-first deferral, not shared-library implementation.

## Changed-File Gate Decision

`git diff --name-only` showed changes in:

- `.github/workflows/windows-ci.yml`;
- `CMakeLists.txt`;
- `INSTALL.md`;
- `README.md`;
- `docs/maintainer_guide.md`;
- `scripts/static_package_deferral_check.sh`;
- `tests/test_cmake_install.sh`;
- Sprint 153 planning artifacts.

No `.c` files or public `.h` headers changed. The full C quality gate
`make format && make lint && make test` is therefore not required for Day 13.
Focused package, report-index, documentation, and whitespace validation is the
appropriate gate for the touched surfaces.

## Focused Validation Results

| Validation | Result | Evidence |
| --- | --- | --- |
| Static deferral guard | Pass | `bash scripts/static_package_deferral_check.sh` passed. |
| Make install/package proof | Pass | `bash tests/test_install.sh` passed with `23` checks and `0` failures. |
| CMake install/export proof | Pass | `bash tests/test_cmake_install.sh` passed with `27` checks, `0` failures, and `0` skips. |
| Package report-index structure | Pass | `python3 scripts/normalize_report_index.py --family package --check` reported `6` rows ok. |
| Package report-index freshness meaning | Pass | `python3 scripts/normalize_report_index.py --family package --check-freshness` reported freshness ok for `6` source-controlled rows. |
| Runtime backend report-index freshness meaning | Pass | `python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness` reported freshness ok for `1` source-controlled row. |

## Stale Wording Review

Reviewed active package/ABI surfaces with a focused search over:

- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `.github/workflows`;
- `sparse.pc.in`;
- `cmake/SparseConfig.cmake.in`;
- `tests/test_install.sh`;
- `tests/test_cmake_install.sh`;
- `scripts/static_package_deferral_check.sh`;
- `CMakeLists.txt`.

The remaining hits are expected and belong to one of these categories:

- explicit non-claims;
- static-first deferral diagnostics;
- guard/test patterns that reject unsupported metadata;
- documentation of the exact blockers for future shared-library support.

No active searched wording claims shared-library packaging, dynamic ABI
compatibility, runtime-loader behavior, package-manager distribution,
static/shared selectors, Windows Makefile parity, or Windows `pkg-config`
execution parity.

## Residual Debt Register

| Residual | Status | Owner Candidate | Close Condition |
| --- | --- | --- | --- |
| Public export/import macro policy | Deferred | Future shared-library sprint | `SPARSE_API` or equivalent is designed, applied, documented, and tested across supported compilers. |
| Symbol visibility/export allowlist | Deferred | Future shared-library sprint | Internal symbols are hidden by default or excluded through export lists/linker scripts/`.def` files, with symbol inspection proof. |
| Dynamic ABI compatibility policy | Deferred | Future ABI governance sprint | Public structs, callbacks, enum values, allocator boundaries, error state, and version metadata have an explicit compatibility policy. |
| Linux `.so` support | Deferred | Future platform/package sprint | Shared build/install creates reviewed `.so` artifacts with SONAME and installed downstream loader proof. |
| macOS `.dylib` support | Deferred | Future platform/package sprint | Shared build/install creates reviewed `.dylib` artifacts with install-name/RPATH and installed downstream loader proof. |
| Windows DLL/import-library support | Deferred | Future Windows package sprint | DLL/import-library naming, install layout, `__declspec`, runtime lookup, and C runtime allocator behavior are reviewed and tested. |
| Static/shared package selectors | Deferred | Future package metadata sprint | CMake and `pkg-config` selector semantics are designed, implemented, documented, and tested for both selected modes. |
| Package-manager distribution | Deferred | Future release/productization sprint | Package-manager claims are backed by actual packaging, install, version, and CI proof. |
| Windows Makefile parity | Deferred | Future Windows build sprint | Windows Makefile behavior is implemented or explicitly remains unsupported with reviewed proof. |
| Windows `pkg-config` execution parity | Deferred | Future Windows package sprint | Windows `pkg-config` execution is installed, exercised, and documented as a reviewed lane. |

## Day 14 Handoff

Day 14 should finalize Sprint 153 artifacts and write a closeout handoff for
Sprint 154. The handoff should state:

- Sprint 153 selected stronger static-first deferral;
- shared-library support remains unimplemented by design;
- CMake rejection diagnostics and deferral guard now name exact blockers;
- Unix and Windows install proofs reject unsupported loader/static-shared
  metadata;
- external comparison work must not infer shared-library, dynamic ABI,
  runtime-loader, or package-manager support from static package proof.
