# Sprint 207 Day 9: Package Guard Alignment

## Purpose

Align package-manager and static-package guards with the Sprint 207 selected
continued-deferral tier and retained non-claims.

## Guard Ownership

| Guard | Owner surface | Day 9 alignment |
| --- | --- | --- |
| `scripts/package_manager_deferral_check.sh` | Package-manager provider support, local Homebrew proof boundary, provider recipe absence, public package non-claims | Now enforces Sprint 207 continued-deferral decision, forbidden public provider claims, release/package artifact non-claims, local proof boundary, and public non-claim wording. |
| `scripts/static_package_deferral_check.sh` | Static-first package contract, shared-library packaging deferral, dynamic ABI deferral, Windows package non-claims | Reviewed and validated unchanged; already covers static/shared selector, shared-library, dynamic ABI, and Windows package execution boundaries. |
| `tests/test_package_manager_deferral_guard.py` | Focused package-manager guard regressions | Expanded to cover Homebrew/core readiness, public tap support, binary package support, bottle support, Linuxbrew support, and release package support overclaims. |

## Package Guard Alignment Changes

Extended `scripts/package_manager_deferral_check.sh` forbidden positive
provider patterns to reject:

- release readiness claims;
- release artifact claims;
- release package claims;
- package release claims.

The existing Day 7/8 patterns already rejected:

- package-manager distribution support;
- Homebrew/core readiness/support;
- public tap support;
- bottle support;
- Linuxbrew support;
- vcpkg support;
- Conan support;
- pkgsrc support;
- distro/system package support;
- binary package support.

## Regression Expansion

Extended `tests/test_package_manager_deferral_guard.py` with these focused
regressions:

| Regression | Injected wording | Expected failure |
| --- | --- | --- |
| Bottle support | `bottle support is available.` | Unsupported provider claim. |
| Linuxbrew support | `Linuxbrew support is provided.` | Unsupported provider claim. |
| Release package support | `release packages are available.` | Unsupported provider claim. |

Existing regressions still cover:

- `Homebrew/core readiness is supported.`;
- `public tap support is available.`;
- `binary package support is provided.`

## Static Guard Review

`scripts/static_package_deferral_check.sh` was reviewed for Day 9. No code
change was needed because it already enforces:

- Sprint 170 static-first package decision record;
- `BUILD_SHARED_LIBS=ON` rejection;
- static archive target and install contract;
- no shared-library install destinations;
- no export/import or dynamic ABI metadata;
- no static/shared package selector in CMake/pkg-config metadata;
- README/INSTALL/maintainer wording that keeps shared-library packaging and
  dynamic ABI support deferred;
- Windows package non-claim wording;
- no unselected Windows Makefile install or `pkg-config` execution.

## Validation

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_package_manager_deferral_guard.py` | Passed | Focused package-provider overclaim regressions passed. |
| `python3 -m py_compile tests/test_package_manager_deferral_guard.py` | Passed | Python regression syntax is valid. |
| `bash -n scripts/package_manager_deferral_check.sh` | Passed | Package guard shell syntax is valid. |
| `bash scripts/static_package_deferral_check.sh` | Passed | Static package, shared-library, dynamic ABI, and Windows package non-claims remain guarded. |
| `bash scripts/package_manager_deferral_check.sh` | Passed | Package-manager deferral, Sprint 207 decision, forbidden provider claims, selected local Homebrew proof, package metadata neutrality, and public non-claims passed. |
| Installed formula scan | Passed | No `sparse-lu-ortho-local` formula remained installed. |
| Temporary proof tap scan | Passed | No `sparse-lu-ortho/local-proof-*` tap remained. |
| Generated Homebrew output scan | Passed | No generated `.tar.gz`, `.tgz`, `.zip`, `.log`, `.rb`, `.bottle.*`, or `Formula/` output appeared under `packaging/homebrew`. |

## Guard Failure Guidance

If package-provider wording needs to change in the future, maintainers must
first add evidence for the exact provider tier:

- public tap/source formula needs provider formula ownership, stable source
  archive policy, checksum provenance, install/test/uninstall proof, cleanup
  proof, and claim-safe docs;
- Homebrew/core readiness needs all public tap evidence plus Homebrew/core
  audit-readiness and release/archive discipline;
- bottles, Linuxbrew, binary packages, vcpkg, Conan, pkgsrc, distro/system
  packages, shared-library packages, dynamic ABI, and release readiness need
  separate evidence before any positive claim.

## Day 9 Completion Criteria

| Criterion | Status |
| --- | --- |
| Item 207.4 is covered by enforceable guard behavior. | Complete. |
| Unselected package paths remain blocked by validation. | Complete. |
| Guard failures explain how to update evidence before changing claims. | Complete through Sprint 207 decision/design checks and this guard-alignment artifact. |

## Day 9 Outcome

Package guard alignment is complete for the selected continued-deferral tier.
The repository now has shell guard coverage, static package guard coverage, and
focused Python regression coverage for the main unsupported package-provider
claim surfaces.
