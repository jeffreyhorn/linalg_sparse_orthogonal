# Day 4 Support Matrix Implementation

## Objective

Add the compact support/readiness matrix and wire top-level docs to it as the
active user-facing support truth.

## Implemented Matrix

`INSTALL.md` now owns `Support Readiness Matrix`, with rows for:

- local source build and first solve;
- Unix Make static install;
- Unix `pkg-config` consumer;
- installed CMake consumer;
- Windows MSVC CMake install/downstream;
- Windows selected Cholesky comparison freshness;
- Linux/macOS selected comparison freshness;
- Linux selected performance freshness;
- local benchmark and sentinel reports;
- local generated API HTML;
- package-manager distribution;
- shared-library and dynamic ABI support;
- broad ecosystem or state-of-the-art parity.

Each row states current user status, primary user path, evidence owner, and
retained non-claims.

## Routing Updates

| File | Routing change |
| --- | --- |
| `README.md` | Added support/readiness to the adoption map, routed CI support interpretation to the matrix, and reduced repeated install/package caveats to a matrix link. |
| `docs/tutorial.md` | Routed support/readiness status to INSTALL while keeping the tutorial focused on local build-tree usage. |
| `docs/cookbook.md` | Routed support/readiness questions to INSTALL and linked package/platform/performance boundaries to the matrix. |
| `docs/solver_selection.md` | Routed install support and support/readiness status to the matrix while retaining solver-choice and diagnostics guidance. |
| `docs/api_reference.md` | Added a matrix link for installed package, generated API HTML, platform, package-manager, shared-library, and ABI boundaries. |
| `examples/README.md` | Routed installed-consumer support interpretation to the matrix. |
| `docs/maintainer_guide.md` | Added the user-facing support matrix to first-user starting points and clarified INSTALL's ownership. |

## Non-Claim Preservation

The implementation does not change support levels. It preserves these
non-claims in the new matrix:

- no Windows Makefile parity;
- no Windows `pkg-config` execution parity;
- no broad Windows parity;
- no broad Windows report freshness;
- no Windows selected oracle or benchmark freshness;
- no package-manager distribution support;
- no Homebrew/core, bottle, Linuxbrew, tap, vcpkg, Conan, pkgsrc, distro, or
  broad provider support;
- no shared-library support;
- no dynamic ABI support;
- no runtime-loader support;
- no broad external-library parity;
- no portable performance superiority;
- no hosted API publication;
- no broad ecosystem or unqualified state-of-the-art claim.

## Changed Surface

Day 4 changed Markdown documentation only:

- `INSTALL.md`
- `README.md`
- `docs/tutorial.md`
- `docs/cookbook.md`
- `docs/solver_selection.md`
- `docs/api_reference.md`
- `examples/README.md`
- `docs/maintainer_guide.md`
- `docs/planning/EPIC_17/SPRINT_194/WORKING_NOTES.md`
- this artifact

No public headers, source files, build scripts, CI workflows, selected target
manifest rows, package templates, install rules, or generated report behavior
were changed.

## Validation

Because support/package/Windows/performance wording changed, these checks were
run:

```sh
git diff --check
bash scripts/static_package_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
make windows-powershell-guard
python3 tests/test_selected_performance_docs.py
```

All commands passed. `make windows-powershell-guard` includes expected
negative-case output for missing or required PowerShell while validating its
test harness, but the overall target exited successfully.

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 4.
