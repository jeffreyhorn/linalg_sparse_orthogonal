# Sprint 207 Day 10 User Package Docs

## Purpose

Day 10 updated user-facing package documentation to match the Sprint 207
continued-deferral decision. The docs now distinguish supported source install
paths from the bounded developer-mode local Homebrew proof and retained
package-provider non-claims.

## Changed User Surfaces

| Surface | Day 10 update |
| --- | --- |
| `README.md` | Clarifies that Sprint 207 selected continued package-provider deferral and that the Homebrew proof is developer-mode local static source formula proof only. |
| `INSTALL.md` | Updates package-manager readiness wording and the support matrix row for package-manager distribution. |
| `packaging/homebrew/README.md` | Clarifies that the formula template remains proof-only and is not a Homebrew install method, public tap formula, Homebrew/core readiness artifact, bottle, Linuxbrew path, binary package, or release package. |
| `docs/planning/EPIC_19/SPRINT_207/WORKING_NOTES.md` | Records Day 10 validation and item 207.5 user-documentation progress. |

## User-Facing Status After Day 10

| Topic | User-facing status |
| --- | --- |
| Source install | Supported through the existing Make and CMake source install guidance. |
| Homebrew proof | Developer-mode local static source formula proof only. |
| User-facing Homebrew install | Not claimed. |
| Public tap or Homebrew/core | Not claimed. |
| Bottles or Linuxbrew | Not claimed. |
| vcpkg, Conan, pkgsrc, distro/system packages | Not claimed. |
| Binary packages, release packages, package-manager release readiness | Not claimed. |
| Shared-library packages or dynamic ABI support | Not claimed. |

## Validation

| Command | Result |
| --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Passed. |
| `make docs-check` | Passed. |
| `python3 tests/test_package_manager_deferral_guard.py` | Passed. |
| `python3 -m py_compile tests/test_package_manager_deferral_guard.py` | Passed. |
| `bash -n scripts/package_manager_deferral_check.sh` | Passed. |
| `bash scripts/package_manager_deferral_check.sh` | Passed, including embedded local Homebrew proof boundary checks. |
| `brew list --formula | rg '^sparse-lu-ortho-local$' || true` | No installed proof formula remained. |
| `brew tap | rg '^sparse-lu-ortho/local-proof-' || true` | No temporary proof tap remained. |
| `find packaging/homebrew -maxdepth 3 ...` | No generated archive, log, formula, bottle, or `Formula/` output was present. |

## Completion Criteria

- Item 207.5 has user-facing documentation implementation.
- Users can tell source builds are the supported install path.
- Users can tell Homebrew evidence is local proof only.
- Public docs do not imply Homebrew/core, bottles, Linuxbrew, ABI, release, or
  broad package-manager support.
