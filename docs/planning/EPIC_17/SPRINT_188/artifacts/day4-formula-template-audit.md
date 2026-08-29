# Sprint 188 Day 4: Formula Template Audit

## Purpose

Audit `packaging/homebrew/sparse-lu-ortho.rb.in` against the Sprint 188
package proof boundary before proof-script hardening begins.

## Audit Summary

| Area | Result | Evidence |
| --- | --- | --- |
| Source-controlled template only | Pass | The file is `sparse-lu-ortho.rb.in`, not a committed installable `.rb` formula. |
| Required placeholders | Pass | Homepage, source archive URL, SHA-256, version, and Homebrew license placeholders are present. |
| License metadata injection | Pass | Formula license metadata is injected only through `__SPARSE_HOMEBREW_LICENSE__`. |
| Static-only install surface | Pass | The install step builds with CMake and checks for `lib/libsparse_lu_ortho.a`. |
| Shared artifact rejection | Pass | The install and test blocks reject `.dylib`, `.so`, `.so.*`, and `.dll` artifacts. |
| Downstream CMake consumer | Pass | `test do` writes a CMake consumer that uses exact-version `find_package(Sparse ...)` and links `Sparse::sparse_lu_ortho`. |
| Generated output hygiene | Pass | No rendered formula, archive, log, bottle, or local tap output was found under `packaging/homebrew`. |
| Provider claim boundary | Pass | The template comments state temporary local formula scope and warn against committing rendered formula, tap, archive, cache, bottle, or install outputs. |

## Placeholder Checklist

| Placeholder | Status | Purpose |
| --- | --- | --- |
| `__SPARSE_HOMEBREW_HOMEPAGE__` | Present | Inject local proof homepage metadata. |
| `__SPARSE_FORMULA_URL__` | Present | Inject temporary `file://` source archive URL. |
| `__SPARSE_FORMULA_SHA256__` | Present | Inject computed source archive checksum. |
| `__SPARSE_VERSION__` | Present | Inject version from `VERSION` and exact downstream CMake requirement. |
| `__SPARSE_HOMEBREW_LICENSE__` | Present | Inject approved Homebrew license identifier once approved root metadata exists. |

## Static Package Checklist

| Requirement | Audit result |
| --- | --- |
| Uses CMake build/install flow | Pass. The template configures, builds, and installs through CMake. |
| Installs the maintained static archive package surface | Pass. The template checks for `lib/libsparse_lu_ortho.a`. |
| Avoids shared-library selectors | Pass. The template does not set `BUILD_SHARED_LIBS=ON` or define static/shared package selectors. |
| Rejects shared artifacts | Pass. The template rejects `.dylib`, `.so`, `.so.*`, and `.dll` outputs in both install and test phases. |
| Keeps dynamic ABI out of scope | Pass. No dynamic ABI or runtime-loader support wording appears in the template. |

## Formula Test Review

The `test do` block remains appropriately narrow:

1. It creates a temporary downstream CMake project.
2. It requires C11 for the test project.
3. It uses `find_package(Sparse #{expected_version} EXACT REQUIRED)`.
4. It links `Sparse::sparse_lu_ortho`.
5. It compiles and runs a minimal executable using the installed public
   headers.
6. It checks the static archive, CMake config, and pkg-config metadata.
7. It rejects shared-library artifacts after test execution.

This proves an installed local static source formula consumer path only. It
does not prove Homebrew/core, bottles, Linuxbrew, public taps, binary packages,
other package managers, shared libraries, or dynamic ABI support.

## Template Corrections

No formula template corrections are required before Day 5 proof-script
render/archive hardening.

The current template should remain unchanged unless later proof runs reveal a
specific render, install, or `brew test` failure.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| Placeholder presence loop for all five required placeholders | Passed | Formula render inputs are still represented in the template. |
| `ruby -c packaging/homebrew/sparse-lu-ortho.rb.in` | Passed | The template parses as Ruby before rendering. |
| `rg` audit of static, shared, provider, and test markers | Passed | Expected static/test markers are present; unsupported claims are limited to comments that reject them. |
| Generated-output search under `packaging/homebrew` | Passed | No generated Homebrew proof outputs are present. |

## Day 5 Handoff

Day 5 can proceed to proof-script render/archive hardening from a clean
template baseline:

1. keep all five placeholders required;
2. fail clearly on missing or unresolved placeholders;
3. keep source archive creation temporary and cleanable;
4. preserve standalone license metadata inclusion once approved metadata
   exists; and
5. keep rendered formula, archives, logs, taps, caches, build trees, install
   prefixes, and bottle outputs out of source control.

## Validation Scope

Day 4 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
