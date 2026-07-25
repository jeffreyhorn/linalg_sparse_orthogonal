# Sprint 133 Day 13 - Integrated Package Validation

## Purpose

Day 13 runs the integrated validation path for Sprint 133 package, install,
consumer, and static-first deferral surfaces. The goal is to prove that the
selected static archive package contract remains coherent across Make install,
CMake install/export, `pkg-config`, downstream consumers, and documentation
claim guards.

## Touched Surface Review

| Surface | Touched in Sprint 133 | Day 13 validation owner |
| --- | --- | --- |
| CMake package contract | Yes; `BUILD_SHARED_LIBS=ON` rejection and install/export proof checks. | `tests/test_cmake_install.sh` and `scripts/static_package_deferral_check.sh`. |
| Make install and `pkg-config` consumer proof | Yes; stricter `.pc` field and downstream checks. | `tests/test_install.sh`. |
| Package support documentation | Yes; static-first support and deferral wording. | `scripts/static_package_deferral_check.sh`, `git diff --check`, and whitespace scan. |
| Shell validation scripts | Yes; package/install proof scripts. | `bash -n` syntax checks plus focused execution. |
| C source and public headers | No. | Full `make format && make lint && make test` gate not required by Sprint instruction. |

## Validation Results

| Check | Result | Evidence |
| --- | --- | --- |
| Shell syntax | Pass | `bash -n tests/test_install.sh && bash -n tests/test_cmake_install.sh && bash -n scripts/static_package_deferral_check.sh`. |
| Touched C/header scan | Pass | `git diff --name-only -- '*.c' '*.h'` returned no paths. |
| Make install and pkg-config proof | Pass | `bash tests/test_install.sh` passed 22 checks, 0 failures. |
| CMake install/export proof | Pass | `bash tests/test_cmake_install.sh` passed 21 checks, 0 failures, 0 skips. |
| Static-first deferral proof | Pass | `bash scripts/static_package_deferral_check.sh` passed all deferral checks. |
| Patch whitespace | Pass | `git diff --check`. |
| Sprint/package whitespace scan | Pass | `rg -n "[[:blank:]]$"` over touched package, script, docs, and Sprint 133 paths returned no matches. |

## Make install and pkg-config Evidence

Successful focused run:

```text
--- Checking installed files ---
  [PASS] static library installed
  [PASS] no shared-library artifacts installed
  [PASS] all 19 headers installed
  [PASS] pkg-config file installed
  [PASS] pkg-config can resolve sparse
  [PASS] pkg-config exact version constraint works
  [PASS] pkg-config prefix points at install prefix
  [PASS] pkg-config libdir points at installed libdir
  [PASS] pkg-config includedir points at installed includedir
  [PASS] pkg-config --cflags returns installed include path
  [PASS] pkg-config --libs returns installed static archive link flags
  [PASS] pkg-config --static libs match current self-contained link flags
  [PASS] pkg-config file has no private dependency stanza
  [PASS] pkg-config file has no unsupported packaging or ABI claims
--- Summary ---
Passed: 22
Failed: 0
ALL INSTALL TESTS PASSED
```

This covers Make install/uninstall, static archive install shape, no shared
artifacts, exact header inventory, `sparse.pc` field semantics, downstream
compile/link/run consumers, and uninstall cleanup.

## CMake install/export Evidence

Successful focused run:

```text
--- Checking installed CMake package metadata ---
  [PASS] CMake imported target is static
  [PASS] CMake imported target uses install include prefix
  [PASS] CMake imported archive uses install prefix
  [PASS] CMake package has no source-tree paths
  [PASS] CMake package has no build-tree paths
--- Summary ---
Passed: 21
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

This covers CMake configure/build/install, static archive install shape, no
shared artifacts, exact header inventory, CMake package files, installed
static imported-target metadata, source/build-tree path leak scans,
`find_package(Sparse)` consumer configure/build/run, exact-version package
resolution, mismatched-version rejection, and installed pkg-config version.

## Static-first Deferral Evidence

Successful focused run:

```text
static-package-deferral-check: BUILD_SHARED_LIBS rejection ok
static-package-deferral-check: static target declaration ok
static-package-deferral-check: no shared export/ABI metadata found ok
static-package-deferral-check: package metadata has no static/shared selector ok
static-package-deferral-check: support wording remains deferred ok
static-package-deferral-check: passed
```

This covers configure-time rejection of shared-library requests, explicit
static CMake target declaration, absence of unsupported public export/import
macros or shared ABI metadata/selectors, and bounded support wording.

## Quality Gate Interpretation

No `.c` or `.h` files changed in the tracked Sprint 133 diff at Day 13. Under
the Sprint instruction, `make format && make lint && make test` is required
only when code files are modified, so the full C quality gate was not run.

The relevant package, build-system, shell-script, documentation, install, and
consumer surfaces all have matching focused validation evidence.

## Residual Validation Queue

| Item | Status |
| --- | --- |
| Reviewed CI promotion for full install proofs | Deferred. Current install proofs remain local unless future work promotes them. |
| Shared-library and dynamic ABI validation | Deferred by product decision. Static-first deferral guard remains the active proof. |
| Package-manager validation | Deferred. No package-manager support is claimed by Sprint 133. |
| Optional thread/OpenMP package flag matrix | Deferred. Day 13 validates the default installed package contract. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every touched package surface has matching validation evidence. | Complete | Make/pkg-config, CMake install/export, static deferral, syntax, and hygiene checks passed. |
| Required quality gates pass or blockers are explicit. | Complete | No `.c`/`.h` diff, so full C gate was not required; focused gates passed. |
| Validation evidence is ready for closeout and PR review. | Complete | This artifact records commands, results, support boundaries, and residual validation queue. |
