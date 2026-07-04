# Sprint 104 Day 12 Cross-Platform Runtime Review

## Purpose

Day 12 reviews backend/runtime behavior across local Make, local CMake, Linux
CI, macOS CI, Windows CI, serial builds, OpenMP builds, optional dense backend
requests, benchmark reporting, and performance sentinels. The goal is to make
platform-specific assumptions explicit before Sprint 104 closeout.

## Review Inputs

| surface | role reviewed |
|---|---|
| `Makefile` | local reviewed quality wrappers, OpenMP target, benchmark targets, canonical reports, sentinel target, wall-check |
| `CMakeLists.txt` | library source list, CMake test registration, optional OpenMP/mutex package metadata, Windows benchmark/test gating |
| `.github/workflows/ci.yml` | Linux enforced and supplemental runtime/benchmark lanes |
| `.github/workflows/macos-ci.yml` | macOS Apple Clang reviewed path, supplemental GCC, wall-check, sanitizer, install/pkg-config confidence lane |
| `.github/workflows/windows-ci.yml` | Windows reviewed CMake subset, expected CTest count, staged exclusions |
| `README.md` | top-level cross-platform CI contract and benchmark command summary |
| `docs/maintainer_guide.md` | reviewed baseline interpretation, packaging/platform scope, OpenMP/runtime-control model, benchmark governance |
| `benchmarks/README.md` | benchmark/sentinel report meaning and non-claims |
| `scripts/performance_sentinels.sh` | recorded runtime context and S5/S2 sentinel behavior |
| `scripts/bench_canonical_report.sh` | threshold-free canonical measurement metadata |

## Platform and Validation Mapping

| platform/surface | enforced or local command | current scope | interpretation |
|---|---|---|---|
| local strongest reviewed baseline | `make quality-review-full` | reviewed Makefile path plus reviewed CMake parity path | strongest local proof point when claiming the branch is inside the maintained reviewed baseline |
| local reviewed Makefile compile quality | `make quality-review-compile` | format check, source-list check, lint/tooling build | source/list/tooling drift gate; not a runtime test claim |
| local reviewed CMake parity | `make quality-review-cmake` | configure, clean rebuild, `ctest -N`, Makefile/CMake test-count parity, full CTest | shared Make/CMake parity proof on POSIX |
| local OpenMP | `make omp` or CMake `-DSPARSE_OPENMP=ON` | alternate OpenMP build and test path | opt-in build/runtime mode; serial remains default |
| local sentinel bundle | `make performance-sentinels` | S5 wall-check hard gate plus S2 threshold-free Cholesky CSC report rows | local regression evidence only; not portable timing evidence |
| Linux CI | `make quality-review-compile`, `make quality-review-cmake`, dead-code jobs | enforced reviewed Make/CMake/dead-code source of truth | strongest reviewed CI source |
| Linux supplemental CI | `make test`, sanitizer paths, `bench-fast`, TSan, coverage | runtime and supplemental quality signals | useful signals but not wider platform/product claims |
| macOS CI | Apple Clang reviewed Make/CMake path, wall-check, sanitizer | enforced Apple Clang reviewed path plus platform-specific wall-check/sanitize | narrower than Linux; Homebrew GCC and install/pkg-config are supplemental |
| Windows CI | CMake configure/build, `ctest -N`, full `ctest` with MSVC | reviewed CMake consumer subset with 51 registered tests | no reviewed Makefile parity, dead-code, bench, or install-validation claim |

## CMake and Test Count Notes

Local Day 12 configuration check:

- `cmake -S . -B build/day12-platform-review`: passed.
- `ctest -N --test-dir build/day12-platform-review`: reported
  `Total Tests: 54`.

The `ctest -N` listing was run after configure and before build, so CTest
printed missing-executable lookup warnings while still listing the registered
test surface. The count is still useful for registration review.

Current reviewed count interpretation:

- POSIX CMake registers 54 tests on this local machine.
- Windows CI expects 51 tests.
- The 3-test delta is intentional:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
- Windows CI documents those staged exclusions and fails if the reviewed CTest
  count drifts from 51.

## Runtime and Backend Coherence

| area | review result |
|---|---|
| builtin dense backend baseline | coherent: docs and tests continue to treat builtin as portable default |
| optional dense backend env vars | coherent: benchmark/sentinel docs record `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND` as context, not broad vendor claims |
| CMake OpenMP | coherent: `SPARSE_OPENMP` is opt-in, adds compile definitions and OpenMP link metadata |
| Make OpenMP | coherent: `make omp` owns the platform-specific OpenMP flags and reset-on-entry behavior |
| serial default | coherent: serial remains default across Make and CMake unless OpenMP is explicitly requested |
| nested parallelism | coherent: maintainer docs and algorithm docs warn against implying public nested-parallel runtime control |
| benchmark rows | coherent: canonical report now uses `category=measurement`; sentinel docs separate S5 hard gate from S2 report rows |
| Windows reviewed scope | coherent: CMake-first consumer proof only; no Makefile, fuzz/property, bench, or install-validation overclaim |

## Coherence Update Decision

No Day 12 source, workflow, or public documentation updates are needed.

Rationale:

- Day 11 already aligned the benchmark and sentinel wording gaps found on Day
  10.
- CMake configure passed locally.
- CTest registration count matches the expected POSIX surface of 54 tests.
- Windows CI already pins and explains the 51-test reviewed subset.
- OpenMP remains opt-in and is documented as runtime-controlled by the OpenMP
  runtime, not by a public library thread-control API.
- The benchmark/sentinel docs now distinguish local measurement, hard
  regression gates, optional backend context, and non-claims.

## Follow-Up List

| follow-up | priority | reason |
|---|---|---|
| Day 13 should run the final touched-file validation set | high | Day 12 is a review artifact; Sprint closeout still needs a final validation reconciliation |
| keep Windows expected CTest count tied to explicit staged exclusions | medium | prevents silent drift in reviewed Windows scope |
| avoid adding `performance-sentinels` to CI without a fresh variance decision | medium | the bundle is local evidence; CI timing variance would need its own baseline policy |
| do not widen optional dense backend wording beyond Cholesky/LDLT seams | medium | avoids broad vendor-backend claims unsupported by current runtime surface |
| keep OpenMP TSan interpretation narrow | medium | Ubuntu libomp suppressions are useful but not proof of every OpenMP/runtime combination |

## Completion Check

| criterion | status |
|---|---|
| workflow and build-system runtime surfaces reviewed | complete |
| CI and local validation mapping written | complete |
| Windows and serial-build scope notes captured | complete |
| platform-specific runtime assumptions documented | complete |
| mismatch list resolved with explicit no-change decision | complete |
| cross-platform follow-ups captured before closeout | complete |
