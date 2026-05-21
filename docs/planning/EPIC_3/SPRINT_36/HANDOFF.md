# Sprint 36 Handoff

**Source sprint:** 36  
**Prepared on:** Day 14  
**Purpose:** Convert Sprint 36's cross-platform parity work into explicit
starting constraints for Sprint 37, Sprint 38, and later Epic 3 quality-gate
expansion work.

## Starting State For Sprint 37

Sprint 36 does **not** hand off broken reviewed-quality flows, a stale
cross-platform contract, or undocumented CI/platform expectations.

Authoritative validated close state at Sprint 36 close:

- `make format`: passed
- `make lint`: passed
- `make test`: passed
- `make quality-review-compile`: passed
- `make quality-review`: passed
- `make quality-review-cmake-compile`: passed
- `make quality-review-cmake`: passed
- `make wall-check`: passed
- `make deadcode-report`: passed
- `make deadcode-check`: passed
- `make sanitize`: passed
- `ctest -N --test-dir build/quality-review-cmake`: `53` registered tests
- full reviewed CMake `ctest`: `53 / 53` passed

Validated timings captured on Day 13:

- `make lint`: `303.32 s`
- `make test`: `264.17 s`
- `make quality-review-compile`: `696.45 s`
- `make quality-review`: `487.59 s`
- `make quality-review-cmake-compile`: `93.11 s`
- `make quality-review-cmake`: `817.17 s`
- full reviewed CMake `ctest` real time: `703.03 s`

## Cross-Platform Quality Contract Now In Force

Sprint 36 did not create identical commands on all platforms. It made the
reviewed contract operationally truthful by platform.

### Linux

Enforced reviewed baseline:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

### macOS

Enforced Apple Clang reviewed/supporting baseline:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make wall-check`
- `make sanitize`

Supplemental/staged macOS surfaces:

- Homebrew GCC direct build/test leg remains supplemental
- dead-code remains staged rather than CI-enforced

### Windows

Enforced reviewed subset:

- reviewed CMake configure
- reviewed CMake build
- `ctest -N`
- full `ctest`

Still staged rather than enforced:

- named local Makefile reviewed-wrapper parity
- dead-code tooling/reporting

Named excluded Windows tests remain explicit:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

## Sprint 34 / Sprint 35 Baselines Still Preserved

Later Epic 3 work should preserve all of these:

- Sprint 34 reviewed Makefile wrappers still define the maintained local
  quality contract
- Sprint 34 reviewed CMake parity wrappers still define the maintained CMake
  parity contract
- Sprint 35 public-doc ownership split remains in force:
  - headers = authoritative API contract
  - `README.md` = concise entrypoint
  - `docs/tutorial.md` = fuller teaching surface
- active CTest registry remains `53` until intentionally changed
- `tests/test_framework_optin.c` remains live opt-in/skip policy coverage

## Highest-Value Shipped Sprint 36 Results

Sprint 36 closed the main phase-2 parity gap left by Sprint 34:

- Linux, macOS, and Windows workflows now use one consistent enforced/staged/
  supplemental vocabulary
- macOS CI now expresses Apple Clang as the explicit reviewed baseline instead
  of only older direct commands
- Windows CI now states and checks its enforced reviewed CMake subset directly,
  including `ctest -N` and the staged count expectation
- the parity report now makes the actual state explicit:
  - reviewed CMake parity is the only fully honest cross-platform reviewed
    baseline today
  - reviewed Makefile wrappers remain Linux/macOS-enforced and Windows-staged
  - dead-code remains Linux-enforced, macOS-staged, Windows-excluded
- reviewed Makefile portability improved in the maintained path by removing
  avoidable `find` and hardcoded `/bin/*` assumptions

## Residual Deferred Queue

Sprint 36 hands off a **bounded staged-parity queue**, not a regression queue.

### Priority A: Windows local reviewed-wrapper parity remains staged

Sprint 36 made the Windows reviewed subset explicit, but did not claim full
local Makefile reviewed-wrapper parity on Windows.

Carried forward:

- local `quality-review-compile` parity on Windows remains staged
- local `quality-review` parity on Windows remains staged

### Priority B: dead-code remains intentionally non-universal

Sprint 36 preserved the truthful dead-code limits instead of overstating them.

Carried forward:

- macOS dead-code remains staged
- Windows dead-code remains excluded
- compile-db coverage gap still persists from Sprint 34:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- shared-path execution model still persists:
  - `build/deadcode-cmake`
  - `build/deadcode/`

### Priority C: sanitizer/build-tree interaction is now an explicit maintainer caveat

Day 13 exposed one operational limitation:

- a prior `make sanitize` run can leave a sanitizer-instrumented `build/` tree
  behind
- a later direct `make lint` run may then fail at benchmark link time unless
  the tree is cleaned first

Carried forward:

- authoritative direct/reviewed validation sweeps should start from a clean
  `build/` tree if a sanitizer path ran immediately before them

## Suggested First-Fix Queue For Sprint 37+

Immediate later-sprint emphasis belongs here:

- Sprint 37:
  - fold the Sprint 36 sanitizer/build-tree caveat into maintainer workflow
    guidance and target normalization
  - keep quality-target naming/reporting consistent while avoiding fake Windows
    Makefile parity claims
- Sprint 38:
  - address the dead-code compile-db exclusion list
  - address dead-code shared-path isolation
  - expand gates only where the staged/enforced boundaries remain truthful

## Reproduction Commands

Use these commands before and after later Epic 3 parity/gate work:

1. `make format`
2. `make lint`
3. `make test`
4. `make quality-review-compile`
5. `make quality-review`
6. `make quality-review-cmake-compile`
7. `make quality-review-cmake`
8. `make wall-check`
9. `make deadcode-report`
10. `make deadcode-check`

If a sanitizer path ran immediately before the direct sweep:

11. `make clean`

Expected stable comparison targets at Sprint 36 close:

- `53` registered CTest tests
- full reviewed CMake `ctest`: `53 / 53` passing
- Linux reviewed dead-code path: green
- macOS/Windows workflow wording: aligned to enforced/staged/supplemental
  contract

## Key References

- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day10-cross-platform-parity-report.md](./artifacts/day10-cross-platform-parity-report.md)
- [day11-final-parity-consistency-pass.md](./artifacts/day11-final-parity-consistency-pass.md)
- [day12-platform-focused-validation.md](./artifacts/day12-platform-focused-validation.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
