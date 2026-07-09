# Sprint 117 Day 4 Full Validation Design

## Purpose

Day 4 defines the validation lanes Sprint 117 should use before final Epic 10
claim calibration. The design separates reviewed proof from supplemental,
local-only, conditional, and staged lanes so Day 5 can execute validation
without widening package, platform, performance, or claim scope.

## Reviewed Baseline Decision

The strongest local reviewed baseline for Sprint 117 closeout is:

```sh
make quality-review-full
```

This target runs:

- `make quality-review`
  - `make format-check`
  - `make lint`
  - `make test`
  - `make deadcode-check`
- `make quality-review-cmake`
  - CMake configure
  - clean CMake build
  - `ctest -N`
  - Makefile/CMake test-count parity
  - full CTest execution

Use this as the default Day 5 execution lane if local tooling and runtime allow
it. If it fails, stop and investigate before proceeding. If local tooling is
unavailable, record the blocker explicitly rather than replacing the reviewed
lane with a weaker claim.

## Validation Command Matrix

| Surface / claim family | Command | Classification | Required when | Expected output |
|---|---|---|---|---|
| Documentation-only Sprint 117 artifacts | `git diff --check`; `rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_117` | Required docs hygiene | Always for Day 4 and other docs-only days | No diff whitespace errors; no trailing-whitespace matches. |
| Full local reviewed closeout baseline | `make quality-review-full` | Strongest local reviewed baseline | Day 5 default, especially before final closeout claims | Makefile reviewed path and CMake reviewed parity path pass. |
| C/header changes | `make format && make lint && make test` | Required by project rule | Any `.c` or `.h` file changes | Formatting applied; lint and full test suite pass. |
| Compile-quality only | `make quality-review-compile` | Reviewed compile-quality path | Source-list, formatting, or lint-focused validation; Linux/macOS reviewed compile lanes | `format-check`, `source-list-check`, and `lint` pass. |
| Makefile reviewed path | `make quality-review` | Reviewed local quality path | Local full Makefile proof without CMake rerun | `format-check`, `lint`, `test`, and `deadcode-check` pass. |
| CMake parity | `make quality-review-cmake` | Reviewed CMake parity path | CMake/test registration/build parity claims or final closeout | Configure, clean build, `ctest -N`, count parity, and CTest pass. |
| Source-list parity | `make source-list-check` | Required focused metadata check | Source files, build metadata, Makefile/CMake source lists, or proof-owner movement change | `scripts/check_library_sources.py` passes. |
| Make install/pkg-config | `bash tests/test_install.sh` | Local Unix-side package proof; supplemental unless promoted | Install/package docs, installed headers, `sparse.pc`, or Make install semantics change | Static library, public headers, pkg-config consumer, and uninstall checks pass. |
| CMake install/export | `bash tests/test_cmake_install.sh` | Local Unix-side package/export proof; supplemental unless promoted | CMake package config/export/version or `find_package(Sparse)` wording changes | CMake install, exported target, exact-version behavior, and consumer proof pass. |
| Benchmark compile surface | `make bench-build` | Supplemental compile confidence | Benchmark source or benchmark command surface changes | Benchmark binaries build. |
| Fast benchmark runtime | `make bench-fast` | Supplemental runtime signal | Benchmark/report wording or runtime signal changes | Fast benchmark subset and `bench_reorder --skip-factor` complete. |
| Canonical benchmark report | `make bench-canonical-report` | Supplemental local report artifact | Final comparison package needs refreshed benchmark report metadata | Threshold-free report generated under benchmark report directory. |
| Performance sentinels | `make performance-sentinels` | Supplemental local sentinel/report bundle | Performance sentinel wording or report evidence changes | Wall-check gate plus threshold-free sentinel bundle complete. |
| Large matrix guardrails | `make large-matrix-guardrails` | Supplemental/reviewed-by-script guardrail depending on mode | Reorder/graph/large-matrix evidence changes | Reviewed guardrails pass; supplemental reports remain non-claims. |
| Wall check | `make wall-check` | Reviewed macOS lane / local runtime gate | macOS-reviewed performance signal or wall-check wording changes | Existing wall-check script passes. |
| Sanitizer | `make sanitize`; `make asan` | Supplemental runtime confidence | Runtime-sensitive code changes or final supplemental validation if selected | UBSan/ASan instrumented tests pass after clean rebuild. |
| Coverage | `make coverage` | Supplemental coverage report | Coverage claims, coverage threshold, or final supplemental report changes | Coverage report generated and threshold check passes. |

## Platform Lane Map

| Platform | Reviewed lane | Supplemental lane | Staged / unsupported boundaries |
|---|---|---|---|
| Linux | `make quality-review-compile`; `make quality-review-cmake`; dead-code report/check completeness | `make test`, `make sanitize`, `make asan`, `make bench-fast`, TSan/OpenMP, `make coverage` | Local install scripts are developer-side package proof unless promoted to CI reviewed lanes. |
| macOS | Apple Clang `make quality-review-compile`; Apple Clang `make quality-review-cmake`; `make wall-check`; Apple Clang `make sanitize` | Homebrew GCC direct build/test/wall-check; Make install/`pkg-config` proof | No full reviewed install/export parity, no dead-code parity, no full coverage parity, no dynamic ABI claim. |
| Windows | MSVC CMake configure/build, `ctest -N`, expected CTest count `51`, full `ctest` | None promoted beyond the reviewed CMake subset | No Makefile parity, no separate install-validation lane, no thread/fuzz/property parity, no package-manager or dynamic ABI support. |

## Expected Outputs And Exclusions

| Area | Expected output | Exclusion / risk handling |
|---|---|---|
| `quality-review-full` | Both reviewed Makefile and CMake lanes pass. | If sanitizer, coverage, OMP, or other instrumented builds were run earlier, reset with `make clean` before returning to normal reviewed paths. |
| CMake parity | CMake and Makefile test counts match; full CTest passes. | A count mismatch is a blocker, not a wording-only issue. |
| Windows reviewed count | Workflow expects `51` registered CTest tests. | Windows staged exclusions remain `test_threads`, `test_sprint4_integration`, and `test_fuzz`; do not claim broader Windows parity. |
| Install scripts | Static archive, headers, pkg-config, CMake package files, and downstream consumers validate. | Passing local Unix scripts does not create reviewed Linux/macOS/Windows install parity by itself. |
| Benchmarks | Reports include local branch, command, compiler/platform, and artifact context where supported. | Benchmark rows are local measurement artifacts, not portable performance guarantees. |
| Coverage | Threshold check passes and HTML report is generated. | Coverage is supplemental evidence, not universal correctness or platform parity proof. |
| Source-list movement | Source-list parity passes and focused Make/CMake proof is captured. | Any source movement without parity and focused consumer proof remains deferred. |

## Day 5 Execution Checklist

Day 5 should execute in this order unless a blocker forces a stop:

1. Record environment basics:
   - branch;
   - current commit;
   - changed files;
   - whether `.c` or `.h` files changed.
2. Run docs hygiene:
   - `git diff --check`
   - `rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_117`
3. Run reviewed baseline if tooling and runtime allow:
   - `make quality-review-full`
4. If `make quality-review-full` fails:
   - stop;
   - record failing phase;
   - rerun only the failing subcommand if needed for diagnosis;
   - do not proceed to claim packaging until resolved.
5. Run conditional supplemental lanes only if the touched surface requires
   them:
   - package/install: `bash tests/test_install.sh` and
     `bash tests/test_cmake_install.sh`;
   - benchmark/report: `make bench-build`, `make bench-fast`,
     `make bench-canonical-report`, or `make performance-sentinels`;
   - source-list: `make source-list-check`;
   - coverage: `make coverage`;
   - sanitizer: `make sanitize` / `make asan`.
6. Record skipped supplemental lanes as intentionally skipped with the reason,
   not as passing reviewed proof.

## Day 6 Packaging Inputs

Day 6 should package:

- command;
- classification;
- result;
- runtime or blocker notes;
- claim family supported;
- surfaces validated;
- explicit exclusions and non-claims preserved.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| All required validation commands are known before execution. | Complete. |
| Command choices match touched surfaces and Epic 10 closeout needs. | Complete. |
| Staged exclusions and supplemental lanes are not mistaken for reviewed proof. | Complete. |
| Day 4 remains documentation-only. | Complete. |
