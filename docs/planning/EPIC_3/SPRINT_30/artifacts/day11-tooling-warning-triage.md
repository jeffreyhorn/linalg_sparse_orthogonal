# Sprint 30 Day 11 Benchmark And Example Warning Triage

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Turn the current `benchmarks/` and `examples/` warning backlog into an explicit Sprint 31 cleanup queue by separating portability issues, stale CLI or API drift, initializer-pattern drift, and incidental numeric-literal warnings.

## Inventory Summary

Using the Day 1 Apple Clang CMake baseline as input:

- total benchmark/example warnings: `14`
- benchmark warnings: `13`
- example warnings: `1`
- warning-bearing files: `6`

By warning class:

- `-Wmissing-field-initializers`: `10`
- `-Wimplicit-function-declaration`: `2`
- `-Wdouble-promotion`: `1`
- `-Wswitch`: `1`

Derived support files:

- `day11-tooling-warning-counts-by-file.txt`
- `day11-tooling-warning-counts-by-file-and-class.txt`
- `day11-tooling-warning-counts-by-class.txt`

Top warning-bearing files:

1. `benchmarks/bench_main.c`: `5`
2. `benchmarks/bench_colamd.c`: `3`
3. `benchmarks/bench_convergence.c`: `2`
4. `benchmarks/bench_ldlt_csc.c`: `2`
5. `benchmarks/bench_chol_csc.c`: `1`
6. `examples/example_colamd.c`: `1`

## Triage Categories

### Category 1: initializer-pattern drift

Dominant signal:

- `10` warnings
- all `-Wmissing-field-initializers`

Affected files:

- `benchmarks/bench_colamd.c`
- `benchmarks/bench_main.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `examples/example_colamd.c`

Representative drift:

- QR options initializers omit the new trailing progress-callback fields.
- Cholesky and LDLT benchmark options still use positional initialization against structs that have grown for backend selection and callback telemetry.
- The public COLAMD example uses the same positional QR options pattern that now warns in benchmark code.

Interpretation:

- This is maintainability drift, not a behavioral bug.
- The warnings track exactly the same public-options evolution issue already seen in the test suite.
- In benchmarks and examples, the problem is slightly worse than in tests because these files also teach downstream users what “normal” call-site style should look like.

Cleanup style implied by the warnings:

- convert touched benchmark/example options structs to designated initializers
- make intentionally-defaulted callback and telemetry fields explicit
- keep benchmark backend-selection intent visible at the call site

### Category 2: stale CLI and API-surface drift

Signal:

- `1` warning directly (`-Wswitch` in `benchmarks/bench_main.c`)
- several user-visible mismatches beyond the warning count

Affected file:

- `benchmarks/bench_main.c`

Observed drift:

- the file header usage text still advertises `--reorder rcm|amd|none`
- `reorder_name()` handles `NONE`, `RCM`, and `AMD`, but not `COLAMD` or `ND`
- CLI parsing still rejects `colamd` and `nd`

Cross-file mismatch:

- `benchmarks/bench_reorder.c` already benchmarks `RCM`, `AMD`, `COLAMD`, `ND`, and `NONE`
- `include/sparse_qr.h` documents `SPARSE_REORDER_COLAMD` and `SPARSE_REORDER_ND` as supported reorder choices

Interpretation:

- This is the highest-signal tooling issue in the current benchmark/example warning set.
- The warning itself is small, but the real problem is usability drift: the main benchmark harness misrepresents the reorder modes the library already supports.
- It is therefore both compile-hygiene debt and developer-tool documentation debt.

### Category 3: portability and feature-test-macro drift

Signal:

- `2` warnings
- both `-Wimplicit-function-declaration`

Affected files:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`

Observed pattern:

- both files define `_POSIX_C_SOURCE 199309L`
- both later call `snprintf`
- Apple Clang warns that the declaration is not visible under that feature-test setting

Interpretation:

- This is real portability debt in benchmark tooling, not just local compiler noise.
- The benchmark layer is part of the project’s engineering workflow, so its platform-facing headers should be held to the same hygiene standard as the main library.
- The likely fix is narrow: either raise feature-test handling to a level that exposes the needed declarations or restructure the portability setup so the standard declarations are visible without relying on an outdated macro setting.

### Category 4: incidental numeric-literal precision warning

Signal:

- `1` warning
- `-Wdouble-promotion`

Affected file:

- `benchmarks/bench_convergence.c`

Observed pattern:

- returning `NAN` from a helper that returns `double`

Interpretation:

- This is the lowest-priority warning in the benchmark/example slice.
- It is mechanical cleanup, not a stale-API or portability blocker.
- It should be handled after the `bench_main` and `_POSIX_C_SOURCE` issues, not before them.

## Documentation Drift Notes

Day 11 found two benchmark/example usability mismatches that matter even if the code still compiles:

1. `benchmarks/bench_main.c` documents and parses only `none`, `rcm`, and `amd`, while the library and other benchmark tooling already support `colamd` and `nd`.
2. `examples/example_colamd.c` still demonstrates positional QR options initialization even though the current public QR options struct has trailing callback fields and already warns under the normal CMake build.

Neither issue breaks the core library. Both weaken the reliability of the project’s developer-facing guidance.

## Sprint 31 First-Fix Subset

### Priority A: benchmark tool correctness and portability

Start with:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`

Reasons:

- `bench_main.c` combines the stale reorder CLI, stale usage text, one switch warning, and multiple initializer warnings in the project’s main benchmark harness
- `bench_convergence.c` carries the other benchmark portability warning around `snprintf`
- fixing these two files first removes the highest-signal usability and portability debt in the tooling layer

Expected outcome:

- benchmark CLI exposes the reorder modes the library actually supports
- `reorder_name()` becomes exhaustive for supported values
- feature-test/header handling no longer hides `snprintf`
- benchmark portability warnings in the main entry points are gone

### Priority B: designated-initializer sweep for benchmark and example options

Next cleanup group:

- `benchmarks/bench_colamd.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `examples/example_colamd.c`

Reasons:

- these files are mechanically simple and all express the same drift pattern
- they are visible examples of public options-struct usage
- once `bench_main.c` is corrected, this group finishes the obvious benchmark/example initializer debt without expanding scope

### Priority C: residual numeric-literal cleanup

After the higher-signal categories:

- `benchmarks/bench_convergence.c` `NAN` site

Reason:

- low ambiguity but low urgency
- best handled opportunistically while touching the file for the portability fix

## Day 11 Conclusion

Benchmark and example warnings are now split into distinct cleanup themes rather than treated as one generic backlog:

- the biggest cluster is initializer-pattern drift in public-facing tooling
- the most important user-facing issue is stale reorder CLI and usage drift in `bench_main.c`
- the most important portability issue is `snprintf` visibility under low `_POSIX_C_SOURCE` settings
- the remaining numeric-literal warning is mechanical follow-through work

That gives Sprint 31 a concrete tooling queue instead of a vague “clean up benchmark warnings” task.

## Validation

End-of-day verification passed:

- `make format`
- `make lint`
- `make test`
