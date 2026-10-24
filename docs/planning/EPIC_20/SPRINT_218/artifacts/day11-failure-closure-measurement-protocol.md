# Day 11: Failure Closure And Measurement Protocol

**State:** Item 218.4's allocation-failure lifecycle is closed for the
explicit matmul allocation sites. The baseline and candidate libraries have
been rebuilt in separate, uninstrumented CMake trees, and the same benchmark
driver has been compiled against each. Day 12 owns the paired measurement
samples and their analysis; the six Day 11 smoke runs below are not a
performance result.

## Allocation-Failure Signoff

| Fail-after position | Site | Registered coverage and rollback |
| --- | --- | --- |
| 0 | Result shell | `test_matmul_result_shell_and_first_node_allocation_failures` checks NULL output, stale ownership, unchanged inputs, and retry. |
| 1-6 | Six result shell arrays | Same test, once per array; partial shell buffers are freed by constructor cleanup. |
| 7-10 | Accumulator, flags, touched list, column tails | Workspace and stale-output tests cover each site and an immediate retry. |
| 11 | First node slab | Result-allocation test covers failure before the first link. |
| 12 and later | Subsequent node slabs | Day 10 sweep fails after partial row/column links, retries every failure, and checks output values and traversal. |

No per-node heap allocation occurs inside a slab. The one-node-slab build
turns every subsequent node request into an injectable allocation; its
eight later positions include mid-row and row-transition failures. Every
source-level explicit allocation in the matmul output path is covered by
the checked fault hook, including the private output shell and slabs. The
single matmul failure cleanup frees workspaces and private output; only the
success path publishes `*C`. The `IDX_MAX` guard is not an allocation site
and is outside this fault-injection map. This signoff does not claim control
over allocator activity internal to system `qsort` or other libraries.

`make matmul-allocation-failure-gate` exited 0 on Day 11: registration
passed, 41 tests passed, 0 failed/skipped, 372262 assertions. Day 10's
one-node-slab ASan/UBSan run passed, and native macOS `leaks` runs on the
default and one-node-slab binaries each reported zero leaks and zero leaked
bytes. The [Day 9 map](day9-allocation-site-failure-map.md) and
[Day 10 instrumentation record](day10-rollback-retry-sanitizer.md) hold
the fixture-level evidence.

## Pinned Builds

| Attribute | Baseline | Candidate |
| --- | --- | --- |
| Source | Detached worktree at `271bd3b704326296332cb397385ed458436ca0b7`, clean | `sprint-218` working tree at the same HEAD, with uncommitted Sprint 218 changes |
| Source directory | `build/s218-day11-baseline-src` | Repository root |
| Build directory | `build/s218-day11-baseline` | `build/s218-day11-candidate` |
| `src/sparse_matrix.c` SHA-256 | `d194f0e72a6fa280a1d9d4519f09d95d0f84fbd50d4e3dfdd75e83a6d6e0ffc2` | `6da7faf6298260d6708401ce17af351e22824fdd4774c7660fe2ad9530f2e1dc` |
| Benchmark executable SHA-256 | `fb8df0a351faa52991ebbae908d7f8176ad893385e9f084f460ce19aa6a4ad33` | `d65c68c52d7482ae3c85a0d06be784a47900a6fd91640ae34e40f8287c4a024c` |

Both CMake caches specify `/usr/bin/clang`, `CMAKE_C_FLAGS=-O2`, an empty
`SANITIZE`, and `SPARSE_OPENMP=OFF`. The `src/sparse_matrix.c`
`compile_commands.json` entries contain identical `-O2 -std=c11` and
project warning flags, apart from source/include/build paths. Both library
targets were built fresh. The benchmark driver was the **same file at
measurement time** for both binaries, SHA-256
`0995357c0f272bfb94ac6150d48f434307e9ec4f77b6368ad1794a8c3b93b3cb`.
The baseline commit predates the benchmark; compiling this one driver
against each archive avoids comparing different harness revisions. Its
post-timer validation now checks every expected value, dimensions, and
`nnz`; setup and validation are outside the kernel timer for both binaries.

Rebuild using the recorded commands, then use the same direct compiler
flags for each driver binary:

```sh
git worktree add --detach build/s218-day11-baseline-src 271bd3b704326296332cb397385ed458436ca0b7
cmake -S build/s218-day11-baseline-src -B build/s218-day11-baseline -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_C_FLAGS=-O2 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake -S . -B build/s218-day11-candidate -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_C_FLAGS=-O2 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/s218-day11-baseline --target sparse_lu_ortho --parallel 4
cmake --build build/s218-day11-candidate --target sparse_lu_ortho --parallel 4
/usr/bin/clang -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 -Ibuild/s218-day11-baseline/include -Ibuild/s218-day11-baseline-src/include benchmarks/bench_matmul.c build/s218-day11-baseline/libsparse_lu_ortho.a -lm -o build/s218-day11-baseline/bench_matmul
/usr/bin/clang -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2 -Ibuild/s218-day11-candidate/include -Iinclude benchmarks/bench_matmul.c build/s218-day11-candidate/libsparse_lu_ortho.a -lm -o build/s218-day11-candidate/bench_matmul
```

The first `git worktree add` command is only for a fresh checkout; reuse
the existing clean baseline worktree when it is already present. If source
or driver hashes change before Day 12, rebuild both and record new hashes
before comparing. Do not reuse Make's `build/libsparse_lu_ortho.a` or an
ASan/UBSan tree for either side.

Post-PR portability note (2026-10-24): the commands above record the original
Day 11 build, not a new measurement recipe against the latest branch tip.
The pinned benchmark source is available at
`git show 806b31ed3efc9959bf585c34a8d4000171c796c5:benchmarks/bench_matmul.c`.
The post-portability-fix source includes a pre-include POSIX feature macro for
strict C11 Linux builds and has SHA-256
`61d5bf2cd0896dc4b9f6b5507195781f5a72be20f4891370e3d683dc3806aa61`.
The later PR #242 review fix added a no-argument mode that runs all three
fixtures; its source SHA-256 is
`11e2321d73bc662f5ba5b81b9ed05662dd0cd34541fbecb74f12496147508877`.
Neither change alters single-fixture timing or validation logic, but
rebuilding from either source produces different provenance. Do not combine
a new binary with the frozen Day 12 samples; a new comparison needs both
variants rebuilt and new hashes and raw samples.

## Day 12 Sampling Contract

The [Day 2 fixture manifest](day2-matmul-baseline.md) fixes `banded`,
`wide-row`, and `rectangular` dimensions, input/output `nnz`, and numeric
coefficients. Inputs are constructed before the timer. The driver records
one `sparse_matmul` call with `CLOCK_MONOTONIC`; it validates expected values
and frees output afterward. A nonzero exit invalidates that sample and its
pair. Both variants must use this exact driver source and fixture names.

1. Set `LC_ALL=C` and run two **unrecorded warmup processes** for each
   fixture/variant combination. A failed warmup stops the comparison.
2. Run ten numbered paired rounds per fixture, each invocation in a fresh
   process. Alternate order: odd rounds baseline then candidate; even
   rounds candidate then baseline. Keep all rounds, including outliers;
   never replace a slow observation selectively.
3. Wrap every measured invocation with `/usr/bin/time -l`. Retain its
   original stdout and stderr separately, plus the exit status, under
   `docs/planning/EPIC_20/SPRINT_218/artifacts/day12-raw/` using names
   `<fixture>-<round>-<variant>.stdout`, `.stderr`, and `.status`. Preserve
   the benchmark CSV header and row, the `maximum resident set size`, and
   `peak memory footprint` lines. These files are the raw record; any Day 12
   summary table must be derived from them.
4. Before calculating a ratio, require 10 successful pairs per fixture,
   identical expected dimensions and `nnz`, and the driver's full value
   validation on every run. Report median elapsed time, min/max, and IQR
   for each variant and shape; show all raw samples and the paired median
   differences. Report RSS and footprint separately as whole-process peaks,
   not matmul-only allocation costs. Disclose regressions and outliers.

From the repository root, the following Bash loop implements the warmup,
alternating paired order, and raw capture. Start with a fresh `day12-raw`
directory; use a new directory name for any repeat run rather than
overwriting an earlier raw record.

```sh
export LC_ALL=C
raw=docs/planning/EPIC_20/SPRINT_218/artifacts/day12-raw
mkdir -p "$raw"
for fixture in banded wide-row rectangular; do
  for variant in baseline candidate; do
    for warmup in 1 2; do
      "build/s218-day11-$variant/bench_matmul" "$fixture" >/dev/null || exit 1
    done
  done
  for round in {1..10}; do
    if (( round % 2 )); then order='baseline candidate'; else order='candidate baseline'; fi
    for variant in $order; do
      sample="$raw/$fixture-$round-$variant"
      if [[ -e "$sample.stdout" || -e "$sample.stderr" || -e "$sample.status" ]]; then
        printf 'existing sample: %s\n' "$sample" >&2
        exit 1
      fi
      /usr/bin/time -l "build/s218-day11-$variant/bench_matmul" "$fixture" \
        >"$sample.stdout" 2>"$sample.stderr"
      status=$?
      printf '%s\n' "$status" >"$sample.status"
      if (( status )); then exit "$status"; fi
    done
  done
done
```

The six Day 11 smoke invocations (one per variant/fixture) all exited 0 and
reported the pinned output `nnz`. Their elapsed times were not retained as
Day 12 samples, because they were not paired, warmed, or memory-instrumented.

## Quality Gates

| Command | Result |
| --- | --- |
| `make matmul-allocation-failure-gate` | Exit 0; registration passed, 41 tests passed, 372262 assertions. |
| `make format-check` and `make source-list-check` | Both exit 0; 49 library sources. |
| `make lint` | Exit 0; benchmark tooling build, strict compile, clang-tidy, and cppcheck completed. |
| `make test` | Exit 0; final output `All tests passed.` |

## Host And Confounders

Captured on 2026-10-23: Darwin 24.6.0, x86_64; Intel Core i9-9980HK
2.40 GHz; 32 GiB RAM; Apple clang 17.0.0 (`clang-1700.6.4.2`);
Command Line Tools SDK at
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`. This is a shared
desktop host, with no CPU pinning or thermal control. Process startup is
outside the driver timer but scheduling, cold caches, allocator behavior,
frequency scaling, and preemption can affect sub-millisecond samples.
`/usr/bin/time -l` memory peaks include input construction, validation,
and process runtime, not just the multiply. The old Day 2 baseline samples
used the pre-validation driver and are context only; Day 12 ratios must use
the paired binaries and protocol above. No portable speed or memory claim
follows from this setup.
