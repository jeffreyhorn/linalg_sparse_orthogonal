# Day 2: Pinned Matmul Baseline

**Source revision:** `271bd3b704326296332cb397385ed458436ca0b7`.
Only the benchmark and its build registration were added; `sparse_matmul`
and the existing tests are unchanged. **Host:** Darwin 24.6.0 x86_64;
Apple clang 17.0.0 (`clang-1700.6.4.2`). **Mode:** dedicated CMake tree,
C11, `-O2`, project warnings, no sanitizer or OpenMP. The measurement is
local evidence, not a portable performance target.

## Fixture Manifest

The driver is [bench_matmul.c](../../../../../benchmarks/bench_matmul.c). All
values are deterministic positive coefficients; there is no random seed and
the expected threshold-drop count is zero. It validates input/output nnz and
returns nonzero if construction or multiplication fails.

| Fixture | A | B | A nnz | B nnz | Expected C nnz | Purpose |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `banded` | 256 x 256 diagonal | 256 x 256 tridiagonal | 256 | 766 | 766 | Narrow sparse rows; control for output size. |
| `wide-row` | 1 x 1024 full row | 1024 x 1024 diagonal | 1024 | 1024 | 1024 | One output row of 1024 retained entries; repeated row-head scans in `sparse_insert` sum to 523,776 comparisons in the sorted flush. |
| `rectangular` | 64 x 32 dense | 32 x 96 dense | 2048 | 3072 | 6144 | Rectangular full output; repeated row and column scans. |

## Reproduction

```sh
cmake -S . -B build/s218-baseline -DCMAKE_C_FLAGS=-O2 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/s218-baseline --target bench_matmul --parallel 4
for f in banded wide-row rectangular; do
  for i in 1 2 3 4 5; do
    printf '%s sample %s\n' "$f" "$i"
    /usr/bin/time -l build/s218-baseline/bench_matmul "$f"
  done
done
```

Configure and build both exited 0. The dedicated tree did not exist before
configuration, so its archive and benchmark executable were freshly built.
The timed interval is exactly one `sparse_matmul` call, after A/B setup and
before result validation/cleanup. Each sample is a new process. macOS
`/usr/bin/time -l` maximum resident set size is in bytes and covers the
**whole process**, including input construction; it is not an isolated
kernel allocation delta. The command's `peak memory footprint` is retained
as a second raw observation. All 15 sample commands exited 0.

## Raw Samples

Values below are transcribed from the benchmark CSV row and the two indicated
`/usr/bin/time -l` lines, in execution order. Time is kernel-only milliseconds;
RSS and footprint are bytes, whole-process peaks.

| Fixture | Sample | Elapsed ms | Max RSS | Peak footprint | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| banded | 1 | 0.391000 | 868352 | 659456 | 0 |
| banded | 2 | 0.084000 | 839680 | 630784 | 0 |
| banded | 3 | 0.138000 | 839680 | 630784 | 0 |
| banded | 4 | 0.175000 | 851968 | 643072 | 0 |
| banded | 5 | 0.138000 | 905216 | 696320 | 0 |
| wide-row | 1 | 1.588000 | 933888 | 724992 | 0 |
| wide-row | 2 | 1.912000 | 950272 | 741376 | 0 |
| wide-row | 3 | 8.069000 | 954368 | 745472 | 0 |
| wide-row | 4 | 1.896000 | 942080 | 733184 | 0 |
| wide-row | 5 | 1.879000 | 987136 | 778240 | 0 |
| rectangular | 1 | 13.284000 | 1179648 | 970752 | 0 |
| rectangular | 2 | 12.881000 | 1134592 | 925696 | 0 |
| rectangular | 3 | 6.328000 | 1204224 | 995328 | 0 |
| rectangular | 4 | 13.045000 | 1150976 | 942080 | 0 |
| rectangular | 5 | 11.412000 | 1126400 | 917504 | 0 |

Medians (elapsed only): banded 0.138 ms, wide-row 1.896 ms,
rectangular 12.881 ms. The respective ranges are 0.084-0.391,
1.588-8.069, and 6.328-13.284 ms. The wide-row outlier and the broad
rectangular spread argue against interpreting any single run as a speedup.
Retain the same build mode, fixture names, and independent-process sampling
on Days 11-12. At this scale, process startup and OS scheduling can dominate
the shortest measurement; the kernel timer excludes startup but not preemption.

## Pre-change Correctness

`make matmul-allocation-failure-gate` exited 0: registration passed; 18 C
tests passed, 0 failed/skipped, 185 assertions. It exercises the existing
three workspace-failure positions and recovery. The benchmark checks output
nnz for its fixed fixture, but does **not** replace the
independent value, ordering, and failure-position tests scheduled for later
days. The current semantic contract remains as documented in the
[Day 1 inventory](day1-matmul-contract-cost-inventory.md).
The fresh CMake tree separately built `test_matmul`; its selected CTest run
passed 1/1 tests.
`make format-check`, `make source-list-check`, `make lint`, and `make test`
also exited 0; the full Make suite ended with `All tests passed.`
