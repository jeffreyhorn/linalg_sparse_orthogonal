# Day 12: Selected Matmul Kernel Profile

**Scope:** Local selected-kernel comparison only. This is not a portable
speed claim, an all-kernel result, or a memory-improvement claim. The
[Day 11 protocol](day11-failure-closure-measurement-protocol.md) pins the
compiler, source and executable hashes, fixtures, timing boundary, warmup,
sample order, and host. Those hashes matched immediately before this run.

## Raw Evidence And Validation

The three pinned fixtures are `banded` (256 x 256 diagonal times 256 x 256
tridiagonal, 766 output nonzeros), `wide-row` (1 x 1024 full row times a
1024 x 1024 diagonal, 1024 nonzeros), and `rectangular` (64 x 32 dense
times 32 x 96 dense, 6144 nonzeros). The same
[`bench_matmul.c`](../../../../../benchmarks/bench_matmul.c) driver checks
the full expected value grid, dimensions, and `nnz` after the kernel timer
on both revisions. A failed value check exits nonzero and invalidates the
sample. No fixture dropped an expected output value at the threshold.

After two unrecorded warmup processes per fixture/variant, ten numbered
independent-process pairs were run for each fixture. Odd pairs ran baseline
then candidate; even pairs reversed order. All 60 measured commands exited
zero. The [raw directory](day12-raw/) contains exactly 180 original
`.stdout`, `.stderr`, and `.status` files, one set per process. The
[per-process CSV](day12-samples.csv) records all elapsed times, dimensions,
counts, RSS, footprint, pair order, and exit statuses; it is derived from,
not a replacement for, the raw logs. The
[SHA-256 manifest](day12-raw-sha256.txt) covers each raw file. Run
`python3 scripts/summarize_matmul_profile.py` with the raw directory and a
destination CSV to revalidate and reproduce the summary. The parser rejects
missing/extra files, nonzero statuses, changed fixture counts, invalid times,
or missing memory lines.

`elapsed_ms` is the driver's single `sparse_matmul` call timed with
`CLOCK_MONOTONIC`; input setup, output validation, and cleanup are outside
that interval. RSS and peak footprint come from `/usr/bin/time -l` and cover
the **entire process**, including setup and validation. The table below uses
the median of 10 samples, full min-max range, and inclusive-quartile IQR
(`statistics.quantiles(..., method="inclusive")`). The ratio divides the
two medians; paired delta is the median of baseline minus candidate within
each numbered pair. Neither is a confidence interval.

```sh
python3 scripts/summarize_matmul_profile.py docs/planning/EPIC_20/SPRINT_218/artifacts/day12-raw docs/planning/EPIC_20/SPRINT_218/artifacts/day12-samples.csv
shasum -a 256 -c docs/planning/EPIC_20/SPRINT_218/artifacts/day12-raw-sha256.txt
```

Both commands exited 0. The CSV SHA-256 after parsing is
`c3b30ae97bf8bc8babd847a239c07a53abefd319228e31cd23dffb81f31024fe`.

## Kernel Time

| Fixture | Baseline median ms | Baseline range ms | Baseline IQR ms | Candidate median ms | Candidate range ms | Candidate IQR ms | Median ratio baseline/candidate | Median paired delta ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `banded` | 0.0695 | 0.065-0.100 | 0.00525 | 0.0610 | 0.058-0.137 | 0.02275 | 1.139 | 0.0095 |
| `wide-row` | 1.3140 | 1.252-1.656 | 0.09525 | 0.1090 | 0.092-0.164 | 0.04775 | 12.055 | 1.2110 |
| `rectangular` | 3.9480 | 3.746-4.525 | 0.44200 | 1.1660 | 1.066-1.541 | 0.16800 | 3.386 | 2.7215 |

The wide-row and rectangular candidate ranges do not overlap the baseline
ranges in this local run; all ten paired candidate calls were faster for
each. The narrow banded ranges overlap, and the candidate was slower in
three of ten pairs. Its roughly 0.0085 ms difference between medians is
small relative to observed scheduling noise, so this run does not establish
a robust narrow-row improvement. The earlier Day 2 five-sample baseline
used a different driver revision and is **not** mixed into these ratios.

## Whole-Process Memory

| Fixture | Baseline median RSS bytes (range) | Candidate median RSS bytes (range) | Baseline median footprint bytes | Candidate median footprint bytes |
| --- | ---: | ---: | ---: | ---: |
| `banded` | 839680 (839680-876544) | 847872 (839680-905216) | 630784 | 638976 |
| `wide-row` | 942080 (897024-974848) | 942080 (905216-983040) | 733184 | 733184 |
| `rectangular` | 1138688 (1126400-1183744) | 1146880 (1093632-1191936) | 929792 | 937984 |

Candidate median RSS and footprint are 8192 bytes higher for `banded`
and `rectangular`, unchanged for `wide-row`. Candidate RSS is higher in
four, three, and five of ten respective pairs. These process peaks are
coarse, overlap broadly, and cannot isolate the matmul allocator. They
provide no evidence of reduced peak memory. The candidate adds a
`B->cols`-length column-tail workspace, so an output-size-linked workspace
cost remains even though linking is faster.

## Mechanism And Limits

The pre-change flush sorts touched columns, then calls `sparse_insert` for
each retained value. Repeated row-head scans can require `Theta(r^2)`
comparisons for a row with `r` retained entries; column-head scans add
cost on dense columns. The candidate keeps the same contribution traversal,
accumulator, drop predicate, and touched-column sort, but links each retained
node through row and column tails in O(1) after sorting. The 1024-entry
wide-row fixture is deliberately sensitive to that removed scan. The
rectangular dense-result fixture exercises both row and column links.

The remaining cost includes accumulation, sorting touched columns, checked
workspace allocation, node-slab allocation, and linked-structure traversal.
The selected shapes are small and do not establish asymptotic wall-clock
behavior or results for arbitrary sparsity, solvers, platforms, compiler
versions, or thread counts. Host frequency scaling, cold caches, process
scheduling, and allocator variability remain confounders despite paired
order. No kernel-only memory profiler was used. Day 13 owns evidence freeze
and further instrumented checks.
