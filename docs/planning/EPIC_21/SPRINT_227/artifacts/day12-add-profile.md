# Day 12 Selected Addition Construction Profile

## Scope and outcome

This is a same-host comparison of three selected `sparse_add` construction workloads. It is not
a whole-library, portable, allocator-independent, or state-of-the-art claim. The collection
followed the [Day 11 frozen protocol](day11-integrated-failure-matrix-and-scaling-protocol.md)
without changing its fixtures, ordering, warmups, repetition count, invalidation rules, or claim
boundary.

All 60 measured processes exited zero and passed the benchmark's complete A/B/C value, shape,
and `nnz` checks. The candidate was faster in every numbered pair for all three fixtures, and its
range did not overlap the baseline range. Whole-process memory ranges overlap; they do not show
an isolated or robust memory reduction.

## Exact source, build, and host receipt

| Field | Frozen value |
| --- | --- |
| Baseline source | Commit `fcdeaaee856821e8c801d805c63b5fd40eb89b2f`; `git archive` SHA-256 `4a6cc675e9006733d9f9e5ada1dc2e770b5d4f67a6dc5fb4687ec09ef91eabcd` |
| Candidate source | Sprint 227 working tree on the same HEAD; tracked binary diff SHA-256 `44907ac20ead8c24f231d8ffb14a1d64791d24c02213d7a7588191adfee4158e`; 93-file [source manifest](day12-source-sha256.txt) |
| Shared driver | `benchmarks/bench_add.c`; SHA-256 `623a7f5ba90fb4be37e3b432c154f5f6c38e8bf8dd25c4bae149d999455b8ed8` |
| Compiler | `/usr/bin/clang`; Apple clang 17.0.0 (`clang-1700.6.4.2`), x86_64 target |
| Library flags | `-std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2`, Command Line Tools SDK, `SPARSE_IDX_BITS=32`; static archive; OpenMP/sanitizers/coverage disabled |
| Driver link | Same flags and source bytes; each executable linked to its own archive with `-lm` |
| Width proof | Both generated stamps contain `32`; generated `sparse_version.h` files have the same SHA-256 `849d4acbd12131aec3f712299f5345fb6de8aac333fd769196f103dbc0f3e13d` |
| Baseline archive | SHA-256 `472eb51dcfcd50ddfef30dc37bcbaecc3e4d89657cef1bd96658967c43fe3d92` |
| Candidate archive | SHA-256 `3e42e54229bb5cdba7aaddcade7e8b06f625877dbe32d4ca45ce7be34eb328d3` |
| Baseline executable | SHA-256 `7e2f2e318d2c73753d18bbf184423a339175a2a8aa04702266dd3384c0a926d7` |
| Candidate executable | SHA-256 `516700b283d24997070eb2c2633b789fa8b37607ff9fd92d3edea0108ae090db` |
| Host | Intel Core i9-9980HK at 2.40 GHz; 8 physical/16 logical cores; 34,359,738,368 bytes memory |
| OS | macOS 15.7.9 build 24G830; Darwin 24.6.0 x86_64 |
| Power/environment | AC power, battery charging; serial collection; `OMP_NUM_THREADS=1`, `LC_ALL=C`, `TZ=UTC`; no frequency or thermal isolation claimed |
| Collection window | 2026-11-25 18:15:12.776050Z through 18:15:16.981996Z; 4.206 seconds |

The baseline source was extracted independently and the candidate used a distinct build tree.
No object, generated header, archive, or executable was shared. The same candidate driver source
was compiled against each static archive. The complete compile recipes are the Make-expanded
flags above followed by the driver compile/link command using the corresponding include, archive,
and output paths.

## Raw evidence and validation

The collector ran two correctness-qualified but unrecorded warmup processes for each fixture and
variant. It then ran ten independent-process pairs per fixture, baseline first in odd pairs and
candidate first in even pairs. Every invocation used a 30-second cap; the collection enforced a
15-minute aggregate cap and killed the process group on timeout. Neither cap fired, no pair was
replaced, and no sample was removed.

The [raw bundle](day12-raw/) contains exactly 300 immutable files: separate benchmark stdout,
benchmark stderr, `/usr/bin/time -l` output, status, and JSON invocation metadata for each of 60
processes. Metadata records argv, fixture, variant, pair, order, timestamps, wall duration, and
both caps. The [raw SHA-256 manifest](day12-raw-sha256.txt) verifies all 300 files. The
[per-process CSV](day12-samples.csv) contains every validated sample and is derived from, not a
replacement for, the raw bundle.

```sh
python3 scripts/summarize_add_profile.py \
  docs/planning/EPIC_21/SPRINT_227/artifacts/day12-raw \
  docs/planning/EPIC_21/SPRINT_227/artifacts/day12-samples.csv
shasum -a 256 -c \
  docs/planning/EPIC_21/SPRINT_227/artifacts/day12-raw-sha256.txt
```

Both commands exit zero. The CSV SHA-256 is
`dd36939fbdb2b7cfdaf9868819711c990564e67789a90eb0acd8ed1e3d8d6afb`; the raw manifest SHA-256
is `0cf43a479b00e187f94d816845fdc7b7e6879c7ff95c64dfb6d3c5f8881b4cb1`.

`elapsed_ms` covers exactly one `sparse_add(A, B, 1, 1, &C)` call. Fixture construction,
validation, and cleanup are outside that clock. RSS and peak footprint cover the whole process,
including setup and complete post-call validation. Quartiles use Python's inclusive method.

## Selected-kernel time

| Fixture (`nnz`, max output row) | Baseline median ms | Baseline min-max ms | Baseline IQR ms | Candidate median ms | Candidate min-max ms | Candidate IQR ms | Median-of-medians baseline/candidate | Median paired ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wide-row` (4096, 4096) | 13.4310 | 11.968-20.567 | 0.26975 | 0.1515 | 0.133-0.182 | 0.02450 | 88.653 | 88.289 |
| `varied-rows` (9216, 256) | 2.1550 | 2.098-2.697 | 0.14975 | 0.2470 | 0.220-0.313 | 0.02475 | 8.725 | 9.149 |
| `narrow-rows` (8192, 8) | 3.5605 | 3.413-3.864 | 0.16025 | 0.2190 | 0.193-0.450 | 0.05850 | 16.258 | 17.156 |

All ten paired ratios exceed 68 for `wide-row`, 7.31 for `varied-rows`, and 7.58 for
`narrow-rows`. The wide baseline has one 20.567 ms high observation and the narrow candidate has
one 0.450 ms high observation; both are retained. Even with those observations, every candidate
sample remains below every corresponding baseline range.

The selected profiles support a local statement that replacing repeated output insertion with
direct sorted row/column-tail linking substantially reduced this operation's elapsed time on all
three recorded structures. They do not isolate row width alone: the old insertion path scans row
and column lists, so the many-row control also reflects its column-chain and per-row behavior.
Output `nnz`, row width, and row count must remain attached to any interpretation.

## Whole-process memory context

| Fixture | Baseline median RSS bytes (range) | Candidate median RSS bytes (range) | Baseline median footprint bytes (range) | Candidate median footprint bytes (range) |
| --- | ---: | ---: | ---: | ---: |
| `wide-row` | 1,249,280 (1,200,128-1,265,664) | 1,261,568 (1,228,800-1,294,336) | 1,040,384 (991,232-1,056,768) | 1,056,768 (1,024,000-1,089,536) |
| `varied-rows` | 1,429,504 (1,400,832-1,486,848) | 1,417,216 (1,404,928-1,486,848) | 1,220,608 (1,191,936-1,277,952) | 1,212,416 (1,200,128-1,282,048) |
| `narrow-rows` | 1,357,824 (1,323,008-1,417,216) | 1,343,488 (1,302,528-1,396,736) | 1,148,928 (1,114,112-1,208,320) | 1,138,688 (1,097,728-1,191,936) |

Candidate RSS is lower/equal/higher in 4/0/6 wide-row pairs, 6/1/3 varied-row pairs, and 7/1/2
narrow-row pairs. Footprint direction is similarly mixed. The ranges overlap broadly and process
peaks include fixture construction and validation. This evidence does **not** establish reduced
peak memory or isolate the builder's column-tail workspace.

## Limitations and claim boundary

- Results apply only to the exact source pair, host, compiler, flags, and three workload IDs in
  this receipt.
- The run is one short local collection without independent replication, frequency locking,
  thermal telemetry, allocator instrumentation, or confidence intervals.
- Process startup, scheduling, caches, system load, and allocator state remain possible
  confounders despite alternating paired order.
- No result extends to in-place addition, matmul, other arithmetic, solvers, arbitrary sparsity,
  another platform, another compiler, or another library.
- No portable speed, memory-improvement, superiority, release, or state-of-the-art claim is
  authorized by this evidence.
