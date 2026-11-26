# Sprint 228 Day 11: Reconstruction Memory Packet

## Outcome

The frozen four-workload packet proves the narrow Day 11 memory claim. The
maintained default route allocated **0 bytes in 0 allocations** for
reconstruction scratch in every invocation. The exact-`off` compatibility
route allocated one 2,097,152-byte `m*n` accumulator in every nonzero
invocation and no accumulator for the all-zero shortcut.

The counters deliberately keep bidiagonalization/decomposition work, retained
economy-SVD factors, sparse shell/maps, and output node slabs visible. They are
not subtracted from RSS and are not described as removed memory.

This development packet is **not promotion-accepted timing evidence**. Its
memory checks pass, but the frozen timing limits fail for the square and wide
cells, and collection used the documented dirty-worktree development override.
All noisy samples remain in the raw packet; no selective rerun was used.

## Frozen execution

- Workloads: square 512x512/rank 16, tall 1024x256/rank 8, wide
  256x1024/rank 24, and zero 512x512/rank 16.
- Routes: unset selector (maintained default) and exact `off` (dense
  compatibility).
- Two warmups per workload and route: 16 retained warmup invocations.
- Ten alternating measured pairs per workload: 80 retained measured
  invocations.
- Fresh process per invocation, `OMP_NUM_THREADS=1`, `LC_ALL=C`, 30-second
  invocation cap, and 15-minute collection cap.
- Collection elapsed time: 3.435 seconds.
- Source snapshot SHA-256:
  `f1f6cf6c5045da990b92eed8a5cf675dd4bfafefdc9e6e405b1a5736bc7955ba`.
- Binary SHA-256:
  `41f6a7ff40db996b8bab9925534373e9f8fcb69226720dee6d4b2a18c40eec0f`.
- Static library SHA-256:
  `abca061fbca054fa769eb20b6ce7bb5cef21dcbb18357563489e4a8fe23489bd`.
- Host/toolchain: macOS 15.7.9 x86_64; Apple Clang 17.0.0; 32-bit sparse
  indices; maintained C11 warning and optimization flags.

The build-local raw packet is retained at
`build/sprint228-day11-memory/`. It contains 291 files: metadata and summary,
16 warmup receipts with raw stdout/stderr, 80 measured receipts with raw
stdout/stderr, and a 290-entry SHA-256 manifest. The collector confirmed that
the source snapshot did not change during collection; that snapshot includes
the collector itself as well as the benchmark, build definitions, headers, and
library sources.
Superseded packets remain separately retained at
`build/sprint228-day11-memory-preformat/` and
`build/sprint228-day11-memory-precollector-hash/`; neither is selected as a
replacement trial.

## Component accounting

Values below are exact checked allocation extents from the executed route,
except process RSS, which is the contextual median fresh-process maximum. The
machine-readable table is in `day11-memory-component-accounting.tsv`.

| Workload | Route | Bidiag/decomposition peak | Retained factors | Reconstruction scratch | Shell/maps | Node slabs | Median peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| square | default | 4,222,952 B / 1,033 allocs | 4,198,400 B / 3 | 0 B / 0 | 16,520 B / 7 | 131,088 B / 1 | 9,408,512 B |
| square | dense | 4,222,952 B / 1,033 allocs | 4,198,400 B / 3 | 2,097,152 B / 1 | 16,520 B / 7 | 131,088 B / 1 | 9,465,856 B |
| tall | default | 4,214,760 B / 521 allocs | 2,623,488 B / 3 | 0 B / 0 | 20,616 B / 7 | 131,088 B / 1 | 6,209,536 B |
| tall | dense | 4,214,760 B / 521 allocs | 2,623,488 B / 3 | 2,097,152 B / 1 | 20,616 B / 7 | 131,088 B / 1 | 8,306,688 B |
| wide | default | 4,366,464 B / 532 allocs | 2,623,488 B / 3 | 0 B / 0 | 20,616 B / 7 | 131,088 B / 1 | 9,375,744 B |
| wide | dense | 4,366,464 B / 532 allocs | 2,623,488 B / 3 | 2,097,152 B / 1 | 20,616 B / 7 | 131,088 B / 1 | 9,400,320 B |
| zero | default | 4,222,952 B / 1,033 allocs | 4,198,400 B / 3 | 0 B / 0 | 16,520 B / 7 | 0 B / 0 | 9,392,128 B |
| zero | dense | 4,222,952 B / 1,033 allocs | 4,198,400 B / 3 | 0 B / 0 | 16,520 B / 7 | 0 B / 0 | 9,433,088 B |

The nonzero dense expectation is
`rows * columns * sizeof(double) = 262,144 * 8 = 2,097,152` bytes. The
observed count and extent equal that expectation in all 30 nonzero dense
measurements. All 40 default measurements report zero reconstruction-scratch
allocations. Both zero routes return before compatibility scratch allocation.

Retained-factor bytes reconcile to
`(k + rows*k + columns*k) * sizeof(double)` with `k = min(rows, columns)`.
Shell/map bytes reconcile to the matrix object, two header arrays, and four
identity maps. One 131,088-byte node slab remains visible for every nonzero
output; the zero control has no slab. Route pairs have identical output
checksum, `nnz`, and all non-scratch component counters.

## Timing and acceptance boundary

| Workload | Median default/dense | Maximum pair ratio | Frozen timing result |
| --- | ---: | ---: | --- |
| square | 1.3923 | 1.5675 | fail |
| tall | 1.2297 | 1.4312 | pass |
| wide | 1.3771 | 1.5764 | fail |
| zero control | 0.9466 | 1.1582 | contextual only |

The frozen nonzero gate requires median ratio at most 1.25 and every pair at
most 1.50. Therefore `timing_pass=false` and `protocol_accepted=false`.
Day 11 accepts only the exact reconstruction-memory result. A later clean,
exact-candidate collection must satisfy the timing gate before promotion-level
timing language is allowed.

## Verification

The focused suite reports 136 tests, 19,369 assertions, zero failures, and zero
skips. `test_s228_lowrank_memory_accounting_categories` independently checks
default, dense, and zero-route counter semantics on a small fixture. The raw
collector reports `memory_pass=true`, no correctness or accounting errors, and
unchanged source identity throughout collection. All 290 manifest entries
verify against their retained files.
