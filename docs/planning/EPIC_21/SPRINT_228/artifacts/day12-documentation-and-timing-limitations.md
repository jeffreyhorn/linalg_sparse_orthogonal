# Sprint 228 Day 12: Documentation and Timing Limitations

## Decision

The Day 11 packet supports one comparative statement: on the four frozen
workloads, the maintained default reconstruction used zero additional `m*n`
scratch while the exact-`off` dense compatibility route used one 2,097,152-byte
accumulator on each nonzero workload. It does not support a whole-SVD
peak-memory reduction or a speed claim.

Promotion timing remains blocked. Square and wide both fail the frozen median
and individual-pair limits. The documentation therefore publishes the memory
ownership distinction and explicitly retains timing as unsuccessful local
development evidence.

## Pinned evidence

- Raw packet: `build/sprint228-day11-memory/`.
- Source snapshot SHA-256:
  `f1f6cf6c5045da990b92eed8a5cf675dd4bfafefdc9e6e405b1a5736bc7955ba`.
- Binary SHA-256:
  `41f6a7ff40db996b8bab9925534373e9f8fcb69226720dee6d4b2a18c40eec0f`.
- Static library SHA-256:
  `abca061fbca054fa769eb20b6ce7bb5cef21dcbb18357563489e4a8fe23489bd`.
- Protocol: two retained warmups per workload/route, ten alternating measured
  pairs, fresh process per invocation, 30-second invocation cap, 15-minute
  collection cap, serial thread context, and retained raw streams/receipts.
- Integrity: all 290 manifest entries verify; source remained unchanged during
  collection; collector reports no correctness or component-accounting error.
- Acceptance state: `memory_pass=true`, `timing_pass=false`, and
  `protocol_accepted=false` because timing failed and the dirty-tree development
  override was explicit.

The complete component table and formulas are in
`day11-reconstruction-memory-packet.md` and
`day11-memory-component-accounting.tsv`.

## Selected timing analysis

Ratios are default-route wall time divided by dense-compatibility wall time in
the same alternating pair. The frozen nonzero limits are median <= 1.25 and
every pair <= 1.50.

| Workload | Ten retained pair ratios | Median | Range | Pairs > 1.25 | Pairs > 1.50 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| square | 1.5190, 1.3712, 1.1678, 1.2543, 1.4306, 1.3086, 1.5675, 1.2358, 1.4626, 1.4133 | 1.3923 | 1.1678-1.5675 | 8/10 | 2/10 | fail median and pair limits |
| tall | 1.1461, 1.1879, 1.3672, 1.2492, 1.4312, 1.2077, 1.1819, 1.3007, 1.2758, 1.2102 | 1.2297 | 1.1461-1.4312 | 4/10 | 0/10 | pass |
| wide | 1.2818, 1.3078, 1.3522, 1.1912, 1.2421, 1.5764, 1.4087, 1.5530, 1.4116, 1.4021 | 1.3771 | 1.1912-1.5764 | 8/10 | 2/10 | fail median and pair limits |
| zero control | 1.1316, 0.9304, 0.8054, 1.1221, 0.9627, 1.0662, 0.8548, 1.1582, 0.8614, 0.9119 | 0.9466 | 0.8054-1.1582 | 0/10 | 0/10 | contextual control |

Square and wide each exceed 1.25 in eight of ten pairs, so their failed
medians are not attributable to one isolated outlier. Each also has two pairs
above 1.50. No selective rerun, outlier deletion, pooling with a superseded
packet, or threshold change is permitted. A future promotion attempt needs a
clean exact-candidate packet and must pass the already frozen limits.

## Memory interpretation

The supported ownership model is:

- default reconstruction scratch: scalar accumulation, zero additional
  `m*n` allocation;
- dense compatibility reconstruction scratch: one `m*n*sizeof(double)`
  allocation on nonzero reconstruction;
- retained factors: `(k + m*k + n*k) * sizeof(double)`, where
  `k = min(m,n)` for the economy SVD currently used by this API;
- decomposition: dense matrix and reflector/work allocations remain;
- sparse result: matrix shell/maps plus node slabs proportional to retained
  `nnz`, subject to slab granularity.

Consequently, "no additional reconstruction accumulator" is supported.
"Sparse-memory SVD", "lower whole-call peak memory", "always less memory",
and "faster" are not supported. A low drop threshold can retain nearly every
cell, making sparse output storage itself approach dense scale.

## Documentation reconciliation

| Surface | Day 12 wording owner | Evidence boundary |
| --- | --- | --- |
| `include/sparse_svd.h` | exact drop inequality/equality, default tolerance, selector, error statuses, and remaining storage | contract and source behavior; no timing number |
| `README.md` | user-facing default/compatibility route and narrow memory explanation | links to this retained evidence; rejects whole-SVD and speed inference |
| `benchmarks/README.md` | commands, JSON component meanings, collector caps, provisional override, and interpretation | raw packet and hashes; RSS contextual only |
| `docs/maintainer_guide.md` | maintainer selector boundary and forbidden claim language | memory pass retained; timing promotion blocked |

The quantitative statements in these surfaces are either formulas fixed by
the implementation contract or values copied from the pinned Day 11 packet.
No external-library ranking, portable performance, release, ABI, or broad SVD
claim is introduced.
