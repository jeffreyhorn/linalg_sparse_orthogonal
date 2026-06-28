# Sprint 93 Day 3: Reviewed Runtime Audit

## Purpose

Reduce Sprint 93's broad runtime-scalability and threading problem to one
ranked live contradiction map centered on the reviewed ND long pole, the
highest-value graph/reorder owners, and the remaining runtime-control and
proof concentration seams.

## Main Result

Sprint 93's broad runtime problem is now reduced to one ranked live
contradiction map:

- strongest first target:
  - the ND recursive driver and its graph-partition pipeline concentrated in
    `src/sparse_reorder_nd.c`, `src/sparse_graph.c`, and
    `src/sparse_graph_refine.c`
- strongest second target:
  - runtime-control and thread-local override complexity across the ND and
    graph pipeline, where tuning and profile hooks are real but now too
    diffuse to read as one clean runtime model
- strongest third target:
  - proof concentration in `tests/test_reorder_nd.c` and `tests/test_graph.c`,
    where major runtime truth still sits inside large single-binary owners
- strongest fourth target:
  - bounded benchmark and evidence follow-through so touched runtime changes
    remain measurable against the maintained reorder lane
- strongest support-only but real target:
  - public and maintainer wording that still needs to stay truthful about
    bounded threading maturity and reviewed runtime claims

## Strongest Current Contradiction

The strongest current contradiction is still the reviewed ND long pole:

- `src/sparse_reorder_nd.c` remains the recursive entry point and still owns
  threshold, profile, leaf, and recursion-side runtime behavior
- the current tree still centers the reviewed runtime hotspot on
  `test_reorder_nd`, not on broad library execution or on backend work
- `benchmarks/bench_reorder.c` already carries a bounded touched rerun lane
  (`--sprint86-slice`, `--nd-threshold`, `--skip-factor`) that matches this
  ownership story rather than widening it

That fixes the strongest first Sprint 93 move:

- the project does not most urgently need another backend, install, or public
  adoption pass
- it needs one clearer runtime-convergence move on the ND recursive and graph
  pipeline
- the current runtime issue is concentrated enough to target surgically, but
  still broad enough that it must be separated from generic threading claims

## Second-Tier Contradictions

### Runtime-Control Complexity

The strongest second contradiction is runtime-control complexity:

- `src/sparse_reorder_nd.c` still exposes profile env and override hooks
- `src/sparse_graph.c` and `src/sparse_graph_internal.h` still carry a growing
  set of thread-local FM / coarsening / separator override seams
- those controls are useful for diagnosis and bounded tuning, but they now
  read as a real cleanup target before the repo can claim a sharper
  runtime/threading model

This is real Sprint 93 work because a bounded runtime reduction is less
valuable if the touched lane still depends on a control story that is too
diffuse to interpret cleanly.

### Proof Concentration

The strongest third contradiction is proof concentration rather than proof
absence:

- `tests/test_reorder_nd.c` remains a giant owner containing fixture loading,
  runtime-control coverage, and major reviewed runtime proof
- `tests/test_graph.c` remains the adjacent giant owner for partition and FM
  behavior
- `tests/test_threads.c` and `tests/test_omp.c` provide concurrency proof, but
  they do not yet rebalance the reviewed-runtime cost concentration centered on
  the ND and graph review owners

This is real Sprint 93 work, but it reads after the first touched runtime seam
rather than before it.

### Benchmark and Evidence Follow-Through

The strongest fourth contradiction is evidence follow-through:

- `benchmarks/bench_reorder.c` is already the bounded runtime evidence owner
- `benchmarks/bench_amd_qg.c` and `benchmarks/bench_iterative_reuse.c` remain
  adjacent measurement surfaces, but they are clearly second-tier relative to
  the reviewed ND lane
- canonical reporting remains real, but it should not become the first
  implementation owner

This remains real Sprint 93 work, but it is explicitly later than the first
implementation seam.

## Fix-Now vs Deferred Split

The current tree now separates cleanly into:

### Contradictions that should drive Sprint 93 implementation

- ND recursive runtime seam
- runtime-control cleanup on touched ND/graph owners
- proof-surface rebalancing only where it reduces reviewed-runtime cost

### Contradictions that remain later or bounded non-claims for now

- fake broad multithreading maturity
- generic graph/reorder rewrite everywhere at once
- broad benchmark-superiority claims
- capability or packaging work outside the touched runtime lane

### Contradictions already materially bounded entering Sprint 93

- compressed-first product convergence
- portable dense/backend maturity package
- install/export contract sharpness
- front-door and support-surface layering

## Strongest Owner Surfaces

The highest-value owner surfaces tied to this audit are now explicit:

- runtime and reordering implementation owners:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_reorder_amd_qg.c`
- proof-owner tests:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `tests/test_threads.c`
  - `tests/test_omp.c`
- benchmark and runtime-evidence owners:
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
  - `benchmarks/bench_iterative_reuse.c`

## Interpretation

The useful Day 3 clarification is now explicit:

- Sprint 93 does not begin with another generic "add threads" pass
- it begins with one ranked runtime contradiction map
- the best first implementation center is the ND recursive runtime seam and
  its adjacent graph-partition owner surfaces
- runtime-control cleanup, proof-surface rebalancing, and bounded
  benchmark/reporting follow-through remain real Sprint 93 work, but they are
  explicitly sequenced behind that first center

## Exit State

- Sprint 93 now has one ranked live reviewed-runtime contradiction map
  grounded in the current post-Sprint-92 tree.
- The first Sprint 93 implementation center is fixed to the ND recursive
  runtime seam and its adjacent graph-partition owners.
- Day 4 can freeze the runtime/threading contract without reopening the ranked
  runtime order.
