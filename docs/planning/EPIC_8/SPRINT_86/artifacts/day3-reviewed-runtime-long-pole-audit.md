# Sprint 86 Day 3: Reviewed Runtime Long-Pole Audit

## Purpose

Reduce Sprint 86's broad reviewed-runtime problem to one ranked live cause map
so the sprint can choose one bounded ND/reorder runtime lane instead of
another generic performance bucket.

## Main Result

Sprint 86's broad reviewed-runtime problem is now reduced to one ranked live
cause map:

- strongest first target:
  - bounded ND runtime reduction centered on `tests/test_reorder_nd.c`,
    `src/sparse_reorder_nd.c`, and the multilevel graph pipeline it drives
- strongest second target:
  - proof-surface concentration rebalancing across `tests/test_reorder_nd.c`
    and adjacent reorder/graph proof owners where repeated heavy fixture work
    is avoidable without weakening correctness ownership
- strongest third target:
  - bounded graph-pipeline follow-through in `src/sparse_graph.c`,
    `src/sparse_graph_coarsen.c`, `src/sparse_graph_bisect.c`, and
    `src/sparse_graph_refine.c`
- strongest fourth target:
  - benchmark/comparison follow-through in `benchmarks/bench_reorder.c` and
    `benchmarks/bench_fillin.c` after a real landed runtime seam exists
- strongest support-only but real target:
  - maintainer/docs wording only where the landed runtime seam changes proof,
    rerun, or reviewed-path expectations

## Strongest Current Contradiction

The strongest current contradiction is now explicit:

- the validated Sprint 85 close already fixed the reviewed long pole to
  `test_reorder_nd` at `283.53 sec` out of `404.15 sec`
- the live tree shows that this is not just a large-test-file problem
- `tests/test_reorder_nd.c` concentrates many large-fixture and env-policy
  proofs while the underlying algorithmic work is split across
  `src/sparse_reorder_nd.c` and the `src/sparse_graph*.c` pipeline
- the heaviest retained fixtures and policy paths are concentrated on the same
  reviewed owner:
  - `bcsstk14`
  - `Pres_Poisson`
  - `Kuu`
  - `s3rmt3m3`
  - `SPARSE_ND_*` override and policy coverage

That fixes the strongest first Sprint 86 move:

- land one bounded ND runtime reduction first
- preserve the current correctness and reviewed proof-owner contract
- treat proof-surface rebalancing as a second seam, not the first runtime
  center
- keep benchmark and support-surface movement behind a real landed runtime
  seam

## Second-Tier Contradictions

### Proof-Surface Concentration

The strongest second contradiction is proof-surface concentration:

- `tests/test_reorder_nd.c` = `2287`
- `tests/test_graph.c` = `2925`
- `tests/test_reorder.c` = `1082`
- `tests/test_reorder_amd_qg.c` = `273`

The live tree shows repeated heavy fixture and policy coverage across the ND
and graph proof owners. That means proof-surface rebalancing is real Sprint 86
work, but it still reads as second after the first runtime-reduction lane
rather than as the initial implementation center.

### Algorithmic and Policy Concentration

The strongest third contradiction is algorithmic and policy concentration:

- `src/sparse_graph.c` = `841`
- `src/sparse_reorder_nd.c` = `757`
- `src/sparse_graph_coarsen.c` = `659`
- `src/sparse_reorder_amd_qg.c` = `611`
- `src/sparse_graph_refine.c` = `602`
- `src/sparse_graph_bisect.c` = `528`
- `src/sparse_reorder.c` = `419`
- `src/sparse_graph_separator.c` = `297`

The current runtime ceiling is therefore not isolated to one test file. It is
also a multilevel ND/graph pipeline problem. This is real Sprint 86 work, but
it still reads as follow-through behind the first bounded ND runtime seam
rather than a generic family-wide rewrite.

### Benchmark and Support Follow-Through

The strongest benchmark and support-only follow-through remains bounded:

- `benchmarks/bench_reorder.c` = `321`
- `benchmarks/bench_fillin.c` = `178`
- `README.md` = `1050`
- `docs/maintainer_guide.md` = `726`

These remain follow-through only where landed runtime work changes benchmark
comparison expectations, reviewed rerun guidance, or maintainer interpretation.
They do not become correctness owners.

## Deferred Runtime and Proof Claims

Broad runtime-claim widening remains lower-value first work:

- no generic suite-wide speedup claim detached from the ND/reorder lane
- no weakening of correctness proof quality to buy runtime wins
- no generic maintainability decomposition restart
- no benchmark-governance or example-governance drift into correctness
  ownership
- no support-surface churn detached from a real landed runtime seam
- no package/platform maturity claim widening

## Interpretation

The useful Day 3 clarification is now explicit:

- the best first Sprint 86 move is not generic "make tests faster"
- it is one bounded ND runtime reduction on the reviewed long-pole lane
- proof-surface rebalancing follows next where repeated heavy proof is
  avoidable without weakening ownership
- graph-pipeline follow-through comes after that where the first runtime
  landing exposes the true second seam
- benchmark surfaces remain informative, not authoritative
- support surfaces stay support-only unless implementation truly changes the
  reviewed-path or rerun contract

The Sprint 80 and Sprint 85 carry-forward reading is now fixed:

- Sprint 80 already fenced the performance contract, so Sprint 86 does not
  need to reopen generic performance governance
- Sprint 85 already handed Sprint 86 a runtime-first queue rather than another
  maintainability-first decomposition sprint
- the first Sprint 86 landing must preserve correctness ownership while
  reducing reviewed runtime on the ND lane

## Exit State

- Sprint 86 now has one ranked live reviewed-runtime contradiction map grounded
  in the current tree and validated Sprint 85 runtime anchors.
- The first implementation center is fixed to one bounded ND runtime reduction
  lane.
- Later proof-surface rebalancing, graph-pipeline follow-through, benchmark
  comparisons, and support-only wording are explicitly ordered behind that
  first lane.
