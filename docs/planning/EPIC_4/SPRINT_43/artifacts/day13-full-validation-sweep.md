# Sprint 43 Day 13: Full Validation Sweep

## Summary

Day 13 ran the full Sprint 40/41/42 validation anchor against the Sprint 43
Phase-1 graph / ND decomposition.

The result is clean:

- the mandatory code-change floor passed
- `make quality-review-full` passed
- reviewed CMake parity remained exact at `53`
- direct graph and ND reruns passed

No reconciliation queue surfaced.

## Validation Runs

### Mandatory floor

- `make format` → passed, `real 4.04`
- `make lint` → passed, `real 356.72`
- `make test` → passed, `real 94.28`

### Strong reviewed proof

- `make quality-review-full` → passed, `real 816.43`

Included reviewed-path components:

- reviewed Makefile path
- `deadcode-check`
- reviewed clean CMake rebuild
- `ctest -N`
- full reviewed CMake `ctest`

### Direct graph / ND reruns

- `./build/test_graph` → passed, `Time: 2.381 s`
- `./build/test_graph_fm_buckets` → passed, `Time: 0.000 s`
- `./build/test_reorder_nd` → passed, `Time: 31.386 s`
- `./build/test_reorder_amd_qg` → passed, `Time: 0.324 s`

## Truthfulness Anchors

The preserved Sprint 40 validation/truthfulness anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 195.15 sec`

Interpretation:

- Sprint 43 did not disturb the maintained local reviewed baseline
- the graph split remains aligned with the established Makefile/CMake parity
  contract

## Graph-Specific Outcome

The graph-focused reruns are important because Sprint 43 moved large internal
ownership seams across real implementation files:

- `src/sparse_graph_core.c`
- `src/sparse_graph_coarsen.c`
- `src/sparse_graph_bisect.c`
- residual `src/sparse_graph.c`

The direct reruns confirm those seams still compose cleanly across:

- graph construction / ownership
- hierarchy / coarsening
- coarse bisection
- FM bucket support
- ND integration
- quotient-graph AMD integration

The Day 12 seam additions stayed green explicitly:

- `test_graph_subgraph_argument_validation`
- `test_graph_subgraph_path_slice`
- `test_bisect_forced_gggp_small_graph`
- `test_bisect_forced_brute_large_graph_falls_back_to_gggp`

## Caveats

None beyond the normal maintained contract:

- dead-code execution remains serialized
- reviewed CMake remains the strongest shared reviewed baseline
- FM refinement, separator lifting, and deeper runtime-strategy cleanup remain
  later-phase work and were not reopened in this sweep

## Outcome

Sprint 43 now closes from a measured, validated Phase-1 decomposition state:

- extracted graph ownership / construction seam validated
- extracted hierarchy / coarsening seam validated
- extracted coarse-bisection seam validated
- updated build/include wiring validated
- focused graph seam tests validated

That is the correct end-state before Sprint 43 closeout and handoff.
