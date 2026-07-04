# Sprint 105 Day 7 Generated Graph-Family Evidence

## Purpose

Day 7 refreshes generated graph-family evidence for graph partition,
separator, quotient-graph, and nested-dissection behavior using deterministic
inputs already owned by the test suite. The goal is to expose structural
behavior and reproducibility, not to create a broad benchmark or portable
performance claim.

## Evidence Boundary

Day 7 keeps generated-family evidence in focused tests and supplemental local
artifacts. No new random fixture source, benchmark schema, or public
performance threshold is introduced.

Selected generated families:

| family | canonical fixture name | owner | Day 7 role |
|---|---|---|---|
| 1D path | `path1d-n20`, `path1d-n41` | `tests/test_graph.c`, `tests/test_reorder_nd.c` | degenerate partition and ND permutation behavior |
| 2D grid | `grid2d-5x5`, `grid2d-10x10`, `grid2d-30x30` | `tests/test_graph.c`, `tests/test_reorder_nd.c` | coarsening, separator, policy-difference, and ND/AMD fill behavior |
| 3D mesh | `mesh3d-5x5x5` | `tests/test_graph.c` | planar separator behavior and deterministic FM evidence |
| clique bridge | `two_cliques-k10` | `tests/test_graph.c` | small separator and partition determinism behavior |
| banded symmetric | `banded-n10000-bw5`, `banded-n256-bw8` | `tests/test_reorder_amd_qg.c`, `tests/test_reorder_nd.c` | quotient-graph AMD guardrail and ND factor-dispatch residual proof |

Deferred generated family:

| family | reason deferred |
|---|---|
| `arrow-n<N>` | current owner is `bench_fillin` human-readable LU context; keep supplemental until LU fill schema work is selected |

## Focused Validation Commands

```sh
make build/test_graph && ./build/test_graph
make build/test_reorder_nd && ./build/test_reorder_nd
make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg
```

## Captured Evidence Summary

### Graph Partition and Separator Families

Source: `make build/test_graph && ./build/test_graph`

Result:

```text
Tests run:    61
Tests failed: 0
Tests skipped: 0
Assertions:   1762
Time:         4.146 s
ALL TESTS PASSED
```

Notable generated-family evidence:

| fixture | evidence | interpretation |
|---|---|---|
| `grid2d-5x5` | coarsens in one step and again in two steps; hierarchy builds | regular-grid coarsening is deterministic and bounded |
| `path1d-n20` | coarsening halves; ND produces a valid permutation | degenerate path behavior is covered |
| `path1d-n8`, `path1d-n41` | brute-force and GGGP bisection paths pass | small and fall-through bisection paths are covered |
| `grid2d-5x6` | GGGP, FM reduction, and FM no-regress paths pass | grid partition and refinement behavior is covered |
| `grid2d-10x10` | partition, determinism, and balanced-boundary smoke paths pass | separator behavior is deterministic on a reviewed grid |
| `grid2d-30x30` | dynamic-K sep `60`, fixed-K sep `30`, partitions differ | policy-difference lane is exercised on a generated grid |
| `mesh3d-5x5x5` | baseline sep `25`, gain-noise sep `25`, partitions differ | 3D mesh separator behavior is covered without named external input |
| `two_cliques-k10` | partition and determinism paths pass | bridge/separator behavior is reproducible |

Named-matrix smoke rows in `test_graph` remain supplemental context:

```text
tests/data/suitesparse/bcsstk14.mtx (n=1806): sep=97, 43.2 ms
tests/data/suitesparse/Pres_Poisson.mtx (n=14822): sep=216, 261.8 ms
```

These timing values are local context only.

### Nested-Dissection Generated Families

Source: `make build/test_reorder_nd && ./build/test_reorder_nd`

Result:

```text
Tests run:    35
Tests failed: 0
Tests skipped: 1
Assertions:   105
Time:         100.892 s
ALL TESTS PASSED
```

The skipped test is explicit and non-failing:

```text
test_analysis_typed_nd_sep_lift_weight_overrides_env
bcsstk04 env balance weight did not differ from typed hybrid baseline
```

Notable generated-family evidence:

| fixture | evidence | interpretation |
|---|---|---|
| `grid2d-4x4` | ND produces a valid permutation with separator-last behavior | small separator structure is pinned |
| `grid2d-10x10` | AMD `nnz(L)=656`, ND `nnz(L)=656`, ND/AMD `1.00` | generated grid fill behavior is deterministic and comparable |
| `path1d-n20` | ND produces a valid permutation | degenerate path case remains covered |
| `banded-n256-bw8` | Cholesky residual under AMD `4.44e-16`, under ND `9.55e-15` | generated banded factor-dispatch path remains numerically coherent |

Named-matrix ND rows in the same test remain supplemental cross-checks for
the Day 6 named-matrix artifact:

```text
Pres_Poisson (n=14822): AMD nnz(L) = 2668793, ND nnz(L) = 2474435 (ND/AMD = 0.927, ND wall 7.40 s)
bcsstk14 (n=1806): NONE=190791, RCM=178311, AMD=116071, ND=132634 (ND/AMD = 1.143)
```

Local timing remains non-portable.

### Quotient-Graph AMD and Banded Guardrail

Source: `make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg`

Result:

```text
Tests run:    7
Tests failed: 0
Tests skipped: 0
Assertions:   2068
Time:         0.687 s
ALL TESTS PASSED
```

Notable evidence:

| fixture | evidence | interpretation |
|---|---|---|
| `banded-n10000-bw5` | AMD on 10,000 x 10,000 banded matrix with `nnz=109970` completed in `0.21 s` locally | quotient-graph AMD handles large regular generated input without the old bitset path |
| `nos4` | wrapper `nnz(L)=637`, qg `nnz(L)=637` | public AMD wrapper and qg implementation agree |
| `bcsstk04` | wrapper `nnz(L)=3143`, qg `nnz(L)=3143` | small structural named-matrix agreement |
| `bcsstk14` | wrapper `nnz(L)=116071`, qg `nnz(L)=116071` | reviewed structural named-matrix agreement |

The banded runtime is local context only; the maintained claim is successful
completion and structural agreement/validity.

## Generated-Family Claim Boundaries

| claim | status |
|---|---|
| generated fixtures require no external downloads | supported |
| grid, path, clique-bridge, mesh, and banded families are deterministic | supported by focused tests |
| generated evidence proves global ordering superiority | not claimed |
| local timing is portable performance evidence | not claimed |
| `bench_fillin` arrow evidence is canonical | deferred |
| generated families replace named Matrix Market evidence | not claimed |

## Day 8 Inputs

Day 8 large-matrix guardrail design should use these generated-family risks:

- `banded-n10000-bw5` is the best current large generated AMD guardrail.
- `grid2d-30x30` exercises policy-difference behavior without external data.
- `grid2d-10x10` and `path1d-n20` are cheap smoke fixtures for deterministic
  separator and ND behavior.
- `mesh3d-5x5x5` covers separator-heavy 3D structure.
- large local timing remains supplemental unless a future threshold source is
  defined.

## Completion Check

| criterion | status |
|---|---|
| at least two generated structural families covered | complete |
| deterministic commands recorded | complete |
| graph partition, separator, qg-AMD, and ND behavior represented | complete |
| skipped/deferred generated lanes explicit | complete |
| local timing non-claims preserved | complete |
