# Sprint 105 Day 8 Large-Matrix Guardrail Design

## Purpose

Day 8 defines deterministic large-matrix guardrails for reorder, graph, and
fill evidence. The design uses the Day 6 named-matrix evidence, Day 7
generated-family evidence, and the existing Sprint 104 sentinel boundaries.
It avoids new portable timing claims and limits hard pass/fail behavior to
structural bounds, smoke limits, and existing threshold sources.

## Design Principles

- Guardrails must target deterministic failure modes first.
- New pass/fail timing thresholds require a baseline, threshold source, and
  machine-class assumption before implementation.
- Local runtime and max-RSS evidence stays report-only unless promoted by a
  later baseline design.
- Reviewed lanes must stay affordable for normal development validation.
- Supplemental and local-only lanes must be explicit so they do not become
  hidden CI requirements.
- Skip behavior is evidence and must name the missing fixture, binary,
  baseline, or platform assumption.

## Large-Matrix Risk Register

| risk | observed source | failure mode | deterministic guardrail direction |
|---|---|---|---|
| memory growth | `bench_amd_qg` and qg-AMD history | dense/bitset-style workspace returns or large generated rows exhaust memory | generated banded structural completion and optional max-RSS report-only row |
| integer overflow | generated matrix builders and index arithmetic | `n`, `nnz`, or allocation sizes wrap silently | size/nnz sanity assertions before allocation-heavy paths |
| recursion depth or stack pressure | ND and graph partition recursion | degenerate path/grid recursion becomes too deep or nonterminating | generated path/grid smoke with valid permutation and bounded completion |
| pathological fill | named matrices and generated grids | fill explodes or ordering path regresses structurally | `nnz_L` structural comparisons against stable baselines or broad bounds |
| runtime cliffs | qg-AMD, ND Pres_Poisson, COLAMD rows | local orderings become unexpectedly slow | keep existing `wall-check`; add report-only runtime rows unless baseline exists |
| separator degeneracy | graph partition on grids, meshes, and bridge fixtures | separator is empty, whole graph, nondeterministic, or violates cut invariant | generated separator tests with partition invariant and nondegenerate bounds |
| unsupported large-lane drift | optional/local commands | local-only jobs become implied quality gates | explicit local-only labels and skip rules |

## Selected Guardrail Lanes

### Reviewed Structural Lanes

These lanes are suitable for maintained focused validation because they are
deterministic and already have test ownership.

| lane id | command | fixture | owner | pass/fail basis | Day 9 recommendation |
|---|---|---|---|---|---|
| `G1` | `make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg` | `banded-n10000-bw5` | `tests/test_reorder_amd_qg.c` | successful qg-AMD completion and valid structural rows | keep reviewed; optionally document as large generated AMD guardrail |
| `G2` | `make build/test_reorder_nd && ./build/test_reorder_nd` | `grid2d-10x10`, `path1d-n20`, `banded-n256-bw8` | `tests/test_reorder_nd.c` | valid permutation, deterministic `nnz_L`, residual bound | keep reviewed; avoid broader runtime threshold |
| `G3` | `make build/test_graph && ./build/test_graph` | `grid2d-10x10`, `grid2d-30x30`, `mesh3d-5x5x5`, `two_cliques-k10` | `tests/test_graph.c` | partition invariant, nondegenerate separator, determinism | keep reviewed; use as graph/separator guardrail owner |
| `G4` | `make bench-reorder-sprint86` | `bcsstk14`, `Pres_Poisson` | `bench_reorder` | exact CSV shape plus structural `nnz_L` report rows | keep reviewed report proof; do not make timing thresholded |

### Supplemental Report Lanes

These lanes are useful for maintainers but should remain report-only unless a
future sprint defines baselines.

| lane id | command | fixture set | metric | reason supplemental |
|---|---|---|---|---|
| `S1` | `build/bench_reorder --skip-factor` | full `bench_reorder` named-matrix slice | `nnz_L`, fill ratios, `reorder_ms` | useful six-fixture context; `reorder_ms` is local |
| `S2` | `build/bench_amd_qg --skip-bitset` | SuiteSparse plus generated banded rows | `reorder_ms`, `peak_rss_mb`, `nnz_L` | max-RSS is platform-sensitive and bitset foil is historical |
| `S3` | `make performance-sentinels` | existing sentinel bundle | S5 pass/fail plus S2 report rows | hard-fail behavior remains limited to existing `wall-check` rows |
| `S4` | `build/bench_reorder --sprint86-slice --skip-factor --reorder-via-analyze` | reviewed two-fixture slice | analyze-path `nnz_L`, `reorder_ms` | useful path comparison; not first reviewed lane |

### Local-Only Lanes

These lanes may be useful during investigation but should not be made reviewed
or required without a fresh baseline design.

| lane id | command pattern | reason local-only | skip behavior |
|---|---|---|---|
| `L1` | full numeric factor pass on `Pres_Poisson` or full `bench_reorder` | too slow and machine-sensitive | `skip` with `notes=factor_skipped` unless explicitly requested |
| `L2` | large ND threshold sweeps | runtime cliffs and policy exploration | `skip` with threshold-source missing unless a sweep artifact owns it |
| `L3` | full bitset side of `bench_amd_qg` large generated rows | historical foil can run for minutes | use `--skip-bitset`; record `notes=historical_bitset_skipped` |
| `L4` | cross-platform max-RSS comparisons | platform units and allocator behavior differ | report local platform; no pass/fail |

## Threshold and Bound Rules

| metric | allowed status | threshold rule |
|---|---|---|
| valid permutation | `pass`/`fail` | deterministic invariant; no machine context needed |
| partition invariant | `pass`/`fail` | deterministic invariant; no machine context needed |
| separator nondegeneracy | `pass`/`fail` | fixture-specific structural bound, such as finite and smaller than graph |
| `nnz_L` exact value | `pass`/`fail` only in tests that already own exact structural expectations | exact or broad bound must be fixture-local and documented |
| residual norm | `pass`/`fail` | numeric tolerance must be documented by test owner |
| `reorder_ms` | `report` unless baseline exists | existing `wall-check` is the only current hard timing gate |
| `peak_rss_mb` | `report` | platform-sensitive; no cross-machine threshold |
| full command wall time | `report` or local-only | no portable threshold without new baseline design |

## Skip and Report Rules

| condition | status | required note |
|---|---|---|
| generated fixture builder fails allocation | `fail` for reviewed structural test | `allocation_failed` |
| optional named matrix missing | `skip` | `fixture_missing` |
| benchmark binary missing | `skip` | `binary_missing` |
| threshold baseline absent | `skip` | `baseline_missing` |
| bitset historical foil skipped | `skip` or `report` | `historical_bitset_skipped` |
| numeric factor intentionally skipped | `report` or `skip` | `factor_skipped` |
| max-RSS platform not comparable | `report` | `platform_local_only` |

## Day 9 Implementation Plan

Day 9 should implement or refresh only a narrow guardrail batch:

1. Prefer documentation and command aggregation around existing reviewed
   owners before adding source.
2. If source is touched, keep it to structural assertions in
   `tests/test_reorder_amd_qg.c`, `tests/test_reorder_nd.c`, or
   `tests/test_graph.c`.
3. Do not add new timing pass/fail thresholds.
4. Keep `banded-n10000-bw5` as the first large generated AMD guardrail.
5. Keep `grid2d-10x10`, `grid2d-30x30`, `path1d-n20`, `mesh3d-5x5x5`, and
   `two_cliques-k10` as graph/ND guardrail fixtures.
6. Record supplemental commands separately from reviewed commands.
7. Run `make format && make lint && make test` if any `.c` or `.h` files are
   modified.

## Validation Plan

Reviewed focused validation:

```sh
make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg
make build/test_reorder_nd && ./build/test_reorder_nd
make build/test_graph && ./build/test_graph
make bench-reorder-sprint86
```

Supplemental report validation:

```sh
build/bench_reorder --skip-factor
build/bench_amd_qg --skip-bitset
make performance-sentinels
```

Docs-only validation:

```sh
git diff --check -- docs/planning/EPIC_10/SPRINT_105
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105
```

## Non-Claims

This design does not claim:

- any new portable timing threshold;
- max-RSS values are comparable across platforms;
- local-only large lanes are required quality gates;
- generated fixtures cover all sparse matrix behavior;
- `bench_fillin` arrow rows are canonical guardrail evidence;
- the existing `wall-check` baseline should be widened without a new baseline
  review.

## Completion Check

| criterion | status |
|---|---|
| large-matrix risks identified | complete |
| reviewed, supplemental, and local-only lanes separated | complete |
| deterministic pass/fail rules separated from report-only metrics | complete |
| skip behavior defined | complete |
| Day 9 implementation guidance written | complete |
