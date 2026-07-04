# Sprint 105 Day 4 Evidence Boundary and Matrix Selection

## Purpose

Day 4 freezes the first Sprint 105 evidence boundary before implementation
work begins. It selects named matrices, generated families, size tiers,
commands, artifact expectations, and deferred lanes using the Day 2 audit and
Day 3 fill/fixture contract.

## Boundary Decision

Sprint 105 will start with `bench_reorder` as the first implementation and
reporting lane because it already has:

- stable CSV output;
- named Matrix Market fixtures already committed to the repo;
- all five ordering labels needed for the first contract pass;
- `nnz_L` as the primary fill field;
- `reorder_ms` and `factor_ms` as local runtime fields;
- direct versus analyze path labeling;
- the current bounded `sprint86` fixture slice.

Sprint 105 will not promote `bench_amd_qg`, `bench_colamd`, or `bench_fillin`
to canonical reports until the `bench_reorder` field contract is stable.
Those adjacent benchmarks remain useful supplemental evidence.

## Selected Named-Matrix Evidence Set

### Reviewed First Slice

The reviewed first slice is the existing bounded `bench_reorder` slice:

| fixture | source | class | reason selected | command owner |
|---|---|---|---|---|
| `bcsstk14` | `tests/data/suitesparse/bcsstk14.mtx` | SPD structural mechanics | existing wall-check/qg-AMD and ND proof fixture; medium size | `bench_reorder`, `wall-check` |
| `Pres_Poisson` | `tests/data/suitesparse/Pres_Poisson.mtx` | SPD PDE/Poisson | existing ND long-pole and fill/runtimes calibration fixture | `bench_reorder`, `wall-check` |

Reviewed first command:

```sh
make bench-reorder-sprint86
```

Expanded equivalent:

```sh
build/bench_reorder --sprint86-slice --skip-factor
```

Expected output shape:

```text
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
...
```

Interpretation:

- `nnz_L` is the primary fill field.
- `reorder_ms` is local timing context only.
- `factor_ms=skip` is intentional for the reviewed first slice.
- `fixture_slice=sprint86` is historical naming for the current bounded
  two-fixture slice, not a new public benchmark claim.

### Supplemental Named-Matrix Slice

The supplemental full `bench_reorder` fixture list remains available for local
or artifact generation once the reviewed first slice is stable:

| fixture | source | class | reason selected | status |
|---|---|---|---|---|
| `nos4` | `tests/data/suitesparse/nos4.mtx` | small SPD | cheap baseline and many tests already use it | supplemental |
| `bcsstk04` | `tests/data/suitesparse/bcsstk04.mtx` | small SPD structural | cheap structural mechanics fixture | supplemental |
| `Kuu` | `tests/data/suitesparse/Kuu.mtx` | SPD bimodal-degree | graph/ND policy-sensitive fixture | supplemental |
| `bcsstk14` | `tests/data/suitesparse/bcsstk14.mtx` | medium SPD structural | reviewed first-slice member | reviewed/supplemental |
| `s3rmt3m3` | `tests/data/suitesparse/s3rmt3m3.mtx` | larger SPD | fill/runtimes context in existing bench list | supplemental/local |
| `Pres_Poisson` | `tests/data/suitesparse/Pres_Poisson.mtx` | large SPD PDE | reviewed first-slice member and local long-pole | reviewed/local |

Supplemental command:

```sh
build/bench_reorder --skip-factor
```

Interpretation:

- use for local before/after evidence;
- do not call it portable performance evidence;
- capture platform, compiler, build mode, and thread context if promoted into
  a planning artifact.

## Selected Generated-Family Evidence Set

Generated-family evidence should start from existing deterministic builders in
tests rather than new random fixtures.

| family | canonical fixture pattern | existing owner | first Sprint 105 role | status |
|---|---|---|---|---|
| 1D path | `path1d-n<N>` | `tests/test_graph.c`, `tests/test_reorder_nd.c` | degenerate ND and bandwidth behavior | reviewed test-local |
| 2D grid | `grid2d-<R>x<C>` | `tests/test_graph.c`, `tests/test_reorder_nd.c` | separator/ND structural behavior | reviewed test-local |
| banded symmetric | `banded-n<N>-bw<B>` | `tests/test_reorder_amd_qg.c`, `bench_amd_qg` | AMD large regular guardrail | reviewed test-local / supplemental benchmark |
| two cliques plus bridge | `two_cliques-k<K>` | `tests/test_graph.c` | graph partition and separator behavior | reviewed test-local |
| arrow | `arrow-n<N>` | `bench_fillin`, `tests/test_etree.c` | fill stress context | supplemental |

Day 4 boundary:

- generated families stay test-local or supplemental until Day 7;
- no new benchmark schema is introduced on Day 4;
- Day 7 may add artifact-friendly generated-family evidence only after Day 5
  proves the first `bench_reorder` contract pass.

## Size Tiers

| tier | intended size | examples | expected command status | interpretation |
|---|---|---|---|---|
| smoke | small, fast, always cheap | `nos4`, `bcsstk04`, `path1d-n20`, `grid2d-5x5` | test or focused benchmark smoke | quick correctness/format check |
| reviewed | bounded and deterministic | `bcsstk14`, `Pres_Poisson` with `--skip-factor`, `banded-n10000-bw5` test | reviewed local/CI-equivalent where already owned | maintained evidence, still not portable timing |
| supplemental | useful but broader | full `bench_reorder --skip-factor`, `Kuu`, `s3rmt3m3`, `bench_colamd` | local artifact or focused maintainer run | comparison context |
| local-only | large/noisy or machine-sensitive | full factor pass on large fixtures, broader ND threshold sweeps, bitset rows | explicit local-only command | no public performance claim |

## Frozen Implementation Lanes

### Lane 1: `bench_reorder` Contract Alignment

Owner:

- `benchmarks/bench_reorder.c`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

Scope:

- preserve or document current CSV header;
- clarify `fixture_slice=sprint86` as current bounded two-fixture slice;
- map current fields to the Day 3 contract;
- keep `factor_ms=skip` explicit;
- avoid changing adjacent benchmarks in the first pass.

Validation candidate:

```sh
make bench-reorder-sprint86
```

If `.c` or `.h` changes occur, run:

```sh
make format && make lint && make test
```

### Lane 2: Named-Matrix Evidence Capture

Owner:

- planning artifact under `docs/planning/EPIC_10/SPRINT_105/artifacts/`

Scope:

- capture reviewed first-slice output after Lane 1;
- optionally capture supplemental full-slice output when runtime is acceptable;
- record platform/build context and non-claims.

Validation candidate:

```sh
build/bench_reorder --sprint86-slice --skip-factor
```

### Lane 3: Generated-Family Boundary

Owner:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`
- `tests/test_reorder_amd_qg.c`
- future Day 7 artifact or helper if needed

Scope:

- use existing deterministic generated families first;
- avoid adding random generated fixtures;
- choose artifact-friendly names from the Day 3 contract if rows are emitted.

Validation candidates:

```sh
make build/test_graph && ./build/test_graph
make build/test_reorder_nd && ./build/test_reorder_nd
make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg
```

Run full C quality gate if any `.c` or `.h` file changes.

## Deferred Lanes

| lane | deferral reason |
|---|---|
| `bench_amd_qg` canonical promotion | historical bitset-vs-qg foil; memory field is useful but local and platform-sensitive |
| `bench_colamd` schema migration | useful, but QR/COLAMD output should wait until `bench_reorder` contract is stable |
| `bench_fillin` schema migration | useful LU fill context, but current output is human-readable and not first-lane critical |
| hard thresholds for reorder/fill reports | Sprint 104 requires baseline design and machine-class assumptions before new hard gates |
| full numeric factor pass on large matrices | too slow/noisy for reviewed first lane |
| splitting large graph/ND tests | useful maintainability work, but should follow touched implementation boundaries |

## Command and Artifact Plan

| day | command or artifact | expected output | status |
|---|---|---|---|
| Day 5 | `make bench-reorder-sprint86` or direct equivalent | CSV rows for bcsstk14 and Pres_Poisson across reorderings | reviewed first lane |
| Day 6 | selected named-matrix artifact | captured and interpreted first-slice rows, optional supplemental rows | planning/report evidence |
| Day 7 | generated-family artifact or focused tests | deterministic graph-family evidence with contract names | reviewed or supplemental |
| Day 8 | guardrail design | selected large-matrix deterministic risks and thresholds/report modes | design |
| Day 9 | guardrail implementation | focused guardrail command output | reviewed/supplemental per boundary |

## Non-Claims

This boundary does not claim:

- `bench_reorder` timing is portable across machines or compilers;
- `sprint86` is a new product-facing benchmark brand;
- ND, AMD, RCM, or COLAMD is universally better than the others;
- generated graph families represent all real sparse workloads;
- max-RSS or local wall timing is comparable across platforms without a fresh
  machine contract;
- full reorder-adjacent benchmark schema migration is complete.

## Completion Check

| criterion | status |
|---|---|
| named-matrix first slice selected | complete |
| supplemental named-matrix slice selected | complete |
| generated-family evidence set selected | complete |
| size tiers defined | complete |
| command and artifact plan written | complete |
| implementation lanes frozen before source/script edits | complete |
