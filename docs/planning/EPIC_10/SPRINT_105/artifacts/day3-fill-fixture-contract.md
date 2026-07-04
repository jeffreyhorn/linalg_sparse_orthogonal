# Sprint 105 Day 3 Fill and Fixture Contract

## Purpose

Day 3 defines the canonical field, fixture naming, skip, and interpretation
contract for Sprint 105 reorder, graph, fill, runtime, and memory artifacts.
The contract is intentionally narrower than a new benchmark schema migration:
it sets the rules that later implementation days must follow before changing
`bench_reorder`, documentation, or adjacent report owners.

## Contract Principles

- Fill, runtime, and memory fields must be separable. A row can carry all
  three, but interpretation must not blur them.
- Local timing is command, fixture, build, backend, and thread-context evidence
  only.
- Generated graph families are deterministic structural fixtures, not
  stand-ins for every real sparse workload.
- Skip and error rows are evidence. They must use the same fixture and
  algorithm identifiers as successful rows.
- Thresholded pass/fail behavior requires a documented baseline, threshold
  source, and machine-class assumption. Report-only rows must not fail.
- Tests and external oracles remain correctness owners. Benchmark fill or
  residual fields are context unless a test explicitly owns the correctness
  claim.

## Canonical Artifact Classes

| class | status values | primary owner | interpretation |
|---|---|---|---|
| reorder/fill report | `report`, `skip`, `error` | benchmark or script owner | threshold-free structural and local runtime evidence |
| large-matrix guardrail | `pass`, `fail`, `skip` or `report` | test, benchmark, or script owner | deterministic smoke/structural bound if thresholded; local evidence if report-only |
| performance sentinel | `pass`, `fail`, `skip`, `report` | sentinel script owner | narrow regression gate only when thresholded |
| proof test | test framework result | test owner | correctness or invariant evidence |
| historical comparison foil | `report`, `skip`, `error` | benchmark owner | local context only, not canonical public performance evidence |

## Canonical Row Fields

Future Sprint 105 reorder/fill artifacts should prefer these fields. Existing
reports may keep their current headers until an implementation day explicitly
migrates them.

| field | required | value contract | interpretation |
|---|---|---|---|
| `artifact_id` | yes for bundled reports | stable artifact or lane name, for example `reorder_fill` | groups rows from one run |
| `status` | yes | `report`, `pass`, `fail`, `skip`, `error` | result state; only `pass`/`fail` imply a threshold |
| `command` | yes for generated bundles | exact command or target | reproducibility and claim boundary |
| `fixture` | yes | canonical fixture identifier from this contract | aggregation key across report/skip/error rows |
| `fixture_source` | recommended | `suitesparse`, `generated`, `synthetic`, `builtin`, or `external` | fixture provenance |
| `fixture_class` | recommended | structural class such as `spd`, `unsymmetric`, `grid2d`, `banded`, `arrow`, `separator_heavy` | explains why the row exists |
| `nrows` | yes when known | integer or `n/a` | size context |
| `ncols` | yes when known | integer or `n/a` | rectangular support context |
| `nnz_A` | recommended | integer or `n/a` | input sparsity context |
| `ordering` | yes | `none`, `rcm`, `amd`, `qg_amd`, `colamd`, `nd`, or a documented extension | algorithm key |
| `ordering_scope` | yes | `symmetric`, `column`, `graph_partition`, `separator`, or `n/a` | prevents COLAMD/symmetric confusion |
| `ordering_path` | recommended | `direct`, `analyze`, `qr`, `lu`, `cholesky`, `ldlt`, or `n/a` | path context |
| `policy` | recommended | policy summary or `default`/`n/a` | ND/graph policy context |
| `fixture_slice` | recommended | named slice such as `all`, `reviewed`, `sprint86`, or `local` | reviewed/supplemental context |
| `fill_metric` | yes when fill is reported | `nnz_L`, `nnz_R`, `nnz_LU`, `fill_ratio`, or documented extension | structural metric name |
| `fill_value` | yes when fill is reported | numeric value or `n/a` | structural metric value |
| `fill_baseline` | recommended | baseline ordering or fixture, or `n/a` | denominator for ratios |
| `fill_ratio` | recommended | numeric ratio or `n/a` | must name baseline |
| `runtime_metric` | recommended | `reorder_ms`, `factor_ms`, `wall_ms`, or `n/a` | local timing metric name |
| `runtime_ms` | recommended | numeric milliseconds or `n/a` | local timing only |
| `memory_metric` | recommended | `peak_rss_mb`, `workspace_ints`, `allocation_guard`, or `n/a` | memory/proxy metric name |
| `memory_value` | recommended | numeric value or `n/a` | memory/proxy value |
| `threshold` | required for pass/fail | value and rule, or `n/a` | thresholded interpretation |
| `threshold_source` | required for pass/fail | file, artifact, or `n/a` | baseline ownership |
| `reviewed_status` | yes | `reviewed`, `supplemental`, `local-only`, or `exploratory` | claim boundary |
| `build_mode` | recommended for runtime rows | `serial`, `openmp`, `unknown`, or `n/a` | runtime context |
| `omp_num_threads` | recommended for runtime rows | value, `unset`, or `n/a` | runtime context |
| `notes` | recommended | short semicolon-delimited notes | skip reason, unsupported case, or caveat |

## Mapping Current Outputs to the Contract

| current owner | current fields | contract interpretation | Day 5+ action |
|---|---|---|---|
| `bench_reorder` | `matrix`, `n`, `reorder`, `nnz_L`, `reorder_ms`, `factor_ms`, `reorder_path`, `fixture_slice`, `nd_base_threshold` | strongest current reorder/fill report; `nnz_L` maps to fill, `reorder_ms` and `factor_ms` map to local timing | first candidate for contract docs or schema enrichment |
| `bench_amd_qg` | `matrix`, `n`, `impl`, `reorder_ms`, `peak_rss_mb`, `nnz_L` | historical qg-AMD vs bitset foil; memory proxy is local max-RSS delta | keep bounded unless a fresh contract promotes it |
| `bench_colamd` | human-readable `nnz(R)` none/AMD/COLAMD rows | COLAMD QR fill context, not artifact-ready | optional later schema alignment |
| `bench_fillin` | human-readable LU `nnz_before`, `nnz_after`, `ratio` | LU fill smoke context, not artifact-ready | optional later schema alignment |
| `wall-check` | parsed qg-AMD/AMD/ND `reorder_ms` versus baselines | thresholded sentinel-like gate | keep hard gate narrow |
| `performance-sentinels` | S5 pass/fail rows and S2 report rows | sentinel bundle with explicit status and context | preserve status/skip discipline |

## Fixture Naming Rules

### General Rules

1. Use the same fixture identifier in `report`, `skip`, and `error` rows.
2. Use lowercase algorithm names but preserve canonical fixture case when the
   file or benchmark already has a stable name, such as `Pres_Poisson`.
3. Named Matrix Market fixtures should use the basename without `.mtx` unless
   an existing benchmark already emits the extension. A schema migration must
   choose one spelling and use it across all statuses.
4. Generated fixtures must encode family and dimensions in the fixture name.
5. Synthetic stress cases must encode the stress shape and size, not the
   historical sprint that introduced them.
6. Fixture-slice labels are metadata, not fixture identity.

### Named Matrix Fixtures

| source | canonical fixture | notes |
|---|---|---|
| `tests/data/suitesparse/nos4.mtx` | `nos4` | keep current `bench_reorder` spelling unless a schema migration changes all statuses together |
| `tests/data/suitesparse/bcsstk04.mtx` | `bcsstk04` | SPD/structural mechanics fixture |
| `tests/data/suitesparse/bcsstk14.mtx` | `bcsstk14` | wall-check qg-AMD fixture and ND proof fixture |
| `tests/data/suitesparse/Pres_Poisson.mtx` | `Pres_Poisson` | preserve existing case because current reports and baselines use it |
| `tests/data/suitesparse/Kuu.mtx` | `Kuu` | preserve existing case |
| other SuiteSparse files | basename without `.mtx` | use exact basename case |

### Generated Family Fixtures

| family | naming pattern | example | intended use |
|---|---|---|---|
| 1D path | `path1d-n<N>` | `path1d-n10000` | degenerate ND and bandwidth behavior |
| 2D grid | `grid2d-<R>x<C>` | `grid2d-100x100` | ND/separator and fill behavior |
| banded symmetric | `banded-n<N>-bw<B>` | `banded-n10000-bw5` | AMD large regular guardrail |
| arrow | `arrow-n<N>` | `arrow-n500` | fill stress |
| two cliques plus bridge | `two_cliques-k<K>` | `two_cliques-k20` | graph partition behavior |
| separator-heavy synthetic | `separator_heavy-<shape>` | `separator_heavy-grid_bridge` | graph/separator proof |

### Fixture Slice Labels

| label | meaning |
|---|---|
| `all` | all fixtures selected by the owning benchmark |
| `reviewed` | bounded reviewed fixture set, if explicitly defined |
| `sprint86` | historical current `bench_reorder` label for bcsstk14/Pres_Poisson slice |
| `local` | maintainer-local or supplemental fixture set |
| `generated` | generated-family fixture set |

Sprint 105 may keep `sprint86` for compatibility in `bench_reorder`, but user
and maintainer docs should describe it as the current bounded two-fixture
slice, not as a new public benchmark claim.

## Ordering and Mode Labels

| label | scope | meaning |
|---|---|---|
| `none` | `symmetric` or `column` | no ordering baseline |
| `rcm` | `symmetric` | Reverse Cuthill-McKee |
| `amd` | `symmetric` | public AMD wrapper, currently quotient-graph AMD |
| `qg_amd` | `symmetric` | explicit quotient-graph AMD implementation row |
| `colamd` | `column` | column approximate minimum degree |
| `nd` | `symmetric` | nested dissection |
| `graph_partition` | `graph_partition` | graph bisection/partition surface |
| `separator` | `separator` | vertex separator or lift surface |

Do not mix `amd` and `colamd` as interchangeable labels. If a QR/COLAMD report
compares `amd` and `colamd`, it must name `ordering_scope=column` for COLAMD
and explain any symmetric AMD comparison path separately.

## Fill Metric Rules

| metric | unit | valid owners | interpretation |
|---|---|---|---|
| `nnz_L` | nonzero count | symbolic Cholesky/LDL-style reports | structural fill context |
| `nnz_R` | nonzero count | QR/COLAMD reports | QR fill context |
| `nnz_LU` | nonzero count | LU fill reports | LU factor structural context |
| `fill_ratio` | ratio | any fill report with a named baseline | value divided by baseline fill |
| `bandwidth` | index distance | RCM/bandwidth reports | structural locality context |
| `separator_size` | vertex count | graph partition reports | separator context, not fill by itself |

Fill ratios must name the denominator in `fill_baseline`. Examples:

- `fill_baseline=none`
- `fill_baseline=amd`
- `fill_baseline=input_nnz`
- `fill_baseline=prior_run:<artifact>`

## Runtime and Memory Rules

Runtime fields:

- `reorder_ms` means local wall or CPU timing of the ordering call, depending
  on the owner. The owner must document which clock it uses.
- `factor_ms` means local factor timing for the selected factorization path.
- `wall_ms` means local wall-clock milliseconds for a command or phase.
- runtime values are report-only unless paired with a threshold and source.

Memory fields:

- `peak_rss_mb` is a process-level local proxy and may be platform-specific.
- `workspace_ints` is an implementation-level proxy only when the owner can
  define it deterministically.
- `allocation_guard` is a pass/fail structural bound, not a memory benchmark.
- memory values must not be compared across platforms without an explicit
  machine and OS contract.

## Skip and Unavailable-Lane Rules

| condition | status | required fields |
|---|---|---|
| fixture file missing | `skip` | canonical `fixture`, `ordering`, `notes=fixture_missing` |
| benchmark binary missing | `skip` | `command`, `notes=binary_missing` |
| ordering unsupported for shape | `skip` or `error` | `fixture`, `ordering`, `notes=unsupported_shape` |
| symbolic analysis fails | `error` | `fixture`, `ordering`, `notes=analysis_failed:<code>` |
| factor intentionally skipped | `report` with `factor_ms=skip` or `skip` row in enriched schema | `notes=factor_skipped` |
| threshold baseline missing | `skip` | `threshold_source`, `notes=baseline_missing` |
| optional backend unavailable | `skip` or `report` depending on owner | explicit fallback note |

Skip rows are not passes. They should be preserved in generated artifacts
where practical so missing evidence is visible.

## Reviewed Status Rules

| status | meaning |
|---|---|
| `reviewed` | expected in maintained local/CI-equivalent validation and documented as such |
| `supplemental` | useful evidence, but not part of the primary reviewed quality claim |
| `local-only` | maintainer-local measurement or machine-sensitive artifact |
| `exploratory` | development investigation; not a public claim surface |

The current bounded `bench-reorder-sprint86` slice remains supplemental or
local-reviewed context unless a later artifact explicitly promotes it.

## Implementation Checklist

Before changing a reorder/fill report owner:

- [ ] Identify whether the owner is report-only, thresholded, supplemental, or
      reviewed.
- [ ] Name the exact command and output artifact.
- [ ] Decide whether existing consumers require backward-compatible headers.
- [ ] Use canonical fixture and ordering labels for report, skip, and error
      rows.
- [ ] Separate fill, runtime, memory, and correctness context fields.
- [ ] Define threshold and threshold source before emitting `pass` or `fail`.
- [ ] Record unsupported cases and skipped lanes explicitly.
- [ ] Update `benchmarks/README.md` if a benchmark schema or interpretation
      changes.
- [ ] Update `docs/maintainer_guide.md` if reviewed/supplemental ownership
      changes.
- [ ] Run focused report generation after script or benchmark changes.
- [ ] Run `make format && make lint && make test` if any `.c` or `.h` file is
      modified.

## Day 4 Starting Point

Day 4 should use this contract to select the first bounded evidence set:

1. named matrices already supported by `bench_reorder`;
2. generated families that can use deterministic names;
3. reviewed versus supplemental fixture slices;
4. the first report owner to align, likely `bench_reorder`;
5. any adjacent benchmark owners that should remain deferred until the
   `bench_reorder` path is stable.

## Completion Check

| criterion | status |
|---|---|
| fill, runtime, memory, and fixture fields have clear semantics | complete |
| naming rules support aggregation across report, skip, and error rows | complete |
| skip and unavailable-lane rules are explicit | complete |
| implementation checklist identifies docs, scripts, benchmarks, and validation | complete |
| no metric is framed as a portable performance claim | complete |
