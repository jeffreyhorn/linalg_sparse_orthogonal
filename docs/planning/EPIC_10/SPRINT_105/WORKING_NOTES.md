# Sprint 105 Working Notes

## Sprint Context

Sprint 105 implements "Reordering, Graph & Large-Matrix Scalability" from
`docs/planning/EPIC_10/PROJECT_PLAN.md`. The sprint improves reorder, graph,
fill, runtime, and memory evidence without turning local measurements into
portable performance or state-of-the-art claims.

The sprint starts from the Sprint 100 evidence templates and the Sprint 104
runtime/benchmark contract. Reorder and graph work must name the command,
fixture scope, metric fields, backend/thread context where relevant, reviewed
status, unsupported cases, and non-claims before it can support public or
maintainer-facing wording.

## Validation Rules

Validation must scale with the touched surface:

| touched surface | required validation |
|---|---|
| planning documentation only | `git diff --check`; trailing-whitespace scan on touched planning files |
| public documentation only | `git diff --check`; trailing-whitespace scan on touched docs |
| benchmark docs or scripts | focused benchmark/report command where runnable; docs hygiene |
| helper script only | focused helper invocation, if executable; docs hygiene |
| generated artifact or benchmark report only | regenerate or inspect the owning command output; docs hygiene |
| test `.c` file | focused affected test binary; `make format`; `make lint`; `make test` |
| library `.c` or public `.h` file | focused affected tests; `make format`; `make lint`; `make test` |
| build or CMake surface | focused Make/CMake configure or build check plus any code-touch gate |
| workflow or package surface | focused workflow/package command where runnable plus any code-touch gate |

If any `.c` or `.h` file is modified, the full required quality chain is:

```sh
make format && make lint && make test
```

All required checks must pass before closeout or PR creation.

## Claim Boundaries

Sprint 105 may earn only bounded reorder, graph, fill, runtime, and
large-matrix scalability evidence tied to named commands, fixtures, metric
definitions, validation commands, and unsupported cases.

Sprint 105 must not claim:

- portable timing superiority across machines, compilers, operating systems,
  optional backends, or OpenMP runtimes;
- broad SuiteSparse, PETSc, Trilinos, or graph-package parity;
- universal best ordering quality for AMD, COLAMD, nested dissection, quotient
  graph, or partitioning paths;
- that local fill or runtime rows replace correctness tests or external
  oracles;
- that generated synthetic graph families represent all real sparse workloads;
- that large-matrix smoke guardrails prove unbounded scalability;
- GPU, distributed-memory, or out-of-core sparse graph support;
- Windows Makefile, benchmark, or install parity beyond reviewed scope;
- that report-only benchmark rows are hard performance gates.

## Day 1 - Scope and Scalability Baseline

### Goal

Convert the Sprint 105 project-plan section and prior Epic 10 handoffs into a
bounded reorder/graph scalability package with clear workstreams, evidence
rules, validation expectations, and claim boundaries.

### Actions

- Re-read the Sprint 105 section of
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read Sprint 100 benchmark and performance evidence guardrails:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day10-benchmark-coverage-performance-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/benchmark-interpretation-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/performance-sentinel-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-claim-non-goal-register.md`
- Re-read Sprint 104 closeout and retrospective handoffs:
  - `docs/planning/EPIC_10/SPRINT_104/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_10/SPRINT_104/RETROSPECTIVE.md`
- Created the Sprint 105 artifacts directory.
- Recorded authoritative Day 1 inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Sprint 105 scope baseline, workstream ownership, validation
  matrix, and claim boundaries in `artifacts/day1-scalability-baseline.md`.

### Findings

- Sprint 100 keeps reorder/fill and graph evidence in candidate state until a
  sprint records named fixtures, fill metric contracts, local timing caveats,
  validation commands, unsupported cases, and non-claims.
- Sprint 104 requires benchmark and sentinel outputs to preserve command,
  fixture, backend, thread, threshold/report-only, skip, and platform context.
- Sprint 105 should audit reorder and graph owners before source edits so
  metric contracts, generated-family evidence, and guardrails attach to real
  maintained surfaces.
- Large-matrix guardrails should target deterministic failure modes such as
  memory growth, overflow, recursion depth, pathological fill, or runtime
  cliffs. They should not become portable timing claims.
- Documentation and reports must distinguish fill, memory, local runtime,
  correctness context, and reviewed versus supplemental lanes.

### Validation Expectations

- Day 1 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 1 Exit State

Day 1 is complete. Sprint 105 now has working notes, authoritative inputs,
scope baseline, workstream ownership, validation expectations, claim
boundaries, and a Day 2 audit starting point.

## Day 2 - Reorder and Graph Surface Audit

### Goal

Re-rank AMD, COLAMD, nested dissection, quotient graph, graph partition, and
fill-report gaps from the live repository before Sprint 105 defines metric
contracts or edits implementation code.

### Actions

- Re-read the Day 2 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Inventoried reorder and graph source owners:
  - `src/sparse_reorder.c`
  - `src/sparse_reorder_amd_qg.c`
  - `src/sparse_colamd.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph_core.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_graph.c`
  - `src/sparse_analysis.c`
- Inventoried public API, test, benchmark, Make, CMake, and documentation
  owners for reorder/fill and graph evidence.
- Classified current evidence by AMD/qg-AMD, COLAMD, ND, graph partition,
  separator, fill, runtime, and memory-reporting family.
- Ranked current gaps by value, determinism, validation cost, and claim risk.
- Wrote `artifacts/day2-reorder-graph-surface-audit.md`.

### Findings

- `bench_reorder` is the strongest current reorder/fill artifact because it
  already emits stable CSV rows with matrix, ordering, `nnz_L`, local
  `reorder_ms`, optional factor timing, path, slice, and ND threshold fields.
- `bench_fillin`, `bench_colamd`, and `bench_amd_qg` are useful adjacent
  evidence, but their output schemas and claim roles differ from
  `bench_reorder`.
- AMD/qg-AMD has strong implementation and stress ownership, including the
  10k banded test and `bench_amd_qg`, but its memory/workspace interpretation
  is split across header prose, tests, and historical benchmark comments.
- COLAMD is correctly separated from symmetric reorderings in public API docs,
  tests, and benchmark docs, but its QR fill reporting is still
  human-readable rather than artifact-contract-owned.
- ND and graph partition coverage is strong but concentrated in large proof
  owners with many historical comments. Cleanup should be tied to future
  touched implementation boundaries.
- The first Sprint 105 fix-now item should be the canonical fill/runtime/memory
  and fixture naming contract, with `bench_reorder` as the likely first
  implementation lane.

### Validation Expectations

- Day 2 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 2 Exit State

Day 2 is complete. Sprint 105 now has a live reorder/graph owner inventory,
current evidence map, ranked gap list, fix-now queue, deferred queue, and
Day 3 field-contract starting point.

## Day 3 - Fill and Fixture Contract Design

### Goal

Define canonical fill, runtime, memory, fixture naming, ordering, skip, and
reviewed-status fields for Sprint 105 reorder and graph artifacts before
implementation changes begin.

### Actions

- Re-read the Day 3 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read Day 2's audit and the Sprint 100 benchmark interpretation template.
- Reviewed current live output contracts for:
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
  - `benchmarks/bench_colamd.c`
  - `benchmarks/bench_fillin.c`
  - `scripts/wall_check.sh`
  - `scripts/performance_sentinels.sh`
- Defined canonical fields for fixture identity, ordering identity, fill
  metrics, local runtime, memory proxies, skip/error rows, thresholds, and
  reviewed status.
- Defined fixture naming rules for named SuiteSparse matrices, generated graph
  families, and synthetic stress fixtures.
- Wrote `artifacts/day3-fill-fixture-contract.md`.

### Findings

- `bench_reorder` is still the best first implementation lane because its CSV
  already maps cleanly to fixture, ordering, fill, runtime, path, slice, and ND
  policy fields.
- `bench_amd_qg` has the only current memory-proxy field
  (`peak_rss_mb`), but it remains a historical qg-AMD versus bitset foil and
  should not become canonical without a fresh decision.
- `bench_colamd` and `bench_fillin` expose useful fill evidence, but their
  current human-readable text output should remain adjacent until the
  `bench_reorder` contract is stable.
- Skip/error rows must use the same fixture and ordering identifiers as report
  rows. This is necessary for aggregation and matches recent Sprint 104
  sentinel review lessons.
- Thresholded statuses require a baseline and threshold source. Report-only
  rows must not imply pass/fail behavior.

### Validation Expectations

- Day 3 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 3 Exit State

Day 3 is complete. Sprint 105 now has a fill, runtime, memory, fixture,
ordering, skip/error, reviewed-status, and implementation contract for reorder
and graph artifacts before Day 4 selects the first evidence set.

## Day 4 - Evidence Boundary and Matrix Selection

### Goal

Select the named matrices, generated graph families, size tiers, commands, and
artifact boundaries that Sprint 105 will use for bounded reorder, graph, fill,
and large-matrix scalability evidence.

### Actions

- Re-read the Day 4 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Compared the Day 2 live surface audit with the Day 3 fill and fixture
  contract.
- Inspected the committed SuiteSparse fixture set under
  `tests/data/suitesparse/`.
- Re-read current `bench_reorder` fixture ownership and generated fixture
  builders in graph/reorder tests.
- Selected the existing `bench_reorder --sprint86-slice --skip-factor` lane as
  the reviewed first named-matrix slice.
- Selected deterministic generated family candidates from existing test
  builders rather than adding new random fixtures.
- Defined smoke, reviewed, supplemental, and local-only size tiers.
- Wrote `artifacts/day4-evidence-boundary.md`.

### Findings

- The first implementation lane should be `bench_reorder` contract alignment,
  not a broad migration of all reorder-adjacent benchmarks.
- The reviewed first slice should stay bounded to `bcsstk14` and
  `Pres_Poisson`, using the existing `make bench-reorder-sprint86` target.
- The current `sprint86` slice label should remain compatible for now but be
  documented as a historical label for the current bounded two-fixture slice.
- Generated families should start from existing deterministic builders:
  `path1d`, `grid2d`, `banded`, `two_cliques`, and `arrow`.
- `bench_amd_qg`, `bench_colamd`, and `bench_fillin` remain supplemental or
  deferred until the first `bench_reorder` contract pass is stable.

### Validation Expectations

- Day 4 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 4 Exit State

Day 4 is complete. Sprint 105 now has frozen named-matrix, generated-family,
size-tier, command, artifact, and deferred-lane boundaries before source or
script implementation begins.

## Day 5 - Reorder/Fill Reporting Batch 1

### Goal

Validate the first Sprint 105 reorder/fill reporting batch against the Day 3
field contract and Day 4 bounded evidence boundary.

### Actions

- Re-read the Day 5 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read the Day 3 fill and fixture contract and the Day 4 evidence boundary.
- Inspected `benchmarks/bench_reorder.c`, `benchmarks/README.md`, and
  `docs/maintainer_guide.md` for current `bench_reorder` schema ownership.
- Confirmed the current `bench_reorder` CSV header already matches the Day 4
  selected first-lane fields.
- Preserved the existing `make bench-reorder-sprint86` target,
  `--sprint86-slice` flag, and `fixture_slice=sprint86` compatibility label.
- Ran `make bench-reorder-sprint86` to regenerate the bounded sample output.
- Ran a focused live parser smoke check against
  `build/bench_reorder --sprint86-slice --skip-factor`.
- Wrote `artifacts/day5-reorder-fill-reporting-batch1.md`.

### Findings

- No source schema migration was necessary for Day 5 because `bench_reorder`
  already emits:
  `matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold`.
- The Day 5 path remains threshold-free report evidence. It should not be
  interpreted as portable performance evidence.
- `factor_ms=skip` is intentional for the bounded reviewed first slice because
  Day 5 uses `--skip-factor`.
- The focused smoke proof should guard the exact header, 10-row output shape,
  selected fixtures, five ordering labels, direct path, sprint86 slice label,
  and current ND base threshold.

### Validation Expectations

- Day 5 changes planning documentation only.
- Required checks:
  - `make bench-reorder-sprint86`
  - focused live parser smoke check for the bounded CSV output
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`
- Full C quality gate is not required unless `.c` or `.h` files are modified.

### Validation Results

- `make bench-reorder-sprint86`: passed; emitted 10 bounded data rows for
  `bcsstk14` and `Pres_Poisson` across `none`, `rcm`, `amd`, `colamd`, and
  `nd`.
- Focused live parser smoke check: passed; header, row count, fixture/order
  coverage, `factor_ms=skip`, `reorder_path=direct`,
  `fixture_slice=sprint86`, and `nd_base_threshold=160` matched expectations.
- `git diff --check -- docs/planning/EPIC_10/SPRINT_105`: passed after
  temporarily marking untracked Sprint 105 files intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 5 Exit State

Day 5 is complete. Sprint 105 now has the first bounded `bench_reorder`
reporting-batch proof, regenerated sample output, parser smoke expectations,
and validation notes without changing the source reporting schema.

## Day 6 - Named-Matrix Evidence Expansion

### Goal

Expand Sprint 105 reorder/fill evidence from the bounded two-fixture Day 5
proof to the full committed `bench_reorder --skip-factor` named-matrix slice.

### Actions

- Re-read the Day 6 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-used the Day 3 metric contract, Day 4 evidence boundary, and Day 5
  reporting-batch proof as the governing inputs.
- Ran `build/bench_reorder --skip-factor` to capture all six committed
  `bench_reorder` fixtures across `none`, `rcm`, `amd`, `colamd`, and `nd`.
- Calculated `ratio_to_none` and `ratio_to_amd` from the captured `nnz_L`
  values.
- Recorded skipped and unavailable lanes for numeric factor timing,
  analyze-path reorder evidence, QR/COLAMD `nnz_R`, LU `nnz_LU`, and external
  matrices outside committed fixtures.
- Ran a focused live parser validation over
  `build/bench_reorder --skip-factor`.
- Wrote `artifacts/day6-named-matrix-evidence.md`.

### Findings

- The full `bench_reorder --skip-factor` slice emits 30 rows: six named
  fixtures times five ordering labels.
- AMD is the strongest `nnz_L` row on most fixtures in this direct-path slice.
- ND ties AMD on `nos4` and `bcsstk04`, stays close on `s3rmt3m3`, and beats
  AMD on `Pres_Poisson`.
- `Kuu` remains a useful bimodal-degree stress fixture because ND is still
  `1.855x` AMD fill in the captured direct-path evidence.
- Day 6 remains report-only evidence. Local `reorder_ms` rows are context, not
  portable timing claims.

### Validation Expectations

- Day 6 changes planning documentation only.
- Required checks:
  - `build/bench_reorder --skip-factor`
  - focused live parser validation for the full named-matrix CSV output
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`
- Full C quality gate is not required unless `.c` or `.h` files are modified.

### Validation Results

- `build/bench_reorder --skip-factor`: passed; emitted 30 named-matrix rows
  for `nos4`, `bcsstk04`, `Kuu`, `bcsstk14`, `s3rmt3m3`, and
  `Pres_Poisson`.
- Focused live parser validation: passed; header, row count, fixture/order
  coverage, `factor_ms=skip`, `reorder_path=direct`, `fixture_slice=all`, and
  `nd_base_threshold=160` matched expectations.
- `git diff --check -- docs/planning/EPIC_10/SPRINT_105`: passed after
  temporarily marking untracked Sprint 105 files intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 6 Exit State

Day 6 is complete. Sprint 105 now has full named-matrix `bench_reorder`
evidence, fill ratios, local runtime context, explicit skipped/deferred lanes,
and focused validation output.

## Day 7 - Generated Graph-Family Expansion

### Goal

Refresh deterministic generated-family evidence for graph partition,
separator, quotient-graph AMD, and nested-dissection behavior using the
families selected on Day 4.

### Actions

- Re-read the Day 7 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read the Day 4 generated-family boundary and size-tier table.
- Inspected generated fixture ownership in `tests/test_graph.c`,
  `tests/test_reorder_nd.c`, `tests/test_reorder_amd_qg.c`,
  `benchmarks/bench_amd_qg.c`, and `benchmarks/bench_fillin.c`.
- Ran `make build/test_graph && ./build/test_graph`.
- Ran `make build/test_reorder_nd && ./build/test_reorder_nd`.
- Ran `make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg`.
- Captured deterministic evidence for path, 2D grid, 3D mesh, clique-bridge,
  and banded generated families.
- Deferred `arrow-n<N>` canonical evidence because its current owner is the
  human-readable `bench_fillin` LU context.
- Wrote `artifacts/day7-generated-graph-family-evidence.md`.

### Findings

- Existing focused tests already cover more than the minimum two generated
  structural families required by Day 7.
- `test_graph` passed 61 tests and covers generated grids, paths,
  clique-bridge separators, 3D mesh separators, policy-difference behavior,
  and determinism.
- `test_reorder_nd` passed with one explicit skip and covers `grid2d-4x4`,
  `grid2d-10x10`, `path1d-n20`, and `banded-n256-bw8` ND/factor-dispatch
  behavior.
- `test_reorder_amd_qg` passed and covers the `banded-n10000-bw5`
  quotient-graph AMD large generated guardrail.
- No source changes were needed. The Day 7 evidence stays in reviewed
  test-local or supplemental artifact lanes.

### Validation Expectations

- Day 7 changes planning documentation only.
- Required checks:
  - `make build/test_graph && ./build/test_graph`
  - `make build/test_reorder_nd && ./build/test_reorder_nd`
  - `make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg`
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`
- Full C quality gate is not required unless `.c` or `.h` files are modified.

### Validation Results

- `make build/test_graph && ./build/test_graph`: passed; 61 tests, 0 failed,
  0 skipped.
- `make build/test_reorder_nd && ./build/test_reorder_nd`: passed; 35 tests,
  0 failed, 1 explicit skip.
- `make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg`: passed; 7
  tests, 0 failed, 0 skipped.
- `git diff --check -- docs/planning/EPIC_10/SPRINT_105`: passed after
  temporarily marking untracked Sprint 105 files intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 7 Exit State

Day 7 is complete. Sprint 105 now has deterministic generated-family evidence
for path, grid, mesh, clique-bridge, and banded families, with focused test
proofs and explicit deferred lanes.

## Day 8 - Large-Matrix Guardrail Design

### Goal

Define deterministic large-matrix memory, runtime, fill, and structural
guardrails that fit reviewed, supplemental, or local-only Sprint 105 lanes.

### Actions

- Re-read the Day 8 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-used Day 6 named-matrix evidence and Day 7 generated-family evidence as
  guardrail inputs.
- Inspected existing `wall-check` and `performance-sentinels` behavior in the
  Makefile and helper scripts.
- Classified large-matrix risks: memory growth, integer overflow, recursion or
  stack pressure, pathological fill, runtime cliffs, separator degeneracy, and
  local-only lane drift.
- Separated reviewed structural lanes from supplemental report lanes and
  local-only noisy lanes.
- Defined threshold and skip rules that preserve the existing hard timing gate
  boundary.
- Wrote `artifacts/day8-large-matrix-guardrail-design.md`.

### Findings

- New pass/fail timing thresholds are not justified without a fresh baseline
  and machine-class assumption.
- Existing structural tests already provide the best reviewed guardrail owners
  for `banded-n10000-bw5`, `grid2d-10x10`, `grid2d-30x30`,
  `path1d-n20`, `mesh3d-5x5x5`, and `two_cliques-k10`.
- `make wall-check` remains the only current hard timing gate.
- `bench_reorder --skip-factor`, `bench_amd_qg --skip-bitset`, and
  `make performance-sentinels` are useful supplemental report lanes, but they
  should not become hidden reviewed requirements.
- Day 9 should prefer narrow structural guardrail implementation or command
  aggregation over broad benchmark/schema migration.

### Validation Expectations

- Day 8 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_105`
- Full C quality gate is not required unless `.c` or `.h` files are modified.

### Validation Results

- `git diff --check -- docs/planning/EPIC_10/SPRINT_105`: passed after
  temporarily marking untracked Sprint 105 files intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105`: passed; no
  matches.

### Day 8 Exit State

Day 8 is complete. Sprint 105 now has a large-matrix guardrail design that
separates deterministic reviewed structural checks from supplemental
report-only lanes and local-only noisy lanes.

## Day 9 - Scalability Guardrail Implementation

### Goal

Implement the selected deterministic large-matrix guardrail batch while
keeping supplemental report lanes opt-in and threshold-free.

### Actions

- Re-read the Day 9 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read the Day 8 large-matrix guardrail design.
- Added `scripts/large_matrix_guardrails.sh`.
- Added the `make large-matrix-guardrails` target.
- Kept reviewed default lanes to:
  - `build/test_graph`
  - `build/test_reorder_nd`
  - `build/test_reorder_amd_qg`
  - `build/bench_reorder --sprint86-slice --skip-factor`
- Kept supplemental lanes opt-in with
  `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1`.
- Ran `bash -n scripts/large_matrix_guardrails.sh`.
- Ran `make large-matrix-guardrails`.
- Wrote `artifacts/day9-scalability-guardrail-implementation.md`.

### Findings

- The new target writes a deterministic report bundle under
  `build/bench-reports/large-matrix-guardrails`.
- The default reviewed run records `G1` through `G4` as pass rows and records
  supplemental lanes `S1` and `S2` as explicit opt-in skips.
- The script validates the bounded `bench_reorder` CSV shape before marking
  `G4` as passed.
- No new timing or max-RSS thresholds were introduced.
- Supplemental full named-matrix and qg-AMD report lanes remain available
  without becoming hidden quality requirements.

### Validation Expectations

- Day 9 changes a script, the Makefile, and planning documentation.
- Required checks:
  - `bash -n scripts/large_matrix_guardrails.sh`
  - `make large-matrix-guardrails`
  - `git diff --check`
  - trailing-whitespace scan on touched planning docs and script
- Full C quality gate is not required unless `.c` or `.h` files are modified.

### Validation Results

- `bash -n scripts/large_matrix_guardrails.sh`: passed.
- `make large-matrix-guardrails`: passed; wrote:
  - `build/bench-reports/large-matrix-guardrails/index.tsv`
  - `build/bench-reports/large-matrix-guardrails/manifest.txt`
  - `build/bench-reports/large-matrix-guardrails/test_graph.txt`
  - `build/bench-reports/large-matrix-guardrails/test_reorder_nd.txt`
  - `build/bench-reports/large-matrix-guardrails/test_reorder_amd_qg.txt`
  - `build/bench-reports/large-matrix-guardrails/bench_reorder_sprint86.csv`
- `git diff --check`: passed after temporarily marking untracked Sprint 105
  files and `scripts/large_matrix_guardrails.sh` intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105 scripts/large_matrix_guardrails.sh Makefile`:
  passed; no matches.

### Day 9 Exit State

Day 9 is complete. Sprint 105 now has an implemented
`make large-matrix-guardrails` target with reviewed structural guardrails,
opt-in supplemental report lanes, generated report artifacts, and residual
large-matrix queue documentation.

## Day 10 - Graph/Reorder Ownership Cleanup

### Goal

Clean up a touched graph/reorder proof owner so comments describe current
ownership and maintained behavior rather than day-by-day implementation
history.

### Actions

- Re-read the Day 10 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Audited graph/reorder source and test owners for stale sprint-history
  comments and unclear proof ownership.
- Selected `tests/test_reorder_amd_qg.c` as the narrow Day 10 cleanup owner
  because it is small, recently used by the Day 9 guardrail target, and had
  stale day-labeled comments.
- Rewrote the qg-AMD proof-owner header around current maintained contracts.
- Replaced stale stub-retirement and day-labeled stress comments with current
  argument-validation, wrapper-delegation, and large-regular-input guardrail
  descriptions.
- Renamed the test suite banner to current ownership wording.
- Left helper extraction out of scope because the local helpers are small and
  extraction would add surface area without reducing meaningful duplication.
- Ran the focused qg-AMD proof-owner test.
- Wrote `artifacts/day10-graph-reorder-ownership-cleanup.md`.

### Findings

- `tests/test_graph.c`, `tests/test_reorder_nd.c`,
  `src/sparse_graph.c`, and `src/sparse_reorder_nd.c` still carry substantial
  history-heavy comments, but they are too large for an opportunistic cleanup
  pass while guardrail behavior is still being stabilized.
- `tests/test_reorder_amd_qg.c` was the right first cleanup owner because its
  current proof surface is clear: argument validation, wrapper delegation,
  symbolic fill equality, and `banded-n10000-bw5` structural guardrail.
- No behavior changes were needed.

### Validation Expectations

- Day 10 modifies a `.c` file, so required checks are:
  - `make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan on touched docs/source/script files

### Validation Results

- `make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg`: passed; 7
  tests, 0 failed, 0 skipped.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed after temporarily marking untracked Sprint 105
  files and `scripts/large_matrix_guardrails.sh` intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105 scripts/large_matrix_guardrails.sh Makefile tests/test_reorder_amd_qg.c`:
  passed; no matches.

### Day 10 Exit State

Day 10 is complete. Sprint 105 now has a cleaned qg-AMD proof-owner comment
surface, preserved behavior, focused validation, full C quality-gate
validation, and a residual graph/reorder cleanup queue.

## Day 11 - Reporting and Documentation Alignment

### Goal

Consolidate reorder, fill, graph, and scalability reporting guidance for users
and maintainers so the implemented Sprint 105 artifacts are interpreted within
their reviewed, supplemental, and local-only boundaries.

### Actions

- Re-read the Day 11 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read the Day 3 fill/fixture contract, Day 8 guardrail design, and Day 9
  guardrail implementation artifact.
- Updated `docs/algorithm.md` with a user-facing reorder/fill reporting
  interpretation section.
- Updated `benchmarks/README.md` with `make large-matrix-guardrails` command,
  output, reviewed-lane, and supplemental-mode guidance.
- Updated `docs/maintainer_guide.md` with large-matrix structural guardrail
  ownership and non-claim rules.
- Wrote `artifacts/day11-reporting-and-documentation-alignment.md`.

### Findings

- Existing docs already separated canonical performance reports and
  performance sentinels; the missing piece was explicit large-matrix guardrail
  documentation.
- The `sprint86` fixture-slice label remains historical in `bench_reorder`,
  so docs now describe it as the current bounded two-fixture slice.
- No benchmark schema migration was needed for Day 11; the existing
  implemented target and artifact names were sufficient.
- No source edits were required.

### Validation Expectations

- Day 11 modifies documentation only, so required checks are:
  - `git diff --check`
  - trailing-whitespace scan on touched documentation and Sprint 105 artifacts
- Full C quality gate is not required because no `.c` or `.h` files were
  modified.

### Validation Results

- `git diff --check`: passed after temporarily marking untracked Sprint 105
  files and `scripts/large_matrix_guardrails.sh` intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/algorithm.md benchmarks/README.md docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_105`:
  passed; no matches.

### Day 11 Exit State

Day 11 is complete. Sprint 105 now has aligned user, benchmark, and maintainer
documentation for reorder/fill metrics, reviewed large-matrix guardrails,
supplemental report lanes, and timing/memory non-claims.

## Day 12 - Integrated Evidence Reconciliation

### Goal

Reconcile named-matrix, generated-family, guardrail, and reporting artifacts
into one coherent Sprint 105 evidence package.

### Actions

- Re-read the Day 12 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read Day 5, Day 6, Day 7, Day 8, Day 9, and Day 11 artifacts.
- Re-ran the full named-matrix evidence lane with
  `build/bench_reorder --skip-factor`.
- Re-ran the implemented large-matrix guardrail bundle with
  `make large-matrix-guardrails`.
- Checked the full named-matrix CSV contract with the Day 6 parser shape.
- Checked the generated guardrail `index.tsv` for reviewed pass rows and
  supplemental skip rows.
- Checked the bounded `bench_reorder_sprint86.csv` shape directly from the
  generated report directory.
- Wrote `artifacts/day12-integrated-evidence-reconciliation.md`.

### Findings

- The full named-matrix report preserved the Day 6 structural values:
  - `nos4`: AMD and ND tie at `637`;
  - `bcsstk04`: AMD and ND tie at `3143`;
  - `Kuu`: AMD remains strongest at `406264`, ND remains `753755`;
  - `bcsstk14`: AMD remains strongest at `116071`, ND remains `132634`;
  - `s3rmt3m3`: AMD remains strongest at `474609`, ND remains `484890`;
  - `Pres_Poisson`: ND remains strongest at `2474435`.
- `make large-matrix-guardrails` regenerated:
  - `index.tsv`;
  - `manifest.txt`;
  - `test_graph.txt`;
  - `test_reorder_nd.txt`;
  - `test_reorder_amd_qg.txt`;
  - `bench_reorder_sprint86.csv`.
- The guardrail index preserved the intended status split:
  - reviewed lanes `G1` through `G4`: `pass`;
  - supplemental lanes `S1` and `S2`: `skip`.
- Local runtime values varied across reruns, as expected; no runtime value was
  promoted to portable performance evidence.
- No immediate source or documentation contradiction required a Day 12 fix.

### Validation Expectations

- Day 12 modifies planning documentation only, so required checks are:
  - rerun selected evidence commands;
  - check generated CSV/index contracts;
  - `git diff --check`;
  - trailing-whitespace scan on touched documentation and Sprint 105 artifacts.
- Full C quality gate is not required because no `.c` or `.h` files were
  modified.

### Validation Results

- `make build/bench_reorder && build/bench_reorder --skip-factor`: passed.
- `make large-matrix-guardrails`: passed.
- full named-matrix CSV contract parser: passed.
- guardrail `index.tsv` contract parser: passed.
- guardrail `bench_reorder_sprint86.csv` contract parser: passed.
- strict generated-report failure scan: passed; no matches.
- `git diff --check`: passed after temporarily marking untracked Sprint 105
  files and `scripts/large_matrix_guardrails.sh` intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105 docs/algorithm.md benchmarks/README.md docs/maintainer_guide.md`:
  passed; no matches.

### Day 12 Exit State

Day 12 is complete. Sprint 105 now has reconciled named-matrix,
generated-family, guardrail, and reporting evidence with parser-checked
contracts, no immediate contradictions requiring fixes, and explicit residual
fix candidates for Day 13/14 consideration.

## Day 13 - Final Fix Batch and Validation Sweep

### Goal

Resolve the highest-priority Day 12 contradictions, rerun focused validation
for every touched surface, run the broader required gate for the branch's C
test change, and record the final residual queue.

### Actions

- Re-read the Day 13 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read the Day 12 reconciliation artifact and contradiction queue.
- Chose a no-op implementation fix batch because Day 12 had no immediate
  source or documentation contradiction requiring a change.
- Ran `bash -n scripts/large_matrix_guardrails.sh`.
- Ran `make large-matrix-guardrails`.
- Ran the full required C quality gate:
  `make format && make lint && make test`.
- Wrote `artifacts/day13-final-validation-and-residual-queue.md`.

### Findings

- The historical `sprint86` label remains a documented compatibility label,
  not a defect requiring Day 13 code churn.
- The supplemental `S1` and `S2` guardrail lanes remain opt-in and should not
  be promoted without fresh baseline design.
- Runtime drift across reruns remained local context only.
- `make large-matrix-guardrails` regenerated the expected report bundle and
  preserved the reviewed/supplemental status split.
- The full C quality gate passed after the branch's qg-AMD test comment
  cleanup.

### Validation Expectations

- Day 13 touches planning documentation and the branch already includes a `.c`
  test-file change, so required checks are:
  - `bash -n scripts/large_matrix_guardrails.sh`
  - `make large-matrix-guardrails`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan on touched documentation, script, Makefile, and
    source/test files

### Validation Results

- `bash -n scripts/large_matrix_guardrails.sh`: passed.
- `make large-matrix-guardrails`: passed.
- `make format && make lint && make test`: passed.
- `git diff --check`: passed after temporarily marking untracked Sprint 105
  files and `scripts/large_matrix_guardrails.sh` intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105 docs/algorithm.md benchmarks/README.md docs/maintainer_guide.md scripts/large_matrix_guardrails.sh Makefile tests/test_reorder_amd_qg.c`:
  passed; no matches.

### Day 13 Exit State

Day 13 is complete. Sprint 105 now has a no-op final fix decision backed by
focused guardrail validation, the full required C quality gate, final
documentation hygiene checks, and an explicit residual queue for closeout.

## Day 14 - Sprint 105 Closeout and Handoff

### Goal

Close Sprint 105 with validated artifacts, documentation, and a clear Sprint
106 handoff queue.

### Actions

- Re-read the Day 14 plan section in
  `docs/planning/EPIC_10/SPRINT_105/PLAN.md`.
- Re-read the Sprint 105 project-plan items in
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Reviewed all Sprint 105 artifact files and Day 13 validation state.
- Wrote `artifacts/day14-closeout-and-handoff.md` with:
  - project-plan item closeout status;
  - artifact index;
  - implemented surface summary;
  - final validation summary;
  - non-claims;
  - Sprint 106 handoff queue;
  - retrospective inputs.

### Findings

- Every Sprint 105 project-plan item has a completed, deferred, or non-claim
  status.
- The implemented guardrail target, documentation updates, qg-AMD ownership
  cleanup, and final validation are all represented in the closeout artifact.
- Remaining work is intentionally residual and should not be treated as hidden
  Sprint 105 scope.
- Day 14 changes planning documentation only.

### Validation Expectations

- Day 14 modifies planning documentation only, so required checks are:
  - `git diff --check`
  - trailing-whitespace scan on touched planning docs and already-touched
    Sprint 105 surfaces
- Full C quality gate is not required for Day 14 because no `.c` or `.h` files
  were modified after the Day 13 full gate.

### Validation Results

- `git diff --check`: passed after temporarily marking untracked Sprint 105
  files and `scripts/large_matrix_guardrails.sh` intent-to-add for coverage.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105 docs/algorithm.md benchmarks/README.md docs/maintainer_guide.md scripts/large_matrix_guardrails.sh Makefile tests/test_reorder_amd_qg.c`:
  passed; no matches.

### Day 14 Exit State

Day 14 is complete. Sprint 105 is closed from validated evidence, explicit
project-plan item status, a Sprint 106 handoff queue, retrospective inputs,
and final closeout hygiene checks.
