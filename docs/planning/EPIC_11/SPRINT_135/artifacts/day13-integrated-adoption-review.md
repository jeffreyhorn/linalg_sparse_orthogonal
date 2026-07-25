# Sprint 135 Day 13 - Integrated Adoption Review

## Purpose

Review the simplified adoption surface as both a first-use reader and a
maintainer. Day 13 checks whether the previous split, cookbook, report-index,
and navigation edits form a coherent path rather than just a set of valid
links.

## First-Use Walkthrough

| Reader question | Entry point | Next handoff | Result |
|---|---|---|---|
| I want one local build and solve. | `README.md` Start Here and Quick Start | `examples/README.md` | Clear. README keeps quick success first and examples provide runnable next steps. |
| I need to choose a solver. | `README.md` Choose a Workflow | `docs/solver_selection.md` | Clear. The decision tree starts from matrix arrival and problem shape. |
| My data is CSR or CSC. | `README.md` Start Here / Adoption Map | `docs/cookbook.md` | Clear. Cookbook starts from compressed input and routes to solver families. |
| My matrix is Matrix Market. | `docs/cookbook.md` and `examples/README.md` | `docs/matrix_market.md` and `example_matrix_market` | Clear. Format authority and runnable load/use example are both linked. |
| I need install/downstream use. | `README.md` Start Here / Adoption Map | `INSTALL.md` | Clear. Install support remains out of cookbook and example docs. |
| I need current algorithm details. | `README.md` Adoption Map / docs index | `docs/algorithm.md` | Clear after Day 13 title fix. |

Small fix applied:

- renamed `docs/algorithm.md` heading from `Algorithm Description` to
  `Algorithm Reference`
- tightened the intro to make current-reference ownership explicit and point
  historical measurements to `docs/algorithm_history.md`

## Benchmark And Report Walkthrough

| Reader question | Entry point | Handoff | Result |
|---|---|---|---|
| I chose an API workflow and now need measurement. | `docs/cookbook.md` Measure section | `benchmarks/README.md` | Clear. Cookbook maps workflow to benchmark family without duplicating command docs. |
| I need generated canonical reports. | `benchmarks/README.md` report handoff | `build/bench-reports/canonical/index.tsv`, `manifest.txt` | Clear. Report is described as threshold-free local snapshot. |
| I need local sentinel behavior. | `benchmarks/README.md` report handoff | `build/bench-reports/sentinels/sentinels.tsv`, `manifest.txt` | Clear. Only wall-check is framed as thresholded. |
| I need large-matrix guardrail artifacts. | `benchmarks/README.md` report handoff | `build/bench-reports/large-matrix-guardrails/index.tsv`, `manifest.txt` | Clear. Reviewed/supplemental rows and skip semantics remain visible. |

Benchmark/report wording remains evidence-bounded:

- examples teach API usage
- tests own regression behavior
- benchmark rows are local measurement artifacts
- generated indexes/manifests are artifact maps and freshness context
- cross-report normalized indexing remains deferred

## Maintainer And Historical Walkthrough

| Maintainer need | Entry point | Handoff | Result |
|---|---|---|---|
| Current algorithm behavior | README docs index / tutorial map | `docs/algorithm.md` | Clear. Current reference is discoverable without historical detail first. |
| Historical measurement context | `docs/algorithm.md` intro and section links | `docs/algorithm_history.md` | Clear. Appendix preserves measurement chronology and caveats. |
| Support-tier and package truth | README/INSTALL navigation | `INSTALL.md`, `docs/maintainer_guide.md` | Clear. Install remains static-first and platform-tier wording is unchanged. |
| Quality-policy interpretation | README Adoption Map / tutorial map | `docs/maintainer_guide.md` | Clear. Maintainer guide remains findable without crowding first-use docs. |

## Compressed-First Discoverability Matrix

| Workflow family | Visible from README | Visible from cookbook | Runnable/example handoff | Benchmark/report handoff |
|---|---|---|---|---|
| Direct | Yes | Yes | `example_compressed_input`, `example_basic_solve`, `example_analysis` | `bench_main`, `bench_refactor`, `bench_refactor_csc` |
| Iterative | Yes | Yes | `example_iterative`, `example_ic_minres`, `example_matrix_free` | `bench_iterative_reuse` |
| Matrix Market | Yes | Yes | `example_matrix_market` | Route by chosen API workflow |
| SVD / low-rank | Yes | Yes | `example_svd_lowrank` | `bench_svd` |
| Symmetric eigensolver | Yes | Yes | `example_eigs` | `bench_eigs`, `bench_eigs_reuse` |
| Benchmark/report | Yes | Yes | `benchmarks/README.md` | canonical, sentinel, and guardrail report artifacts |

## Final Small-Fix Queue

Applied during Day 13:

- `docs/algorithm.md` title and intro alignment

No additional Day 13 cleanup is required before closeout. Day 14 should focus
on metrics, project-plan reconciliation, final claim-boundary summary, and
Sprint 136 handoff.

## Completion Criteria

- adoption docs have a coherent reader path from first contact to examples
- maintainer and historical material is discoverable without crowding first-use
  guidance
- all compressed-first workflow families have visible entry points
