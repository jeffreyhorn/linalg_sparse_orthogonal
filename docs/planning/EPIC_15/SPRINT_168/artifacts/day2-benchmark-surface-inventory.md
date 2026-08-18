# Sprint 168 Day 2: Benchmark Surface Inventory

## Purpose

Day 2 inventories the current benchmark/report command surface, generated
output conventions, documentation boundaries, and reusable freshness patterns
before Sprint 168 selects the hosted performance publication lane.

## Makefile Benchmark Owners

| Target or variable | Current role | Sprint 168 implication |
| --- | --- | --- |
| `bench` | Runs all benchmark binaries. | Too broad and historically too slow for hosted publication. Do not use as the Sprint 168 hosted lane. |
| `bench-build` | Builds benchmark binaries without running them. | Existing compile-drift guard; useful prerequisite but not performance evidence. |
| `tooling-build` | Builds benchmark and example binaries without running workloads. | Standard compile-only support check; not a performance publication owner. |
| `bench-fast` | Runs a small CI-friendly benchmark subset plus `bench_reorder --skip-factor`. | Runtime smoke evidence only. It must not become the methodology-bound publication lane by itself. |
| `BENCH_CANONICAL_REPORT_BINS` | Defines canonical report binaries: `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, `bench_eigs_reuse`. | Primary candidate pool for Sprint 168. |
| `BENCH_CANONICAL_REPORT_DIR` | Writes canonical outputs under `build/bench-reports/canonical`. | Existing ignored generated-output location for selected hosted artifacts. |
| `BENCH_CANONICAL_REPORT_LABEL` | Optional bounded label passed to canonical report metadata. | Reusable for CI labels such as branch, PR, lane, or hosted-run scope. |
| `bench-canonical-report` | Generates four threshold-free canonical CSVs plus `index.tsv` and `manifest.txt`. | Strongest existing owner for Sprint 168 hosted performance publication. |
| `performance-sentinels` | Generates local sentinel bundle and runs existing wall-check hard gate. | Useful adjacent evidence; not the first hosted publication candidate because it mixes hard gate and advisory measurements. |
| `large-matrix-guardrails` | Generates structural guardrail reports and optional supplemental lanes. | Out of scope for Sprint 168 performance publication unless later narrowed. |

## Canonical Report Script Inventory

| Script behavior | Current state | Sprint 168 reuse |
| --- | --- | --- |
| Entry point | `scripts/bench_canonical_report.sh <report_dir> <bench_refactor_csc> <bench_chol_csc> <bench_iterative_reuse> <bench_eigs_reuse>` | Can be wrapped by a more selective target or reused as the report generator. |
| Generated CSVs | `bench_refactor_csc.csv`, `bench_chol_csc.csv`, `bench_iterative_reuse.csv`, `bench_eigs_reuse.csv` | Sprint 168 should select one row/family or keep all four only if runtime is bounded. |
| Fixed fixture commands | `bench_refactor_csc` and `bench_chol_csc` use `tests/data/suitesparse/nos4.mtx --repeat 1`; reuse/eigs benchmarks use defaults. | `bench_refactor_csc` already has a concrete fixture and repeat count suitable for Day 3 scoring. |
| Metadata files | `index.tsv` and `manifest.txt` | Existing metadata structure can support hosted freshness if required fields are tightened. |
| Context fields | timestamp, git commit, git branch, platform, compiler, build mode, `OMP_NUM_THREADS`, artifact, relative path, command, fixture/workload, repeat semantics. | Good starting metadata set for hosted publication. |
| Claim fields | `report_family=benchmark`, `status=measurement`, `support_tier=local_only`, `claim_boundary=local_threshold_free`, `baseline=n/a`, `threshold=n/a`, `methodology_notes=threshold_free_local_measurement;not_portable_performance_claim`. | Sprint 168 must update or supplement support-tier/claim-boundary semantics for the hosted selected lane without implying superiority. |
| Control-character guards | Rejects tabs/newlines/carriage returns in TSV metadata fields. | Reusable safety pattern for any new hosted metadata fields. |
| Build-mode detection | Uses `SPARSE_CANONICAL_BUILD_MODE` override or OpenMP runtime detection from binaries. | Useful for CI; Day 5 should decide whether hosted lane sets the override explicitly. |

## Performance Sentinel Script Inventory

| Script behavior | Current state | Sprint 168 implication |
| --- | --- | --- |
| Entry point | `scripts/performance_sentinels.sh` via `make performance-sentinels`. | Adjacent evidence, not the preferred Day 3 lane. |
| Hard gate | Existing `wall-check` rows are thresholded local wall gates. | Do not mix hard gate semantics with threshold-free publication unless explicitly selected. |
| Advisory rows | Cholesky CSC and LDLT KKT context rows are threshold-free. | Useful methodology examples for support tier, claim boundary, repeat semantics, warmup, variance, and backend context. |
| Generated output | `build/bench-reports/sentinels/sentinels.tsv`, `manifest.txt`, and CSV/text artifacts. | Keep generated outputs ignored and do not commit local sentinel artifacts. |
| Claim boundary | Script comments explicitly say local regression evidence, not portable performance. | Preserve this wording; hosted publication should remain methodology-bound. |

## Documentation Inventory

| Document | Current benchmark/performance boundary | Sprint 168 implication |
| --- | --- | --- |
| `README.md` | Keeps only a high-level performance story and points detailed benchmark interpretation to `benchmarks/README.md`. Describes `make bench-canonical-report` as a bounded local snapshot and generated rows as branch-local measurement artifacts, not portable guarantees. | Update only after hosted lane exists; keep README short and scoped. |
| `benchmarks/README.md` | Defines benchmarks as local measurement tools, lists canonical/runtime/exploratory split, documents `bench-canonical-report`, `performance-sentinels`, generated metadata, report-index handoff, and non-portable interpretation. | Primary docs owner for Sprint 168 hosted/local distinction and selected lane interpretation. |
| `docs/maintainer_guide.md` | Interprets reviewed proof and maintainer policy surfaces. | Update only if hosted lane changes reviewed/supplemental evidence classification. |
| `tests/corpus/manifests/report_families.tsv` | Contains benchmark canonical row family as generated local/advisory evidence with `make bench-canonical-report` and non-claims. | Candidate metadata owner for promoted hosted status, if Sprint 168 chooses to register hosted benchmark freshness. |

## Generated Output Conventions

| Output | Current path | Current interpretation |
| --- | --- | --- |
| Canonical benchmark CSVs | `build/bench-reports/canonical/*.csv` | Local threshold-free measurements. |
| Canonical index | `build/bench-reports/canonical/index.tsv` | Structured generated metadata for canonical artifacts. |
| Canonical manifest | `build/bench-reports/canonical/manifest.txt` | Human-readable command, artifact, and methodology context. |
| Sentinel rows | `build/bench-reports/sentinels/sentinels.tsv` | Local wall-check hard gate plus threshold-free context rows. |
| Normalized report index | `build/report-index/normalized-index.tsv` | Cross-report navigation and freshness diagnostics; not a broad claim owner. |

Generated benchmark, sentinel, guardrail, and normalized-index outputs belong
under ignored `build/` paths. Sprint 168 should not stage generated report
files unless it explicitly changes the publication policy.

## Reusable Freshness Patterns

| Pattern | Current owner | Reuse candidate |
| --- | --- | --- |
| Regenerate before checking freshness | `report-index-oracle-freshness`, `report-index-comparison-freshness` | Sprint 168 hosted lane should generate the selected performance report before checking it. |
| `normalize_report_index.py --check-freshness` | Report-index freshness commands | Candidate for benchmark-family freshness if current parser supports the selected benchmark row semantics. |
| `--require-generated <family>` | Existing oracle/comparison freshness checks | Candidate pattern for requiring the selected performance artifact to exist. |
| Source-controlled report-family metadata | `tests/corpus/manifests/report_families.tsv` | Candidate owner for selected benchmark hosted/local policy rows. |
| Clear pass/fail echo messages | Makefile freshness targets | Reuse for hosted CI so failures name missing/stale selected performance rows. |
| Artifact upload path naming | Existing Linux generated-report freshness job | Reuse for hosted report bundle upload after CI lane design. |

## Local-Only Versus Hosted Boundary

| Surface | Current status | Sprint 168 target status |
| --- | --- | --- |
| `make bench-canonical-report` | Local-only threshold-free generated report bundle. | Candidate generator for one hosted selected report scope. |
| `bench_refactor_csc` canonical row | Local-only measurement today. | Preferred candidate for hosted publication if Day 3/Day 4 confirm runtime and output stability. |
| `bench-fast` | Supplemental Linux runtime smoke. | Remains smoke evidence, not hosted methodology-bound publication. |
| `performance-sentinels` | Local hard/advisory sentinel bundle. | Remains local unless explicitly selected in a later sprint. |
| Benchmark docs | Source-controlled interpretation guidance. | May be updated after hosted lane implementation to distinguish local and hosted selected rows. |
| Hosted CI artifacts | No hosted performance publication owner today. | Sprint 168 must add named workflow/job/artifact ownership before hosted claims are made. |

## Day 3 Handoff

Day 3 should score candidates from the canonical maintained surface:

1. `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`
2. `bench_chol_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`
3. `bench_iterative_reuse` default workload
4. `bench_eigs_reuse` default workload

The selection should prioritize bounded runtime, stable metadata, clear user
value, and low risk of backend-superiority or portable-performance overclaim.

## Validation Notes

Day 2 changed only Sprint 168 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Current benchmark/report owners are known. | Complete | Makefile, canonical report script, sentinel script, docs, and manifest owners are inventoried. |
| Local-only versus hosted evidence boundaries are explicit. | Complete | Boundary table keeps canonical reports and sentinels local-only until a named hosted lane exists. |
| Reusable report/freshness conventions are identified. | Complete | Freshness patterns table identifies regeneration, normalization, required-generated checks, report-family metadata, messages, and upload paths. |
