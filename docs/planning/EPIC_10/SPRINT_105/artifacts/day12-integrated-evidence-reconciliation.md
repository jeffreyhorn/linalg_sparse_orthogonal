# Sprint 105 Day 12 Integrated Evidence Reconciliation

## Purpose

Day 12 reconciles Sprint 105 named-matrix, generated-family, guardrail, and
documentation evidence into one coherent package. The reconciliation verifies
that the implemented report fields match the Day 3 contract, the selected
guardrail lanes match the Day 8 design, and the Day 11 docs do not overstate
local runtime or memory evidence.

## Regenerated Evidence Checklist

| evidence owner | command | regenerated artifact or output | result |
|---|---|---|---|
| full named-matrix reorder/fill report | `build/bench_reorder --skip-factor` | stdout CSV with 30 rows | passed |
| full named-matrix CSV contract | `build/bench_reorder --skip-factor \| awk ...` | schema, fixture, ordering, and metadata check | passed |
| large-matrix guardrail bundle | `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/` | passed |
| guardrail lane index | `awk ... build/bench-reports/large-matrix-guardrails/index.tsv` | 4 reviewed pass rows and 2 supplemental skip rows | passed |
| bounded two-fixture guardrail CSV | `awk ... build/bench-reports/large-matrix-guardrails/bench_reorder_sprint86.csv` | schema, fixture, ordering, and metadata check | passed |

Generated guardrail artifacts:

```text
build/bench-reports/large-matrix-guardrails/index.tsv
build/bench-reports/large-matrix-guardrails/manifest.txt
build/bench-reports/large-matrix-guardrails/test_graph.txt
build/bench-reports/large-matrix-guardrails/test_reorder_nd.txt
build/bench-reports/large-matrix-guardrails/test_reorder_amd_qg.txt
build/bench-reports/large-matrix-guardrails/bench_reorder_sprint86.csv
```

## Integrated Sprint 105 Evidence Table

| lane | source | fixture coverage | primary fields | reviewed status | interpretation |
|---|---|---|---|---|---|
| named-matrix full report | `build/bench_reorder --skip-factor` | `nos4`, `bcsstk04`, `Kuu`, `bcsstk14`, `s3rmt3m3`, `Pres_Poisson` | `matrix`, `n`, `reorder`, `nnz_L`, `reorder_ms`, `factor_ms`, `reorder_path`, `fixture_slice`, `nd_base_threshold` | supplemental/report | structural fill rows plus local timing context; not a portable timing gate |
| bounded two-fixture report | `build/bench_reorder --sprint86-slice --skip-factor` | `bcsstk14`, `Pres_Poisson` | same `bench_reorder` CSV fields | reviewed in `large-matrix-guardrails` as `G4` | CSV-shape and structural fill evidence; `sprint86` is a historical slice label |
| graph generated-family tests | `build/test_graph` | grids, paths, mesh, clique bridge, named Matrix Market smoke fixtures | test framework pass/fail plus fixture-specific printed context | reviewed as `G3` | graph partition, separator, determinism, and generated-family structural proof |
| ND generated-family tests | `build/test_reorder_nd` | grids, path, banded SPD, named Matrix Market fixtures | test framework pass/fail plus `nnz(L)` and residual context | reviewed as `G2` | ND permutation, fill, policy, and factor-dispatch structural proof |
| qg-AMD generated-family tests | `build/test_reorder_amd_qg` | `banded-n10000-bw5`, `nos4`, `bcsstk04`, `bcsstk14` | test framework pass/fail plus wrapper/helper `nnz(L)` equality | reviewed as `G1` | public AMD wrapper delegation and large regular generated-input guardrail |
| supplemental full named-matrix report | `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails` | full `bench_reorder --skip-factor` fixture set | same full-report CSV fields | supplemental opt-in as `S1` | useful local context; skipped in default reviewed bundle |
| supplemental qg-AMD report | `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails` | `bench_amd_qg --skip-bitset` fixture set | `reorder_ms`, `peak_rss_mb`, `nnz_L` | supplemental opt-in as `S2` | local qg-AMD and generated-banded context; max-RSS remains platform-local |

## Regenerated Structural Values

The regenerated full named-matrix lane preserved the structural values from
Day 6:

| fixture | strongest row by `nnz_L` | current structural interpretation |
|---|---|---|
| `nos4` | `amd` and `nd` tie at `637` | small smoke fixture; ND falls through to AMD-equivalent fill |
| `bcsstk04` | `amd` and `nd` tie at `3143` | small structural fixture; direct-path fill remains stable |
| `Kuu` | `amd` at `406264` | bimodal-degree stress fixture; ND remains `753755` / `1.855x` AMD fill |
| `bcsstk14` | `amd` at `116071` | reviewed irregular fixture; ND remains `132634` / `1.143x` AMD fill |
| `s3rmt3m3` | `amd` at `474609` | supplemental irregular fixture; ND remains close to AMD at `484890` |
| `Pres_Poisson` | `nd` at `2474435` | reviewed 2D PDE fixture; ND remains better than AMD `2668793` |

Runtime fields varied from earlier artifacts, as expected. Those values remain
local context only and were not used as pass/fail evidence.

## Contract Reconciliation

| contract rule | current evidence | status |
|---|---|---|
| fill, runtime, and memory fields remain separable | `nnz_L`, `reorder_ms`, `factor_ms`, and `peak_rss_mb` are documented with separate interpretations | consistent |
| `bench_reorder` rows retain stable fixture and ordering labels | full and bounded parser checks passed | consistent |
| skip rows are visible evidence | `factor_ms=skip` remains explicit; supplemental guardrail lanes emit `skip` by default | consistent |
| reviewed vs supplemental lanes are explicit | guardrail `index.tsv` has `G1`-`G4` reviewed pass rows and `S1`-`S2` supplemental skip rows | consistent |
| local timing is not portable performance evidence | Day 11 docs and this artifact frame timing as local context | consistent |
| max-RSS is not cross-platform pass/fail evidence | `S2` remains supplemental and report-only | consistent |

## Contradictions and Fix Candidates

| issue | disposition | fix candidate |
|---|---|---|
| The `sprint86` fixture-slice label is historical and can confuse readers | documented in Day 11 and preserved for compatibility | optional future alias or schema migration if consumers can tolerate it |
| `bench_reorder --skip-factor` is useful full named-matrix evidence but not default-reviewed in the guardrail bundle | resolved by explicit `S1` supplemental lane | no immediate fix needed |
| `bench_amd_qg --skip-bitset` carries max-RSS context that can be overread | resolved by explicit `S2` supplemental lane and docs non-claim | no immediate fix needed |
| Runtime values vary across reruns while structural fields remain stable | expected and documented | keep runtime out of pass/fail checks unless a fresh baseline is designed |
| `bench_fillin` arrow generated-family context remains outside the integrated package | deferred | Day 13/14 residual queue if LU fill schema work becomes relevant |

No immediate source or documentation contradiction required a Day 12 fix.

## Validation Summary

Commands run:

```sh
make build/bench_reorder && build/bench_reorder --skip-factor
make large-matrix-guardrails
build/bench_reorder --skip-factor | awk -F, '... full named-matrix contract ...'
awk -F'\t' '... guardrail index contract ...' build/bench-reports/large-matrix-guardrails/index.tsv
awk -F, '... bounded CSV contract ...' build/bench-reports/large-matrix-guardrails/bench_reorder_sprint86.csv
rg -n "\[FAIL\]|Tests failed:[[:space:]]*[1-9]|ERROR|error" build/bench-reports/large-matrix-guardrails/*.txt
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105 docs/algorithm.md benchmarks/README.md docs/maintainer_guide.md
```

Results:

- full named-matrix lane: passed;
- full named-matrix parser contract: passed;
- large-matrix guardrail bundle: passed;
- guardrail index contract: passed;
- bounded guardrail CSV contract: passed;
- strict generated-report failure scan: passed; no matches;
- `git diff --check`: passed;
- trailing-whitespace scan: passed; no matches.

No `.c` or `.h` files were modified for Day 12, so the full C quality gate is
not required by the sprint instructions.

## Completion Check

| criterion | status |
|---|---|
| named-matrix evidence rerun | complete |
| generated-family and guardrail evidence rerun | complete |
| metric and fixture contract checked | complete |
| contradictions identified and assigned | complete |
| local runtime results kept non-portable | complete |
