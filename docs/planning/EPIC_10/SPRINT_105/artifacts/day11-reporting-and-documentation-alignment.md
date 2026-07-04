# Sprint 105 Day 11 Reporting and Documentation Alignment

## Purpose

Day 11 aligned user and maintainer documentation with the Sprint 105
reorder/fill reporting contract, large-matrix guardrail design, and implemented
`make large-matrix-guardrails` target.

## Updated Documentation

| file | update | reason |
|---|---|---|
| `docs/algorithm.md` | Added reorder/fill reporting interpretation guidance | Gives users a compact explanation of structural fill metrics, local timing, max-RSS caveats, reviewed guardrails, and supplemental large-matrix reports |
| `benchmarks/README.md` | Added `make large-matrix-guardrails` command, artifact, lane, and interpretation guidance | Gives benchmark users the current command and output shape without treating supplemental reports as reviewed gates |
| `docs/maintainer_guide.md` | Added maintainer ownership rules for the large-matrix structural guardrail bundle | Keeps reviewed lanes, supplemental lanes, and non-claims aligned with the implemented target |

## User-Facing Interpretation

The updated user guidance separates:

- structural fields such as `nnz_L`, `nnz_R`, `nnz_LU`, `fill_ratio`,
  `bandwidth`, and `separator_size`;
- local runtime fields such as `reorder_ms`, `factor_ms`, and command wall
  time;
- platform-local memory proxies such as `peak_rss_mb`;
- reviewed structural guardrail output from `make large-matrix-guardrails`;
- supplemental large-matrix reports enabled with
  `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1`.

The main public rule is that structural rows are fixture-bound, runtime rows
are local context unless backed by `wall-check`, and supplemental large-matrix
reports are not new reviewed quality gates.

## Maintainer Interpretation

The updated maintainer guidance records the current guardrail bundle:

| lane | command | category | interpretation |
|---|---|---|---|
| `G1` | `build/test_reorder_amd_qg` | reviewed | qg-AMD wrapper and large banded structural guardrail |
| `G2` | `build/test_reorder_nd` | reviewed | ND generated-family, named-matrix, policy, and residual structural coverage |
| `G3` | `build/test_graph` | reviewed | graph partition, separator, generated-family, and determinism coverage |
| `G4` | `build/bench_reorder --sprint86-slice --skip-factor` | reviewed | bounded CSV-shape and structural fill rows for `bcsstk14` and `Pres_Poisson` |
| `S1` | `build/bench_reorder --skip-factor` | supplemental | threshold-free full named-matrix reorder/fill report |
| `S2` | `build/bench_amd_qg --skip-bitset` | supplemental | threshold-free qg-AMD and generated-banded report; max-RSS remains platform-local |

## Benchmark Reporting Notes

`benchmarks/README.md` now documents the default output directory:

```text
build/bench-reports/large-matrix-guardrails/
```

and the default artifacts:

- `index.tsv`;
- `manifest.txt`;
- `test_reorder_amd_qg.txt`;
- `test_reorder_nd.txt`;
- `test_graph.txt`;
- `bench_reorder_sprint86.csv`.

The docs explicitly keep the `sprint86` fixture-slice label as historical and
interpret it as the current bounded two-fixture slice.

## Residual Queue

| item | status | reason |
|---|---|---|
| richer schema migration for `bench_reorder` | deferred | Day 11 aligned docs only; no benchmark row migration was needed |
| hard timing thresholds for supplemental lanes | deferred | requires fresh baseline and machine-class design |
| cross-platform max-RSS interpretation | deferred | platform-sensitive and intentionally report-only |
| broader user tutorial examples for reorder/fill artifacts | deferred | current update covered algorithm and benchmark interpretation; tutorial walkthrough can follow later |

## Validation

Validation commands:

```sh
git diff --check
rg -n "[ \t]+$" docs/algorithm.md benchmarks/README.md docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_105
```

Results:

- `git diff --check`: passed after temporarily marking untracked Sprint 105
  files and `scripts/large_matrix_guardrails.sh` intent-to-add for coverage.
- trailing-whitespace scan: passed; no matches.

No `.c` or `.h` files were modified for Day 11, so the full C quality gate is
not required by the sprint instructions.

## Completion Check

| criterion | status |
|---|---|
| users can interpret fill and runtime outputs without overreading them | complete |
| maintainers know reviewed, supplemental, and local-only lane boundaries | complete |
| benchmark docs point at current target and outputs | complete |
| documentation validation recorded | complete |
