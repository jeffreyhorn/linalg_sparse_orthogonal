# Sprint 105 Day 13 Final Validation and Residual Queue

## Purpose

Day 13 resolves the highest-priority Day 12 contradictions, reruns the focused
and full required validation surface for the branch, and records the remaining
Sprint 105 residual queue before Day 14 closeout.

## Fix Batch Decision

Day 12 found no immediate contradiction that required source, script, or public
documentation changes:

- the historical `sprint86` fixture-slice label is already documented as a
  compatibility label;
- the full `bench_reorder --skip-factor` lane is explicitly supplemental in
  the guardrail bundle;
- the `bench_amd_qg --skip-bitset` max-RSS context is explicitly supplemental
  and platform-local;
- runtime drift across reruns is expected and remains local context only;
- `bench_fillin` arrow generated-family context remains deferred.

The Day 13 bounded fix batch is therefore a no-op implementation decision, not
an omission. The value of the day is the final validation sweep and explicit
residual queue.

## Validation Commands

Focused script and guardrail validation:

```sh
bash -n scripts/large_matrix_guardrails.sh
make large-matrix-guardrails
```

Full required C quality gate, because this branch modifies
`tests/test_reorder_amd_qg.c`:

```sh
make format && make lint && make test
```

Documentation and diff hygiene:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_105 docs/algorithm.md benchmarks/README.md docs/maintainer_guide.md scripts/large_matrix_guardrails.sh Makefile tests/test_reorder_amd_qg.c
```

## Validation Results

| command | result | notes |
|---|---|---|
| `bash -n scripts/large_matrix_guardrails.sh` | passed | script syntax check passed before guardrail execution |
| `make large-matrix-guardrails` | passed | regenerated `build/bench-reports/large-matrix-guardrails/` |
| `make format && make lint && make test` | passed | full format, lint, and test gate completed successfully |
| `git diff --check` | passed | checked after temporarily marking untracked Sprint 105 files and `scripts/large_matrix_guardrails.sh` intent-to-add |
| trailing-whitespace scan | passed | no matches across touched documentation, script, Makefile, and qg-AMD test file |

## Regenerated Guardrail Artifacts

`make large-matrix-guardrails` regenerated:

```text
build/bench-reports/large-matrix-guardrails/index.tsv
build/bench-reports/large-matrix-guardrails/manifest.txt
build/bench-reports/large-matrix-guardrails/test_graph.txt
build/bench-reports/large-matrix-guardrails/test_reorder_nd.txt
build/bench-reports/large-matrix-guardrails/test_reorder_amd_qg.txt
build/bench-reports/large-matrix-guardrails/bench_reorder_sprint86.csv
```

The regenerated guardrail run preserved the reviewed/supplemental split:

- reviewed lanes `G1` through `G4` passed;
- supplemental lanes `S1` and `S2` remained explicit skips in default mode.

## Remaining Residual Queue

| item | status | handoff |
|---|---|---|
| `sprint86` fixture-slice label compatibility | residual | keep documented; consider a future alias/schema migration only if consumers can tolerate it |
| `bench_fillin` arrow generated-family context | residual | candidate future LU fill schema/reporting work; not part of Sprint 105 reviewed surface |
| supplemental full named-matrix guardrail lane `S1` | residual | keep opt-in until a future baseline promotes it |
| supplemental qg-AMD/max-RSS lane `S2` | residual | keep opt-in and platform-local; no cross-platform max-RSS threshold |
| graph/reorder history-heavy comments outside `test_reorder_amd_qg.c` | residual | candidate future cleanup in `tests/test_graph.c`, `tests/test_reorder_nd.c`, `src/sparse_graph.c`, and `src/sparse_reorder_nd.c` |
| hard timing thresholds beyond `wall-check` | residual | require fresh baseline, threshold source, and machine-class design |

## Completion Check

| criterion | status |
|---|---|
| highest-priority Day 12 contradictions resolved or explicitly deferred | complete |
| focused script and guardrail validation passed | complete |
| broader required C quality gate passed | complete |
| regenerated guardrail artifacts recorded | complete |
| residual queue explicit | complete |
