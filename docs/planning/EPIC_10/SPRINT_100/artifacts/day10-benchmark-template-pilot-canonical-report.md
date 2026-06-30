# Sprint 100 Day 10 Pilot: Canonical Benchmark Report

## Summary

| field | value |
|---|---|
| benchmark surface | canonical maintained benchmark report |
| benchmark binary or target | `make bench-canonical-report` |
| command | `make bench-canonical-report` |
| artifact owner | `scripts/bench_canonical_report.sh`, `Makefile`, `benchmarks/README.md` |
| implementation owner | `benchmarks/bench_refactor_csc.c`, `benchmarks/bench_chol_csc.c`, `benchmarks/bench_iterative_reuse.c`, `benchmarks/bench_eigs_reuse.c` |
| report owner | `build/bench-reports/canonical/` |
| category | canonical maintained |
| fixture scope | fixed maintained fixtures and default reuse workloads |
| metric scope | local timing plus benchmark-local residual/convergence context |
| claim state before work | earned as threshold-free local report |
| claim state after work | earned as threshold-free local report; not promoted to sentinel |

## Claim Boundary

Bounded claim:

> The canonical report target emits a repeatable, artifact-friendly local
> snapshot for the maintained direct, Cholesky CSC, iterative reuse, and
> eigensolver reuse benchmark surfaces.

Disallowed broader claim:

> This artifact does not prove portable performance superiority, broad
> ecosystem benchmark parity, or a pass/fail performance regression threshold.

## Command and Environment

| field | value |
|---|---|
| command | `make bench-canonical-report` |
| working directory | repository root |
| build mode | Makefile benchmark binaries under `build/` |
| compiler and flags | caller-local Makefile defaults |
| platform | caller-local |
| CPU / machine class | caller-local; must be recorded by the operator when comparing runs |
| thread settings | benchmark-local defaults unless explicitly overridden |
| BLAS / SuiteSparse / optional backend state | current local build and probe state |
| repeat count | `--repeat 1` for the SuiteSparse-backed direct/Cholesky entries; benchmark-local defaults for reuse binaries |
| random seed policy | deterministic/default benchmark construction |

## Fixture Set

| fixture | source | dimensions | nnz | class | reason selected |
|---|---|---:|---:|---|---|
| `nos4.mtx` | `tests/data/suitesparse/nos4.mtx` | 100 | documented by Matrix Market fixture | SuiteSparse SPD | fixed compact direct/Cholesky canonical fixture |
| iterative reuse defaults | `bench_iterative_reuse` internal construction | benchmark-defined | benchmark-defined | public iterative handle reuse | maintained reuse workflow measurement |
| eigensolver reuse defaults | `bench_eigs_reuse` internal construction | benchmark-defined | benchmark-defined | public eigensolver handle reuse | maintained reuse workflow measurement |

## Metrics

| metric | unit | emitted by | interpretation |
|---|---|---|---|
| refactor, solve, and speedup fields | milliseconds / ratio | `bench_refactor_csc` | local direct repeated-run lifecycle measurement |
| Cholesky backend timing fields | milliseconds | `bench_chol_csc` | local linked-list / CSC scalar / CSC supernodal comparison context |
| iterative one-shot, reuse, speedup, iteration, residual fields | milliseconds / ratio / counts / residual | `bench_iterative_reuse` | local public iterative handle reuse measurement with convergence context |
| eigensolver one-shot, reuse, speedup, iteration, residual, basis fields | milliseconds / ratio / counts / residual / basis size | `bench_eigs_reuse` | local public eigensolver handle reuse measurement with convergence context |
| bundle metadata | text / TSV | `scripts/bench_canonical_report.sh` | artifact identity and command mapping |

Metric ownership checklist:

- [x] local timing metrics are explicitly labeled local
- [x] throughput or speedup metrics identify their comparison baseline
- [x] residual or correctness fields identify benchmark-local context, not oracle ownership
- [x] fill, memory, or basis-size fields are separated from wall time
- [x] unsupported rows or skipped workloads remain benchmark-output responsibilities

## Output Artifacts

| artifact | format | owner | interpretation |
|---|---|---|---|
| `bench_refactor_csc.csv` | CSV | `bench_refactor_csc` | direct repeated-run lifecycle snapshot |
| `bench_chol_csc.csv` | CSV | `bench_chol_csc` | Cholesky CSC backend-aware snapshot |
| `bench_iterative_reuse.csv` | CSV | `bench_iterative_reuse` | iterative handle reuse snapshot |
| `bench_eigs_reuse.csv` | CSV | `bench_eigs_reuse` | eigensolver handle reuse snapshot |
| `manifest.txt` | text | `scripts/bench_canonical_report.sh` | command mapping and artifact inventory |
| `index.tsv` | TSV | `scripts/bench_canonical_report.sh` | one structured row per emitted canonical artifact |

## Acceptance or Reporting Mode

| field | value |
|---|---|
| mode | threshold-free report |
| pass/fail threshold | `none` |
| threshold source | `n/a` |
| expected status | report-only success when all benchmark binaries complete and artifacts are written |
| rerun guidance | compare emitted CSV rows across labeled local runs; do not compare unlabeled timings across unlike machines |

## Evidence Summary

| evidence type | result | notes |
|---|---|---|
| local timing | reported | each CSV owns local timing fields |
| fill or memory | partial/contextual | eigensolver basis-size fields are contextual, not a memory guarantee |
| residual or correctness context | contextual only | tests and external oracle lanes remain correctness owners |
| artifact generation | defined | bundle writes four CSVs plus manifest and index |
| unsupported cases | benchmark-local | skipped or unsupported rows must be interpreted from benchmark output |

## Non-Claims

This artifact does not claim:

- portable performance superiority;
- pass/fail runtime regression detection;
- broad benchmark ecosystem parity;
- correctness ownership for direct, iterative, or eigensolver APIs;
- Windows/MSVC benchmark parity unless separately run and recorded there.

## Follow-Up Work

| follow-up | owner | reason |
|---|---|---|
| add filled templates for `wall-check` | Sprint 104 or Sprint 109 | promote sentinel interpretation without blurring report-only benchmarks |
| add filled coverage evidence when coverage topology changes | Sprint 105 or Sprint 109 | keep supplemental coverage status explicit |
| record machine-class metadata in future benchmark artifacts | Sprint 104 | improve local before/after comparison hygiene |
