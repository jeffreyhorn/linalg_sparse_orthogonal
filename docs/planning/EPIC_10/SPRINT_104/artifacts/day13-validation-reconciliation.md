# Sprint 104 Day 13 Validation Reconciliation

## Purpose

Day 13 reconciles the Sprint 104 artifacts with the final implementation,
documentation, script, and validation state before the closeout day. Because
Sprint 104 modified `.c` files and tests, the final required quality gate is
the full:

```sh
make format && make lint && make test
```

## Final Touched Surface Summary

| surface | files | Sprint 104 role |
|---|---|---|
| source comments | `src/sparse_matrix.c`, `src/sparse_eigs.c` | document OpenMP runtime ownership beside the actual parallel regions |
| tests | `tests/test_chol_csc_supernodal.c`, `tests/test_ldlt.c` | prove invalid optional dense backend env requests fall back to builtin |
| Make/report scripts | `Makefile`, `scripts/performance_sentinels.sh`, `scripts/bench_canonical_report.sh` | add bounded local sentinel target and align canonical report metadata |
| user/maintainer docs | `README.md`, `benchmarks/README.md`, `docs/algorithm.md`, `docs/maintainer_guide.md` | align benchmark, sentinel, OpenMP, backend, and platform wording |
| sprint artifacts | `docs/planning/EPIC_10/SPRINT_104/**` | capture baseline, audits, implementation notes, validation, and handoff |

## Artifact-to-Implementation Reconciliation

| artifact | implementation status |
|---|---|
| Day 1 runtime baseline | complete; claim boundaries and validation rules are reflected in later artifacts |
| Day 2 backend consumer audit | complete; Cholesky and LDLT dense-backend seams received focused fallback tests |
| Day 3 runtime contract | complete; builtin fallback, optional dense backend context, OpenMP runtime ownership, and observability split are preserved |
| Day 4 descriptor boundary | complete; no public API widening was needed |
| Day 5 descriptor batch | complete; invalid Cholesky/LDLT dense backend requests now have focused fallback coverage |
| Day 6 threading audit | complete; cleanup candidates and validation needs fed Day 7 |
| Day 7 threading cleanup | complete; source comments and public docs clarify that OpenMP thread count remains runtime-owned |
| Day 8 sentinel design | complete; first sentinel batch uses S5 hard gate plus S2 report-only rows |
| Day 9 sentinel batch | complete; `make performance-sentinels` generates the documented sentinel bundle |
| Day 10 reporting audit | complete; stale wording and claim risks drove Day 11 alignment |
| Day 11 reporting alignment | complete; docs and canonical metadata now use measurement language |
| Day 12 platform review | complete; POSIX/Windows CTest count split and no-change decision are documented |

## Final Validation Command Log

| command | result | note |
|---|---|---|
| `bash -n scripts/performance_sentinels.sh && bash -n scripts/bench_canonical_report.sh` | passed | shell syntax for touched report scripts |
| `make build/test_chol_csc_supernodal build/test_ldlt build/test_omp build/test_eigs` | passed | focused affected binaries already up to date |
| `./build/test_chol_csc_supernodal` | passed | 62 tests, 0 failures, 0 skips, 8170 assertions |
| `./build/test_ldlt` | passed | 89 tests, 0 failures, 0 skips, 912 assertions |
| `./build/test_omp` | passed | 12 tests, 0 failures, 0 skips, 831 assertions |
| `./build/test_eigs` | passed | 31 tests, 0 failures, 0 skips, 310 assertions |
| `make bench-canonical-report` | passed | generated canonical report artifacts |
| canonical metadata inspection | passed | `index.tsv` and `manifest.txt` use `category=measurement` |
| `make performance-sentinels` | passed | generated S5 pass rows and S2 report rows |
| sentinel artifact inspection | passed | recorded serial build, `OMP_NUM_THREADS=unset`, dense backend env unset, builtin dense kernel |
| `make format && make lint && make test` | passed | required full quality gate for `.c`/`.h` touched branch |

## Representative Final Report Rows

Final `make performance-sentinels` emitted:

| sentinel | status | metric | value | threshold |
|---|---|---|---:|---|
| S5 | pass | `bcsstk14 qg_amd_reorder_ms` | 57.1 | `2x` |
| S5 | pass | `Pres_Poisson amd_reorder_ms` | 3814.5 | `2x` |
| S5 | pass | `Pres_Poisson nd_reorder_ms` | 3480.7 | `1.5x` |
| S2 | report | `nos4.mtx factor_ll_ms` | 0.256 | `n/a` |
| S2 | report | `nos4.mtx factor_csc_ms` | 0.312 | `n/a` |
| S2 | report | `nos4.mtx factor_csc_sn_ms` | 0.293 | `n/a` |
| S2 | report | `nos4.mtx speedup_csc` | 0.82 | `n/a` |
| S2 | report | `nos4.mtx speedup_csc_sn` | 0.87 | `n/a` |

These rows are local measurement context only. S5 is the existing hard
wall-check lane; S2 remains threshold-free.

## Known Limitations and Non-Claims

- Optional dense acceleration is not required for correctness or installation.
- Invalid or unavailable optional dense backend requests fall back to builtin
  under current behavior.
- `performance-sentinels` is local regression evidence, not portable timing
  evidence.
- OpenMP remains opt-in; serial remains the default reviewed behavior.
- OpenMP timing depends on the external OpenMP runtime and thread settings.
- Windows remains the reviewed CMake-first consumer subset, not Makefile,
  benchmark, fuzz/property, or install-validation parity.
- Benchmark residual/agreement fields remain diagnostic context unless a test
  or oracle artifact owns the correctness claim.

## Sprint 105 Handoff Candidates

| priority | candidate | reason |
|---|---|---|
| high | keep benchmark/sentinel wording tied to generated fields | prevents local timing rows from becoming product claims |
| high | preserve Windows CTest count discipline | avoids silent reviewed-scope drift when adding tests |
| medium | consider same-worktree baseline design before adding S1/S3/S4 hard thresholds | avoids CI/local timing variance overclaim |
| medium | keep optional backend widening local to concrete seams | prevents Cholesky/LDLT dense backend work from implying broad vendor-provider parity |
| medium | keep OpenMP runtime ownership explicit near any future parallel region | avoids accidental `SPARSE_*` to OpenMP thread-control coupling |

## Completion Check

| criterion | status |
|---|---|
| artifacts reconciled with implementation | complete |
| focused backend/runtime/report checks rerun | complete |
| required full quality gate passed | complete |
| final report rows inspected | complete |
| limitations and non-claims captured | complete |
| Sprint 105 handoff candidates drafted | complete |
