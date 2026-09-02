# Sprint 192 Day 12: Integrated Local Validation

## Summary

Day 12 ran the integrated local validation set for the selected
methodology-bound performance lane. The public local freshness target
regenerated the canonical benchmark bundle, the selected checker passed, the
workflow/docs/report-index guards passed, and generated artifacts remained
ignored under `build/`.

## Regenerated Artifact Set

Command:

```sh
make bench-canonical-report-freshness
```

Result:

```text
bench-canonical-report: wrote build/bench-reports/canonical
  - bench_refactor_csc.csv
  - bench_chol_csc.csv
  - bench_iterative_reuse.csv
  - bench_eigs_reuse.csv
  - index.tsv
  - manifest.txt
bench-canonical-report-freshness: checking selected canonical performance report
bench-canonical-freshness: passed (mode=local; artifact=bench_refactor_csc; report_dir=build/bench-reports/canonical)
bench-canonical-report-freshness: passed (selected threshold-free performance report freshness)
```

## Selected Artifact Inspection

| Field | Value |
| --- | --- |
| CSV `benchmark` | `bench_refactor_csc` |
| CSV `matrix` | `nos4.mtx` |
| CSV `n` | `100` |
| CSV `nnz` | `594` |
| CSV `scenario` | `chol_spd` |
| CSV `ldlt_dense_backend_request` | `n/a` |
| CSV `ldlt_dense_backend_selected` | `n/a` |
| CSV `ldlt_dense_backend_fallback` | `n/a` |
| Index `git_commit` | `557535c1` |
| Index `git_branch` | `sprint-192` |
| Index `runner_context` | `local` |
| Index `build_flags` | `not_recorded` |
| Index `cpu_model` | `unknown` |
| Index `build_mode` | `serial` |
| Index `omp_num_threads` | `unset` |
| Index `status` | `measurement` |
| Index `support_tier` | `local_only` |
| Index `claim_boundary` | `local_threshold_free` |
| Index `fixture_or_workload` | `nos4.mtx` |
| Index `matrix_size` | `n=100` |
| Index `repeat_semantics` | `configured_repeat_1` |
| Index `warmup` | `none_configured` |
| Index `variance` | `not_computed_single_sample` |
| Index `baseline` | `n/a` |
| Index `threshold` | `n/a` |
| Index `backend_context` | `n/a` |
| Index `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` |
| Manifest `selected_artifact` | `bench_refactor_csc` |
| Manifest `selected_matrix_size` | `n=100` |
| Manifest `baseline` | `n/a` |
| Manifest `threshold` | `n/a` |
| Manifest `warmup` | `none_configured` |
| Manifest `variance` | `not_computed_single_sample` |
| Manifest `methodology_notes` | `threshold_free_local_measurement;not_portable_performance_claim` |

The generated local artifact also recorded the local Darwin platform and Apple
Clang compiler in `index.tsv` and `manifest.txt`. Those values are context for
the local validation run, not portable performance evidence.

## Ignored Artifact Check

`git check-ignore -v` confirmed that the generated canonical benchmark files
are ignored through `.gitignore:2:build/`:

- `build/bench-reports/canonical/bench_refactor_csc.csv`;
- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`;
- generated unselected canonical CSV files.

`git status --short --ignored build/bench-reports/canonical` reported
`!! build/`, confirming generated artifacts are not tracked branch changes.

## Validation

Commands run:

```sh
make bench-canonical-report-freshness
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 -m py_compile scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py tests/test_bench_canonical_freshness.py tests/test_selected_performance_docs.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected canonical benchmark freshness passed through the public Make target;
- selected-performance docs guard passed;
- selected workflow guard passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- selected benchmark freshness regression tests passed;
- report-index normalization regression tests passed;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 12.

## Residual Queue Draft

- Hosted timing thresholds remain deferred until a hosted baseline, variance
  model, repeat/warmup policy, tolerance, and same-machine comparison design
  exist.
- Unselected canonical benchmark CSV files remain generated local context, not
  uploaded selected hosted performance evidence.
- Windows and macOS selected benchmark freshness remain outside this lane.
- The selected performance lane remains methodology-bound and threshold-free,
  not portable performance or state-of-the-art evidence.

## Day 13 Inputs

Day 13 should review the full diff for accidental broadening, duplicated
authority, generated artifact leakage, brittle markers, and claim wording that
could imply broad performance support.
