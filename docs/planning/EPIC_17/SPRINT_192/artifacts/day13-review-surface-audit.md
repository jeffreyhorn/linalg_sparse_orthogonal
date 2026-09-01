# Sprint 192 Day 13: Review Surface and Residual Audit

## Summary

Day 13 reviewed the Sprint 192 branch surface for accidental broadening,
generated artifact leakage, brittle claim wording, and selected-lane identity
drift. The branch reads as one methodology-bound selected performance lane for
`bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`.

## Changed Surface Audit

| Surface | Review result |
| --- | --- |
| `.github/workflows/ci.yml` | Hosted selected performance job is bounded to Linux, `timeout-minutes: 10`, hosted metadata, direct hosted checker invocation, exact three-file upload scope, and `if-no-files-found: error`. |
| `scripts/check_bench_canonical_freshness.py` | Selected checker validates selected artifact presence, `index.tsv`, manifest agreement, claim boundaries, and selected CSV content. |
| `tests/test_bench_canonical_freshness.py` | Covers positive local/hosted behavior and negative selected metadata, CSV shape, policy, and unselected-row drift. |
| `tests/test_selected_comparison_workflow.py` | Guards selected performance workflow timeout, hosted metadata, upload artifact, retention, exact paths, broad upload rejection, and unselected CSV rejection. |
| `tests/test_selected_performance_docs.py` | Guards selected performance doc markers and forbidden broad-performance overclaims. |
| `tests/test_normalize_report_index.py` | Confirms normalized benchmark rows preserve selected methodology metadata and required benchmark artifacts fail clearly when missing. |
| `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, corpus docs | Documentation states threshold-free selected evidence and non-claims without promoting broad performance support. |

No `.c` or `.h` files are modified.

## Selected Lane Trace

| Field | Value |
| --- | --- |
| Selected target | `SRT-BENCH-REFACTOR-CSC-NOS4` |
| Family/subfamily | `benchmark` / `canonical` |
| Target key | `bench_refactor_csc` |
| Workload | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| Artifact pattern | `build/bench-reports/canonical/bench_refactor_csc.csv` |
| Required files | `bench_refactor_csc.csv;index.tsv;manifest.txt` |
| Expected rows | `1` |
| Expected row id | `bench_refactor_csc` |
| Workflow | `.github/workflows/ci.yml` |
| Workflow job | `hosted-performance-freshness` |
| Workflow artifact | `sprint168-selected-performance-freshness` |
| Workflow platforms | `linux` |
| Support tier | `hosted_selected` in selected manifest; local regenerated rows remain `local_only` unless hosted env is supplied. |
| Freshness policy | `generated_local_advisory` |

## Claim-Boundary Checklist

- Selected row remains `status=measurement`.
- Selected row remains `baseline=n/a` and `threshold=n/a`.
- Selected row records `warmup=none_configured`.
- Selected row records `variance=not_computed_single_sample`.
- Hosted mode requires `hosted_selected` and
  `hosted_selected_threshold_free`.
- Hosted upload scope is exactly the selected CSV plus `index.tsv` and
  `manifest.txt`.
- Unselected canonical CSV files are not uploaded as selected hosted evidence.
- Local sentinel thresholds remain separate from hosted selected-performance
  freshness.
- Docs retain non-claims for portable performance, release benchmark proof,
  algorithmic superiority, platform parity, package/ABI support,
  runtime-loader support, external-library parity, OpenMP speedup evidence,
  backend superiority, and state-of-the-art status.

## Generated Artifact Audit

The Day 12 regenerated canonical benchmark files remain ignored through
`.gitignore:2:build/`. `git status --short --ignored
build/bench-reports/canonical` reported only:

```text
!! build/
```

No generated benchmark, normalized-index, or Python cache artifacts are intended
to be committed.

## Residual Queue

| Residual | Status |
| --- | --- |
| Hosted timing threshold | Deferred until a hosted baseline, variance model, repeat/warmup policy, tolerance, and same-machine comparison policy exist. |
| Unselected canonical benchmark publication | Out of scope; generated locally but not uploaded as selected hosted performance evidence. |
| Windows selected benchmark freshness | Out of scope; Windows benchmark freshness remains separate from the Linux hosted lane. |
| macOS selected benchmark freshness | Out of scope; no selected benchmark freshness lane is promoted for macOS. |
| Portable performance / state-of-the-art claims | Out of scope and guarded as non-claims. |
| Package, ABI, runtime-loader, package-manager evidence | Out of scope and explicitly separate from selected performance freshness. |

## Day 14 Closeout Checklist

- Rerun the Day 12 focused validation set.
- Verify generated canonical benchmark files are still ignored.
- Verify no Python cache files remain.
- Verify no `.c` or `.h` files changed before deciding whether the full C
  quality gate is required.
- Check the final diff for accidental broad performance wording.
- Ensure `WORKING_NOTES.md` and Day 14 artifact summarize completed scope,
  validation, residuals, and retrospective inputs.

## Validation

Commands run:

```sh
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py tests/test_bench_canonical_freshness.py tests/test_selected_performance_docs.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
git status --short --ignored build/bench-reports/canonical
```

Results:

- selected-performance docs guard passed;
- selected workflow guard passed;
- selected target schema validation passed;
- selected benchmark freshness regression tests passed;
- report-index normalization regression tests passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 13;
- generated canonical benchmark files remain ignored.
