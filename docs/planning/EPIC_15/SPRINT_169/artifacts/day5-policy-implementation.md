# Sprint 169 Day 5: Statistical And Schema Implementation

## Purpose

Day 5 implements the Day 3 statistical policy and Day 4 schema-normalization
design for the selected performance report lane. The implementation keeps the
canonical index at 29 columns, makes weak methodology values explicit, and
preserves the selected versus unselected row claim boundary.

## Implemented Generator Changes

Updated `scripts/bench_canonical_report.sh`:

| Field | Previous value | New value or behavior |
| --- | --- | --- |
| `warmup` | `not_recorded` | `none_configured` |
| `variance` | `not_recorded` | `not_computed_single_sample` |
| selected `matrix_size` | `not_recorded` | `n=100` for `bench_refactor_csc` |
| `bench_chol_csc` `matrix_size` | `not_recorded` | `n=100`, still unselected/local-only |
| default canonical row `matrix_size` | global `not_recorded` | row-specific argument; remains `not_recorded` for rows without stable dimensions |
| manifest matrix-size field | `matrix_size=not_recorded` | `selected_matrix_size=n=100` |

The generator command interface is unchanged:

```text
scripts/bench_canonical_report.sh <report_dir> <bench_refactor_csc> <bench_chol_csc> <bench_iterative_reuse> <bench_eigs_reuse>
```

The selected hosted support boundary is unchanged:

- selected row can be `hosted_selected` /
  `hosted_selected_threshold_free` in hosted mode;
- unselected rows remain `local_only` / `local_threshold_free`.

## Implemented Freshness-Checker Changes

Updated `scripts/check_bench_canonical_freshness.py`:

- requires selected `matrix_size=n=100`;
- requires selected `warmup=none_configured`;
- requires selected `variance=not_computed_single_sample`;
- extends manifest agreement to include `warmup`, `variance`, and selected
  matrix size;
- maps selected row `matrix_size` to `manifest.txt`
  `selected_matrix_size` so the manifest remains explicit about selected
  scope;
- preserves existing checks for selected row identity, threshold-free
  baseline/threshold values, hosted-mode metadata, and unselected row
  `local_only` / `local_threshold_free` boundaries.

## Documentation Updates

Updated:

- `benchmarks/README.md`;
- `docs/maintainer_guide.md`.

The docs now describe canonical warmup/variance semantics as explicit
methodology fields:

- `warmup=none_configured`;
- `variance=not_computed_single_sample`.

They continue to say these values must not be interpreted as
warmup-controlled timing, repeated samples, computed variance, confidence
intervals, portable performance, or broad benchmark evidence.

## Generated Output Verification

Focused validation generated a hosted-style canonical report and confirmed the
row-level schema:

| Artifact | Matrix size | Warmup | Variance | Support tier | Claim boundary |
| --- | --- | --- | --- | --- | --- |
| `bench_refactor_csc` | `n=100` | `none_configured` | `not_computed_single_sample` | `hosted_selected` | `hosted_selected_threshold_free` |
| `bench_chol_csc` | `n=100` | `none_configured` | `not_computed_single_sample` | `local_only` | `local_threshold_free` |
| `bench_iterative_reuse` | `not_recorded` | `none_configured` | `not_computed_single_sample` | `local_only` | `local_threshold_free` |
| `bench_eigs_reuse` | `not_recorded` | `none_configured` | `not_computed_single_sample` | `local_only` | `local_threshold_free` |

The selected row now has machine-readable statistical limitations while
remaining threshold-free.

## Preserved Non-Claims

Day 5 does not add:

- timing regression thresholds to selected publication rows;
- portable performance superiority;
- broad benchmark-family publication;
- external-library parity;
- platform parity;
- package, shared-library, dynamic ABI, or runtime-loader claims;
- state-of-the-art sparse linear algebra performance claims.

## Validation

Day 5 changed shell, Python, documentation, and planning artifacts. No `.c` or
`.h` files were modified, so the full C quality gate is not required for this
day.

Validation run:

```sh
bash -n scripts/bench_canonical_report.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/check_bench_canonical_freshness.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_bench_canonical_freshness.py --help
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness
env BENCH_CANONICAL_REPORT_LABEL=sprint-169-hosted-performance \
  SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
  SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
  SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest \
  SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
  SPARSE_CANONICAL_CPU_MODEL=unknown \
  SPARSE_CANONICAL_BUILD_MODE=serial \
  PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
git diff --check
```

All checks passed.

Generated canonical output remained under ignored `build/` paths and should
not be staged as source.

## Day 5 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected report output matches the Day 3 and Day 4 policy. | Complete | `matrix_size=n=100`, `warmup=none_configured`, and `variance=not_computed_single_sample` are generated and checked for the selected row. |
| Local freshness still has a conservative interpretation. | Complete | `make bench-canonical-report-freshness` passes without changing threshold-free interpretation. |
| Unselected rows are not promoted by the implementation. | Complete | Hosted-style output keeps unselected rows `local_only` / `local_threshold_free`, and the checker still enforces that invariant. |
