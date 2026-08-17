# Day 5 Comparison Harness Implementation

## Summary

Day 5 implemented descriptor-backed comparison targets in
`scripts/run_external_comparison.py` and added local generated output support
for the selected `qr-compatible-ls` family.

The existing `qr-minnorm` target remains supported and continues to write to
`build/comparison/qr_minnorm/`. The new `qr-compatible-ls` target writes to
`build/comparison/qr_compatible_ls/`.

## Implementation Changes

| Area | Change |
| --- | --- |
| Target model | Added a `TARGETS` descriptor table for comparison-family metadata. |
| Existing target | Preserved `qr-minnorm` fixture key, output root, row IDs, claim scope, and summary wording. |
| New target | Added `qr-compatible-ls` for `qr_overdetermined_compatible_5x3`. |
| Project probe | Generalized generated C probe inputs by descriptor; `qr-compatible-ls` uses `sparse_qr_factor` plus `sparse_qr_solve`. |
| Baseline parser | Generalized expected helper protocol count: `OK 6` for `qr-minnorm`, `OK 4` for `qr-compatible-ls`. |
| Row generation | Generalized study row IDs, fixture keys, subfamily, operation, claim scope, and tolerances by descriptor. |
| Validation | Generalized selected-row validation and self-check coverage by descriptor. |
| Summary | Generalized generated summary title and scope text by descriptor. |

## New Target Contract

| Field | Value |
| --- | --- |
| Target | `qr-compatible-ls` |
| Fixture key | `qr_overdetermined_compatible_5x3` |
| Subfamily | `qr_compatible_ls` |
| Operation | `least_squares_solve` |
| Output root | `build/comparison/qr_compatible_ls/` |
| Baseline command | `python3 tests/qr_external_dense_reference.py qr_overdetermined_compatible_5x3` |
| Project solve path | `sparse_qr_factor` followed by `sparse_qr_solve` |
| Expected solution | `1,-2,0.5` |
| Expected solution norm | `2.2912878474779199` |
| Residual tolerance | `1e-10` |
| Solution tolerance | `1e-10` |
| Claim scope | fixture-local QR compatible least-squares comparison only |

## Generated Outputs

`python3 scripts/run_external_comparison.py --target qr-compatible-ls` writes:

- `build/comparison/qr_compatible_ls/project_observations.tsv`
- `build/comparison/qr_compatible_ls/baseline_observations.tsv`
- `build/comparison/qr_compatible_ls/dependency_status.tsv`
- `build/comparison/qr_compatible_ls/study.tsv`
- `build/comparison/qr_compatible_ls/summary.md`
- `build/comparison/qr_compatible_ls/manifest.tsv`

## Selected Rows Emitted

| Row ID | Status |
| --- | --- |
| `comparison_qr_overdetermined_compatible_5x3_project_status_v1` | `pass` |
| `comparison_qr_overdetermined_compatible_5x3_baseline_status_v1` | `pass` |
| `comparison_qr_overdetermined_compatible_5x3_residual_norm_v1` | `pass` |
| `comparison_qr_overdetermined_compatible_5x3_solution_norm_v1` | `pass` |
| `comparison_qr_overdetermined_compatible_5x3_solution_values_v1` | `pass` |
| `comparison_qr_overdetermined_compatible_5x3_project_vs_baseline_max_abs_delta_v1` | `pass` |

Observed local values:

| Metric | Value |
| --- | --- |
| residual delta | `1.7342238036525468e-15` |
| solution-norm delta | `4.4408920985006262e-16` |
| max solution component delta | `4.4408920985006262e-16` |

The generated study recorded a dirty worktree because Day 5 ran before commit.
That is expected for local generated artifacts and remains an explicit caveat,
not release proof.

## Validation

Commands run:

```sh
python3 -m py_compile scripts/run_external_comparison.py
python3 scripts/run_external_comparison.py --self-check
python3 scripts/run_external_comparison.py --target qr-minnorm
python3 scripts/run_external_comparison.py --target qr-compatible-ls
```

All four commands passed.

## Preserved Boundaries

- `qr-minnorm` behavior and output path remain available.
- `qr-compatible-ls` is fixture-local only.
- No broad QR, external-library, platform, package, ABI, performance, release,
  or state-of-the-art claim is introduced.
- No C/header implementation file changed.
- Report-index freshness enforcement for the new selected family remains a
  later-day task so tests and metadata can be updated deliberately.

## Day 6 Handoff

Day 6 should integrate the selected comparison family with maintained fixture
and report metadata. Specifically:

- decide whether `tests/corpus/manifests/report_families.tsv` needs a new
  `comparison/qr_compatible_ls` row before Day 10 freshness enforcement;
- confirm whether fixture metadata is already sufficient or whether a
  source-controlled manifest row is needed;
- keep generated output under `build/comparison/qr_compatible_ls/`;
- preserve the Day 3 six-row contract for later normalizer enforcement.
