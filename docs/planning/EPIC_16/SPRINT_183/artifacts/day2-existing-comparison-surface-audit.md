# Sprint 183 Day 2: Existing Comparison Surface Audit

## Purpose

Audit the current external comparison runner, selected manifest rows, generated
artifact conventions, freshness checks, tests, and workflow guards before
selecting a fifth bounded external comparison family.

## Runner Inventory

`scripts/run_external_comparison.py` currently owns four selected targets:

| Target | Comparison kind | Fixture | Output directory | Rows |
| --- | --- | --- | --- | ---: |
| `qr-minnorm` | QR solve | `qr_underdetermined_minnorm_2x4` | `build/comparison/qr_minnorm/` | 6 |
| `qr-compatible-ls` | QR solve | `qr_overdetermined_compatible_5x3` | `build/comparison/qr_compatible_ls/` | 6 |
| `partial-svd-diag6-k2` | Partial SVD | `partial_svd_diag6_k2` | `build/comparison/partial_svd_diag6_k2/` | 10 |
| `lu-nonsym-square-5` | LU solve | `lu_nonsym_square_5` | `build/comparison/lu_nonsym_square_5/` | 6 |

The runner uses deterministic target dictionaries for fixture key, subfamily,
operation, output directory, expected metrics, tolerances, claim scope, summary
text, and non-claims. A Sprint 183 target should follow that model unless Day
7 deliberately designs a justified extension.

## Baseline Helper Surface

| Family | Helper | Current role |
| --- | --- | --- |
| QR | `tests/qr_external_dense_reference.py` | Source-controlled dense QR reference helper for selected QR comparison rows. |
| Partial SVD | `tests/svd_external_dense_reference.py` | Source-controlled dense SVD reference helper for selected partial-SVD comparison rows. |
| LU | `tests/lu_external_dense_reference.py` | Source-controlled dense LU reference helper for selected linked-list LU comparison rows. |

Optional NumPy/SciPy dependency rows remain deferred context and are not pass
evidence. A new family should prefer a source-controlled helper and keep
optional package rows out of selected pass evidence.

## Output File Contract

Every selected comparison target emits:

- `project_observations.tsv`;
- `baseline_observations.tsv`;
- `dependency_status.tsv`;
- `study.tsv`;
- `summary.md`;
- `manifest.tsv`.

The selected workflow guard requires these files for every manifest-owned
comparison directory and rejects broad `build/comparison/**` uploads.

## Selected Manifest Surface

The four current comparison rows in
`tests/corpus/manifests/selected_report_targets.tsv` share these properties:

| Field | Current comparison value |
| --- | --- |
| `selection_scope` | `reviewed_cross_platform_selected` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `workflow_file` | `.github/workflows/ci.yml;.github/workflows/macos-ci.yml` |
| `workflow_job` | `generated-report-freshness;selected-comparison-freshness` |
| `workflow_platforms` | `linux;macos` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |

Sprint 183 should preserve this exactness for the new selected row. Windows
must remain absent unless a separate Windows-safe promotion path is implemented
with manifest, workflow, guard, and documentation support.

## Freshness Target

`make report-index-comparison-freshness` currently runs:

1. `python3 scripts/run_external_comparison.py --target qr-minnorm`
2. `python3 scripts/run_external_comparison.py --target qr-compatible-ls`
3. `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2`
4. `python3 scripts/run_external_comparison.py --target lu-nonsym-square-5`
5. `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`

A new selected target must be generated before the required freshness check can
expect its rows.

## Test Surface

| Test file | Current responsibility |
| --- | --- |
| `tests/test_run_external_comparison.py` | Verifies target support diagnostics, generated files, study row IDs, metric sets, dependency rows, and report-family metadata. |
| `tests/test_selected_comparison_workflow.py` | Verifies Linux/macOS selected comparison workflow integration, exact uploaded files, fail-closed uploads, summary checks, and Windows deferral. |
| `tests/test_selected_report_targets_manifest.py` | Verifies selected manifest structure, workflow metadata cardinality, unsupported families, and Windows non-selection. |
| `tests/test_normalize_report_index.py` | Verifies report-index selected target handling and freshness behavior. |

The new family should extend these tests where the selected row, runner target,
expected row IDs, or workflow artifact expectations change.

## Generated Artifact Observation

Existing generated comparison files are present locally under
`build/comparison/` for all four current targets. Day 2 inspected the file
names only. `git status --short -- build/comparison` reports no tracked or
staged generated artifact changes.

## New-Family Invariants

- Add exactly one bounded family.
- Keep generated output under `build/comparison/<subfamily>/`.
- Emit the six standard output files.
- Use exact expected row IDs and expected row count.
- Keep selected report rows `local_only` unless a later sprint changes support
  semantics.
- Keep optional dependency rows as defer context, not pass evidence.
- Keep workflow uploads exact and fail-closed.
- Keep Windows report freshness deferred unless deliberately promoted.
- Preserve broad parity, package, ABI, performance, release, and
  state-of-the-art non-claims.

## Day 3 Handoff

Day 3 should inventory candidate families against this existing surface. The
best candidates will reuse the runner's current solve or factorization shape,
have a small deterministic fixture, and need minimal workflow/report-index
contract changes.

## Validation

Day 2 changes planning artifacts only. Validation:

- `git status --short -- build/comparison`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Notes |
| --- | --- | --- |
| New family design starts from the existing selected comparison contract. | Complete | Runner, manifest, freshness target, docs, and tests were audited. |
| Artifact, row-count, and guard invariants are explicit. | Complete | Required files, row counts, workflow artifacts, and broad-upload rejection are recorded. |
| Generated local outputs remain unstaged. | Complete | Existing `build/comparison/` files were observed but not staged. |
