# Day 10: Report Integration Design

## Product Decision

Keep the first Sprint 154 comparison study artifact-only for now.

Do not add a source-controlled `comparison` report family on Day 10. The Day 9
runner now emits schema-complete local generated rows in
`build/comparison/qr_minnorm/study.tsv`, but report-index promotion should wait
until Day 11 can implement and validate explicit freshness, row-count,
stale-output, local-only, and non-claim behavior in the normalizer.

This is a deliberate product decision, not a missing implementation. The
comparison rows are useful local evidence, but they are not yet maintained
report-index evidence.

## Reviewed Inputs

Day 10 reviewed the Day 9 generated output contract:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/comparison/qr_minnorm/manifest.tsv`

The selected Day 9 study rows are:

| Metric | Current status | Report status |
| --- | --- | --- |
| `project_status` | `pass` | artifact-only |
| `baseline_status` | `pass` | artifact-only |
| `residual_norm` | `pass` | artifact-only |
| `solution_norm` | `pass` | artifact-only |
| `solution_values` | `pass` | artifact-only |
| `project_vs_baseline_max_abs_delta` | `pass` | artifact-only |

## Why Not Promote Immediately

The current normalized report index has mature treatment for these families:

- source-controlled corpus metadata;
- generated-local oracle rows with selected strict freshness gates;
- benchmark, sentinel, guardrail, dead-code, and coverage advisory rows;
- package, CI, documentation, and runtime-backend policy rows.

Adding comparison rows now would create a new generated-local evidence family
without enough policy around:

- exact selected comparison row count;
- selected row ids;
- required `pass` semantics;
- missing-row handling;
- duplicate-row handling;
- stale source commit handling;
- dirty worktree caveats;
- optional package `defer` handling;
- local-only support-tier wording;
- generated summary interpretation;
- how `--require-generated comparison` should behave.

The Day 9 runner already enforces many of these rules for its own outputs. The
report-index normalizer should not partially duplicate or reinterpret those
rules until Day 11 can add a complete integration.

## Selected Path

Day 10 selects this path:

1. Keep `build/comparison/qr_minnorm/study.tsv` as the canonical generated row
   artifact for the first narrow study.
2. Keep `build/comparison/qr_minnorm/summary.md` as the human-readable local
   generated study summary.
3. Keep `build/comparison/qr_minnorm/manifest.tsv` as the run-level provenance
   source.
4. Treat the comparison output as local-only generated evidence.
5. Treat all comparison proof claims as fixture-local and selected-row-only.
6. Do not count `skip` or `defer` rows as proof.
7. Do not imply hosted CI, release, package, ABI, platform, performance,
   external-library ecosystem, NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or
   state-of-the-art proof.

## Future Normalization Policy

If Day 11 promotes comparison rows into the normalized report index, use this
policy.

### Report-Family Contract

Add a single source-controlled manifest row:

| Field | Value |
| --- | --- |
| `report_family` | `comparison` |
| `subfamily` | `qr_minnorm` |
| `row_meaning` | `external_process_dense_reference_comparison` |
| `row_origin` | `generated_local` |
| `status` | `unknown` |
| `support_tier` | `local_only` |
| `freshness_policy` | `generated_compare_inputs` |
| `generator_command` | `python3 scripts/run_external_comparison.py --target qr-minnorm` |
| `artifact_pattern` | `build/comparison/qr_minnorm/study.tsv` |

### Selected Row Count

Require exactly six selected generated rows:

1. `comparison_qr_underdetermined_minnorm_2x4_project_status_v1`
2. `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1`
3. `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1`
4. `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1`
5. `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`
6. `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1`

Missing or duplicate selected rows should be `error` when comparison freshness
is required.

### Freshness

Use strict generated-input freshness only when a caller asks for comparison
freshness, for example:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
```

Freshness should be interpreted this way:

- source commit equal to current `HEAD`: `fresh`;
- source commit different from current `HEAD`: `stale`;
- missing `study.tsv`: `not_generated`;
- selected row status `fail` or `error`: freshness error when required;
- selected row status `skip` or `defer`: not proof and should not satisfy a
  required comparison gate;
- dirty worktree: allowed only as explicit provenance and caveat, not release
  proof.

### Status Mapping

Comparison row status should map as:

| Study status | Normalized status | Meaning |
| --- | --- | --- |
| `pass` | `pass` | Selected row met its fixture-local tolerance. |
| `fail` | `fail` | Required metric missed tolerance. |
| `skip` | `skip` | Optional future comparison lane was skipped. |
| `defer` | `defer` | Optional future baseline or target was intentionally not selected. |
| `error` | `fail` or `unknown` | Required command, parser, or output contract failed. |

Only `pass` rows may support the fixture-local claim, and only if all selected
rows are present and pass.

## Non-Claim Register

The comparison report family, if added later, must preserve these non-claims:

- no broad QR parity;
- no NumPy parity;
- no SciPy parity;
- no LAPACK parity;
- no SuiteSparse parity;
- no Eigen parity;
- no external-library ecosystem parity;
- no hosted CI proof;
- no release proof;
- no platform portability proof;
- no package-manager proof;
- no shared-library or ABI proof;
- no performance superiority;
- no state-of-the-art claim.

## Day 11 Implementation Checklist

Day 11 should either implement the selected report-index integration completely
or keep the artifact-only decision documented.

If implementing integration:

1. Add a `comparison/qr_minnorm` contract row to
   `tests/corpus/manifests/report_families.tsv`.
2. Extend `scripts/validate_corpus_schema.py` only if existing manifest enums
   reject the new family, origin, status, support tier, or freshness policy.
3. Add a `comparison_generated_rows()` loader to
   `scripts/normalize_report_index.py`.
4. Map `study.tsv` fields into `NORMALIZED_FIELDS` without dropping command,
   commit, branch, platform, compiler, configuration, support tier, claim
   scope, non-claims, status, caveat, or artifact path.
5. Add selected comparison diagnostics for exact row ids and row count.
6. Teach freshness diagnostics that required comparison rows fail closed on
   missing, stale, duplicate, non-pass, or malformed selected rows.
7. Add focused normalizer checks for:
   - missing `study.tsv`;
   - present fresh `study.tsv`;
   - stale source commit;
   - selected-row count mismatch;
   - selected-row non-pass status;
   - optional dependency `defer` not counting as proof.
8. Update maintainer docs only after the behavior is implemented.

If not implementing integration on Day 11:

1. Leave report families unchanged.
2. Keep the runner-generated `summary.md` and `study.tsv` as the canonical
   local generated artifacts.
3. Document the deferral reason in the Day 11 artifact.
4. Re-run the Day 9 harness and normal report-index checks to ensure no
   accidental report promotion happened.
