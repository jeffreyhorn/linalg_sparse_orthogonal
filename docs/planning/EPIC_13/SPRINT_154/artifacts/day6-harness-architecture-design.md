# Sprint 154 Day 6 Harness Architecture Design

## Purpose

Day 6 designs the command flow, project-side runner, baseline runner, output
paths, provenance collection, diagnostics, and implementation checklist for
the first narrow comparison harness.

The selected target remains `qr_underdetermined_minnorm_2x4`.

## Existing Implementation Inputs

Useful source patterns already exist:

- `scripts/run_corpus_oracle.py` builds temporary static-library probes,
  captures compiler/configuration metadata, emits generated-local rows, and
  validates expected row status.
- `build_sprint150_minnorm_qr_rows()` in `scripts/run_corpus_oracle.py`
  already generates project-side QR minimum-norm observations for the selected
  Sprint 150 fixture family.
- `tests/qr_external_dense_reference.py` already emits the selected
  external-process dense-reference values with protocol `OK <value_count>`
  followed by one numeric value per line.
- `tests/corpus/expected/qr_underdetermined_minnorm_2x4.tsv` already defines
  the four selected expected rows and tolerances.

The comparison harness should reuse these patterns without changing the
selected oracle freshness gate or report-index policy until Days 10-11.

## Proposed Harness Entry Point

Add a new script:

- `scripts/run_external_comparison.py`

Initial supported mode:

```sh
python3 scripts/run_external_comparison.py --target qr-minnorm
```

Optional arguments:

- `--root <path>`: repository root, default inferred from script location;
- `--output-dir <path>`: default `build/comparison/qr_minnorm`;
- `--keep-temp`: keep temporary build/probe files for debugging;
- `--check`: validate an existing generated artifact against current source
  commit and selected row policy without regenerating, if implemented later.

Unsupported targets should fail with an explicit error. The first version
should not accept broad target names such as `qr`, `svd`, `scipy`, `numpy`, or
`all`.

## Command Flow

The harness should run in this order:

1. Resolve repository root and output directory.
2. Collect source provenance:
   - commit;
   - branch;
   - worktree state;
   - project version.
3. Create or reset `build/comparison/qr_minnorm/`.
4. Run the source-controlled baseline command:
   `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`.
5. Parse and validate the baseline output protocol:
   - first line must be `OK 6`;
   - next four values are baseline solution values;
   - fifth value is baseline residual norm;
   - sixth value is baseline solution norm.
6. Build or reuse the static library needed for project-side probe execution.
7. Compile and run a temporary project-side probe for
   `qr_underdetermined_minnorm_2x4`.
8. Parse project-side observations:
   - status;
   - residual norm;
   - solution norm;
   - solution values.
9. Compare project output against expected rows and baseline output.
10. Emit `study.tsv`, `manifest.tsv`, and `summary.md`.
11. Exit non-zero for required dependency errors, malformed output, duplicate
    selected rows, missing selected rows, or selected metric failures.

## Project-Side Runner Design

The project-side runner should use the same fixture as the corpus proof owner:

- matrix rows: `2`;
- matrix cols: `4`;
- nonzeros: `4`;
- equations: `x0 + x1 = 1`, `x2 + x3 = 1`;
- RHS: `[1.0, 1.0]`;
- expected solution: `[0.5, 0.5, 0.5, 0.5]`.

The probe should:

- construct the sparse matrix directly from the deterministic fixture entries;
- call `sparse_qr_solve_minnorm(A, b, x, NULL)`;
- print stable key/value lines:
  - `status=SPARSE_SUCCESS` or an error status;
  - `residual_norm=<float>`;
  - `solution_norm=<float>`;
  - `solution_values=<comma-separated floats>`.

The implementation may reuse helper patterns from
`scripts/run_corpus_oracle.py`, including temporary C source generation and
static-library linking, but should keep comparison-specific output separate.

## Baseline Runner Design

The baseline runner should:

- invoke the selected command using the current Python executable;
- require `OK 6`;
- parse six values:
  - `solution_values[0..3]`;
  - `residual_norm`;
  - `solution_norm`;
- record baseline command, helper path, Python executable, and Python version;
- classify malformed output as `error`;
- classify non-zero exit as `error`;
- never call NumPy, SciPy, package managers, or network resources in the
  selected first-study path.

## Output Path Policy

Generated local outputs:

- root: `build/comparison/qr_minnorm/`;
- metric rows: `build/comparison/qr_minnorm/study.tsv`;
- run metadata: `build/comparison/qr_minnorm/manifest.tsv`;
- human summary: `build/comparison/qr_minnorm/summary.md`.

The output directory may be reset before writing a new run. Resetting this
directory must not affect corpus oracle output under `build/corpus/`.

Generated comparison outputs are local artifacts. They should remain ignored
unless a later day deliberately publishes a source-controlled study artifact.

## Row Emission Design

Emit one row per selected metric:

- `comparison_qr_underdetermined_minnorm_2x4_project_status_v1`;
- `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1`;
- `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1`;
- `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1`;
- `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`;
- `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1`.

All rows should carry:

- source commit, branch, and worktree state;
- generated timestamp;
- baseline command and Python version;
- project command, compiler, and configuration;
- fixture key;
- metric;
- expected/project/baseline/delta values;
- tolerance;
- status and stable status reason;
- local-only claim scope and non-claims.

## Failure Messages

Use stable and specific failure classes:

| Failure Class | Trigger |
| --- | --- |
| `unsupported_target` | Target is not `qr-minnorm`. |
| `missing_python` | Python executable cannot run the baseline helper. |
| `missing_baseline_helper` | `tests/qr_external_dense_reference.py` is missing. |
| `baseline_command_failed` | Baseline command exits non-zero. |
| `baseline_malformed_output` | Baseline output does not match `OK 6` plus six numeric values. |
| `project_build_failed` | Static library or probe build fails. |
| `project_probe_failed` | Project-side probe exits non-zero or omits required keys. |
| `metric_tolerance_miss` | Project or baseline metric misses selected tolerance. |
| `missing_selected_row` | A required comparison row was not emitted. |
| `duplicate_selected_row` | A required comparison row was emitted more than once. |
| `unsupported_claim_boundary` | Claim scope includes forbidden broad wording. |

The harness should report the fixture key, metric, expected value, observed
value, tolerance, and artifact path whenever possible.

## Day 7 Implementation Checklist

Day 7 should implement the project-side scaffold first:

1. Add `scripts/run_external_comparison.py`.
2. Implement CLI parsing for `--target qr-minnorm`, `--root`, and
   `--output-dir`.
3. Implement repository, source, project version, platform, and Python
   provenance capture.
4. Implement selected fixture metadata as constants.
5. Implement output directory creation and artifact path constants.
6. Implement project-side probe generation, static-library build/reuse, probe
   compilation, and key/value parsing, reusing `scripts/run_corpus_oracle.py`
   patterns where practical.
7. Emit a provisional `manifest.tsv` and project-side observations for smoke
   validation.
8. Add a focused smoke mode or check that verifies the selected fixture key,
   expected solution values, and project-side observation keys.

Day 8 should then add baseline command execution and parser handling.
