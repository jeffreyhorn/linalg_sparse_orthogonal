# Sprint 154 Day 4 Dependency Pinning Policy

## Purpose

Day 4 defines dependency discovery, version capture, skip/defer behavior,
provenance, and external-package boundaries for the selected Sprint 154
comparison target `qr_underdetermined_minnorm_2x4`.

## Selected Dependency Posture

The first narrow comparison study will use a source-controlled
external-process dense reference as the baseline.

Selected baseline:

- baseline name: `qr_external_dense_reference`;
- baseline type: `external_process_dense_reference`;
- baseline command:
  `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`;
- baseline implementation:
  `tests/qr_external_dense_reference.py`;
- baseline algorithm for this fixture:
  solve `A A^T y = b`, then compute `x = A^T y`, residual norm, and solution
  norm using Python stdlib floating-point arithmetic;
- baseline output protocol:
  first line `OK <value_count>`, followed by one floating-point value per
  output line.

No optional external package is required for the first study. NumPy and SciPy
remain deferred until a later sprint day or future sprint explicitly adds
optional-package discovery, version capture, and skip/defer rows.

## Required Local Dependencies

| Dependency | Required | Discovery | Version Capture | Missing Behavior |
| --- | --- | --- | --- | --- |
| repository source tree | yes | current Git worktree | `git rev-parse HEAD` and dirty/clean status | `error`; the harness cannot run without source provenance. |
| `python3` interpreter | yes | `sys.executable` or configured `PYTHON` path | `sys.version` and executable path | `error`; the selected baseline is source-controlled Python. |
| `tests/qr_external_dense_reference.py` | yes | source-controlled path relative to repo root | helper path plus helper source commit | `error`; the selected baseline is unavailable. |
| project static library/test executable | yes | local build path selected by harness design | compiler/configuration/build command fields | `error`; no project-side comparison can be emitted. |
| NumPy | no | not used in selected Day 4 policy | not applicable | `defer`; optional package baseline is not selected. |
| SciPy | no | not used in selected Day 4 policy | not applicable | `defer`; optional package baseline is not selected. |

The Day 4 policy intentionally avoids package-manager installation steps. A
future optional package baseline may discover locally installed packages, but
that discovery must not be worded as package-manager support.

## Version And Provenance Capture

Every generated comparison artifact or row should capture these fields:

- `source_commit`: `git rev-parse HEAD`;
- `source_branch`: current branch name if available;
- `worktree_state`: `clean` or `dirty`;
- `baseline_name`: `qr_external_dense_reference`;
- `baseline_type`: `external_process_dense_reference`;
- `baseline_version`: source-controlled helper version, initially represented
  by the source commit plus helper path;
- `baseline_command`: exact command line used to invoke the helper;
- `baseline_python_executable`: resolved Python executable path;
- `baseline_python_version`: full Python version string;
- `project_command`: exact project-side command or helper invocation;
- `project_version`: project version from `VERSION` if available;
- `platform`: OS/platform string;
- `compiler`: compiler identity for project-side build if available;
- `configuration`: build configuration and relevant flags;
- `fixture_key`: `qr_underdetermined_minnorm_2x4`;
- `metric`: selected metric name;
- `tolerance_kind` and `tolerance_value`;
- `status`;
- `caveat`;
- `artifact_path`.

If Day 5 decides to keep comparison rows artifact-only, these fields should
still appear in the study artifact.

## Status And Dependency Semantics

| Condition | Status | Evidence Meaning |
| --- | --- | --- |
| Source-controlled baseline helper runs and selected metrics match tolerance. | `pass` | Fixture-local comparison proof only. |
| Baseline helper runs but output is malformed. | `error` | No proof; harness or helper contract must be fixed. |
| Baseline helper runs but metric comparison misses tolerance. | `fail` | No proof; publish failure with metric diagnostics. |
| Required Python interpreter is missing. | `error` | No proof; selected baseline cannot run. |
| Required helper path is missing. | `error` | No proof; selected baseline cannot run. |
| Project-side build or runner is missing. | `error` | No proof; project output cannot be compared. |
| Optional NumPy/SciPy package is missing. | `defer` | No proof and no failure; optional package baseline is not selected. |
| Optional package baseline is explicitly disabled. | `defer` | No proof; selected source-controlled helper remains the only baseline. |

Because the selected baseline is required and source-controlled, missing
`python3` or missing helper files are errors, not skips. Skips are reserved for
future optional dependencies. Deferred optional-package rows must not count as
proof.

## Discovery Policy

The later harness should discover dependencies in this order:

1. Resolve repository root from the script location or an explicit `--root`.
2. Resolve Python from `sys.executable` unless a deliberate `--python` option
   is added.
3. Verify `tests/qr_external_dense_reference.py` exists under the repository
   root.
4. Run the helper with `qr_underdetermined_minnorm_2x4`.
5. Parse the `OK <value_count>` protocol and exact numeric row count.
6. Resolve project-side output from the harness design selected on Day 6.
7. Emit provenance before metric status if any downstream step fails.

The harness should not search system package managers, install dependencies,
download wheels, clone external repositories, or infer support from a user's
global environment.

## Security And Reproducibility Boundaries

The selected baseline is source-controlled and executes local Python code only.
The harness must:

- avoid network access;
- avoid package installation;
- avoid executing arbitrary discovered binaries;
- avoid importing optional packages unless a later policy explicitly selects
  them;
- record exact command lines and paths;
- keep generated artifacts under ignored local `build/` paths unless a later
  day explicitly source-controls a study artifact;
- treat dirty worktree state as a provenance caveat, not a hidden condition.

Future optional NumPy/SciPy support must add:

- explicit import/discovery code;
- `numpy.__version__` and/or `scipy.__version__` capture;
- import failure as `defer`, not `pass`;
- package-baseline caveats that avoid package-manager, hosted CI, or ecosystem
  parity wording;
- tests for missing optional dependency behavior.

## Non-Claims Preserved By This Policy

This dependency policy does not claim:

- NumPy or SciPy availability;
- package-manager support;
- broad external-library parity;
- hosted CI proof;
- platform portability beyond the recorded local run;
- shared-library support or dynamic ABI compatibility;
- performance superiority;
- state-of-the-art behavior.

It also does not make the source-controlled dense-reference helper a
replacement for LAPACK, NumPy, SciPy, or any external library. It is a
fixture-local external-process reference for the first narrow study.

## Day 5 Handoff

Day 5 should design the output schema around the selected dependency policy.
At minimum, comparison rows or artifact records need:

- source and worktree provenance;
- baseline command, type, helper path, and Python version;
- project command and build/configuration metadata;
- fixture key;
- metric name;
- expected/project/baseline values where applicable;
- tolerance;
- status;
- caveat;
- artifact path;
- local-only support tier.

Day 5 should also decide whether Sprint 154 creates normalized comparison rows
or keeps the first study artifact-only until the schema has more experience.
