# Sprint 154 Day 7 Harness Project Runner Scaffold

## Purpose

Day 7 implements the first comparison harness scaffold for the selected
`qr_underdetermined_minnorm_2x4` target. This batch focuses on project-side
fixture execution, provenance capture, and output-path scaffolding. Baseline
execution and full metric comparison remain Day 8-9 work.

## Implementation

Added:

- `scripts/run_external_comparison.py`

Initial supported command:

```sh
python3 scripts/run_external_comparison.py --target qr-minnorm
```

Initial generated local output path:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/manifest.tsv`

## Supported Target

The scaffold intentionally supports only:

- target: `qr-minnorm`;
- fixture: `qr_underdetermined_minnorm_2x4`;
- operation: `minnorm_solve`;
- expected solution: `0.5,0.5,0.5,0.5`;
- expected residual tolerance: `1e-10`;
- expected solution norm: `1.0`;
- expected solution tolerance: `1e-10`.

Unsupported target names fail with `unsupported_target`.

## Project-Side Runner

The script:

1. resolves repository root and output directory;
2. builds `build/libsparse_lu_ortho.a` if missing;
3. generates a temporary C probe for the selected QR minimum-norm fixture;
4. compiles the probe against the static library;
5. runs `sparse_qr_solve_minnorm(A, b, x, NULL)`;
6. parses stable key/value probe output:
   - `status`;
   - `residual_norm`;
   - `solution_norm`;
   - `solution_values`;
7. compares project-side output against the selected Day 3 expected values;
8. writes project observations and manifest metadata.

## Provenance Captured

`manifest.tsv` records:

- target;
- fixture key;
- generated UTC timestamp;
- source commit;
- source branch;
- worktree state;
- project version;
- platform;
- compiler identity;
- Day 7 configuration caveat;
- project probe command;
- project observations path.

The Day 7 configuration explicitly records
`baseline_status=not_yet_integrated` so the scaffold cannot be mistaken for a
complete external comparison study.

## Output Semantics

`project_observations.tsv` records project-side metrics only:

- `project_status`;
- `residual_norm`;
- `solution_norm`;
- `solution_values`.

All rows must pass the selected fixture-local tolerance policy before the
script exits successfully. These rows are project-side scaffold evidence, not
external-baseline comparison proof.

## Non-Claims

Day 7 does not claim:

- baseline comparison proof;
- external-library parity;
- broad QR or minimum-norm correctness;
- hosted CI proof;
- performance proof;
- package, ABI, platform, or runtime-loader support;
- state-of-the-art behavior.

## Validation

Day 7 validation:

- `python3 scripts/run_external_comparison.py --target qr-minnorm` passed;
- generated `build/comparison/qr_minnorm/manifest.tsv`;
- generated `build/comparison/qr_minnorm/project_observations.tsv`;
- project-side `project_status` passed with `SPARSE_SUCCESS`;
- project-side `residual_norm` passed with `1.5700924586837752e-16`;
- project-side `solution_norm` passed with `0.99999999999999989`;
- project-side `solution_values` passed with
  `0.49999999999999989,0.49999999999999989,0.5,0.5`;
- `python3 -m py_compile scripts/run_external_comparison.py` passed;
- `python3 scripts/validate_corpus_schema.py` passed;
- `python3 scripts/normalize_report_index.py --family corpus --family oracle --check`
  passed with `78` rows ok;
- `git diff --check` passed.

Because Day 7 adds a Python script and planning artifacts but does not modify
`.c` files or public `.h` headers, the full C quality gate is not required.

## Day 8 Handoff

Day 8 should add the baseline runner:

- invoke `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`;
- parse the `OK 6` protocol;
- record baseline command, Python executable, and Python version;
- write baseline status into the generated artifacts;
- preserve source-controlled dense-reference non-claims;
- keep optional NumPy/SciPy rows deferred.
