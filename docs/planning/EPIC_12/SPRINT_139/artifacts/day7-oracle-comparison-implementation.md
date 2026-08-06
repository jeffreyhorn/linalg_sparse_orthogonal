# Sprint 139 Day 7: Oracle Comparison Implementation

## Purpose

Day 7 implements the solver-backed QR oracle lane for the selected Sprint 139
fixture. The implementation extends the maintained corpus oracle runner with an
explicit opt-in solver QR path while preserving the existing generated-reference
rows and optional-data skip semantics.

## Implementation Summary

Updated `scripts/run_corpus_oracle.py` with:

- `--include-solver-qr`, an opt-in flag that appends solver-backed QR rows;
- `--solver-library`, defaulting to `build/libsparse_lu_ortho.a`;
- a temporary C probe that builds `qr_rank_deficient_6x4_nullspace_v1`, links
  against the static library, runs `sparse_qr_factor()`, records rank, nullity,
  and normalized nullspace residual, and then exits without creating
  source-controlled C files;
- separate solver-backed row IDs:
  - `qr_rank_deficient_6x4_nullspace_v1_qr_rank`
  - `qr_rank_deficient_6x4_nullspace_v1_qr_nullity`
  - `qr_rank_deficient_6x4_nullspace_v1_qr_nullspace_residual`
- solver-backed row provenance with `solver_family=qr`, compiler identity,
  `proof_owner=runtime_qr_probe`, fixture hashes, support tier, command, and
  non-claim fences;
- manifest metadata for oracle row count, solver families, and solver QR row
  count.

The default command remains backward-compatible: without `--include-solver-qr`,
it emits the original generated-reference rows only.

## Row Semantics

| Row family | Row IDs | `solver_family` | Meaning |
| --- | --- | --- | --- |
| Generated reference | existing first-lane row IDs | `unknown` | Regenerates deterministic fixture metadata and expected reference facts; not QR solver pass evidence. |
| Solver-backed QR | new `_qr_*` row IDs | `qr` | Runs the QR implementation through a temporary static-library probe and records observed rank, nullity, and normalized residual. |

The split avoids silently changing the meaning of the existing Sprint 138 rows.

## Solver-Backed Observations

The Day 7 local run emitted:

| Row ID | Observed result | Expected result | Status |
| --- | ---: | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_qr_rank` | `3` | `3` | `pass` |
| `qr_rank_deficient_6x4_nullspace_v1_qr_nullity` | `1` | `1` | `pass` |
| `qr_rank_deficient_6x4_nullspace_v1_qr_nullspace_residual` | approximately `2.22e-16` | `<= 1e-10` | `pass` |

These rows may support only fixture-local solver-backed QR rank, nullity, and
normalized nullspace residual evidence for
`qr_rank_deficient_6x4_nullspace_v1`.

## Command

Default generated-reference command:

```sh
python3 scripts/run_corpus_oracle.py --root tests/corpus --oracle-dir build/corpus/oracle --report-dir build/corpus-reports
```

Solver-backed QR command:

```sh
env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py --root tests/corpus --oracle-dir build/corpus/oracle --report-dir build/corpus-reports --include-solver-qr
```

The solver-backed command requires a built static library at
`build/libsparse_lu_ortho.a` or an explicit `--solver-library` path.

## Generated Outputs

Local generated outputs remain ignored and uncommitted:

- `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

The explicit Day 7 run produced six oracle rows:

- three generated-reference rows with `solver_family=unknown`;
- three solver-backed QR rows with `solver_family=qr`.

The report index includes all six oracle comparison rows plus the optional-data
skip row for `suitesparse_rank_deficient_qr_subset_v1`.

## Provenance and Non-Claims

Solver-backed QR rows record:

- compiler identity;
- `build_profile=static_default`;
- `optional_data_policy=disabled`;
- `proof_owner=runtime_qr_probe`;
- structure and value hashes;
- `qr_tolerance=1e-10`;
- `support_tier=local_only`.

Solver-backed QR rows preserve these non-claims:

- no broad QR correctness;
- no raw-basis parity;
- no global rank-threshold policy;
- no broad rank-deficient solve;
- no minimum-norm or least-squares claim;
- no SuiteSparse parity;
- no external-library parity;
- no platform parity;
- no performance or state-of-the-art claim.

## Validation

Commands run:

```sh
python3 -m py_compile scripts/validate_corpus_schema.py scripts/run_corpus_oracle.py
python3 -B scripts/validate_corpus_schema.py
env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py --root tests/corpus --oracle-dir build/corpus/oracle --report-dir build/corpus-reports
env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py --root tests/corpus --oracle-dir build/corpus/oracle --report-dir build/corpus-reports --include-solver-qr
tmpdir=$(mktemp -d) && (cd "$tmpdir" && env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B /Users/jeff/experiments/linalg_sparse_orthogonal/scripts/run_corpus_oracle.py --include-solver-qr)
env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py --root tests/corpus --oracle-dir build/corpus/oracle --report-dir build/corpus-reports --include-solver-qr
python3 - <<'PY'
import csv
from pathlib import Path
with open('build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv', newline='') as handle:
    oracle_rows = list(csv.DictReader(handle, delimiter='\t'))
with open('build/corpus-reports/index.tsv', newline='') as handle:
    report_rows = list(csv.DictReader(handle, delimiter='\t'))
manifest = Path('build/corpus-reports/manifest.txt').read_text()
generated = [r for r in oracle_rows if r['solver_family'] == 'unknown']
qr = [r for r in oracle_rows if r['solver_family'] == 'qr']
assert len(generated) == 3
assert len(qr) == 3
assert {r['comparison_status'] for r in qr} == {'pass'}
expected_qr_ids = {
    'qr_rank_deficient_6x4_nullspace_v1_qr_rank',
    'qr_rank_deficient_6x4_nullspace_v1_qr_nullity',
    'qr_rank_deficient_6x4_nullspace_v1_qr_nullspace_residual',
}
assert {r['oracle_row_id'] for r in qr} == expected_qr_ids
assert expected_qr_ids <= {r['row_subject'] for r in report_rows}
assert 'oracle_row_count=6' in manifest
assert 'solver_families=qr,unknown' in manifest
assert 'solver_qr_row_count=3' in manifest
PY
```

Results:

- Python compile checks passed.
- Corpus schema validation passed.
- Default generated-reference oracle command passed.
- Solver-backed QR oracle command passed.
- Non-repo-CWD solver-backed smoke test passed.
- Solver QR oracle/report metadata check passed.

No `.c` or `.h` files were modified by Day 7, so the full C quality gate was
not required. The temporary C probe is generated under a temporary directory at
runtime and is not a source-controlled project file.

## Day 7 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Oracle rows are mechanically parseable and reproducible. | Complete | `--include-solver-qr` emits TSV rows with stable `_qr_*` IDs, key/value configuration, compiler provenance, and manifest row counts. |
| Passing rows reflect only the selected fixture-local QR behavior. | Complete | Solver-backed rows use `solver_family=qr`, `support_tier=local_only`, and explicit QR non-claims. |
| Generated reports include freshness and provenance fields. | Complete | Report rows and manifest record command, commit, branch, platform, compiler/configuration, solver families, and row counts. |
