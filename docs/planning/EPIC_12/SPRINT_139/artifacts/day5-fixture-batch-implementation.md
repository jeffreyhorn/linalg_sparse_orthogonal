# Sprint 139 Day 5: Fixture Batch Implementation

## Purpose

Day 5 implements the Day 4 fixture batch decision. The selected Sprint 139 QR
closure fixture already exists in the maintained corpus layout, so Day 5 is a
confirmation and validation pass rather than a broad fixture expansion.

This artifact records that no new source-controlled QR fixture rows are needed
before oracle and proof-owner work begins.

## Implementation Decision

No corpus metadata changes are required on Day 5.

The existing source-controlled rows already implement the first-class Sprint
139 fixture batch:

- `tests/corpus/manifests/fixtures.tsv`
- `tests/corpus/manifests/generators.tsv`
- `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv`

Reasoning:

- Day 4 selected a single first-class fixture:
  `qr_rank_deficient_6x4_nullspace_v1`.
- The fixture row already exists, is generated, and points to
  `qr_rank_deficient_6x4_nullspace_generator_v1`.
- The generator row already records deterministic algorithm metadata, stable
  parameters, structure hash, value hash, canonical format, and change policy.
- The expected-result file already contains ready rows for rank, nullity, and
  normalized residual.
- Adding new QR fixture rows now would widen the sprint beyond the selected
  closure and risk partial progress across more residuals.

## Confirmed Fixture Row

| Field | Confirmed value |
| --- | --- |
| `fixture_key` | `qr_rank_deficient_6x4_nullspace_v1` |
| `fixture_family` | `qr_rank_deficient` |
| `storage_kind` | `generated` |
| `matrix_path` | empty |
| `generator_key` | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| `rows` | `6` |
| `cols` | `4` |
| `nnz` | `14` |
| `rank_status` | `rank_deficient` |
| `expected_rank` | `3` |
| `nullity` | `1` |
| `conditioning_class` | `moderate` |
| `scale_class` | `unit` |
| `sparsity_class` | `structured_sparse` |
| `expected_behavior` | `success` |
| `support_tier` | `local_only` |

Confirmed claim boundary:

> Fixture-local generated reference rank/nullity and normalized null-vector
> residual metadata.

Confirmed non-claims:

- no QR solver pass evidence before Sprint 139 proof work;
- no raw-basis parity;
- no broad QR correctness;
- no global minimum-norm guarantee;
- no SuiteSparse parity;
- no broad corpus completeness;
- no SVD correctness claim.

## Confirmed Generator Row

| Field | Confirmed value |
| --- | --- |
| `generator_key` | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| `generator_version` | `1` |
| `algorithm` | `fixed_columns_c3_equals_c0_plus_c1` |
| `seed` | `none` |
| `parameters` | `rows=6;cols=4;expected_rank=3;nullity=1;dependency=c3-c0-c1` |
| `expected_structure_hash` | `81496065f83410049f2c32556a3cb705375fe1e076112149a750489b4854f505` |
| `expected_value_hash` | `2c6e0846a8a8bbe2c67786c25c029237acfccc891817ed3038b0b0e3646c36e2` |
| `canonical_format` | `coo_zero_based_row_col_value_f64_text_v1` |
| `regeneration_command` | `python3 scripts/run_corpus_oracle.py` |

Local hash reproduction confirmed:

```text
structure_hash=81496065f83410049f2c32556a3cb705375fe1e076112149a750489b4854f505
value_hash=2c6e0846a8a8bbe2c67786c25c029237acfccc891817ed3038b0b0e3646c36e2
rows=6 cols=4 nnz=14
```

## Confirmed Expected Rows

| Row ID | Status | Comparison | Expected result | Tolerance |
| --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `ready_for_oracle` | rank | `3` | `exact=0` |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `ready_for_oracle` | nullity | `1` | `exact=0` |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `ready_for_oracle` | residual norm | `normalized_null_vector_residual<=1e-10` | `absolute=1e-10` |

These rows remain expected-result rows. They are prerequisites for solver-backed
observed evidence, not solver pass evidence by themselves.

## Deferred Fixture Additions

Day 5 intentionally does not add:

- a duplicate-column projector source-controlled fixture;
- near-dependent rank-threshold source-controlled fixtures;
- least-squares or minimum-norm source-controlled fixtures;
- COLAMD/reordered QR fixtures;
- SuiteSparse optional-data pass fixtures;
- broad external-library parity fixture rows.

These remain deferred for the reasons recorded in Day 2 and Day 4. The next
implementation work should focus on oracle comparison design and the dedicated
QR proof owner for the selected fixture.

## Validation

Commands run:

```sh
python3 scripts/validate_corpus_schema.py
ruby -e 'ok=true; Dir.glob("tests/corpus/**/*.tsv").each do |f|; rows=File.readlines(f, chomp: true); next if rows.empty?; widths=rows.map { |r| r.split("\t", -1).length }.uniq; if widths.length != 1; warn "#{f}: inconsistent TSV widths #{widths.inspect}"; ok=false; end; end; exit(ok ? 0 : 1)'
env -u SPARSE_CORPUS_OPTIONAL_DATA_DIR python3 -B scripts/run_corpus_oracle.py --root tests/corpus --oracle-dir build/corpus/oracle --report-dir build/corpus-reports
python3 - <<'PY'
from scripts.validate_corpus_schema import GENERATED_FIXTURES, canonical_structure_text, canonical_value_text, sha256_text
fixture = GENERATED_FIXTURES['qr_rank_deficient_6x4_nullspace_generator_v1']
entries = fixture['entries']()
print('structure_hash=' + sha256_text(canonical_structure_text(fixture['rows'], fixture['cols'], entries)))
print('value_hash=' + sha256_text(canonical_value_text(fixture['rows'], fixture['cols'], entries)))
print(f"rows={fixture['rows']} cols={fixture['cols']} nnz={len(entries)}")
PY
```

Results:

- corpus schema validation passed;
- corpus TSV width consistency passed;
- oracle/report command generated local ignored outputs under `build/`;
- reproduced structure and value hashes match the generator manifest;
- no source-controlled corpus row changes were necessary.

## Day 5 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| New fixture rows pass maintained schema validation. | Complete | No new rows were needed; existing selected fixture rows pass `python3 scripts/validate_corpus_schema.py`. |
| Expected-result rows preserve explicit non-claims. | Complete | Existing rank, nullity, and residual rows retain QR solver-pass, broad QR, SuiteSparse, and raw-basis non-claims. |
| Generated or stored fixture metadata is reproducible and reviewable. | Complete | Hash reproduction matched the generator manifest and oracle/report generation completed locally. |
