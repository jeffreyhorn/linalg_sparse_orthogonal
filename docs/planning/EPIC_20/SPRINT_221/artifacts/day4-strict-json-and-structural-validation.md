# Sprint 221 Day 4: Strict JSON And Structural Validation

## Scope

The v1 parser and structural checker now execute the source-controlled
`external_baseline_evidence_v1.json` field shape using only Python's standard
library. This day does **not** declare a generated record valid or enable
the CHOLMOD pilot. `--self-test` and any structurally valid `--input` still
exit 1 with `implementation_pending` until the Day 5-8 semantic and byte
checks have negative-fixture coverage.

The accepted Day 7 planning blueprint is a positive **shape** fixture via
`validate_shape()`. It is still `record_kind=blueprint`,
`evidence_origin=planning`, `evidence_status=planned`, not an observed
correctness result. The historical Day 10 attempt remains a blocked
pre-solve attempt.

## Executable Checks

`read_json()` requires UTF-8 text and an object root. It retains object
pairs during decoding so a duplicate key anywhere in a nested object or
array receives its JSON pointer, rather than being overwritten by the
decoder. It rejects `NaN`, `Infinity`, `-Infinity`, exponent overflow to
infinity, malformed JSON, invalid UTF-8 and escaped unpaired surrogates.
Diagnostics report a stable code and pointer; the CLI also reports
`record_id` when parsing made it available.

`validate_shape()` interprets only the schema keywords present in this v1
authority: local `$defs` references, `oneOf`, exact `type`, `const`, `enum`,
required/properties/additionalProperties, bounded array/string/object
lengths, numeric minimum and regular-expression syntax. No value coercion
occurs: `true` is not an integer, and `4.0` is not the pinned integer `4`.
Missing and extra fields are found recursively, not just at the record
root. Pinned selection controls are checked before any status promotion:
selection ID, matrix path/hash/shape, operation, RHS/oracle, precision,
project backend/ordering/API and CHOLMOD version/source/API/mode controls.
The blueprint's abbreviated matrix-origin commit is retained only as a
planning shape; Day 5 must require the full commit for generated runs.

## Focused Mutation Results

The `StructuralTests` cases cover a positive planning blueprint, root and
nested missing fields, root and nested unknown fields, empty IDs, boolean
and float-valued dimensions, unsupported schema/status/side enums, changed
matrix hash, operation, RHS and backend, wrong timing unit, malformed hash
and short Git syntax, unsafe relative artifact path, nested duplicate keys,
nonfinite numbers and invalid UTF-8/Unicode. Each negative case asserts
the rule code and JSON pointer. A CLI mutation also verifies the offending
record ID in the diagnostic.

The code mapping at this stage is structural: `invalid_json`,
`duplicate_or_unknown_field`, `missing_required_field`, `exact_type`,
`unsupported_enum`, `selection_mismatch`, `missing_provenance`,
`unit_or_sample_contract`, and `nonfinite_or_missing_value_abuse`.
Day 5-8 must still enforce status/null relationships, complete cell-ID
recomputation, observed provenance, actual matrix/methodology/artifact
hashes, numerical thresholds, timing sample statistics and publication
claim boundaries. No structural pass is a solver pass.

## Pilot Gate

The existing `run_cholmod_pilot.py` calls `--self-test` before building or
running the helper. That command still fails with `implementation_pending`,
so an optional local invocation stops at `schema_validation_failed` before
solver execution. The selected manifests, normalizer, canonical benchmark
and public documentation claims remain unchanged.
