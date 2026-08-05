# Sprint 138 Day 10 - Maintained Oracle/Report Command

## Purpose

Day 10 adds the first maintained corpus/oracle command. The command validates
the deterministic first corpus lane, emits observed oracle rows under ignored
`build/` paths, and writes a report index that can feed Sprint 141
normalization.

This implementation does not change solver source code, add public claims,
commit generated outputs, add optional external data, or claim broad QR
correctness.

## Maintained Command

Run:

```sh
python3 scripts/run_corpus_oracle.py
```

Default outputs:

| Path | Meaning | Source-control policy |
| --- | --- | --- |
| `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` | Observed oracle rows for the first deterministic fixture lane. | Generated; do not commit. |
| `build/corpus-reports/index.tsv` | Report-index rows for corpus oracle comparisons. | Generated; do not commit. |
| `build/corpus-reports/manifest.txt` | Human-readable local run manifest. | Generated; do not commit. |

## Emitted Oracle Rows

| Oracle row ID | Comparison | Expected | Observed design | Status |
| --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `rank` | `3` | Deterministic reference rank `3`. | `pass` when equal. |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `nullity` | `1` | Deterministic reference nullity `1`. | `pass` when equal. |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `subspace_distance` | `projector_distance<=1e-10` | Reference null-vector residual normalized by vector norm. | `pass` when below tolerance. |

The command validates the corpus/reference lane. Sprint 139 still owns QR
solver behavior closure and any stronger QR implementation claim.

## Report-Index Compatibility

`build/corpus-reports/index.tsv` includes the Sprint 141-compatible fields:

- `report_row_id`;
- `report_family`;
- `row_kind`;
- `row_subject`;
- `artifact_path`;
- `generator_command`;
- `source_commit`;
- `source_branch`;
- `generated_at_utc`;
- `platform`;
- `compiler`;
- `configuration`;
- `support_tier`;
- `status`;
- `status_reason`;
- `row_meaning`;
- `claim_scope`;
- `non_claims`;
- `freshness_status`;
- `freshness_reason`.

Rows remain `local_only` until a reviewed platform lane promotes the evidence.

## Source Metadata Updates

| File | Update |
| --- | --- |
| `tests/corpus/manifests/fixtures.tsv` | Replaced the Day 10 validation-command placeholder with `python3 scripts/run_corpus_oracle.py` and set `introduced_in` to `Sprint 138 Day 9`. |
| `tests/corpus/manifests/generators.tsv` | Replaced the regeneration-command placeholder with `python3 scripts/run_corpus_oracle.py`. |
| `tests/corpus/expected/qr_rank_deficient_6x4_nullspace_v1.tsv` | Marked the projector/subspace expected row as `ready_for_oracle`. |
| `tests/corpus/README.md` | Added command usage and generated-output boundary notes. |
| `tests/corpus/schemas/oracle_fields.md` | Added command usage and updated the first-lane projector expected result. |

## Validation Evidence

Day 10 validation used:

```sh
python3 -B scripts/validate_corpus_schema.py
python3 -B scripts/run_corpus_oracle.py
```

Additional checks covered Python syntax, TSV row widths, trailing whitespace,
focused Markdown links, generated-output source-control boundaries, and absence
of `.c`/`.h` changes.

## Claim Boundaries

The emitted rows are fixture-local corpus/oracle evidence. They do not claim:

- raw QR basis parity;
- broad QR correctness;
- global minimum-norm behavior;
- SuiteSparse or external-library parity;
- broad corpus completeness;
- SVD correctness;
- release readiness;
- package, platform, performance, coverage, or state-of-the-art status.

## Day 10 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The command validates the first corpus lane. | Complete | `scripts/run_corpus_oracle.py` validates corpus schema, regenerates first-lane reference metadata, compares rank/nullity/projector rows, and emits observed oracle rows. |
| Emitted rows include required provenance and interpretation fields. | Complete | Oracle rows include command, commit, branch, timestamp, platform, configuration, support tier, expected/observed values, tolerance, status, claim scope, and non-claims. |
| Report rows do not imply release, performance, or broad correctness proof. | Complete | Report rows carry `local_only` support tier, fixture-local row meaning, explicit non-claims, and freshness metadata. |
