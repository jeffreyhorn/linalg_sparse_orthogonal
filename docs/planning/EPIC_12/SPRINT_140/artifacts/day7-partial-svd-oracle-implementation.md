# Day 7 Partial-SVD Oracle Implementation

## Summary

Day 7 implements an opt-in partial-SVD oracle path for the Sprint 140 generated
fixture. The default QR oracle command remains backward-compatible; callers
must pass `--include-partial-svd` to append the partial-SVD rows.

The Day 7 rows are generated-reference oracle rows. They prove the corpus row
semantics, parser behavior, report metadata, and claim boundaries for the
selected fixture. They are not solver-backed pass evidence yet; Days 8-9 own
the focused compiled proof path.

## Implemented Files

| File | Change |
| --- | --- |
| `scripts/run_corpus_oracle.py` | Added fixture-specific expected-row loading, generic comparison parsing, opt-in partial-SVD generated-reference rows, combined output naming, and manifest counts. |
| `docs/planning/EPIC_12/SPRINT_140/WORKING_NOTES.md` | Recorded Day 7 implementation notes and validation expectations. |

## Command Behavior

| Command | Oracle output | Meaning |
| --- | --- | --- |
| `python3 scripts/run_corpus_oracle.py` | `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` | Existing QR generated-reference lane only. |
| `python3 scripts/run_corpus_oracle.py --include-partial-svd` | `build/corpus/oracle/corpus.oracle.tsv` | Existing QR generated-reference rows plus the Sprint 140 partial-SVD generated-reference rows. |
| `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | `build/corpus/oracle/corpus.oracle.tsv` | Existing QR generated-reference rows, opt-in solver-backed QR rows, and partial-SVD generated-reference rows. |

Generated oracle, report, skip, and manifest files remain under ignored
`build/` paths.

## Partial-SVD Rows

The opt-in partial-SVD path emits these rows with
`solver_family=partial_svd` and `support_tier=local_only`:

| Row family | Observed result |
| --- | --- |
| Singular values | `top_k=10,10,9.999999;max_abs_error=0` |
| Left subspace | `left_projector_distance=0` |
| Right subspace | `right_projector_distance=0` |
| Vector residual | `max_triplet_residual=0` |
| Orthogonality | `max_orthogonality_residual=0` |
| Default status | `SPARSE_SUCCESS` |
| Tight-budget status | `SPARSE_ERR_NOT_CONVERGED` |
| Tight-budget diagnostic | `no_partial_sigma_u_vt_on_failure` |

The configuration records `proof_owner=generated_partial_svd_reference` and
`solver_execution=none` so the generated rows cannot be confused with a
compiled solver proof.

## Comparison Parser Coverage

The shared comparison function now supports:

- `rank` and `nullity` with exact integer comparison;
- `value` with `top_k` vector parsing, descending sort, and max absolute error;
- `subspace_distance` with projector tolerance;
- `residual_norm` with scalar or key/value observations;
- `status` with exact status-token comparison;
- `diagnostic` with exact diagnostic-token comparison.

Malformed vectors, missing keys, non-finite metrics, wrong vector lengths, and
reported `max_abs_error` mismatches raise validation errors instead of becoming
pass evidence.

## Report And Manifest Metadata

The manifest now records:

- total oracle row count;
- solver families present;
- solver-backed QR row count;
- partial-SVD row count;
- fixture keys covered by the generated output;
- shared fixture-local claim boundary and non-claims.

## Claim Boundary

Passing Day 7 partial-SVD rows support only generated-reference interpretation
for `partial_svd_clustered_repeated_diag8x6_k3_v1`. They do not support broad
partial-SVD correctness, raw singular-vector identity, broad repeated-spectrum
coverage, external-library parity, performance claims, or partial-result
guarantees.

## Validation Commands

Required Day 7 validation:

```sh
python3 scripts/validate_corpus_schema.py
python3 -m py_compile scripts/validate_corpus_schema.py scripts/run_corpus_oracle.py
python3 scripts/run_corpus_oracle.py
python3 scripts/run_corpus_oracle.py --include-partial-svd
```

Generated files under `build/` should remain uncommitted.

## Day 8 Handoff

Day 8 should design the compiled proof owner for the same fixture. The proof
owner should reuse these row IDs, observed-result formats, and non-claim
boundaries while replacing `solver_execution=none` with actual solver-backed
measurements.
