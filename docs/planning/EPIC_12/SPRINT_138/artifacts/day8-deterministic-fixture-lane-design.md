# Sprint 138 Day 8 - Deterministic Fixture Lane Design

## Purpose

Day 8 defines the exact first deterministic corpus fixture lane so Day 9 can
land generator metadata and expected-result rows without reselecting the
fixture. The lane remains fixture-local QR evidence and does not widen public
claims.

This is a design artifact with targeted metadata cleanup. It does not add a
generator implementation, observed oracle rows, report command, source tests,
optional external data, generated outputs, or public claim updates.

## Selected Fixture Family

| Field | Value |
| --- | --- |
| Fixture family | `qr_rank_deficient` |
| Fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| Generator key | `qr_rank_deficient_6x4_nullspace_generator_v1` |
| Storage kind | `generated` |
| Solver family | `qr` |
| Primary operation | `rank_info` and `nullspace` |
| Shape | Tall rectangular, `rows=6`, `cols=4` |
| Nonzeros | `14` |
| Expected rank | `3` |
| Nullity | `1` |
| Expected null vector | `[-1, -1, 0, 1]` spans the right nullspace. |
| Conditioning class | `moderate` |
| Scale class | `unit` |
| Sparsity class | `structured_sparse` |
| Expected behavior | `success` |
| Support tier before validation | `local_only` |

## Generator Algorithm

The generator is deterministic and non-random. It emits a 6x4 sparse matrix
with columns `c0`, `c1`, `c2`, and `c3`, where `c3 = c0 + c1`.

```text
c0 = [1, 0, 0, 1, 0, 1]^T
c1 = [0, 1, 0, 1, 1, 0]^T
c2 = [0, 0, 1, 0, 1, 1]^T
c3 = [1, 1, 0, 2, 1, 1]^T
```

The first three columns are independent, so the matrix has rank 3. Because
`c3 - c0 - c1 = 0`, the vector `[-1, -1, 0, 1]^T` spans one right-nullspace
direction and the nullity is 1.

## Expected Matrix Entries

Canonical entries are sorted by row, then column. Rows and columns are
zero-based.

| Row | Col | Value |
| --- | --- | --- |
| 0 | 0 | `1.0` |
| 0 | 3 | `1.0` |
| 1 | 1 | `1.0` |
| 1 | 3 | `1.0` |
| 2 | 2 | `1.0` |
| 3 | 0 | `1.0` |
| 3 | 1 | `1.0` |
| 3 | 3 | `2.0` |
| 4 | 1 | `1.0` |
| 4 | 2 | `1.0` |
| 4 | 3 | `1.0` |
| 5 | 0 | `1.0` |
| 5 | 2 | `1.0` |
| 5 | 3 | `1.0` |

## Generator Metadata Design

| Field | Value |
| --- | --- |
| `generator_version` | `1` |
| `algorithm` | `fixed_columns_c3_equals_c0_plus_c1` |
| `seed` | `none` |
| `parameters` | `rows=6;cols=4;expected_rank=3;nullity=1;dependency=c3-c0-c1` |
| `canonical_format` | `coo_zero_based_row_col_value_f64_text_v1` |
| `expected_structure_hash` | `TBD_DAY9` |
| `expected_value_hash` | `TBD_DAY9` |
| `floating_policy` | Exact integer structure/values for generation; exact rank/nullity; projector/subspace distance tolerance `1e-10`. |
| `regeneration_command` | `TBD_DAY10_CORPUS_ORACLE_COMMAND` until the maintained command lands. |
| `change_policy` | Update generator version, fixture metadata, expected results, oracle rows, validation command, and docs together. |

## Canonical Hash Policy

Day 9 should compute hashes from deterministic canonical text, not from
compiler-dependent binary memory.

Canonical text policy:

```text
format coo_zero_based_row_col_value_f64_text_v1
rows 6
cols 4
nnz 14
0 0 1.0000000000000000
0 3 1.0000000000000000
...
5 3 1.0000000000000000
```

Hash policy:

| Hash | Input |
| --- | --- |
| Structure hash | Header plus row/column pairs only. |
| Value hash | Header plus row/column/value triples. |
| Algorithm | SHA-256 over LF-terminated canonical text. |

## Expected Results

| Oracle row ID | Operation | Comparison kind | Expected result | Tolerance |
| --- | --- | --- | --- | --- |
| `qr_rank_deficient_6x4_nullspace_v1_rank` | `rank_info` | `rank` | `3` | `exact=0` |
| `qr_rank_deficient_6x4_nullspace_v1_nullity` | `rank_info` | `nullity` | `1` | `exact=0` |
| `qr_rank_deficient_6x4_nullspace_v1_projector_residual` | `nullspace` | `subspace_distance` | `projector_distance<=1e-10` | `projector=1e-10` |

The projector/subspace row should compare subspace/projector behavior, not raw
QR basis orientation or signs.

## Validation Command Design

The Day 10 maintained command should:

1. Read `tests/corpus/manifests/fixtures.tsv`.
2. Read `tests/corpus/manifests/generators.tsv`.
3. Regenerate `qr_rank_deficient_6x4_nullspace_v1` from the generator row.
4. Verify structure and value hashes.
5. Run QR rank/nullity and nullspace projector/subspace comparisons.
6. Emit observed oracle rows under `build/corpus/oracle/`.
7. Emit a report index under `build/corpus-reports/`.
8. Preserve `local_only` support tier unless a reviewed platform lane runs it.
9. Treat skipped/deferred/unsupported rows as non-pass evidence.

Until that command lands, manifest rows may retain
`TBD_DAY10_CORPUS_ORACLE_COMMAND`.

## Sprint 139 QR Handoff Fields

| Field | Handoff value |
| --- | --- |
| Fixture key | `qr_rank_deficient_6x4_nullspace_v1` |
| Expected rank | `3` |
| Nullity | `1` |
| Null vector reference | `[-1, -1, 0, 1]` up to scale. |
| Primary comparison | Projector or two-way projection distance. |
| Projector tolerance | `1e-10` initial fixture-local tolerance. |
| Non-claims | No raw-basis parity, no broad QR correctness, no global minimum-norm guarantee, no SuiteSparse parity, no broad corpus completeness. |

## Day 9 Implementation Handoff

| Task | Required Day 9 action |
| --- | --- |
| Generator metadata | Replace `TBD_DAY9` hash fields with computed SHA-256 values. |
| Expected result rows | Keep rank/nullity/projector expected rows aligned with this design. |
| Validation helper | Extend `scripts/validate_corpus_schema.py` only if Day 9 adds hash-format or command checks. |
| Generated output boundary | Do not commit `build/corpus/` or `build/corpus-reports/` outputs. |
| Claim boundary | Keep first-lane evidence fixture-local and leave public claims unchanged. |

## Day 8 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The first fixture lane is deterministic and reproducible by design. | Complete | Fixed matrix entries, no seed, canonical text, and SHA-256 hash policy are defined. |
| Fixture metadata can be checked against the manifest schema. | Complete | Manifest fields are mapped to concrete rows, columns, nnz, rank, nullity, classes, generator key, and support tier. |
| The lane supports Sprint 139 without claiming broad corpus coverage. | Complete | QR handoff fields and non-claims preserve fixture-local rank/nullity/subspace evidence only. |
