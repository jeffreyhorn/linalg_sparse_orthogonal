# Sprint 150 Day 7: Oracle Data Implementation

## Purpose

Implement the executable QR oracle data path for the Sprint 150 selected
fixtures, normalize expected values into machine-comparable encodings, and
validate the generated local oracle/report rows.

## Implemented Oracle Changes

Updated `scripts/run_corpus_oracle.py` to extend `--include-solver-qr` beyond
the original Sprint 139 QR fixture.

Implemented fixture tables for:

- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Implemented generalized temporary C probes for:

- rank-deficient rectangular QR rank/nullity;
- solver-produced nullspace normalized residual;
- projector-distance comparison against deterministic reference null vectors;
- underdetermined minimum-norm solve status;
- minimum-norm residual, solution norm, and solution vector observations.

Extended QR oracle comparison support for:

- scalar `solution_norm=<value>` rows;
- vector `solution_values=<comma-vector>` rows with `max_abs_error`;
- existing partial-SVD `top_k` value rows without changing their semantics.

## Expected-Data Normalization

Normalized minimum-norm expected rows from free-form values to executable
key/value encodings:

| Fixture | Row | Expected Encoding |
| --- | --- | --- |
| `qr_underdetermined_minnorm_2x4` | solution norm | `solution_norm=1.0` |
| `qr_underdetermined_minnorm_2x4` | solution values | `solution_values=0.5,0.5,0.5,0.5` |
| `qr_minnorm_3x6_exact_values` | solution norm | `solution_norm=2.8982753492378879` |
| `qr_minnorm_3x6_exact_values` | solution values | `solution_values=1.2,1.2,1.0,0.6,0.4,2.0` |
| `qr_minnorm_5x10_exact_values` | solution norm | `solution_norm=3.3166247903553998` |
| `qr_minnorm_5x10_exact_values` | solution values | `solution_values=0.4,0.8,1.2,1.6,2.0,0.2,0.4,0.6,0.8,1.0` |

## Generated Oracle Validation

Command run:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

Result:

- wrote `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv`;
- wrote `build/corpus-reports/index.tsv`;
- wrote `build/corpus-reports/skips.tsv`;
- wrote `build/corpus-reports/manifest.txt`;
- generated `26` oracle rows;
- `26` rows passed;
- generated `23` solver-backed QR rows;
- generated rows covered the existing 6x4 QR seed plus the five Sprint 150 QR
  fixtures.

Local generated rows remain ignored build artifacts. They are evidence for the
local command, commit, branch, platform, compiler, configuration, support tier,
claim scope, and non-claims recorded in the generated rows.

## Report-Index Validation

Commands run:

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

Results:

- oracle freshness check passed with advisory generated-present rows;
- corpus/oracle normalized report-index check passed with `78` rows.

## Claim Boundary

The generated solver-backed rows support only fixture-local QR claims for the
named rows and command output. They do not support:

- broad QR correctness;
- raw QR or nullspace basis identity;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad minimum-norm or least-squares behavior;
- rank-deficient minimum-norm recovery;
- inconsistent-system behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art claims.

## Day 8 Handoff

Day 8 should design focused proof-owner tests that exercise the same row
semantics in `tests/test_qr_corpus.c` or a similarly focused QR corpus test
owner:

1. load/build the selected fixtures through existing helpers or new focused
   helpers;
2. assert rank/nullity and nullspace residual for rank-deficient rectangular
   fixtures;
3. assert projector distance without raw-basis identity;
4. assert minimum-norm status, residual, solution norm, and exact values for
   deterministic full-row-rank underdetermined fixtures;
5. keep diagnostics fixture-key oriented so failures map back to expected rows.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Expected data is deterministic and source-controlled as needed. | Complete | Minimum-norm expected rows now use deterministic key/value encodings; rank-deficient rows already used deterministic encodings. |
| Generation commands are reproducible. | Complete | `python3 scripts/run_corpus_oracle.py --include-solver-qr` generated local oracle/report rows successfully. |
| Expected data matches the selected oracle semantics. | Complete | The solver-backed oracle emitted 26 rows with 26 pass statuses for the selected QR fixtures. |
