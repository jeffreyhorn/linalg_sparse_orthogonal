# Day 11 Maintainer Guidance and Residual Queue

## Scope

Day 11 records how maintainers regenerate, interpret, and bound the Sprint 139
QR corpus lane after the Day 9 proof owner and Day 10 public wording updates.
The guidance stays fixture-local to `qr_rank_deficient_6x4_nullspace_v1` and
does not broaden QR, external-library, platform, performance, corpus
completeness, or state-of-the-art claims.

## Updated Maintainer Surfaces

- `docs/maintainer_guide.md` now has a Sprint 139 QR corpus maintenance section
  with regeneration commands, expected generated outputs, stale-report signals,
  support-tier interpretation, and the remaining QR residual queue.
- `tests/corpus/README.md` now documents the opt-in solver-backed QR output
  shape, stale or unsupported report signals, support-tier boundary, and
  residuals that remain outside Sprint 139.

## Regeneration Commands

```sh
python3 scripts/validate_corpus_schema.py
make build/test_qr_corpus && ./build/test_qr_corpus
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

The focused C proof is expected to report four passing `test_qr_corpus` tests.
The opt-in oracle path is expected to emit:

- `build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv` with six
  rows: three generated-reference rows and three solver-backed QR rows.
- `build/corpus-reports/index.tsv` with pass rows for the rank, nullity, and
  normalized nullspace residual comparisons.
- `build/corpus-reports/skips.tsv` with optional external-data skip/defer rows.
- `build/corpus-reports/manifest.txt` with command, row count, solver families,
  support tier, and `solver_qr_row_count=3`.

## Stale or Unsupported Report Signals

Maintainers should regenerate or reject interpretation when:

- the report command does not include `--include-solver-qr`;
- the report predates changes to corpus manifests, expected rows, schemas,
  `scripts/run_corpus_oracle.py`, `tests/test_qr_corpus.c`, or
  `tests/test_qr_helpers.h`;
- the manifest commit, branch, compiler, configuration, support tier, command,
  or generated path does not match the report under review;
- the oracle output is missing the three `solver_family=qr` rows, omits `qr`
  from solver families, or reports `solver_qr_row_count` other than `3`;
- any Sprint 139 QR comparison row has a non-pass status;
- optional-data skip/defer rows are cited as QR pass evidence.

## Support Tier and Optional Data

The Sprint 139 QR lane remains `local_only` evidence. Generated oracle and
report files stay ignored local outputs. Optional SuiteSparse or external data
rows are availability/provenance policy evidence only until a later sprint
promotes reviewed hosted evidence.

## Remaining Residual Queue

- Global QR rank-threshold policy across scales and perturbations remains open.
- Broad rank-deficient QR solve, residual-only least-squares, and minimum-norm
  behavior remain open because they require solve-side fixtures and oracle
  semantics.
- COLAMD/reordered QR remains open because ordering and fill behavior have
  separate semantics from the nullspace residual lane.
- SuiteSparse, LAPACK, NumPy, SciPy, platform, performance, corpus
  completeness, and state-of-the-art claims remain open.
- Raw QR basis/sign/orientation parity is intentionally not closed; Sprint 139
  uses residual/subspace-safe evidence.
- Partial-SVD clustered/repeated singular-value and rank-deficient
  range-projector follow-through is owned by Sprint 140.

## Day 11 Validation

Day 11 is a documentation-only update. The planned validation is documentation
hygiene: diff whitespace checks, trailing-whitespace scans, and focused
relative-link validation for the edited documentation and Sprint 139 planning
artifacts. All three checks passed.
