# Sprint 150 Day 11: Report Integration Implementation

## Purpose

Implement the Sprint 150 QR report-generation and normalized report-index
follow-through designed on Day 10. Day 11 keeps generated report artifacts
under `build/` and updates only source-controlled generator behavior and sprint
documentation.

## Implementation Summary

Updated `scripts/run_corpus_oracle.py` so each local oracle/report generation
run resets its generated output surface before writing the current run.

The command already generated the selected QR rows, but the report normalizer
reads `build/corpus/oracle/*.tsv`. A stale ignored `corpus.oracle.tsv` from an
older combined QR/partial-SVD run caused duplicate generated-local QR rows and
stale partial-SVD rows to appear in normalization. The Day 11 fix makes the
generator own the local output directory state:

- remove prior `build/corpus/oracle/*.tsv` files before writing the current
  oracle TSV;
- remove prior `build/corpus-reports/index.tsv`;
- remove prior `build/corpus-reports/skips.tsv`;
- remove prior `build/corpus-reports/manifest.txt`;
- then write the current oracle, report, skip, and manifest files.

No generated `build/` files are source-controlled.

## Generated QR Report Surface

The maintained QR command is:

```sh
python3 scripts/run_corpus_oracle.py --include-solver-qr
```

After the cleanup fix, the generated oracle directory contains only the current
QR-only oracle artifact:

```text
build/corpus/oracle/qr_rank_deficient_6x4_nullspace_v1.oracle.tsv
```

The generated manifest reports:

- `oracle_row_count=26`
- `solver_families=qr,unknown`
- `solver_qr_row_count=23`
- `partial_svd_row_count=0`
- `support_tier=local_only`

Selected fixture keys in the generated report:

- `qr_rank_deficient_6x4_nullspace_v1`
- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

## Normalized Report Index

The normalized report-index checks are generated, not hand-authored.

Validation after the cleanup fix:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

Result:

```text
normalize-report-index: 78 rows ok
```

Oracle freshness validation:

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check
```

Result:

```text
normalize-report-index: freshness ok (28 rows)
```

The freshness command still emits the expected advisory
`generated_present_unchecked` warnings for generated-local oracle rows. Those
warnings are acceptable for Sprint 150 because the generated rows remain local
evidence tied to the recorded command, commit, branch, platform, compiler,
configuration, support tier, and artifact path.

## Validation

Day 11 validation commands:

```sh
python3 -m py_compile scripts/run_corpus_oracle.py
python3 scripts/run_corpus_oracle.py --include-solver-qr
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness --check
```

All commands passed.

No `.c` or `.h` files were modified on Day 11, so the full C gate was not
required for Day 11. Day 9 already ran `make format`, `make lint`, and
`make test` after the Sprint 150 C changes.

## Source-Controlled Owners

The Day 11 report integration is owned by:

- `scripts/run_corpus_oracle.py`
- `scripts/normalize_report_index.py`
- `tests/corpus/manifests/report_families.tsv`
- `tests/corpus/manifests/fixtures.tsv`
- `tests/corpus/manifests/generators.tsv`
- `tests/corpus/expected/*.tsv`
- `tests/test_qr_corpus.c`

No source-controlled normalized index output was added.

## Claim Boundary

Day 11 supports fixture-local generated report evidence only. It does not
claim:

- broad QR correctness;
- raw QR basis or raw nullspace basis identity;
- sign, orientation, scale, or column-order parity;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- broad minimum-norm or least-squares behavior;
- SVD-pseudoinverse global-oracle behavior;
- external-library parity;
- platform, package, ABI, performance, or state-of-the-art status.

## Day 12 Handoff

Documentation can now cite:

- the selected six QR fixture keys;
- the `23` solver-backed QR generated-local rows;
- `python3 scripts/run_corpus_oracle.py --include-solver-qr`;
- the normalized corpus/oracle check result of `78` rows;
- the oracle freshness check result of `28` generated-local rows with advisory
  warnings.

Day 12 should update user-facing and maintainer-facing documentation without
turning generated-local rows into release, hosted-platform, package, ABI,
performance, or state-of-the-art claims.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Selected QR families have report/index coverage. | Complete | QR-only generation reports six fixture keys and `23` solver-backed QR rows. |
| Report commands and normalized rows validate locally. | Complete | Oracle generation, corpus/oracle normalization, and oracle freshness checks passed. |
| Report wording remains bounded to source-controlled evidence. | Complete | Generated rows remain `local_only`; stale ignored artifacts are cleared before each run; non-claims remain fixture-local. |
