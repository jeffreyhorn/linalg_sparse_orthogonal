# Sprint 163 Day 12 Cross-Surface Validation And Quality Gate

## Purpose

Day 12 re-runs the affected report, documentation, package-boundary, and
quality checks after the selected Sprint 163 report-script and documentation
updates. This validation matches the changed-file surface: shell scripts,
Python report-index code, public docs, benchmark docs, maintainer docs,
report-index schema notes, and planning artifacts changed; no C or header files
changed.

## Changed-File Gate Decision

| Changed Surface | Files | Required Gate |
| --- | --- | --- |
| Shell report scripts | `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh` | Shell syntax and selected report command execution. |
| Python report-index code | `scripts/normalize_report_index.py` | Focused normalizer regression test and selected benchmark/sentinel normalization. |
| Public and maintainer docs | `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, `tests/corpus/schemas/report_index_fields.md` | Unsupported-claim scan and whitespace checks. |
| Planning docs | `docs/planning/EPIC_14/SPRINT_163/*` | Whitespace checks. |
| C / header files | none | `make format`, `make lint`, and `make test` not required by the Sprint 163 Day 12 gate. |

## Commands Run

```sh
bash -n scripts/bench_canonical_report.sh scripts/performance_sentinels.sh
make bench-canonical-report
make performance-sentinels
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel \
  --output build/report-index/normalized-index.tsv
python3 scripts/validate_corpus_schema.py
bash scripts/static_package_deferral_check.sh
rg -n "state-of-the-art|superiority|package-manager|shared-library|dynamic ABI|runtime-loader|broad platform|portable performance|OpenMP speedup|backend superiority" README.md benchmarks/README.md docs/maintainer_guide.md tests/corpus/schemas/report_index_fields.md
git diff --check
```

## Results

| Check | Result | Notes |
| --- | --- | --- |
| Shell syntax | Pass | Both selected report scripts parse with `bash -n`. |
| Canonical report generation | Pass | `make bench-canonical-report` wrote canonical CSVs, `index.tsv`, and `manifest.txt`. |
| Sentinel report generation | Pass | `make performance-sentinels` wrote `sentinels.tsv`, manifest, wall-check output, and raw S2/S3 CSVs. |
| Normalizer regression | Pass | `python3 tests/test_normalize_report_index.py` reported `test-normalize-report-index: ok`. |
| Benchmark/sentinel normalization | Pass | Normalized benchmark/sentinel index wrote `26` rows. |
| Corpus/report-family schema | Pass | `validate-corpus-schema` reported the corpus directory is ok. |
| Static package deferral guard | Pass | Guard preserved shared-library, dynamic ABI, runtime-loader, package-manager, and Windows package non-claims. |
| Unsupported-claim scan | Pass | Hits are non-claims or boundary wording, not positive claims. |
| Whitespace | Pass | `git diff --check` reported no issues. |

## Generated Row Semantics Check

Focused generated-row inspection confirmed:

- canonical rows:
  - `4` rows;
  - all `status=measurement`;
  - all `claim_boundary=local_threshold_free`;
  - all `baseline=n/a` and `threshold=n/a`;
  - repeat semantics are `configured_repeat_1` or `benchmark_default`.
- sentinel rows:
  - `19` rows total;
  - S5: `3` rows, `status=pass`, `claim_boundary=local_wall_gate`;
  - S2: `8` rows, `status=report`,
    `claim_boundary=local_threshold_free`;
  - S3: `8` rows, `status=report`,
    `claim_boundary=local_threshold_free`;
  - S5/S2/S3 rows all carry repeat semantics, `warmup=not_recorded`,
    `variance=not_recorded`, and methodology notes.

## Hosted-Only Verification Checklist

Sprint 163 local validation does not create hosted proof. A future hosted
publication lane must explicitly provide:

- workflow/job name;
- runner OS and image;
- compiler and version;
- build mode and thread setting;
- selected command;
- generated artifact or log;
- row-state interpretation;
- claim boundary and non-superiority caveats.

Do not cite local generated rows as hosted CI proof, package proof, ABI proof,
runtime-loader proof, broad platform proof, OpenMP speedup evidence, backend
superiority proof, or state-of-the-art evidence.

## Residual Notes

- No validation-driven fixes were required on Day 12.
- Generated report artifacts remain under ignored `build/` paths.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Completion Check

- Validation matches the changed-file surface.
- Required checks passed before closeout.
- Hosted-only expectations are explicit.
