# Sprint 152 Day 12 Integrated Regeneration Validation

## Purpose

Day 12 proves that the selected generated report freshness policy can be run as
one integrated local workflow: regenerate the selected oracle reports, normalize
the selected report families, check required and strict freshness modes, verify
advisory/deferred families, and run focused tests.

## Regenerated Outputs

Command:

```sh
make report-index-oracle-freshness
```

Result: passed.

Generated local outputs:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`
- `build/report-index/normalized-index.tsv`

These files remain ignored build artifacts and are not source-controlled release
proof.

## Selected Oracle Evidence

`build/corpus/oracle/corpus.oracle.tsv` contained `52` selected oracle rows:

- `23` QR solver-backed rows
- `26` partial-SVD solver-backed rows
- `3` generated-reference rows with `unknown` solver family

`build/corpus-reports/manifest.txt` recorded:

- `oracle_row_count=52`
- `solver_families=partial_svd,qr,unknown`
- `solver_qr_row_count=23`
- `partial_svd_row_count=26`
- `command=scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd`
- the selected QR and partial-SVD fixture-key set

The normalized corpus/oracle index was regenerated with:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle --output build/report-index/normalized-index.tsv
```

Result: passed with `128` normalized rows:

- `74` corpus rows
- `54` oracle rows

The corresponding check passed:

```sh
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
```

## Freshness Modes

Required selected oracle freshness passed:

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness
```

Result: exit `0`, `0` freshness errors, and `52` expected
`generated_present_unchecked` row-level warnings.

Strict selected oracle freshness passed:

```sh
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --strict-generated --check-freshness
```

Result: exit `0`, `0` freshness errors, and `52` expected
`generated_present_unchecked` row-level warnings. The selected aggregate policy
passed because row counts, solver families, fixture keys, command metadata, and
source commit metadata matched the current local output.

Advisory and source-controlled package, coverage, and dead-code freshness
passed:

```sh
python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --check-freshness
```

Result: exit `0` with `10` advisory/source-controlled rows. Missing local
coverage and dead-code generated artifacts remain advisory and are not counted
as claim-bearing generated freshness proof.

Runtime-backend freshness passed:

```sh
python3 scripts/normalize_report_index.py --family runtime_backend --check-freshness
```

Result: exit `0` with the runtime-backend governance row classified as
source-controlled advisory evidence.

## Focused Tests

Commands:

```sh
python3 scripts/validate_corpus_schema.py
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
```

Results:

- corpus schema validation passed
- Python compile check passed
- `test-normalize-report-index: ok`

## Documentation And Whitespace Checks

Searched active documentation and report-family metadata for stale selected
oracle command wording:

```sh
rg -n "run_corpus_oracle.py --include-solver-qr|run_corpus_oracle.py --include-partial-svd|require-generated oracle|report-index-oracle-freshness|QR-only|partial-SVD-only" \
  README.md docs/maintainer_guide.md docs/solver_selection.md docs/algorithm.md \
  tests/corpus/schemas/report_index_fields.md tests/corpus/manifests/report_families.tsv
```

Remaining QR-only and partial-SVD-only command references are intentionally
documented as focused debugging variants. Selected freshness wording points at
`make report-index-oracle-freshness`.

Whitespace validation passed:

```sh
git diff --check
```

## Non-Claims

This validation does not claim hosted CI oracle proof, release artifact proof,
package-manager availability, shared-library ABI support, dynamic-loader
support, broad platform support, compiler portability, broad QR correctness,
broad partial-SVD correctness, external-library parity, portable performance,
benchmark superiority, complete coverage, zero dead code, or state-of-the-art
sparse linear algebra status.
