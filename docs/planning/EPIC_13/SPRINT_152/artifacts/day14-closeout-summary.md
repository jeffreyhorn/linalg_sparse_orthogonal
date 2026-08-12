# Sprint 152 Day 14 Closeout Summary

## Purpose

Day 14 finalizes Sprint 152 generated report freshness publication artifacts,
records final validation, and prepares the Sprint 153 shared-library ABI
product-decision handoff.

## Completed Sprint 152 Outputs

Sprint 152 leaves the selected generated freshness publication surface centered
on one maintained local command:

```sh
make report-index-oracle-freshness
```

The command regenerates the selected combined oracle output and checks required
oracle freshness without promoting ignored local generated artifacts into
release, hosted CI, package, ABI, performance, platform, or state-of-the-art
proof.

Source-controlled changes cover:

- `Makefile` local freshness target
- selected oracle policy logic in `scripts/normalize_report_index.py`
- focused report-index tests in `tests/test_normalize_report_index.py`
- selected oracle report-family metadata in
  `tests/corpus/manifests/report_families.tsv`
- report-index schema documentation in
  `tests/corpus/schemas/report_index_fields.md`
- user and maintainer documentation in `README.md`, `docs/algorithm.md`,
  `docs/solver_selection.md`, and `docs/maintainer_guide.md`
- Sprint 152 plan, working notes, and artifacts

## Final Validation

Commands run:

```sh
make report-index-oracle-freshness
python3 scripts/validate_corpus_schema.py
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 scripts/normalize_report_index.py --family corpus --family oracle --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py --family corpus --family oracle --check
python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --strict-generated --check-freshness
python3 scripts/normalize_report_index.py --family coverage --family deadcode --family package --family runtime_backend --check-freshness
git diff --check
```

All commands passed.

Final selected oracle generated output:

- `52` total rows in `build/corpus/oracle/corpus.oracle.tsv`
- `23` QR solver-backed rows
- `26` partial-SVD solver-backed rows
- `3` generated-reference rows with `unknown` solver family

Final normalized corpus/oracle index:

- `128` total rows
- `74` corpus rows
- `54` oracle rows

Strict selected oracle freshness exited `0` with no freshness errors and `52`
expected row-level `generated_present_unchecked` warnings. Advisory/source-
controlled coverage, dead-code, package, and runtime-backend checks exited `0`
with `11` advisory/source-controlled rows.

## Stale Reference And Generated Output Review

Active documentation and report-family metadata were searched for stale selected
oracle command wording and stale row-count wording. Selected freshness guidance
points at `make report-index-oracle-freshness`; remaining QR-only and
partial-SVD-only command references are intentionally documented as focused
debugging variants that do not satisfy the selected combined row-count policy.

Generated output remains under ignored paths, including:

- `build/corpus/oracle/`
- `build/corpus-reports/`
- `build/report-index/`

Python `__pycache__` output was removed after validation.

## Retrospective Inputs

### Closed

- Selected generated family choice: oracle generated-reference and
  solver-backed rows.
- Selected local command surface: `make report-index-oracle-freshness`.
- Selected row-count policy: `52` total oracle rows with QR, partial-SVD, and
  generated-reference splits.
- Required/strict aggregate policy coverage for missing, stale, failing,
  partial, missing-solver-family, and missing-fixture-key selected oracle rows.
- Documentation alignment for local-only generated freshness and non-claims.

### Intentionally Residual

- Benchmark, sentinel, guardrail, dead-code, and coverage generated families
  remain advisory or later-sprint owned.
- Hosted CI logs remain external evidence and are not local generated
  freshness artifacts.
- Package and runtime-backend governance rows remain source-controlled evidence,
  not generated report freshness proof.
- Generated local oracle output remains ignored and uncommitted.

### Follow-Up Risks

- Future sprints can accidentally cite selected local oracle output as hosted
  or release proof unless non-claims stay visible.
- Strict selected oracle freshness still has row-level
  `generated_present_unchecked` warnings; the aggregate selected policy closes
  Sprint 152, but row-level comparison semantics remain a possible later
  refinement.
- Sprint 153 package/ABI work must avoid treating selected local oracle
  freshness as package, shared-library, loader, or ABI evidence.

## Closeout Checklist

- Sprint 152 working notes updated through Day 14.
- Artifact set complete for Days 1-14.
- Sprint 153 handoff prepared.
- Final lightweight report/schema/freshness checks passed.
- No C or header files changed during Sprint 152.
- Ignored generated report output is not staged.
- Python cache output removed after validation.
- Generated-report evidence boundary is explicit.

## Non-Claims

Sprint 152 closeout does not claim hosted CI oracle proof, release artifact
proof, package-manager availability, shared-library ABI support, dynamic-loader
support, broad platform support, compiler portability, broad QR correctness,
broad partial-SVD correctness, external-library parity, portable performance,
benchmark superiority, complete coverage, zero dead code, or state-of-the-art
sparse linear algebra status.
