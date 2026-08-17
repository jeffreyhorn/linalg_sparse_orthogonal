# Day 10 Report Integration Implementation

## Summary

Day 10 implemented the Day 9 report-integration design for the selected
two-family QR comparison surface.

The selected freshness gate now reports both selected study artifacts in
row-set and non-pass diagnostics, and maintainer/public wording no longer
describes `make report-index-comparison-freshness` as minimum-norm-only.

## Code Changes

| File | Change |
| --- | --- |
| `scripts/normalize_report_index.py` | Added selected comparison artifact constants for both `qr_minnorm` and `qr_compatible_ls`; row-set mismatch and non-pass selected-row diagnostics now name both selected study files. |
| `tests/test_normalize_report_index.py` | Added focused assertions that selected comparison row-set and non-pass failures report both selected study artifacts. |

## Documentation Changes

| File | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Updated selected comparison freshness guidance to describe both selected QR comparison families, both artifact groups, 12 generated rows, and `local_only` claim boundaries. |
| `docs/solver_selection.md` | Updated QR solver-selection wording to describe selected fixture-local minimum-norm and compatible least-squares comparisons. |
| `README.md` | Updated report-freshness command comments and QR evidence wording from minimum-norm-only to selected QR comparison freshness. |
| `tests/corpus/README.md` | Updated hosted freshness wording from QR minimum-norm-only to selected QR comparison freshness. |

## Selected Artifacts

Freshness diagnostics now identify both selected study files:

```text
build/comparison/qr_minnorm/study.tsv
build/comparison/qr_compatible_ls/study.tsv
```

This makes missing, unexpected, duplicate, and non-pass selected-row failures
actionable for the full selected row set.

## Claim Boundary

The selected generated rows remain `local_only` fixture-level evidence:

- `qr_underdetermined_minnorm_2x4` minimum-norm comparison;
- `qr_overdetermined_compatible_5x3` compatible least-squares comparison.

The changes do not add broad QR, external-library, platform, package, ABI,
performance, release, or state-of-the-art claims. Optional NumPy/SciPy rows
remain deferred context only.

## Validation

Commands to run:

```sh
python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --self-check
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
make report-index-comparison-freshness
git diff --check
```

No `.c` or `.h` files changed on Day 10, so the full
`make format && make lint && make test` gate is not required.

## Completion Check

- Selected QR comparison rows appear in normalized reports.
- Selected freshness fails on stale, missing, unexpected, duplicate, or invalid
  selected rows.
- Freshness errors name both selected study artifacts.
- Non-selected and optional/deferred rows remain non-proof context.
- Public and maintainer wording matches the two-family selected comparison
  design.

## Day 11 Handoff

Day 11 should continue documentation alignment by reviewing QR corpus,
maintainer, solver-selection, and public non-claim wording as one surface and
preparing the Sprint 161 partial-SVD comparison handoff.
