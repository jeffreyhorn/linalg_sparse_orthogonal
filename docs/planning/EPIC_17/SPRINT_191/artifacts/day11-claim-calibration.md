# Sprint 191 Day 11: Documentation and Claim Calibration

## Summary

Day 11 updated current documentation to describe the new selected
`qr-incompatible-ls` comparison family without broadening QR, least-squares,
platform, package, performance, or state-of-the-art claims.

## Documentation Updates

| File | Update |
| --- | --- |
| `docs/solver_selection.md` | Added selected QR incompatible least-squares evidence for `qr_overdetermined_incompatible_4x2` and repeated broad non-claims. |
| `README.md` | Updated selected comparison wording to include QR minimum-norm, compatible least-squares, and incompatible least-squares fixtures. |
| `docs/cookbook.md` | Added incompatible least-squares rows to the QR evidence note. |
| `docs/maintainer_guide.md` | Added QR incompatible least-squares to selected comparison interpretation and clarified Windows metadata remains unpromoted for this target. |
| `tests/corpus/README.md` | Added broad least-squares parity to selected comparison non-claims. |
| `tests/corpus/schemas/report_index_fields.md` | Added broad least-squares parity to report-index selected comparison non-claims. |
| `scripts/check_qr_header_docs_guard.sh` | Updated guard checks so future docs validation requires incompatible least-squares wording. |

## Evidence Statement

The selected QR incompatible least-squares evidence is limited to:

- target key `qr-incompatible-ls`;
- subfamily `qr_incompatible_ls`;
- fixture `qr_overdetermined_incompatible_4x2`;
- selected source-controlled dense QR reference helper;
- status, residual norm, solution norm, solution values, and
  project-vs-baseline max absolute delta;
- `1e-10` tolerances;
- local generated rows and reviewed Linux/macOS selected comparison freshness.

## Non-Claims

The updated docs explicitly retain these boundaries:

- no broad QR correctness;
- no broad least-squares parity;
- no raw QR basis identity;
- no Q sign or orientation identity;
- no global rank-threshold policy;
- no broad rank-deficient solve behavior;
- no external-library parity;
- no broad Windows report freshness;
- no package-manager or ABI proof;
- no performance or release proof;
- no state-of-the-art status.

## Validation

| Command | Result |
| --- | --- |
| `bash scripts/check_qr_header_docs_guard.sh` | Pass |
| `python3 scripts/validate_corpus_schema.py` | Pass |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass |
| active-doc stale wording scan | Pass |

No `.c` or `.h` files changed, so the full C quality gate is not required for
Day 11.

## Day 12 Handoff

Day 12 should run integrated local validation for the selected comparison
family, including the full selected comparison freshness gate, target-specific
freshness for `qr-incompatible-ls`, focused runner and normalizer tests, and
manual inspection of generated `summary.md` and `manifest.tsv` fields.
