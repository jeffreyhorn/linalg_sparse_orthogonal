# Day 11 Documentation Alignment

Day 11 aligned the public and maintainer documentation with the selected
partial-SVD comparison family promoted earlier in Sprint 161.

## Updated Documentation

| File | Alignment |
| --- | --- |
| `README.md` | Updated quick validation and hosted freshness wording from selected QR-only comparison freshness to selected QR plus partial-SVD comparison freshness. |
| `docs/maintainer_guide.md` | Documented the selected comparison freshness gate as three families: `qr-minnorm`, `qr-compatible-ls`, and `partial-svd-diag6-k2`. |
| `docs/solver_selection.md` | Added the selected `partial_svd_diag6_k2` comparison boundary to SVD workflow guidance without broadening solver claims. |
| `tests/corpus/README.md` | Expanded the selected comparison freshness section with the partial-SVD target, row meanings, artifact path, and non-claims. |
| `tests/corpus/schemas/report_index_fields.md` | Added the selected comparison freshness gate contract, expected generated row counts, and failure modes. |

## Selected Partial-SVD Comparison Meaning

The selected `partial_svd_diag6_k2` comparison rows are local generated
evidence for one diagonal top-k fixture. They compare project-side partial-SVD
output against the source-controlled dense SVD reference helper and preserve
the following generated row meanings:

- `project_status`
- `baseline_status`
- `singular_value_0`
- `singular_value_1`
- `singular_values_max_abs_delta`
- `residual_norm`
- `u_orthogonality`
- `v_orthogonality`
- `u_projector_diag`
- `v_projector_diag`

## Non-Claims Preserved

The docs explicitly keep the selected comparison evidence out of broad
partial-SVD correctness, raw singular-vector identity, vector sign/orientation
identity, repeated-spectrum ordering, external-library parity, hosted/release
proof, platform proof, package proof, ABI proof, performance proof, and
state-of-the-art claims.

## Validation

Day 11 validation should confirm the documentation references match the
freshness gate behavior:

- `make report-index-comparison-freshness`
- `python3 scripts/validate_corpus_schema.py`
- `git diff --check`
