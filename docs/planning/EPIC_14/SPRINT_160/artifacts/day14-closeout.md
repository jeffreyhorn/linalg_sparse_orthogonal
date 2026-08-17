# Day 14 Closeout And Retrospective Prep

## Summary

Day 14 completed Sprint 160 closeout.

The sprint now has one additional bounded QR comparison family,
`qr-compatible-ls`, integrated with the descriptor-backed comparison runner,
source-controlled report-family metadata, selected comparison freshness,
focused tests, documentation, evidence review, and Sprint 161 partial-SVD
handoff.

## Final Deliverables

| Deliverable | Status | Evidence |
| --- | --- | --- |
| New bounded QR comparison family | Complete | `scripts/run_external_comparison.py` target `qr-compatible-ls`. |
| Existing QR minimum-norm comparison preserved | Complete | `tests/test_run_external_comparison.py`; `make report-index-comparison-freshness`. |
| Selected comparison row set expanded to 12 rows | Complete | `scripts/normalize_report_index.py`; `tests/test_normalize_report_index.py`. |
| Source-controlled report metadata | Complete | `tests/corpus/manifests/report_families.tsv` row for `comparison/qr_compatible_ls`. |
| Selected comparison freshness gate | Complete | `Makefile` target `report-index-comparison-freshness` regenerates both targets. |
| Focused runner tests | Complete | `tests/test_run_external_comparison.py`. |
| Normalizer row-state tests | Complete | `tests/test_normalize_report_index.py`. |
| Documentation alignment | Complete | `README.md`, `docs/maintainer_guide.md`, `docs/solver_selection.md`, `tests/corpus/README.md`. |
| Sprint 161 handoff | Complete | `day11-docs-alignment.md`; `day13-evidence-review.md`. |

## Final Validation

Commands run:

```sh
python3 -m py_compile scripts/run_external_comparison.py scripts/normalize_report_index.py tests/test_normalize_report_index.py tests/test_run_external_comparison.py
python3 scripts/run_external_comparison.py --self-check
python3 tests/test_run_external_comparison.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
make report-index-comparison-freshness
git diff --check
```

All commands passed.

Observed comparison freshness summary:

```text
external-comparison: qr-minnorm project-vs-baseline comparison passed
external-comparison: qr-compatible-ls project-vs-baseline comparison passed
normalize-report-index: freshness ok (14 rows)
report-index-comparison-freshness: passed (local-only generated comparison freshness)
```

No `.c` or `.h` files changed in Sprint 160, so the full C quality gate
`make format && make lint && make test` was not required.

## Selected Row Closeout

The selected comparison set is:

| Family | Rows | Artifact |
| --- | ---: | --- |
| `qr-minnorm` / `qr_underdetermined_minnorm_2x4` | 6 | `build/comparison/qr_minnorm/study.tsv` |
| `qr-compatible-ls` / `qr_overdetermined_compatible_5x3` | 6 | `build/comparison/qr_compatible_ls/study.tsv` |

Each selected family contributes:

- `project_status`
- `baseline_status`
- `residual_norm`
- `solution_norm`
- `solution_values`
- `project_vs_baseline_max_abs_delta`

Strict selected comparison freshness now fails on missing, unexpected,
duplicate, stale, non-pass, skip, or defer selected rows.

## Deferred And Non-Proof Rows

Optional NumPy/SciPy dependency rows remain deferred context:

- `status=defer`
- `status_reason=optional_package_baseline_not_selected`
- `required=no`
- `caveat=deferred rows are not pass evidence`

They cannot create selected pass evidence.

## Claim Review

The aligned public and maintainer wording supports only:

- fixture-local QR minimum-norm comparison for
  `qr_underdetermined_minnorm_2x4`;
- fixture-local QR compatible least-squares comparison for
  `qr_overdetermined_compatible_5x3`;
- selected freshness for those rows when
  `make report-index-comparison-freshness` passes.

It does not claim:

- broad QR parity;
- raw QR basis identity;
- Q sign/orientation identity;
- global rank-threshold behavior;
- broad rank-deficient solve behavior;
- NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity;
- broad hosted platform proof;
- package-manager behavior;
- shared-library ABI;
- performance superiority;
- release proof;
- state-of-the-art status.

## Retrospective Inputs

What went well:

- Descriptor-backed target configuration allowed `qr-compatible-ls` to be
  added without weakening `qr-minnorm`.
- Focused CLI tests caught output-path contract details and now protect both
  target families.
- Normalizer row-state tests stayed the owner for missing, unexpected,
  duplicate, stale, fail, and defer selected-row behavior.
- Documentation alignment prevented minimum-norm-only wording from surviving
  after the selected comparison set became two-family.

What to watch:

- Report diagnostics must list every selected artifact when a selected row set
  spans multiple generated files.
- Historical sprint artifacts may mention earlier one-family behavior; current
  public/maintainer docs should be the source for present-tense claims.
- Hosted artifact names or CI summaries may need future alignment if hosted
  lanes start publishing both comparison families separately.

Sprint 161 handoff:

- Start with a low-risk partial-SVD target such as `partial_svd_diag6_k2`.
- Define subspace-safe metric rows before implementation.
- Reuse the Sprint 160 descriptor, metadata, focused-test, normalizer, and
  freshness pattern.
- Avoid raw singular-vector identity, sign/order identity, repeated-spectrum
  overclaims, convergence-rate claims, broad partial-SVD correctness,
  external-library parity, and platform/package/performance/ABI claims.

## Cleanup

Python bytecode cache directories generated by validation were removed from
`scripts/` and `tests/`.
