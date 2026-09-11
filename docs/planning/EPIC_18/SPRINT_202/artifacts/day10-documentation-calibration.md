# Sprint 202 Day 10: Documentation Calibration

## Summary

Day 10 calibrated public and maintainer documentation for the Sprint 202
macOS hosted selected benchmark freshness lane. The documentation now describes
Linux and macOS hosted selected freshness for the same manifest-selected
`bench_refactor_csc` row while preserving threshold-free, non-portable
performance wording.

## Updated Surfaces

| Surface | Calibration |
| --- | --- |
| `README.md` | Top-level benchmark guidance now says the selected performance freshness row is mirrored by reviewed Linux and macOS hosted CI lanes. |
| `INSTALL.md` | Support matrix row now reads `Linux/macOS selected performance freshness` and keeps package-manager, timing-threshold, platform-parity, and state-of-the-art non-claims. |
| `benchmarks/README.md` | Benchmark interpretation now names Linux and macOS hosted selected-performance lanes, their runner contexts, and the retained lack of Linux/macOS performance parity. |
| `docs/maintainer_guide.md` | Maintainer guidance now lists both selected hosted artifacts and both hosted runner metadata sources. |
| `tests/corpus/README.md` | Corpus selected-target interpretation now records Linux/macOS hosted metadata scope and excludes comparable timing claims. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index field guidance now states that selected benchmark hosted workflow metadata covers only Linux/macOS for the exact selected row. |
| `tests/test_selected_performance_docs.py` | Documentation guard now enforces the Linux/macOS selected-performance markers, macOS artifact, macOS runner context, and INSTALL support-matrix wording. |

## Claim Boundary

The calibrated wording supports only hosted selected benchmark freshness for
the existing `SRT-BENCH-REFACTOR-CSC-NOS4` row:

- artifact: `bench_refactor_csc.csv`;
- workload: `tests/data/suitesparse/nos4.mtx --repeat 1`;
- status: `measurement`;
- support tier: `hosted_selected` in hosted mode;
- claim boundary: `hosted_selected_threshold_free`;
- baseline: `n/a`;
- threshold: `n/a`;
- warmup: `none_configured`;
- variance: `not_computed_single_sample`.

The docs continue to reject:

- portable performance;
- timing thresholds;
- Linux/macOS performance parity;
- Windows selected benchmark freshness;
- broad benchmark-family publication;
- package-manager distribution or bottles;
- package/ABI proof;
- release readiness;
- external-library or backend superiority;
- state-of-the-art sparse linear algebra status.

## Sprint 192 Terminology Cross-Check

The wording remains aligned with Sprint 192 methodology terminology:

- selected hosted lane;
- threshold-free methodology evidence;
- `hosted_selected`;
- `hosted_selected_threshold_free`;
- `not_portable_performance_claim`;
- generated reports under ignored `build/` paths;
- selected row freshness rather than timing pass/fail proof.

## Validation

The updated documentation is guarded by:

```sh
python3 tests/test_selected_performance_docs.py
```

Day 11 will rerun the broader focused freshness and workflow guard set after
this documentation calibration.
