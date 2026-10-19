# Sprint 215 Day 1: External Baseline Evidence Intake

## Purpose And Scope

Inventory what the current selected comparison, corpus and benchmark surfaces
actually establish before selecting a third-party solver or pilot. This is an
intake record, not a baseline-selection or state-of-the-art decision. Item
owners and provisional checks are tracked in `../WORKING_NOTES.md`.

## Existing Evidence

| Source | Observed contract | Does not establish |
| --- | --- | --- |
| `../../SPRINT_212/artifacts/day5-product-policy-decision.md` | Selected canonical `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1` has threshold-free freshness, `status=measurement`, `baseline=n/a`, `threshold=n/a`, no warmup and no computed variance. | Hosted timing threshold, platform timing parity or speed superiority. |
| `benchmarks/README.md` and `scripts/bench_canonical_report.sh` | Canonical CSV bundle at `build/bench-reports/canonical/`, with `manifest.txt` and `index.tsv`; generated metadata carries command, git/platform/compiler/build context, support tier and methodology fields. | Third-party solver identity, correctness parity or portable performance. |
| `benchmarks/bench_refactor_csc.c` | Internal analyze/refactor/solve timings, selected path and `res_public`/`res_csc` residuals for its bounded workflow. | Matched external factorization or solve phases; `speedup_refactor` is internal-path ratio only. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Six selected comparison targets plus `SRT-BENCH-REFACTOR-CSC-NOS4`; Linux/macOS workflow metadata is attached to selected rows. | A new Day 1 hosted run or an external-library baseline. |
| `tests/corpus/manifests/report_families.tsv` and `tests/corpus/schemas/report_index_fields.md` | Comparison rows are generated-local, fixture-level dense-helper comparisons; canonical benchmark family is generated-local advisory. Selected freshness validates identity and generated artifacts. | Broad solver-family parity or state-of-the-art comparison. |
| `scripts/run_external_comparison.py` | Emits `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv` per target under `build/comparison/*/`. Baseline type is `external-process-source-controlled-helper`; its version string is a helper filename. | An independently released NumPy, SciPy, LAPACK, SuiteSparse or other third-party solver/version. |
| `tests/corpus/README.md` and manifest schemas | Source-controlled fixture/generator/optional-data metadata defines local claims, provenance fields and skip/defer semantics. | Availability of optional external corpus data or a numerical pass from metadata alone. |

The comparison `study.tsv` already has baseline name/type/version/command,
fixture and operation, metric, project/baseline values, delta, tolerance,
status, support tier, claim scope, non-claims, compiler/platform and commit.
Those fields are reusable vocabulary, but they are not yet a complete
third-party package provenance or performance methodology contract.

## Fixture And Operation Inventory

| Current slice | Matrix/provenance | Operation and current evidence limit |
| --- | --- | --- |
| Selected QR comparisons | Small source-controlled generator or inline matrices: `qr_underdetermined_minnorm_2x4`, `qr_overdetermined_compatible_5x3`, `qr_overdetermined_incompatible_4x2`. | Minimum-norm and compatible/incompatible least-squares solve; residual, solution norm/values and helper delta for named fixtures only. |
| Selected partial SVD | Inline diagonal `partial_svd_diag6_k2` comparison. | Top-k singular values, residual, orthogonality and projector diagnostics against source-controlled dense helper; not raw vector identity. |
| Selected LU | Inline `lu_nonsym_square_5` comparison. | One nonsymmetric square solve with status, residual and solution diagnostics; not sparse-direct or pivoting superiority. |
| Selected Cholesky | Inline `cholesky_spd_tridiag_5` comparison. | One SPD tridiagonal solve with status, residual and solution diagnostics; not general SPD or reordering parity. |
| Selected canonical benchmark | Checked-in Matrix Market `tests/data/suitesparse/nos4.mtx` identifies `HB/nos4` in its header; benchmark guide identifies emitted `n=100`. | SPD/Cholesky internal refactor and solve measurement for `--repeat 1`; not a third-party solver comparison. Do not interpret `n=100` as `nnz` or corpus breadth. |
| Broader corpus inputs | `tests/corpus/manifests/fixtures.tsv` has local fixture shape, property, RHS, generator and claim metadata; `tests/data/suitesparse/` has additional checked-in `.mtx` files. `optional_data.tsv` names a disabled-by-default SuiteSparse rank-deficient QR subset with terms requiring review. | No Day 1 matrix subset chosen; file origin/header is not a completed license or redistribution determination. |

Current selected comparison fixtures are defined by the comparison harness;
not every harness key is a row in `fixtures.tsv`. A future pilot must bind its
matrix bytes/hash, RHS, operation and solver semantics explicitly rather than
assuming the selected target name is sufficient provenance.

## Report Paths And Lane Separation

| Lane | Command / artifact | Evidence boundary |
| --- | --- | --- |
| Local comparison | `make report-index-comparison-freshness` generates six `build/comparison/*/study.tsv` targets and runs `scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`. | Fresh generated local helper rows for named fixtures. |
| Hosted selected comparison | `.github/workflows/ci.yml` and `macos-ci.yml` own selected comparison freshness artifacts; Windows has a bounded Cholesky path, while selected manifest platform scope remains Linux/macOS. | Workflow definitions are not observed hosted outcomes; no broad Windows promotion. |
| Local canonical benchmark | `make bench-canonical-report` writes four CSVs plus `index.tsv` and `manifest.txt`; `make bench-canonical-report-freshness` checks the selected `bench_refactor_csc` row. | Threshold-free internal measurement and metadata freshness. |
| Hosted selected performance | Linux `sprint168-selected-performance-freshness` and macOS `sprint202-macos-selected-performance-freshness` review the same selected row with hosted metadata. | No cross-runner speed ratio, hosted threshold or external baseline. |
| Local sentinel | `make performance-sentinels` retains its separately justified local gate and contextual rows. | Sentinel thresholds must not be copied into canonical or external-baseline policy. |

## Unknowns To Resolve

1. Which exact external project/release, package source, license, build
   options, backend/BLAS, solver entry point and supported platform will be
   compared? No candidate is selected on Day 1.
2. Which bounded matrix bytes, hashes, source and redistribution terms are
   acceptable, and is optional data actually available?
3. Are project and external operations equivalent in precision, ordering,
   pivoting, preconditioning, stopping criteria, reuse and timed phase?
4. Which correctness metrics/tolerances and failure dispositions must be
   fixed before measurement? Existing `study.tsv` tolerances cover the current
   helper fixtures only.
5. Which warmup, repeats, variance, CPU/compiler/package provenance, raw
   artifact and retention rules would make an external timing interpretable?
6. Should third-party evidence extend an existing selected report or use a
   separate schema so helper comparisons and benchmark freshness keep their
   present meaning?

## Claim Boundary And Handoff

The current repository supports narrow, source-controlled dense-helper
comparison freshness and one threshold-free canonical benchmark freshness
row. It does **not** support a claim of third-party solver parity, broad
SuiteSparse coverage, portable or cross-platform performance, external speed
superiority, or state-of-the-art status. A future local pilot, if feasible,
would remain a named comparison cell until methodology and actual evidence
justify a more specific statement.

Day 2 should survey pinned external candidates and a matrix shortlist with
license/source/hash evidence, then mark each feasible, excluded or blocked.
Day 3 should select the exact cell or record a concrete blocker. No benchmark
or comparison command was executed for this Day 1 documentation-only intake.
`make docs-check` passed locally, including API documentation coverage for 18
checked-in public headers. No `.c` or `.h` file changed, so the full C
quality chain was not required.
