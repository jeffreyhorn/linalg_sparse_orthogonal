# Sprint 154 Day 1 Comparison Boundary

## Purpose

Day 1 establishes the Sprint 154 evidence boundary before selecting a direct
external comparison target. The sprint must produce one narrow
evidence-backed comparison study, not broad ecosystem parity.

## Source Plan

Sprint 154 implements
`docs/planning/EPIC_13/PROJECT_PLAN.md` section
`Sprint 154: External Comparison Harness And First Narrow Study`.

The sprint goal is to build the first direct external comparison harness and
publish one narrow study while preserving non-claims for broad external-library
parity, package-manager support, shared-library ABI, hosted CI, performance,
platform support, and state-of-the-art status.

## Handoff Inputs

| Input | Evidence Available | Day 1 Interpretation |
| --- | --- | --- |
| Sprint 150 QR corpus | Six maintained QR fixtures with source-controlled metadata, expected rows, focused proof owner, and generated-local oracle rows. | Candidate comparison family, fixture-local only. |
| Sprint 151 partial-SVD corpus | Four maintained partial-SVD fixtures with source-controlled metadata, expected rows, focused proof owner, and generated-local oracle rows. | Candidate comparison family, fixture-local only. |
| Sprint 152 report freshness | `make report-index-oracle-freshness` regenerates combined QR plus partial-SVD oracle rows and checks selected local freshness. | Useful local generated evidence policy; not hosted CI, package, ABI, performance, or release proof. |
| Sprint 153 package/ABI decision | Static-first package support remains maintained; shared-library support remains deferred with exact blockers. | External comparison must not infer dynamic linking, dynamic ABI, or package-manager support. |
| Maintainer solver evidence table | Existing bounded external dense-reference lanes and solver-family non-claims are documented in `docs/maintainer_guide.md`. | Wording input and prior-art boundary, not broad parity proof. |

## Current Corpus Inventory

Maintained QR fixtures from `tests/corpus/manifests/fixtures.tsv`:

- `qr_rank_deficient_6x4_nullspace_v1`
- `qr_rankdef_duplicate_5x4_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`

Maintained partial-SVD fixtures from `tests/corpus/manifests/fixtures.tsv`:

- `partial_svd_clustered_repeated_diag8x6_k3_v1`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`
- `partial_svd_fail_closed_diag6_k2_v1`

Expected-result rows are source-controlled under `tests/corpus/expected/`:

| Fixture File | Expected Rows |
| --- | ---: |
| `qr_rank_deficient_6x4_nullspace_v1.tsv` | 3 |
| `qr_rankdef_duplicate_5x4_v1.tsv` | 4 |
| `qr_rankdef_dependent_row_4x3_v1.tsv` | 4 |
| `qr_underdetermined_minnorm_2x4.tsv` | 4 |
| `qr_minnorm_3x6_exact_values.tsv` | 4 |
| `qr_minnorm_5x10_exact_values.tsv` | 4 |
| `partial_svd_clustered_repeated_diag8x6_k3_v1.tsv` | 8 |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1.tsv` | 7 |
| `partial_svd_lowrank_rect5x7_k3_sparse_output_v1.tsv` | 6 |
| `partial_svd_fail_closed_diag6_k2_v1.tsv` | 5 |

The Day 1 source-controlled expected-row baseline is `49` rows.

## Current Report And Freshness Inventory

Report-family policy rows live in
`tests/corpus/manifests/report_families.tsv`.

Relevant existing families:

- `corpus/fixtures`, `corpus/generators`, `corpus/optional_data`, and
  `corpus/expected`: source-controlled metadata and expected-row policy.
- `oracle/generated_reference` and `oracle/solver_backed`: generated-local
  comparison rows under ignored `build/` paths.
- `package/static_install`, `ci/reviewed_lanes`,
  `documentation/report_guidance`, and `runtime_backend/governance`:
  source-controlled interpretation anchors, not generated comparison output.
- `benchmark`, `sentinel`, `guardrail`, `deadcode`, and `coverage`: generated
  or advisory local report families that must not be treated as external
  comparison proof without a separate policy.

Maintained local commands and owners:

- `make report-index-oracle-freshness` regenerates combined QR plus
  partial-SVD local oracle rows and runs the selected oracle freshness gate.
- `python3 scripts/run_corpus_oracle.py --include-solver-qr` is a QR-focused
  debugging variant and does not satisfy the Sprint 152 selected combined
  freshness policy by itself.
- `python3 scripts/run_corpus_oracle.py --include-partial-svd` is a
  partial-SVD-focused debugging variant and does not satisfy the Sprint 152
  selected combined freshness policy by itself.
- `python3 scripts/normalize_report_index.py --check` validates normalized
  report-row construction.
- `python3 scripts/normalize_report_index.py --check-freshness` inspects
  report freshness diagnostics according to family policy.

## Comparison Non-Claim Register

Sprint 154 comparison work must not claim:

- broad QR correctness;
- broad partial-SVD correctness;
- LAPACK, NumPy, SciPy, SuiteSparse, Eigen, CHOLMOD, PETSc, Trilinos, or
  ecosystem parity;
- raw QR basis, sign, orientation, or column-order parity;
- raw singular-vector identity, sign, phase, or basis-order parity;
- broad rank-threshold, nullspace, minimum-norm, repeated-spectrum,
  low-rank-output, drop-tolerance, convergence-rate, or partial-result
  guarantees;
- portable performance superiority;
- hosted CI proof from local generated artifacts;
- package-manager distribution support;
- shared-library packaging or dynamic ABI compatibility;
- Linux `.so`, macOS `.dylib`, Windows DLL/import-library, SONAME,
  install-name, RPATH, or runtime-loader support;
- Windows Makefile parity or Windows `pkg-config` execution parity;
- state-of-the-art sparse linear algebra status.

## Stop Conditions

Stop and redesign if any of these happen:

- The selected comparison target requires an external dependency that cannot be
  discovered, versioned, skipped, or deferred cleanly.
- The harness would treat an optional missing dependency as pass evidence.
- Generated comparison rows cannot identify library, version, command,
  platform, compiler, fixture, metric, tolerance, status, caveat, and artifact
  path or cannot explicitly document a deferred field.
- The first study needs timing or memory measurements to look meaningful; that
  would require a benchmark methodology outside the Day 1 boundary.
- A comparison result would be worded as broad external-library parity rather
  than a fixture-local result.
- Report-index integration would promote local generated output into hosted CI,
  package, ABI, platform, performance, release, or state-of-the-art evidence.
- `.c` or public `.h` files change and the sprint cannot run the required full
  quality gate.

## Day 2 Handoff

Day 2 should audit QR and partial-SVD as candidate first-study targets.

The audit should compare:

- fixture count and expected-row shape;
- external baseline availability and likely local installation path;
- metric clarity and tolerance stability;
- parsing complexity;
- optional dependency behavior;
- report-index integration cost;
- documentation/non-claim risk;
- likely implementation size for a complete first study.

The preferred Day 3 target should be the smallest candidate that can publish a
useful, reproducible, fixture-local comparison without broad parity claims.
