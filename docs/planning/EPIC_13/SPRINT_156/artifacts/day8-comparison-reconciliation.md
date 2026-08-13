# Sprint 156 Day 8: Comparison Study Reconciliation

## Purpose

Reconcile the Sprint 154 external comparison harness and first narrow
`qr-minnorm` study against final Epic 13 claim boundaries. The goal is to keep
the comparison evidence tied to one selected fixture, one selected baseline
type, explicit dependency statuses, and explicit residual work.

## Inputs Reviewed

- `docs/planning/EPIC_13/SPRINT_154/artifacts/day3-comparison-target-selection.md`
- `docs/planning/EPIC_13/SPRINT_154/artifacts/day11-report-integration-implementation.md`
- `docs/planning/EPIC_13/SPRINT_154/artifacts/day13-integrated-validation-and-study-publication.md`
- `docs/planning/EPIC_13/SPRINT_154/artifacts/first-narrow-qr-minnorm-comparison-study.md`
- `scripts/run_external_comparison.py`
- `scripts/normalize_report_index.py`
- `tests/qr_external_dense_reference.py`
- `tests/corpus/manifests/report_families.tsv`
- `README.md`
- `docs/maintainer_guide.md`

## Commands Run

| Command | Result | Notes |
| --- | --- | --- |
| `python3 scripts/run_external_comparison.py --self-check` | Passed | Harness self-check completed. |
| `make report-index-comparison-freshness` | Passed | Regenerated the selected local comparison output and passed required comparison freshness. |
| `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | Passed | Required freshness passed with `7` rows. |
| `python3 scripts/normalize_report_index.py --family comparison --output build/report-index/day8-comparison-normalized.tsv` | Passed | Wrote ignored local inspection index with `7` rows. |

The final `make report-index-comparison-freshness` run was executed with
`PYTHONDONTWRITEBYTECODE=1` after clearing transient Python bytecode so the
generated manifest records `worktree_state=clean`. Generated outputs remain
ignored under `build/` and are not committed proof.

## Selected Study

| Field | Value |
| --- | --- |
| Target | `qr-minnorm` |
| Fixture | `qr_underdetermined_minnorm_2x4` |
| Operation | `minnorm_solve` |
| Project lane | `sparse_qr_solve_minnorm` |
| Baseline name | `source-controlled-dense-qr-reference` |
| Baseline type | `external-process-source-controlled-helper` |
| Baseline helper | `tests/qr_external_dense_reference.py` |
| Support tier | `local_only` |
| Source commit in generated manifest | `c00a349b9cab7edd58be79c0f6496c9f1097261b` |
| Source branch in generated manifest | `sprint-156` |
| Worktree state in generated manifest | `clean` |
| Platform in generated manifest | `darwin-x86_64` |
| Project version | `2.2.0` |

## Selected Row Reconciliation

The required comparison freshness gate expects one source-controlled contract
row plus six generated rows for `qr_underdetermined_minnorm_2x4`. Day 8
validated all six generated rows as `pass`:

| Metric | Delta | Tolerance | Status |
| --- | --- | --- | --- |
| `project_status` | status-only | status-only | `pass` |
| `baseline_status` | status-only | status-only | `pass` |
| `residual_norm` | `1.5700924586837752e-16` | `1e-10` absolute | `pass` |
| `solution_norm` | `1.1102230246251565e-16` | `1e-10` absolute | `pass` |
| `solution_values` | `1.1102230246251565e-16` | `1e-10` absolute per component | `pass` |
| `project_vs_baseline_max_abs_delta` | `1.1102230246251565e-16` | `1e-10` absolute | `pass` |

Freshness output reported all six generated rows as fresh because their
`source_commit` matched current `HEAD`; the seventh row is the
source-controlled comparison contract row.

## Dependency And Provenance Status

| Dependency | Status | Required | Interpretation |
| --- | --- | --- | --- |
| `python3` | `pass` | yes | Required interpreter was available for the source-controlled helper. |
| `tests/qr_external_dense_reference.py` | `pass` | yes | Selected helper was available and used as the baseline. |
| `numpy` | `defer` | no | Optional package baseline was not selected; this is not pass evidence. |
| `scipy` | `defer` | no | Optional package baseline was not selected; this is not pass evidence. |

No skip or defer row was counted as comparison proof.

## Claim Boundary

Day 8 supports only this fixture-local statement:

`sparse_qr_solve_minnorm` agrees with the selected source-controlled dense
reference helper for `qr_underdetermined_minnorm_2x4` on project status,
baseline status, residual norm, solution norm, solution values, and maximum
absolute project-vs-baseline solution delta under the recorded command,
commit, branch, platform, compiler, configuration, tolerance policy, and
local-only support tier.

This evidence does not claim:

- broad QR correctness or broad QR parity;
- broad minimum-norm behavior;
- broad rank-deficient solve, nullspace, economy-mode, reorder, or sparse-mode
  behavior;
- raw Q/R basis identity, sign, orientation, pivot-order, or rank-threshold
  policy;
- SVD-pseudoinverse global-oracle behavior;
- NumPy, SciPy, LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, or ecosystem
  parity;
- hosted CI, release, platform portability, package-manager, shared-library,
  loader, ABI, performance, or state-of-the-art proof.

## Wording Audit Notes

The public and maintainer wording reviewed on Day 8 keeps the comparison lane
bounded:

- `README.md` names `make report-index-comparison-freshness` as a narrow local
  comparison freshness gate for only `qr_underdetermined_minnorm_2x4`;
- `docs/maintainer_guide.md` says selected comparison rows are fixture-local
  only and that skip/defer dependency rows are visible non-proof states;
- the Sprint 154 publication artifact explicitly lists broad QR, external
  package, hosted CI, release, platform, package-manager, shared-library, ABI,
  performance, and state-of-the-art non-claims.

No public wording change was needed on Day 8.

## Residual Comparison Queue

| Residual | Owner | Promotion criteria |
| --- | --- | --- |
| QR comparison beyond `qr_underdetermined_minnorm_2x4` | QR and report owners | Add target-selection artifact, expected metrics, tolerances, generated rows, freshness gate expectations, and claim-boundary docs. |
| Optional NumPy baseline | Comparison owner | Define package discovery/version capture, skip semantics, tolerance policy, and non-package-manager wording before turning defer into generated comparison rows. |
| Optional SciPy baseline | Comparison owner | Same as NumPy; absence or deferral must remain non-proof. |
| LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, and ecosystem baselines | Comparison owner | Add one bounded fixture family at a time with dependency provenance, source/reference selection, and explicit non-parity wording. |
| QR raw Q/R, sign/orientation, pivot-order, and rank-threshold comparisons | QR owner | Define basis/sign/order-safe metrics or intentionally avoid raw-basis metrics. |
| Broad rank-deficient, nullspace, economy-mode, reorder, and sparse-mode QR comparisons | QR and reorder owners | Define family-local metrics and tolerances separate from the minimum-norm fixture. |
| Partial-SVD comparison publication | SVD and report owners | Define subspace-safe/repeated-spectrum-safe metrics and selected comparison freshness policy. |
| Portable runtime or performance comparison | Benchmark owner | Use benchmark methodology and sentinel governance; do not reuse correctness comparison rows as timing proof. |
| Hosted CI comparison publication | CI and comparison owners | Promote the selected comparison gate to a reviewed hosted lane and update support tiers. |
| Package-manager, shared-library, loader, and ABI comparison lanes | Package owner | Require separate product decisions and validation contracts before any comparison wording. |

## Completion Criteria Check

- Comparison evidence supports only the selected narrow study.
- Optional dependency deferrals remain visible non-proof states.
- Freshness rows are current for the generated local comparison output.
- Public wording remains bounded and does not make broad ecosystem or
  performance parity claims.
- Future comparison work is staged behind explicit owners and promotion gates.
