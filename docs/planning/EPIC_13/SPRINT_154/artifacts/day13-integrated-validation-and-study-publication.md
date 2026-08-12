# Day 13: Integrated Validation And Study Publication

## Scope

Day 13 ran the selected comparison freshness gate, published the first narrow
source-controlled comparison study snapshot, and validated the affected
harness, report-index, schema, documentation, and whitespace surfaces.

## Published Study Artifact

Published:

- `docs/planning/EPIC_13/SPRINT_154/artifacts/first-narrow-qr-minnorm-comparison-study.md`

The published study snapshots the generated local comparison output for
`qr_underdetermined_minnorm_2x4` and preserves:

- target and fixture identity;
- command-level reproducibility;
- source commit and branch;
- worktree state caveat;
- project and baseline values;
- selected row statuses;
- dependency status;
- residual comparative gaps;
- non-claim boundaries.

Regenerate the local source data with:

```sh
make report-index-comparison-freshness
```

## Integrated Validation

Ran:

```sh
make report-index-comparison-freshness
```

Result:

- regenerated `build/comparison/qr_minnorm/project_observations.tsv`;
- regenerated `build/comparison/qr_minnorm/baseline_observations.tsv`;
- regenerated `build/comparison/qr_minnorm/dependency_status.tsv`;
- regenerated `build/comparison/qr_minnorm/study.tsv`;
- regenerated `build/comparison/qr_minnorm/summary.md`;
- regenerated `build/comparison/qr_minnorm/manifest.tsv`;
- required comparison freshness passed with seven normalized rows: one
  source-controlled contract row and six generated selected rows.

Selected generated rows:

| Metric | Delta | Tolerance | Status |
| --- | --- | --- | --- |
| `project_status` | | status-only | `pass` |
| `baseline_status` | | status-only | `pass` |
| `residual_norm` | `1.5700924586837752e-16` | `1e-10` | `pass` |
| `solution_norm` | `1.1102230246251565e-16` | `1e-10` | `pass` |
| `solution_values` | `1.1102230246251565e-16` | `1e-10` | `pass` |
| `project_vs_baseline_max_abs_delta` | `1.1102230246251565e-16` | `1e-10` | `pass` |

Dependency status:

| Dependency | Status | Proof interpretation |
| --- | --- | --- |
| `python3` | `pass` | required interpreter available |
| `tests/qr_external_dense_reference.py` | `pass` | selected helper available |
| `numpy` | `defer` | optional package baseline not selected; not proof |
| `scipy` | `defer` | optional package baseline not selected; not proof |

## Focused Checks

Day 13 validation should be reproduced with:

```sh
make report-index-comparison-freshness
python3 scripts/run_external_comparison.py --self-check
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness
git diff --check
```

The stale wording search from Day 12 remains the active documentation audit:
matches in public and maintainer docs are non-claims or scoped boundaries, not
broad parity claims.

## Quality-Gate Decision

Day 13 changed documentation only.

No `.c` or public `.h` files were modified on Day 13, so the required full
`make format && make lint && make test` gate is not required for this day.

The focused report-index, comparison, schema, and whitespace checks are the
selected Day 13 gate.

## Residual Comparative Gap Register

Still deferred:

- QR comparison beyond `qr_underdetermined_minnorm_2x4`;
- optional NumPy baseline;
- optional SciPy baseline;
- LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, and other ecosystem baselines;
- QR raw Q/R basis identity;
- QR sign/orientation/order parity;
- pivot-order and rank-threshold comparison;
- broad rank-deficient solve comparison;
- broad nullspace and economy-mode comparison;
- sparse-mode and reorder comparison;
- partial-SVD comparison publication under the normalized `comparison` family;
- portable runtime or performance comparison;
- hosted CI comparison publication;
- package-manager, shared-library, loader, and ABI comparison lanes.

## Day 14 Handoff

Day 14 should close Sprint 154 by:

- confirming Day 13 study publication remains reproducible;
- rerunning the selected focused checks;
- reviewing generated/local-only non-claim boundaries;
- recording final residuals and Sprint 155 handoff items;
- preparing the sprint retrospective inputs.
