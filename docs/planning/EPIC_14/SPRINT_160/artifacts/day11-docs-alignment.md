# Day 11 Documentation Alignment

## Summary

Day 11 aligned the selected QR comparison documentation with the implemented
two-family report surface and drafted the Sprint 161 partial-SVD comparison
handoff.

The documentation now describes `make report-index-comparison-freshness` as a
selected QR comparison gate for both `qr-minnorm` and `qr-compatible-ls`, not
as a minimum-norm-only lane.

## Documentation Alignment

| Surface | Alignment |
| --- | --- |
| `tests/corpus/README.md` | Added a selected QR comparison freshness section with both target families, artifacts, row meanings, and non-claims. |
| `docs/maintainer_guide.md` | Confirmed maintainer instructions list both targets, both artifact groups, 12 selected generated rows, and local-only claim boundaries. Corrected stale “two underlying commands” wording. |
| `docs/solver_selection.md` | Confirms QR solver-selection wording refers to selected fixture-local minimum-norm and compatible least-squares comparisons. |
| `README.md` | Confirms public freshness command and QR evidence wording describe selected QR comparison freshness without broadening claims. |

## Selected QR Comparison Contract

| Target | Fixture | Artifact | Claim Boundary |
| --- | --- | --- | --- |
| `qr-minnorm` | `qr_underdetermined_minnorm_2x4` | `build/comparison/qr_minnorm/study.tsv` | fixture-local minimum-norm solve comparison only |
| `qr-compatible-ls` | `qr_overdetermined_compatible_5x3` | `build/comparison/qr_compatible_ls/study.tsv` | fixture-local compatible least-squares comparison only |

Each selected family contributes:

- `project_status`
- `baseline_status`
- `residual_norm`
- `solution_norm`
- `solution_values`
- `project_vs_baseline_max_abs_delta`

The generated rows remain `local_only`. Reviewed hosted execution may prove
the selected gate ran and passed on a reviewed Linux surface, but it does not
convert the rows into broad platform, release, package, ABI, performance, or
external-library evidence.

## Non-Claim Preservation

The aligned wording preserves explicit non-claims for:

- broad QR parity;
- raw QR basis identity;
- Q sign/orientation identity;
- global rank-threshold behavior;
- broad rank-deficient solve behavior;
- NumPy/SciPy/LAPACK/SuiteSparse/Eigen parity;
- hosted platform proof beyond the selected lane;
- package-manager behavior;
- shared-library ABI;
- performance superiority;
- release proof;
- state-of-the-art status.

Optional NumPy and SciPy dependency rows remain deferred context only. They
cannot create selected pass evidence.

## Sprint 161 Partial-SVD Comparison Handoff

Sprint 161 should publish the first bounded partial-SVD comparison family using
the Sprint 160 comparison pattern.

### Recommended Starting Target

Start with a diagonal or low-risk source-controlled fixture before attempting
subspace-sensitive rank-deficient or repeated-spectrum behavior.

Recommended first candidate:

| Candidate | Rationale |
| --- | --- |
| `partial_svd_diag6_k2` | Existing dense helper support, deterministic singular values, simple top-k semantics, and low risk of raw singular-vector identity claims. |

Alternates to consider after the first candidate is stable:

| Candidate | Reason to defer initially |
| --- | --- |
| `partial_svd_tall_diag_8x5_k3` | Still diagonal but adds rectangular/tall shape and zero-row interpretation. |
| `partial_svd_nonsym_rect10x8_k3` | Nonsymmetric rectangular fixture needs clearer metric and tolerance design. |
| `partial_svd_clustered_repeated_diag8x6_k3_v1` | Repeated/clustered spectra require subspace-safe comparison, not raw vector identity. |
| `partial_svd_rankdef_diag6x4_k2_range_projector_v1` | Rank-deficient projector semantics need careful row design and non-claims. |

### Required Design Lessons From Sprint 160

1. Select one fixture family before implementation.
2. Define metric rows, tolerances, skip/defer states, stale behavior, and
   non-claims before code changes.
3. Use descriptor-backed target definitions rather than one-off globals.
4. Keep generated artifacts isolated under a stable subdirectory.
5. Add source-controlled `report_families.tsv` metadata before interpreting
   generated rows.
6. Add focused runner tests for target dispatch, generated files, row IDs,
   metadata, and optional dependency context.
7. Add normalizer tests for complete, missing, unexpected, duplicate, stale,
   fail, and defer selected rows.
8. Keep C proof-owner tests unchanged unless solver implementation or fixture
   helpers change.
9. Require `make report-index-comparison-freshness` or a future equivalent to
   regenerate selected outputs before strict freshness normalization.
10. Preserve local-only support tier unless a later sprint explicitly promotes
    hosted interpretation.

### Candidate Partial-SVD Row Shape

The first partial-SVD comparison should avoid broad vector identity. For a
simple diagonal first target, suggested selected rows are:

- `project_status`
- `baseline_status`
- `singular_values`
- `singular_value_max_abs_delta`
- `triplet_residual_norm`
- `orthogonality_residual`
- `project_vs_baseline_metric_status`

For repeated-spectrum or rank-deficient targets, use subspace/projector
distances instead of raw singular-vector ordering or sign identity.

### Sprint 161 Non-Goals

Sprint 161 should not claim:

- broad partial-SVD correctness;
- raw singular-vector identity;
- vector sign or ordering identity across repeated spectra;
- convergence-rate superiority;
- partial-result guarantees after fail-closed outcomes;
- broad sparse-output/drop-tolerance optimality;
- NumPy/SciPy/LAPACK parity;
- performance, package, ABI, platform, release, or state-of-the-art evidence.

## Validation Plan

Day 11 changed documentation only. Required validation:

```sh
git diff --check
```

No `.c`, `.h`, Python script, Makefile, or manifest changes were made on Day
11, so script and C quality gates are not required for this day.

## Day 12 Handoff

Day 12 should run the focused local validation pass for the full Sprint 160
changed-file surface, including the runner tests, normalizer tests, corpus
schema validation, selected comparison freshness, and documentation hygiene.
