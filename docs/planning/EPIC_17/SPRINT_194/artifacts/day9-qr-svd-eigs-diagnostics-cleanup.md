# Sprint 194 Day 9 QR/SVD/Eigensolver Diagnostics Cleanup

## Objective

Apply the Day 7 diagnostics wording contract to QR, SVD, and symmetric
eigensolver documentation without changing numerical behavior, public result
fields, backend selection, tolerance semantics, or evidence scope.

## Inputs

- `docs/planning/EPIC_17/SPRINT_194/artifacts/day7-diagnostics-wording-contract.md`
- `README.md`
- `docs/solver_selection.md`
- `docs/cookbook.md`
- `examples/README.md`
- `docs/maintainer_guide.md`
- public QR, SVD, and eigensolver headers under `include/`

## QR Wording Changes

| Location | Before | After |
| --- | --- | --- |
| `README.md` API list | QR residual and rank entries were correct but broad. | QR solve now names the QR-local `||b - A*x||_2` residual; QR rank and rank-info entries now say QR-local/tolerance-local diagnostics. |
| `docs/solver_selection.md` QR evidence boundary | Fixture-local rank/nullity/nullspace residual behavior was stated without explicit tolerance locality. | The boundary now states the behavior is under fixture tolerances owned by the gate and retains sign/orientation non-claims. |
| `examples/README.md` least-squares example | Described the least-squares solution and residuals as teaching output. | Now says QR least-squares solution and problem-local per-equation residuals. |
| `examples/README.md` QR corpus note | Said the corpus proof "proves" fixture-local behavior. | Now says the corpus supports only fixture-local rank, nullity, and nullspace residual behavior under fixture tolerances. |

## SVD Wording Changes

| Location | Before | After |
| --- | --- | --- |
| `README.md` capability summary | Partial-SVD proof wording named fixtures but did not enumerate the diagnostic scope. | Now frames the proof as fixture-local evidence for named top-k, rank, projector, triplet-residual, orthogonality, sparse-output, and recovery diagnostics. |
| `README.md` SVD API list | Rank and condition wording could read broader than API-local diagnostics; low-rank entries used "best rank-k" shorthand. | Now says SVD-local rank, 2-norm condition estimate, SVD iteration non-convergence, and rank-k output from the SVD API. |
| `docs/solver_selection.md` SVD workflow | Rank and corpus wording was mostly correct but not consistently SVD-local. | Now says SVD-local numerical rank and tight-budget `SPARSE_ERR_NOT_CONVERGED` fail-closed behavior. |
| `docs/cookbook.md` SVD workflow | Did not explicitly say SVD rank/condition/residual/orthogonality wording is SVD-local. | Added SVD-local diagnostic scope and retained raw-vector, repeated-spectrum, sparse-output, and state-of-the-art non-claims. |
| `examples/README.md` SVD example | Described condition and rank estimation generally. | Now says SVD-local condition estimate and SVD-local rank estimation at selected tolerances. |

## Symmetric Eigensolver Wording Changes

| Location | Before | After |
| --- | --- | --- |
| `README.md` eigensolver summary | Used shorthand `result.*` fields and described AUTO backend selection without an explicit non-superiority boundary in the summary. | Now names `sparse_eigs_t.residual_norm`, `used_csc_path_ldlt`, `peak_basis_size`, and `backend_used`, and states AUTO routing is not universal backend superiority. |
| `README.md` eigensolver API list | Listed backend and shift-invert fields but omitted residual/convergence count in the same handoff. | Now includes `result.residual_norm`, `result.n_converged`, shift-invert backend, basis telemetry, and concrete AUTO routing. |
| `docs/cookbook.md` eigensolver workflow | Told users to leave AUTO unless profiling/workload-specific control justified explicit backend selection. | Now also says AUTO is routing policy and points users at `sparse_eigs_t` fields before changing backend, shift-invert, or preconditioner settings. |
| `examples/README.md` eigensolver example | Reported eigen-equation residuals and backend paths as example output. | Now says residuals are problem-local and tells users to inspect `sparse_eigs_t` fields before treating backend selection as the next tuning target. |
| `docs/maintainer_guide.md` wording rules | Already distinguished iterative/eigs/SVD residuals. | Added explicit QR-local, SVD-local, and eigensolver AUTO-routing rules. |

## Retained Non-Claims

Day 9 explicitly retained non-claims for:

- broad QR correctness;
- broad least-squares parity;
- raw QR basis parity;
- QR sign or orientation identity;
- global rank-threshold policy;
- broad rank-deficient solve behavior;
- broad minimum-norm behavior;
- broad SVD or partial-SVD correctness;
- raw singular-vector identity;
- vector sign or orientation identity;
- repeated-spectrum ordering;
- broad sparse-output optimality;
- nonsymmetric eigensolver support;
- backend superiority;
- portable preconditioner superiority;
- external-library parity;
- broad platform parity;
- package or ABI proof;
- portable performance;
- state-of-the-art status.

## Retained Semantics

- No QR, SVD, or eigensolver API behavior changed.
- No public result field changed.
- No tolerance or rank-threshold behavior changed.
- No backend selection policy changed.
- No report freshness target changed.
- No selected target manifest changed.
- No generated report evidence changed.

## Validation Plan

Day 9 changes are documentation-only, but they touch user-facing docs and
selected evidence wording. Validate with:

```sh
git diff --check
python3 tests/test_selected_performance_docs.py
```

No `.c` or `.h` files were modified, so the full
`make format && make lint && make test` gate is not required for this day.
