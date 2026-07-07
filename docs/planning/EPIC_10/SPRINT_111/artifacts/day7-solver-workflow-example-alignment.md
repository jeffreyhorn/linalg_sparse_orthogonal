# Day 7 Solver Workflow Example Alignment

## Purpose

Day 7 aligns the runnable example map with the new
`docs/solver_selection.md` guide. Day 6 already updated the compressed-first
direct example source to cover CSR and CSC input. Day 7 therefore focuses on
making the existing direct, iterative, reuse, QR, LDLT, preconditioner,
condition, matrix-free, SVD, eigensolver, and reorder examples discoverable
without exposing test-only helpers or maintainer proof language.

## Touched Files

- `examples/README.md`
- `docs/planning/EPIC_10/SPRINT_111/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_111/artifacts/day7-solver-workflow-example-alignment.md`

## Example Alignment Summary

| Workflow Need | Example Route | Day 7 Action |
|---|---|---|
| Choose a solver workflow first | `docs/solver_selection.md` | Added front-door links from `examples/README.md`. |
| Smallest one-shot direct solve | `example_basic_solve` | Kept as the first-success route. |
| CSR/CSC compressed-first input | `example_compressed_input` | Updated docs to reflect Day 6 CSR and CSC source coverage. |
| Stable-pattern repeated direct solve | `example_analysis` | Kept as the analyze/factor/solve/refactor route. |
| Symmetric indefinite direct solve | `example_ldlt` | Added a discoverable README section. |
| Overdetermined QR least-squares | `example_least_squares` | Kept as the rectangular QR route. |
| Underdetermined minimum-norm QR | `example_minnorm` | Added a discoverable README section. |
| Reorder/fill and COLAMD | `example_colamd` | Added a discoverable README section with symmetric-vs-column ordering wording. |
| One-shot iterative solve | `example_iterative` | Kept as GMRES/ILU(0) route. |
| IC(0), CG, and MINRES | `example_ic_minres` | Added a discoverable README section. |
| Matrix-free iterative solve | `example_matrix_free` | Added a discoverable README section. |
| SVD/low-rank | `example_svd_lowrank` | Kept and clarified line wrapping in docs. |
| Condition number | `example_condition` | Added a discoverable README section. |
| Symmetric eigensolver | `example_eigs` | Kept as one-shot symmetric eigensolver route. |
| Installed downstream consumer | `examples/cmake_example/` | Kept separate from local build-tree examples. |

## Public API Guardrails

- No test-only helper pattern was promoted into user documentation.
- No private headers or private source owners were exposed.
- The repeated-run direct lifecycle remains explicit in `example_analysis`.
- Iterative handle reuse remains limited to CG, GMRES, and MINRES.
- `BiCGSTAB` remains a one-shot compatibility route.
- Benchmark docs remain measurement guidance, not portable performance proof.
- Matrix Market load/use remains scheduled for Day 8 rather than being implied
  by the eigensolver fixture-loading example.

## Validation

Day 7 changed documentation only, so required validation is:

- `git diff --check`
- trailing-whitespace scan over touched docs

No example source changed on Day 7, and Day 6 already rebuilt and ran the
modified compressed-input example.

## Completion Criteria Status

- Solver example documentation follows public APIs only.
- Guide text and examples now agree on matrix format, lifecycle, and reuse
  boundaries.
- Validation expectations are documentation-only for Day 7.
- Examples remain minimal and copyable; Day 7 did not grow any source file.
