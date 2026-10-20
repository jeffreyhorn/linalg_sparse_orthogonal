# Sprint 217 Day 12: User Guidance And Claim Boundary

**Scope:** [Day 12 plan](../PLAN.md), [Day 11 header traceability](day11-public-vector-solver-contract.md), and [Day 9 ILUT evidence](day9-ilut-validity-failure-behavior.md).

## Claim Matrix

| Surface | Supported statement | Boundary retained |
| --- | --- | --- |
| `vec_norm2` and `vec_norminf` | Scaled 2-norm avoids false whole-norm zero/infinity from direct squaring for finite representable results; both norms propagate NaN; NULL/nonpositive length retains zero | Tiny contributions can round away, the final 2-norm can exceed `double`, and norm arithmetic alone does not prove accuracy for ill-conditioned solves |
| Selected iterative solvers | CG, GMRES, MINRES, BiCGSTAB, their covered matrix-free variants, and block CG distinguish nonfinite caller input (`BADARG`) from solver-detected numeric failure (`NUMERIC`, no convergence, NaN diagnostic) | No all-family convergence or independent block-adapter parity; callback and other hard-error result fields remain API-specific |
| ILUT factor-system diagnostic | The selected `west0067` ILUT solve has finite `z` and an independent, scale-aware `L(Uz)` backward error (`8.249302e-80 <= 1e-12` on the Day 9 host); `sparse_ilu_solve` rejects nonfinite factor-system work | `L*U` is incomplete, so this does not establish `A*z=r` or useful preconditioning; selected 500-iteration GMRES checks did not converge |
| Project standing | The source-controlled API and solver-selection routes explain the bounded contracts | No state-of-the-art, external-library parity, portable performance, or broad numerical reliability claim follows from this sprint |

## Item 217.5 Documentation Freeze

The 20-hour planned allocation (Day 11: 10 h; Day 12: 10 h) is complete for
this scoped documentation item. The final touched public-contract list is
`include/sparse_vector.h`, `include/sparse_iterative.h`, and
`include/sparse_ilu.h`. The final touched user-guide list is `README.md`,
`docs/solver_selection.md`, and `docs/api_reference.md`. The evidence list
is this artifact, the Day 11 artifact, and `WORKING_NOTES.md`. No
`docs/maintainer_guide.md` edit is needed: its existing support/evidence
boundaries remain applicable.

No new prose-matching documentation guard was added. The guidance links to
the exact public headers and selected fixture. Existing API routing checks
exercise the stable links; the external-baseline claim test covers the
selected CHOLMOD pilot, not every state-of-the-art claim. The broader Sprint
217 wording was reviewed against the claim matrix above. A new phrase-sensitive
guard without a new machine-verifiable contract would be brittle.

## Validation

- `make -s docs-check`: pass; 18 public-header reference pages and 18 source
  pages covered. The regenerated vector page shows the narrowed scaled-norm
  wording.
- `python3 scripts/check_api_docs_routing.py`: pass; seven routing documents
  checked. `python3 tests/test_api_docs_routing.py`: pass.
- `python3 tests/test_external_baseline_claim_docs.py`: pass (5 tests),
  retaining the selected external-pilot comparison boundary.
- New local link targets resolve to the checked-in vector, iterative, and
  ILU headers and `tests/test_ilu.c`; no link points to generated HTML.
- `make -s format-check` and `git diff --check`: pass. No C behavior changed
  on Day 12; Sprint 217's focused/instrumented and final full-quality runs
  remain assigned to Days 13-14.
