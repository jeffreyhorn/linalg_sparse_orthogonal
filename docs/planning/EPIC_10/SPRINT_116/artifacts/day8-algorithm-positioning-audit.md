# Day 8 Algorithm Reference Positioning Audit

## Purpose

Day 8 decides whether `docs/algorithm.md` should be treated as public adoption
guidance or technical background, and whether Day 9 needs focused cleanup for
unsupported claims or unclear audience boundaries.

## Inputs Reviewed

| Input | Reviewed for |
|---|---|
| `docs/algorithm.md` | Audience role, public/reference boundary, performance wording, historical evidence, internal-helper wording, and unsupported claims. |
| `README.md` | How the front door points to algorithm documentation. |
| `docs/solver_selection.md` | Whether solver choice guidance already lives outside algorithm background. |
| `examples/README.md` | Whether adoption examples already own first-use workflows. |
| `benchmarks/README.md` | Whether performance measurement interpretation already owns benchmark caveats. |

## Positioning Decision

`docs/algorithm.md` should remain technical background and reference material,
not first-use adoption guidance.

The adoption path is already owned by:

- `README.md` for the front door and quick start;
- `docs/solver_selection.md` for solver-family choice;
- `examples/README.md` for runnable public API examples;
- `benchmarks/README.md` for local measurement and benchmark interpretation;
- `INSTALL.md` for install and downstream consumer workflows.

`docs/algorithm.md` is still useful, but its role is different: it explains
data structures, algorithms, complexity, local evidence, historical decisions,
and implementation tradeoffs.

## Public-Versus-Background Boundary

| Area | Current role | Day 9 decision |
|---|---|---|
| Orthogonal linked-list data structure | Technical background for storage tradeoffs. | Keep. |
| LU, Cholesky, LDL^T, QR, iterative, SVD, and eigensolver sections | Technical explanation of algorithms and implementation behavior. | Keep. |
| Performance tables and fixture measurements | Local historical evidence, not portable guarantees. | Keep with positioning note. |
| Reordering advisory knobs | Technical/historical tuning context for known fixture classes. | Keep with positioning note. |
| OpenMP section | Technical implementation and runtime ownership note. | Keep. |
| API consistency notes | Technical reference context, not the first-use API guide. | Keep. |
| Adoption workflow guidance | Owned elsewhere. | Add a top positioning note that points readers elsewhere. |

## Unsupported-Claim Audit

| Claim class | Finding | Disposition |
|---|---|---|
| State-of-the-art claims | No broad state-of-the-art claim found. The document contains fixture-specific wins, losses, regressions, and retired targets. | No substantive edit required. |
| Universal performance claims | No universal performance claim found. Measurements are tied to fixtures, local benchmarks, or caveats. | Add positioning note to make this easier for readers to interpret. |
| Package/platform support claims | No package, platform, package-manager, shared-library, or ABI support claim found. | No edit required. |
| Public API expansion claims | Some sections discuss internal helpers and implementation details, but not as public adoption contracts. | Add positioning note so internal detail is not mistaken for public API guidance. |
| Maintainer-proof-first wording | Historical sprint evidence is extensive, but it is not promoted as the README adoption path. | Add positioning note rather than rewriting historical sections. |

## Day 9 Edit Checklist

| Item | Edit decision | Rationale |
|---|---|---|
| Add a compact positioning note near the top of `docs/algorithm.md` | Edit | Makes the document role explicit and prevents technical/historical content from being mistaken for first-use adoption guidance. |
| Do not rewrite algorithm sections | No edit | The sections remain useful technical background. |
| Do not change benchmark tables or historical measurements | No edit | Day 10-11 own broader performance wording; Day 9 only needs role clarification. |
| Do not add package/platform or support wording | No edit | Package/platform support is owned by `INSTALL.md` and Sprint 115 guardrails. |
| Do not add implementation claims | No edit | Sprint 116 is adoption QA and documentation boundary work only. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| `docs/algorithm.md` has a clear adoption-facing role | Partially complete; role is clear from surrounding docs but should be explicit in the file. |
| No maintainer-proof-first wording is promoted into adoption guidance | Complete. |
| Day 9 can apply focused cleanup if needed | Complete; add top positioning note only. |

## Validation Notes

- Day 8 changed Sprint 116 planning documentation only.
- `docs/algorithm.md` and related adoption docs were inspected but not edited.
- No `.c` or `.h` files were modified.
