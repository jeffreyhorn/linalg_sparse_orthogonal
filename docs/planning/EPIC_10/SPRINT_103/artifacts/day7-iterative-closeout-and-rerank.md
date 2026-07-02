# Sprint 103 Day 7 Iterative Closeout and Rerank

## Purpose

Day 7 closes out the Sprint 103 iterative comparison batch before spectral and
SVD implementation begins. It validates the touched iterative surface, checks
whether Day 6 introduced helper or reporting debt, and reranks the remaining
eigen, thick-restart, LOBPCG, and SVD comparison work against the evidence now
available.

## Iterative Evidence Closed Out

Day 6 implemented the selected BiCGSTAB comparison batch in
`tests/test_bicgstab.c`.

| evidence lane | fixture key | closeout status |
|---|---|---|
| deterministic nonsymmetric known-solution comparison | `bicgstab_nonsym_known_5` | validated against constructed `x_true`, LU, and true residual |
| corpus residual comparison | `bicgstab_steam1_ilu_vs_gmres30` | validated with BiCGSTAB+ILU and GMRES(30)+ILU true residuals below the declared threshold |
| expected non-convergence boundary | `bicgstab_small_budget_unsym_tridiag` | validated as an expected `SPARSE_ERR_NOT_CONVERGED` path with finite residual |

The batch reduces the Day 2 BiCGSTAB comparison gap from "highest priority" to
"bounded iterative evidence landed." It does not close all BiCGSTAB comparison
work because no independent external solver helper was added and the corpus
coverage remains limited to named fixtures.

## Day 7 Validation Results

| command | result |
|---|---|
| `make build/test_bicgstab build/test_iterative build/test_stagnation` | passed; all targets already up to date |
| `./build/test_bicgstab` | passed; 61 tests, 0 failures, 0 skips, 466 assertions |
| `./build/test_iterative` | passed; 80 tests, 0 failures, 0 skips, 711 assertions |
| `./build/test_stagnation` | passed; 46 tests, 0 failures, 0 skips, 308 assertions |

Focused iterative validation confirms that the new BiCGSTAB comparison tests do
not disturb the adjacent CG, GMRES, MINRES, stagnation, residual-history, or
breakdown surfaces covered by the existing iterative binaries.

## Helper, Fixture, and Reporting Debt Review

| area | Day 7 finding | disposition |
|---|---|---|
| shared helper extraction | Day 6 kept all new comparison code local to `tests/test_bicgstab.c` | no extraction debt created |
| public API or build files | no public headers, library sources, CMake files, or Makefile targets changed | no API or build-system debt created |
| fixture ownership | new deterministic builder is local, small, and tied to one fixture key | acceptable for current evidence scope |
| diagnostic reporting | tests print descriptive residual and iteration data only | preserve as bounded evidence; no performance claim |
| external helper status | no external helper was added | keep deferred until a future sprint explicitly scopes helper availability and skip semantics |
| iterative follow-ups | CG and MINRES remain lower-gap families with existing direct-solver and residual coverage | defer unless later spectral work exposes reusable reporting infrastructure |

## Updated Solver-Family Ranking

Day 2 ranked BiCGSTAB first because nonsymmetric iterative comparison evidence
had the highest gap-to-impact ratio. After Day 6, the remaining Sprint 103
implementation queue is:

| rank | family or lane | updated rationale |
|---:|---|---|
| 1 | LOBPCG residual and orthogonality comparison lane | Still has high user value and high numerical risk; current evidence is mostly deterministic or internal, and preconditioned spectral workflows need bounded comparison proof. |
| 2 | thick-restart eigensolver independent fixture comparison lane | Restart-specific behavior still leans on grow-m parity; the next best improvement is a bounded fixture with residual and orthogonality checks independent of broad ARPACK claims. |
| 3 | SVD singular-value/rank/reconstruction follow-through | Broad invariant coverage exists, but rank and reconstruction evidence should follow spectral fixture scoping so singular-value claims stay bounded. |
| 4 | grow-m eigensolver documentation and residual interpretation | Core eigensolver evidence is useful, but the most immediate value is clearer interpretation unless Day 8 identifies a low-cost reference fixture. |
| 5 | MINRES consolidated comparison artifact | Existing coverage remains strong; useful as taxonomy documentation or a future external-helper consumer rather than a Sprint 103 implementation priority. |
| 6 | CG convergence-profile consumer lane | Visible and important, but current known-solution, residual, SuiteSparse, preconditioner, and direct-solver checks are stronger than the remaining spectral gaps. |
| 7 | BiCGSTAB external-helper expansion | Day 6 reduced the immediate gap; further work should wait until external helper status and oracle independence are explicitly scoped. |

## Residual Follow-Up Queue

| follow-up | owner window | notes |
|---|---|---|
| LOBPCG comparison design | Day 8 | bind diagonal, Laplacian, and optional SuiteSparse fixtures to residual and orthogonality criteria |
| thick-restart comparison design | Day 8 or Day 9 | avoid treating grow-m parity as independent proof; use it only as secondary context |
| SVD overlap scoping | Day 10 | share rank, singular-value, reconstruction, and orthogonality threshold language with spectral evidence |
| residual interpretation docs | Day 12 | explain true residuals, relative residuals, convergence status, iteration descriptions, and non-claims |
| external iterative helper | future sprint or explicit Day 14 handoff | requires helper availability, skip reason, subprocess status, and oracle-version ownership |

## Claim Boundaries After Closeout

Day 7 preserves the Day 6 claim boundaries:

- Sprint 103 has bounded BiCGSTAB evidence for three named fixtures.
- GMRES(30) comparison is an internal cross-check, not an external oracle.
- Local iteration counts are descriptive and are not performance claims.
- No PETSc, SciPy, Trilinos, ARPACK, LAPACK, or broad package parity claim is
  earned by the iterative batch.
- The remaining spectral and SVD work must cite fixture taxonomy entries and
  declare residual, orthogonality, rank, or reconstruction thresholds before
  implementation.

## Day 8 Handoff

Day 8 should design the spectral comparison batch from the updated ranking. The
highest-value target is LOBPCG, with thick-restart close behind. The design
should:

- keep eigenvalue agreement separate from eigenpair residuals;
- keep vector orthogonality separate from convergence status;
- name any grow-m parity as internal comparison evidence only;
- avoid broad ARPACK, SciPy, or package-parity wording;
- record focused validation commands before code changes.

## Day 7 Conclusion

The iterative comparison batch is validated and closed. BiCGSTAB no longer
blocks spectral implementation, and Sprint 103 can move into LOBPCG,
thick-restart, and SVD evidence with a refreshed ranking and explicit deferred
iterative ownership.
