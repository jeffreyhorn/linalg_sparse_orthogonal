# Sprint 103 Day 5 Iterative Oracle Batch Design

## Purpose

Day 5 selects and freezes the first Sprint 103 iterative comparison batch
before implementation. The design uses the Day 2 ranking, Day 3 fixture
taxonomy, and Day 4 helper/reporting boundary to define exact fixtures,
reference behavior, tolerances, ownership, validation commands, and non-claims.

## Selected Family

Primary family: **BiCGSTAB**

Rationale:

- Day 2 ranked BiCGSTAB as the highest iterative comparison gap.
- Nonsymmetric iterative solves are high user-impact and numerically fragile.
- Existing BiCGSTAB tests already cover many API and residual behaviors, but
  the evidence is distributed across many tests and lacks one focused
  comparison artifact with fixture taxonomy, reference behavior, and claim
  boundaries.
- Day 4 concluded that a new shared helper is not required before the first
  iterative batch.

## Existing Coverage to Preserve

| existing lane | file | current evidence |
|---|---|---|
| small known solutions | `tests/test_bicgstab.c` | `test_bicgstab_3x3_known_solution`, `test_bicgstab_5x5_known_solution` |
| direct-solver cross-check | `tests/test_bicgstab.c` | `test_bicgstab_vs_lu_direct` |
| true residual check | `tests/test_bicgstab.c` | `test_bicgstab_true_residual_matches` |
| preconditioner behavior | `tests/test_bicgstab.c` | ILU and ILUT lanes, preconditioner iteration comparison |
| corpus fixtures | `tests/test_bicgstab.c` | `west0067`, `steam1`, `orsirr_1` |
| GMRES comparison | `tests/test_bicgstab.c` | `steam1` and nonsymmetric tridiagonal comparison |
| stagnation/breakdown | `tests/test_stagnation.c` | BiCGSTAB stagnation and breakdown tests |

Day 6 should not remove or weaken these lanes. The new batch should add a
claim-owned comparison layer rather than refactor existing coverage.

## Selected Batch

| batch item | fixture key | taxonomy class | profile | reference behavior | expected result |
|---|---|---|---|---|---|
| deterministic nonsymmetric solve comparison | `bicgstab_nonsym_known_5` | `nonsym-known-solution` | `fast-exact` | constructed `x_true`; compare BiCGSTAB against LU solution and true residual | converged; residual `< 1e-10`; solution matches LU and `x_true` within tolerance |
| corpus residual comparison | `bicgstab_steam1_ilu_vs_gmres30` | `nonsym-mm-medium` / `ill-conditioned-scale` | `slow-convergent` | BiCGSTAB+ILU and GMRES(30)+ILU true residual comparison on `steam1` | both converge; both true residuals `< 1e-4`; no iteration superiority claim |
| expected hard-case boundary | `bicgstab_small_budget_unsym_tridiag` | `nonsym-known-solution` | `expected-nonconvergent` | deliberately small iteration budget on deterministic nonsymmetric tridiagonal | returns `SPARSE_ERR_NOT_CONVERGED` or non-converged result with finite residual |

The batch deliberately excludes an external Python dense helper. Constructed
solutions, LU cross-checks, and GMRES residual comparisons are sufficient for
the first iterative comparison lane and avoid adding helper status complexity
before it is needed.

## Fixture and Tolerance Matrix

| fixture key | construction/load path | dimensions | RHS/target | tolerance model |
|---|---|---:|---|---|
| `bicgstab_nonsym_known_5` | local deterministic builder in `tests/test_bicgstab.c` using the existing nonsymmetric 5x5 pattern | 5 x 5 | `x_true = {1, -1, 2, -2, 3}`, `b = A*x_true` | BiCGSTAB residual `< 1e-10`; `max |x_bicgstab - x_true| < 1e-8`; `max |x_bicgstab - x_lu| < 1e-8` |
| `bicgstab_steam1_ilu_vs_gmres30` | `tests/data/suitesparse/steam1.mtx` | 240 x 240 | `x_true[i] = i + 1`, `b = A*x_true` | BiCGSTAB and GMRES true residuals `< 1e-4`; both result structs report convergence |
| `bicgstab_small_budget_unsym_tridiag` | existing `build_unsym_tridiag(n, 4.0, -1.0, -2.0)` or equivalent | 50 x 50 | deterministic sine or `i + 1` RHS | max iteration budget intentionally too small; finite residual; expected non-convergence is the pass condition |

## Implementation Ownership

Expected Day 6 touched file:

| file | intended change |
|---|---|
| `tests/test_bicgstab.c` | add a Sprint 103 comparison section with the selected batch tests and register them in the existing `RUN_TEST` list |

No public headers, library sources, CMake files, Makefile targets, or Python
helpers are planned for Day 6.

Shared helper changes are explicitly out of scope unless implementation finds
meaningful duplication that cannot be handled locally. If that happens, Day 6
must stop and record the helper gap before editing shared `.h` files.

## Reporting Contract for Day 6 Tests

Each selected test should make the following evidence visible in code comments
or diagnostic output:

| field | required content |
|---|---|
| fixture key | one of the selected keys above |
| taxonomy class | Day 3 class |
| reference behavior | constructed solution, LU cross-check, GMRES comparison, or expected non-convergence |
| acceptance thresholds | residual and solution thresholds before assertions |
| non-claim | no external package parity and no iteration performance superiority |

Diagnostic prints may include iteration counts and true residuals, but those
counts are descriptive unless explicitly thresholded above.

## Focused Validation Commands

Because Day 6 is expected to modify a `.c` test file, required validation is:

```sh
make build/test_bicgstab
./build/test_bicgstab
make format && make lint && make test
git diff --check
rg -n "[ \t]+$" tests/test_bicgstab.c docs/planning/EPIC_10/SPRINT_103
```

If only documentation changes unexpectedly occur, the reduced validation is:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103
```

## Deferred Iterative Follow-Ups

| follow-up | disposition |
|---|---|
| external Python dense solve helper for BiCGSTAB | deferred; not needed for the first batch and would require helper output/status design |
| CG external comparison lane | deferred; Day 2 found lower comparison gap than BiCGSTAB |
| MINRES consolidated comparison artifact | deferred; useful after BiCGSTAB design validates the reporting contract |
| shared residual/reporting helper extraction | deferred; Day 4 found reporting commonality but no safe implementation extraction yet |
| GMRES restart comparison expansion | deferred; outside the Sprint 103 named Day 5 priority and already has broad existing coverage |

## Non-Claims Preserved

The Day 5 design does not claim:

- BiCGSTAB parity with PETSc, SciPy, Trilinos, or other external packages;
- external helper evidence for BiCGSTAB;
- portable iteration-count or performance superiority;
- correctness on all nonsymmetric sparse systems;
- that GMRES comparison is an independent external oracle;
- any public API change.

## Day 5 Conclusion

Day 6 should implement a focused BiCGSTAB comparison batch in
`tests/test_bicgstab.c`: one deterministic known-solution plus LU comparison,
one `steam1` residual comparison against GMRES(30), and one expected
non-convergence boundary with a deliberately small iteration budget. This fits
the Sprint 103 budget, avoids premature helper extraction, and gives Day 7 a
clear iterative evidence package to validate and rerank.
