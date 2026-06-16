# Sprint 72 Retrospective

**Sprint:** 72 — Core Matrix/Product Model Convergence Phase 1  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 72 scope and implementation-day validation baseline were captured before ownership work began
- [x] the strongest product-model contradiction centers were re-ranked from the live repo before the first implementation fence was fixed
- [x] the first landing stayed bounded to the public direct-workflow seam and did not widen into a broad `SparseMatrix` rewrite
- [x] the first ownership batch landed across the matrix shell and direct-workflow headers without widening into CSC/CSR redesign
- [x] `sparse_reset_perms()` now drops stale one-shot solve compatibility and recovers a plain matrix shell
- [x] the strongest compressed-path seam was narrowed to Cholesky CSC publish-back and landed as a bounded ownership cleanup
- [x] family-local and integration proof owners were aligned to the landed Day 6 and Day 9 boundaries
- [x] public contract follow-through stayed bounded to header-local truth rather than reopening broader public-surface cleanup
- [x] the full Sprint 72 branch passed the standard code-day gate, the strongest reviewed baseline, and the targeted follow-on proof/install surfaces
- [x] Sprint 72 closed with one explicit first-phase product-model convergence package and a ranked Sprint 73 carry-forward queue

## What Went Well

1. **Sprint 72 reduced two real product-model contradictions instead of only restating them.**
   The branch landed substantive ownership cleanup at:
   - `src/sparse_matrix.c`
   - `src/sparse_chol_csc.c`
   - `include/sparse_matrix.h`
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
   and tied those changes to focused proof in:
   - `tests/test_integration.c`
   - `tests/test_chol_csc.c`

2. **The first lane stayed properly bounded.**
   Sprint 72 did not collapse into:
   - a broad `SparseMatrix` rewrite
   - generic matrix arithmetic redesign
   - LDL^T or LU CSR ownership widening
   - capability/platform/docs spill
   That kept the sprint aligned with the Sprint 70 architecture contract.

3. **The direct-workflow boundary is now materially clearer.**
   The key Day 6 ownership rule is now explicit and enforced:
   - copied factored `SparseMatrix` shells are short-lived one-shot compatibility shells
   - `sparse_reset_perms()` recovers a plain matrix shell
   - stale reordered/factored solve compatibility does not survive permutation reset
   - repeated-run analysis/factor surfaces read more clearly as the long-lived reuse owner

4. **The strongest compressed Cholesky seam was improved without widening the family scope.**
   The Day 9 batch turned `chol_csc_writeback_to_sparse(...)` into a cleaner publish-back pipeline and proved the result in the strongest local owner:
   - solve-ready published shell
   - explicit reorder payload
   - identity internal row/column permutation shells
   - correct SPD solve behavior

5. **Proof ownership and public contract wording stayed aligned with the landed code.**
   Sprint 72 correctly followed through in:
   - `include/sparse_matrix.h`
   - `include/sparse_cholesky.h`
   - `docs/maintainer_guide.md`
   without reopening:
   - `README.md`
   - `docs/tutorial.md`
   - example adoption surfaces
   - benchmark interpretation surfaces

6. **The validated close state is strong.**
   Sprint 72 ended with:
   - `make format` passed
   - `make lint` passed
   - `make test` passed
   - `make quality-review-full` passed
   - reviewed CMake parity still exact at `53`
   - Makefile/CMake parity still `53 vs 53`
   - reviewed CMake `ctest` still `53 / 53`
   - install/package regressions still clean

## What Didn't Go Well

1. **Sprint 72 only completes the first product-model phase.**
   The branch cleaned the strongest direct-workflow and Cholesky CSC seams, but it did not yet resolve:
   - mixed logical/physical/permuted-state semantics across the wider matrix API
   - later compressed-path seams in LDL^T or LU CSR
   - broader family-local factor/publication convergence

2. **The linked-list public center still exists as a compatibility shell rather than a fully re-centered product model.**
   That is the correct Sprint 72 outcome, but not the end-state Epic 7 is aiming at.

3. **The strongest capability and configuration ceilings remain untouched by design.**
   Sprint 72 correctly did not widen into:
   - index-width modernization
   - scalar-surface broadening
   - unsymmetric eigensolver expansion
   - residual env-var/default-policy convergence
   Those stay open because they were not the right first-phase product-model target.

4. **Runtime asymmetry in the reviewed suite remains visible.**
   The full reviewed path passed, but `test_reorder_nd` still dominated reviewed CMake time. That is not a Sprint 72 blocker, but it remains operational friction for later proof-heavy work.

5. **The branch depended on disciplined non-moves.**
   Sprint 72’s success required not reopening public docs, examples, benchmarks, or platform surfaces unless the landed ownership work truly forced it. That discipline held, but it means later sprints still need to maintain that fence carefully.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 72 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `334.55 sec` |
| reviewed `test_reorder_nd` time | `240.93 sec` |
| install regression | `11 / 11` |
| CMake install regression | `13 / 13` |

### Sprint 72 artifact package

| Metric | Sprint 72 close state |
|---|---:|
| total artifact files under `SPRINT_72/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/landing artifacts | `6` |
| review/closeout artifacts | `3` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-product-model-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-rerun-recheck.md`
  - `day3-product-model-surface-audit.md`
  - `day4-first-product-model-boundary.md`
  - `day7-post-landing-audit-and-rerank.md`
- design/landing artifacts:
  - `day5-ownership-convergence-design.md`
  - `day6-ownership-convergence-batch1.md`
  - `day8-compressed-path-ownership-design.md`
  - `day9-compressed-path-ownership-batch.md`
  - `day10-public-contract-and-example-adoption-design.md`
  - `day11-public-contract-and-example-adoption-batch.md`
- review/closeout artifacts:
  - `day12-regression-expansion-and-build-alignment.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed product-model package

| Metric | Sprint 72 close state |
|---|---:|
| public headers touched in landed package | `5` |
| source files touched in landed package | `2` |
| focused proof-owner tests touched | `2` |
| maintained proof/policy docs touched | `1` |
| representative example residual anchors retained | `2` |
| maintained benchmark proof rows retained | `2` |

Notes:

- public headers touched:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- source files touched:
  - `src/sparse_matrix.c`
  - `src/sparse_chol_csc.c`
- focused proof-owner tests touched:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
- maintained proof/policy docs touched:
  - `docs/maintainer_guide.md`
- representative example residual anchors retained:
  - `example_analysis` -> `4.44e-16`
  - `example_basic_solve` -> `0.00e+00`
- maintained benchmark proof rows retained:
  - `bench_refactor_csc nos4` -> `speedup_refactor = 1.69`
  - `bench_chol_csc nos4` -> residuals `7.06e-16`, `5.89e-16`, `5.89e-16`

## Residual Deferred Debt

Sprint 72 deliberately stopped after the first product-model phase. The main
open work it intentionally hands forward is:

- next-phase product-model convergence on the remaining matrix-state and compressed-path seams
- configuration modernization only where the remaining env-var/default-policy seams still carry real ownership cost
- capability modernization led by index width, with scalar breadth later
- benchmark-governed backend/performance maturity without widening product or platform claims
- later permanent-surface cleanup only where future implementation work moves ownership again

Still consciously constrained rather than silently “solved”:

- no broad `SparseMatrix` rewrite
- no LDL^T or LU CSR ownership widening in the first-phase batch
- no capability/platform/docs spill hidden inside product-model work
- no reinterpretation of benchmark rows or install checks as broader reviewed platform claims
- no weakening of the Sprint 70 truthfulness fence

Not carried forward as unresolved Sprint 72 debt:

- the product-model baseline and first implementation fence
- the Day 6 matrix-shell ownership convergence batch
- the Day 9 Cholesky CSC publish-back ownership batch
- the bounded public contract follow-through in the touched headers
- the maintainer-guide proof-owner alignment
- the full Day 13 validation sweep
- the Day 14 closeout and ranked Sprint 73 handoff queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-product-model-baseline.md](./artifacts/day1-scope-and-product-model-baseline.md)
- [day1-authoritative-inputs.txt](./artifacts/day1-authoritative-inputs.txt)
- [day2-validation-baseline-and-rerun-recheck.md](./artifacts/day2-validation-baseline-and-rerun-recheck.md)
- [day3-product-model-surface-audit.md](./artifacts/day3-product-model-surface-audit.md)
- [day4-first-product-model-boundary.md](./artifacts/day4-first-product-model-boundary.md)
- [day5-ownership-convergence-design.md](./artifacts/day5-ownership-convergence-design.md)
- [day6-ownership-convergence-batch1.md](./artifacts/day6-ownership-convergence-batch1.md)
- [day7-post-landing-audit-and-rerank.md](./artifacts/day7-post-landing-audit-and-rerank.md)
- [day8-compressed-path-ownership-design.md](./artifacts/day8-compressed-path-ownership-design.md)
- [day9-compressed-path-ownership-batch.md](./artifacts/day9-compressed-path-ownership-batch.md)
- [day10-public-contract-and-example-adoption-design.md](./artifacts/day10-public-contract-and-example-adoption-design.md)
- [day11-public-contract-and-example-adoption-batch.md](./artifacts/day11-public-contract-and-example-adoption-batch.md)
- [day12-regression-expansion-and-build-alignment.md](./artifacts/day12-regression-expansion-and-build-alignment.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom-Line Closeout

Sprint 72 succeeded because it converted the strongest first-phase
product-model contradictions into explicit, validated ownership rules without
widening into a broad rewrite.

The branch now closes with:

- one cleaner direct-workflow matrix-shell boundary
- one cleaner Cholesky CSC publish-back boundary
- one clearer header-level ownership story for one-shot compatibility versus repeated-run factor ownership
- one explicit proof-owner alignment around the landed boundaries
- one validated close state from the strongest reviewed baseline
- one ranked carry-forward queue for Sprint 73 and later Epic 7 work

That is the right Sprint 72 outcome: Sprint 73 can now continue product-model
convergence from a cleaner, proven first-phase package instead of re-solving
the same ownership contradictions.
