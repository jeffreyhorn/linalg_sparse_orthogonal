# Sprint 50 Working Notes

## Day 1

**Objective:** Turn the Sprint 50 project-plan scope plus the Epic 5
review/todo and Epic 4 inherited closeout contract into a concrete
direct-solver lifecycle starting point by confirming the preserved reviewed
baseline, naming the Sprint 50 workstreams explicitly, and defining the
authoritative direct-solver public-surface, analysis/refactor, benchmark,
example, and validation inputs before lifecycle API design begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 50 project-plan source and the new sprint plan:
   - `sed -n '1,220p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
3. Re-read the Epic 5 review and remediation todo:
   - `sed -n '1,220p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
4. Re-read the inherited Epic 4 closeout / residual baseline:
   - `sed -n '1,240p' docs/planning/EPIC_4/EPIC_4_RETROSPECTIVE.md`
5. Refresh one recent Day 1 artifact and working-notes pattern:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_49/artifacts/day1-scope-and-lifecycle-api-baseline.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_49/WORKING_NOTES.md`
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_49/artifacts/day1-authoritative-inputs.txt`
6. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
7. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
8. Measure the live direct-solver public-header, implementation, example,
   benchmark, and regression hotspot sizes:
   - `wc -l include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_iterative.h include/sparse_eigs.h src/sparse_chol_csc.c src/sparse_ldlt_csc.c examples/example_analysis.c benchmarks/bench_refactor.c benchmarks/bench_refactor_csc.c tests/test_chol_csc.c tests/test_ldlt_csc.c tests/test_etree.c README.md docs/tutorial.md`
9. Re-read the main public analysis/refactor precedent and direct-solver public
   lifecycle/state surfaces:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '220,420p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_ldlt.h`
   - `sed -n '1,220p' README.md`
   - `sed -n '1,220p' examples/example_analysis.c`

### Day 1 Findings

#### 1. Sprint 50 starts from a preserved Epic 4 closeout baseline, not from lifecycle-baseline repair work

The inherited starting contract remains explicit and stable:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- Epic 4 already closed the internal repeated-run and documentation-ownership
  groundwork:
  - iterative/eigensolver repeated-run handles exist publicly in bounded form
  - the maintainer-guide / README boundary is already in place
  - the review-driven residual queue is explicit in `EPIC_4_RETROSPECTIVE.md`

Interpretation:

- Sprint 50 is not a validation-recovery sprint
- Sprint 50 is a direct-solver lifecycle design sprint on top of a preserved
  reviewed baseline

#### 2. The key Sprint 50 asymmetry is now precise: the direct-solver side still lacks the explicit lifecycle clarity already present elsewhere

The repo already has one public reusable-lifecycle precedent:

- `include/sparse_analysis.h`
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_refactor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_factor_free(...)`

But the direct-solver one-shot surfaces remain the dominant public usage story:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Interpretation:

- Sprint 50 does not need to invent lifecycle language from scratch
- it needs to reconcile the existing analysis/factor/refactor precedent with
  the still-dominant one-shot direct-solver caller model

#### 3. The main direct-solver tradeoff is still the compatibility-facing mutable `SparseMatrix` / in-place factor path

The live public direct-solver docs still show the same core tradeoff called out
by the Epic 5 review:

- LU remains in-place on a copied `SparseMatrix`
- Cholesky usage is still taught primarily as a copied/in-place factor path
- LDL^T already factors into a separate output struct, but its broader public
  lifecycle relationship to `sparse_analysis_t` is not the dominant caller
  story
- `example_analysis.c` demonstrates the clearest explicit direct repeated-run
  workflow in the repo today

Interpretation:

- Sprint 50’s design target should center on making the direct repeated
  workflow explicit without breaking the one-shot compatibility path
- the analysis/factor example is already a strong precedent and should be
  treated as such

#### 4. The direct Sprint 50 hotspots are already explicit and concentrated

The live Day 1 sizes make the main public and implementation surfaces clear:

- public headers:
  - `include/sparse_analysis.h` = `334`
  - `include/sparse_lu.h` = `327`
  - `include/sparse_cholesky.h` = `191`
  - `include/sparse_ldlt.h` = `310`
- supporting broader public-surface context:
  - `include/sparse_iterative.h` = `718`
  - `include/sparse_eigs.h` = `680`
- main direct-solver implementation hotspots relevant to Sprint 50 follow-on
  planning:
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_ldlt_csc.c` = `2723`
- strongest direct repeated-workflow support surfaces:
  - `examples/example_analysis.c` = `191`
  - `benchmarks/bench_refactor.c` = `159`
  - `benchmarks/bench_refactor_csc.c` = `388`
- strongest direct-solver regression concentrations:
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_ldlt_csc.c` = `3637`
  - `tests/test_etree.c` = `2962`

Interpretation:

- Sprint 50 is correctly focused on lifecycle design, not on code decomposition
  yet
- but the largest later implementation and regression surfaces are already easy
  to name before Sprint 51 begins

#### 5. The direct analysis/factor/refactor path is real, but it is still documented as a partially optimized bridge rather than the default public direct lifecycle

The most important Day 1 wording in `include/sparse_analysis.h` remains
explicit:

- the symbolic structure computed by `sparse_analyze()` is “available for
  future optimizations”
- it is “not currently used to bypass internal symbolic work” in the delegated
  factorization routines

Interpretation:

- Sprint 50 should treat this as a first-class design input
- the Epic 5 lifecycle model must decide whether to extend this public bridge,
  wrap it differently, or expose complementary lifecycle affordances around it
- Day 1 confirms that the main direct-solver problem is integration clarity,
  not absence of any lifecycle precedent

#### 6. The docs already contain the split Sprint 50 must preserve

The current documentation boundary is useful and should not be reopened:

- `README.md` remains the user/operator entry point
- `docs/maintainer_guide.md` remains the maintainer-policy home
- `example_analysis.c` provides the strongest shipped direct repeated-workflow
  illustration

Interpretation:

- Sprint 50 should define the direct-solver lifecycle target cleanly enough
  that later docs can explain it without redistributing policy again
- migration guidance belongs after the contract is designed, not before

#### 7. Sprint 50 is a design sprint, not an implementation sprint

The current repo state and Sprint 50 plan together make the intended order
clear:

1. baseline and validation recheck
2. direct public-surface inventory
3. precedent inventory
4. lifecycle gap analysis
5. bounded public lifecycle design
6. explicit non-goals and implementation landing plan

Interpretation:

- Sprint 50 should not spend time acting like Sprint 51
- the correct Day 1 close is a clean design baseline and authoritative-input
  package
