# Sprint 69 Working Notes

## Day 1 - Scope Audit & Public Surface Baseline Setup

### Goal

Freeze the Sprint 69 starting point before implementation work begins by
reconfirming the inherited Sprint 68 contract, the preserved reviewed
baseline, the strongest live public-surface and Epic-closeout hotspots, and
the most important docs/header/example/benchmark/project surfaces the sprint
will touch next.

### Actions

1. Re-read the Sprint 69 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 68 retrospective, and
   the Sprint 68 Day 14 closeout artifact.
2. Re-read the landed Sprint 69 plan and fixed the bounded workstreams that
   the sprint should actually carry:
   - public surface audit
   - docs/examples productization
   - cross-surface compatibility sweep
   - full validation
   - Epic 6 summary and handoff
   - project-level residual finalization
3. Reconfirmed the strongest reviewed baseline surfaces:
   - `make quality-review-full`
   - `make -n quality-review-full`
4. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Measured the strongest likely Sprint 69 touch surfaces directly from the
   live tree across:
   - maintained public product surfaces
   - public header/reference surfaces
   - strongest proof/adoption/reporting surfaces
   - project-level planning and closeout surfaces

### Findings

#### 1. Sprint 69 starts from the Sprint 68 giant-test and assurance close, not from renewed subsystem work

Sprint 68 already landed the last bounded giant-test and second-layer
assurance package Epic 6 still needed:

- first-wave `test_chol_csc` maintainability relief
- stronger large-`n` CSC-backed Cholesky public-path oracle coverage
- bounded seeded lifecycle property follow-through
- docs/examples/benchmarks/test ownership alignment
- tighter platform-confidence wording for the reviewed Windows subset

That means Sprint 69 is not reopening:

- backend abstraction or build-option work
- benchmark-governance redesign
- packaging/ABI/platform convergence as a primary implementation target
- large-source or giant-test decomposition as the main story

Interpretation:

- Sprint 69 is the first Epic 6 sprint centered primarily on final integrated
  public product closure and epic-level handoff
- implementation files are now support surfaces only where a touched public
  surface or final compatibility contradiction truly proves they must move

#### 2. The strongest local reviewed baseline remains the authoritative Sprint 69 starting point

The maintained Day 1 truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 69 inherits the exact same reviewed baseline story as the Sprint 68
  close
- public-surface and Epic-closeout work does not get a weaker truth surface
  just because much of the sprint is docs/integration oriented

#### 3. The highest-value Sprint 69 problem is concentrated in final public-surface reconciliation, not in another isolated feature lane

The live repo shows the strongest remaining pressure in:

- top-level product and adoption surfaces
- benchmark/example/test ownership wording
- public header/reference interpretation
- maintainer and project-level residual-story alignment
- final cross-surface truthfulness around what is taught, what is proved, and
  what is merely carried as context

The project-plan scope therefore reduces cleanly to:

1. public surface audit
2. docs/examples productization
3. cross-surface compatibility sweep
4. full validation
5. Epic 6 summary and handoff
6. project-level residual finalization

Interpretation:

- Sprint 69 should not pretend every remaining Epic 6 surface is an equal
  target
- the highest-value work is concentrated in the final public-story seams where
  docs, examples, benchmarks, headers, tests, and project-level summary
  artifacts still need one integrated reading

#### 4. The strongest live Sprint 69 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- maintained public product surfaces:
  - `README.md` = `1034`
  - `docs/tutorial.md` = `477`
  - `examples/README.md` = `161`
  - `benchmarks/README.md` = `356`
  - `docs/maintainer_guide.md` = `578`
- likely public header/reference surfaces:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `232`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest proof/adoption/reporting support surfaces:
  - `tests/test_integration.c` = `2411`
  - `tests/test_chol_csc.c` = `4608`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_fuzz.c` = `651`
  - `tests/test_framework_optin.c` = `85`
  - `examples/example_analysis.c` = `210`
  - `examples/example_basic_solve.c` = `110`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`
- project-level closeout surfaces:
  - `docs/planning/EPIC_6/PROJECT_PLAN.md` = `344`

Interpretation:

- the strongest remaining Epic 6 pressure is concentrated in a smaller set of
  permanent public surfaces plus the proof/adoption/reporting surfaces they
  reference
- Sprint 69 should start by reranking those cross-surface seams, not by
  inventing another subsystem sprint inside the closeout

#### 5. The Day 1 non-goal fence is now explicit before deeper audit begins

Sprint 69 Day 1 confirms the following non-goals:

- no fake product simplification that weakens the maintained truthfulness
  contract
- no broad implementation work disguised as public-surface cleanup
- no inflated cross-platform confidence story beyond reviewed evidence
- no reopening settled Sprint 60-68 seams unless a touched public surface
  proves it is necessary
- no broad style-only docs churn disconnected from real product-story
  contradictions
- no fake “Epic closeout” that skips the measured final validation baseline

### Day 1 Close

Sprint 69 now starts from one explicit public-surface and Epic-closeout
baseline:

- the Sprint 68 giant-test and assurance close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor is re-established locally at `53`
- the broad Epic 6 closeout claim has already narrowed to public audit,
  docs/examples productization, cross-surface compatibility, validation,
  Epic 6 handoff, and project-level residual finalization
- the next step is to validate that live rerun and truthfulness contract
  precisely before writing the bounded public-surface audit follow-through
