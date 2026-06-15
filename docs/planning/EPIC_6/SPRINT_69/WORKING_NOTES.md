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

## Day 2 - Validation Baseline & Final Rerun Recheck

### Goal

Reconfirm the reviewed baseline and the targeted final rerun set that Sprint
69 public-surface, compatibility, and Epic-closeout work must preserve before
any implementation or reconciliation work lands.

### Actions

1. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
2. Re-read the reviewed baseline wrapper surface:
   - `make -n quality-review-full`
3. Reconfirmed the authoritative validation split for:
   - bounded `*.c` / `*.h` days
   - substantial cross-surface integration or closeout work
   - docs-only days
4. Rechecked build-tree availability of the most relevant Sprint 69 proof and
   regression surfaces:
   - integration and family-local proof owners
   - representative examples
   - maintained benchmark/reporting surfaces
   - assurance support surfaces that still define the final Epic 6 story
5. Reconfirmed the strongest likely Sprint 69 touched-surface classes from the
   live branch state after the Day 1 baseline.

### Findings

#### 1. The strongest reviewed baseline is unchanged at Sprint 69 start

The strongest local reviewed baseline is still:

- `make quality-review-full`

The maintained reviewed CMake parity anchor is still:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 69 inherits the same reviewed-baseline authority split as Sprint 68
- final public-surface and Epic-closeout work is not allowed to drift onto a
  weaker local truth surface

#### 2. The authoritative validation split is now explicit before code or closeout work begins

The Day 2 validation contract is now fixed as:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial cross-surface integration, compatibility,
  or closeout work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

Interpretation:

- this matches the maintained repo contract rather than inventing a lighter
  Sprint-69-specific rule set
- the final sprint still closes from the same strongest baseline as the rest
  of Epic 6

#### 3. The high-signal Sprint 69 rerun set is now fixed around the actual final product and closeout-risk surface

The targeted rerun set present in `build/` is:

- cross-family/orchestration and public-proof owners:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_reorder_nd`
- final assurance support surfaces:
  - `./build/test_fuzz`
  - `./build/test_framework_optin`
  - `./build/test_iterative`
  - `./build/test_eigs`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

Interpretation:

- this is the right Day 2 shape because it covers the actual Sprint 69 lanes
  without turning the sprint into a repo-wide rerun mandate on every day
- the rerun set is centered on the surfaces most likely to define the final
  Epic 6 public story, proof ownership, and measured closeout baseline

#### 4. Sprint 69’s likely touched-surface class is narrower than the full reviewed suite, even though the closeout is repo-wide in meaning

Day 2 confirms the most likely Sprint 69 touched lane is concentrated in:

- maintained public product surfaces
- public header/reference interpretation surfaces
- proof/adoption/reporting surfaces only where ownership wording or final
  compatibility interpretation truly moves
- project-level planning and residual-summary surfaces only where the landed
  final story requires it

Interpretation:

- the sprint should stay bounded to the highest-value final product and
  closeout seams rather than widening into generic cleanup everywhere
- even the final Epic 6 sprint still needs a sharp touched-surface fence

### Day 2 Close

Sprint 69 now has one explicit validation contract before deeper audit and
implementation work:

- strongest local reviewed baseline is still `make quality-review-full`
- reviewed CMake parity remains explicit at `53`
- bounded code-touching days must run `make format`, `make lint`, and
  `make test`
- substantial integration or closeout work should default to
  `make quality-review-full`
- the high-signal Sprint 69 rerun set is fixed around the actual final
  product, proof-owner, example, and maintained benchmark/report surfaces
  present in `build/`

## Day 3 - Public Surface Audit I

### Goal

Re-rank the final public product surfaces by contradiction density, adoption
value, and Epic 6 closeout payoff before any final simplification work lands.

### Actions

1. Audited the highest-value maintained public product surfaces:
   - `README.md`
   - `docs/tutorial.md`
   - `examples/README.md`
   - `benchmarks/README.md`
   - `docs/maintainer_guide.md`
2. Re-read the strongest public reference and interpretation surfaces:
   - `include/sparse_cholesky.h`
   - `include/sparse_analysis.h`
   - `include/sparse_iterative.h`
   - `include/sparse_eigs.h`
3. Measured current size concentration across those public surfaces.
4. Ran targeted terminology scans for:
   - example / benchmark / test ownership wording
   - workflow / proof / property / oracle wording
   - install / platform / canonical-report wording
5. Ranked the live contradiction set by:
   - public product-story importance
   - cross-surface duplication density
   - risk of user or maintainer misread
   - likely simplification payoff in Sprint 69

### Findings

#### 1. The strongest remaining Sprint 69 problem is duplicated product-story ownership, not missing coverage

The live repo already documents the final Epic 6 state in many places. The
main residual pressure is that the same story is still told across too many
surfaces:

- workflow choice and repeated-run adoption
- examples vs benchmarks vs tests ownership
- benchmark canonical/reporting interpretation
- platform/install confidence limits
- maintainer-policy vs user-facing explanation boundaries

Interpretation:

- Sprint 69 should optimize for simplification and sharper authority splits,
  not for adding more explanatory surface area
- the strongest risk is not a missing statement; it is drift or reader
  overload from repeated parallel explanations

#### 2. README is the strongest first target because it carries the densest mix of product narrative, proof ownership, platform truth, and install story

`README.md` remains the strongest Day 3 hotspot because it combines:

- top-level product narrative
- workflow-choice guidance
- tests/examples/benchmarks ownership summary
- benchmark canonical/reporting summary
- quality/platform truth summary
- install/package summary

Interpretation:

- this is the highest-value first landing because small README clarifications
  can remove duplicated explanation pressure from several adjacent surfaces
- the risk is not that README is wrong in one obvious place; it is that it is
  now the densest multi-role surface in the final Epic 6 state

#### 3. Tutorial is the strongest second target because it still repeats product-story framing that now belongs more compactly elsewhere

`docs/tutorial.md` is the strongest second hotspot because it still carries:

- workflow-choice framing
- examples vs benchmarks interpretation
- repeated-run direct-path handoff language
- public-product explanation that partly overlaps with README and
  `examples/README.md`

Interpretation:

- tutorial is still valuable, but its strongest residual risk is explanatory
  overlap rather than missing usage content
- it reads like the best second landing once README has been simplified first

#### 4. The maintainer guide is the strongest policy authority surface, but it should not be the first simplification target by volume alone

`docs/maintainer_guide.md` is large and policy-heavy, but it is already the
right home for:

- documentation ownership interpretation
- benchmark-governance interpretation
- packaging/platform residual interpretation
- proof-ownership policy

Interpretation:

- it remains a crucial support surface, but the main Sprint 69 pressure is not
  “make the maintainer guide smaller” in isolation
- the first win is to reduce what README/tutorial/examples/benchmarks still
  need to say because the maintainer guide already owns the policy layer

#### 5. Public headers are real final product surfaces, but most are weaker first targets than the public docs

The live header ranking is now clearer:

- strongest header target:
  - `include/sparse_cholesky.h`
- real but lower-priority support headers:
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

Why `include/sparse_cholesky.h` stands out:

- it carries the densest public-path explanation tied to the final Epic 6
  product story:
  - transparent CSC dispatch
  - benchmark/test reference notes
  - backend-contract error semantics
  - one-shot vs repeated-run lifecycle interpretation

Interpretation:

- `include/sparse_cholesky.h` is the best header-side support candidate for
  Sprint 69
- the other large headers are substantial, but their residual pressure is more
  reference breadth than final product-story contradiction

#### 6. Examples and benchmark READMEs are important support surfaces, but they are weaker first targets than README/tutorial

`examples/README.md` and `benchmarks/README.md` are already relatively sharp:

- examples README stays local to adoption entry points
- benchmarks README stays local to benchmark categories, schemas, and
  ownership limits

Their main remaining risk is support-side drift relative to README/tutorial,
not that they need broad redesign first.

Interpretation:

- they are likely part of the first touched set
- they are not the best first design center for Sprint 69

### Day 3 Close

Sprint 69’s broad “public product surface finalization” claim is now reduced
to one ranked live seam map:

- strongest first target:
  - `README.md`
- strongest second target:
  - `docs/tutorial.md`
- strongest policy/support surface:
  - `docs/maintainer_guide.md`
- strongest header-side support candidate:
  - `include/sparse_cholesky.h`
- important support surfaces, but weaker first design centers:
  - `examples/README.md`
  - `benchmarks/README.md`
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
