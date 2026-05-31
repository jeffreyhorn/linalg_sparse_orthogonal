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

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 50 must preserve, then define the smallest authoritative
validation boundary for this design sprint versus the broader rerun set later
direct-solver implementation days will need once `*.c` / `*.h` edits begin.

### Commands Run

1. Re-read the Sprint 50 Day 2 plan item and existing working-notes context:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the direct-solver public lifecycle/state surfaces that will later
   define the touched implementation set:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,260p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`
3. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
5. Re-read the live quality-contract wording sources:
   - `rg -n "quality-review-full|deadcode-check|reviewed baseline|strongest local reviewed baseline|quality-review-cmake" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
6. Reconfirm the direct-solver example, benchmark, and regression binaries that
   later API-implementation batches are most likely to touch:
   - `rg -n "example_analysis|bench_refactor|bench_refactor_csc|test_chol_csc|test_ldlt_csc|test_etree|test_cholesky|test_ldlt" Makefile CMakeLists.txt tests benchmarks examples`

### Day 2 Findings

#### 1. The strongest local reviewed baseline remains explicit, stable, and already uses the exact wording Sprint 50 should preserve

The maintained wrapper surface still says exactly:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review'
  or 'make quality-review-cmake'`

The README and maintainer guide stay aligned with that same language:

- `README.md` calls `make quality-review-full` the strongest local reviewed
  baseline
- `docs/maintainer_guide.md` treats that phrasing as the authoritative local
  close state

Interpretation:

- Sprint 50 should keep using the exact “strongest local reviewed baseline”
  language
- later direct-solver lifecycle work should not invent narrower wording unless
  it is making a genuinely narrower claim

#### 2. The reviewed CMake parity anchor remains exact and is still the main truthfulness backstop for later public API work

The maintained reviewed CMake path still resolves cleanly to:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The concrete reviewed suite list still includes the direct-solver and
supporting structural tests Sprint 50 is most likely to care about later:

- `test_cholesky`
- `test_ldlt`
- `test_etree`
- `test_chol_csc`
- `test_ldlt_csc`

Interpretation:

- later Sprint 50+ implementation batches should treat the exact `53` count as
  a truthfulness anchor, not as a fuzzy “about the same” metric
- the direct-solver lifecycle work must preserve both the count and the
  Makefile/CMake parity contract

#### 3. The reviewed quality contract is intentionally layered, and Day 2 fixes the authority split Sprint 50 should use

The live repo still divides authority cleanly:

- `make quality-review-full`:
  - strongest local reviewed baseline
- `make quality-review`:
  - reviewed Makefile local path
  - `format-check + lint + test + deadcode-check`
- `make quality-review-cmake`:
  - reviewed CMake parity path with full suite execution
- `make deadcode-check`:
  - report-completeness gate, not a zero-findings or removal-ready gate

Interpretation:

- Sprint 50 Day 2 should use this same split rather than restating the whole
  quality contract ad hoc
- later direct-solver public API days that touch `*.c` / `*.h` should rerun the
  full required gate and then escalate to `make quality-review-full` for
  substantial lifecycle-surface batches

#### 4. Sprint 50 as a design sprint has a much smaller authoritative validation requirement than the later implementation sprints

For the design-only days, the smallest authoritative validation set is:

- preserve the wording and meaning of:
  - `make quality-review-full`
  - reviewed CMake parity
  - `53` tests
- rerun targeted sanity checks only when docs or planning artifacts change

For later implementation days that touch `*.c` / `*.h`, the minimum
authoritative gate should remain:

- `make format`
- `make lint`
- `make test`

And for substantial public lifecycle API batches, the stronger default should
remain:

- `make quality-review-full`

Interpretation:

- Day 2 fixes the validation boundary before the design artifacts get deeper
- Sprint 50 should not blur docs-only design work with code-touch validation
  claims

#### 5. The later direct-solver touched-surface follow-on list is already clear enough to freeze before API design begins

The most likely later direct-solver follow-on binaries and regression surfaces
are explicit in the live build/test graph:

- examples:
  - `./build/example_analysis`
- benchmarks:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
- regression tests:
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_etree`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`

Interpretation:

- Sprint 50 later implementation days should treat these as the highest-signal
  touched follow-ons when public direct lifecycle work lands
- broader full-suite and CMake parity execution still belongs to the
  authoritative reviewed baseline, not to an improvised subset

#### 6. The direct-solver validation story is now bounded enough that Day 3 can stay focused on public-surface inventory

Day 2 resolves the validation ambiguity that could otherwise distort the design
work:

- strongest local reviewed baseline meaning remains fixed
- reviewed CMake test-count truth remains fixed at `53`
- the design sprint has a small sanity-check boundary
- later implementation follow-ons are already named

Interpretation:

- Day 3 can now inventory the direct-solver public lifecycle surface without
  also trying to rediscover the quality contract
- the main remaining Sprint 50 work is design narrowing, not validation-policy
  argument

## Day 3

**Objective:** Re-map the live direct-solver public API surface across LU,
Cholesky, LDL^T, the analysis/refactor bridge, and QR-as-contrast so Sprint 50
can reduce the lifecycle problem to named public seams instead of a generic
state-model complaint.

### Commands Run

1. Re-read the Sprint 50 Day 3 plan item and the current sprint notes:
   - `sed -n '60,180p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `sed -n '1,420p' docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the main direct-solver public lifecycle/state headers:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,240p' include/sparse_cholesky.h`
   - `sed -n '1,320p' include/sparse_ldlt.h`
3. Re-read the public contrast surface where factor ownership is already
   explicit but refactor/reuse is not:
   - `sed -n '1,260p' include/sparse_qr.h`
4. Re-read the strongest shipped direct repeated-workflow caller and the main
   refactor benchmarks:
   - `sed -n '1,260p' examples/example_analysis.c`
   - `sed -n '1,240p' benchmarks/bench_refactor.c`
   - `sed -n '1,260p' benchmarks/bench_refactor_csc.c`
5. Re-read the public README and local example/benchmark README language around
   repeated-run, one-shot, and analyze/refactor workflows:
   - `sed -n '200,340p' README.md`
   - `sed -n '1,120p' examples/README.md`
   - `sed -n '50,90p' benchmarks/README.md`
   - `rg -n "analysis|refactor|Repeated-Run|one-shot|lifecycle" README.md examples/README.md benchmarks/README.md include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h include/sparse_qr.h`

### Day 3 Findings

#### 1. The live public direct-solver surface reduces cleanly to five lifecycle seam classes rather than one generic “state model” problem

The current public surface now maps cleanly to:

- matrix-mutating one-shot factor-and-solve:
  - `sparse_lu_factor(...)`
  - `sparse_lu_factor_opts(...)`
  - `sparse_lu_solve(...)`
  - `sparse_cholesky_factor(...)`
  - `sparse_cholesky_factor_opts(...)`
  - `sparse_cholesky_solve(...)`
- factor-object one-shot lifecycle:
  - `sparse_ldlt_factor(...)`
  - `sparse_ldlt_factor_opts(...)`
  - `sparse_ldlt_solve(...)`
  - `sparse_ldlt_free(...)`
- explicit analysis/factor/refactor bridge:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_refactor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_analysis_free(...)`
  - `sparse_factor_free(...)`
- backend/reorder/telemetry side paths that affect implementation choice more
  than lifecycle ownership:
  - `sparse_chol_backend_t`
  - `sparse_ldlt_backend_t`
  - `used_csc_path`
  - reorder fields in the one-shot opts structs
- comparison surface for lifecycle expectations, not a direct Sprint 50 target:
  - `sparse_qr_t`
  - `sparse_qr_factor(...)`
  - `sparse_qr_factor_opts(...)`
  - `sparse_qr_free(...)`

Interpretation:

- Sprint 50’s lifecycle problem is not “direct solvers have no state model”
- it is that the repo already exposes multiple different state models, and the
  explicit repeated direct workflow is not yet the dominant public story

#### 2. The current direct-solver workflows split into three real caller buckets plus two later verification buckets

The public direct-solver caller stories group into:

- one-shot matrix-copy then in-place factor:
  - LU
  - Cholesky
- one-shot factor object:
  - LDL^T
  - QR as a lifecycle contrast
- analyze-once / factor-many:
  - `sparse_analysis_t` + `sparse_factors_t`

And the shipped verification/support caller stories sit separately:

- example surface:
  - `example_analysis.c`
- benchmark surface:
  - `bench_refactor.c`
  - `bench_refactor_csc.c`

Interpretation:

- the first public lifecycle landing should target the real API buckets, not
  the example/benchmark mirror surfaces
- Day 3 now separates caller contract from proof/demo surface

#### 3. The strongest hidden-state / documentation-discipline gap remains concentrated in the matrix-mutating one-shot paths

LU and Cholesky still rely most heavily on caller discipline:

- examples teach `sparse_copy()` before factorization
- factor state and permutations live inside the mutated `SparseMatrix`
- preserving the original matrix view is mainly a documentation rule, not an
  explicit lifecycle object boundary

LDL^T improves that story materially:

- factors live in `sparse_ldlt_t`
- the input matrix is not modified
- solve/free ownership is explicit

But LDL^T still does not make the analysis/refactor bridge the dominant direct
public caller story.

Interpretation:

- the highest-value Sprint 50 lifecycle tension is the gap between
  matrix-mutating one-shot compatibility and explicit caller-owned lifecycle
- LDL^T and QR show that explicit factor containers are already acceptable repo
  patterns

#### 4. The explicit analysis/refactor bridge is already the closest thing to the target public lifecycle, but it still reads as a partial bridge instead of the default direct contract

The analysis path already exposes the strongest direct repeated-run contract:

- analyze once
- factor numerically
- solve
- refactor with same sparsity pattern
- solve again
- free explicit owned objects

But the public docs and comments still frame it as:

- an analyze/refactor workflow
- a future-optimization bridge
- a specialist path alongside delegated one-shot factorization routines

Interpretation:

- Sprint 50’s first design target should anchor on `sparse_analysis.h`
- the key question is how to make this bridge relate cleanly to LU, Cholesky,
  and LDL^T without pretending the one-shot APIs are going away

#### 5. QR matters to Sprint 50 mainly as a boundary-setting contrast surface

QR is not a direct solver, but it is still useful Day 3 contrast:

- it already uses a caller-owned factor object (`sparse_qr_t`)
- it has explicit factor / solve / free ownership
- it does not expose a public refactor/reuse workflow
- its reorder policy is direct and local to QR’s own structural model

Interpretation:

- Sprint 50 should not pull QR into the first implementation target set
- but QR confirms that explicit factor-object lifecycle ownership is already a
  normalized public pattern in the repo

#### 6. The true first landing targets are now narrower than the full caller/support surface

The strongest first public landing targets are:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

The strongest later compatibility and verification surfaces are:

- `README.md`
- `examples/example_analysis.c`
- `examples/README.md`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/README.md`
- `include/sparse_qr.h`

Interpretation:

- Day 3 now fixes the real target boundary before Day 4 precedent inventory and
  Day 5 gap analysis begin
- the lifecycle landing should start at the direct public headers, not at the
  example/benchmark/docs perimeter

## Day 4

**Objective:** Map the existing lifecycle precedents Sprint 50 can reuse,
separate them from the direct-solver-specific structural seams that should stay
private, and fix a bounded “borrow vs keep direct-specific” rule set before the
gap-analysis day.

### Commands Run

1. Re-read the Sprint 50 Day 4 plan item and the current working-notes state:
   - `sed -n '110,230p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `sed -n '1,520p' docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the direct public lifecycle precedent in full:
   - `sed -n '1,340p' include/sparse_analysis.h`
3. Refresh the Epic 4 explicit handle precedents for lifecycle shape and
   terminology:
   - `sed -n '180,340p' include/sparse_iterative.h`
   - `sed -n '500,680p' include/sparse_eigs.h`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_49/artifacts/day3-public-lifecycle-api-design.md`
4. Re-read the direct implementation bridge seams most relevant to later public
   lifecycle exposure:
   - `rg -n "sparse_factor_numeric|sparse_refactor_numeric|chol_csc_from_sparse_with_analysis|ldlt_csc_from_sparse_with_analysis|sparse_analyze\\(|sparse_factor_solve\\(" src include`
   - `sed -n '1,340p' src/sparse_analysis.c`
   - `sed -n '340,760p' src/sparse_analysis.c`
   - `sed -n '1,260p' src/sparse_chol_csc_internal.h`
   - `sed -n '1,260p' src/sparse_ldlt_csc_internal.h`

### Day 4 Findings

#### 1. Sprint 50 already has two different kinds of lifecycle precedent, and they should be reused differently

The current repo gives Sprint 50:

- a direct public lifecycle precedent:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_refactor_numeric(...)`
  - `sparse_factor_solve(...)`
- a generic public repeated-run handle precedent:
  - `sparse_iter_handle_t`
  - `sparse_iter_handle_init(...)`
  - `sparse_iter_handle_prepare_*`
  - `sparse_solve_*_with_handle(...)`
  - `sparse_iter_handle_free(...)`
  - `sparse_eigs_handle_t`
  - `sparse_eigs_handle_prepare(...)`
  - `sparse_eigs_sym_with_handle(...)`

Interpretation:

- Sprint 50 does not need to invent public lifecycle shape from scratch
- but it also should not mirror Sprint 49 handles mechanically, because the
  direct-solver side already has a public analyze/factor/refactor bridge

#### 2. `sparse_analysis.h` is the main direct-solver design anchor because it already matches the domain’s real repeated-run workflow

The direct public bridge already captures the right solver-domain sequence:

1. analyze once
2. factor numerically
3. solve
4. refactor on new values with the same sparsity pattern
5. solve again
6. free explicit owned state

The implementation side reinforces that this bridge is real:

- `sparse_factor_numeric(...)` builds a working copy, dispatches by direct
  factor type, and owns factor payload handoff
- `sparse_factor_solve(...)` applies analysis permutation state and delegates
  to the factor-specific solve path
- `sparse_refactor_numeric(...)` preserves old factors on error and rewrites
  through the same bridge

Interpretation:

- Sprint 50 should borrow its direct lifecycle sequence primarily from
  `sparse_analysis.h`
- later public direct lifecycle design should feel like an extension or
  centering of this bridge, not a replacement vocabulary borrowed from a
  different solver family

#### 3. The iterative/eigensolver handle work is still valuable precedent, but mostly for handle discipline and public-contract rules

The Epic 4 handle surfaces contribute the generic public repeated-run rules:

- zero-init or init helper
- explicit prepare step
- repeated run through one-shot-compatible option/result surfaces
- free safe on zeroed state
- reuse preserves allocation capacity, not old numerical state

Interpretation:

- Sprint 50 should borrow:
  - lifecycle-centric wording
  - initialize / prepare / run / free discipline
  - “one-shot remains first-class” compatibility framing
- Sprint 50 should not borrow:
  - opaque-workspace-first framing as the primary direct design anchor
  - a purely dimension/workspace-centric model detached from analysis/refactor

#### 4. The internal direct-solver structural seams are implementation precedents, not public-shape precedents

The main direct implementation bridge seams are now explicit:

- `src/sparse_analysis.c`
  - permutation-aware working-copy construction
  - factor payload ownership transfer
  - factor-type dispatch
  - solve dispatch
  - safe refactor overwrite semantics
- `src/sparse_chol_csc_internal.h`
  - `chol_csc_from_sparse_with_analysis(...)`
  - symbolic-aware CSC preallocation
  - analysis-driven CSC working-format path
- `src/sparse_ldlt_csc_internal.h`
  - `ldlt_csc_from_sparse_with_analysis(...)`
  - pre-pass / pre-permuted indefinite path rules
  - CSC-side factor object writeback and row-adjacency scaffolding

Interpretation:

- Sprint 50 should treat these as structural implementation precedents that
  later public lifecycle work can route through
- Sprint 50 should not expose these names, layouts, or CSC-specific sequencing
  as the public API target

#### 5. The main “borrow vs direct-specific” split is now clean enough to guide Day 5

Borrow from existing precedents:

- explicit owned lifecycle objects
- zero-init / init helper safety
- explicit prepare / analyze step before repeated runs
- one-shot API preservation
- reuse preserves setup/capacity, not old numerical state
- free safe on zeroed/empty state

Keep direct-solver-specific:

- factor-type differences between LU / Cholesky / LDL^T
- symbolic-analysis semantics
- same-pattern refactor contract
- permutation / reordered-copy interaction
- CSC/native dispatch and backend telemetry
- mutable-`SparseMatrix` compatibility realities in LU / Cholesky

Interpretation:

- Day 4 now fixes which parts of Sprint 49 are generic pattern and which parts
  are solver-family-specific implementation detail

#### 6. Sprint 50’s public lifecycle target should read as analysis-centric first, handle-centric second

The strongest design signal after Day 4 is:

- generic repeated-run handles are a good public-contract model
- but on the direct-solver side the public repeated-run truth already centers
  on analysis/factor/refactor

Interpretation:

- if Sprint 50 exposes any new public lifecycle layer later, it should compose
  around the analysis/refactor story instead of displacing it
- Day 5 can now analyze gaps against a much narrower and better-grounded
  precedent set

## Day 5

**Objective:** Turn the Day 3 surface inventory and Day 4 precedent map into a
ranked direct-solver lifecycle gap analysis that is explicit about usability,
correctness, efficiency, maintainability, and compatibility tradeoffs before
the API-design days begin.

### Commands Run

1. Re-read the Sprint 50 Day 5 plan item and the current notes:
   - `sed -n '150,280p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `sed -n '1,760p' docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the Epic 5 review and remediation todo to keep the gap framing
   aligned with the project-level queue:
   - `sed -n '1,260p' docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md`
3. Re-read the Day 3 public-surface inventory and Day 4 precedent inventory:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_50/artifacts/day3-direct-solver-public-surface-inventory.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_50/artifacts/day4-lifecycle-precedent-inventory.md`

### Day 5 Findings

#### 1. The highest-value lifecycle gap is still usability: the repeated direct workflow is real but not the dominant public caller story

The repo already supports an explicit repeated direct workflow through:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_refactor_numeric(...)`
- `sparse_factor_solve(...)`

But the dominant public direct-solver mental model is still:

- copy a matrix
- factor it in place or into a family-specific object
- solve
- rely on docs/examples to decide when and how to preserve original state

Gap class:

- usability
- product-shape clarity

Interpretation:

- the biggest problem is no longer missing capability
- it is that the strongest repeated direct workflow is still under-centered and
  therefore easy for callers to miss

#### 2. The strongest correctness-risk gap is still hidden mutable state and caller-discipline dependence in the one-shot LU / Cholesky path

The one-shot LU and Cholesky story still depends most on undocumented-in-code
caller discipline:

- copy before factorization if the original matrix view matters
- understand that reorder/factor state now lives inside the factor-bearing
  `SparseMatrix`
- know which later workflows require identity permutations or unfactored input

Gap class:

- correctness-risk by misuse
- API ergonomics

Interpretation:

- this is not an immediate numerical bug
- it is the strongest remaining “easy to use wrong” public direct-solver seam

#### 3. The strongest efficiency gap is that analysis-driven factor-many is public but still not clearly the default performance story

The analysis/refactor bridge already exists, but Day 1 and the Epic 5 review
still show that it is framed as:

- a bridge
- a partial optimization hook
- a specialist repeated-run path

rather than the obvious public answer to stable-pattern direct solves.

Gap class:

- efficiency
- performance-story clarity

Interpretation:

- the project risks leaving real factor-many efficiency on the table not
  because the mechanism is absent, but because the public contract does not yet
  center it strongly enough

#### 4. The strongest maintainability gap is the mismatch between three public direct lifecycle models that are individually reasonable but not yet reconciled

Current direct public models:

- matrix-mutating one-shot:
  - LU
  - Cholesky
- factor-object one-shot:
  - LDL^T
- explicit analysis/factor/refactor bridge

Each model is defensible alone, but together they create public-shape drift.

Gap class:

- maintainability
- long-term API coherence

Interpretation:

- the repository already has enough direct-solver public surface that leaving
  these models only loosely related will keep generating docs, test, and
  example drift

#### 5. The docs/examples gap is real but secondary: they still over-center the one-shot path because the contract itself is not fully centered yet

Examples and user-facing docs still lean heavily one-shot-first:

- that is intentional for simplicity
- but it also means the repeated direct workflow remains easy to read as
  advanced or specialist rather than as the supported stable-pattern path

Gap class:

- documentation
- migration clarity

Interpretation:

- docs are not the root cause
- they are reflecting an API-centering gap that the later design days need to
  resolve first

#### 6. The main compatibility constraint remains explicit and should stay that way

Sprint 50 still has to preserve:

- one-shot direct public APIs as first-class supported paths
- mutable `SparseMatrix` compatibility behavior where already public
- family-specific factor semantics that are real API differences, not mere
  naming noise

Interpretation:

- Sprint 50 cannot solve its lifecycle gap by pretending the old surface is
  going away
- the correct target is a bounded additive or centering move, not a redesign

#### 7. The smallest credible public direct-lifecycle exposure is now narrow enough to state explicitly

The minimum credible target that would materially improve the system without
reopening broad API churn is:

- make the analysis/factor/refactor lifecycle the explicit repeated direct-run
  public story
- clarify or extend its ownership/lifecycle contract where needed
- keep LU / Cholesky / LDL^T one-shot entries as compatibility-first wrappers
  or peer entry points
- do not expose CSC/native internal layout, backend-specific storage, or broad
  new public solver-family abstractions in Sprint 50

Interpretation:

- Day 5 rules out both extremes:
  - no-op documentation-only cleanup
  - broad new direct-solver framework redesign

#### 8. The ranked gap list is now concrete enough for the Day 6 design batch

Ranked highest to lowest:

1. repeated direct workflow is under-centered publicly
2. hidden mutable matrix-state dependence in one-shot LU / Cholesky
3. factor-many efficiency story is public but not first-class
4. multiple direct lifecycle models remain unreconciled
5. docs/examples still over-center one-shot usage

Interpretation:

- Day 6 should attack the first three directly
- Day 7 can then audit the resulting contract for whether it narrows the last
  two enough without overreaching

## Day 6

**Objective:** Convert the Day 5 ranked gap list into the first bounded public
direct-solver lifecycle contract, with explicit decisions on abstraction
shape, lifecycle stages, naming, and first-model family coverage before the
post-design audit.

### Commands Run

1. Re-read the Day 6 plan item and the current Sprint 50 notes:
   - `sed -n '120,320p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the Day 5 gap-analysis artifact:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_50/artifacts/day5-direct-solver-lifecycle-gap-analysis.md`
3. Re-read the strongest direct public lifecycle anchor and adjacent public
   family headers:
   - `sed -n '1,420p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`
4. Re-read the Epic 4 repeated-run public-lifecycle design shape for generic
   init / prepare / run / reuse / free precedent:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_49/artifacts/day3-public-lifecycle-api-design.md`

### Day 6 Findings

#### 1. The strongest first-pass abstraction shape is a bounded hybrid, but it is analysis-centric rather than handle-centric

Day 6 compared three plausible directions:

- keep only the existing analysis/factor/refactor API and merely document it
  better
- add a brand-new generic public direct handle
- keep the current analysis/factor model as the direct repeated-run core and
  allow only small additive lifecycle clarifications around it

The strongest choice is the third:

- analysis-centric bounded hybrid

Interpretation:

- docs-only centering is too small
- a new generic direct handle would be too broad and duplicative
- the repo already has a meaningful public direct lifecycle through:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_refactor_numeric(...)`
  - `sparse_factor_solve(...)`

#### 2. The public repeated direct lifecycle stages are now concrete enough to name explicitly

Day 6 fixes the target public stages as:

1. initialize / zero
2. analyze / prepare
3. factor
4. solve
5. refactor / reuse
6. free

Interpretation:

- this preserves the direct domain vocabulary already present in
  `sparse_analysis.h`
- it also borrows the clearer lifecycle sequencing discipline proven on the
  iterative/eigensolver side during Epic 4

#### 3. “Prepare” should remain analysis-specific vocabulary, not a generic storage-reserve abstraction

The direct repeated-run side is not primarily a workspace-reserve story.
Its true prepare step is:

- choose factor family
- choose reorder policy
- compute symbolic structure
- establish same-pattern reuse preconditions

Interpretation:

- Sprint 50 should not flatten direct solver preparation into a generic
  “allocate handle buffers” concept
- the analysis step is the real direct public preparation contract

#### 4. The first model must explicitly cover LU, Cholesky, and LDL^T

Day 6 rejects a smaller direct-lifecycle model that would only speak clearly
about one or two factor families.

The first explicit lifecycle coverage set should be:

- LU
- Cholesky
- LDL^T

And it should not try to pull QR in as a first landing target.

Interpretation:

- LU / Cholesky / LDL^T are already the direct families tied to the public
  analysis/factor/refactor bridge
- QR remains a useful contrast surface, but broadening the first contract to
  include QR would widen scope without closing the highest-value gap

#### 5. One-shot direct APIs should remain first-class peer entry points, not be reframed as deprecated leftovers

Day 6 fixes the relationship between the repeated direct lifecycle and the
family-specific one-shot APIs:

- one-shot APIs remain:
  - simple/default caller path
  - compatibility-preserving path
  - first-class supported path
- explicit analysis/factor/refactor remains:
  - stable-pattern repeated-run path
  - factor-many performance path
  - clearer lifecycle path for higher-context callers

Interpretation:

- Sprint 50 should center the repeated-run contract without lying about the
  continued importance of one-shot LU / Cholesky / LDL^T usage

#### 6. The right naming is domain-specific first: analysis and factors, not generic handles

Day 6 explicitly prefers:

- analysis
- factors
- refactor
- repeated direct run

over generic public nouns such as:

- handle
- workspace
- context

Interpretation:

- `sparse_analysis.h` is already the strongest direct precedent
- generic naming would hide direct-solver semantics that callers actually need
  to reason about

#### 7. Reuse semantics are now narrow enough to state cleanly

Public direct reuse should mean:

- preserve symbolic/permutation setup
- reuse the analyzed structure for new values
- replace numeric factor state on success

It should not mean:

- preserve old triangular numeric state as an incremental-update contract
- promise backend-specific CSC/native storage layout
- validate structural compatibility beyond the current caller precondition

Interpretation:

- this is the direct-solver analogue of the Epic 4 repeated-run rule:
  preserve setup investment, not old numerical iteration state

#### 8. Day 6 leaves Day 7 a concrete audit target rather than a generic design backlog

The remaining high-value questions are now narrow:

- whether the analysis-centric shape fully closes the repeated-run centering gap
- what should stay one-shot-first even after the lifecycle story is centered
- whether any tiny additive helper surface is justified later

Interpretation:

- Sprint 50 now has a real first-pass lifecycle contract
- Day 7 can audit a bounded design instead of re-opening architecture search

## Day 7

**Objective:** Audit the Day 6 lifecycle design against the Day 5 ranked gaps
and the inherited Epic 4 compatibility boundary, then separate true Sprint 50
public-contract decisions from Sprint 51 implementation details.

### Commands Run

1. Re-read the Day 7 plan item and the latest Sprint 50 notes:
   - `sed -n '240,320p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `tail -n 220 docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the Day 6 design artifact:
   - `sed -n '1,320p' docs/planning/EPIC_5/SPRINT_50/artifacts/day6-public-direct-solver-lifecycle-api-design-batch1.md`
3. Re-read the strongest current repeated direct-run caller surfaces:
   - `sed -n '1,260p' examples/example_analysis.c`
   - `sed -n '1,260p' benchmarks/bench_refactor.c`
   - `sed -n '1,240p' benchmarks/bench_refactor_csc.c`
   - `sed -n '1,260p' examples/README.md`

### Day 7 Findings

#### 1. The Day 6 design closes the main repeated-run centering gap well enough to keep

The Day 5 highest-ranked issue was that repeated direct workflow was real but
not centered enough publicly.

After the Day 7 audit, the Day 6 model still looks correct:

- explicit analysis/factor/refactor lifecycle
- one-shot compatibility paths preserved
- no unnecessary new generic direct handle

Interpretation:

- no stronger competing abstraction surfaced during audit
- the analysis-centric bounded hybrid remains the right Sprint 50 shape

#### 2. The biggest remaining public question is now relationship wording, not architecture

The highest-value unresolved contract work is no longer:

- whether to add a lifecycle
- whether to introduce a new direct handle

It is now:

- how strongly to frame analysis/factor/refactor as the intended stable-pattern
  repeated-run path
- how to describe one-shot LU / Cholesky / LDL^T as:
  - peer entry points
  - simple/default path
  - or both depending on caller context

Interpretation:

- Sprint 50’s remaining queue is now mostly wording and boundary work rather
  than model selection

#### 3. The one-shot mutable-matrix story should remain explicit rather than being papered over

The audit confirms that Sprint 50 should not try to “solve” the one-shot LU /
Cholesky mutation model by hiding it behind more abstract public wording.

What should stay one-shot-first:

- simple single-solve LU on a copied matrix
- simple single-solve Cholesky on a copied matrix
- small examples whose job is to teach basic public factor-and-solve flow

Interpretation:

- the repeated lifecycle should be centered for stable-pattern repeated work
- the one-shot mutable path should remain visibly supported where it is the
  simpler caller story

#### 4. The internal-only boundary is now concrete enough to protect the Sprint 50 scope

The audit confirms that the following should remain internal-only:

- CSC/native factor storage layout
- analysis-aware CSC helper names
- backend-selection plumbing beyond the existing public option structs
- structural-pattern validation machinery beyond the current caller
  precondition
- generic direct-handle storage abstractions

Interpretation:

- this keeps Sprint 50 focused on the public repeated-run contract rather than
  implementation plumbing

#### 5. The example and benchmark adoption boundary should be selective, not universal

The live caller surfaces make the adoption boundary fairly clear.

Should adopt the final repeated-run story early:

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

Can lag intentionally:

- small one-shot examples
- one-shot example wording in `examples/README.md`
- `benchmarks/bench_refactor_csc.c`

Interpretation:

- the highest-signal repeated-run surfaces should align early
- backend-heavy or intentionally simple one-shot surfaces do not need to be
  pulled forward prematurely

#### 6. The remaining Sprint 50 “must decide” list is now small and public-contract-specific

Day 7 narrows the real remaining Sprint 50 decisions to:

1. exact zero/init/free expectations
2. exact analyze-once / factor-refactor-many wording
3. exact one-shot vs repeated-lifecycle relationship wording
4. exact reuse meaning and non-meaning

Interpretation:

- these are true public-contract questions
- they are now cleanly separated from Sprint 51 implementation details

#### 7. The rest should wait for Sprint 51+

The audit confirms that several questions are real but not Sprint 50 design
work:

- exact header/source implementation shape
- regression-test expansion details
- broad README/tutorial/example rewrites
- benchmark-framework adjustments

Interpretation:

- Sprint 50 stays bounded away from solving implementation in advance
- Day 8 can now finalize the contract without reopening code-shape planning

## Day 8

**Objective:** Finalize the caller-facing direct repeated-run lifecycle
contract: zero/init, analyze/factor/refactor/solve/free semantics, reuse
meaning, struct expectations, and the exact one-shot versus repeated-run
relationship.

### Commands Run

1. Re-read the Day 8 plan item and the latest Sprint 50 notes:
   - `sed -n '280,360p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `tail -n 260 docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the Day 6 design artifact and Day 7 audit:
   - `sed -n '1,320p' docs/planning/EPIC_5/SPRINT_50/artifacts/day6-public-direct-solver-lifecycle-api-design-batch1.md`
   - `sed -n '1,340p' docs/planning/EPIC_5/SPRINT_50/artifacts/day7-post-design-audit.md`
3. Re-read the Epic 4 migration-path wording shape to keep the final
   relationship language aligned with the repo’s recent public-contract style:
   - `sed -n '1,320p' docs/planning/EPIC_4/SPRINT_49/artifacts/day8-migration-path-documentation-batch.md`

### Day 8 Findings

#### 1. Zero-init plus explicit free is the right final lifecycle baseline

Day 7 left open whether Sprint 50 should require only zero-init or should
design around new init helpers immediately.

Day 8 fixes the contract as:

- `sparse_analysis_t` may begin zeroed
- `sparse_factors_t` may begin zeroed
- zeroed structs are the normative initial state
- free is explicit and safe on zeroed state

Interpretation:

- Sprint 50 does not need new public init helpers to make the lifecycle
  coherent
- optional additive helpers can remain a later implementation question if
  Sprint 51 proves they are justified

#### 2. Analyze-once / factor-refactor-many is now the final repeated-run direct story

Day 8 finalizes the repeated direct-run contract as:

1. zero / init
2. analyze / prepare
3. factor
4. solve
5. refactor / reuse
6. free

And it fixes the main caller-facing meaning:

- analyze once
- factor / solve
- refactor / solve many

Interpretation:

- this is no longer just the strongest implementation path
- it is the intended stable-pattern repeated-run public contract for:
  - LU
  - Cholesky
  - LDL^T

#### 3. “Prepare” remains analysis-specific public vocabulary

Day 8 confirms that the direct public prepare step should still mean:

- choose factor family
- choose reorder policy
- compute reusable symbolic/permutation state
- establish the same-pattern reuse contract

Interpretation:

- Sprint 50 should not flatten the direct lifecycle into generic
  workspace/handle language
- `sparse_analyze(...)` remains the real public analysis/prepare entry point

#### 4. Reuse meaning and non-meaning are now explicit enough to stabilize later docs

Day 8 fixes the one-sentence behavioral truth as:

- reuse preserves symbolic/permutation setup, not old numeric factor state

It also fixes the negative boundary:

- no incremental-update guarantee on prior triangular numeric data
- no backend-specific CSC/native storage persistence contract
- no automatic structural-pattern validation beyond the current caller
  precondition

Interpretation:

- this is the direct-solver analogue of the Epic 4 repeated-run truth anchor

#### 5. The one-shot versus repeated-run relationship is now explicit instead of only implied

Day 8 finalizes the relationship as:

- one-shot APIs are first-class peer entry points
- for one-off or low-context solves, they are also the simple/default path
- the analysis/factor/refactor lifecycle is the explicit opt-in path for
  stable-pattern repeated direct runs

Interpretation:

- Sprint 50 can now say both truths clearly:
  - one-shot APIs remain fully supported
  - the repeated lifecycle is the intended factor-many performance story

#### 6. The struct/option story is now clear enough for later implementation and docs adoption

Day 8 fixes the caller-facing struct expectations as:

- `sparse_analysis_t` owns symbolic/permutation state
- `sparse_factors_t` owns numeric factor state
- neither object should be described as owning the source matrix
- designated initializers remain the preferred public style for option structs
- family-specific option structs stay family-specific where that reflects real
  semantics

Interpretation:

- Sprint 51 now has enough contract clarity to align header wording and tests
  without reopening the public model

#### 7. Sprint 50’s remaining design queue is now fence work, not contract-shape work

After Day 8, the remaining design work is no longer about the repeated-run
public contract itself.

What remains for later Sprint 50 design days is:

- non-goal recording
- compatibility fence wording
- sprint-to-sprint boundary documentation

Interpretation:

- the public direct lifecycle design is now complete enough to drive Sprint 51
  implementation

## Day 9

**Objective:** Turn the finalized Day 8 lifecycle contract into an explicit
scope and compatibility fence so Sprint 50-52 stay additive, preserve the
one-shot public story, and avoid widening into a broad direct-solver rewrite.

### Commands Run

1. Re-read the Day 9 plan item and the latest Sprint 50 notes:
   - `sed -n '320,420p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `tail -n 260 docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the finalized Day 8 contract:
   - `sed -n '1,340p' docs/planning/EPIC_5/SPRINT_50/artifacts/day8-public-direct-solver-lifecycle-api-design-batch2.md`

### Day 9 Findings

#### 1. The strongest remaining risk after Day 8 is scope drift, not contract ambiguity

After Day 8, the repeated direct-run contract is already concrete enough to
implement. The largest remaining risk is therefore not “what should the public
model be?” but:

- accidental API widening
- accidental demotion of one-shot paths
- accidental exposure of internal CSC/native structures
- accidental benchmark/docs churn masquerading as lifecycle progress

Interpretation:

- Day 9 is the right point to convert the scope boundary into an explicit
  written fence

#### 2. Sprint 50-52 should be allowed to clarify and strengthen the existing analysis/factor/refactor story, not replace it

The allowed change set should include:

- making the repeated direct-run story easier to discover
- aligning public headers and docs with the Day 8 contract
- adding only bounded lifecycle-supporting public refinements where justified
- improving tests and benchmarks around the explicit repeated-run path

Interpretation:

- Epic 5 is an additive lifecycle-centering effort
- it is not a ground-up direct-solver API replacement

#### 3. The non-goal list is now explicit enough to stop the main likely overreaches

Day 9 fixes the main explicit non-goals as:

- no broad public factor-container redesign everywhere at once
- no removal or demotion of one-shot direct APIs
- no raw internal storage exposure
- no unrelated solver-family expansion
- no broad benchmark-framework redesign
- no structural-pattern verifier redesign in Sprint 50-52

Interpretation:

- the main likely forms of design drift are now named and fenced off directly

#### 4. One-shot compatibility preservation is now recorded as a conscious contract

Day 9 fixes that Sprint 50-52 must preserve that callers can still use:

- `sparse_lu_factor(...)`
- `sparse_lu_factor_opts(...)`
- `sparse_cholesky_factor(...)`
- `sparse_cholesky_factor_opts(...)`
- `sparse_ldlt_factor(...)`
- `sparse_ldlt_factor_opts(...)`

And that these are still:

- supported
- documented
- appropriate for one-off or low-context solves

Interpretation:

- one-shot direct APIs are not tolerated leftovers
- they are an explicit compatibility commitment inside Epic 5

#### 5. Mutable-`SparseMatrix` one-shot behavior remains an accepted tradeoff rather than a hidden future promise

Day 9 fixes that Epic 5 may clarify the one-shot LU / Cholesky mutation model
more clearly, but does not remove it.

That includes:

- factorization on a copied matrix when the original matrix view matters
- mutation of matrix-carried factor/reorder state in the one-shot path

Interpretation:

- the compatibility boundary is now honest about what Epic 5 is and is not
  trying to change

#### 6. The Sprint 50-to-51 boundary is now clean

Sprint 50 design owns:

- contract wording
- non-goals
- compatibility fence
- adoption-boundary decisions
- validation/landing planning

Sprint 51+ implementation owns:

- header edits
- source integration
- targeted test additions
- selected example/benchmark adoption
- validation execution

Interpretation:

- Sprint 50 now has a clear stop line
- implementation planning can proceed later without reopening the public model

#### 7. The adoption boundary remains selective and now inherits the compatibility fence

Early adopters should still be:

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`
- the most directly related public headers

Intentional lagging surfaces remain:

- small one-shot examples
- `examples/README.md` one-shot teaching surfaces
- `benchmarks/bench_refactor_csc.c`
- broader README/tutorial reshaping

Interpretation:

- the fence protects against broad surface churn before the core lifecycle
  implementation lands

## Day 10

**Objective:** Define the validation contract, targeted follow-ons, and
implementation order for the later public direct-solver lifecycle landing so
Sprint 51 begins from an explicit execution plan rather than sprint memory.

### Commands Run

1. Re-read the Day 10 plan item and the latest Sprint 50 notes:
   - `sed -n '360,460p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `tail -n 260 docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the Day 9 scope/compatibility fence:
   - `sed -n '1,320p' docs/planning/EPIC_5/SPRINT_50/artifacts/day9-non-goal-and-compatibility-fence.md`
3. Recheck the live build/test/example/benchmark target surfaces:
   - `rg -n "example_analysis|bench_refactor|bench_refactor_csc|test_cholesky|test_ldlt|test_etree|test_chol_csc|test_ldlt_csc" Makefile CMakeLists.txt tests benchmarks examples`
   - `sed -n '1,260p' examples/example_analysis.c`

### Day 10 Findings

#### 1. Later direct-lifecycle code days should use the same baseline validation gate as the other public Epic 4/5 landings

Day 10 fixes the mandatory gate for later `*.c` / `*.h` lifecycle batches as:

- `make format`
- `make lint`
- `make test`

And for substantial public API batches:

- `make quality-review-full`

Interpretation:

- direct lifecycle work is public-surface work
- it should use the strongest established local validation contract rather than
  a lighter sprint-local shortcut

#### 2. The highest-signal targeted follow-ons are now explicit and grounded in the live repo surfaces

Later lifecycle implementation should treat these as the main targeted reruns:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Interpretation:

- the explicit repeated-run teaching surface is already real in
  `example_analysis`
- the factor-many benchmark surfaces are already real in `bench_refactor*`
- the family-level regression binaries already exist and should be used instead
  of inventing a new ad hoc validation story

#### 3. The implementation order should remain public-first, then behavior, then adoption

Day 10 fixes the intended order as:

1. public headers / API surface
2. implementation and wrapper integration
3. high-signal example / benchmark adoption
4. compatibility sweep
5. final validation

Interpretation:

- header review should happen before broad source churn
- the first docs/example/benchmark adoption should follow stable behavior, not
  race ahead of it

#### 4. The most likely early landing targets are now concrete

Primary public header targets remain:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Primary early adoption surfaces remain:

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

Interpretation:

- Day 10 keeps the landing plan aligned with the strongest repeated-run direct
  seams instead of widening into every direct surface at once

#### 5. The validation plan also inherits the Day 9 scope fence

The landing plan explicitly stays out of:

- raw CSC/native storage exposure
- broad benchmark framework redesign
- structural-pattern verifier redesign
- sweeping example conversion
- large tutorial rewrite
- generic direct-handle introduction as the main landing

Interpretation:

- the validation/landing plan reinforces the scope boundary instead of quietly
  weakening it

#### 6. Sprint 50 now has a complete pre-implementation package for Sprint 51

By the end of Day 10, Sprint 50 has:

- baseline/truthfulness anchors
- public-surface inventory
- precedent map
- ranked gap analysis
- first-pass lifecycle design
- post-design audit
- final public contract
- scope/compatibility fence
- validation and landing plan

Interpretation:

- the remaining Sprint 50 work can now focus on caller-surface audit, summary,
  validation sweep, and closeout rather than more contract discovery

## Day 11

**Objective:** Re-audit the live caller-facing docs, examples, benchmark docs,
and direct public headers from the perspective of the finished Sprint 50
direct-lifecycle contract, then bound the later adoption set.

### Commands Run

1. Re-read the Day 11 plan item and the latest Sprint 50 notes:
   - `sed -n '420,520p' docs/planning/EPIC_5/SPRINT_50/PLAN.md`
   - `tail -n 260 docs/planning/EPIC_5/SPRINT_50/WORKING_NOTES.md`
2. Re-read the current top-level and local caller docs:
   - `sed -n '1,260p' README.md`
   - `sed -n '1,260p' examples/README.md`
   - `sed -n '1,220p' benchmarks/README.md`
   - `sed -n '1,260p' docs/tutorial.md`
3. Re-check the strongest direct public lifecycle and family-local header
   surfaces:
   - `sed -n '1,260p' include/sparse_analysis.h`
   - `sed -n '1,220p' include/sparse_lu.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
   - `sed -n '1,260p' include/sparse_ldlt.h`
4. Search for repeated-run / refactor / copy-discipline wording drift:
   - `rg -n "analy(z|s)e once|refactor|repeated-run|same-pattern|copy\\(|identity permutations|one-shot|simple/default path|peer entry" README.md docs/tutorial.md examples/README.md benchmarks/README.md include/sparse_analysis.h include/sparse_lu.h include/sparse_cholesky.h include/sparse_ldlt.h`

### Day 11 Findings

#### 1. `include/sparse_analysis.h` is already the strongest repeated-run direct contract surface

The header already teaches:

- analyze once
- factor
- solve
- refactor
- free

And it already uses zeroed `analysis` / `factors` in the public workflow
example.

Interpretation:

- this header needs later wording alignment, not conceptual redesign
- it remains the main repeated-run direct header contract

#### 2. The family-local direct headers are aligned enough as one-shot-first surfaces

`include/sparse_lu.h`, `include/sparse_cholesky.h`, and `include/sparse_ldlt.h`
already do the right main thing:

- they teach their family-local contract honestly
- LU / Cholesky explicitly teach copy-before-in-place-factorization
- LDL^T explicitly teaches separate factor-object output and identity-
  permutation input expectations

Interpretation:

- these surfaces should stay family-local
- later work should mostly add bounded relationship wording or cross-reference,
  not rewrite them into repeated-run guides

#### 3. The top-level README is only partially aligned: it names the direct repeated-run workflow but does not yet frame it as a caller decision path

The README already:

- lists `sparse_analysis.h` in the API overview
- names `sparse_analyze`, `sparse_factor_numeric`, `sparse_refactor_numeric`,
  and `sparse_factor_solve`
- describes analyze-once / factor-many in feature terms

But unlike the iterative/eigensolver repeated-run story, it does not yet give
direct callers a migration-style explanation for:

- when to stay on one-shot direct APIs
- when to use analysis/factor/refactor
- what direct reuse preserves and what it does not

Interpretation:

- `README.md` is a high-signal later adoption target

#### 4. `docs/tutorial.md` should stay mostly one-shot-first but will need a bounded repeated-run cross-reference

The tutorial already teaches:

- copy-before-factorization
- identity-permutation discipline for QR and preconditioners

But it does not yet contain a bounded explicit repeated direct-run note.

Interpretation:

- the tutorial should not be broadly rewritten
- it should later gain only a small repeated-run section or cross-reference

#### 5. `examples/README.md` should stay one-shot-first by design, but it currently omits the strongest direct repeated-run example

The examples README already correctly says:

- shipped examples lean on one-shot public APIs
- those one-shot paths remain first-class

But it does not list `example_analysis`.

Interpretation:

- the file’s one-shot-first scope is correct
- the omission of `example_analysis` is a real later fix target

#### 6. `benchmarks/README.md` contains one real behavior/documentation contradiction today

The benchmark table currently says:

- `bench_refactor` = “LDL^T re-factor with cached symbolic”

But the live driver is a Cholesky analyze-once / factor-many benchmark.

Interpretation:

- this is an actual docs drift, not merely a future adoption opportunity
- it should be corrected when the direct repeated-run docs/benchmark adoption
  work lands

#### 7. The later caller-surface adoption set is now small and explicit

Highest-signal later updates:

1. `README.md`
2. `examples/example_analysis.c` supporting docs around it
3. `examples/README.md`
4. `benchmarks/README.md`

Lower-priority or bounded later updates:

1. `docs/tutorial.md` repeated-run cross-reference only
2. family-local direct headers only if touched during implementation

Interpretation:

- Sprint 51+ does not need a broad caller-doc rewrite
- it needs a narrow adoption batch around the real repeated-run direct surfaces
