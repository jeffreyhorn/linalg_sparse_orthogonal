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
