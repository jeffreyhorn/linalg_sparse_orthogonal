# Sprint 53 Working Notes

## Day 1

**Objective:** Turn the Sprint 53 project-plan scope plus the Sprint 52
validated closeout package into a concrete CSC direct-solver follow-through
starting point by confirming the preserved reviewed baseline, naming the
Sprint 53 implementation workstreams explicitly, and defining the
authoritative CSC header, implementation, benchmark, regression, and
validation hotspots before any CSC follow-through edits begin.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 53 project-plan source and the new sprint plan:
   - `sed -n '121,149p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
3. Re-read the strongest Sprint 52 closeout and handoff artifacts:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_52/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_52/RETROSPECTIVE.md`
   - `sed -n '1,200p' docs/planning/EPIC_5/SPRINT_52/artifacts/day1-scope-and-phase2-lifecycle-baseline.md`
4. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
6. Measure the live CSC header, implementation, benchmark, regression, and
   caller-facing hotspot sizes:
   - `wc -l include/sparse_analysis.h include/sparse_cholesky.h include/sparse_ldlt.h src/sparse_analysis.c src/sparse_cholesky.c src/sparse_ldlt.c src/sparse_chol_csc.c src/sparse_ldlt_csc.c src/sparse_chol_csc_internal.h src/sparse_ldlt_csc_internal.h benchmarks/bench_refactor_csc.c tests/test_integration.c tests/test_chol_csc.c tests/test_ldlt_csc.c README.md benchmarks/README.md examples/example_analysis.c`
7. Reconfirm the live CSC dispatch and analysis-aware indefinite references:
   - `rg -n "ldlt_csc_from_sparse_with_analysis|chol_csc_from_sparse_with_analysis|SPARSE_FACTOR_LDLT|supernodal|dispatch" include src tests benchmarks examples README.md`
8. Re-scan planning notes that already mention the relevant deferred CSC
   follow-through seams:
   - `rg -n "ldlt_csc_from_sparse_with_analysis|deferred|Sprint 17|Sprint 19|CSC" docs/planning/EPIC_5 docs/planning/EPIC_4 docs/planning -g '*.md'`

### Day 1 Findings

#### 1. Sprint 53 starts from a preserved Sprint 52 validated Phase 2 package, not from renewed direct-lifecycle design work

The inherited starting state is already explicit and stable:

- Sprint 52 already closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- Sprint 52 already left the public repeated-run direct contract intact:
  - one-shot LU / Cholesky / LDL^T remain first-class
  - repeated direct runs remain analysis/factors-centric
  - reuse preserves symbolic/permutation setup, not stale numeric factor
    contents
- Sprint 52 already deepened the strongest shared Cholesky and LDL^T CSC
  repeated-run paths

Interpretation:

- Sprint 53 is not a baseline-repair sprint
- Sprint 53 is not a public API redesign sprint
- Sprint 53 is a CSC follow-through sprint on top of a validated Phase 2 base

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible on all substantial CSC batches

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 53 should keep using the exact “strongest local reviewed baseline”
  phrasing
- substantial CSC direct-solver batches should continue to treat the reviewed
  CMake count and parity contract as truthfulness anchors

#### 3. The real Sprint 53 queue is now concentrated in CSC-specific completion seams, not in the shared repeated-run direct contract

The Sprint 53 plan items and live repo state narrow to six bounded work
classes:

1. analysis-aware LDL^T indefinite path completion
2. transparent LDL^T dispatch follow-through
3. indefinite CSC factor-many proof
4. Cholesky / LDL^T dispatch reconciliation
5. targeted benchmark and regression refresh
6. validation and closeout

Interpretation:

- the shared `sparse_analysis_t` / `sparse_factors_t` public contract is
  already strong enough to keep
- the remaining work is mostly inside CSC behavior, dispatch clarity, and
  proof surfaces

#### 4. The strongest architectural seam is still the analysis-aware LDL^T CSC indefinite path

The live code and planning references already point to the same seam:

- `ldlt_csc_from_sparse_with_analysis(...)` is the named analysis-aware LDL^T
  CSC path
- Sprint 52 Day 5 already reused this path directly from the shared
  repeated-run integration when the scalar BK pre-pass stayed compatible with
  the caller analysis
- Sprint 53 now needs to audit whether that path is complete and uniformly
  supported enough for the remaining deferred workloads

Interpretation:

- the highest-value Day 3-5 work is likely inside LDL^T CSC preparation and
  dispatch details, not broad public-surface changes

#### 5. The live hotspot map is already concentrated enough to name directly

The main touched surfaces are clear before any new CSC edits begin:

- public/shared contract:
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `320`
- shared / family implementation:
  - `src/sparse_analysis.c` = `818`
  - `src/sparse_cholesky.c` = `494`
  - `src/sparse_ldlt.c` = `1494`
- CSC implementation hotspots:
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_ldlt_csc.c` = `2723`
  - `src/sparse_chol_csc_internal.h` = `994`
  - `src/sparse_ldlt_csc_internal.h` = `805`
- strongest proof/adoption surfaces:
  - `benchmarks/bench_refactor_csc.c` = `388`
  - `tests/test_integration.c` = `1529`
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_ldlt_csc.c` = `3637`
  - `README.md` = `930`
  - `benchmarks/README.md` = `191`
  - `examples/example_analysis.c` = `210`

Interpretation:

- Sprint 53 is correctly centered on CSC-heavy source and regression files
- the largest proof concentration remains `test_chol_csc.c`,
  `test_ldlt_csc.c`, and `test_integration.c`

#### 6. The strongest deferred-work bridge from earlier sprints is now explicit in both code and planning notes

The inherited planning bridge is now concrete:

- Sprint 53 explicitly depends on Sprint 17/19 CSC deferred follow-ons being
  inventoried
- live docs and tests still carry those seams forward in:
  - `README.md`
  - `tests/test_sprint20_integration.c`
  - `tests/test_ldlt.c`
  - Sprint 50 / Sprint 52 working notes and artifacts

Interpretation:

- Sprint 53 does not need to rediscover the CSC deferred queue
- it needs to close or materially reduce the highest-value pieces of that
  already-known queue

#### 7. The Sprint 53 workstreams are now explicit before code changes begin

The Day 1 implementation workstreams are:

1. CSC baseline and validation recheck
2. analysis-aware indefinite path audit
3. analysis-aware LDL^T integration
4. transparent LDL^T dispatch follow-through
5. indefinite factor-many benchmark proof
6. Cholesky / LDL^T dispatch reconciliation
7. targeted CSC regression and validation closeout

Interpretation:

- the Sprint 53 queue is already narrowed to CSC completion slices, not broad
  direct-solver research
- the correct Day 1 close is a clean CSC baseline and authoritative-input
  package

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 53 must preserve, then define the smallest authoritative
validation boundary for the later CSC implementation days and the high-signal
CSC rerun set those code-touch batches should use.

### Commands Run

1. Re-read the Sprint 53 Day 2 plan item and the current sprint notes:
   - `sed -n '78,123p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
2. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
4. Re-read the live quality-contract wording sources:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
5. Reconfirm the targeted CSC follow-on binaries already present in the build
   tree:
   - `ls build/bench_refactor_csc build/test_chol_csc build/test_ldlt_csc build/test_cholesky build/test_ldlt build/test_etree build/test_integration build/example_analysis`

### Day 2 Findings

#### 1. The strongest local reviewed baseline and truthfulness anchors remain exact

The maintained Sprint 53 baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The authority split is still the same:

- `make quality-review-full`
  - strongest local reviewed baseline
- `make quality-review`
  - reviewed Makefile path
- `make quality-review-cmake`
  - reviewed CMake parity path
- `make deadcode-check`
  - report-completeness gate, not a zero-findings gate

Interpretation:

- Sprint 53 should keep using the exact “strongest local reviewed baseline”
  phrasing
- the sprint should treat the reviewed CMake count and parity contract as
  truthfulness anchors rather than as loose guidance

#### 2. The later CSC code-day gate is simple and should stay explicit

The mandatory gate for later `*.c` / `*.h` CSC work remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial shared direct-solver or CSC dispatch
batches remains:

- `make quality-review-full`

Interpretation:

- Sprint 53 does not need a sprint-specific validation invention
- it needs to preserve the same code-day and substantial-batch boundary that
  already governs the repo

#### 3. The high-signal Sprint 53 CSC rerun set is now fixed from the live build tree

The targeted CSC follow-on binaries already present and ready to rerun are:

- `./build/bench_refactor_csc`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_integration`
- `./build/example_analysis`

Interpretation:

- Sprint 53 does not need to guess which post-patch binaries matter most
- the strongest follow-on set is already concrete before Day 3 auditing and
  Day 4+ CSC patches begin

#### 4. The quality-contract wording remains internally aligned across the main authority sources

The current wording still agrees across:

- `Makefile`
- `README.md`
- `docs/maintainer_guide.md`

The key retained truths are:

- `quality-review-full` is still the strongest local reviewed baseline
- `quality-review-cmake-compile` / `quality-review-cmake` still own the
  reviewed CMake parity path
- `deadcode-check` still means report completeness, not zero findings

Interpretation:

- Sprint 53 can safely cite the existing quality contract without rewording it
- Day 2 does not surface any need to reopen the Sprint 48 quality-contract
  simplification work

#### 5. The smallest authoritative validation boundary is now explicit

For Sprint 53:

- docs-only days:
  - preserve the reviewed wording/count anchors
  - use targeted sanity checks only
- `*.c` / `*.h` CSC days:
  - `make format`
  - `make lint`
  - `make test`
- substantial shared direct-solver or dispatch batches:
  - add `make quality-review-full`
- targeted CSC follow-ons:
  - rerun only the binaries justified by the touched seam

Interpretation:

- Sprint 53 now has a clean validation boundary before any CSC implementation
  work starts
- there is no ambiguity around when a batch needs only sanity checks versus
  full code-day validation

## Day 3

**Objective:** Audit the live analysis-aware LDL^T indefinite CSC path so
Sprint 53 can start from named fallback and proof seams instead of a generic
"complete the path" instruction.

### Commands Run

1. Re-read the Sprint 53 Day 3 plan item and the current sprint notes:
   - `sed -n '124,159p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
   - `sed -n '1,420p' docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
2. Re-scan the main analysis-aware LDL^T references across the shared path,
   public headers, tests, and README:
   - `rg -n "ldlt_csc_from_sparse_with_analysis|sparse_factor_numeric\\(|sparse_refactor_numeric\\(|SPARSE_FACTOR_LDLT|used_csc_path|supernodal|pivot" src/sparse_analysis.c src/sparse_ldlt.c src/sparse_ldlt_csc.c src/sparse_ldlt_csc_internal.h include/sparse_analysis.h include/sparse_ldlt.h tests/test_integration.c tests/test_ldlt.c tests/test_ldlt_csc.c tests/test_sprint20_integration.c benchmarks/bench_refactor_csc.c README.md`
3. Re-read the live shared repeated-run direct contract and the shared LDL^T
   CSC factor path:
   - `sed -n '230,340p' include/sparse_analysis.h`
   - `sed -n '300,820p' src/sparse_analysis.c`
4. Re-read the live one-shot LDL^T CSC dispatch path and its public options
   contract:
   - `sed -n '820,1160p' src/sparse_ldlt.c`
   - `sed -n '120,220p' include/sparse_ldlt.h`
5. Re-read the internal CSC helper contract that the shared path depends on:
   - `sed -n '220,360p' src/sparse_ldlt_csc_internal.h`
6. Re-read the strongest existing dispatch and parity proof surfaces:
   - `sed -n '150,260p' tests/test_sprint20_integration.c`
   - `sed -n '2360,2465p' tests/test_ldlt.c`
7. Re-read the strongest prior sprint artifact for the shared LDL^T CSC reuse
   landing:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_52/artifacts/day5-numeric-reuse-integration-batch2.md`
8. Re-read the current CSC factor-many benchmark surface:
   - `sed -n '1,320p' benchmarks/bench_refactor_csc.c`

### Day 3 Findings

#### 1. The analysis-aware indefinite CSC path is already real on the shared repeated-run direct surface

The current shared path is materially stronger than a generic wrapper:

- `sparse_factor_numeric(...)` routes `SPARSE_FACTOR_LDLT` through
  `factor_ldlt_with_analysis_csc(...)` on `n >= SPARSE_CSC_THRESHOLD`
- the public header already states the real current contract:
  - reuse caller analysis directly when the scalar BK pre-pass does not add
    extra swaps
  - rebuild analysis only when the final symmetric permutation moves beyond the
    caller's reorder
- `sparse_validate_analysis_input_matrix(...)` already enforces:
  - matching dimension
  - original row/col state
  - cheap gross-structure match through `source_nnz`

Interpretation:

- Sprint 53 does not start from a missing analysis-aware indefinite path
- it starts from a partial completion problem: the path exists, but it still
  carries concentrated rebuild and proof seams

#### 2. The main residual fallback is concentrated inside `factor_ldlt_with_analysis_csc(...)`

The shared LDL^T CSC helper still does the expensive orchestration itself on
every factorization:

1. run `ldlt_csc_from_sparse(...)`
2. run scalar BK elimination
3. compare the resulting permutation against the caller analysis
4. if permutations differ:
   - build `A_perm`
   - rerun `sparse_analyze(..., REORDER_NONE)`
   - call `ldlt_csc_from_sparse_with_analysis(...)` on the pre-permuted matrix
5. seed `pivot_size`
6. run `ldlt_csc_eliminate_supernodal(...)`
7. fall back to the scalar factor if the batched path does not complete cleanly

Interpretation:

- the core Phase 3 implementation seam is not “make the path exist”
- it is “reduce or better bound the amount of hidden scalar-prepass /
  reanalysis orchestration still happening inside the shared CSC helper”

#### 3. `sparse_refactor_numeric(...)` is still only a safe wrapper around full re-entry to `sparse_factor_numeric(...)`

The refactor path still:

- validates the new matrix against the stored analysis/factor contract
- allocates a temporary factors object
- calls `sparse_factor_numeric(...)` again
- swaps in the new factors only on success

That means the LDL^T CSC repeated-run path still pays the same Day 3 helper
orchestration on every refactor attempt:

- scalar BK pre-pass
- possible pre-permute + reanalyze
- CSC factor build
- possible supernodal fallback

Interpretation:

- Sprint 53’s strongest repeated-run seam is now explicit
- the main factor-many follow-through work is tied directly to this helper and
  the proof around it, not to a different public API

#### 4. The one-shot transparent LDL^T dispatch path is complete enough for callers, but it still duplicates CSC orchestration with the shared path

`sparse_ldlt_factor_opts(...)` and `ldlt_factor_csc_path(...)` already give the
public one-shot surface a coherent story:

- AUTO / LINKED_LIST / CSC dispatch is explicit
- `used_csc_path` is published early and stable on success/error
- the CSC path already wraps:
  - scalar pre-pass
  - pre-permute
  - analyze
  - `ldlt_csc_from_sparse_with_analysis(...)`
  - supernodal attempt
  - scalar fallback
  - writeback

But the shared repeated-run path and the one-shot path still own similar
orchestration in two places rather than one shared internal abstraction.

Interpretation:

- that duplication is now a named drift seam
- Sprint 53 should treat it as a bounded CSC follow-through target, not as a
  reason to redesign the public direct API

#### 5. The internal helper contract still exposes why the indefinite path is harder than Cholesky

`ldlt_csc_from_sparse_with_analysis(...)` already documents the real
indefinite boundary:

- SPD inputs may call it directly
- indefinite batched use still requires:
  - scalar pre-pass
  - final symmetric permutation resolution
  - pre-permuted matrix
  - analysis on that pre-permuted matrix

Interpretation:

- the helper itself is not the missing piece
- the missing follow-through is better end-to-end ownership of that pre-pass /
  pre-permute / reanalysis sequence and stronger proof that the resulting path
  behaves predictably on intended indefinite workloads

#### 6. The proof surface is still asymmetric: dispatch routing is covered better than indefinite factor-many behavior

What is already well covered:

- `tests/test_sprint20_integration.c`
  - AUTO routing below and above threshold
  - indefinite KKT routing through the CSC path
- `tests/test_ldlt.c`
  - forced CSC backend factorization and cross-backend agreement
- `tests/test_ldlt_csc.c`
  - deep CSC factor kernel invariants

What is still missing or under-centered:

- no LDL^T-specific analogue of the Day 8 `bench_refactor` proof
- `bench_refactor_csc.c` is still Cholesky-only despite its shared repeated-run
  framing
- the public analysis/factors repeated-run LDL^T path does not yet have a
  dedicated benchmark or equally clear regression story for same-pattern
  indefinite updates

Interpretation:

- Sprint 53’s strongest proof gap is now explicit: indefinite factor-many
  evidence is weaker than dispatch-routing evidence

#### 7. The ranked Sprint 53 CSC target list is now concrete

Highest-value Phase 3 targets:

1. reduce or better bound hidden scalar-prepass / reanalysis work inside the
   shared LDL^T CSC repeated-run path
2. tighten shared-vs-one-shot LDL^T CSC orchestration so dispatch behavior is
   easier to reason about
3. add real indefinite factor-many benchmark proof
4. refresh public regression coverage around the shared indefinite repeated-run
   path
5. reconcile README / header wording only after the CSC ownership seams are
   clearer

Explicit non-goals from the audit:

- no public direct-solver redesign
- no raw CSC/native storage exposure
- no full structural-pattern verifier redesign
- no broad tutorial or example rewrite
- no new generic direct-handle abstraction

## Day 4

**Objective:** Reduce the strongest Day 3 CSC ownership seam by unifying the
analysis-aware LDL^T CSC completion half that was still duplicated between the
shared repeated-run path and the one-shot CSC dispatch path, while adding
focused indefinite regression proof and closing the full code-day validation
gate.

### Commands Run

1. Re-read the Sprint 53 plan and Day 3 audit before touching code:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day3-analysis-aware-indefinite-path-audit.md`
   - `sed -n '320,520p' docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
2. Re-read the live shared and one-shot LDL^T CSC orchestration hotspots:
   - `rg -n "factor_ldlt_with_analysis_csc|ldlt_factor_csc_path|ldlt_csc_from_sparse_with_analysis|ldlt_csc_eliminate_supernodal|ldlt_csc_writeback_to_ldlt" src/sparse_analysis.c src/sparse_ldlt.c src/sparse_ldlt_csc_internal.h`
   - `sed -n '360,560p' src/sparse_analysis.c`
   - `sed -n '720,980p' src/sparse_ldlt.c`
   - `sed -n '1,140p' src/sparse_ldlt_csc_internal.h`
3. Re-read the strongest current shared-path integration coverage surface:
   - `rg -n "explicit_analysis_path|ldlt_factor_opts|sparse_factor_numeric|sparse_factor_solve" tests/test_integration.c`
   - `sed -n '1,260p' tests/test_integration.c`
   - `sed -n '860,1080p' tests/test_integration.c`
4. Land the bounded shared-helper and regression batch:
   - `apply_patch` on:
     - `src/sparse_ldlt_csc_internal.h`
     - `src/sparse_ldlt.c`
     - `src/sparse_analysis.c`
     - `tests/test_integration.c`
5. Review the exact patch:
   - `git diff -- src/sparse_ldlt_csc_internal.h src/sparse_ldlt.c src/sparse_analysis.c tests/test_integration.c`
6. Run the required code-day gate:
   - `make format`
   - `make lint`
   - `make test`
7. Run the stronger reviewed baseline:
   - `make quality-review-full`

### Day 4 Findings

#### 1. The strongest Day 3 duplication seam is now materially smaller

Day 4 extracted a new shared internal helper:

- `ldlt_csc_factor_with_resolved_analysis(...)`

That helper now owns the analysis-aware LDL^T CSC completion half that both
paths previously carried separately:

- `ldlt_csc_from_sparse_with_analysis(...)`
- seeding the CSC factor's `pivot_size` from the resolved scalar pre-pass
- supernodal attempt via `ldlt_csc_eliminate_supernodal(...)`
- scalar fallback when the supernodal path is not retained
- writeback into public `sparse_ldlt_t`

Interpretation:

- Sprint 53 did not redesign the indefinite CSC path
- it reduced the amount of LDL^T CSC completion logic that had to be reasoned
  about in two places

#### 2. The one-shot CSC dispatch path and the shared repeated-run path now share the same completion helper

The new helper is now used from both:

- `ldlt_factor_csc_path(...)`
- `factor_ldlt_with_analysis_csc(...)`

That means the two paths still differ where they genuinely need to differ:

- scalar BK pre-pass
- whether the caller analysis can be reused directly
- whether a pre-permuted matrix plus derived analysis is needed

But they no longer duplicate the later CSC completion half once that analysis
state has been resolved.

Interpretation:

- the Sprint 53 CSC ownership split is cleaner
- the dispatch story is easier to reason about without broadening the public
  API

#### 3. The shared repeated-run indefinite path kept its bounded semantics

Day 4 did not change the preserved Phase 2 contract:

- one-shot LDL^T remains first-class
- repeated direct runs remain analysis/factors-centric
- the LDL^T shared path still reuses caller analysis directly only when the
  scalar BK pre-pass does not force extra swaps beyond the caller reorder
- otherwise the path still rebuilds analysis only on the pre-permuted matrix

Interpretation:

- Day 4 tightened internal ownership
- it did not overpromise that indefinite LDL^T is as simple as Cholesky

#### 4. The focused regression proof is now stronger on the explicit indefinite shared path

Day 4 added a new integration test centered on the above-threshold indefinite
KKT case:

- `test_ldlt_factor_opts_matches_explicit_analysis_path_indefinite_kkt`

The new proof checks:

- one-shot `sparse_ldlt_factor_opts(...)` on an indefinite CSC-routed KKT
  matrix
- explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)` on the same
  matrix
- solve parity against the same exact right-hand side

Interpretation:

- Sprint 53 now has more direct proof that the public explicit-analysis LDL^T
  path and the one-shot CSC dispatch path stay behaviorally aligned on the
  intended indefinite workload class

#### 5. The full code-day validation gate closed cleanly

Day 4 touched `*.c` / `*.h`, so the required gate ran:

- `make format`
- `make lint`
- `make test`

All passed.

The stronger reviewed baseline also passed:

- `make quality-review-full`

Maintained truthfulness anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 131.51 sec`

Interpretation:

- the Day 4 CSC batch is validated from the repo's strongest local reviewed
  baseline
- no new validation drift surfaced while reducing the indefinite CSC
  orchestration seam

#### 6. Day 4 leaves a cleaner Day 5 starting point

What Day 4 materially improved:

- shared-vs-one-shot LDL^T CSC completion logic is more unified
- explicit indefinite shared-path proof is stronger

What Day 4 intentionally did not solve:

- the scalar BK pre-pass itself is still required
- derived-analysis fallback still exists when that pre-pass changes the final
  symmetric permutation
- there is still no LDL^T-specific factor-many benchmark equivalent to the
  Cholesky repeated-run proof

Interpretation:

- Day 5 can now focus more directly on deeper indefinite repeated-run follow-
  through and proof, rather than on duplicated CSC completion plumbing

## Day 5

**Objective:** Remove the next highest-value indefinite CSC seam by unifying
the shared scalar-prepass / resolved-analysis preparation front half used by
the one-shot CSC dispatch path and the repeated-run analysis/factors path,
while adding direct proof that the public LDL^T refactor workflow still stays
inside the bounded same-pattern reuse contract on an indefinite CSC workload.

### Commands Run

1. Re-read the Sprint 53 Day 5 plan item plus the Day 3 and Day 4 results:
   - `sed -n '196,228p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day3-analysis-aware-indefinite-path-audit.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day4-analysis-aware-ldlt-integration-batch1.md`
   - `tail -n 220 docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
2. Re-read the remaining shared LDL^T preparation and refactor hotspots:
   - `sed -n '420,560p' src/sparse_analysis.c`
   - `sed -n '780,980p' src/sparse_ldlt.c`
   - `sed -n '320,380p' src/sparse_ldlt_csc_internal.h`
   - `sed -n '940,1325p' tests/test_integration.c`
3. Reconfirm the later sprint ordering so Day 5 would not steal Day 8-10
   benchmark or dispatch work:
   - `sed -n '140,340p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
4. Land the bounded shared-preparation and regression batch:
   - `apply_patch` on:
     - `src/sparse_ldlt.c`
     - `src/sparse_analysis.c`
     - `src/sparse_ldlt_csc_internal.h`
     - `tests/test_integration.c`
5. Review the exact patch:
   - `git diff -- src/sparse_analysis.c src/sparse_ldlt.c src/sparse_ldlt_csc_internal.h tests/test_integration.c`
6. Run the required code-day gate:
   - `make format`
   - `make lint`
   - `make test`
7. Run the touched follow-ons justified by the batch:
   - `./build/test_integration`
   - `./build/test_ldlt`
   - `./build/test_ldlt_csc`
   - `./build/test_sprint20_integration`
   - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/example_analysis`

### Day 5 Findings

#### 1. The remaining duplicated indefinite CSC preparation seam is now materially smaller

Day 5 extracted a second shared internal helper:

- `ldlt_csc_prepare_resolved_analysis(...)`

That helper now owns the indefinite CSC preparation front half that Day 4 had
left duplicated:

- scalar BK pre-pass via `ldlt_csc_from_sparse(...)`
- `ldlt_csc_eliminate_native(...)`
- comparison against caller analysis when one exists
- pre-permuted matrix build when the resolved BK permutation diverges
- `SPARSE_FACTOR_LDLT` + `SPARSE_REORDER_NONE` derived analysis on that
  pre-permuted matrix

Interpretation:

- Day 4 unified the CSC completion half
- Day 5 unified the resolved-analysis preparation half
- the main indefinite CSC control flow is now split more cleanly into:
  - preparation
  - analysis-aware CSC completion

#### 2. The one-shot CSC path and the repeated-run shared path now share the same preparation boundary

Day 5 now routes both of these through the new helper:

- `ldlt_factor_csc_path(...)`
- `factor_ldlt_with_analysis_csc(...)`

That means the two paths still differ only where they should:

- one-shot path has no caller analysis hint, so it always resolves through a
  pre-permuted matrix plus derived analysis
- repeated-run path can still reuse the caller analysis directly when the BK
  pre-pass does not introduce extra swaps beyond that analysis

But they no longer duplicate the actual decision and setup logic that chooses
between those two cases.

Interpretation:

- Sprint 53 reduced another real CSC ownership seam without redesigning the
  public direct-solver surface
- later dispatch-follow-through days can now start from a cleaner internal
  base

#### 3. The bounded reuse semantics stayed honest

Day 5 did not change the preserved LDL^T repeated-run contract:

- one-shot LDL^T remains first-class
- repeated direct runs remain analysis/factors-centric
- reuse preserves symbolic/permutation setup when the resolved BK structure
  stays compatible with the caller analysis
- the path still performs a fresh scalar BK pre-pass each time
- the path still does not promise reuse of stale numeric factor values or old
  pivot choices

Interpretation:

- the batch strengthens reuse of symbolic/permutation setup
- it does not overclaim reuse of stale pivot or numeric state

#### 4. The public indefinite refactor story is now better proved

Day 5 added a new integration test:

- `test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt`

That proof uses an above-threshold indefinite KKT matrix and checks:

- initial explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)`
- correct solve on the original indefinite matrix
- same-pattern value perturbation on a fresh KKT matrix
- `sparse_refactor_numeric(...)` success on that perturbed indefinite matrix
- correct solve after refactor using the same public analysis/factors objects

Interpretation:

- Sprint 53 now has direct proof that the public LDL^T repeated-run refactor
  path works on a same-pattern indefinite CSC workload
- the proof is still honest: the contract is same-pattern numeric refresh, not
  stale-factor reuse

#### 5. The full code-day gate and touched follow-ons closed cleanly

Because `*.c` / `*.h` changed, Day 5 ran:

- `make format`
- `make lint`
- `make test`

All passed.

The touched follow-ons also passed:

- `./build/test_integration`
- `./build/test_ldlt`
- `./build/test_ldlt_csc`
- `./build/test_sprint20_integration`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/example_analysis`

Representative direct results:

- `test_integration` = `35 / 35`
- `test_ldlt` = `83 / 83`
- `test_ldlt_csc` = `95 / 95`
- `test_sprint20_integration` = `20 / 20`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 1.70x`
  - `res_ll = 8.24e-16`
  - `res_csc = 7.06e-16`
- `example_analysis` residual stayed `4.44e-16`

Interpretation:

- the Day 5 batch is validated from the normal code-day gate plus the
  highest-signal LDL^T/CSC follow-ons justified by the touched seam

#### 6. Day 5 leaves a cleaner Day 6 dispatch starting point

What Day 5 materially improved:

- duplicated indefinite CSC preparation logic is smaller
- repeated-run indefinite refactor proof is stronger

What Day 5 intentionally did not solve:

- public backend/telemetry wording and reasoning are still Day 6-7 work
- there is still no LDL^T-specific factor-many benchmark batch yet
- the scalar BK pre-pass remains the authoritative permutation-resolution step

Interpretation:

- Day 6 can now focus on dispatch reasoning and public path clarity instead of
  more duplicated indefinite preparation plumbing

## Day 6

**Objective:** Make the LDL^T CSC-vs-linked-list dispatch contract easier to
reason about on the highest-value public path by centralizing backend
selection, reusing one selected-backend execution seam across reorder and
no-reorder control flow, clarifying the `used_csc_path` contract, and adding
focused proof that selected-path telemetry is still published before later
validation failures.

### Commands Run

1. Re-read the Sprint 53 Day 6 plan item plus the Day 3-5 results:
   - `sed -n '229,260p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day3-analysis-aware-indefinite-path-audit.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day4-analysis-aware-ldlt-integration-batch1.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day5-analysis-aware-ldlt-integration-batch2.md`
   - `tail -n 220 docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
2. Re-read the live LDL^T dispatch and backend-telemetry seams:
   - `sed -n '96,180p' include/sparse_ldlt.h`
   - `sed -n '960,1165p' src/sparse_ldlt.c`
   - `sed -n '2380,2795p' tests/test_ldlt.c`
   - `sed -n '1,220p' tests/test_sprint20_integration.c`
3. Land the bounded dispatch-selection and public-proof batch:
   - `apply_patch` on:
     - `include/sparse_ldlt.h`
     - `src/sparse_ldlt.c`
     - `tests/test_ldlt.c`
4. Review the exact patch:
   - `git diff -- include/sparse_ldlt.h src/sparse_ldlt.c tests/test_ldlt.c`
5. Run the required code-day gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
6. Run the touched LDL^T/dispatch follow-ons justified by the batch:
   - `./build/test_ldlt`
   - `./build/test_sprint20_integration`
   - `./build/test_integration`
   - `./build/example_analysis`

### Day 6 Findings

#### 1. The public LDL^T dispatch decision is now centralized instead of being partly inlined in the one-shot wrapper

Day 6 extracted a new helper in `src/sparse_ldlt.c`:

- `ldlt_dispatch_select_backend(...)`

That helper now owns the explicit backend decision for:

- `SPARSE_LDLT_BACKEND_LINKED_LIST`
- `SPARSE_LDLT_BACKEND_CSC`
- `SPARSE_LDLT_BACKEND_AUTO`
- the `n == 0` empty-matrix exception

Interpretation:

- the public backend contract is easier to audit because the selected-path
  decision is now one named seam
- later LDL^T dispatch-follow-through days can start from a clearer selector
  instead of more wrapper-local branching

#### 2. Reorder and no-reorder LDL^T factor flow now share one selected-backend execution seam

Day 6 also extracted:

- `ldlt_factor_selected_backend(...)`

That helper now owns the actual selected-path execution after backend
resolution:

- CSC path:
  - `ldlt_factor_csc_path(...)`
- linked-list path:
  - `ldlt_factor_internal(...)`

And `sparse_ldlt_factor_opts(...)` now uses that one helper in both:

- reorder path after `sparse_permute(...)`
- no-reorder direct path

Interpretation:

- Day 6 did not redesign the LDL^T factor kernels
- it reduced another reasoning seam by making reorder and no-reorder branches
  share the same selected-backend contract

#### 3. The public backend telemetry contract is now more explicit and more honest

Day 6 tightened the public header wording in `include/sparse_ldlt.h` for:

- `SPARSE_LDLT_BACKEND_CSC`
- `used_csc_path`

The important clarifications are now explicit:

- `used_csc_path` reports the actual selected numeric path, not just the
  caller-requested backend enum
- forced CSC still reports linked-list on the `n == 0` empty-matrix edge case
  because the CSC scalar pre-pass has no meaningful empty input to factor

Interpretation:

- the wording now matches the implementation better
- later README or broader dispatch-summary work can refer back to a clearer
  header-level source of truth

#### 4. Sprint 53 now has direct proof that selected-path telemetry survives later reorder validation failure

Day 6 added a focused public regression in `tests/test_ldlt.c`:

- `test_ldlt_backend_csc_reports_selected_path_before_reorder_error`

That proof checks:

- forced CSC backend request
- `used_csc_path` output pointer set
- later invalid reorder enum
- selected-path telemetry still published as `1` before the later
  `SPARSE_ERR_BADARG` return

Interpretation:

- the repo now proves a higher-signal part of the dispatch contract than it
  did before
- callers can trust that the telemetry reports the selected numeric path even
  when later wrapper validation rejects the call

#### 5. The full code-day gate, stronger reviewed baseline, and touched follow-ons all closed cleanly

Because `*.c` / `*.h` changed, Day 6 ran:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 120.16 sec`

The touched follow-ons also passed:

- `./build/test_ldlt`
- `./build/test_sprint20_integration`
- `./build/test_integration`
- `./build/example_analysis`

Representative direct results:

- `test_ldlt` = `84 / 84`
- `test_sprint20_integration` = `20 / 20`
- `test_integration` = `35 / 35`
- `example_analysis` residual stayed `4.44e-16`

Interpretation:

- the Day 6 batch is validated from the normal code-day gate, the strongest
  local reviewed baseline, and the most relevant LDL^T/dispatch follow-ons

#### 6. Day 6 leaves a cleaner Day 7 dispatch-reconciliation starting point

What Day 6 materially improved:

- selected-path backend reasoning is now more explicit
- selected-backend execution is more centralized
- public backend telemetry wording is more honest
- direct public proof around dispatch telemetry is stronger

What Day 6 intentionally did not solve:

- LDL^T-specific factor-many benchmark proof is still later work
- broader Cholesky/LDL^T dispatch reconciliation is still later work
- the scalar BK pre-pass remains the authoritative indefinite permutation
  resolution step

Interpretation:

- Day 7 can now focus on cross-surface dispatch follow-through instead of
  another round of basic LDL^T selector cleanup

## Day 7

**Objective:** Tighten the next LDL^T CSC dispatch seam by making the
shared analysis-aware CSC completion helper distinguish between the intended
batched-path rejection fallback and real helper failures, while aligning the
public CSC wording with that narrower internal contract and adding direct
proof that contract violations do not get silently masked as scalar fallback.

### Commands Run

1. Re-read the Sprint 53 Day 7 plan item plus the Day 3-6 results:
   - `sed -n '261,320p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day3-analysis-aware-indefinite-path-audit.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day4-analysis-aware-ldlt-integration-batch1.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day5-analysis-aware-ldlt-integration-batch2.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day6-ldlt-dispatch-batch1.md`
   - `tail -n 220 docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
2. Re-read the live CSC completion and cross-surface dispatch wording seams:
   - `sed -n '120,210p' include/sparse_ldlt.h`
   - `sed -n '780,860p' src/sparse_ldlt.c`
   - `sed -n '2388,2488p' tests/test_ldlt.c`
   - `sed -n '1,260p' tests/test_sprint20_integration.c`
   - `sed -n '1540,1615p' tests/test_ldlt_csc.c`
3. Land the bounded CSC-completion and proof batch:
   - `apply_patch` on:
     - `include/sparse_ldlt.h`
     - `src/sparse_ldlt.c`
     - `tests/test_ldlt.c`
     - `tests/test_sprint20_integration.c`
     - `tests/test_ldlt_csc.c`
4. Review the exact patch:
   - `git diff -- include/sparse_ldlt.h src/sparse_ldlt.c tests/test_ldlt.c tests/test_sprint20_integration.c tests/test_ldlt_csc.c`
5. Run the required code-day gate:
   - `make format`
   - `make lint`
   - `make test`
6. Run the touched LDL^T/CSC follow-ons justified by the batch:
   - `./build/test_ldlt`
   - `./build/test_ldlt_csc`
   - `./build/test_sprint20_integration`
   - `./build/example_analysis`

### Day 7 Findings

#### 1. The shared LDL^T CSC completion helper now distinguishes real fallback from real failure

Day 7 tightened `ldlt_csc_factor_with_resolved_analysis(...)` in
`src/sparse_ldlt.c`.

Before Day 7:

- any non-`SPARSE_OK` return from `ldlt_csc_eliminate_supernodal(...)`
  silently fell back to the resolved scalar pre-pass factor

After Day 7:

- `SPARSE_OK`
  - retains the batched supernodal completion
- `SPARSE_ERR_BADARG`
  - falls back to the resolved scalar pre-pass factor because the batched path
    rejected the cached pivot pattern
- all other errors
  - now propagate directly instead of being masked as dispatch fallback

Interpretation:

- the CSC completion seam is more honest and easier to reason about
- Day 7 removed a real silent-failure risk instead of just rewording comments

#### 2. The helper contract is now explicit enough to reject invalid completion configuration up front

Day 7 also tightened the entry validation in
`ldlt_csc_factor_with_resolved_analysis(...)`:

- `analysis->type` must be `SPARSE_FACTOR_LDLT`
- `min_size` must be at least `1`

Interpretation:

- misuse of the shared completion helper is now a direct contract failure
- it no longer aliases into the same path used for intended supernodal
  rejection fallback

#### 3. The public LDL^T CSC wording now matches the actual dispatch layering better

Day 7 updated `include/sparse_ldlt.h` and the related test commentary so the
cross-surface story is clearer:

- `SPARSE_LDLT_BACKEND_CSC` now means the CSC pipeline, not a promise that the
  batched supernodal completion always survives
- `used_csc_path` continues to report the selected CSC-vs-linked-list path
- the internal CSC completion variants are now described explicitly as:
  - batched supernodal completion
  - resolved scalar-prepass fallback

Interpretation:

- the public wording is now closer to Cholesky where it should be
- the valid indefinite-family exception remains explicit instead of being
  flattened away

#### 4. Sprint 53 now has direct proof that contract violations are rejected instead of being silently treated as fallback

Day 7 added a new focused regression in `tests/test_ldlt_csc.c`:

- `test_s53_with_analysis_invalid_min_size_rejected`

That proof checks:

- valid KKT-style resolved-analysis setup
- invalid `min_size = 0` passed to
  `ldlt_csc_factor_with_resolved_analysis(...)`
- direct `SPARSE_ERR_BADARG` return

Interpretation:

- the helper no longer quietly “succeeds” through scalar fallback on an
  invalid completion configuration
- the LDL^T CSC dispatch story is tighter at the exact seam Day 7 targeted

#### 5. The code-day gate and touched LDL^T/CSC follow-ons closed cleanly

Because `*.c` / `*.h` changed, Day 7 ran:

- `make format`
- `make lint`
- `make test`

All passed.

The touched follow-ons also passed:

- `./build/test_ldlt`
- `./build/test_ldlt_csc`
- `./build/test_sprint20_integration`
- `./build/example_analysis`

Representative direct results:

- `test_ldlt` = `84 / 84`
- `test_ldlt_csc` = `96 / 96`
- `test_sprint20_integration` = `20 / 20`
- `example_analysis` residual stayed `4.44e-16`

Interpretation:

- the Day 7 batch is validated from the normal code-day gate plus the most
  relevant LDL^T/CSC follow-ons for the touched seam

#### 6. Day 7 leaves a cleaner Day 8 benchmark-proof starting point

What Day 7 materially improved:

- the CSC completion fallback seam is narrower
- public CSC wording is more accurate about what “CSC selected” actually means
- the internal helper contract has direct regression proof

What Day 7 intentionally did not solve:

- LDL^T-specific factor-many benchmark proof is still Day 8 work
- broader benchmark/documentation claims still need to stay bounded to
  measured evidence
- the scalar BK pre-pass remains the authoritative indefinite permutation
  resolution step

Interpretation:

- Day 8 can now focus on measured indefinite factor-many evidence instead of
  more cleanup around hidden CSC completion semantics

## Day 8

**Objective:** Turn the deferred LDL^T-specific factor-many proof into real
measured evidence without redesigning the benchmark framework, while keeping
the benchmark claims truthful about the live public repeated-run path and the
direct CSC completion seam.

### Commands Run

1. Re-read the Sprint 53 Day 8 plan item plus the Day 3-7 findings:
   - `sed -n '321,380p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day3-analysis-aware-indefinite-path-audit.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day7-ldlt-dispatch-batch2.md`
   - `tail -n 220 docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
2. Re-read the live benchmark and direct-lifecycle proof surfaces:
   - `sed -n '1,260p' benchmarks/bench_refactor_csc.c`
   - `sed -n '1,220p' benchmarks/README.md`
   - `sed -n '1030,1115p' tests/test_integration.c`
   - `sed -n '1340,1425p' tests/test_ldlt.c`
   - `sed -n '400,520p' benchmarks/bench_ldlt_csc.c`
3. Land the bounded benchmark / correctness / regression batch:
   - `apply_patch` on:
     - `benchmarks/bench_refactor_csc.c`
     - `benchmarks/README.md`
     - `src/sparse_analysis.c`
     - `tests/test_integration.c`
4. Run targeted proof and regression checks while iterating:
   - `./build/test_integration`
   - `./build/test_etree`
   - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/bench_refactor_csc --indefinite-kkt --repeat 1`
   - `./build/example_analysis`
5. Run the required code-day gate on the corrected state:
   - `make format`
   - `make test`
   - `make lint`

### Day 8 Findings

#### 1. Day 8 now leaves a real indefinite factor-many benchmark mode instead of only a Cholesky/SPD proof surface

`benchmarks/bench_refactor_csc.c` now has two bounded workflows:

- `chol_spd`
  - the existing SPD / Cholesky repeated-run surface
- `ldlt_kkt`
  - a new synthetic above-threshold indefinite KKT repeated-run surface

The new mode is intentionally narrow:

- single built-in `kkt-150` workload
- same-pattern numeric perturbations only
- one analyze call amortized across later refactors
- public repeated-run path versus direct CSC completion path

Interpretation:

- Sprint 53 now has measured indefinite factor-many evidence on the intended
  workload class instead of only dispatch and regression proof
- Day 8 did not introduce a new benchmark framework or a larger corpus design

#### 2. The benchmark contract is now more truthful about the live direct repeated-run surfaces

Day 8 removed the stale generic “LL side” wording from the benchmark and its
local README.

The benchmark now reports:

- `workflow`
- `refactor_public_ms`
- `refactor_csc_ms`
- `solve_public_ms`
- `solve_csc_ms`
- `speedup_refactor`
- `res_public`
- `res_csc`

Interpretation:

- the benchmark now matches the live repo state where the public repeated-run
  path is not always a pure linked-list-only story
- the local benchmark docs now describe the SPD and indefinite modes directly
  instead of overclaiming one generic surface

#### 3. The first indefinite benchmark run exposed a real LDL^T permutation bug in the shared solve wrapper

The first `--indefinite-kkt` run produced:

- CSC residual at round-off
- public repeated-run residual badly wrong

That turned out not to be a benchmark mistake. It exposed a real contract bug:

- `sparse_factor_solve(...)` was still pre/post-applying `analysis->perm` for
  LDL^T factors
- but the factor object may already carry a final composed symmetric
  permutation

Interpretation:

- Day 8 became both a benchmark-proof batch and a correctness batch
- the benchmark did exactly what it should do here: surface a real mismatch
  between the public repeated-run contract and the internal permutation state

#### 4. Day 8 fixed the LDL^T analysis/factor permutation contract so reordered indefinite repeated runs solve correctly

The final Day 8 landing tightened `src/sparse_analysis.c` in two places:

- `sparse_factor_solve(...)`
  - LDL^T factors now skip the extra outer `analysis->perm` shuffle because
    `sparse_ldlt_solve(...)` already owns the final factor permutation
- small-path LDL^T `sparse_factor_numeric(...)`
  - after factoring an already permuted working copy, the factor object's
    stored permutation is now composed back into original matrix coordinates

Interpretation:

- LDL^T factor objects now carry one consistent permutation contract across:
  - below-threshold linked-list analysis/factor path
  - above-threshold CSC analysis/factor path
  - later repeated-run refactor path
- the public shared solve wrapper no longer double-permutes reordered
  indefinite workloads

#### 5. Sprint 53 now has direct regression proof for the reordered indefinite repeated-run case that Day 8 fixed

Day 8 added:

- `test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt_amd`

That regression proves:

- `SPARSE_FACTOR_LDLT` analysis with `SPARSE_REORDER_AMD`
- first factorization on a KKT workload
- same-pattern indefinite refactor on perturbed values
- `sparse_factor_solve(...)` still recovers the exact known solution after the
  refactor

The existing proof also remained relevant:

- `test_public_lifecycle_ldlt_refactor_same_pattern_indefinite_kkt`
  - same contract under `SPARSE_REORDER_NONE`

Interpretation:

- Day 8 closed the permutation bug with a direct future-facing regression, not
  only with a benchmark observation

#### 6. The measured Day 8 benchmark outputs now stay numerically clean on both the SPD and indefinite proof surfaces

Representative Day 8 results after the bug fix:

- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `workflow = chol_spd`
  - `analyze_ms = 0.313`
  - `refactor_public_ms = 0.157`
  - `refactor_csc_ms = 0.110`
  - `solve_public_ms = 0.010`
  - `solve_csc_ms = 0.004`
  - `speedup_refactor = 1.43x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`
- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`
  - `workflow = ldlt_kkt`
  - `analyze_ms = 0.124`
  - `refactor_public_ms = 0.172`
  - `refactor_csc_ms = 0.137`
  - `solve_public_ms = 0.006`
  - `solve_csc_ms = 0.002`
  - `speedup_refactor = 1.26x`
  - `res_public = 2.96e-16`
  - `res_csc = 2.96e-16`

Interpretation:

- the new indefinite benchmark now proves the intended same-pattern workload at
  round-off accuracy on both sides
- Day 8 leaves measured evidence rather than only a narrative claim

#### 7. The focused follow-on proof surfaces stayed clean

Focused reruns after the bug fix:

- `./build/test_integration`
  - `36 / 36`
- `./build/test_etree`
  - `97 / 97`
- `./build/example_analysis`
  - residual stayed `4.44e-16`

Interpretation:

- the Day 8 LDL^T permutation fix did not only help the benchmark
- it also kept the higher-value integration and analysis/factor proof surfaces
  clean

#### 8. Day 8 closes the benchmark-proof gap and leaves a cleaner Day 9 audit starting point

What Day 8 materially improved:

- added a real indefinite factor-many benchmark mode
- made the repeated-run benchmark wording more truthful
- exposed and fixed a real reordered indefinite solve bug
- added missing AMD repeated-run regression proof

What Day 8 intentionally did not solve:

- broader Cholesky / LDL^T dispatch reconciliation is still later work
- public top-level README wording is still later work
- the scalar BK pre-pass still remains the authoritative indefinite
  permutation-resolution step

Interpretation:

- Day 9 can now audit the remaining adoption/documentation queue from a
  measured and correctness-checked benchmark surface rather than from a proof
  gap

## Day 9: Cholesky / LDL^T Dispatch Reconciliation Audit

### Commands run

- `sed -n '332,380p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
- `sed -n '280,360p' README.md`
- `sed -n '1,220p' benchmarks/README.md`
- `sed -n '1,220p' include/sparse_ldlt.h`
- `sed -n '180,360p' include/sparse_analysis.h`
- `rg -n "bench_refactor_csc|CSC|LDL\\^T|Cholesky|used_csc_path|repeated-run" README.md include/sparse_ldlt.h include/sparse_cholesky.h tests/test_chol_csc.c tests/test_ldlt_csc.c tests/test_ldlt.c`
- `sed -n '1,260p' include/sparse_cholesky.h`
- `sed -n '1500,1635p' tests/test_ldlt_csc.c`
- `sed -n '2400,2495p' tests/test_ldlt.c`

### Findings

#### 1. The remaining Sprint 53 reconciliation queue is now mostly a top-level README problem, not a code or test problem

The Day 9 audit checked the live dispatch story across:

- `README.md`
- `benchmarks/README.md`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`
- CSC-specific dispatch tests

Result:

- `README.md` is now the main place where the Cholesky / LDL^T CSC story is
  still lower-resolution than the landed code
- the other inspected surfaces are already more aligned than the top-level
  README

Interpretation:

- Day 10 should not reopen Sprint 53 implementation work
- it should land one bounded public-story clarification batch instead

#### 2. `include/sparse_ldlt.h` is already the strongest authoritative wording for the LDL^T CSC dispatch contract

The LDL^T header already states the critical Sprint 53 distinctions:

- AUTO vs forced backend selection
- forced CSC means the CSC pipeline, not an unconditional promise of batched
  supernodal completion
- `used_csc_path` reports the selected numeric path
- the `n == 0` empty-matrix exception stays explicit

Interpretation:

- Day 10 should not spend effort rewriting the LDL^T header again unless a
  real contradiction appears while touching the primary target

#### 3. `benchmarks/README.md` is aligned enough after Day 8

The benchmark-local README now correctly distinguishes:

- `bench_refactor`
  - Cholesky analyze-once / refactor-many proof
- `bench_refactor_csc`
  - default SPD / Cholesky repeated-run proof
  - optional indefinite LDL^T KKT repeated-run proof
  - public repeated-run path vs direct CSC completion path

Interpretation:

- the benchmark-local docs no longer look like the main remaining drift source
- Day 10 should leave them alone unless a primary README edit forces one tiny
  consistency tweak

#### 4. The CSC-specific tests are already describing the LDL^T pipeline accurately enough

The inspected test commentary already reflects the landed layered LDL^T story:

- scalar BK pre-pass remains the authoritative indefinite
  permutation-resolution step
- the CSC pipeline may retain batched completion or resolved scalar fallback
- helper contract violations no longer silently alias into fallback

Interpretation:

- tests are already a stronger description of the live LDL^T CSC contract than
  the top-level README
- they are not the right Day 10 target by default

#### 5. The real remaining drift is that `README.md` still compresses Cholesky and LDL^T dispatch into one story that is now too coarse

What the live code now supports:

- Cholesky
  - simpler size-based linked-list vs CSC backend selection
  - forced CSC means the CSC backend
- LDL^T
  - similar outer size-based dispatch
  - but forced CSC means the CSC pipeline
  - that pipeline still begins from the scalar BK pre-pass
  - completion may retain the batched path or fall back to the resolved scalar
    factor

Interpretation:

- Day 10 needs to clarify the top-level dispatch story without pretending the
  two families are internally identical

#### 6. Day 8's new indefinite factor-many benchmark proof is still under-centered in the top-level README

After Day 8:

- `bench_refactor_csc` now has a real `--indefinite-kkt` LDL^T mode
- that mode measures the public repeated-run path vs the direct CSC completion
  path
- the benchmark now closes at round-off residuals after the Day 8 LDL^T
  permutation fix

Interpretation:

- Day 10 should probably add a small README-level benchmark-story
  reconciliation
- it should point readers to `benchmarks/README.md` rather than restating the
  full benchmark contract

### Ranked Day 10 targets

#### Primary target

- `README.md`

Why:

- it is now the highest-visibility place where the dispatch story still lags
  the landed CSC reality

What to do:

- tighten the Cholesky / LDL^T CSC wording
- mention the new indefinite repeated-run proof surface briefly
- keep the detailed truth in:
  - `include/sparse_ldlt.h`
  - `benchmarks/README.md`

#### Secondary target only if truly needed

- one very small header touch in:
  - `include/sparse_cholesky.h`
  - or `include/sparse_analysis.h`

Constraint:

- only if the README clarification would otherwise contradict a local header
  sentence

### Explicit non-goals

- no tutorial-scale rewrite
- no broad example rewrite
- no benchmark-framework redesign
- no new public direct-solver API work
- no fake symmetry rewrite that hides the real Cholesky vs LDL^T CSC
  differences

### Day 9 outcome

Sprint 53's remaining reconciliation queue is now concrete:

- main target:
  - `README.md`
- already aligned enough:
  - `benchmarks/README.md`
  - `include/sparse_ldlt.h`
  - CSC-specific dispatch tests
- optional tiny follow-on only if needed:
  - `include/sparse_cholesky.h`
  - `include/sparse_analysis.h`

This was a docs-only audit day, so I did not rerun `make format`, `make lint`,
`make test`, or `make quality-review-full`.

## Day 10: Dispatch Reconciliation Batch

### Commands run

- `sed -n '381,440p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
- `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day9-cholesky-ldlt-dispatch-reconciliation-audit.md`
- `sed -n '280,380p' README.md`
- `rg -n "bench_refactor_csc|Cholesky|LDL\\^T|CSC pipeline|used_csc_path|Repeated-Run Direct Workflow|direct workflow|factor-many|indefinite" README.md`
- `sed -n '510,625p' README.md`
- `sed -n '1,140p' benchmarks/README.md`
- `rg -n "bench_refactor_csc|indefinite-kkt|Bunch-Kaufman|CSC pipeline|batched path|resolved scalar-prepass factor|Cholesky CSC dispatch|LDL\\^T CSC dispatch" README.md benchmarks/README.md include/sparse_ldlt.h include/sparse_cholesky.h`
- `sed -n '548,648p' README.md`
- `wc -l README.md benchmarks/README.md include/sparse_ldlt.h include/sparse_cholesky.h`

### Findings

#### 1. Day 9's target ranking held: only `README.md` needed a real follow-through patch

The Day 10 batch rechecked the live top-level and local docs surfaces before
editing:

- `README.md`
- `benchmarks/README.md`
- `include/sparse_ldlt.h`
- `include/sparse_cholesky.h`

Result:

- the benchmark-local README and headers were already aligned enough
- the remaining real drift was still concentrated in the top-level README

Interpretation:

- Day 10 stayed on the Day 9 primary target without reopening secondary
  surfaces

#### 2. The top-level Cholesky CSC story is now explicit about being the simpler family-local case

The Day 10 README patch now says the repeated-run CSC story on the Cholesky
side is intentionally simple:

- AUTO picks linked-list vs CSC by size
- forcing CSC means the CSC backend directly
- the highest-signal repeated-run proof surfaces are:
  - `bench_refactor`
  - default SPD mode in `bench_refactor_csc`

Interpretation:

- the top-level README no longer implies Cholesky and LDL^T share one identical
  internal CSC model

#### 3. The stale pre-Sprint-53 LDL^T CSC wording is now gone

Before Day 10, the LDL^T CSC section still read as if the analysis-aware
follow-through was mainly a future Sprint 20 direction.

The Day 10 patch replaced that stale wording with the current bounded Sprint 53
contract:

- forcing CSC means the CSC pipeline, not a blanket promise that the batched
  completion path wins every indefinite input
- the scalar Bunch-Kaufman pre-pass remains the authoritative indefinite
  permutation-resolution step
- once the CSC pipeline is selected, completion may:
  - retain the batched path
  - or fall back to the resolved scalar-prepass factor when the batched path
    rejects the cached pivot pattern

Interpretation:

- the README now matches the live LDL^T dispatch layering instead of the older
  pre-follow-through mental model

#### 4. Day 8's new indefinite factor-many proof is now visible at the top-level README layer

The Day 10 patch added a compact README-level handoff for:

- `bench_refactor_csc --indefinite-kkt`

It now states that this mode:

- measures the public repeated-run LDL^T path against the direct
  resolved-analysis CSC completion path
- uses a bounded same-pattern KKT workload
- closes at round-off residuals on both sides after the Sprint 53
  permutation-contract fix

Interpretation:

- the indefinite repeated-run proof is no longer hidden only in benchmark-local
  docs and sprint artifacts

#### 5. No README-driven contradiction forced a header follow-on

The targeted Day 10 sanity checks confirmed:

- `include/sparse_ldlt.h` already used the same CSC-pipeline vocabulary
- `include/sparse_cholesky.h` already matched the simpler Cholesky CSC story
- `benchmarks/README.md` already matched the new README benchmark references

Interpretation:

- Day 10 stayed narrowly bounded to the README
- no header churn was justified

### Day 10 output

Landed bounded reconciliation in:

- `README.md`

Recorded artifact:

- `docs/planning/EPIC_5/SPRINT_53/artifacts/day10-dispatch-reconciliation-batch.md`

### Targeted sanity checks

Because this was docs-only, I did not run `make format`, `make lint`,
`make test`, or `make quality-review-full`.

I ran targeted Day 10 wording checks instead:

- `rg -n "bench_refactor_csc|indefinite-kkt|Bunch-Kaufman|CSC pipeline|batched path|resolved scalar-prepass factor|Cholesky CSC dispatch|LDL\\^T CSC dispatch" README.md benchmarks/README.md include/sparse_ldlt.h include/sparse_cholesky.h`
- `sed -n '548,648p' README.md`
- `wc -l README.md benchmarks/README.md include/sparse_ldlt.h include/sparse_cholesky.h`

All were clean.

## Day 11: Regression Expansion Batch

### Commands run

- `sed -n '440,520p' docs/planning/EPIC_5/SPRINT_53/PLAN.md`
- `tail -n 220 docs/planning/EPIC_5/SPRINT_53/WORKING_NOTES.md`
- `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_53/artifacts/day10-dispatch-reconciliation-batch.md`
- `rg -n "Sprint 53|ldlt|chol|refactor|same-pattern|indefinite|used_csc_path|pipeline|bench_refactor_csc|test_public_lifecycle|analysis-aware" tests src include benchmarks README.md docs/planning/EPIC_5/SPRINT_53 -g '!build'`
- `rg -n "test_public_lifecycle_ldlt|used_csc_path|refactor_same_pattern|indefinite_kkt|analysis-aware|resolved_analysis|csc pipeline|mismatch|nnz drift|preserve|old factors" tests/test_integration.c tests/test_ldlt.c tests/test_ldlt_csc.c tests/test_sprint20_integration.c`
- `sed -n '820,1225p' tests/test_integration.c`
- `sed -n '1225,1365p' tests/test_integration.c`
- `sed -n '430,520p' tests/test_integration.c`
- `make format`
- `make lint`
- `make test`
- `./build/test_integration`
- `./build/test_ldlt_csc`
- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`

### Findings

#### 1. The strongest remaining proof gap was the bounded failure contract on the high-value indefinite repeated-run path

Before Day 11, Sprint 53 already had:

- one-shot vs explicit-analysis indefinite KKT parity
- same-pattern indefinite refactor success
- reordered indefinite repeated-run success
- measured indefinite factor-many benchmark proof

Sprint 52 already had the generic failure-side proof on the SPD path:

- cheap `nnz`-drift rejection
- old-factor preservation on failed refactor

What Sprint 53 still lacked was that same bounded failure-side proof on the
main above-threshold indefinite KKT repeated-run path.

Interpretation:

- Day 11 should add one focused regression on that path instead of broad new
  CSC coverage

#### 2. Day 11 added the missing indefinite `nnz`-drift + old-factor-preservation proof

Added in `tests/test_integration.c`:

- `test_public_lifecycle_ldlt_refactor_rejects_nnz_drift_and_preserves_old_factors_amd`

What it proves:

1. analyze `kkt-150` with:
   - `SPARSE_FACTOR_LDLT`
   - `SPARSE_REORDER_AMD`
2. factor once through the public repeated-run path
3. remove one symmetric coupling pair from a copied matrix to create obvious
   `nnz` drift
4. `sparse_refactor_numeric(...)` returns:
   - `SPARSE_ERR_BADARG`
5. the old factors remain valid for the original RHS/solution pair afterward

Interpretation:

- the cheap gross-structure guard is now directly proved on the high-value
  indefinite CSC path
- old-factor preservation is also now directly proved on that same path

#### 3. This stayed tightly bounded to proof, not implementation

Day 11 changed:

- `tests/test_integration.c`

Day 11 did not change:

- `src/`
- public headers
- benchmarks
- README

Interpretation:

- this was the intended Day 11 shape: close one real proof gap without
  reopening Sprint 53's implementation or docs queue

### Validation

Because `tests/test_integration.c` changed, I ran the full required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Focused Day 11 follow-ons also passed:

- `./build/test_integration`
  - `37 / 37`
- `./build/test_ldlt_csc`
  - `96 / 96`
- `./build/bench_refactor_csc --indefinite-kkt --repeat 1`
  - `workflow = ldlt_kkt`
  - `speedup_refactor = 1.26x`
  - `res_public = 2.96e-16`
  - `res_csc = 2.96e-16`

### Day 11 outcome

Sprint 53's indefinite repeated-run proof surface is now better balanced:

- success-path proof already existed
- bounded failure-path proof now exists too
- the measured indefinite benchmark surface stayed numerically unchanged and
  healthy
