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
