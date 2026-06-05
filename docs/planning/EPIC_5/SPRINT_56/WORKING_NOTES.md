# Sprint 56 Working Notes

## Day 1

**Objective:** Turn the Sprint 56 project-plan scope plus the Sprint 55
validated large-source decomposition close state into a concrete Phase 2
starting point by confirming the preserved reviewed baseline, naming the
Sprint 56 CSC/SVD implementation workstreams explicitly, and defining the
authoritative direct-solver and dense-algorithm implementation, proof, and
caller-surface hotspots before any extraction work begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 56 project-plan source and the new sprint plan:
   - `sed -n '219,246p' docs/planning/EPIC_5/PROJECT_PLAN.md`
   - `sed -n '1,120p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
3. Re-read the strongest inherited Phase 1 closeout sources:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_55/artifacts/day14-closeout-and-handoff.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_55/RETROSPECTIVE.md`
4. Re-read the Epic 5 large-source review/todo guidance for the remaining
   CSC/SVD queue:
   - `rg -n "ldlt_csc|chol_csc|sparse_svd\\.c|Large-Source Decomposition Phase 2|large-source|decomposition" docs/planning/EPIC_5/reviews/review-codex-2026-05-31.md docs/planning/EPIC_5/reviews/todo-codex-2026-05-31.md docs/planning/EPIC_5/PROJECT_PLAN.md docs/planning/EPIC_5/SPRINT_55/RETROSPECTIVE.md`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed wrapper surface:
   - `make -n quality-review-full`
7. Measure the live CSC direct-solver, SVD, proof, and caller-surface
   hotspots:
   - `wc -l src/sparse_ldlt_csc.c src/sparse_chol_csc.c src/sparse_svd.c src/sparse_ldlt_csc_internal.h src/sparse_chol_csc_internal.h src/sparse_svd_internal.h tests/test_ldlt_csc.c tests/test_chol_csc.c tests/test_svd.c tests/test_integration.c benchmarks/bench_refactor_csc.c examples/example_analysis.c include/sparse_ldlt.h include/sparse_cholesky.h include/sparse_svd.h README.md docs/maintainer_guide.md`

### Day 1 Findings

#### 1. Sprint 56 starts from a validated decomposition baseline, not from renewed solver-lifecycle or API design work

The inherited starting state is already explicit and stable:

- Sprint 55 closed with:
  - bounded eigensolver decomposition complete enough to reduce
    `src/sparse_eigs.c` from `3233` to `1534`
  - bounded iterative decomposition complete enough to reduce
    `src/sparse_iterative.c` from `2377` to `1985`
  - no public API redesign
  - no repeated-run solver support-boundary drift
- Sprint 55 also closed from:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- the inherited caller-facing contract remains unchanged:
  - one-shot APIs remain first-class entry points
  - repeated-run lifecycle support remains the validated Sprint 50-54 shape

Interpretation:

- Sprint 56 is not a public lifecycle redesign sprint
- Sprint 56 is not a validation-recovery sprint
- Sprint 56 is a bounded maintainability and ownership sprint

#### 2. The strongest local reviewed baseline remains unchanged and should stay visible on all substantial decomposition batches

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

And the wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

Interpretation:

- Sprint 56 should keep using the exact `strongest local reviewed baseline`
  phrasing
- substantial extraction batches should continue treating the reviewed CMake
  count and parity contract as truthfulness anchors

#### 3. The Epic 5 large-source review queue is now concentrated in the CSC direct-solver production files plus `src/sparse_svd.c`

The Epic 5 review and todo notes already pointed to:

- `src/sparse_ldlt_csc.c`
- `src/sparse_chol_csc.c`
- `src/sparse_svd.c`

The live repo state confirms that the queue is still current:

- `src/sparse_ldlt_csc.c` = `2723`
- `src/sparse_chol_csc.c` = `2194`
- `src/sparse_svd.c` = `1728`

Interpretation:

- Sprint 56 should treat the review queue as still live, not historical
- `src/sparse_ldlt_csc.c` remains the clearest first direct-solver extraction
  target after the Sprint 53 CSC completion work
- `src/sparse_chol_csc.c` and `src/sparse_svd.c` remain large enough that
  ownership improvement should still dominate over cosmetic cleanup

#### 4. The real Sprint 56 queue is decomposition-first, not feature-first

The Sprint 56 plan items and live repo state narrow to seven bounded work
classes:

1. `sparse_ldlt_csc.c` residual audit
2. LDLT CSC decomposition batch
3. `sparse_chol_csc.c` residual audit
4. Cholesky CSC decomposition batch
5. `sparse_svd.c` maintainability batch
6. touched-doc and comment reconciliation
7. validation and closeout

Interpretation:

- Sprint 56 should reduce ownership ambiguity in the remaining large CSC/SVD
  files before widening any other Epic 5 queue
- the sprint should explicitly prefer helper-vs-orchestration splits over
  generic “split by size” edits

#### 5. The live hotspot map is already concentrated enough to name directly

The main touched surfaces are clear before any extraction work begins:

- public headers:
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_svd.h` = `257`
- main implementations:
  - `src/sparse_ldlt_csc.c` = `2723`
  - `src/sparse_chol_csc.c` = `2194`
  - `src/sparse_svd.c` = `1728`
  - `src/sparse_ldlt_csc_internal.h` = `877`
  - `src/sparse_chol_csc_internal.h` = `994`
  - `src/sparse_svd_internal.h` = `21`
- strongest proof surfaces:
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_chol_csc.c` = `4643`
  - `tests/test_svd.c` = `3746`
  - `tests/test_integration.c` = `1803`
  - `benchmarks/bench_refactor_csc.c` = `611`
- strongest caller-facing adoption surface:
  - `examples/example_analysis.c` = `210`
  - `README.md` = `987`
  - `docs/maintainer_guide.md` = `294`

Interpretation:

- the strongest implementation risk seams are now concentrated in the CSC
  direct-solver files, not in the already-split iterative/eigensolver fronts
- the proof surfaces for CSC and SVD are also large enough that extraction
  work must preserve test and benchmark parity deliberately

#### 6. The inherited direct-solver lifecycle and solver support boundary is already fixed, which gives Sprint 56 a clean non-goal fence

The inherited public and lifecycle boundary remains:

- one-shot APIs remain first-class peer entry points
- the analysis/factors repeated direct-run path remains the validated direct
  lifecycle shape
- repeated-run solver handles remain the validated Sprint 54 support set
- no raw CSC/native storage exposure
- no broad public direct-handle redesign
- no broad solver-family or dense-algorithm API expansion

Interpretation:

- Sprint 56 should preserve those already-validated boundaries while changing
  implementation ownership underneath them
- public API expansion is not the right success criterion for this sprint

#### 7. Comment and wording normalization remains a real Sprint 56 work item, but only after ownership seams land

Sprint 55 already proved the right cleanup style:

- preserve durable algorithm and ownership commentary
- remove stale sprint-history narrative in touched permanent code

Interpretation:

- Sprint 56 should again treat comment cleanup as a bounded implementation
  quality task, not as optional polish
- the right order is still:
  - land ownership seams first
  - normalize touched comments and any coupled wording afterward

## Day 1 Close

Sprint 56 now has an explicit starting point:

- preserved reviewed baseline
- inherited validated decomposition and public-contract fence
- named CSC direct-solver and SVD maintainability hotspots
- clear decomposition-first workstreams
- explicit non-goal fence against public API redesign

That is enough to move to the Day 2 validation and touched-surface recheck
without reopening Sprint 50-55 public contract decisions.

## Day 2

**Objective:** Reconfirm the maintained reviewed baseline and truthfulness
anchors Sprint 56 must preserve, then define the smallest authoritative
validation boundary for the later CSC direct-solver and SVD extraction days
and the high-signal rerun set those code-touch batches should use.

### Commands Run

1. Re-read the Sprint 56 Day 2 plan item and the current sprint notes:
   - `sed -n '78,122p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_56/WORKING_NOTES.md`
2. Reconfirm the maintained reviewed CMake truthfulness anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
3. Reconfirm the maintained reviewed wrapper authority surface:
   - `make -n quality-review-full`
4. Re-read the live quality-contract wording sources:
   - `rg -n "strongest local reviewed baseline|quality-review-full|quality-review-cmake|deadcode-check" README.md docs/maintainer_guide.md Makefile .github/workflows -g '!build'`
5. Reconfirm the main Sprint 56 follow-on binaries already present in the
   build tree:
   - `ls build/test_chol_csc build/test_ldlt_csc build/test_cholesky build/test_ldlt build/test_etree build/test_svd build/test_integration build/bench_refactor_csc build/example_analysis`
6. Measure the live size of those main proof/adoption surfaces:
   - `wc -l tests/test_chol_csc.c tests/test_ldlt_csc.c tests/test_svd.c tests/test_integration.c benchmarks/bench_refactor_csc.c examples/example_analysis.c`

### Day 2 Findings

#### 1. The strongest local reviewed baseline and truthfulness anchors remain exact

The maintained Sprint 56 baseline remains:

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

- Sprint 56 should keep using the exact `strongest local reviewed baseline`
  phrasing
- the reviewed CMake count and Makefile/CMake parity contract remain the
  authoritative truthfulness anchors for later extraction days

#### 2. The later decomposition code-day gate is simple and should stay explicit

The mandatory gate for later `*.c` / `*.h` decomposition work remains:

- `make format`
- `make lint`
- `make test`

And the stronger default for substantial implementation ownership batches
remains:

- `make quality-review-full`

Interpretation:

- docs-only audit/design/summary days do not need the full code-day gate
- substantial extraction batches should continue to run both the direct gate
  and the stronger reviewed baseline path

#### 3. The live quality-contract wording still matches the maintained split across README, maintainer guide, and Makefile

The quality-contract wording remains aligned across the main authority
surfaces:

- `README.md`
  - user-facing command map
  - strongest local reviewed baseline wording
  - explicit `deadcode-check` completeness-gate wording
- `docs/maintainer_guide.md`
  - maintainer-facing authority framing
  - reviewed CMake parity anchor
  - dead-code interpretation boundary
- `Makefile`
  - executable reviewed-target authority
  - current rerun guidance
  - current test-count parity checks

Interpretation:

- Sprint 56 does not need to reopen any quality-contract wording work on Day 2
- the maintained reviewed baseline language is already stable enough to carry
  forward unchanged

#### 4. The high-signal Sprint 56 rerun set is now fixed explicitly from the live build tree

The main Sprint 56 follow-on binaries already present in `build/` are:

- `./build/test_chol_csc`
- `./build/test_ldlt_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_svd`
- `./build/test_integration`
- `./build/bench_refactor_csc`
- `./build/example_analysis`

Interpretation:

- Sprint 56 can keep its rerun set focused on the CSC direct-solver and SVD
  families actually touched by the large-source decomposition work
- no broader default rerun set is needed on Day 2

#### 5. The proof and adoption surfaces are now large enough that parity preservation is part of the extraction work itself

The live proof/adoption surface sizes are now:

- `tests/test_chol_csc.c` = `4643`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_svd.c` = `3746`
- `tests/test_integration.c` = `1803`
- `benchmarks/bench_refactor_csc.c` = `611`
- `examples/example_analysis.c` = `210`

Interpretation:

- Sprint 56 extraction work should assume that proof-surface legibility and
  parity matter alongside implementation-file size reduction
- the rerun set is not ceremonial; it is the main defense against accidental
  behavior drift while ownership moves under the hood

## Day 2 Close

Sprint 56 now has an explicit validation boundary:

- preserved reviewed baseline wording
- exact reviewed CMake count anchor
- explicit code-day gate
- explicit stronger reviewed-baseline default
- authoritative CSC direct-solver and SVD rerun set from the live build tree

That is enough to move to the Day 3 `sparse_ldlt_csc.c` residual ownership
audit without any remaining ambiguity around validation expectations.

## Day 3

**Objective:** Reduce `src/sparse_ldlt_csc.c` to a concrete extraction map by
separating the live LDLT CSC ownership bands, ranking the real bounded seam
options, and fixing the strongest first extraction target before any code
movement begins.

### Commands Run

1. Re-read the Sprint 56 Day 3 plan item and current sprint notes:
   - `sed -n '122,159p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
   - `sed -n '337,520p' docs/planning/EPIC_5/SPRINT_56/WORKING_NOTES.md`
2. Re-read the current private LDLT CSC header surface:
   - `sed -n '1,260p' src/sparse_ldlt_csc_internal.h`
3. Re-read the front portion of the production file and file-level design note:
   - `sed -n '1,260p' src/sparse_ldlt_csc.c`
4. Build a function map for the whole translation unit:
   - `rg -n "^static |^sparse_err_t |^int |^void |^double |^idx_t " src/sparse_ldlt_csc.c`
5. Re-read the strongest direct proof surface:
   - `sed -n '1,260p' tests/test_ldlt_csc.c`
6. Re-read the CSC repeated-run benchmark surface:
   - `sed -n '1,260p' benchmarks/bench_refactor_csc.c`

### Day 3 Findings

#### 1. The `src/sparse_ldlt_csc.c` problem now reduces cleanly to five ownership bands instead of one generic large-file target

The live function map breaks into five clear bands:

1. lifecycle / storage / structural conversion
   - `ldlt_csc_free(...)`
   - `ldlt_csc_alloc(...)`
   - `ldlt_csc_row_adj_append(...)`
   - `ldlt_csc_detect_supernodes(...)`
   - `ldlt_csc_from_sparse(...)`
   - `ldlt_csc_from_sparse_with_analysis(...)`
   - `ldlt_csc_to_sparse(...)`
   - `ldlt_csc_writeback_to_ldlt(...)`
   - `ldlt_csc_validate(...)`
2. legacy wrapper and compatibility path
   - `csc_to_full_symmetric_matrix(...)`
   - `ldlt_csc_eliminate_wrapper(...)`
3. scalar/native LDLT CSC kernel and dispatch core
   - `ldlt_csc_symmetric_swap(...)`
   - `ldlt_csc_eliminate(...)`
   - workspace alloc/free
   - Bunch-Kaufman helpers
   - scatter / lookup / cmod / one-step elimination
   - `ldlt_csc_eliminate_native(...)`
   - `ldlt_csc_solve(...)`
4. supernodal LDLT CSC helper cluster
   - `ldlt_csc_supernode_extract(...)`
   - `ldlt_csc_supernode_writeback(...)`
   - `ldlt_csc_supernode_eliminate_diag(...)`
   - `ldlt_csc_supernode_eliminate_panel(...)`
   - `ldlt_csc_eliminate_supernodal(...)`
5. small local helper seams serving the larger clusters
   - `ldlt_csc_bsearch_row_map(...)`
   - row-adjacency and dense-column clear helpers

Interpretation:

- Sprint 56 no longer needs to talk about `sparse_ldlt_csc.c` as one monolith
- the real choice is which owned cluster to move first without weakening the
  scalar/native CSC kernel or the public compatibility path

#### 2. The strongest first extraction target is the supernodal helper cluster, not the scalar/native kernel

The supernodal cluster is the clearest first owned slice because it already
reads as a bounded subsystem:

- extraction and dense writeback
- dense diagonal block factor step
- dense panel solve step
- top-level supernodal elimination driver

Why it is stronger than the scalar/native kernel for Batch 1:

- it is already grouped contiguously near the end of the file
- it has its own vocabulary and helper surface
- it is easier to move without reopening the main scalar Bunch-Kaufman control
  loop
- it is directly exercised by the CSC-specific tests, which gives clearer proof
  boundaries after extraction

Interpretation:

- the supernodal cluster is the highest-value first extraction seam because it
  offers real ownership reduction without forcing a riskier rewrite of the
  scalar/native elimination heart of the file

#### 3. The scalar/native kernel is the largest residual seam, but it is a better second target than first target

The middle scalar/native band remains the biggest residual ownership mass:

- symmetric swap
- workspace lifecycle
- Bunch-Kaufman scan helpers
- scatter / lookup / cmod
- one-step elimination
- native elimination driver
- solve path

This band is important, but it is more intertwined than the supernodal slice:

- `ldlt_csc_eliminate_native(...)` depends on multiple local helpers and the
  row-adjacency population path
- `ldlt_csc_symmetric_swap(...)` and cmod behavior are tightly coupled to the
  scalar elimination path
- the solve path carries user-visible correctness expectations, so a purely
  mechanical extraction would be risky

Interpretation:

- this is the strongest second-phase seam after a cleaner supernodal-first
  extraction has reduced file size and clarified the remaining core
- Sprint 56 should not start by splitting the scalar kernel just because it is
  the biggest line-count region

#### 4. Conversion / validation / writeback is real ownership, but a weak first extraction target

The top lifecycle/conversion band is substantial, but it is not the best first
batch:

- it contains real owned logic
- it touches public-facing conversion semantics indirectly
- it is less behaviorally cohesive than the supernodal cluster

Moving it first would risk a lower-value split:

- more files
- less line-count relief in the numerically dense backend section
- weaker improvement to the main implementation readability problem

Interpretation:

- this band is a valid later cleanup seam
- it should not outrank the supernodal cluster or the scalar/native kernel in
  the first extraction order

#### 5. The wrapper path is intentionally secondary and should not drive the decomposition order

The wrapper/compatibility path is narrow:

- full symmetric linked-list expansion helper
- wrapper elimination path retained for comparison and regression purposes

Its main role is compatibility and A/B proof, not primary ownership:

- it is already bounded
- it is not where most maintainability weight lives
- extracting it first would reduce little risk and little line count

Interpretation:

- keep it visible as a seam
- do not let it distort the first extraction order

#### 6. The proof surfaces argue for CSC-native ownership bands, not arbitrary utility-file slicing

The main proof surfaces remain:

- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_integration.c` still carries public repeated-run direct-lifecycle
  coverage
- `benchmarks/bench_refactor_csc.c` = `611`

And the benchmark is especially informative:

- it proves both SPD Cholesky and indefinite LDLT repeated-run CSC paths
- it names the direct CSC completion seam explicitly

Interpretation:

- Sprint 56 should extract ownership bands that those tests and benchmarks
  already imply exist
- a utility-first split that cuts across these proof boundaries would be harder
  to validate and explain

#### 7. The ranked extraction order is now explicit

Ranked `src/sparse_ldlt_csc.c` target order from strongest to weakest:

1. supernodal LDLT CSC helper cluster
2. scalar/native elimination kernel cluster
3. conversion / validation / writeback cluster
4. wrapper/compatibility cluster
5. small residual local helper cleanup

That gives Sprint 56 a concrete first-batch recommendation:

- first extraction target:
  - supernodal LDLT CSC helper cluster
- keep in the main file initially:
  - public-ish lifecycle/conversion entry points
  - wrapper compatibility path
  - scalar/native Bunch-Kaufman core
  - solve path

## Day 3 Close

Sprint 56 now has a concrete LDLT CSC decomposition map:

- named ownership bands
- a ranked extraction order
- an explicit first target centered on the supernodal helper cluster
- a clear reason not to start with the scalar/native kernel despite its size

That is enough to move to the Day 4 LDLT CSC decomposition design without
leaving the first batch boundary ambiguous.

## Day 4

**Objective:** Freeze the first LDLT CSC extraction boundary before editing
permanent implementation files by turning the Day 3 supernodal-first ranking
into an exact file split, declaration strategy, and preserved-behavior
checklist.

### Commands Run

1. Re-read the Sprint 56 Day 4 plan item and the Day 3 closing state:
   - `sed -n '160,194p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
   - `sed -n '520,620p' docs/planning/EPIC_5/SPRINT_56/WORKING_NOTES.md`
2. Re-read the earlier decomposition-design artifact shape for reference:
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_55/artifacts/day4-eigensolver-decomposition-batch1-design.md`
3. Re-read the Day 3 LDLT CSC seam audit:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_56/artifacts/day3-ldlt-csc-residual-ownership-audit.md`
4. Re-read the selected supernodal helper cluster in the live production file:
   - `sed -n '2140,2615p' src/sparse_ldlt_csc.c`

### Day 4 Findings

#### 1. Sprint 56 Batch 1 should extract the supernodal LDLT CSC helper cluster into its own source file

The Day 3 ranking still holds after re-reading the concrete helper block:

- new file:
  - `src/sparse_ldlt_csc_supernodal.c`

The first moved function set should be:

- `ldlt_csc_supernode_extract(...)`
- `ldlt_csc_supernode_writeback(...)`
- `ldlt_csc_supernode_eliminate_diag(...)`
- `ldlt_csc_supernode_eliminate_panel(...)`
- `ldlt_csc_eliminate_supernodal(...)`

These functions already form a contiguous owned cluster and share the same
dense-workflow vocabulary:

- supernode extraction
- dense diagonal/panel work
- CSC writeback
- top-level supernodal elimination driver

Interpretation:

- this is a real ownership seam, not a cosmetic file move
- moving it first reduces `src/sparse_ldlt_csc.c` meaningfully while leaving
  the scalar/native kernel intact for a later phase

#### 2. The retained main file should keep the public-facing lifecycle and scalar/native kernel ownership

### Keep in `src/sparse_ldlt_csc.c`

- lifecycle / storage / structural conversion:
  - alloc / free
  - row-adjacency growth
  - supernode detection
  - sparse-to-CSC conversion
  - analysis-aware sparse-to-CSC conversion
  - CSC-to-sparse conversion
  - writeback to `sparse_ldlt_t`
  - validation
- wrapper / compatibility path:
  - linked-list expansion helper
  - wrapper elimination path
- scalar/native kernel core:
  - symmetric swap
  - workspace alloc/free
  - Bunch-Kaufman scan helpers
  - scatter / lookup / cmod helpers
  - one-step elimination
  - native elimination driver
  - solve path
- small residual local helpers not specific to the moved supernodal cluster

### Move to `src/sparse_ldlt_csc_supernodal.c`

- row-map lookup helper used only by the supernodal cluster:
  - `ldlt_csc_bsearch_row_map(...)`
- supernodal extract/writeback helpers
- supernodal diagonal-block eliminate helper
- supernodal panel eliminate helper
- supernodal elimination driver

Interpretation:

- Batch 1 should create one clean “supernodal-owned” file
- the retained main file should stay the home of the CSC-native scalar/control
  path

#### 3. Sprint 56 Phase 2 should keep using the existing internal header rather than open a new private-header taxonomy

The first batch should keep declarations in the existing:

- `src/sparse_ldlt_csc_internal.h`

Reason:

- Batch 1 already changes one major ownership axis:
  - source-file extraction
- adding a new private-header taxonomy in the same batch would combine:
  - source extraction
  - header redesign
- the current internal header is already the authoritative private contract for:
  - `LdltCsc`
  - `LdltCscWorkspace`
  - scalar/native helpers
  - supernodal helper declarations

Deferred by design:

- creation of a dedicated `src/sparse_ldlt_csc_supernodal_internal.h`
- broader narrowing or repartitioning of `src/sparse_ldlt_csc_internal.h`

Interpretation:

- Sprint 56 should separate source ownership first
- header taxonomy cleanup is a later maintainability choice, not a Day 5
  dependency

#### 4. The first batch must preserve native/wrapper semantics, permutation behavior, and CSC proof parity exactly

### Public and compatibility invariants

- `ldlt_csc_eliminate(...)` runtime override behavior unchanged
- native versus wrapper routing unchanged
- linked-list comparison path retained for regression and benchmark use
- no public header/API changes

### Numerical and storage invariants

- permutation semantics unchanged
- pivot-size and `D` / `D_offdiag` semantics unchanged
- row-adjacency assumptions unchanged
- residual and inertia behavior unchanged
- direct CSC repeated-run completion path unchanged

### Proof-surface invariants

- `tests/test_ldlt_csc.c` remains the primary direct proof surface
- `tests/test_integration.c` continues to prove public repeated-run direct
  lifecycle correctness
- `benchmarks/bench_refactor_csc.c` continues to prove:
  - SPD repeated-run Cholesky CSC behavior
  - indefinite repeated-run LDLT CSC behavior

Interpretation:

- the Day 5 extraction is successful only if the file boundary changes while
  all these behaviors remain exact

#### 5. The minimal comment policy for Batch 1 is again ownership-truthful rather than sprint-historical

Preserve:

- durable algorithm meaning
- supernodal/scalar ownership boundaries
- pivot/permutation invariants
- writeback/drop-threshold semantics

Reduce where touched:

- sprint chronology
- implementation-history narrative
- comments that explain landing order instead of present code truth

Do not try in Batch 1:

- repo-wide LDLT CSC comment normalization
- private-header taxonomy cleanup
- broad CSC doc rewriting

#### 6. The expected Day 5 touched-file set is now explicit

Primary expected touched set:

- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c` (new)
- `src/sparse_ldlt_csc_internal.h`
- `Makefile`
- `CMakeLists.txt`

Secondary touch only if truly needed:

- `tests/test_ldlt_csc.c`

Avoid by default:

- `include/sparse_ldlt.h`
- `tests/test_integration.c`
- `benchmarks/bench_refactor_csc.c`
- `src/sparse_chol_csc.c`

## Day 4 Close

Day 4 fixes the first LDLT CSC extraction boundary explicitly:

- move the supernodal LDLT CSC helper cluster into
  `src/sparse_ldlt_csc_supernodal.c`
- keep lifecycle/conversion, wrapper compatibility, and scalar/native kernel
  ownership in `src/sparse_ldlt_csc.c`
- reuse the existing internal header for Phase 2
- preserve the full native/wrapper/permutation/proof contract exactly

That gives Sprint 56 a concrete, bounded, maintainability-first Day 5 landing
plan.

## Day 5

**Objective:** Land the first bounded LDLT CSC source split by moving the
supernodal helper cluster out of `src/sparse_ldlt_csc.c` while preserving the
existing private-header contract, build surfaces, and CSC proof behavior.

### Commands Run

1. Re-read the Day 4 design boundary and the live moved-function set:
   - `sed -n '1,240p' docs/planning/EPIC_5/SPRINT_56/artifacts/day4-ldlt-csc-decomposition-design.md`
   - `rg -n "ldlt_csc_bsearch_row_map|ldlt_csc_supernode_extract|ldlt_csc_supernode_writeback|ldlt_csc_supernode_eliminate_diag|ldlt_csc_supernode_eliminate_panel|ldlt_csc_eliminate_supernodal" src/sparse_ldlt_csc.c src/sparse_ldlt_csc_internal.h`
   - `sed -n '1,220p' src/sparse_ldlt_csc_internal.h`
   - `nl -ba src/sparse_ldlt_csc.c | sed -n '2138,2725p'`
2. Re-read the current build-list patterns before editing:
   - `sed -n '40,85p' Makefile`
   - `sed -n '92,110p' CMakeLists.txt`
3. Land the bounded split:
   - `apply_patch` on:
     - `src/sparse_ldlt_csc.c`
     - `src/sparse_ldlt_csc_supernodal.c` (new)
     - `src/sparse_ldlt_csc_internal.h`
     - `Makefile`
     - `CMakeLists.txt`
4. Run the required validation gate:
   - `make format && make lint && make test && make quality-review-full`
5. Run the focused Day 5 follow-ons:
   - `./build/test_ldlt_csc`
   - `./build/test_ldlt`
   - `./build/test_integration`
   - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/example_analysis`
6. Capture post-split size/state checks:
   - `wc -l src/sparse_ldlt_csc.c src/sparse_ldlt_csc_supernodal.c src/sparse_ldlt_csc_internal.h`
   - `git status --short --branch`

### Day 5 Findings

#### 1. The first LDLT CSC decomposition batch landed cleanly with the exact Day 4 boundary

The moved Batch 1 cluster now lives in:

- `src/sparse_ldlt_csc_supernodal.c`

Moved function set:

- `ldlt_csc_bsearch_row_map(...)`
- `ldlt_csc_supernode_extract(...)`
- `ldlt_csc_supernode_writeback(...)`
- `ldlt_csc_supernode_eliminate_diag(...)`
- `ldlt_csc_supernode_eliminate_panel(...)`

Retained in `src/sparse_ldlt_csc.c`:

- lifecycle/conversion ownership
- wrapper compatibility path
- scalar/native Bunch-Kaufman kernel
- top-level `ldlt_csc_eliminate_supernodal(...)`
- solve path

Interpretation:

- the split is real ownership reduction, not just comment motion
- `src/sparse_ldlt_csc.c` now reads more clearly as the retained CSC
  orchestration/native home
- the supernodal cluster is now an owned backend slice with one narrow private
  dependency surface

#### 2. Phase 2 kept the existing private contract instead of mixing in a taxonomy redesign

The batch reused the existing:

- `src/sparse_ldlt_csc_internal.h`

Only bounded header change:

- top-level usage wording now reflects both:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_csc_supernodal.c`

Interpretation:

- Sprint 56 still separated source ownership first
- no new private-header hierarchy was opened in the same batch

#### 3. The ownership reduction is measurable

Post-split line counts:

- `src/sparse_ldlt_csc.c` = `2289`
- `src/sparse_ldlt_csc_supernodal.c` = `392`
- `src/sparse_ldlt_csc_internal.h` = `878`

Compared with the Day 1 baseline:

- `src/sparse_ldlt_csc.c`: `2723 -> 2289`

Interpretation:

- the main LDLT CSC file dropped by `434` lines in the first bounded batch
- the new source file size is large enough to be a real owned slice but still
  narrow enough to stay reviewable

#### 4. The full validation and reviewed parity baseline stayed intact

Required gate:

- `make format` → passed
- `make lint` → passed
- `make test` → passed
- `make quality-review-full` → passed

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 296.94 sec`

Interpretation:

- the source split did not disturb the reviewed baseline
- the new source file is fully represented in both Makefile and CMake paths

#### 5. Focused LDLT CSC follow-ons stayed behavior-stable

Focused reruns:

- `./build/test_ldlt_csc` → `96 / 96`
- `./build/test_ldlt` → `84 / 84`
- `./build/test_integration` → `37 / 37`
- `./build/example_analysis`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative direct results:

- `example_analysis` residual stayed `4.44e-16`
- `bench_refactor_csc nos4`:
  - `speedup_refactor = 1.52x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

Interpretation:

- the split preserved both the CSC-specific regression surface and the repeated-run
  direct proof surface
- no new reconciliation queue surfaced from the extraction itself

## Day 5 Close

Day 5 lands the first bounded LDLT CSC decomposition batch:

- extracted the supernodal helper cluster into
  `src/sparse_ldlt_csc_supernodal.c`
- kept lifecycle/conversion, wrapper compatibility, and the scalar/native
  kernel in `src/sparse_ldlt_csc.c`
- preserved the existing private-header contract in
  `src/sparse_ldlt_csc_internal.h`
- updated both build systems to compile the new owned source file
- preserved full local and reviewed validation parity

That gives Sprint 56 a validated Phase 2 decomposition landing rather than
only a design boundary.

## Day 6

**Objective:** Reduce `src/sparse_chol_csc.c` to concrete ownership seams by
auditing the live Cholesky CSC implementation bands, comparing them against
the landed LDLT CSC split, and fixing a ranked extraction order plus a
defensible first Cholesky CSC boundary before any code movement begins.

### Commands Run

1. Re-read the Sprint 56 Day 6 plan item and the current sprint notes:
   - `sed -n '200,236p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
   - `sed -n '1,980p' docs/planning/EPIC_5/SPRINT_56/WORKING_NOTES.md`
2. Re-read the landed LDLT audit/design state for comparison:
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_56/artifacts/day3-ldlt-csc-residual-ownership-audit.md`
   - `sed -n '1,220p' docs/planning/EPIC_5/SPRINT_56/artifacts/day4-ldlt-csc-decomposition-design.md`
3. Audit the live Cholesky CSC production file and internal contract:
   - `rg -n "^(static |sparse_err_t |void |double |idx_t )" src/sparse_chol_csc.c`
   - `sed -n '1,520p' src/sparse_chol_csc_internal.h`
   - `sed -n '1210,2210p' src/sparse_chol_csc.c`
4. Reconfirm the strongest direct Cholesky proof surfaces:
   - `rg -n "chol_csc_detect_supernodes|chol_dense_factor|chol_dense_solve_lower|chol_csc_eliminate_supernodal|chol_csc_supernode_extract|chol_csc_supernode_eliminate_diag|chol_csc_supernode_eliminate_panel|chol_csc_supernode_writeback|chol_csc_writeback_to_sparse" tests/test_chol_csc.c benchmarks/bench_refactor_csc.c examples/example_analysis.c`
5. Measure the coarse Cholesky ownership bands for ranking support:
   - ```sh
     python3 - <<'PY'
     from pathlib import Path
     text = Path('src/sparse_chol_csc.c').read_text().splitlines()
     ranges = {
         'lifecycle_conversion': (136, 713),
         'workspace_scalar_core': (714, 1223),
         'chol_supernodal_backend_candidate': (1224, 2093),
         'writeback_to_sparse': (2094, 2194),
         'shared_ldlt_dense_primitives_within_supernode_band': (1406, 1672),
     }
     for name, (a, b) in ranges.items():
         print(f"{name}: {b - a + 1}")
     PY
     ```

### Day 6 Findings

#### 1. The Cholesky CSC large-file problem is now reduced to named ownership bands

The current `src/sparse_chol_csc.c` function map separates into six real
ownership bands:

1. lifecycle / storage / structural conversion
2. scalar workspace and native elimination/solve core
3. wrapper / dispatch-specific glue
4. Cholesky-owned supernodal backend
5. compatibility-facing CSC writeback seam
6. shared dense indefinite primitive seam

Interpretation:

- the file is large, but it is no longer ambiguous
- the remaining design question is which ownership band should move first
- Cholesky also has one important family-specific wrinkle:
  - `ldlt_dense_factor(...)` still lives here even though it is not purely
    Cholesky-owned

#### 2. The strongest first Cholesky extraction target is the full Cholesky-owned supernodal backend

The strongest first extraction target is the Cholesky-owned supernodal backend
as one coherent file-owned slice.

Recommended moved set:

- `columns_in_same_supernode(...)`
- `chol_csc_detect_supernodes(...)`
- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`
- `chol_csc_eliminate_supernodal(...)`
- `chol_csc_bsearch_row_map(...)`
- `chol_csc_supernode_extract(...)`
- `chol_csc_supernode_eliminate_diag(...)`
- `chol_csc_supernode_eliminate_panel(...)`
- `chol_csc_supernode_writeback(...)`

Why it outranks the scalar/native kernel:

- it is already contiguous and internally cohesive
- it carries a clean SPD-only vocabulary
- it is the clearest line-count relief in the file
- `tests/test_chol_csc.c` already treats it like a real backend boundary
- `bench_refactor_csc.c` already names the same CSC completion seam directly

Measured ownership value:

- approximate supernodal backend candidate band:
  - `1224..2093`
  - about `870` lines

Interpretation:

- unlike the LDLT batch, Cholesky can justify moving the top-level batched
  driver together with its helper cluster
- that would create a real backend-owned file rather than a narrower helper
  spillover module

#### 3. The scalar workspace/native elimination core is still the strongest second seam

The scalar workspace and native elimination/solve core remains the strongest
second target:

- it still carries a large ownership mass
- it is still the highest-risk numerical band
- it becomes easier to reason about once the supernodal backend no longer
  shares the same file

Approximate ownership mass:

- scalar/workspace/elimination/solve band:
  - `714..1223`
  - about `510` lines

Interpretation:

- Sprint 56 still should not start by splitting the scalar kernel only because
  it is large
- the cleaner first maintainability win is still the supernodal backend

#### 4. The LDLT comparison is still useful, but Cholesky should not copy it mechanically

The landed LDLT Batch 1 remains the right comparison point, but not the exact
template.

Shared pattern with LDLT:

- supernodal backend work is still the best first seam
- keep the existing private header
- avoid mixing source extraction with private-header taxonomy redesign

Intentional differences from LDLT:

1. Cholesky's first seam should be wider.
   - LDLT kept the top-level supernodal driver in the main file.
   - Cholesky's `chol_csc_eliminate_supernodal(...)` is more cleanly part of
     the same SPD backend cluster as its detect/extract/diag/panel/writeback
     helpers.
2. Cholesky should move its dense Cholesky primitives with the backend.
   - `chol_dense_factor(...)` and `chol_dense_solve_lower(...)` are naturally
     owned by the same slice.
3. Cholesky should leave the shared dense LDLT primitive behind.
   - `ldlt_dense_factor(...)` is used by LDLT CSC and should not be blurred
     into a Cholesky-owned module.

Interpretation:

- the right Day 6 outcome is not "repeat the LDLT split"
- it is "reuse the LDLT decision logic, then pick the seam that matches the
  Cholesky file's real ownership"

#### 5. The proof surfaces already reinforce the Cholesky backend boundary

The current proof surfaces already imply a real Cholesky backend seam:

- `tests/test_chol_csc.c` directly names:
  - supernode detection
  - dense Cholesky helpers
  - supernodal elimination
  - extract / diag / panel / writeback helpers
  - writeback-to-sparse
- `benchmarks/bench_refactor_csc.c` directly exercises:
  - `chol_csc_from_sparse_with_analysis(...)`
  - `chol_csc_eliminate_supernodal(...)`
- `examples/example_analysis.c` remains the high-signal caller-facing repeated
  direct workflow proof surface

Interpretation:

- the best extraction seam is the one the proof surfaces already imply exists
- utility-first slicing across those proof boundaries would be harder to
  validate and harder to explain

#### 6. The ranked extraction order is now explicit

Ranked `src/sparse_chol_csc.c` target order from strongest to weakest:

1. Cholesky-owned supernodal backend cluster
2. scalar workspace and native elimination/solve core
3. lifecycle / conversion / validation cluster
4. CSC writeback-to-sparse seam
5. wrapper / dispatch glue
6. shared dense indefinite primitive cleanup

That gives Sprint 56 a concrete first-batch recommendation:

- proposed file:
  - `src/sparse_chol_csc_supernodal.c`
- keep in the main file initially:
  - lifecycle/conversion entry points
  - scalar workspace/native elimination/solve core
  - wrapper/dispatch glue
  - `chol_csc_writeback_to_sparse(...)`
  - shared dense LDLT primitive helpers

## Day 6 Close

Sprint 56 now has a concrete Cholesky CSC decomposition map:

- named ownership bands
- a ranked extraction order
- an explicit first target centered on the full Cholesky-owned supernodal
  backend
- a clear family-difference boundary versus the landed LDLT split

That is enough to move to the next Sprint 56 Cholesky design/implementation
step without leaving the first Cholesky batch boundary ambiguous.

## Day 7

**Objective:** Freeze the first Cholesky CSC extraction boundary before
editing permanent implementation files by turning the Day 6
supernodal-first ranking into an exact file split, declaration strategy, and
preserved-behavior checklist.

### Commands Run

1. Re-read the Sprint 56 Day 7 plan item and the Day 6 closing state:
   - `sed -n '230,304p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_56/artifacts/day6-cholesky-csc-residual-ownership-audit.md`
2. Re-read the live private contract and build-list state before fixing the
   Cholesky boundary:
   - `sed -n '1,260p' src/sparse_chol_csc_internal.h`
   - `sed -n '1,140p' Makefile`
   - `sed -n '80,120p' CMakeLists.txt`
3. Re-read the earlier Phase 2 LDLT design batch for shape/reference:
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_56/artifacts/day4-ldlt-csc-decomposition-design.md`

### Day 7 Findings

#### 1. Sprint 56 Batch 2 should extract the full Cholesky-owned supernodal backend into its own source file

The Day 6 ranking still holds after re-reading the concrete helper block and
the private contract:

- new file:
  - `src/sparse_chol_csc_supernodal.c`

The first moved function set should be:

- `columns_in_same_supernode(...)`
- `chol_csc_detect_supernodes(...)`
- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`
- `chol_csc_eliminate_supernodal(...)`
- `chol_csc_bsearch_row_map(...)`
- `chol_csc_supernode_extract(...)`
- `chol_csc_supernode_eliminate_diag(...)`
- `chol_csc_supernode_eliminate_panel(...)`
- `chol_csc_supernode_writeback(...)`

Interpretation:

- this is a real backend-owned seam, not a cosmetic file move
- Cholesky can justify moving the top-level batched driver together with the
  helper cluster because the ownership is cleaner than the earlier LDLT case

#### 2. The retained main file should keep lifecycle/conversion, scalar/native core, wrapper glue, CSC writeback, and shared dense LDLT helpers

### Keep in `src/sparse_chol_csc.c`

- lifecycle / storage / structural conversion
- sparse-to-CSC and analysis-aware conversion entry points
- validation
- scalar workspace and native elimination/solve core
- wrapper / dispatch-specific glue
- `chol_csc_writeback_to_sparse(...)`
- shared dense indefinite primitive helpers:
  - `ldlt_dense_sym_swap(...)`
  - `ldlt_dense_factor(...)`

### Move to `src/sparse_chol_csc_supernodal.c`

- supernode detection
- dense Cholesky factor/solve primitives
- supernodal row-map lookup helper
- supernode extract logic
- supernode diagonal-block eliminate helper
- supernode panel eliminate helper
- supernode CSC writeback helper
- top-level supernodal elimination driver

Interpretation:

- Batch 2 should create one clean SPD backend-owned file
- the retained main file should stay the home of the compatibility-facing
  conversion/scalar/control path

#### 3. Sprint 56 should keep using the existing internal header rather than mix in a private-header taxonomy redesign

The first Cholesky batch should keep declarations in the existing:

- `src/sparse_chol_csc_internal.h`

Reason:

- Batch 2 already changes one major ownership axis:
  - source-file extraction
- opening a new private-header taxonomy in the same batch would combine:
  - source extraction
  - private-header redesign
- the current internal header already contains the authoritative private
  contract for:
  - `CholCsc`
  - `CholCscWorkspace`
  - scalar/native helpers
  - Cholesky supernodal helpers
  - shared dense helper declarations

Interpretation:

- Sprint 56 should separate source ownership first
- private-header taxonomy cleanup remains a later maintainability choice, not
  a Day 8 dependency

#### 4. The first batch must preserve scalar/supernodal parity, writeback semantics, dispatch behavior, and CSC proof parity exactly

### Public and compatibility invariants

- no public header/API changes
- no user-visible direct-solver lifecycle behavior change
- no change to the public analysis/factors path that ultimately reaches the
  Cholesky CSC completion seam

### Scalar/supernodal parity invariants

- scalar versus supernodal result parity unchanged
- supernode-detection semantics unchanged
- `min_size` threshold behavior unchanged
- dense Cholesky diagonal/panel behavior unchanged

### Dispatch/writeback invariants

- one-shot and shared analysis-aware CSC routing unchanged
- `chol_csc_factor(...)` and `chol_csc_factor_solve(...)` behavior unchanged
- `chol_csc_writeback_to_sparse(...)` semantics unchanged
- drop-threshold and diagonal-preservation behavior unchanged

### Proof-surface invariants

- `tests/test_chol_csc.c` remains the primary direct proof surface
- `tests/test_cholesky.c` remains unchanged in meaning
- `tests/test_integration.c` remains unchanged in meaning
- `benchmarks/bench_refactor_csc.c` keeps its current repeated-run CSC proof
- `examples/example_analysis.c` keeps its current caller-facing repeated direct
  workflow proof

Interpretation:

- Day 8 should preserve the current Cholesky CSC proof story exactly
- the extraction is successful only if the source ownership changes while the
  behavior contracts stay still

#### 5. The bounded non-goal fence is now explicit before the first Cholesky code move

The first Cholesky batch should not:

- redesign CSC dispatch
- change public APIs or header shape
- create a new private-header taxonomy
- move the shared dense LDLT primitive into the Cholesky-owned file
- widen into broader Cholesky/LDLT CSC code reconciliation
- reopen benchmark or example design beyond parity checks

Interpretation:

- the Sprint 56 Cholesky implementation batch is now bounded clearly enough
  that the Day 8 touch set should stay narrow

#### 6. The expected Day 8 touch set and landing checklist are now fixed

Primary expected touched set:

- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_supernodal.c`
- `src/sparse_chol_csc_internal.h`
- `Makefile`
- `CMakeLists.txt`

Secondary touch only if truly needed:

- `tests/test_chol_csc.c`

Avoid by default:

- `include/sparse_cholesky.h`
- `tests/test_cholesky.c`
- `tests/test_integration.c`
- `benchmarks/bench_refactor_csc.c`
- `src/sparse_ldlt_csc.c`

## Day 7 Close

Day 7 fixes the first Cholesky CSC extraction boundary explicitly:

- move the full Cholesky-owned supernodal backend into
  `src/sparse_chol_csc_supernodal.c`
- keep lifecycle/conversion, scalar/native core, wrapper glue, CSC writeback,
  and shared dense LDLT helpers in `src/sparse_chol_csc.c`
- reuse the existing internal header for Phase 2
- preserve the full scalar/supernodal, dispatch, threshold, and proof contract
  exactly

That gives Sprint 56 a concrete, bounded, maintainability-first Day 8 landing
plan.

## Day 8

**Objective:** Land the first bounded Cholesky CSC source split from the Day 7
design by extracting the full Cholesky-owned supernodal backend into its own
owned file while preserving CSC proof parity and the full reviewed validation
baseline.

### Commands Run

1. Re-read the Sprint 56 Day 8 plan item and the Day 7 design boundary:
   - `sed -n '305,380p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_5/SPRINT_56/artifacts/day7-cholesky-csc-decomposition-design.md`
2. Re-read the live Cholesky CSC implementation, internal contract, and build
   surfaces before editing:
   - `sed -n '1,2400p' src/sparse_chol_csc.c`
   - `sed -n '1,260p' src/sparse_chol_csc_internal.h`
   - `sed -n '1,140p' Makefile`
   - `sed -n '80,120p' CMakeLists.txt`
3. Land the Cholesky CSC extraction batch:
   - edited `src/sparse_chol_csc.c`
   - added `src/sparse_chol_csc_supernodal.c`
   - edited `src/sparse_chol_csc_internal.h`
   - edited `Makefile`
   - edited `CMakeLists.txt`
4. Run the required validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
5. Run the high-signal Day 8 follow-ons:
   - `./build/test_chol_csc`
   - `./build/test_cholesky`
   - `./build/test_integration`
   - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/example_analysis`
6. Capture the post-split size deltas:
   - `wc -l src/sparse_chol_csc.c src/sparse_chol_csc_supernodal.c src/sparse_chol_csc_internal.h`
   - `git diff --stat`

### Day 8 Findings

#### 1. Sprint 56 Batch 2 successfully extracted the full Cholesky-owned supernodal backend into its own owned source file

The landed new file is:

- `src/sparse_chol_csc_supernodal.c`

Moved function set:

- `columns_in_same_supernode(...)`
- `chol_csc_detect_supernodes(...)`
- `chol_dense_factor(...)`
- `chol_dense_solve_lower(...)`
- `chol_csc_eliminate_supernodal(...)`
- `chol_csc_bsearch_row_map(...)`
- `chol_csc_supernode_extract(...)`
- `chol_csc_supernode_eliminate_diag(...)`
- `chol_csc_supernode_eliminate_panel(...)`
- `chol_csc_supernode_writeback(...)`

Interpretation:

- the extracted file is a real SPD/backend-owned slice
- the first Cholesky CSC split stayed aligned with the Day 7 ownership model
  rather than degrading into a cosmetic helper spill

#### 2. The retained `src/sparse_chol_csc.c` boundary stayed inside the Day 7 fence

Retained in the main file:

- lifecycle / conversion ownership
- validation
- scalar workspace and native elimination/solve core
- wrapper / dispatch glue
- `chol_csc_writeback_to_sparse(...)`
- shared dense LDLT helpers:
  - `ldlt_dense_sym_swap(...)`
  - `ldlt_dense_factor(...)`

Interpretation:

- the retained file is still the compatibility-facing control path
- Batch 2 did not blur the boundary by moving writeback, shared dense
  indefinite primitives, or wrapper-specific logic into the Cholesky-owned
  file

#### 3. The private-header strategy stayed bounded, but the retained helper surfaces needed one analyzer-facing cleanup pass

The batch kept the existing internal contract in:

- `src/sparse_chol_csc_internal.h`

Bounded header cleanup landed:

- top-level usage wording now names:
  - `src/sparse_chol_csc.c`
  - `src/sparse_chol_csc_supernodal.c`
- the Cholesky dense/supernodal status comments now match the live ownership
  and behavior contract

The retained main file also needed a bounded static-analysis cleanup in:

- `bsearch_row(...)`
- `chol_csc_scatter(...)`
- `chol_csc_gather(...)`

The cleanup tightened those paths to explicit bounded slice/count logic
without changing the behavioral contract.

Interpretation:

- Day 8 stayed decomposition-first
- the extra retained-file cleanup was a proof/maintainability tightening, not
  a hidden semantic redesign

#### 4. The ownership reduction is real and measurable

Post-split line counts:

- `src/sparse_chol_csc.c` = `1625`
- `src/sparse_chol_csc_supernodal.c` = `544`
- `src/sparse_chol_csc_internal.h` = `979`

Compared with the Sprint 56 Day 1 baseline:

- `src/sparse_chol_csc.c`: `2194 -> 1625`

Interpretation:

- the retained main file dropped by `569` lines
- the new file is large enough to represent a true backend-owned seam

#### 5. The full required validation baseline remained exact after the Cholesky split

Required gate:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 279.57 sec`

High-signal follow-ons also remained green:

- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_cholesky` -> `21 / 21`
- `./build/test_integration` -> `37 / 37`
- `./build/example_analysis` -> residual `4.44e-16`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `speedup_refactor = 1.53x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

Interpretation:

- the first Cholesky CSC split preserved both the direct CSC proof surface and
  the broader reviewed project baseline

## Day 8 Close

Sprint 56 now has a landed second decomposition batch:

- the Cholesky-owned supernodal CSC backend has its own source file
- the retained main Cholesky CSC file is materially smaller and more focused
- the internal-header and build-surface changes stayed bounded
- the analyzer-facing retained-helper tightening stayed local and
  behavior-preserving
- the full local and reviewed validation baseline remained exact

That gives Sprint 56 a real Cholesky CSC maintainability landing rather than
only a design artifact.

## Day 9

**Objective:** Reduce `src/sparse_svd.c` to a bounded Sprint 56
maintainability target and pick the specific Day 10 landing direction before
any SVD code movement begins.

### Commands Run

1. Re-read the Sprint 56 Day 9 plan item:
   - `sed -n '260,347p' docs/planning/EPIC_5/SPRINT_56/PLAN.md`
2. Inspect the live SVD hotspot, internal header, and proof surfaces:
   - `wc -l src/sparse_svd.c src/sparse_svd_internal.h tests/test_svd.c benchmarks/bench_svd.c examples/example_svd_lowrank.c`
   - `sed -n '1,260p' src/sparse_svd_internal.h`
   - `sed -n '1,260p' src/sparse_svd.c`
   - `sed -n '260,760p' src/sparse_svd.c`
   - `sed -n '760,1280p' src/sparse_svd.c`
   - `sed -n '1280,1760p' src/sparse_svd.c`
3. Build a function/seam map from the live file:
   - `rg -n "^(static )?[A-Za-z_][A-Za-z0-9_ *]+\\([^;]*\\)\\s*\\{$" src/sparse_svd.c`
   - `nl -ba src/sparse_svd.c | sed -n '1,1750p' | rg "sparse_svd_extract_uv|bidiag_svd_step|bidiag_svd_iterate|pad_orthonormal_basis|sparse_svd_compute|sparse_svd_partial|sparse_svd_rank|sparse_pinv|sparse_svd_lowrank\\(|sparse_svd_lowrank_sparse|sparse_cond|parse_svd_lowrank_outer|sparse_svd_lowrank_outer_product|hh_apply"`
   - `rg -n "^/\\* ═|^/\\* Sprint|^sparse_err_t sparse_svd_partial|^static void bidiag_svd_step|^sparse_err_t bidiag_svd_iterate|^sparse_err_t sparse_svd_compute|^sparse_err_t sparse_svd_lowrank_sparse|^double sparse_cond" src/sparse_svd.c`
4. Re-read the benchmark and test surfaces most coupled to the SVD seam choice:
   - `sed -n '1,220p' benchmarks/bench_svd.c`
   - `rg -n "partial|lowrank|pinv|cond|rank|compute_uv|economy" tests/test_svd.c | tail -n 120`

### Day 9 Findings

#### 1. `src/sparse_svd.c` now reduces cleanly to five ownership bands instead of one generic large-file target

The live file separates into:

1. low-rank sparse reconstruction toggle plus outer-product path
2. bidiagonal reflector extraction plus implicit QR core
3. full-SVD orchestration plus full-mode basis padding
4. partial-SVD Lanczos backend
5. application wrappers and reporting utilities

Interpretation:

- Sprint 56 no longer needs to treat SVD as a vague cleanup bucket
- the file is structured enough to support a bounded maintainability landing
  if the right seam is chosen

#### 2. The strongest remaining SVD maintainability target is the partial-SVD Lanczos backend

The partial-SVD band centered on:

- `sparse_svd_partial(...)`

already owns a distinct algorithm family:

- Lanczos subspace sizing
- `A^T` construction/reuse
- `P/Q/alpha/beta` Lanczos storage lifecycle
- bidiagonalization loop
- small bidiagonal solve
- singular-value sorting and vector recovery

Interpretation:

- this is the strongest first owned slice because it is both large and
  behaviorally cohesive
- it is a better first target than a mechanical file-local cleanup or a
  deeper split of the shared bidiagonal QR core

#### 3. The bidiagonal QR helper cluster is real, but it is the wrong first extraction target

The cluster around:

- `hh_apply(...)`
- `sparse_svd_extract_uv(...)`
- `bidiag_svd_step(...)`
- `bidiag_svd_iterate(...)`

is cohesive algorithmically, but it also remains tightly central to the main
full-SVD/public orchestration path and the current internal test surface.

Interpretation:

- moving the QR/bidiagonal core first would force the broadest private-contract
  expansion
- Sprint 56 should keep the full-SVD/public core stable in the main file and
  extract the cleaner partial-SVD backend first

#### 4. The proof surfaces already support a bounded partial-SVD extraction better than any other SVD seam

The strongest explicit proof surfaces already cluster around partial SVD:

- `tests/test_svd.c`
  - partial sigma-only coverage
  - partial vector recovery coverage
  - timing/parity coverage
- `benchmarks/bench_svd.c`
  - explicit partial-vs-full timing/reporting

Interpretation:

- the partial-SVD backend has a naturally bounded proof envelope
- that makes it the safest maintainability landing relative to the size of the
  ownership gain

#### 5. Day 10 should emphasize helper extraction, not broad file-local cleanup

Chosen Day 10 direction:

- helper extraction

Specific landing direction:

- move the partial-SVD Lanczos backend into:
  - `src/sparse_svd_partial.c`

Keep in `src/sparse_svd.c`:

- full-SVD/public orchestration
- reflector extraction and bidiagonal QR machinery
- full-mode basis padding
- application wrappers:
  - `sparse_svd_rank(...)`
  - `sparse_pinv(...)`
  - `sparse_svd_lowrank(...)`
  - `sparse_svd_lowrank_sparse(...)`
  - `sparse_cond(...)`

Interpretation:

- Day 10 now has a bounded extraction target rather than a generic “make
  sparse_svd.c nicer” task
- the public full-SVD front door stays stable while Sprint 56 reduces one real
  backend seam

## Day 9 Close

Sprint 56 now has an explicit SVD maintainability direction:

- `src/sparse_svd.c` reduces to named ownership bands
- the strongest first target is the partial-SVD Lanczos backend
- Day 10 should land helper extraction, not a broad in-place cleanup pass
- the full-SVD/public orchestration path should stay in the main file this
  sprint

That is enough to start the Day 10 SVD batch from a concrete maintainability
plan instead of a vague residual cleanup scope.
