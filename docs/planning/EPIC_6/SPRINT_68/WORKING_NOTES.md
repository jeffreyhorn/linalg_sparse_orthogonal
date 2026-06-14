# Sprint 68 Working Notes

## Day 1 - Scope Audit & Giant-Test Baseline Setup

### Goal

Freeze the Sprint 68 starting point before implementation work begins by
reconfirming the inherited Sprint 67 contract, the preserved reviewed
baseline, the strongest live giant-test and assurance hotspots, and the most
important docs/proof/support surfaces the sprint will touch next.

### Actions

1. Re-read the Sprint 68 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 67 retrospective, and
   the Sprint 67 Day 14 closeout artifact.
2. Re-read the landed Sprint 68 plan and fixed the bounded workstreams that
   the sprint should actually carry:
   - giant-test residual audit
   - giant-test refactor batch
   - differential/oracle coverage
   - property/fuzz expansion
   - platform-test follow-through
   - validation and closeout
3. Reconfirmed the strongest reviewed baseline surfaces:
   - `make quality-review-full`
   - `make -n quality-review-full`
4. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
5. Measured the strongest likely Sprint 68 touch surfaces directly from the
   live tree across:
   - maintained truth surfaces
   - giant tests
   - supporting examples and benchmark/reporting surfaces

### Findings

#### 1. Sprint 68 starts from the Sprint 67 maintainability close, not from renewed implementation-boundary work

Sprint 67 already closed the strongest remaining large-source ownership seams
that justified a dedicated maintainability phase. That means Sprint 68 is not
reopening:

- graph/reorder ownership extraction as a primary implementation target
- shared ND compatibility/default-policy convergence
- the large-`n` Cholesky analysis-to-CSC handoff lane
- broad source-ownership decomposition work disguised as test refactoring

Interpretation:

- Sprint 68 is the first post-Sprint-67 Epic 6 sprint centered primarily on
  giant-test maintenance cost and second-layer assurance again
- implementation files are now support surfaces only where test or oracle work
  proves they truly need to move

#### 2. The strongest local reviewed baseline remains the authoritative Sprint 68 starting point

The maintained Day 1 truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 68 inherits the exact same reviewed baseline story as the Sprint 67
  close
- giant-test or assurance work does not get a weaker truth surface just
  because the sprint topic is test architecture rather than new implementation

#### 3. The highest-value Sprint 68 problem is concentrated in giant tests and second-layer assurance lanes, not in generic test churn

The live repo shows the strongest pressure in:

- giant CSC and integration proof surfaces
- giant graph/reorder and iterative/eigensolver proof surfaces
- hard numerical lanes that still benefit from stronger differential/oracle
  assurance
- bounded property/fuzz surfaces where added invariants could materially pay
  off

The project-plan scope therefore reduces cleanly to:

1. giant-test residual audit
2. giant-test refactor batch
3. differential/oracle coverage
4. property/fuzz expansion
5. platform-test follow-through
6. validation and closeout

Interpretation:

- Sprint 68 should not pretend every large test is equally urgent
- the highest-value work is concentrated in the largest maintenance surfaces
  and the hardest numerical proof lanes

#### 4. The strongest live Sprint 68 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- maintained truth surfaces:
  - `README.md` = `1025`
  - `docs/maintainer_guide.md` = `561`
  - `benchmarks/README.md` = `351`
- strongest giant-test seams:
  - `tests/test_chol_csc.c` = `4751`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_graph.c` = `2900`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_svd.c` = `2766`
  - `tests/test_integration.c` = `2371`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_eigs.c` = `1522`
- giant-test support or smaller assurance surfaces:
  - `tests/test_sparse_lu.c` = `908`
  - `tests/test_fuzz.c` = `497`
  - `tests/test_suitesparse.c` = `288`
  - `tests/test_framework_optin.c` = `85`
- strongest proof/adoption/reporting support surfaces:
  - `examples/example_analysis.c` = `210`
  - `examples/example_basic_solve.c` = `110`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`

Interpretation:

- the strongest remaining Epic 6 test-maintenance pressure is concentrated in
  a smaller set of giant permanent proof surfaces
- Sprint 68 should start by reranking those seams, not by inventing new test
  workstreams first

#### 5. The Day 1 non-goal fence is now explicit before deeper audit begins

Sprint 68 Day 1 confirms the following non-goals:

- no fake assurance wins that only add brittle fixed outputs
- no broad solver-feature work disguised as test refactoring
- no reopening Sprint 67 implementation-boundary work unless a touched test
  seam proves it is necessary
- no weakening of the reviewed truthfulness contract
- no broad style-only cleanup wave disconnected from giant-test ownership pain
- no inflated platform-confidence claims beyond reviewed evidence

### Day 1 Close

Sprint 68 now starts from one explicit giant-test and assurance baseline:

- the Sprint 67 maintainability close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor is re-established locally at `53`
- the broad Epic 6 giant-test claim has already narrowed to residual audit,
  test refactor, oracle coverage, property/fuzz expansion, platform-test
  follow-through, and closeout
- the next step is to rank those live giant-test seams precisely before
  writing the Day 2 validation and Day 3 hotspot follow-through

## Day 2 - Validation Baseline & Giant-Test/Proof Rerun Recheck

### Goal

Reconfirm the reviewed baseline and the targeted giant-test and assurance
rerun set that Sprint 68 refactor work must preserve before any
implementation work lands.

### Actions

1. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
2. Re-read the reviewed baseline wrapper surface:
   - `make -n quality-review-full`
3. Reconfirmed the authoritative validation split for:
   - bounded `*.c` / `*.h` days
   - substantial giant-test or oracle/property assurance days
   - docs-only days
4. Rechecked build-tree availability of the most relevant Sprint 68 proof and
   regression surfaces:
   - giant direct-family tests
   - giant graph/reorder and iterative/eigensolver tests
   - property/fuzz and opt-in framework proof surfaces
   - representative examples
   - maintained benchmark/reporting surfaces
5. Reconfirmed the strongest likely Sprint 68 touched-surface classes from the
   live branch state after the Day 1 baseline.

### Findings

#### 1. The strongest reviewed baseline is unchanged at Sprint 68 start

The strongest local reviewed baseline is still:

- `make quality-review-full`

The maintained reviewed CMake parity anchor is still:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 68 inherits the same reviewed-baseline authority split as Sprint 67
- giant-test and assurance work is not allowed to drift onto a weaker local
  truth surface

#### 2. The authoritative validation split is now explicit before code work begins

The Day 2 validation contract is now fixed as:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial test-architecture, oracle, or
  platform-confidence work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

Interpretation:

- Sprint 68 should treat giant-test refactors as real code moves, not “docs
  adjacent” work
- assurance expansion that materially changes proof surfaces should default to
  the stronger reviewed gate

#### 3. The high-signal Sprint 68 rerun set is now fixed around the real giant-test and assurance-risk surface

The targeted Sprint 68 rerun set present in `build/` is:

- cross-family/orchestration proof:
  - `./build/test_integration`
- giant direct-family proofs:
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_qr`
  - `./build/test_svd`
- giant graph/reorder and iterative/eigensolver proofs:
  - `./build/test_graph`
  - `./build/test_reorder_nd`
  - `./build/test_iterative`
  - `./build/test_eigs`
- assurance-support surfaces:
  - `./build/test_fuzz`
  - `./build/test_framework_optin`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

Interpretation:

- this is the smallest meaningful rerun set that still spans the strongest
  Sprint 68 maintenance and assurance lanes
- later days should justify any expansion beyond this set, not default to it

#### 4. The current giant-test/refactor lane is already narrower than the full test tree

Even though the repo has a broad reviewed suite, Sprint 68 Day 2 confirms the
highest-value likely touched lane is concentrated in:

- giant CSC and direct-family proofs
- giant graph/reorder proofs
- giant iterative/eigensolver proofs
- property/fuzz support where added invariants may materially pay off
- representative examples and maintained benchmark/reporting surfaces only
  where proof ownership truly moves

Interpretation:

- Sprint 68 should not turn into a repo-wide test reorganization
- the targeted rerun set is already narrow enough to keep later audit and
  landing decisions honest

### Day 2 Close

Sprint 68 now has one explicit validation contract before implementation
begins:

- strongest local reviewed baseline is still `make quality-review-full`
- reviewed CMake parity remains explicit at `53`
- bounded `*.c` / `*.h` days must run `make format`, `make lint`, and
  `make test`
- substantial giant-test or assurance work should default to
  `make quality-review-full`
- the targeted Sprint 68 rerun set is now fixed around the actual giant-test,
  assurance, example, and maintained benchmark surfaces present in `build/`

## Day 3 - Giant-Test Residual Audit

### Goal

Reduce Sprint 68’s broad giant-test and assurance claim to a ranked live seam
map so the sprint can land bounded high-value refactors instead of generic
test cleanup.

### Actions

1. Measured the strongest remaining giant tests by:
   - total line count
   - test count / `RUN_TEST(...)` fan-out
   - helper density
   - sprint/day chronology density
2. Re-read the visible section structure and chronology markers across the
   highest-value giant-test files:
   - `tests/test_chol_csc.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_graph.c`
   - `tests/test_iterative.c`
   - `tests/test_ldlt.c`
   - `tests/test_svd.c`
   - `tests/test_integration.c`
   - `tests/test_reorder_nd.c`
   - `tests/test_eigs.c`
3. Compared “large because broad but coherent” against “large because too many
   unrelated scenarios and chronology layers still coexist in one permanent
   file.”
4. Re-ranked the strongest Sprint 68 refactor and assurance candidates from
   that live state.
5. Recorded the explicit hotspot map and first-order narrowing for Day 4.

### Findings

#### 1. The broad Sprint 68 claim is now reduced to a ranked live seam map

The current giant-test field separates into three real classes:

- strongest first-lane giant-test refactor candidates:
  - `tests/test_chol_csc.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_ldlt_csc.c`
- strongest second-lane oracle/assurance candidates:
  - `tests/test_integration.c`
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `tests/test_svd.c`
- large but more internally coherent or lower-priority follow-through surfaces:
  - `tests/test_qr.c`
  - `tests/test_graph.c`
  - `tests/test_ldlt.c`

Interpretation:

- Sprint 68 should not chase every large test equally
- the highest-value work is concentrated in a smaller set of files where size,
  chronology, helper sprawl, and mixed proof ownership still collide

#### 2. `tests/test_chol_csc.c` is the strongest first refactor target

Measured pressure:

- lines = `4751`
- tests = `144`
- `RUN_TEST(...)` fan-out = `145`
- helper-ish support functions = `14`
- chronology density:
  - `Sprint` mentions = `35`
  - `Day` mentions = `60`

Why it is the strongest first target:

- it combines:
  - family-local CSC factorization behavior
  - dense primitive coverage
  - supernodal extract/writeback plumbing
  - dispatch and backend-contract proof
  - large corpus and regression lanes
- that means it is large not just because the feature surface is broad, but
  because multiple ownership layers still live in one permanent test file
- it also has the best near-term assurance leverage because Cholesky CSC still
  anchors several of Epic 6’s hardest retained direct-path claims

Interpretation:

- if Sprint 68 only lands one giant-test refactor batch, this is currently the
  best first target

#### 3. `tests/test_reorder_nd.c` is the strongest second refactor target, but for a different reason

Measured pressure:

- lines = `2262`
- tests = `34`
- chronology density:
  - `Sprint` mentions = `81`
  - `Day` mentions = `99`

Why it still ranks very high:

- it has less raw line count than `test_chol_csc.c`, but the strongest
  residual pain is chronology and compatibility-story layering
- the file still mixes:
  - public ND behavior
  - compatibility/env-policy proof
  - post-Sprint-27 and Sprint-28 follow-through contracts
  - enum dispatch and supernodal-postorder validation
- that makes it a strong refactor candidate, but slightly worse than
  `test_chol_csc.c` as a first landing because its maintenance pressure is
  more about chronology and proof layering than about one obvious helper split

Interpretation:

- `test_reorder_nd.c` is likely the best second target or the main competitor
  if Day 4 prefers chronology reduction over CSC-family-local helper
  extraction

#### 4. `tests/test_ldlt_csc.c` is large and real, but cleaner than the first two targets

Measured pressure:

- lines = `3680`
- tests = `96`
- helper-ish support = `23`
- chronology density:
  - `Sprint` mentions = `30`
  - `Day` mentions = `51`

Why it is not first:

- it is large and helper-heavy, but its structure reads more consistently as a
  family-local owner than `test_chol_csc.c`
- compared with `test_reorder_nd.c`, it carries less cross-family compatibility
  layering
- compared with `test_chol_csc.c`, it carries fewer distinct proof roles in
  one permanent file

Interpretation:

- it remains a strong later Sprint 68 seam, but not the best first landing

#### 5. Some large files are big, but not the best first refactor targets

Current lower-priority or later lanes:

- `tests/test_qr.c`
  - large (`3197` lines) but segmented into more coherent numerical phases
- `tests/test_graph.c`
  - large (`2900` lines) and chronology-heavy, but Sprint 67 already reduced
    adjacent ownership pressure in the implementation layer, so the immediate
    Sprint 68 payoff is weaker than in `test_chol_csc.c`
- `tests/test_svd.c`
  - very broad and chronology-heavy, but much of its size comes from coherent
    phase-by-phase algorithm coverage rather than one obvious first split seam
- `tests/test_integration.c`
  - high-value assurance owner, but it is better treated first as an
    oracle/parity surface than as the first giant-test refactor target

Interpretation:

- size alone is not enough to justify the first Sprint 68 landing
- the first target should be chosen where maintenance pain and bounded split
  opportunity coincide

### Day 3 Close

Sprint 68’s broad giant-test claim is now reduced to one ranked live seam map:

- strongest first target:
  - `tests/test_chol_csc.c`
- strongest second target:
  - `tests/test_reorder_nd.c`
- strongest later giant direct-family target:
  - `tests/test_ldlt_csc.c`
- strongest oracle/assurance owner:
  - `tests/test_integration.c`
- strongest later assurance/follow-through owners:
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `tests/test_svd.c`

The next step is to turn that ranking into one explicit first-landing boundary
instead of a generic shortlist.

## Day 4 - Hotspot Follow-Through & First-Landing Boundary

### Goal

Turn the Day 3 giant-test ranking into one exact first implementation fence so
Sprint 68 starts from a bounded Cholesky CSC test refactor instead of a
generic multi-file cleanup target set.

### Actions

1. Re-read the Day 3 ranked giant-test audit and the Sprint 68 plan fence.
2. Re-read the strongest likely first-landing region in:
   - `tests/test_chol_csc.c`
3. Re-read the strongest likely second-lane chronology region in:
   - `tests/test_reorder_nd.c`
4. Rechecked the current family-local support surface already available under:
   - `tests/test_chol_csc_supernodal_helpers.h`
5. Fixed the exact first-landing boundary from the live repo state:
   - required refactor surface
   - support only if needed
   - explicit deferred/non-touch set

### Findings

#### 1. Sprint 68 now has one exact first landing boundary instead of a generic giant-test shortlist

The exact first landing is now fixed to:

- `tests/test_chol_csc.c`

This is the right first batch because that file still carries the strongest
combined burden of:

- raw size
- family-local CSC factorization proof
- dense primitive proof
- supernodal helper/plumbing proof
- backend-contract and dispatch proof
- large corpus and regression lanes

Interpretation:

- Sprint 68 should not start by touching every large permanent test file
- it should start where one file still owns the densest mix of proof roles and
  helper pressure

#### 2. `tests/test_reorder_nd.c` is now fixed as the strongest second batch, not a co-equal first landing

`tests/test_reorder_nd.c` remains a high-value Sprint 68 seam because it still
mixes:

- public ND behavior
- compatibility/env-policy proof
- supernodal-postorder validation
- Sprint-history-heavy chronology

But it stays out of the first landing because its strongest pressure is
chronology and proof layering, not the same kind of bounded helper and owner
split opportunity that now stands out in `tests/test_chol_csc.c`.

Interpretation:

- Day 4 closes the competition between the first two candidates
- Cholesky CSC refactor first; ND chronology follow-through second

#### 3. The current helper support surface is context, not a reason to widen the first batch

The first landing may use the already-existing family-local support surface:

- `tests/test_chol_csc_supernodal_helpers.h`

But that file is support only if the Day 5 design proves it necessary.

Why that matters:

- the first batch should stay family-local by default
- Sprint 68 should not invent new shared test abstractions before proving the
  bounded `test_chol_csc.c` split
- cross-family helper widening would blur whether the sprint is still reducing
  one giant owner or starting a broad test-framework redesign

#### 4. `tests/test_integration.c` stays in the oracle lane, not the first refactor lane

`tests/test_integration.c` remains the strongest shared assurance owner, but it
is explicitly outside the first landing fence.

Why it stays out:

- its value is public-path parity and oracle follow-through
- the first Sprint 68 move is not to refactor the oracle owner
- pulling it into the first landing would blur the difference between
  family-local test-architecture cleanup and second-layer assurance expansion

Interpretation:

- refactor-first and oracle-first remain separate lanes on purpose

#### 5. The non-touch set for the first landing is now explicit

The following stay outside the Day 5-7 first batch unless the design proves a
truly necessary support edit:

- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`
- `tests/test_graph.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `tests/test_svd.c`
- `tests/test_integration.c`
- implementation `src/` files
- benchmark/docs truth surfaces

Interpretation:

- Sprint 68 now has a real first-batch fence instead of a soft hotspot cloud
- success means reducing `tests/test_chol_csc.c` ownership pressure without
  widening into unrelated assurance or implementation work

### Day 4 Close

Sprint 68 Day 4 fixes the implementation order as:

1. exact first landing:
   - `tests/test_chol_csc.c`
2. support only if needed:
   - `tests/test_chol_csc_supernodal_helpers.h`
3. strongest second target:
   - `tests/test_reorder_nd.c`
4. explicit oracle/assurance lane:
   - `tests/test_integration.c`
5. later/deferred giant-test follow-through:
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_graph.c`
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`
   - `tests/test_svd.c`

That gives Day 5 one exact job:

- define the bounded ownership and helper-extraction contract inside
  `tests/test_chol_csc.c`

## Day 5 - Giant-Test Refactor Design

### Goal

Turn the Day 4 first-landing fence into one explicit ownership and extraction
contract so Day 6 can reduce `tests/test_chol_csc.c` maintenance pressure
without widening into generic test-framework churn or shared assurance work.

### Actions

1. Re-read the Day 4 boundary artifact and Sprint 68 plan fence.
2. Re-read the live section structure and helper density in:
   - `tests/test_chol_csc.c`
3. Re-read the existing family-local helper seam in:
   - `tests/test_chol_csc_supernodal_helpers.h`
4. Mapped the current ownership split inside the first-landing file across:
   - baseline CSC-format and scalar-kernel proof
   - supernodal helper/plumbing proof
   - writeback and dispatch proof
   - main `RUN_TEST(...)` fan-out and chronology
5. Reduced that map to one bounded Day 6-7 implementation contract and one
   explicit non-widening fence.

### Findings

#### 1. Sprint 68 now has one exact refactor contract for the first giant-test landing

The first landing should not try to "finish" Cholesky CSC test cleanup across
every proof role in the file. It should instead make the permanent owner read
more clearly as:

- one canonical `test_chol_csc` binary and proof owner
- one family-local helper seam for supernodal/writeback/dispatch support code
- one main file that keeps the actual assertions and proof intent readable

Interpretation:

- Day 6 should optimize for lower local maintenance pressure inside the
  existing owner
- it should not behave like a broad test-suite architecture redesign

#### 2. `tests/test_chol_csc.c` should stay the canonical family-local proof owner, not split into multiple test binaries

The live file still owns multiple proof lanes, but that alone does not justify
splitting it into separate `test_*.c` binaries in the first batch.

Why the first landing should keep one binary:

- the current file already serves as the clear family-local owner for CSC
  Cholesky behavior
- several late sections depend on shared local builders, comparison helpers,
  and internal-family context
- widening into multiple binaries would immediately drag in build-list churn,
  test registration churn, and proof-ownership ambiguity

The Day 5 design implication is now explicit:

- keep `tests/test_chol_csc.c` as the canonical proof owner
- reduce local clutter by extracting bounded helper/support code, not by
  multiplying test binaries on the first pass

#### 3. The first extraction seam is the supernodal/writeback/dispatch support lane, not the scalar/core proof lane

The live file has two materially different categories of content:

- durable proof sections that should stay in `tests/test_chol_csc.c`:
  - CSC allocation / growth / conversion / validation proof
  - scalar elimination and solve proof
  - high-level family-local assertions and scenario bodies
- support-heavy local seams that are better extraction candidates:
  - supernode detection allocation helpers
  - factored CSC comparison helpers for scalar-vs-batched checks
  - large SPD fixture builders for dispatch and backend-path checks
  - repetitive round-trip scaffolding for writeback and supernodal parity

The strongest bounded candidate functions are now explicit:

- keep using the existing helper-header lane for family-local support like:
  - `detect_supernodes_alloc(...)`
  - `day8_count_supernodes(...)`
  - `day9_assert_batched_matches_scalar(...)`
  - `day11_build_spd(...)`
- likely move additional supernodal/writeback support there if Day 6 needs it:
  - `day8_chol_csc_match(...)`
  - `day7_chol_csc_get(...)`
  - `day10_factored_matches(...)`
  - `day10_roundtrip_check(...)`

Design consequence:

- keep scenario assertions and proof bodies in the main file
- move only family-local support scaffolding where that materially clarifies
  the supernodal/writeback/dispatch tail

#### 4. The first batch should tighten `RUN_TEST(...)` chronology locally, not redesign the whole runner surface

The giant `RUN_TEST(...)` tail is part of the maintenance burden, but it is
not the first thing to abstract away behind clever registration machinery.

The safer Day 6 contract is:

- keep one explicit `RUN_TEST(...)` owner in `tests/test_chol_csc.c`
- allow bounded regrouping or local ordering cleanup only where helper
  extraction would otherwise make the proof story harder to follow
- avoid introducing macro-driven or data-driven runner indirection just to make
  the file look shorter

Interpretation:

- Sprint 68 should remove real local ownership pressure
- it should not trade one giant readable owner for opaque registration tricks

#### 5. The Day 6-7 touched-file fence is now fixed and small

Required first-batch implementation surface:

- `tests/test_chol_csc.c`

Support only if the landed extraction truly needs it:

- `tests/test_chol_csc_supernodal_helpers.h`

Proof/support surfaces that stay out unless the landed refactor unexpectedly
moves ownership wording:

- `tests/test_integration.c`
- `README.md`
- `docs/maintainer_guide.md`
- `benchmarks/README.md`

Interpretation:

- the first implementation batch can still stay family-local by default
- oracle/docs widening is now explicitly conditional rather than assumed

#### 6. The explicit non-widening fence is now strong enough to keep the landing honest

The first Cholesky CSC test landing should not widen into:

- new `tests/test_chol_csc_*.c` binaries
- shared cross-family test helper layers
- `tests/test_integration.c`
- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- implementation `src/` files
- benchmark or maintained-doc truth surfaces

That matters because Sprint 68 still has real later lanes after the first
refactor landing:

- ND chronology follow-through
- oracle/parity expansion
- property/fuzz expansion
- platform-confidence follow-through

### Day 5 Close

Sprint 68 Day 5 closes with one exact first implementation contract:

1. required first batch:
   - `tests/test_chol_csc.c`
2. support only if needed:
   - `tests/test_chol_csc_supernodal_helpers.h`
3. keep as durable owner:
   - one canonical `test_chol_csc` binary
   - scenario assertions and proof bodies in the main file
4. likely extraction lane:
   - supernodal/writeback/dispatch support helpers only
5. explicit non-touch set:
   - oracle lane
   - other giant tests
   - implementation files
   - benchmark/docs truth surfaces

That gives Day 6 one exact job:

- land one bounded `test_chol_csc.c` helper-extraction batch without widening
  into a broader test-suite redesign

## Day 6 - Giant-Test Refactor Batch 1

### Goal

Land the first bounded giant-test refactor batch inside the Day 5 fence by
reducing local support-helper pressure in `tests/test_chol_csc.c` without
widening into new test binaries, oracle surfaces, or implementation files.

### Actions

1. Re-read the Day 5 design contract and the live first-batch targets:
   - `tests/test_chol_csc.c`
   - `tests/test_chol_csc_supernodal_helpers.h`
2. Identified the strongest family-local support helpers that could move
   cleanly without shifting proof ownership:
   - supernode diagonal-block lookup support
   - scalar-vs-batched factored CSC comparison support
   - writeback round-trip comparison scaffolding
3. Landed the bounded helper extraction into the existing family-local header:
   - `day7_chol_csc_get(...)`
   - `day8_chol_csc_match(...)`
   - `day10_factored_matches(...)`
   - `day10_roundtrip_check(...)`
4. Verified that the batch stayed inside the Day 5 fence:
   - no new `test_chol_csc_*.c` binaries
   - no `tests/test_integration.c` widening
   - no shared cross-family helper layer
   - no implementation-file edits
5. Ran the required validation and the stronger reviewed-quality path.

### Findings

#### 1. `tests/test_chol_csc.c` now reads more like the canonical proof owner and less like a mixed support-helper bucket

The Day 6 batch moved four family-local support helpers into the existing
`tests/test_chol_csc_supernodal_helpers.h` seam:

- `day7_chol_csc_get(...)`
- `day8_chol_csc_match(...)`
- `day10_factored_matches(...)`
- `day10_roundtrip_check(...)`

Those helpers were previously embedded directly in the main test owner even
though they primarily serve:

- supernode diagonal-block reference lookup
- scalar-vs-batched factored CSC comparisons
- writeback round-trip scaffolding

Interpretation:

- the main file now spends less visible surface area on local support plumbing
- the family-local proof bodies remain in the canonical owner instead of being
  split into new binaries

#### 2. The landed batch kept the family-local helper seam narrow and specific

The helper extraction stayed inside the existing header:

- `tests/test_chol_csc_supernodal_helpers.h`

No new generic test helper layer was introduced.

That matters because the extracted helpers are still tightly tied to:

- Cholesky CSC factored-structure comparisons
- supernodal/writeback proof scaffolding
- dispatch-fixture preparation inside this family only

Interpretation:

- Sprint 68 got a real local maintainability win
- it did not pay for that win by creating a vague cross-family abstraction

#### 3. The first landed extraction stayed inside the exact Day 5 fence

Touched test surfaces:

- `tests/test_chol_csc.c`
- `tests/test_chol_csc_supernodal_helpers.h`

The batch did not widen into:

- new `tests/test_chol_csc_*.c` binaries
- `tests/test_integration.c`
- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- any `src/` implementation file
- benchmark or maintained-doc truth surfaces

Interpretation:

- the landed batch is a real helper extraction, not a disguised broader
  assurance or architecture wave
- the remaining Sprint 68 lanes stay available for later rerank

#### 4. The reviewed baseline stayed intact after the giant-test refactor batch

Because `*.c` / `*.h` changed, the required validation set was:

- `make format`
- `make lint`
- `make test`

And because this was substantial giant-test architecture work, the stronger
reviewed path was also run:

- `make quality-review-full`

The reviewed CMake parity anchor remained:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

No reviewed CMake failed-test log was left behind after the run.

### Day 6 Close

Sprint 68 Day 6 now hands off one concrete first landing result:

1. `tests/test_chol_csc.c`
   - retains the scenario assertions and canonical family-local proof bodies
2. `tests/test_chol_csc_supernodal_helpers.h`
   - now owns more of the narrow supernodal/writeback support scaffolding
3. the batch stayed inside the exact two-file landing fence
4. validation and the stronger reviewed path completed from the landed state

That gives Day 7 one exact follow-through job:

- rerank the remaining giant-test and assurance queue after the first landed
  `test_chol_csc` helper extraction

## Day 7 - Post-Landing Audit & Assurance Rerank

### Goal

Rerank the remaining Sprint 68 queue from the landed Day 6 branch state so the
next batch is chosen from the actual maintenance pressure that remains, not
from the pre-landing hotspot map.

### Actions

1. Re-read the Day 5 design contract and the landed Day 6 artifact.
2. Re-measured the strongest remaining giant-test and assurance-owner surfaces:
   - `tests/test_chol_csc.c`
   - `tests/test_chol_csc_supernodal_helpers.h`
   - `tests/test_reorder_nd.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_integration.c`
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`
   - `tests/test_svd.c`
3. Re-read the live large-`n` Cholesky public/oracle proof lanes in:
   - `tests/test_integration.c`
   - `tests/test_chol_csc.c`
4. Re-ranked the residual queue across:
   - remaining pure giant-test refactor seams
   - strongest second-layer oracle opportunity
   - later property/fuzz and platform-confidence follow-through
5. Fixed the exact Day 8-10 target set from the live post-Day-6 state.

### Findings

#### 1. The Day 6 batch closed the strongest pure helper-extraction contradiction inside `tests/test_chol_csc.c`

After the landed helper move:

- `tests/test_chol_csc.c` dropped to `4608` lines from the pre-Day-6 `4751`
  line state
- the family-local helper seam in
  `tests/test_chol_csc_supernodal_helpers.h` now carries more of the narrow
  supernodal/writeback support load
- the main file still remains large, but it now reads more consistently as the
  canonical family-local proof owner

Interpretation:

- a second immediate `test_chol_csc.c` helper-only batch is no longer the
  strongest next move
- the biggest first-order helper-pressure contradiction in that file is already
  materially smaller

#### 2. `tests/test_reorder_nd.c` is now the strongest remaining pure refactor seam, but not the highest-value next move

`tests/test_reorder_nd.c` still stands out because it remains:

- `2262` lines
- the strongest chronology-heavy residual giant test
- the clearest next pure refactor lane if Sprint 68 wanted a second
  maintainability-first batch

But it is no longer the strongest immediate value because:

- Day 6 already captured the best first helper-extraction payoff
- Sprint 68 still owes a stronger second-layer assurance batch
- the public large-`n` CSC-backed Cholesky lane now has the better
  confidence-per-change opportunity

Interpretation:

- `tests/test_reorder_nd.c` is now the strongest deferred pure refactor target
- it should not displace the next oracle lane

#### 3. The strongest remaining next move is now the shared oracle owner on the large-`n` CSC-backed Cholesky public path

The highest-value next batch now sits in:

- `tests/test_integration.c`

Why this lane now ranks first:

- it is the shared owner for public one-shot versus explicit repeated-run proof
- the large-`n` CSC-backed Cholesky path is still one of Epic 6's hardest
  retained numerical lanes
- Day 6 reduced family-local helper clutter enough that the next best value is
  stronger public-path oracle coverage, not another immediate local split

The family-local support context remains:

- `tests/test_chol_csc.c`

Interpretation:

- Day 8 should design one bounded large-`n` Cholesky oracle/parity batch
- the owner should be the public/oracle surface first, not the next giant-test
  refactor seam

#### 4. The exact Day 8-10 target set is now fixed

Strongest next batch:

- large-`n` CSC-backed Cholesky public-path oracle/parity expansion

Required likely owner:

- `tests/test_integration.c`

Likely support only if the oracle shape truly needs a family-local comparison
fixture:

- `tests/test_chol_csc.c`

Current likely non-touch set for the next batch:

- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `tests/test_svd.c`
- implementation `src/` files
- benchmark/docs truth surfaces

Interpretation:

- the queue is now smaller and more explicit
- Sprint 68 should move from first-wave test refactor to second-layer oracle
  work next

### Day 7 Close

Sprint 68 Day 7 fixes the post-Day-6 ranked order as:

1. strongest next move:
   - `tests/test_integration.c`
   - one bounded large-`n` CSC-backed Cholesky oracle/parity batch
2. likely support only if needed:
   - `tests/test_chol_csc.c`
3. strongest deferred pure refactor seam:
   - `tests/test_reorder_nd.c`
4. later assurance/follow-through owners:
   - `tests/test_ldlt_csc.c`
   - `tests/test_iterative.c`
   - `tests/test_eigs.c`
   - `tests/test_svd.c`

That gives Day 8 one exact job:

- define the bounded large-`n` CSC-backed Cholesky oracle/parity contract in
  the public integration owner

## Day 8 - Differential/Oracle Coverage Design

### Goal

Define one bounded second-layer oracle/parity batch for the large-`n`
CSC-backed Cholesky public path so Day 9 strengthens public confidence on the
hardest retained direct lane without duplicating family-local proof.

### Actions

1. Re-read the Day 7 rerank artifact and Sprint 68 plan fence.
2. Re-read the current public large-`n` CSC-backed Cholesky proof owner in:
   - `tests/test_integration.c`
3. Re-read the current family-local support context in:
   - `tests/test_chol_csc.c`
4. Mapped the current public-path proof split across:
   - one-shot `factor_opts` vs explicit analysis-path parity
   - repeated-run same-pattern refactor vs one-shot parity
   - failure-preservation and path-selection publication
5. Reduced that map to one bounded Day 9 oracle/parity contract, one explicit
   tolerance/failure contract, and one small file fence.

### Findings

#### 1. The strongest Day 9 assurance owner is the existing public lifecycle parity lane in `tests/test_integration.c`

The next batch should center on the current public owner instead of inventing a
new oracle lane:

- `test_cholesky_factor_opts_matches_explicit_analysis_path(...)`
- `test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky(...)`

Why this is the right owner:

- it already owns the public one-shot versus explicit repeated-run contract
- it already sits on the large-`n` CSC-backed side of the Cholesky path
- it can absorb one stronger parity/oracle batch without widening into
  implementation details

Interpretation:

- Day 9 should strengthen the public-path integration proof directly
- it should not create a parallel second oracle in a family-local test first

#### 2. The strongest additive proof is a staged public-path parity oracle, not another family-local helper check

The current public lane already proves:

- one-shot `factor_opts` matches the explicit analysis path
- same-pattern repeated-run refactor matches one-shot at two stages

The missing strength is that those proofs are still separated instead of
showing one continuous public-path story:

- build one large-`n` baseline on the CSC side
- confirm one-shot and explicit repeated-run agree
- refactor to a same-pattern second SPD matrix and confirm they still agree
- refactor to a same-pattern third SPD matrix and confirm they still agree
- keep the exact-solution oracle fixed so every stage checks both:
  - public-path parity
  - external-style numerical correctness

Interpretation:

- the new batch should unify and deepen the public-path parity story
- it should add one stronger oracle lane rather than duplicating existing
  family-local CSC helper checks

#### 3. The Day 9 tolerance and failure-classification contract is now explicit

Intended Day 9 oracle contract:

- matrix size must stay on the CSC side:
  - `n >= SPARSE_CSC_THRESHOLD`
- every public one-shot solve and explicit repeated-run solve must agree with
  the fixed exact solution to:
  - `1e-12`
- every public one-shot solve and explicit repeated-run solve pair must agree
  with each other to:
  - `1e-12`
- if path-publication state is observed, the test should assert CSC-side
  routing explicitly rather than assuming it implicitly

Failure classification:

- not a failure-preservation batch
- not a family-local kernel/residual batch
- not a benchmark/throughput batch
- one bounded public oracle/parity batch on valid same-pattern SPD transitions

Interpretation:

- Day 9 should improve assurance on the success-path parity lane
- it should not mix in unrelated error-path or performance claims

#### 4. `tests/test_chol_csc.c` is support context only, and likely not required for the landing

The family-local file remains useful background because it already owns:

- large-`n` analysis-backed helper parity
- CSC-side dispatch/path publication proof
- supernodal residual and writeback family-local contracts

But the current design does not require touching it if the public integration
test can carry the full oracle/parity batch alone.

Interpretation:

- likely required Day 9 owner:
  - `tests/test_integration.c`
- likely support:
  - none
- `tests/test_chol_csc.c` should stay untouched unless the final test shape
  proves it truly needs a shared family-local fixture

#### 5. The Day 9 file fence is now fixed and small

Required likely implementation surface:

- `tests/test_integration.c`

Support only if the final oracle shape truly needs it:

- `tests/test_chol_csc.c`

Current explicit non-touch set:

- `tests/test_reorder_nd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_iterative.c`
- `tests/test_eigs.c`
- `tests/test_svd.c`
- implementation `src/` files
- benchmark/docs truth surfaces

Interpretation:

- the next batch can stay truthful and bounded
- Sprint 68 can strengthen assurance without reopening the refactor lane

### Day 8 Close

Sprint 68 Day 8 closes with one exact Day 9 oracle contract:

1. owner:
   - `tests/test_integration.c`
2. likely support only if needed:
   - `tests/test_chol_csc.c`
3. proof shape:
   - large-`n` CSC-backed Cholesky public-path staged parity across multiple
     same-pattern SPD states
4. oracle/tolerance contract:
   - exact-solution agreement at `1e-12`
   - one-shot vs explicit repeated-run agreement at `1e-12`
   - explicit CSC-side routing assertion when publication state is observed
5. explicit non-touch set:
   - other giant tests
   - implementation files
   - benchmark/docs truth surfaces

That gives Day 9 one exact job:

- land one bounded large-`n` CSC-backed Cholesky public-path oracle/parity
  batch in `tests/test_integration.c`

## 2026-06-13 - Day 9: Large-`n` CSC-backed Cholesky public-path oracle/parity batch

### Goal

Land the bounded Day 8 oracle batch in `tests/test_integration.c` by
strengthening the large-`n` CSC-backed Cholesky public-path success-path proof
across multiple same-pattern SPD states, without widening into family-local
helper tests, implementation files, or unrelated giant-test seams.

### Actions

1. Re-read the existing large-`n` public-path parity owner in
   `tests/test_integration.c`, focusing on:
   - `test_cholesky_factor_opts_matches_explicit_analysis_path(...)`
   - `test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky(...)`
2. Confirm the Day 8 target still fit one bounded integration-owner batch:
   - baseline one-shot versus repeated-run parity was already present in pieces
   - the missing strength was one continuous staged oracle across baseline plus
     later same-pattern refactors on the CSC side
3. Extend
   `test_public_lifecycle_refactor_same_pattern_matches_one_shot_cholesky(...)`
   to carry three full stages:
   - baseline factor/solve on the explicit repeated-run lane
   - refactor stage 1
   - refactor stage 2
4. Add one-shot peers for all three stages and attach `used_csc_path` capture
   to each one-shot Cholesky call so CSC-side routing is asserted explicitly
   when publication state is observed.
5. Keep the fixed exact-solution oracle and strengthen the numerical contract at
   each stage:
   - repeated-run solve matches exact solution to `1e-12`
   - one-shot solve matches exact solution to `1e-12`
   - repeated-run and one-shot agree to `1e-12`
6. Keep the batch inside the Day 8 fence:
   - no `tests/test_chol_csc.c` edit
   - no other giant-test edits
   - no `src/` implementation edits
   - no benchmark/docs truth-surface churn

### Files Touched

- `tests/test_integration.c`

### Validation

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed
- reviewed CMake parity remained exact at `53`
- Makefile/CMake parity remained `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 452.07 sec`

### Outcome

The Day 9 batch landed exactly one stronger public-path oracle story:

- the large-`n` CSC-backed Cholesky repeated-run lane is now checked at the
  baseline matrix and two later same-pattern SPD refactor states
- each stage is paired against a one-shot Cholesky solve on the same matrix
- each one-shot stage now asserts `used_csc_path == 1`
- the public-path owner now proves one continuous staged CSC-backed parity lane
  instead of a more fragmented baseline/refactor split

### Notes

- This stayed a success-path oracle/parity batch only.
- It did not widen into failure-preservation, family-local kernel plumbing, or
  performance benchmarking claims.
