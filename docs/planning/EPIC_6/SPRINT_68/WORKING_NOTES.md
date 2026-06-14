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
