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
