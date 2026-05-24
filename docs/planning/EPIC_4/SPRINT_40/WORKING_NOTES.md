# Sprint 40 Working Notes

## Day 1

**Objective:** Turn the Sprint 40 project-plan scope plus the Epic 4 review
and remediation plan into a concrete baseline/setup package by confirming the
validated starting contract inherited from Epic 3, naming the Sprint 40
workstreams explicitly, and defining the authoritative input surfaces before
deeper lifecycle and architecture auditing begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 40 plan and Epic 4 source artifacts:
   - `sed -n '1,240p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`
3. Inspect current Epic 4 sprint-folder state:
   - `ls -R docs/planning/EPIC_4`
4. Re-read a representative prior sprint Day 1 structure:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_39/WORKING_NOTES.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_39/artifacts/day1-final-audit-baseline.md`
5. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
6. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`

### Day 1 Findings

#### 1. Sprint 40 starts from a validated quality baseline, not from a fresh cleanup queue

The inherited Epic 3 baseline remains stable and usable as the starting point
for Epic 4 architecture work:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- dead-code reporting/checking is already part of the maintained surface:
  - `make deadcode-report`
  - `make deadcode-check`
- the Epic 4 plan is therefore not opening with “rebuild the baseline”
  work; it is opening with inventory and contract-definition work layered on a
  stable reviewed/dead-code/readiness floor

Interpretation:

- Sprint 40 is an audit-and-architecture sprint, not a quality-repair sprint
- the main preserve-not-reopen contract is the post-Epic-3 validated local
  reviewed baseline

#### 2. The Sprint 40 workstreams are already well-bounded by the Epic 4 review and remediation plan

Day 1 did not surface a hidden extra scope beyond what Sprint 40 already says
it will do. The workstreams remain:

- baseline capture
- lifecycle inventory
- state-model taxonomy
- future handle-model design
- quality-contract ownership map
- public migration-risk audit
- validation anchor

Interpretation:

- Sprint 40 should stay audit-first and architecture-first
- it should not drift into early implementation batches that belong to Sprints
  41-42 and later

#### 3. The authoritative input surfaces for Sprint 40 are explicit

Highest-value authoritative inputs for the sprint are:

- Sprint 40 plan:
  - `docs/planning/EPIC_4/SPRINT_40/PLAN.md`
- Epic 4 review:
  - `docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`
- Epic 4 remediation plan:
  - `docs/planning/EPIC_4/reviews/todo-codex-2026-05-21.md`
- Epic 4 project-plan section:
  - `docs/planning/EPIC_4/PROJECT_PLAN.md`
- inherited baseline contracts:
  - `README.md`
  - `Makefile`
  - `CMakeLists.txt`
  - `build/deadcode/report.md`
  - `build/deadcode/report.tsv`
  - `build/deadcode/coverage-notes.txt`

Interpretation:

- Sprint 40 architecture decisions should trace back to these sources rather
  than ad hoc assumptions
- the sprint has enough documented input to proceed without inventing a new
  review surface

#### 4. The current maintainability/correctness risks are structural, which matches Sprint 40 exactly

The Epic 4 review still points to the same high-value structural themes:

- overloaded mutable `SparseMatrix` lifecycle/state responsibilities
- monolithic subsystem hotspots like `src/sparse_graph.c`
- fragmented allocation/overflow helper patterns
- duplicated quality-contract truth across commands/scripts/docs
- benchmark/tooling surfaces that lag the library’s stronger contracts

Interpretation:

- Sprint 40 should focus on naming architecture boundaries and migration risks
- it should not try to prematurely solve these issues before the inventory and
  taxonomy are complete

#### 5. The first concrete audit order is now clear

The strongest Day 1 to Day 4 sequence is:

1. preserve and restate the reviewed/dead-code/CMake/platform baseline
2. inventory hotspot and allocation-density surfaces
3. begin lifecycle inventory in LU/Cholesky/LDLT, then extend across QR/SVD /
   analysis / iterative / eigensolver surfaces
4. delay handle-model design until after the lifecycle inventory and taxonomy
   have concrete inputs

Interpretation:

- Sprint 40 now has a stable front half:
  - baseline
  - scope
  - hotspot inventory
  - lifecycle inventory kickoff
- that order reduces the risk of architecture design drifting away from the
  current code/documentation truth

## Day 2

**Objective:** Capture the current maintained local reviewed-quality baseline
as a concrete Epic 4 reference point by separating direct commands, reviewed
Makefile wrappers, reviewed CMake parity wrappers, and adjacent dead-code / CI
surfaces without conflating their roles.

### Commands Run

1. Re-read the Sprint 40 Day 2 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
2. Re-read the current Sprint 40 working baseline:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/WORKING_NOTES.md`
3. Re-read the top-level build command map in `README.md`:
   - `sed -n '80,220p' README.md`
4. Re-read the maintained quality wrapper and dead-code sections in
   `Makefile`:
   - `sed -n '430,760p' Makefile`
5. Sweep current reviewed/dead-code/CI contract wording:
   - `rg -n "quality-review|quality-review-cmake|quality-review-full|deadcode|Cross-Platform CI Contract|Quality Readiness Checklist|warning-workflow" README.md Makefile .github/workflows/*.yml`
6. Re-read the detailed reviewed/dead-code/CI contract sections in
   `README.md`:
   - `sed -n '620,860p' README.md`
7. Re-read the Linux CI workflow as the strongest enforced reviewed baseline:
   - `sed -n '1,220p' .github/workflows/ci.yml`

### Day 2 Findings

#### 1. The maintained local reviewed baseline is a three-layer model, not one command

The current local quality surface splits into three related but distinct
layers:

- direct commands:
  - `make format`
  - `make format-check`
  - `make lint`
  - `make test`
  - `make check`
- reviewed Makefile wrappers:
  - `make quality-review-compile`
  - `make quality-review`
- reviewed CMake parity wrappers:
  - `make quality-review-cmake-compile`
  - `make quality-review-cmake`

Then one top-level aggregate sits above both wrapper families:

- strongest local reviewed baseline:
  - `make quality-review-full`

Interpretation:

- Day 2 confirms that the reviewed-quality baseline is not a single flat
  command list
- later Epic 4 ownership work must preserve the distinction between:
  - direct commands
  - reviewed Makefile path
  - reviewed CMake parity path
  - strongest combined local reviewed baseline

#### 2. The reviewed Makefile path is the local authoritative command layer for formatting, static analysis, runtime tests, and dead-code completeness

The current reviewed Makefile path is:

- `quality-review-compile`
  - `format-check`
  - `lint`
- `quality-review`
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

This makes the Makefile reviewed path the authoritative local owner of:

- formatting truth
- static-analysis truth
- direct runtime test execution
- local dead-code completeness invocation

Interpretation:

- `quality-review` is the local “reviewed Makefile path”, not merely a synonym
  for `check`
- the Makefile reviewed path already owns more than compile cleanliness; it
  includes runtime and dead-code completeness in the maintained local contract

#### 3. The reviewed CMake path is parity-oriented and intentionally narrower in responsibility

The reviewed CMake wrappers own:

- configure / clean rebuild of `build/quality-review-cmake`
- `ctest -N`
- Makefile/CMake test-count parity check
- full CTest execution through `quality-review-cmake`

But they explicitly do **not** replace:

- Makefile-authoritative formatting checks
- Makefile-authoritative static analysis
- Makefile dead-code/reporting commands

Interpretation:

- the reviewed CMake path is parity proof, not the full local quality-policy
  owner
- later Epic 4 ownership mapping should preserve this narrower role instead of
  flattening everything into “CMake and Makefile do the same thing”

#### 4. `quality-review-full` is the strongest local reviewed baseline because it composes the two reviewed paths, not because it supersedes them conceptually

`quality-review-full` is currently defined as:

- `quality-review`
- then `quality-review-cmake`

It is the strongest local reviewed baseline because it composes:

- Makefile reviewed correctness/static-analysis/runtime/dead-code completeness
- CMake rebuild and active-suite parity truth

Interpretation:

- Day 2 confirms the precise meaning of “strongest local reviewed baseline”
- later Epic 4 docs/tooling work should keep this as a composition of two
  reviewed paths rather than collapsing it into a monolithic opaque command

#### 5. The main operational caveats are stable and load-bearing

Two caveat classes remain part of the maintained baseline:

- tree-mutating alternate modes:
  - `sanitize`
  - `asan`
  - `sanitize-all`
  - `tsan`
  - `omp`
  - `coverage*`
  - returning to the normal direct/reviewed path still requires:
    - `make clean`
- serialized dead-code execution:
  - `deadcode*` targets still share:
    - `build/deadcode-cmake`
    - `build/deadcode/`
  - concurrent invocation remains a known race

Interpretation:

- these are not incidental footnotes; they are part of the true local
  operator contract
- Sprint 40 should carry them forward into the future ownership map as
  load-bearing caveats

#### 6. README, Makefile, and CI already have a mostly coherent division of labor, but the ownership boundary is still descriptive rather than explicitly designed

Current practical ownership shape is:

- `Makefile`
  - concrete local commands
  - rerun guidance
  - dead-code invocation and wrapper semantics
- `README.md`
  - operator-facing command map
  - reviewed-quality explanations
  - cross-platform CI contract
  - readiness checklist
- `.github/workflows/*.yml`
  - enforced/staged CI realization of the documented contract

Interpretation:

- the current system is coherent enough to use as a baseline
- but it is still exactly the kind of distributed ownership model the Epic 4
  review said should be tightened and simplified later

## Day 3

**Objective:** Capture the remaining validated support baseline beyond the
direct local wrappers by recording reviewed CMake parity truth, dead-code
report/check semantics, cross-platform enforced/staged/excluded boundaries, and
the current benchmark/example/tooling support surfaces that later Epic 4
refactors must not regress.

### Commands Run

1. Re-read the Sprint 40 Day 3 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
2. Re-read the current Sprint 40 working notes:
   - `sed -n '1,360p' docs/planning/EPIC_4/SPRINT_40/WORKING_NOTES.md`
3. Reconfirm the reviewed CMake suite baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Re-read the current dead-code report and coverage notes:
   - `sed -n '1,240p' build/deadcode/report.md`
   - `sed -n '1,220p' build/deadcode/coverage-notes.txt`
5. Re-read the README support-contract sections:
   - `sed -n '640,840p' README.md`
6. Re-read the platform workflow contracts:
   - `sed -n '1,220p' .github/workflows/ci.yml`
   - `sed -n '1,220p' .github/workflows/macos-ci.yml`
   - `sed -n '1,220p' .github/workflows/windows-ci.yml`
7. Recheck current compile-only/support-target surfaces:
   - `make -n tooling-build bench-build examples-build wall-check coverage | sed -n '1,240p'`

### Day 3 Findings

#### 1. Reviewed CMake parity remains the strongest shared reviewed baseline beyond the Makefile wrappers

The current reviewed CMake parity baseline remains fully explicit:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- `quality-review-cmake-compile` owns:
  - configure
  - clean rebuild
  - `ctest -N`
  - Makefile/CMake test-count parity
- `quality-review-cmake` owns:
  - the compile wrapper
  - full `ctest`

Interpretation:

- Day 3 confirms that reviewed CMake parity is the strongest shared reviewed
  baseline across platforms
- it remains a support baseline adjacent to, but distinct from, the direct
  local Makefile reviewed path

#### 2. The dead-code contract is now a closeout-state completeness workflow, not a discovery queue

The current dead-code report state is:

- compile-db translation-unit counts:
  - `src = 25`
  - `tests = 53`
  - `benchmarks = 14`
  - `examples = 12`
- compile-db benchmark/example coverage gap:
  - closed
- current bucket state:
  - `coverage-gap = 0`
  - `definitely-unused-internal-candidate = 0`
  - `public-surface-review = 4`
  - `secondary-candidate-signal = 35`
  - `non-deadcode-static-analysis-noise = 6`

The current semantics remain:

- `deadcode-report`
  - refreshes evidence and writes classified report outputs
- `deadcode-check`
  - validates report completeness
  - is not a zero-findings gate
- authoritative execution remains serialized because `deadcode*` still shares:
  - `build/deadcode-cmake`
  - `build/deadcode/`

Interpretation:

- the dead-code support baseline is now about truthful reporting and
  completeness, not “find and delete more code”
- later Epic 4 ownership work should preserve this distinction explicitly

#### 3. The cross-platform baseline remains intentionally asymmetric, but the asymmetry is honest and documented

Current enforced/staged/excluded status remains:

- Linux enforced:
  - `make quality-review-compile`
  - `make quality-review-cmake`
  - `make deadcode-report`
  - `make deadcode-check`
- macOS enforced:
  - Apple Clang `make quality-review-compile`
  - Apple Clang `make quality-review-cmake`
  - `make wall-check`
  - `make sanitize`
- macOS staged:
  - `make deadcode-report`
  - `make deadcode-check`
- macOS supplemental:
  - Homebrew GCC direct `make`
  - Homebrew GCC direct `make test`
  - Homebrew GCC `make wall-check`
  - install/pkg-config validation
- Windows enforced:
  - reviewed CMake configure/build
  - `ctest -N`
  - full `ctest`
  - expected registered test count = `50`
- Windows staged:
  - `make quality-review-compile`
  - `make quality-review`
  - dead-code
- Windows excluded:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`

Interpretation:

- Day 3 confirms the current platform baseline is truthfully asymmetric rather
  than incomplete-by-accident
- later Epic 4 ownership work should treat this as deliberate contract
  structure, not a wording error to erase

#### 4. The current benchmark/example/tooling support baseline is broader than the reviewed local wrappers alone

Current support targets still matter as preserve-not-regress surfaces:

- `tooling-build`
  - builds benchmark and example binaries without execution
- `bench-build`
  - current dry-run reports nothing pending because the binaries are already
    built, but it remains the benchmark compile-only support target
- `examples-build`
  - same current state for example binaries
- `wall-check`
  - maintained performance-regression gate
- `coverage`
  - supplemental coverage gate
  - current threshold remains `80%`

Interpretation:

- these are not the strongest local reviewed baseline commands
- but they are part of the inherited support surface that later Epic 4
  refactors must not silently break

#### 5. The support baseline is now broad enough for later ownership mapping work

After Day 3, Sprint 40 now has a captured baseline for:

- direct local reviewed commands
- reviewed CMake parity
- dead-code report/check semantics
- cross-platform enforced/staged/excluded truth
- compile-only/support-target surfaces

Interpretation:

- Sprint 40 can move on to hotspot and lifecycle inventory work without having
  to come back later and reconstruct the inherited support baseline

## Day 4

**Objective:** Measure the structural hotspots that later Epic 4 refactors are
supposed to target by separating true multi-concern maintainability hotspots
from large but legitimate feature-owner files.

### Commands Run

1. Re-read the Sprint 40 Day 4 plan section:
   - `sed -n '1,320p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
2. Re-read the current Sprint 40 working notes:
   - `sed -n '1,520p' docs/planning/EPIC_4/SPRINT_40/WORKING_NOTES.md`
3. Measure the largest source/test/benchmark/example/script files:
   - `python3` line-count inventory over `src/`, `tests/`, `benchmarks/`,
     `examples/`, and `scripts/`
4. Measure raw allocation-density signal across the main C surfaces:
   - `python3` regex count inventory over `malloc`, `calloc`, `realloc`, and
     `free` calls in `src/`, `tests/`, `benchmarks/`, and `examples/`

### Day 4 Findings

#### 1. The highest-signal structural hotspot is still `src/sparse_graph.c`

The largest implementation files in `src/` are:

- `src/sparse_graph.c` = `3555` lines
- `src/sparse_eigs.c` = `3143` lines
- `src/sparse_ldlt_csc.c` = `2723` lines
- `src/sparse_iterative.c` = `2349` lines
- `src/sparse_chol_csc.c` = `2208` lines

Allocation-density signal in `src/` is also led by:

- `src/sparse_graph.c` = `208`
- `src/sparse_svd.c` = `182`
- `src/sparse_ldlt_csc.c` = `156`
- `src/sparse_iterative.c` = `156`
- `src/sparse_ldlt.c` = `148`
- `src/sparse_etree.c` = `144`

Interpretation:

- Day 4 strongly confirms the Epic 4 review judgment that `src/sparse_graph.c`
  is the first-tier architectural decomposition target
- the next implementation-heavy hotspot tier after graph is:
  - eigensolvers
  - iterative solvers
  - CSC LDLT / Cholesky
  - SVD / etree safety-helper concentration

#### 2. The largest tests are real maintainability risks, but many are still feature-owner files rather than immediate split candidates

The largest files in `tests/` are:

- `tests/test_chol_csc.c` = `4643` lines
- `tests/test_svd.c` = `3712` lines
- `tests/test_ldlt_csc.c` = `3637` lines
- `tests/test_qr.c` = `3259` lines
- `tests/test_etree.c` = `2890` lines
- `tests/test_iterative.c` = `2795` lines
- `tests/test_ldlt.c` = `2774` lines
- `tests/test_graph.c` = `2628` lines

Allocation-density signal in `tests/` is led by:

- `tests/test_chol_csc.c` = `451`
- `tests/test_qr.c` = `424`
- `tests/test_ilu.c` = `398`
- `tests/test_iterative.c` = `387`
- `tests/test_etree.c` = `360`
- `tests/test_svd.c` = `339`
- `tests/test_ldlt_csc.c` = `269`
- `tests/test_minres.c` = `266`

Interpretation:

- the large-test queue is real, especially around helper concentration and raw
  allocation density
- but the biggest test files are still strongly organized around feature-owner
  binaries under the repo’s one-binary-per-test model
- the best Day 4 conclusion is not “split every giant test”; it is:
  - prioritize helper extraction and fixture consolidation first
  - treat whole-file splits as second-order work when ownership boundaries are
    genuinely cleaner afterward

#### 3. Benchmark hotspots are narrower and more mixed-concern than the tests

The largest benchmark files are:

- `benchmarks/bench_eigs.c` = `958` lines
- `benchmarks/bench_main.c` = `774` lines
- `benchmarks/bench_ldlt_csc.c` = `516` lines
- `benchmarks/bench_convergence.c` = `421` lines
- `benchmarks/bench_chol_csc.c` = `393` lines

Allocation-density signal in benchmarks is led by:

- `benchmarks/bench_main.c` = `75`
- `benchmarks/bench_convergence.c` = `50`
- `benchmarks/bench_refactor_csc.c` = `44`
- `benchmarks/bench_ldlt_csc.c` = `44`

Interpretation:

- `bench_eigs.c` and `bench_main.c` remain the highest-value benchmark
  maintainability hotspots
- the backend-comparison pair (`bench_chol_csc.c` / `bench_ldlt_csc.c`) is no
  longer the strongest hotspot after Sprint 37’s helper extraction; Day 4
  confirms that that earlier narrow consolidation worked
- the benchmark queue is therefore now centered on:
  - CLI parsing / mode ownership in `bench_main.c`
  - eigensolver workload concentration in `bench_eigs.c`

#### 4. Examples are not the leading maintainability hotspot class

The largest examples are:

- `examples/example_eigs.c` = `284` lines
- `examples/example_ic_minres.c` = `232` lines
- `examples/example_analysis.c` = `191` lines
- `examples/example_ldlt.c` = `186` lines

Allocation-density signal in examples remains small relative to `src/`,
`tests/`, and `benchmarks`.

Interpretation:

- examples remain a secondary safety/usability surface, not a first-tier
  maintainability hotspot
- later Epic 4 example work should be driven by consistency and safety
  standardization rather than large-file decomposition

#### 5. Scripts are concentrated around two real support hotspots

The largest scripts are:

- `scripts/deadcode_report.py` = `523` lines
- `scripts/epic3_warning_workflow.sh` = `215` lines
- `scripts/deadcode_workflow.sh` = `189` lines
- `scripts/wall_check.sh` = `162` lines

Interpretation:

- `scripts/deadcode_report.py` remains the strongest script-side maintainability
  hotspot
- `scripts/deadcode_workflow.sh` remains a meaningful support hotspot because
  it carries workflow topology, tool invocation, and artifact-generation logic
- `epic3_warning_workflow.sh` is larger than the other shell helpers, but it is
  more narrowly scoped and not the leading Epic 4 multi-concern risk

#### 6. The first-tier Day 4 structural target list is now concrete

Current first-tier maintainability/architecture hotspots are:

- `src/sparse_graph.c`
- `src/sparse_eigs.c`
- `src/sparse_iterative.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_svd.c`
- `tests/test_chol_csc.c`
- `tests/test_svd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`
- `benchmarks/bench_main.c`
- `benchmarks/bench_eigs.c`
- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

Interpretation:

- the hotspot queue now aligns tightly with the Epic 4 review:
  - graph decomposition
  - workspace-heavy numeric internals
  - large helper-dense tests
  - benchmark CLI / workload concentration
  - dead-code support-script ownership
- Day 4 therefore gives later Epic 4 sprints a measured target set instead of
  a generic “large files are bad” narrative

## Day 5

**Objective:** Audit the first major stateful matrix/factor API cluster
(LU, Cholesky, LDLT) for in-place mutation, original-matrix assumptions,
permutation assumptions, factored-state ownership, copy-before-reuse
requirements, and cancellation/partial-mutation risk so later Epic 4
handle-model design starts from actual code/documentation truth.

### Commands Run

1. Re-read the Sprint 40 Day 5 plan section:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
2. Re-read the current Sprint 40 working baseline:
   - `sed -n '1,760p' docs/planning/EPIC_4/SPRINT_40/WORKING_NOTES.md`
3. Re-read the public LU / Cholesky / LDLT headers:
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,260p' include/sparse_cholesky.h`
   - `sed -n '1,320p' include/sparse_ldlt.h`
4. Sweep the current public and implementation wording for lifecycle cues:
   - `rg -n "sparse_lu_factor|sparse_cholesky_factor|sparse_ldlt_factor|identity permutations|sparse_copy\\(|row_perm|col_perm|copy-before|bit-identical|factor_norm|factored" README.md docs/tutorial.md src include`
5. Re-read the main README direct-solver / lifecycle / thread-safety sections:
   - `sed -n '150,330p' README.md`
   - `sed -n '520,575p' README.md`
6. Re-read the tutorial’s direct-solver section:
   - `sed -n '100,190p' docs/tutorial.md`
7. Re-read the detailed LDLT header contract block:
   - `sed -n '120,310p' include/sparse_ldlt.h`

### Day 5 Findings

#### 1. LU and Cholesky still carry the heaviest lifecycle overloading on `SparseMatrix`

The LU and Cholesky paths both use a single `SparseMatrix *` as:

- original coefficient container
- mutable factorization workspace
- permutation owner
- post-factor solve handle
- `factored` / `factor_norm` owner

In both APIs, the normal usage pattern is explicitly:

- make a `sparse_copy()` if the original matrix is still needed
- factor the copy in place
- solve through that same now-factored matrix object

Interpretation:

- these are the clearest “matrix object becomes factor handle” surfaces in the
  current public API
- Day 5 confirms the Epic 4 review’s core lifecycle concern with real
  subsystem evidence, not just general suspicion

#### 2. LDLT is already much closer to the future explicit factor-handle model

The LDLT path separates concerns more cleanly:

- factorization takes `const SparseMatrix *A`
- factorization writes to `sparse_ldlt_t`
- solve consumes `const sparse_ldlt_t *`
- cleanup is explicit via `sparse_ldlt_free()`

The input matrix remains the coefficient owner instead of being repurposed as
the factor handle.

Interpretation:

- LDLT already provides an internal example of the kind of state split Epic 4
  is likely to generalize later
- the handle-model design work does not need to invent this boundary from
  scratch; it already exists in one important factorization family

#### 3. “Use a fresh copy” means different things across the cluster

All three surfaces care about an original/original-like matrix state, but for
different reasons:

- LU / Cholesky:
  - because factorization mutates or destroys the matrix representation
- LDLT:
  - because factorization expects identity permutations / original logical
    state even though it does not mutate the input matrix

Interpretation:

- the current public docs can teach “copy before factoring,” but the real
  lifecycle reason differs across APIs
- that difference is exactly the kind of ambiguity Epic 4 should resolve into
  a clearer state taxonomy

#### 4. Cancellation semantics are a first-class lifecycle risk, not a small callback footnote

LU and Cholesky both document and implement early mutation before the first
progress callback can return:

- `factored` is cleared immediately
- `factor_norm` is cached immediately
- Cholesky can also strip the upper triangle before cancellation returns

By contrast, LDLT cancellation:

- frees partial factor output
- leaves the input matrix bit-identical

Interpretation:

- the important risk is not only “factorization mutates”; it is “cancellation
  may leave a non-original object almost immediately”
- that risk belongs in the future lifecycle/handle contract directly

#### 5. Solve-handle identity is inconsistent even within this one factorization cluster

Post-factor solve handle is:

- LU: `SparseMatrix *`
- Cholesky: `SparseMatrix *`
- LDLT: `sparse_ldlt_t *`

Interpretation:

- this is one of the strongest concrete Day 5 findings in support of later
  explicit-handle design work
- direct solvers do not currently present one uniform lifecycle model to the
  caller

#### 6. The documentation layers currently agree on behavior, but they also expose the architectural split

For this cluster, the three main documentation layers are aligned:

- headers carry precise preconditions and cancellation semantics
- `README.md` teaches copy-before-factor and solve-after-factor usage
- `docs/tutorial.md` shows the expected call patterns

Interpretation:

- Sprint 40 Day 5 did not find a truthfulness problem here
- it found an architecture-shape problem: the documented contract is accurate,
  but it is split between matrix-mutating and separate-factor-object designs

#### 7. Day 5 gives the later handle-model work a concrete starting split

The first lifecycle cluster now divides cleanly into:

- matrix-mutating direct factorization paths:
  - LU
  - Cholesky
- separate factor-object path:
  - LDLT

Interpretation:

- Day 5 provides a real architecture input for Days 6-9:
  - what is already close to an explicit handle model
  - what still overloads mutable state onto `SparseMatrix`

## Day 6

**Objective:** Extend the lifecycle inventory across QR, SVD, symbolic
analysis / numeric factorization, iterative solvers, preconditioner builders,
and eigensolvers so the later Epic 4 taxonomy and handle-model design work
sees the full major stateful API surface instead of only the LU/Cholesky/LDLT
cluster.

### Commands Run

1. Re-read the Sprint 40 Day 6 plan section:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/PLAN.md`
2. Re-read the current Sprint 40 working baseline and Day 5 inventory:
   - `tail -n 220 docs/planning/EPIC_4/SPRINT_40/WORKING_NOTES.md`
3. Re-read the public QR / SVD / analysis / iterative / eigensolver headers:
   - `sed -n '1,260p' include/sparse_qr.h`
   - `sed -n '1,280p' include/sparse_svd.h`
   - `sed -n '1,320p' include/sparse_analysis.h`
   - `sed -n '1,320p' include/sparse_iterative.h`
   - `sed -n '1,360p' include/sparse_eigs.h`
4. Re-read the preconditioner-builder headers that shape iterative lifecycle use:
   - `sed -n '1,220p' include/sparse_ilu.h`
   - `sed -n '1,180p' include/sparse_ic.h`
5. Sweep public docs and implementation wording for lifecycle cues:
   - `rg -n "identity permutations|factored|sparse_copy\\(|not modified|progress_cb|SPARSE_ERR_CANCELLED|const SparseMatrix \\*A|sparse_analysis|sparse_factor_numeric|sparse_refactor_numeric|sparse_solve_cg|sparse_solve_gmres|sparse_eigs_sym|sparse_qr_factor|sparse_svd_compute|sparse_svd_partial" README.md docs/tutorial.md include src`
6. Re-read the README / tutorial sections that currently teach these contracts:
   - `sed -n '230,330p' README.md`
   - `sed -n '140,310p' docs/tutorial.md`
7. Re-read key implementation enforcement points:
   - `sed -n '140,320p' include/sparse_analysis.h`
   - `sed -n '1330,1525p' src/sparse_qr.c`
   - `sed -n '700,1045p' src/sparse_svd.c`
   - `sed -n '1,220p' src/sparse_analysis.c`
   - `sed -n '210,330p' include/sparse_iterative.h`
   - `sed -n '540,640p' include/sparse_eigs.h`

### Day 6 Findings

#### 1. QR, SVD, analysis, and preconditioner builders already prefer separate result handles

The strongest Day 6 structural result is that much of the second-half API
surface is already not matrix-as-handle:

- QR uses `sparse_qr_t`
- SVD uses `sparse_svd_t`
- analyze-once uses `sparse_analysis_t` + `sparse_factors_t`
- ILU / ILUT / IC use `sparse_ilu_t`

Interpretation:

- this cluster is much closer to the future explicit-handle direction than
  the LU / Cholesky paths from Day 5
- Epic 4 does not need to force every subsystem into a new model; several
  already live near it

#### 2. The main lifecycle burden in this cluster is strict matrix-eligibility rules

QR, SVD, analysis, ILU, ILUT, and IC all share a similar input-state rule:

- the matrix is not modified
- but the matrix must still represent the original, identity-permutation,
  unfactored physical storage view

Interpretation:

- the risk here is less “hidden mutation” and more “easy-to-violate caller
  preconditions”
- Day 6 therefore broadens the Epic 4 lifecycle problem: it is not only about
  mutable state ownership; it is also about hard-to-remember matrix
  eligibility rules

#### 3. The tutorial and README are already compensating for that matrix-eligibility complexity

The current user-facing docs explicitly teach “fresh copy of the original” or
“use the original unfactored / unreordered matrix” around:

- QR
- SVD
- analyze-once workflow
- ILU / ILUT / IC preconditioner setup

Interpretation:

- the docs are truthful
- but they are also acting as lifecycle safety scaffolding for an API model
  that still requires careful state reasoning from the caller

#### 4. Iterative solvers are best understood as read-only operator consumers

The iterative solver entry points differ sharply from the factor-building APIs:

- `A` is read-only
- no persistent factor object is created by the solver itself
- mutable state lives in:
  - `x`
  - result structs
  - residual history
  - Krylov workspaces
  - optional preconditioner context

Interpretation:

- iterative solvers belong in a different lifecycle class from both
  matrix-mutating direct solvers and separate-handle factor builders
- this gives Day 7 a clean taxonomy candidate: “operator consumers”

#### 5. Preconditioner builders sit between the iterative layer and the factor-handle layer

ILU / ILUT / IC are important because they explain where iterative lifecycle
complexity really comes from:

- the solver itself is read-only on `A`
- the practical workflow still depends on a separate preconditioner handle
- building that handle reintroduces the strict original / identity-permutation
  preconditions

Interpretation:

- for taxonomy purposes, preconditioner builders should group with QR/SVD/
  analysis-style handle builders, not with the iterative solvers that consume
  them

#### 6. Eigensolvers are also operator consumers, but with richer external context composition

The eigensolver layer is read-only on `A` and uses caller-owned result buffers,
but it can compose with:

- external preconditioner contexts for LOBPCG
- internal LDLT factorization for shift-invert
- internal LDLT refinement for post-pass accuracy tightening

Interpretation:

- eigensolvers are still not matrix-as-handle APIs
- but they depend more heavily on other explicit-handle subsystems than the
  plain iterative solve entry points do

#### 7. Analysis is already a transitional explicit-handle subsystem

The analyze-once workflow is the strongest Day 6 bridge case:

- explicit symbolic and numeric handles already exist publicly
- callers already think in terms of:
  - analysis
  - factors
  - solve
- but the factor payload still wraps a `SparseMatrix *F` internally, so the
  old matrix-centric model has not fully disappeared

Interpretation:

- analysis is likely to be one of the most important later Epic 4 migration
  surfaces because it already presents the public shape of the desired
  architecture

#### 8. Cancellation semantics now fall into at least three clear lifecycle classes

Across Day 5 + Day 6, the cancellation classes are now explicit:

- LU / Cholesky:
  - early matrix mutation before callback return
- QR / LDLT / SVD-family handle builders:
  - free intermediates, preserve input matrix
- iterative / eigensolver operator consumers:
  - preserve input matrix and return latest iterate / partial convergence

Interpretation:

- cancellation behavior is now a real taxonomy input, not just a doc note
- later handle-model design must decide which of these classes remain
  user-visible and which should become more internalized
