# Sprint 96 Working Notes

## Day 1 - Sprint 96 Scope & Hotspot Baseline

### Goal

Open Sprint 96 with a live map of the remaining large-source and giant-test
maintainability candidates after the Sprint 95 public narrative and selected
proof-owner cleanup.

### Actions

- Re-read the Sprint 96 section of
  `docs/planning/EPIC_9/PROJECT_PLAN.md`.
- Re-read the Sprint 96 plan Day 1 scope.
- Re-read the Sprint 90 maintainability/coherence audit and closeout handoff.
- Re-read the Sprint 95 closeout and Sprint 96 handoff queue.
- Measured current source, test, benchmark, example, and public-header file
  sizes from the merged `master` baseline.
- Separated implementation hotspots from proof-owner, benchmark, support-doc,
  public-header, and intentionally historical planning surfaces.
- Recorded authoritative inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Day 1 baseline in
  `artifacts/day1-scope-and-hotspot-baseline.md`.

### Findings

- Sprint 90's broad maintainability map still broadly holds after Sprints
  94-95, but some names and line counts changed.
- `src/sparse_ldlt_csc.c` remains the largest source owner and the clearest
  direct-family candidate.
- `src/sparse_iterative.c`, `src/sparse_lu_csr.c`, `src/sparse_qr.c`,
  `src/sparse_ldlt.c`, and `src/sparse_eigs.c` remain strong solver or
  algorithm candidates for the second cleanup lane.
- `tests/test_chol_csc.c`, `tests/test_ldlt_csc.c`,
  `tests/test_integration.c`, and `tests/test_qr.c` remain the largest
  proof-owner concentrations.
- The Sprint 95 rename moved the selected direct CSC owners to product-oriented
  filenames, so Day 2 should not rerank stale `test_sprint18/19/20` names.
- `docs/algorithm.md` remains a large chronology cleanup candidate, but Day 1
  classifies it as support/documentation follow-through rather than the first
  implementation hotspot.

### Validation

- Day 1 changed planning documentation only.
- No `.c` or `.h` files were modified for Day 1.
- Full `make format && make lint && make test` is not required for this
  docs-only baseline pass.
- Follow-up hygiene checks should include `git diff --check` and
  trailing-whitespace scans on Sprint 96 planning files.

### Day 1 Exit State

Sprint 96 now has a scoped baseline, authoritative inputs, and a current
large-owner candidate list. Day 2 can rank the live implementation and
proof-owner hotspots from measured evidence instead of inheriting Sprint 90
counts directly.

## Day 2 - Hotspot Rerank

### Goal

Rank the Day 1 hotspot candidates by review cost, ownership ambiguity,
extraction risk, and validation blast radius so Sprint 96 has a bounded
fix-now queue.

### Actions

- Re-read the Day 1 scope and hotspot baseline.
- Measured function-like entry density in the top source candidates.
- Measured test/register entry density in the top proof-owner candidates.
- Inspected the largest source owners for mixed responsibility clusters.
- Inspected the largest proof-owner tests for registration and proof clusters.
- Ranked implementation hotspots, proof-owner hotspots, fix-now work, and
  residual maintainability work.
- Recorded the rerank in `artifacts/day2-hotspot-rerank.md`.

### Findings

- `src/sparse_ldlt_csc.c` is the strongest direct-family implementation target:
  it is the largest source owner, has the highest measured function-like entry
  density, and mixes backend dispatch, conversion/writeback, symbolic helpers,
  native numeric work, solve paths, and supernodal behavior.
- `src/sparse_iterative.c` is the strongest default solver-family target: it
  mixes handle/workspace helpers, convergence policy, CG, GMRES, block solver,
  MINRES, and BiCGStab paths.
- `tests/test_chol_csc.c` is the strongest giant-test architecture target:
  it remains the largest proof owner and has the highest measured
  test/register density.
- `tests/test_ldlt_csc.c`, `tests/test_integration.c`, and `tests/test_qr.c`
  remain important proof-owner candidates, but should be kept residual unless
  the selected source cleanup creates a narrow ownership reason to touch them.
- Broad public-header redesigns, benchmark renames, generated documentation
  edits, multi-file source decompositions, and simultaneous giant-test splits
  are outside the Day 2 fix-now scope.

### Validation

- Day 2 changed planning documentation only.
- No `.c` or `.h` files were modified for Day 2.
- Full `make format && make lint && make test` is not required for this
  docs-only rerank pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 96 planning files.

### Day 2 Exit State

Sprint 96 now has a ranked live hotspot map. Day 3 can design a bounded source
extraction plan around `src/sparse_ldlt_csc.c`, keep `src/sparse_iterative.c`
as the default solver-family target, and treat `tests/test_chol_csc.c` as the
provisional giant-test architecture target.

## Day 3 - Source Extraction Design

### Goal

Define the bounded direct-family and solver-family source cleanup boundaries
before moving code.

### Actions

- Re-read the Day 2 hotspot rerank and Day 3 plan.
- Inspected `src/sparse_ldlt_csc.c` function clusters and the existing
  `src/sparse_ldlt_csc_internal.h` declarations.
- Inspected `src/sparse_iterative.c`, `src/sparse_iterative_internal.h`, and
  solver test/build registrations.
- Selected one direct-family cleanup boundary and one solver-family cleanup
  boundary.
- Defined expected touched files, proof owners, stale-reference scans, and
  deferral reasons for adjacent hotspots.
- Recorded the design in `artifacts/day3-source-extraction-design.md`.

### Findings

- The best direct-family extraction boundary is the LDLT dense block factor
  and runtime-selected backend cluster currently embedded near the top of
  `src/sparse_ldlt_csc.c`.
- The existing internal declarations for `ldlt_dense_factor(...)`,
  `ldlt_dense_factor_selected(...)`, and
  `ldlt_dense_factor_backend_name(...)` let the direct cleanup preserve the
  current internal contract while moving implementation ownership.
- The recommended direct implementation shape is a new focused source owner,
  likely `src/sparse_ldlt_dense.c`, registered in both Makefile and CMake.
- The best default solver-family cleanup boundary is the iterative
  block-solver wrapper cluster in `src/sparse_iterative.c`.
- The recommended solver implementation shape, if Day 7 freezes a split, is a
  new focused source owner such as `src/sparse_iterative_block.c` with public
  block solver signatures unchanged.
- Adjacent direct, QR, eigensolver, SVD, matrix-shell, benchmark, public
  header, and generated documentation cleanup remains deferred unless a later
  boundary freeze finds a concrete dependency.

### Validation

- Day 3 changed planning documentation only.
- No `.c` or `.h` files were modified for Day 3.
- Full `make format && make lint && make test` is not required for this
  docs-only design pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 96 planning files.

### Day 3 Exit State

Sprint 96 now has source cleanup boundaries documented before code movement.
Day 4 can freeze the direct-family implementation batch around LDLT
dense/backend ownership, while Day 7 can later freeze the iterative
block-wrapper cleanup batch.

## Day 4 - Direct-Family Cleanup Boundary Freeze

### Goal

Freeze the first direct-family implementation batch and its proof-owner
expectations before editing source files.

### Actions

- Re-read the Day 3 source extraction design and the Day 4 plan.
- Inspected the current dense/backend implementation block in
  `src/sparse_ldlt_csc.c`.
- Checked current test references to `ldlt_dense_factor(...)`,
  `ldlt_dense_factor_selected(...)`, backend environment contracts, and direct
  CSC proof owners.
- Checked Makefile and CMake source/test registrations for LDLT CSC owners.
- Identified include dependencies that should move with the dense/backend
  source owner.
- Recorded the frozen direct-family landing plan in
  `artifacts/day4-direct-family-boundary-freeze.md`.

### Findings

- Day 5 should create `src/sparse_ldlt_dense.c` and move the dense/backend
  implementation block currently at `src/sparse_ldlt_csc.c:38-616`.
- `src/sparse_ldlt_csc_internal.h` already owns the needed internal function
  declarations, so no public API or public header change is needed.
- `Makefile` and `CMakeLists.txt` both need a new
  `src/sparse_ldlt_dense.c` source registration.
- `src/sparse_ldlt_csc.c` should keep `<math.h>`, `<stdlib.h>`, and
  `<string.h>` after the move, but the dense/backend-specific
  `<limits.h>`, `<stdatomic.h>`, `<stdint.h>`, and `<dlfcn.h>` includes should
  move to the new file if compile checks confirm they are no longer used in the
  CSC owner.
- `tests/test_chol_csc.c` owns the direct dense LDLT primitive checks, while
  `tests/test_ldlt_csc.c` owns LDLT CSC supernodal behavior that depends on
  `ldlt_dense_factor_selected(...)`.
- Direct CSC dispatch and regression tests should remain in the targeted proof
  list because this is direct-family source movement.

### Validation

- Day 4 changed planning documentation only.
- No `.c` or `.h` files were modified for Day 4.
- Full `make format && make lint && make test` is not required for this
  docs-only boundary-freeze pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 96 planning files.

### Day 4 Exit State

The first direct-family cleanup batch is ready for implementation without scope
drift. Day 5 should perform one bounded extraction around LDLT dense/backend
ownership, preserve internal signatures, update both build systems, and run the
required full code-day quality chain.

## Day 5 - Direct-Family Source Cleanup Batch 1

### Goal

Land the first bounded direct-family cleanup by separating LDLT dense/backend
ownership from the large CSC implementation owner.

### Actions

- Created `src/sparse_ldlt_dense.c`.
- Moved the dense LDLT block factor and runtime-selected backend implementation
  out of `src/sparse_ldlt_csc.c`.
- Kept the existing internal declarations for `ldlt_dense_factor(...)`,
  `ldlt_dense_factor_selected(...)`, and
  `ldlt_dense_factor_backend_name(...)`.
- Updated `src/sparse_ldlt_csc.c` file-level ownership comments.
- Registered `src/sparse_ldlt_dense.c` in both `Makefile` and
  `CMakeLists.txt`.
- Removed a stale direct line-number comment from `src/sparse_ldlt_csc.c`.
- Ran stale-reference scans for dense factor call sites, build registrations,
  and moved ownership comments.
- Recorded the implementation batch in
  `artifacts/day5-direct-family-source-cleanup-batch1.md`.

### Findings

- The extraction was behavior-preserving: no public headers, public APIs,
  numerical recurrence logic, test registrations, benchmark drivers, or
  generated documentation changed.
- `src/sparse_ldlt_csc.c` remains the owner for CSC allocation, conversion,
  writeback, validation, native elimination, solve, and top-level supernodal
  orchestration.
- `src/sparse_ldlt_dense.c` now owns the dense block factor, backend
  environment parsing, dynamic-loader probe, and backend-name reporting.
- The direct proof owners continue to exercise the moved dense implementation
  through the same internal declarations.
- Day 6 should decide whether any remaining internal-header or local rationale
  comments need cleanup after the extraction.

### Validation

- Day 5 modified `.c` files and build files, so the full required code-day
  quality chain was run.
- `make format && make lint && make test` passed.
- Targeted direct proof owners passed inside the full test run:
  `test_chol_csc`, `test_ldlt_csc`, `test_direct_csc_dispatch`,
  `test_direct_csc_regression`, `test_ldlt`, and
  `test_ldlt_backend_dispatch`.

### Day 5 Exit State

The first direct-family source cleanup batch is landed and validated. Day 6 can
focus on reconciling direct-family comments, proof ownership notes, and any
small residual cleanup created by the extraction.

## Day 6 - Direct-Family Source Cleanup Batch 2

### Goal

Complete the selected direct-family cleanup by reconciling ownership comments,
proof-owner validation, and residual direct-family follow-up after the Day 5
LDLT dense/backend extraction.

### Actions

- Re-read the Day 6 plan and Day 5 implementation artifact.
- Inspected `src/sparse_ldlt_csc_internal.h`,
  `src/sparse_ldlt_csc.c`, `src/sparse_ldlt_dense.c`, and
  `src/sparse_ldlt_csc_supernodal.c` for stale ownership comments.
- Updated `src/sparse_ldlt_csc_internal.h` comments so the internal contract
  names `sparse_ldlt_dense.c` as a current internal consumer and dense
  implementation owner.
- Re-ran the full required code-day quality chain.
- Recorded the direct-family cleanup closeout in
  `artifacts/day6-direct-family-cleanup-closeout.md`.

### Findings

- The Day 5 extraction did not require additional behavior changes.
- The only warranted Day 6 code cleanup was comment-only internal contract
  reconciliation in `src/sparse_ldlt_csc_internal.h`.
- The direct-family ownership boundary is now clear:
  `src/sparse_ldlt_dense.c` owns dense/backend code,
  `src/sparse_ldlt_csc.c` owns CSC scalar/native orchestration, and
  `src/sparse_ldlt_csc_supernodal.c` owns the extracted supernodal helper
  cluster.
- Remaining direct-family work should stay residual unless later sprint days
  identify a narrow dependency.

### Validation

- Day 6 modified an internal header comment, so the full required code-day
  quality chain was run.
- `make format && make lint && make test` passed.
- Targeted direct proof owners passed inside the full test run:
  `test_chol_csc`, `test_ldlt_csc`, `test_direct_csc_dispatch`,
  `test_direct_csc_regression`, `test_ldlt`, and
  `test_ldlt_backend_dispatch`.

### Day 6 Exit State

The selected direct-family source cleanup is complete and validated. Sprint 96
can move to the solver/algorithm cleanup lane with the direct-family residuals
kept explicit and out of scope.

## Day 7 - Solver/Algorithm Cleanup Boundary Freeze

### Goal

Freeze the second implementation cleanup batch in one solver/algorithm hotspot
before editing solver source files.

### Actions

- Re-read the Day 7 plan and the Day 3 source extraction design.
- Inspected the iterative block-solver cluster in `src/sparse_iterative.c`.
- Checked current public block solver declarations in
  `include/sparse_iterative.h`.
- Checked block solver proof owners and build registrations in Makefile,
  CMake, and solver tests.
- Selected one bounded solver cleanup boundary around iterative block-solver
  ownership.
- Defined expected moved symbols, helper dependencies, build updates, proof
  owners, benchmark sanity checks, stale-reference scans, and explicit
  non-goals.
- Recorded the frozen solver landing plan in
  `artifacts/day7-solver-algorithm-boundary-freeze.md`.

### Findings

- The second implementation batch should create
  `src/sparse_iterative_block.c` and move the iterative block-solver ownership
  cluster out of `src/sparse_iterative.c`.
- The frozen move boundary includes `sparse_cg_solve_block(...)`, the shared
  independent-column block dispatch helper, and the GMRES, MINRES, and
  BiCGStab block wrappers.
- The public declarations in `include/sparse_iterative.h` should remain
  unchanged.
- `src/sparse_iterative_internal.h` should be touched only if Day 8 needs a
  narrow private declaration for shared result-reset or converged-state
  helpers.
- Scalar CG, scalar GMRES, matrix-free GMRES, scalar MINRES, scalar and
  matrix-free BiCGStab, workspace internals, QR, eigensolver, SVD, LU, LDLT,
  Cholesky, benchmark-driver, public-header, and generated-documentation
  cleanup remain out of scope.
- The primary proof owners for Days 8-9 are `tests/test_block_solvers.c`,
  `tests/test_minres.c`, `tests/test_bicgstab.c`, `tests/test_iterative.c`,
  and `tests/test_sprint10_integration.c`.

### Validation

- Day 7 changed planning documentation only.
- No `.c` or `.h` files were modified for Day 7.
- Full `make format && make lint && make test` is not required for this
  docs-only boundary-freeze pass.
- Required hygiene checks: `git diff --check` and a trailing-whitespace scan on
  Sprint 96 planning files.

### Day 7 Exit State

The solver/algorithm cleanup lane is ready for implementation. Day 8 should
land the first bounded source split by creating `src/sparse_iterative_block.c`,
preserving public block solver behavior, registering the new source in both
build systems, and running the full required code-day quality chain.

## Day 8 - Solver/Algorithm Source Cleanup Batch 1

### Goal

Land the first bounded solver/algorithm cleanup by separating iterative
block-solver ownership from the main iterative solver source file.

### Actions

- Created `src/sparse_iterative_block.c`.
- Moved the block CG, block GMRES, block MINRES, and block BiCGStab public
  implementations out of `src/sparse_iterative.c`.
- Moved the shared independent-column block dispatch helper and block column
  adapter helpers into the new source owner.
- Kept scalar CG, scalar GMRES, matrix-free GMRES, scalar MINRES, scalar
  BiCGStab, and matrix-free BiCGStab in their existing owners.
- Added a narrow private CG default accessor and shared result helper
  declarations to `src/sparse_iterative_internal.h`.
- Registered `src/sparse_iterative_block.c` in both `Makefile` and
  `CMakeLists.txt`.
- Ran the Day 7 stale-reference scans for block solver implementations and
  build registrations.
- Recorded the implementation batch in
  `artifacts/day8-solver-source-cleanup-batch1.md`.

### Findings

- The source split is behavior-preserving: public solver declarations,
  option/result structs, tests, benchmarks, and generated docs are unchanged.
- Block CG still owns the block workspace and block SpMV algorithm, but now
  gets its default options through `s85_iter_cg_defaults(...)` instead of
  referencing a file-local static directly.
- Block GMRES, MINRES, and BiCGStab still solve each right-hand side through
  the corresponding scalar public solver.
- The only internal-header contract expansion was the minimum needed to share
  result reset/converged semantics and CG default options with the new source
  owner.
- Day 9 should be a closeout pass for comments, helper-name clarity, and any
  stale ownership wording found after the split.

### Validation

- Day 8 modified `.c`, `.h`, and build files, so the full required code-day
  quality chain was run.
- `make format` passed.
- `make lint` passed.
- `make test` passed.
- Solver proof owners passed inside the full test run:
  `test_block_solvers`, `test_minres`, `test_bicgstab`, `test_iterative`,
  `test_sprint10_integration`, and `test_sprint13_integration`.
- Stale-reference scans confirmed public declarations stayed in
  `include/sparse_iterative.h`, block implementations moved to
  `src/sparse_iterative_block.c`, and both build systems register the new
  source file.

### Day 8 Exit State

The selected solver/algorithm source split is landed and validated. Day 9 can
finish the solver cleanup lane by reconciling ownership comments and internal
helper documentation while keeping public behavior unchanged.

## Day 9 - Solver/Algorithm Source Cleanup Batch 2

### Goal

Complete the selected solver/algorithm cleanup by reconciling current ownership
comments, helper documentation, validation notes, and residual queue items
after the Day 8 iterative block-solver source split.

### Actions

- Re-read the Day 8 implementation artifact and Day 9 plan.
- Inspected `src/sparse_iterative.c`, `src/sparse_iterative_block.c`, and
  `src/sparse_iterative_internal.h` for stale ownership wording and unclear
  helper comments.
- Added a current-ownership comment to `src/sparse_iterative_block.c` so it
  names the multiple-RHS public entry points and per-column adapter glue it now
  owns.
- Added a narrow comment in `src/sparse_iterative.c` explaining why
  `s85_iter_cg_defaults(...)` is shared with the block solver owner.
- Added an internal-header comment clarifying that shared result/default
  helpers remain private to source files under `src/`.
- Re-ran the full required code-day quality chain.
- Recorded the solver cleanup closeout in
  `artifacts/day9-solver-cleanup-closeout.md`.

### Findings

- No solver behavior changes were needed after the Day 8 split.
- The selected solver boundary is now complete: block solver public entry
  points live in `src/sparse_iterative_block.c`, public declarations remain in
  `include/sparse_iterative.h`, and scalar solver owners remain unchanged.
- The only warranted Day 9 code cleanup was comment-only ownership and helper
  documentation.
- No additional Sprint 96 solver behavior work is queued.
- Deferred solver-adjacent work remains outside this cleanup lane: scalar
  algorithm rewrites, iterative workspace implementation changes, public API
  restructuring, benchmark-driver restructuring, and proof-owner test splitting.

### Validation

- Day 9 modified `.c` and `.h` comments, so the full required code-day quality
  chain was run.
- `make format && make lint && make test` passed.
- Solver proof owners passed inside the full test run:
  `test_block_solvers`, `test_minres`, `test_bicgstab`, `test_iterative`,
  `test_sprint10_integration`, and `test_sprint13_integration`.
- The lint build compiled benchmark and example binaries without executing
  them.

### Day 9 Exit State

The selected solver/algorithm source cleanup is complete and validated. Sprint
96 can move to the giant-test architecture lane with the direct-family and
solver-family implementation cleanup lanes closed for the selected scope.

## Day 10 - Giant-Test Architecture Design

### Goal

Design one bounded reduction in giant proof-owner concentration before moving
test code.

### Actions

- Re-read the Day 10 plan and the Day 9 solver cleanup closeout.
- Re-measured the largest retained `tests/test_*.c` proof owners.
- Reviewed the largest candidate, `tests/test_chol_csc.c`, including its
  runner groups and existing `tests/test_chol_csc_supernodal_helpers.h`
  helper boundary.
- Checked Makefile and CMake test registration consequences for adding one new
  Cholesky CSC test executable.
- Selected a bounded split target:
  `tests/test_chol_csc_supernodal.c`.
- Recorded the split boundary, build consequences, validation contract, and
  residual queue in
  `artifacts/day10-giant-test-architecture-design.md`.

### Findings

- `tests/test_chol_csc.c` is the largest retained proof-owner test at 5029
  lines.
- The file already separates core CSC, supernodal, writeback, and dispatch
  coverage through runner groups, so it can be split without changing
  assertions or proof intent.
- The first split should move supernode detection, postorder, dense
  supernodal primitives, dense backend contracts, extract/writeback plumbing,
  diagonal factor, panel, parametrised scalar/batched cross-check, and CSC
  linked-list writeback tests into `tests/test_chol_csc_supernodal.c`.
- Allocation/growth, conversion, permutation cache, symbolic validation,
  workspace scaffold, scalar kernel, solve/residual/shim, transparent
  dispatch, and external dense-reference tests should remain in
  `tests/test_chol_csc.c`.
- Leaving dispatch in the core file avoids expanding platform and public
  option-routing churn during the first split.
- `tests/test_ldlt_csc.c` remains the next strongest direct-family proof-owner
  candidate but should not be mixed into this batch.

### Validation

- Day 10 changed planning documentation only.
- No `.c` or `.h` files were modified by the Day 10 artifact.
- The implementation-day validation contract for the planned split is:
  `make format && make lint && make test`.

### Day 10 Exit State

The giant-test architecture lane is ready for implementation. Day 11 should
create `tests/test_chol_csc_supernodal.c`, register it in Makefile and CMake,
move only the selected supernodal/writeback proof groups, and preserve all test
semantics.

## Day 11 - Giant-Test Cleanup Batch 1

### Goal

Implement the first bounded giant-test reduction by splitting the Cholesky CSC
supernodal/writeback proof groups into their own test executable.

### Actions

- Created `tests/test_chol_csc_supernodal.c`.
- Moved supernode detection, supernodal postorder, dense primitive/backend,
  dense LDLT cross-check, supernode extract/writeback, diagonal factor, panel,
  parametrised scalar/batched, and CSC writeback proof groups out of
  `tests/test_chol_csc.c`.
- Kept allocation/growth, conversion, permutation cache, symbolic validation,
  workspace, scalar kernel, solve/residual/shim, transparent dispatch, and
  external dense-reference proof groups in `tests/test_chol_csc.c`.
- Registered `test_chol_csc_supernodal` in both Makefile and CMake.
- Reconciled `tests/test_chol_csc_supernodal_helpers.h` so it describes the
  new supernodal/writeback proof owner instead of the old monolithic file.
- Moved `test_factor_with_analysis_large_n_matches_explicit_supernodal_route`
  back to `tests/test_chol_csc.c` after the focused build showed its runner
  still belongs to the solve/residual/shim group.
- Recorded the implementation batch in
  `artifacts/day11-giant-test-cleanup-batch1.md`.

### Findings

- The split is behavior-preserving: assertions, tolerances, matrix fixtures,
  and skip/platform behavior were not intentionally changed.
- `tests/test_chol_csc.c` now contains the core Cholesky CSC and dispatch
  proof groups.
- `tests/test_chol_csc_supernodal.c` owns the internal supernodal/writeback
  proof groups under suite label `chol_csc_supernodal`.
- The deterministic dispatch SPD builder is local to the core test file
  because dispatch remained there.
- The helper header remains family-local and is included by the new
  supernodal/writeback test owner.

### Validation

- Focused checks passed:
  `make build/test_chol_csc build/test_chol_csc_supernodal`,
  `./build/test_chol_csc`, and `./build/test_chol_csc_supernodal`.
- Focused suite results:
  `test_chol_csc` ran 92 tests with 0 failures, and
  `test_chol_csc_supernodal` ran 60 tests with 0 failures.
- Stale-reference scans confirmed supernodal/writeback runners live in
  `tests/test_chol_csc_supernodal.c`, dispatch remains in
  `tests/test_chol_csc.c`, and both build systems register the new test.
- Required code-day quality chain passed:
  `make format && make lint && make test`.

### Day 11 Exit State

The first giant-test cleanup batch is complete and fully validated. Cholesky
CSC proof ownership is now split between core/dispatch coverage and
supernodal/writeback coverage without changing production behavior.

## Day 12 - Internal Comment & Rationale Cleanup

### Goal

Remove stale chronology on files touched in Days 5-11 while preserving durable
algorithm rationale, ownership boundaries, invariants, and compatibility
constraints.

### Actions

- Re-scanned direct-family, solver, Cholesky CSC proof-owner, helper, and build
  files touched in Days 5-11 for stale `Sprint`/`Day` chronology and helper
  names.
- Rewrote `tests/test_chol_csc.c` comments so they describe current core CSC
  proof groups instead of the original development sequence.
- Rewrote `tests/test_chol_csc_supernodal.c` comments so they describe current
  supernodal/writeback proof groups and durable corpus-safety contracts.
- Renamed Cholesky CSC helper functions from day-number names to
  intent-based names.
- Rewrote selected LDLT CSC comments in `src/sparse_ldlt_csc.c`,
  `src/sparse_ldlt_dense.c`, and `src/sparse_ldlt_csc_internal.h` to describe
  current dense/backend, 2x2 atomicity, analysis-aware conversion,
  wrapper/native, workspace, solve, and batched supernodal contracts.
- Re-scanned solver cleanup files and left them unchanged because the Day 9
  ownership comments already describe current behavior.
- Recorded the cleanup in
  `artifacts/day12-internal-comment-rationale-cleanup.md`.

### Findings

- Most stale chronology was concentrated in proof-owner tests that were
  originally grouped by implementation day.
- Helper names such as `day8_count_supernodes` and `day10_roundtrip_check`
  obscured current intent after the Day 11 split; intent-based names are now
  clearer.
- Some LDLT wrapper/native history remains intentionally because it explains
  why the wrapper path is still compiled and why tests can override kernel
  selection.
- Build-system historical comments unrelated to Sprint 96 touched ownership
  were not changed.

### Validation

- Focused checks passed:
  `make build/test_chol_csc build/test_chol_csc_supernodal build/test_ldlt_csc build/test_direct_csc_dispatch build/test_direct_csc_regression`,
  `./build/test_chol_csc`, `./build/test_chol_csc_supernodal`,
  `./build/test_ldlt_csc`, `./build/test_direct_csc_dispatch`, and
  `./build/test_direct_csc_regression`.
- Required code-day quality chain passed:
  `make format && make lint && make test`.

### Day 12 Exit State

The rationale cleanup is implemented. Touched files now emphasize current
ownership and invariants, with historical details retained only where they
explain active compatibility contracts.

## Day 13 - Full Validation & Residual Queue

### Goal

Validate the Sprint 96 source and proof-owner cleanup as a whole, re-check
registrations and renamed owners, and freeze the residual maintainability queue
for closeout.

### Actions

- Ran the required full code-day quality chain:
  `make format && make lint && make test`.
- Re-checked Makefile and CMake registrations for the new Sprint 96 owners:
  `src/sparse_ldlt_dense.c`, `src/sparse_iterative_block.c`, and
  `tests/test_chol_csc_supernodal.c`.
- Re-ran stale-helper scans for the Day 12 Cholesky CSC helper and dispatch
  names that were replaced with intent-based names.
- Re-measured current source and proof-owner line counts for completed and
  residual Day 2 hotspots.
- Recorded the validation and residual queue in
  `artifacts/day13-validation-and-residual-queue.md`.

### Findings

- The full quality chain passed, including formatting, lint, strict warning
  compilation, example/benchmark builds, and the full test suite.
- The new Sprint 96 source and proof-owner files are present in both Makefile
  and CMake registrations.
- Removed Day-numbered Cholesky CSC helper names do not remain in the split
  Cholesky proof-owner files.
- The completed fix-now queue is bounded to three lanes: LDLT CSC dense
  extraction, iterative block extraction, and Cholesky CSC proof-owner split.
- Remaining large source/proof owners are residual work, not hidden Sprint 96
  completion gaps.

### Validation

- Required code-day quality chain passed:
  `make format && make lint && make test`.
- Registration scan passed:
  `rg -n "test_chol_csc_supernodal|sparse_iterative_block|sparse_ldlt_dense" Makefile CMakeLists.txt`.
- Removed-helper scan passed with no matches in the split Cholesky CSC proof
  files.

### Day 13 Exit State

Branch-wide validation is complete. The residual maintainability queue is
frozen for Day 14 closeout, with completed Sprint 96 work separated from
deferred cleanup candidates and intentional non-goals.

## Day 14 - Sprint 96 Closeout

### Goal

Close Sprint 96 from validated evidence, map every project-plan item to a
clear done/deferred status, and hand Sprint 97 a bounded maintainability queue.

### Actions

- Re-read the Sprint 96 Epic 9 project-plan section against completed
  artifacts.
- Confirmed all seven project-plan items are done or explicitly bounded by the
  residual queue.
- Recorded final ownership snapshots for the direct-family, solver-family, and
  Cholesky CSC proof-owner cleanup lanes.
- Created the Day 14 closeout artifact:
  `artifacts/day14-sprint96-closeout.md`.
- Created the Sprint 96 retrospective:
  `RETROSPECTIVE.md`.
- Used Day 13's full quality chain as the final source validation anchor
  because Day 14 changed planning documentation only.

### Findings

- Sprint 96 completed the selected fix-now queue: LDLT CSC dense/backend
  extraction, iterative block-wrapper extraction, Cholesky CSC proof-owner
  split, touched-owner rationale cleanup, and full validation.
- The sprint improved owner clarity more than it reduced total touched-area
  line count; extracted and split code moved to named owners.
- The highest-priority Sprint 97 queue should start with
  `tests/test_ldlt_csc.c`, QR source/proof cleanup, eigensolver lifecycle
  cleanup, and `tests/test_integration.c` split design.
- Public-header redesign, benchmark renames, generated documentation edits,
  repo-wide chronology cleanup, and simultaneous multi-owner splits remain
  intentional non-goals.

### Validation

- Day 14 changed planning documentation only.
- Final Sprint 96 source validation remains the Day 13 passed chain:
  `make format && make lint && make test`.
- Final Day 14 hygiene checks should include `git diff --check` and trailing
  whitespace scans on Sprint 96 planning docs plus touched source/test/build
  files.

### Day 14 Exit State

Sprint 96 is closed. The retrospective, closeout artifact, final validation
anchor, artifact index, and Sprint 97 handoff queue are in place.
