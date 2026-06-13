# Sprint 67 Working Notes

## Day 1 - Scope Audit & Maintainability Baseline Setup

### Goal

Freeze the Sprint 67 starting point before implementation work begins by
reconfirming the inherited Sprint 66 contract, the preserved reviewed
baseline, the strongest live large-source maintainability hotspots, and the
most important docs/build/implementation/proof surfaces the sprint will touch
next.

### Actions

1. Re-read the Sprint 67 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 66 retrospective, and
   the Sprint 66 Day 14 closeout artifact.
2. Re-read the landed Sprint 67 plan and fixed the bounded workstreams that
   the sprint should actually carry:
   - residual hotspot audit
   - graph/reorder decomposition
   - CSC/iterative residual decomposition
   - comment/chronology cleanup
   - build/regression alignment
   - validation and closeout
3. Reconfirmed the strongest reviewed baseline surfaces:
   - `make quality-review-full`
   - `make -n quality-review-full`
4. Materialized the reviewed CMake parity tree locally and rechecked:
   - `make quality-review-cmake-compile`
   - `ctest -N --test-dir build/quality-review-cmake`
5. Measured the strongest likely Sprint 67 touch surfaces directly from the
   live tree across:
   - maintained truth/build surfaces
   - public coordination headers
   - graph/reorder implementation hotspots
   - CSC/iterative implementation hotspots
   - proof/adoption/regression support surfaces

### Findings

#### 1. Sprint 67 starts from the Sprint 66 productization close, not from renewed build or platform churn

Sprint 66 already closed the main packaging/productization contradiction and
made the platform truthfulness split explicit. That means Sprint 67 is not
reopening:

- static-first package-shape clarification
- install/export/productization tightening
- narrow ABI/version-story clarification
- workflow and CI contract reconciliation
- the deferred platform residual queue except where a touched maintainability
  seam proves it is truly necessary

Interpretation:

- Sprint 67 is the first post-productization Epic 6 sprint centered on source
  ownership and large-file maintainability again
- packaging/platform/build surfaces remain support surfaces only, not the
  implementation center

#### 2. The strongest local reviewed baseline remains the authoritative Sprint 67 starting point

The maintained Day 1 truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The Day 1 supporting rechecks are now explicit:

- `make -n quality-review-full` still expands to the expected reviewed local
  quality path followed by the reviewed CMake parity path
- `make quality-review-cmake-compile` completed cleanly and re-established the
  local reviewed CMake tree before the parity count was recorded

Interpretation:

- Sprint 67 inherits the exact same reviewed baseline story as the Sprint 66
  close
- maintainability work does not get a weaker truth surface just because the
  main sprint topic is decomposition rather than new behavior

#### 3. The highest-value Sprint 67 problem is concentrated in mixed-ownership large-source seams, not in generic repo-wide cleanup

The live repo shows the strongest pressure in:

- graph/reorder orchestration mixed with family-local helpers
- CSC direct-family residuals still carrying too much local orchestration
- iterative/eigensolver residual seams that still compete with coordination in
  large permanent files
- stale sprint-local chronology inside touched permanent implementation
  surfaces

The project-plan scope therefore reduces cleanly to:

1. residual hotspot audit
2. graph/reorder decomposition
3. CSC/iterative residual decomposition
4. comment/chronology cleanup
5. build/regression alignment
6. validation and closeout

Interpretation:

- Sprint 67 should not pretend every large file is equally urgent
- the highest-value work is ownership extraction on the files where
  orchestration, family-local logic, and residual chronology are still
  colliding

#### 4. The strongest live Sprint 67 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- maintained truth/build surfaces:
  - `README.md` = `1020`
  - `docs/maintainer_guide.md` = `548`
  - `CMakeLists.txt` = `413`
  - `Makefile` = `897`
- public coordination headers likely to matter if ownership boundaries move:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_reorder.h` = `186`
  - `include/sparse_cholesky.h` = `232`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest graph/reorder implementation hotspots:
  - `src/sparse_graph.c` = `801`
  - `src/sparse_graph_coarsen.c` = `641`
  - `src/sparse_graph_bisect.c` = `528`
  - `src/sparse_graph_refine.c` = `629`
  - `src/sparse_graph_separator.c` = `297`
  - `src/sparse_reorder_nd.c` = `743`
  - `src/sparse_reorder_amd_qg.c` = `611`
- strongest CSC/iterative residual hotspots:
  - `src/sparse_analysis.c` = `1020`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2130`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
- strongest proof/adoption/regression support surfaces:
  - `tests/test_integration.c` = `2367`
  - `tests/test_graph.c` = `2900`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_chol_csc.c` = `4716`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_eigs.c` = `1522`
  - `examples/example_analysis.c` = `210`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`

Interpretation:

- the strongest remaining Epic 6 maintainability pressure is not generic across
  the whole repo
- it is concentrated in a smaller set of orchestration-heavy files and the
  proof surfaces that will have to move with them

#### 5. The Day 1 non-goal fence is now explicit before deeper audit begins

Sprint 67 Day 1 confirms the following non-goals:

- no fake maintainability wins that blur real ownership
- no broad feature work disguised as decomposition
- no reopening packaging/platform/build-surface churn unless a touched seam
  truly requires it
- no weakening of the reviewed truthfulness contract
- no broad style-only cleanup wave disconnected from actual ownership seams
- no chronology scrub that removes durable explanations just to erase sprint
  history references

### Day 1 Close

Sprint 67 now starts from one explicit maintainability implementation baseline:

- the Sprint 66 productization close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor is re-established locally at `53`
- the broad Epic 6 maintainability claim has already narrowed to hotspot
  audit, graph/reorder decomposition, CSC/iterative residual decomposition,
  chronology cleanup, and build/regression alignment
- the next step is to rank those live hotspot seams precisely before writing
  the bounded Day 2 validation and Day 3 audit follow-through

## Day 2 - Validation Baseline & Hotspot/Proof Rerun Recheck

### Goal

Reconfirm the reviewed baseline and the targeted rerun set that Sprint 67
decomposition work must preserve before any implementation work lands.

### Actions

1. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
2. Re-read the reviewed baseline wrapper surface:
   - `make -n quality-review-full`
3. Reconfirmed the authoritative validation split for:
   - bounded `*.c` / `*.h` days
   - substantial decomposition/build-alignment days
   - docs-only days
4. Rechecked build-tree availability of the most relevant Sprint 67 proof and
   regression surfaces:
   - graph/reorder proofs
   - CSC proofs
   - iterative/eigensolver proofs
   - representative examples
   - maintained benchmark/reporting surfaces
5. Reconfirmed the strongest likely Sprint 67 touched-surface classes from the
   live branch state after the Day 1 baseline.

### Findings

#### 1. The strongest reviewed baseline is unchanged at Sprint 67 start

The strongest local reviewed baseline is still:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 67 starts from the same reviewed truthfulness baseline as the Sprint
  66 close
- decomposition work does not get a weaker local validation contract just
  because the main sprint topic is maintainability rather than new end-user
  behavior

#### 2. The Day 2 authority split is now explicit

The authoritative split for Sprint 67 is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial decomposition, ownership-boundary, or
  build/regression-alignment work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

Interpretation:

- Sprint 67 should treat ownership-boundary changes as contract-sensitive work,
  not as cheap refactors
- the stronger reviewed baseline remains the default for any change that could
  distort build, proof, or ownership truthfulness

#### 3. The high-signal Sprint 67 rerun set is now fixed around the actual decomposition-risk surface

The high-signal rerun set at Sprint 67 start is:

- cross-family and orchestration proof surfaces:
  - `./build/test_integration`
- graph/reorder family proofs:
  - `./build/test_graph`
  - `./build/test_reorder_nd`
- CSC direct-family proofs:
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- iterative and eigensolver residual proofs:
  - `./build/test_iterative`
  - `./build/test_eigs`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
- maintained benchmark/reporting surfaces likely to matter in alignment work:
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

All of those surfaces were present in the current `build/` tree at Day 2.

Interpretation:

- the Sprint 67 rerun set is anchored to the actual decomposition-risk surface
  rather than to every executable in the repo
- maintained benchmark/reporting surfaces remain part of the live Sprint 67
  validation story because later build/regression alignment can still touch
  them indirectly

#### 4. The strongest likely Sprint 67 touch surfaces remain ownership-heavy implementation and proof surfaces, not packaging/productization files

The highest-signal likely Sprint 67 touch surfaces at Day 2 remain:

- implementation hotspots:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
- proof/support surfaces:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_integration.c`
- likely coordination headers only if the design proves they must move:
  - `include/sparse_analysis.h`
  - `include/sparse_reorder.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

Interpretation:

- Sprint 67 still starts from a maintainability and ownership surface, not a
  productization or platform surface
- the heaviest likely touched source and proof seams are already explicit
  before the deeper hotspot audit begins

### Day 2 Close

Sprint 67 now has:

- one explicit reviewed validation contract for decomposition work
- one fixed rerun set centered on the actual graph/reorder, CSC, and
  iterative/eigs proof surface
- one clear Day 3 starting point for the residual hotspot audit
