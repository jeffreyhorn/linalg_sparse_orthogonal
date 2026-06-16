# Sprint 72 Working Notes

## Day 1 - Baseline Setup

### Goal

Turn the Sprint 72 project-plan scope plus the Sprint 70-71 handoff into one
bounded first-phase product-model convergence sprint, with the strongest
likely touch surfaces and non-goal fence fixed before deeper audit begins.

### Actions

1. Re-read the Sprint 72 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`.
2. Re-read the Sprint 71 retrospective and Sprint 71 closeout artifact.
3. Re-read the Sprint 72 plan and confirm the intended day-by-day workstream
   order.
4. Recheck the strongest local reviewed baseline wrapper shape with
   `make -n quality-review-full`.
5. Recheck the reviewed CMake parity anchor with
   `ctest -N --test-dir build/quality-review-cmake`.
6. Re-measure the strongest likely Sprint 72 touch surfaces with raw Day 1
   `wc -l` counts from the live tree.

### Findings

#### 1. Sprint 72 starts from a real implementation-facing queue, not from another planning reset

Sprint 71 already cleared the strongest public/reference drag out of the way.
That means Sprint 72 can start directly from the strongest next Epic 7 queue:

- product-model convergence from the public direct-workflow seam
- not another public-surface cleanup wave
- not capability widening
- not packaging/platform contract churn

Interpretation:

- the Sprint 72 starting point is implementation-facing
- but it still needs to stay bounded by the Sprint 70 architecture contract

#### 2. The strongest local reviewed baseline is still `make quality-review-full`

The Day 1 reread of `make -n quality-review-full` confirms that the strongest
local reviewed baseline still reads as:

- reviewed Makefile path:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`
- reviewed CMake parity path:
  - configure/build
  - `ctest -N`
  - full `ctest`

That remains the right strongest baseline for substantial Sprint 72 ownership
work.

#### 3. Reviewed CMake parity remains explicit and measurable

The Day 1 live parity anchor remains:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 72 starts from the same reviewed-truth anchor carried through the
  late Epic 6 and early Epic 7 sprints
- Day 2 can build the rerun set from a stable live reviewed baseline

#### 4. The highest-value Sprint 72 pressure is now clearly narrowed

The Sprint 72 work is now explicitly narrowed to:

- product-model surface audit
- ownership convergence design
- direct-workflow hardening
- compressed-path ownership cleanup
- public contract/example follow-through
- regression expansion
- validation and closeout

This excludes several tempting but incorrect widenings:

- no broad `SparseMatrix` rewrite
- no type/capability widening disguised as ownership work
- no generic abstraction layer campaign
- no platform/install/package reinterpretation

#### 5. The strongest likely Sprint 72 touch surfaces are now explicit from the live tree

Maintained public/product surfaces:

- `README.md` = `1037`
- `docs/maintainer_guide.md` = `578`
- `INSTALL.md` = `237`
- `include/sparse_matrix.h` = `583`
- `include/sparse_analysis.h` = `498`
- `include/sparse_iterative.h` = `765`
- `include/sparse_eigs.h` = `650`

Strongest product-model / numeric-path seams:

- `src/sparse_matrix.c` = `1052`
- `src/sparse_ldlt_csc.c` = `2130`
- `src/sparse_iterative.c` = `1985`
- `src/sparse_lu_csr.c` = `1665`
- `src/sparse_chol_csc.c` = `1536`
- `src/sparse_qr.c` = `1563`
- `src/sparse_eigs.c` = `1534`

Direct-workflow public-boundary support surfaces:

- `include/sparse_lu.h` = `362`
- `include/sparse_cholesky.h` = `215`
- `include/sparse_ldlt.h` = `334`

Strongest proof/adoption surfaces:

- `tests/test_chol_csc.c` = `4608`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_qr.c` = `3197`
- `tests/test_graph.c` = `2900`
- `tests/test_iterative.c` = `2802`
- `tests/test_svd.c` = `2766`
- `tests/test_integration.c` = `2411`
- `tests/test_sparse_matrix.c` = `1054`
- `examples/example_analysis.c` = `210`
- `examples/example_basic_solve.c` = `110`

Interpretation:

- the strongest direct-workflow and ownership pressure is still concentrated in
  `SparseMatrix` plus the CSC/CSR-backed direct paths
- the strongest proof cost is still concentrated in the existing high-value
  test owners rather than in new proof surfaces

### Day 1 Exit State

Sprint 72 Day 1 closes with one stable starting package:

1. the Sprint 72 implementation queue is fixed from the Sprint 70-71 handoff
2. the strongest reviewed baseline remains `make quality-review-full`
3. the reviewed CMake parity anchor remains `53`
4. the strongest likely touch surfaces are explicit from the live tree
5. the non-goal fence is fixed before deeper audit begins

That gives Day 2 one exact job:

- recheck the implementation-day validation contract and the highest-signal
  rerun surfaces Sprint 72 must preserve before ownership work starts

## Day 2 - Validation Baseline & Rerun Recheck

### Goal

Reconfirm the Sprint 72 implementation-day validation contract and fix the
highest-signal rerun set before any ownership convergence work lands.

### Actions

1. Re-read the Day 2 scope in `docs/planning/EPIC_7/SPRINT_72/PLAN.md`.
2. Recheck the reviewed CMake parity anchor with
   `ctest -N --test-dir build/quality-review-cmake`.
3. Reconfirm the strongest local reviewed baseline reading from the Day 1
   `make -n quality-review-full` wrapper recheck.
4. Reconfirm the live proof-surface split across:
   - reviewed CMake tree
   - maintained root benchmark binaries
   - maintained install/package proof scripts
5. Fix the authoritative Sprint 72 rerun set and validation split in writing.

### Findings

#### 1. The strongest local reviewed baseline is still `make quality-review-full`

The Day 2 reread confirms Sprint 72 still starts from the same strongest local
reviewed baseline carried through late Epic 6 and early Epic 7:

- `make quality-review-full`

That still means:

- bounded `*.c` / `*.h` landing days require:
  - `make format`
  - `make lint`
  - `make test`
- substantial architecture or ownership-boundary batches should escalate to:
  - `make quality-review-full`
- docs-only audit/design/review days use targeted sanity checks only

#### 2. Reviewed CMake parity remains the main truthfulness anchor

The Day 2 live parity anchor remains:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 72 still begins from a stable reviewed-truth surface
- the implementation sprint does not need a new validation reading

#### 3. The proof-surface split is now explicit for Sprint 72

The Day 2 recheck confirms the live local proof split reads as:

- reviewed CMake tree:
  - key proof-owner tests
  - representative examples
- root `build/` tree:
  - maintained benchmark binaries
- scripts:
  - maintained install/package proof

Specifically confirmed present:

- reviewed CMake proof owners and representative examples:
  - `build/quality-review-cmake/test_sparse_matrix`
  - `build/quality-review-cmake/test_integration`
  - `build/quality-review-cmake/test_chol_csc`
  - `build/quality-review-cmake/test_ldlt_csc`
  - `build/quality-review-cmake/test_iterative`
  - `build/quality-review-cmake/test_eigs`
  - `build/quality-review-cmake/example_analysis`
  - `build/quality-review-cmake/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`
- maintained install/package proof scripts:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

#### 4. The highest-signal Sprint 72 rerun set is now fixed

The strongest likely Sprint 72 rerun set is now explicit:

- direct-workflow and ownership-boundary proof:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
- direct CSC-family proof owners:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
- likely support family proofs:
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_eigs`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- maintained install/package proof scripts:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

Interpretation:

- Sprint 72 now has a precise rerun set tied to the actual ownership seam
- the sprint does not need to improvise its proof surface later

### Day 2 Exit State

Sprint 72 Day 2 closes with one explicit validation contract:

1. strongest local reviewed baseline remains `make quality-review-full`
2. reviewed CMake parity remains the main truthfulness anchor at `53`
3. the reviewed CMake versus root benchmark versus script-owned proof split is
   explicit
4. the highest-signal Sprint 72 rerun set is fixed before ownership work starts

That gives Day 3 one exact job:

- audit the live product-model surfaces and reduce the broad ownership problem
  to a ranked contradiction map
