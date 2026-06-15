# Sprint 70 Working Notes

## Day 1 - Scope Audit & Baseline Setup

### Goal

Freeze the Sprint 70 starting point before implementation work begins by
reconfirming the inherited Epic 6 closeout contract, the preserved reviewed
baseline, the strongest live product-model and capability hotspots, and the
most important public, implementation, proof, and project-level surfaces that
Epic 7 will touch next.

### Actions

1. Re-read the Sprint 70 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Epic 6 retrospective, the Epic 7 review, and the Epic 7 todo.
2. Re-read the landed Sprint 70 plan and fixed the bounded workstreams that
   the sprint should actually carry:
   - baseline recheck
   - product-model gap inventory
   - capability ceiling audit
   - public-surface and proof-surface audit
   - validation and platform contract freeze
   - closeout and handoff
3. Reconfirmed the strongest reviewed baseline surfaces:
   - `make quality-review-full`
   - `make -n quality-review-full`
4. Re-materialized the reviewed CMake parity tree and rechecked the parity
   anchor:
   - `make quality-review-cmake-compile`
   - `ctest -N --test-dir build/quality-review-cmake`
5. Measured the strongest likely Sprint 70 touch surfaces directly from the
   live tree across:
   - maintained public product and policy surfaces
   - core product-model and numeric-path implementation seams
   - strongest proof/benchmark/reporting support surfaces
   - project-level Epic 7 review and planning surfaces

### Findings

#### 1. Sprint 70 starts from the Epic 6 validated close and the Epic 7 review package, not from a blank-slate architecture exercise

Epic 6 already closed the highest-value foundational gaps that previously kept
the repo from reading like a real product:

- typed high-value analysis and reorder options
- clearer direct repeated-run lifecycle ownership
- canonical benchmark governance and threshold-free reporting
- truthful packaging/install/platform wording
- stronger large-source, giant-test, and assurance follow-through
- a final coherent public product story across README, tutorial, examples,
  benchmarks, tests, and maintainer surfaces

That means Sprint 70 is not inventing Epic 7 from scratch. It is fixing the
starting contract for the work that the Epic 7 review already identified:

- product-model convergence
- capability-surface expansion
- configuration modernization phase 2
- backend/performance maturity phase 2
- packaging/platform convergence phase 2
- residual source/test/public-surface cleanup

Interpretation:

- Sprint 70 is a bounded baseline and architecture-contract sprint, not a
  disguised implementation sprint
- the Epic 7 review and todo are already real inputs, so Day 1 should sharpen
  them against the live tree rather than restate them abstractly

#### 2. The strongest local reviewed baseline remains the authoritative Sprint 70 starting point

The maintained Day 1 truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The reviewed CMake parity tree was re-materialized locally through:

- `make quality-review-cmake-compile`

Interpretation:

- Sprint 70 inherits the exact same reviewed baseline story as the Epic 6
  close
- even though Day 1 is docs-only planning work, it still starts from the
  strongest reviewed truth surface rather than a lighter planning-only proxy

#### 3. The highest-value Epic 7 pressure is now structural, not foundational

The live repo still reads as a serious engineering-grade sparse linear algebra
library. The remaining pressure is concentrated in ceilings rather than missing
core solver families or missing quality discipline.

The strongest current Day 1 Epic 7 pressure reduces to:

1. core product-model convergence
2. capability-surface expansion on the biggest ceilings
3. residual configuration modernization
4. backend/performance architecture deepening
5. packaging/platform/install convergence beyond the current asymmetric truth
   surface
6. residual large-source, giant-test, and permanent-surface cleanup

Interpretation:

- Sprint 70 should not pretend the repo still needs broad “stabilization”
- the real Day 1 task is freezing the exact target and non-goal fence for
  those structural ceilings before later sprints start landing code

#### 4. The strongest live Sprint 70 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- maintained public product and policy surfaces:
  - `README.md` = `1034`
  - `docs/maintainer_guide.md` = `578`
  - `INSTALL.md` = `237`
  - `include/sparse_matrix.h` = `583`
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `232`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
  - `include/sparse_types.h` = `233`
- core product-model and numeric-path implementation seams:
  - `src/sparse_matrix.c` = `1052`
  - `src/sparse_chol_csc.c` = `1536`
  - `src/sparse_ldlt_csc.c` = `2130`
  - `src/sparse_lu_csr.c` = `1665`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
  - `src/sparse_graph.c` = `821`
  - `src/sparse_reorder_nd.c` = `739`
  - `src/sparse_graph_coarsen.c` = `641`
  - `src/sparse_graph_refine.c` = `629`
  - `src/sparse_reorder_amd_qg.c` = `611`
- strongest proof/benchmark/reporting support surfaces:
  - `tests/test_integration.c` = `2411`
  - `tests/test_chol_csc.c` = `4608`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_graph.c` = `2900`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_eigs.c` = `1522`
  - `tests/test_qr.c` = `3197`
  - `tests/test_svd.c` = `2766`
  - `tests/test_fuzz.c` = `651`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`
- project-level Epic 7 planning and review surfaces:
  - `docs/planning/EPIC_7/PROJECT_PLAN.md` = `356`
  - `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md` = `542`

Interpretation:

- the strongest remaining Epic 7 pressure is concentrated in a finite set of
  permanent public, implementation, proof, and planning surfaces
- Sprint 70 should start by reranking those seams, not by widening into new
  solver or packaging implementation work

#### 5. The Day 1 non-goal fence is now explicit before deeper audit begins

Sprint 70 Day 1 confirms the following non-goals:

- no fake “state-of-the-art” claims not grounded in the live repo
- no broad implementation work disguised as baseline or audit setup
- no reopening solved Epic 6 seams unless a touched Sprint 70 audit seam truly
  forces it
- no weakening of the reviewed truthfulness contract
- no broad cleanup wave disconnected from real product-model, capability,
  performance, or platform ceilings
- no capability or backend promises before the exact first modernization fences
  are frozen

### Day 1 Close

Sprint 70 now starts from one explicit Epic 7 baseline:

- the Epic 6 validated close and the Epic 7 review package are both active and
  unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor has been re-established locally at `53`
- the broad Epic 7 goal has already narrowed to product-model convergence,
  capability expansion, configuration modernization, backend/performance
  maturity, packaging/platform convergence, and residual source/test/surface
  cleanup
- the next step is to recheck the exact validation and rerun contract before
  the deeper product-model and capability audits begin

## Day 2 - Validation Baseline & Rerun Recheck

### Goal

Reconfirm the reviewed baseline and the targeted rerun set that Sprint 70
planning and later Epic 7 implementation sprints must preserve before deeper
audit and architecture-contract work continues.

### Actions

1. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
2. Re-read the reviewed baseline wrapper surface:
   - `make -n quality-review-full`
3. Reconfirmed the authoritative validation split for:
   - bounded `*.c` / `*.h` days
   - substantial architecture, capability, benchmark, or platform work
   - docs-only days
4. Rechecked the current availability of the most relevant Sprint 70 proof and
   regression surfaces across:
   - the reviewed CMake tree
   - maintained benchmark binaries in the root `build/` tree
   - install/package regression scripts in `tests/`
5. Reconfirmed the strongest likely Sprint 70 touched-surface classes from the
   live branch state after the Day 1 baseline.

### Findings

#### 1. The strongest reviewed baseline is unchanged at Sprint 70 start

Sprint 70 still starts from:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 70 inherits the exact same reviewed-baseline authority split as the
  Epic 6 close
- even though the sprint opens with docs-only planning work, later Epic 7
  implementation sprints should still default back to the same reviewed truth
  surfaces rather than inventing a lighter local rule set

#### 2. The validation split is now explicit before any architecture or implementation movement

The validation contract for Sprint 70 is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial architecture, capability,
  benchmark-governance, or platform work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

Interpretation:

- later Epic 7 work keeps the same authority split the repo already maintains
- Sprint 70 is not allowed to invent a planning-specific validation story that
  would be weaker than the live library contract

#### 3. The high-signal Sprint 70 rerun set is now fixed around the actual Epic 7 risk surface

The targeted Sprint 70 rerun set present in the reviewed CMake tree is:

- cross-family/orchestration and direct-family proof owners:
  - `build/quality-review-cmake/test_integration`
  - `build/quality-review-cmake/test_chol_csc`
  - `build/quality-review-cmake/test_ldlt_csc`
  - `build/quality-review-cmake/test_reorder_nd`
- assurance and broader numerical proof support:
  - `build/quality-review-cmake/test_fuzz`
  - `build/quality-review-cmake/test_framework_optin`
  - `build/quality-review-cmake/test_iterative`
  - `build/quality-review-cmake/test_eigs`
  - `build/quality-review-cmake/test_graph`
  - `build/quality-review-cmake/test_qr`
  - `build/quality-review-cmake/test_svd`
- representative examples:
  - `build/quality-review-cmake/example_analysis`
  - `build/quality-review-cmake/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`
  - and the same maintained benchmark binaries are also present in
    `build/quality-review-cmake/`
- maintained install/package proof scripts:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

Interpretation:

- the reviewed CMake tree is currently the authoritative local binary surface
  for the key proof-owner tests and representative examples
- the root `build/` tree is still carrying the maintained benchmark binaries
  and remains relevant for benchmark-governance and reporting follow-through
- install/package proof remains script-owned rather than binary-owned

#### 4. Sprint 70’s likely touched-surface class is already narrower than the full reviewed suite

Day 2 confirms the most likely Sprint 70 touched lane is concentrated in:

- maintained public product and policy surfaces
- product-model, capability, and configuration audit seams
- proof/adoption/reporting surfaces only where later architecture or
  contradiction analysis truly points to them
- project-level Epic 7 planning and review surfaces

Interpretation:

- Sprint 70 should stay bounded to baseline, audit, and contract work rather
  than widening into generic repo churn
- the rerun set is intentionally broader than the likely touched surfaces
  because it is preserving later Epic 7 truthfulness, not predicting every
  Day 3-14 edit target

### Day 2 Close

Sprint 70 now has one explicit validation contract before deeper audits begin:

- strongest local reviewed baseline is still `make quality-review-full`
- reviewed CMake parity remains explicit at `53`
- bounded code-touching days must run `make format`, `make lint`, and
  `make test`
- substantial architecture, capability, benchmark, or platform work should
  default to `make quality-review-full`
- the high-signal Sprint 70 rerun set is fixed around the reviewed CMake proof
  tree, maintained benchmark binaries, and install/package regression scripts

## Day 3 - Product-Model Gap Inventory I

### Goal

Audit the strongest remaining linked-list/product-model and conversion-heavy
workflow seams in the live library so Epic 7 can target real ownership and
usability ceilings rather than broad “matrix model” rhetoric.

### Actions

1. Re-read the Sprint 70 Day 3 plan target and the Epic 7 review’s product-
   model finding.
2. Re-read the strongest public product-model surfaces:
   - `README.md`
   - `include/sparse_matrix.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_lu.h`
   - `include/sparse_ldlt.h`
   - `include/sparse_analysis.h`
3. Re-read the strongest implementation seams defining the current
   linked-list-versus-compressed split:
   - `src/sparse_matrix.c`
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_lu_csr.c`
4. Rechecked the highest-signal conversion and publication paths:
   - `chol_csc_from_sparse_with_analysis`
   - `chol_csc_writeback_to_sparse`
   - `ldlt_csc_from_sparse_with_analysis`
   - `ldlt_csc_writeback_to_ldlt`
   - `lu_csr_from_sparse`
   - `lu_csr_to_sparse`
5. Rechecked the strongest user-facing workflow friction signals:
   - copy-first one-shot direct examples and header guidance
   - identity-permutation preconditions
   - logical vs physical accessor split
   - same-pattern repeated-run lifecycle ownership

### Findings

#### 1. The strongest exact product-model seam is still the one-shot direct workflow centered on copied `SparseMatrix` mutation

The dominant public one-shot direct story is still:

- construct or load a `SparseMatrix`
- `sparse_copy()` it to preserve the original
- factor the copy in place
- solve through the factored copy
- manage permutations or reordered publication implicitly through the matrix

That pattern is explicit across the public surfaces:

- `README.md`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

This remains a real strength for straightforward usage:

- one object
- one familiar sparse matrix type
- low ceremony for small or occasional solves

But it is also the strongest current Epic 7 burden because it keeps the public
center of gravity on:

- matrix copying
- in-place factor mutation
- publication through the same general-purpose matrix container
- implicit lifecycle knowledge around when the original matrix view is still
  valid

Interpretation:

- the largest exact seam is not “there is a linked-list type”
- it is that the default direct-solver product story still makes the mutable
  linked-list matrix the main object callers live inside even when the numeric
  backend is no longer really centered there

#### 2. The second strongest seam is the matrix API’s mixed logical/physical/permuted-state contract

`include/sparse_matrix.h` and `src/sparse_matrix.c` still expose a broad matrix
surface where many operations depend on whether the matrix is in:

- original identity-permutation state
- reordered state
- factored state
- logical-accessor view versus physical-storage view

The current API makes that explicit through:

- separate logical and physical accessors
- public permutation-array accessors
- `sparse_reset_perms(...)`
- many operation notes that warn “do not use on matrices with non-identity
  permutations”
- same-matrix factored-state markers and compatibility checks

This remains a real strength for power users because:

- the representation is inspectable
- permutation state is not hidden
- advanced callers can reason about exact storage/layout transitions

But it is also a product-model cost because a large part of the generic matrix
API is conditionally safe or meaningful depending on matrix state.

Interpretation:

- the next strongest product-model burden is not raw file size alone
- it is that `SparseMatrix` still tries to be:
  - mutable construction container
  - generic arithmetic container
  - reordered/factored state carrier
  - permutation owner
  - interoperability shell

#### 3. The third strongest seam is compressed backend work that still converts out of and publishes back into `SparseMatrix`

The highest-value fast numeric paths are now clearly compressed working-format
paths:

- CSC Cholesky
- CSC LDL^T
- CSR LU

But the public product story still routes through repeated translation layers:

- `SparseMatrix` → CSC/CSR conversion
- compressed factor/elimination/solve
- writeback or publication back into a linked-list-facing result model

The clearest exact examples are:

- `lu_csr_from_sparse(...)` and `lu_csr_to_sparse(...)`
- `chol_csc_from_sparse_with_analysis(...)`
- `chol_csc_writeback_to_sparse(...)`
- `ldlt_csc_from_sparse_with_analysis(...)`
- `ldlt_csc_writeback_to_ldlt(...)`

The fast paths are real and worthwhile, but the public product center is still
not a compressed-first workflow model.

Interpretation:

- this is the strongest current performance-facing product-model burden
- the compressed backends no longer look experimental, but they still read as
  implementation-owned accelerators living behind a linked-list-first public
  center

#### 4. The shared repeated-run direct lifecycle is a real convergence step, but it is still a parallel surface rather than the single dominant product model

The shared direct lifecycle in `include/sparse_analysis.h` is already a major
improvement:

- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

It owns:

- symbolic/permutation reuse
- family-agnostic same-pattern repeated-run story
- clearer factor/workspace separation than the one-shot direct APIs

But the Day 3 reread still shows a split product story:

- one-shot family headers remain the default front door
- the shared lifecycle is the advanced repeated-run path
- some family-local factor objects still coexist beside shared
  `sparse_factors_t`
- the shared path still assumes callers understand identity-permutation and
  same-pattern preconditions at a fairly detailed level

Interpretation:

- the repeated-run lifecycle is not a weakness
- the weakness is that the library still presents two strong product models in
  parallel rather than one clearly dominant long-term ownership story

#### 5. The strongest Day 3 ranking now reduces the broad product-model problem to four concrete seams

The current ranked Day 3 product-model seams are:

1. copy-first, in-place one-shot direct workflow centered on `SparseMatrix`
2. mixed logical/physical/permuted-state semantics across the generic matrix
   API
3. compressed backend conversion/writeback ownership split
4. shared repeated-run lifecycle versus family-local one-shot and owned-factor
   parallel surfaces

Lower but still real support context:

- compile-time tuning and threshold knobs documented on public matrix/direct
  surfaces
- public permutation-array accessors that expose how much state still lives on
  the matrix object itself
- interop and Matrix Market flow that still reinforce `SparseMatrix` as the
  universal public shell

### Day 3 Close

Sprint 70 now has one explicit first product-model hotspot map:

- the broad Epic 7 product-model concern is reduced to four concrete seams
- the strongest exact target is the copied-matrix, in-place one-shot direct
  workflow
- the second strongest target is the mixed logical/physical/permuted-state
  contract on the generic matrix API
- the third strongest target is the compressed backend conversion/writeback
  ownership split
- the next step is to rerank those seams against user cost, performance cost,
  compatibility burden, and proof burden before fixing the first true Epic 7
  product-model boundary
