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

## Day 4 - Product-Model Gap Inventory II & First Boundary

### Goal

Rerank the Day 3 product-model seams against user cost, performance cost,
compatibility burden, and proof burden, then freeze one exact first Epic 7
product-model boundary that later implementation sprints should respect.

### Actions

1. Re-read the Sprint 70 Day 4 plan target and the Day 3 product-model audit.
2. Re-ranked the Day 3 seam set against:
   - user-facing workflow importance
   - numeric-path performance cost
   - compatibility burden
   - proof burden
3. Re-checked the strongest public direct-workflow ownership surfaces:
   - `README.md`
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
4. Re-checked the strongest support-only implementation surfaces behind that
   ownership boundary:
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_lu_csr.c`
5. Fixed the first Epic 7 product-model boundary and the explicit non-goal
   fence in writing.

### Findings

#### 1. The strongest first Epic 7 product-model target is the public direct-workflow ownership boundary, not a generic matrix-API rewrite

The Day 3 ranking holds, but the rerank clarifies the first implementation
boundary:

- the copy-first, in-place one-shot direct workflow is still the strongest
  exact seam
- however, the first realistic Epic 7 landing should start at the public
  workflow ownership boundary rather than at a broad rewrite of the generic
  matrix API

Why this is the correct first target:

- highest user-facing payoff
- strongest coherence gain across one-shot and repeated-run workflows
- lower proof burden than rewriting broad matrix arithmetic or accessor
  semantics first
- creates the right architecture center for later compressed-path and matrix-
  state work instead of reopening everything at once

Interpretation:

- Epic 7 should first converge the direct-solver product story around clearer
  workflow ownership and factor/workspace boundaries
- it should not begin by trying to redesign every logical/physical/permutation
  surface on `SparseMatrix`

#### 2. The exact first product-model boundary is now fixed to the shared direct-workflow ownership lane

The exact first Epic 7 product-model boundary is now:

- public workflow and ownership surfaces:
  - `README.md`
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

Likely support only if needed:

- `examples/example_analysis.c`
- `examples/example_basic_solve.c`
- `tests/test_integration.c`

This boundary is intentionally centered on:

- one-shot versus repeated-run ownership wording
- factor/workspace ownership clarity
- copy discipline and preserved-original discipline
- same-pattern reuse interpretation

It is intentionally not yet centered on:

- generic matrix arithmetic redesign
- permutation-accessor redesign
- compressed storage publication rewrites

#### 3. The mixed logical/physical/permuted-state generic matrix API is now support context, not the first batch center

The Day 3 second-ranked seam remains real:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`

But it is not the correct first batch center because:

- it is broader than the first direct-workflow convergence lane
- it has a higher proof burden across arithmetic, accessors, permutation
  exposure, and matrix-state transitions
- it risks turning Sprint 72 into a generic matrix redesign rather than a
  bounded product-model convergence sprint

Interpretation:

- this seam stays important
- it now reads as the strongest second-phase or support seam rather than the
  correct first landing

#### 4. Compressed backend conversion/writeback ownership is also support context, not the first batch center

The CSC/CSR conversion and writeback seam remains the strongest
performance-facing product-model burden:

- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`

But it should stay outside the first Epic 7 product-model landing because:

- the public workflow ownership story should be clarified before deeper
  compressed/publication surgery
- these paths are lower-ceremony for internal implementation than for public
  callers today
- moving them first would blur product-model convergence into backend rewrite

Interpretation:

- compressed path ownership remains a real Epic 7 lane
- it now reads as a later first-phase or second-phase target after the public
  workflow boundary is stabilized

#### 5. The strongest Day 4 non-goal fence is now explicit

Sprint 70 Day 4 confirms the following non-goals for the first Epic 7
product-model landing:

- no broad rewrite of every `SparseMatrix` entry point
- no generic matrix arithmetic redesign in the first batch
- no fake abstraction layer added only for aesthetics
- no capability-surface widening disguised as product-model cleanup
- no broad CSC/CSR publication rewrite before the public direct-workflow
  boundary is clarified
- no proliferation of new parallel ownership surfaces without retiring or
  shrinking an older one

### Day 4 Close

Sprint 70 now has one explicit first product-model boundary:

- first target:
  - public direct-workflow ownership convergence
- support only if needed:
  - example adoption surfaces
  - cross-family direct proof in `tests/test_integration.c`
- later/deferred:
  - broad generic matrix-API redesign
  - compressed backend conversion/writeback convergence
  - capability-adjacent type or storage widening

That gives later Sprint 70 and Sprint 72 planning one exact job:

- converge the direct-solver product story first, then widen only where the
  bounded ownership changes prove it is necessary

## Day 5 - Capability Ceiling Audit I

### Goal

Reduce the broad Epic 7 capability-expansion question to one ranked live
ceiling map so later modernization work starts from the strongest product
constraints instead of a vague "more features" backlog.

### Inputs Rechecked

I re-read the live public capability surfaces and the strongest existing Epic
7 review claims:

- `include/sparse_types.h`
- `README.md`
- `include/sparse_eigs.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md`

### Findings

#### 1. The strongest current capability ceiling is still the 32-bit index model

The live repo still fixes the global sparse index type to:

- `typedef int32_t idx_t;`
- `#define IDX_MAX INT32_MAX`

This is the strongest capability ceiling because it hits all three layers at
once:

- public product limits:
  - matrix dimensions and nnz cap at roughly 2.1 billion
- implementation assumptions:
  - allocator, overflow, and workspace math are built around `idx_t`
- compatibility implications:
  - widening this is a public-header, ABI, and downstream rebuild event

Interpretation:

- this is not a narrow implementation detail
- it is the broadest capability-width limit in the current product line
- it is therefore the strongest first modernization candidate

#### 2. The second strongest ceiling is real-only scalar support

The repo still presents itself as:

- real-only
- double-precision only

This is visible across the live public API:

- solver inputs and outputs use `double *`
- eigensolver buffers use `double *`
- preconditioner and matrix-free callbacks use `double *`
- factor/result structs store `double` state directly

Why this ranks second:

- it excludes major sparse-library workloads directly:
  - complex Hermitian / unsymmetric problems
  - single-precision or mixed-precision workflows
  - integer-valued exact or symbolic-adjacent product lines
- it cuts across almost every public header, not just one subsystem
- the implementation burden is even broader than the index-width burden

Interpretation:

- real-only is a deeper eventual modernization lane than index width
- but it has higher surface area and proof burden than the 32-bit index ceiling
- it therefore ranks as the strongest second capability ceiling, not the first

#### 3. The third strongest ceiling is the symmetric-only sparse eigensolver surface

The live public eigensolver contract is still:

- `sparse_eigs_sym(...)`
- grow-m Lanczos
- thick-restart Lanczos
- explicit LOBPCG
- symmetric matrices only

This is a real state-of-the-art positioning limit because:

- the library has a credible sparse symmetric eigensolver story now
- but it still has no public unsymmetric sparse eigensolver story
- so the capability ceiling is product-facing, not merely internal

Why it ranks third instead of first:

- it is narrower than index width or scalar support
- it affects one important capability family rather than nearly every family
- the current symmetric story is already relatively strong within its lane

Interpretation:

- this is a meaningful Epic 7 capability target
- but it is a narrower modernization lane than the global width/type ceilings

#### 4. Public caveats, implementation assumptions, and compatibility implications are now separated

The Day 5 split is now explicit:

- public caveats:
  - 32-bit indices
  - real-only scalar support
  - symmetric-only eigensolver scope
- implementation assumptions:
  - widespread `idx_t` use in dimensions, nnz, permutations, and workspaces
  - widespread `double` storage in solver, iterative, and eigensolver contracts
  - symmetric-specific eigensolver naming, docs, and buffer/result semantics
- compatibility implications:
  - index-width widening is a public typedef and ABI event
  - scalar-type widening is a larger API and packaging event still
  - eigensolver-family widening is less ABI-heavy than scalar generalization,
    but it still changes the public product promise and proof burden

This is the most useful Day 5 clarification:

- Epic 7 capability work should not treat all ceilings as one kind of problem
- some are type-width/product-line ceilings
- some are algorithm-family ceilings
- some are broad package/ABI events even before implementation difficulty is
  considered

#### 5. The strongest Day 5 modernization shortlist is now explicit

Sprint 70 Day 5 now leaves one ranked shortlist for later capability design:

1. first capability modernization candidate:
   - 32-bit index ceiling
2. strongest second modernization candidate:
   - real-only scalar ceiling
3. narrower but still important capability candidate:
   - unsymmetric sparse eigensolver gap
4. later/deferred or support context:
   - broader precision-product expansion
   - broader algorithm-family wishlist not yet justified by the live public
     contract

### Day 5 Close

Sprint 70 now has one concrete capability-ceiling ranking instead of a generic
"more capability" plan:

- first:
  - 32-bit index-width ceiling
- second:
  - real-only scalar ceiling
- third:
  - symmetric-only eigensolver ceiling

That gives Day 6 one exact job:

- separate the first realistic Epic 7 capability modernization lane from the
  larger deferred ambitions without blurring width, scalar, and algorithm
  expansion into one fake batch

## Day 6 - Capability Ceiling Audit II & Modernization Fence

### Goal

Turn the Day 5 ranked capability ceilings into one exact first Epic 7
modernization fence so later implementation starts from a bounded capability
lane rather than from a broad "64-bit + complex + more algorithms" promise.

### Inputs Rechecked

I re-read the live Day 5 ranking and the Epic 7 capability-planning inputs:

- `docs/planning/EPIC_7/SPRINT_70/artifacts/day5-capability-ceiling-audit.md`
- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md`
- `docs/planning/EPIC_7/reviews/todo-codex-2026-06-15.md`
- `include/sparse_types.h`
- `README.md`
- `include/sparse_eigs.h`

### Findings

#### 1. The first realistic Epic 7 capability lane is index-width modernization, not broad type generalization

Re-ranking the Day 5 ceilings against:

- user value
- ecosystem impact
- implementation risk
- proof burden

leaves the same ordering, but with a sharper first lane:

1. first bounded modernization candidate:
   - index-width path
2. medium-term capability target:
   - scalar-surface broadening
3. later capability target:
   - unsymmetric sparse eigensolver expansion

Why index width stays first:

- it is the broadest current product ceiling
- it is easier to isolate into one explicit typedef/overflow/build contract
  than scalar-type generalization
- it gives the repo a real capability-modernization path without requiring a
  full product-line rewrite

Interpretation:

- the first Epic 7 capability lane should be "make width widening real"
- not "make every numeric surface type-generic"

#### 2. Real-only scalar support is the strongest second lane, but it is not the right first landing

The Day 5 scalar ranking still holds, but Day 6 makes the implementation
interpretation explicit:

- scalar breadth is a bigger market and product-line ceiling than the
  symmetric eigensolver gap
- but it has the highest surface-area burden across:
  - public headers
  - factor/result structs
  - callbacks
  - examples/tests/docs
  - packaging expectations

Interpretation:

- scalar-surface work should start as preparation and seam definition
- it should not be represented as the first shipped capability closure
- the first lane can make scalar generalization easier later, but should not
  pretend to deliver it outright

#### 3. Unsymmetric sparse eigensolver expansion is important, but it is now explicit later work

The eigensolver-family gap remains real:

- no public unsymmetric sparse eigensolver story

But Day 6 makes its place in the queue explicit:

- it is narrower than width and scalar breadth
- it depends on broader capability and product-model maturity first
- it carries a large proof and documentation burden for a narrower first Epic 7
  payoff

Interpretation:

- this is a credible later Epic 7 or next-epic target
- it should not compete with the first bounded modernization lane

#### 4. The strongest exact first modernization fence is now explicit

The first bounded Epic 7 capability fence should center on the index-width
contract:

- required first modernization center:
  - `include/sparse_types.h`
  - `README.md`
  - `docs/maintainer_guide.md`
- likely support only if needed:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `INSTALL.md`
- proof and safety support only if the implementation batch demands it:
  - width-sensitive overflow and bounds tests
  - install/package sanity where the public typedef or build reading changes

Why this is the right fence:

- `idx_t` is the public width center
- README and maintainer policy already own the current documented width limit
- the first lane can define one real modernization path without immediately
  widening every numeric algorithm family

Interpretation:

- first capability work should start by making the width contract real and
  bounded
- public numeric headers beyond `sparse_types.h` should move only where the
  width path truly forces them

#### 5. The strongest Day 6 non-goal fence is now explicit

Sprint 70 Day 6 confirms the following capability non-goals for the first Epic
7 modernization lane:

- no claim of full type-generic conversion in one sprint
- no fake "complex-ready" story without end-to-end proof
- no broad unsymmetric eigensolver wishlist disguised as a first capability
  batch
- no package/ABI claim wider than the actually landed width seam
- no simultaneous attempt to solve width, scalar breadth, and algorithm-family
  expansion in one lane

### Day 6 Close

Sprint 70 now has one exact capability modernization order:

1. first bounded lane:
   - index-width modernization path
2. strongest second lane:
   - scalar-surface preparation and broadening
3. later capability lane:
   - unsymmetric sparse eigensolver expansion

That gives later Sprint 70 and Sprint 74 planning one exact job:

- land the first real width-modernization seam first, then widen only where
  that bounded path proves broader scalar or algorithm-family work is justified

## Day 7 - Public-Surface and Proof-Surface Audit I

### Goal

Reduce the broad Epic 7 public-surface and proof-surface cleanup question to
one ranked contradiction map so later cleanup work starts from the strongest
remaining drift instead of from generic documentation or test churn.

### Inputs Rechecked

I re-read the strongest maintained public and proof surfaces:

- `README.md`
- `docs/tutorial.md`
- `INSTALL.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `include/sparse_cholesky.h`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`
- `tests/test_reorder_nd.c`
- `tests/test_fuzz.c`

I also rechecked chronology/policy density directly with targeted `rg`
searches across those surfaces.

### Findings

#### 1. The strongest remaining public-surface contradiction is still the overloaded front door in `README.md`

`README.md` is still the strongest public-surface cleanup target because it
continues to mix several roles at once:

- compact product front door
- workflow chooser
- maintained benchmark/reporting summary
- platform/install quality summary
- deep algorithm/performance history
- giant test inventory and maintainer routing notes

The file is now far more coherent than the pre-Epic-6 state, but it still
reads as the densest place where mature product guidance and delivery-history
accumulation compete with each other.

Interpretation:

- this is the strongest user-facing cleanup seam
- the problem is no longer missing ownership
- it is residual density and chronology overexposure

#### 2. The strongest header-side contradiction is `include/sparse_cholesky.h`

Among the likely public headers, `include/sparse_cholesky.h` still carries the
strongest mix of:

- API-local caveats that belong there
- older sprint-history and ABI-history explanation
- benchmark references and performance-reading guidance
- dispatch/telemetry interpretation that partially overlaps public docs and
  maintainer policy

This makes it the strongest current header-side candidate because it most
clearly risks reading broader or more archival than a mature reference header
should.

Interpretation:

- this is not a correctness problem
- it is the strongest public-reference cleanup seam
- it should rank ahead of broader header churn because it concentrates the
  most mixed ownership in one high-signal place

#### 3. The strongest support-surface drift is `INSTALL.md`, not the examples or benchmark READMEs

`INSTALL.md` still carries the strongest support-surface drift because:

- most of the operator contract is already truthful
- but the platform notes still preserve older sprint-history explanation more
  directly than the current mature install surface needs

By contrast:

- `examples/README.md` is already relatively clean and bounded
- `benchmarks/README.md` is still dense, but much of that density is
  legitimately benchmark-local rather than accidental drift
- `docs/maintainer_guide.md` is policy-dense by design and therefore does not
  rank as the strongest contradiction center

Interpretation:

- `INSTALL.md` is the strongest support-surface cleanup candidate
- examples and benchmarks now read more like bounded support surfaces than like
  the main contradiction centers

#### 4. The strongest proof-surface contradiction is `tests/test_reorder_nd.c`

`tests/test_reorder_nd.c` is now the strongest remaining proof-surface cleanup
candidate because it still carries the densest concentration of:

- sprint-day chronology
- historical tuning commentary
- preserved planning archaeology
- live proof mixed with large volumes of historical rationale

This is a stronger contradiction than simple file size alone:

- it makes current regression ownership harder to review quickly
- it blurs durable algorithm explanation with sprint-local history
- it is the clearest remaining giant-test surface where the repo still reads
  like an archive inside a permanent proof owner

Interpretation:

- this is the strongest later giant-test cleanup seam
- stronger than `tests/test_integration.c`
- and stronger than any current docs/support surface except the front door

#### 5. The second strongest proof-surface contradiction is `tests/test_chol_csc.c`

`tests/test_chol_csc.c` remains a major proof-surface burden because it mixes:

- family-local CSC correctness
- supernodal helper proof
- dispatch proof
- large fixture proof
- substantial retained sprint chronology

It ranks behind `tests/test_reorder_nd.c` because:

- its ownership split is somewhat clearer after recent Epic 6 and 7 cleanup
- it still contains a large amount of sprint-local history, but less policy
  ambiguity than the ND test surface

Interpretation:

- this remains one of the strongest future giant-test cleanup candidates
- but it is now clearly second to the ND proof surface

#### 6. The current lower-priority surfaces are now explicit

The Day 7 lower-priority or support-only surfaces are now clearer:

- `docs/tutorial.md`
  - still dense, but its current density is mostly teaching flow rather than
    contradiction
- `examples/README.md`
  - bounded and relatively clean after Epic 6/69 alignment
- `benchmarks/README.md`
  - dense, but much of that density is benchmark-local truth rather than
    drift
- `docs/maintainer_guide.md`
  - intentionally policy-heavy; it is a support authority, not the strongest
    contradiction center
- `tests/test_integration.c`
  - still important, but more coherent than the largest remaining giant-test
    hotspots
- `tests/test_fuzz.c`
  - bounded and readable relative to the larger proof owners

### Day 7 Close

Sprint 70 now has one concrete public/proof-surface contradiction ranking:

1. strongest public front-door contradiction:
   - `README.md`
2. strongest header/reference contradiction:
   - `include/sparse_cholesky.h`
3. strongest support-surface contradiction:
   - `INSTALL.md`
4. strongest proof-surface contradiction:
   - `tests/test_reorder_nd.c`
5. strongest second proof-surface contradiction:
   - `tests/test_chol_csc.c`

That gives Day 8 one exact job:

- separate the highest-value future cleanup lanes from lower-value surface
  churn and freeze the support-surface fence before Sprint 70 moves on to the
  validation/platform contract audit

## Day 8 - Public-Surface and Proof-Surface Audit II

### Goal

Turn the Day 7 contradiction ranking into one explicit Epic 7 cleanup fence so
later work starts from the highest-value public, header, and proof lanes
instead of reopening every documentation or giant-test surface at once.

### Inputs Rechecked

I re-read the live Day 7 ranking and the Epic 7 project-plan contract around
later cleanup lanes:

- `docs/planning/EPIC_7/SPRINT_70/artifacts/day7-public-and-proof-surface-audit.md`
- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `README.md`
- `INSTALL.md`
- `include/sparse_cholesky.h`
- `tests/test_reorder_nd.c`
- `tests/test_chol_csc.c`

### Findings

#### 1. The highest-value future public cleanup lane is the front-door and install pair, not a broad multi-doc rewrite

Re-ranking the Day 7 public/support surfaces against:

- user-facing product value
- contradiction density
- likelihood of durable simplification
- risk of reopening solved Epic 6/69 ownership work

leaves the strongest public cleanup lane as:

- `README.md`
- `INSTALL.md`

Why this pair comes first:

- README still carries the densest mixed public-story burden
- INSTALL is the strongest remaining operator/support drift surface
- both are high-value user-facing surfaces that can likely be simplified
  without reopening deeper policy or benchmark-governance work

Interpretation:

- the later public-product cleanup lane should be front-door plus install
- not a broad simultaneous rewrite of README/tutorial/examples/benchmarks

#### 2. The strongest later header/reference lane is narrow, centered on `include/sparse_cholesky.h`

The header/reference queue is now sharp:

- first header/reference candidate:
  - `include/sparse_cholesky.h`
- later support only if needed:
  - other public headers

Why this is the right lane:

- `include/sparse_cholesky.h` concentrates the strongest mixed reference burden
- widening immediately into more headers would blur one concrete cleanup seam
  into generic header churn
- the other headers do not currently show the same contradiction density

Interpretation:

- the header/reference lane should start narrow
- do not widen into all public headers just because one header is history-dense

#### 3. The strongest later proof-surface cleanup lane is ND-first, CSC-Cholesky second

The proof-surface queue is now explicit:

1. first giant-test cleanup candidate:
   - `tests/test_reorder_nd.c`
2. strongest second giant-test cleanup candidate:
   - `tests/test_chol_csc.c`
3. support only if later cleanup truly needs it:
   - `tests/test_integration.c`
   - `tests/test_fuzz.c`

Why this order holds:

- ND still carries the densest live chronology burden
- CSC Cholesky remains large and history-heavy, but with clearer family-local
  ownership
- integration and fuzz are important proof owners, but they are not the
  strongest current contradiction centers

Interpretation:

- later proof cleanup should stay targeted to the biggest archaeology-heavy
  owners first
- not widen into every proof-owner file

#### 4. The support-surface fence is now explicit

Sprint 70 Day 8 fixes the likely later support-surface fence as:

- likely future public-product cleanup:
  - `README.md`
  - `INSTALL.md`
- likely future header/reference cleanup:
  - `include/sparse_cholesky.h`
- likely future giant-test cleanup:
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
- support only if a later batch truly forces it:
  - `docs/tutorial.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `tests/test_integration.c`
  - `tests/test_fuzz.c`

This is the most useful Day 8 clarification:

- the remaining Epic 7 cleanup problem is no longer "all docs and all giant
  tests are too big"
- it is now a bounded set of likely lanes with explicit support-only surfaces

#### 5. The strongest Day 8 non-goal fence is now explicit

Sprint 70 Day 8 confirms the following surface-cleanup non-goals:

- no repo-wide rewrite of every public doc surface in one sprint
- no broad public-header cleanup campaign without a ranked header center
- no giant-test breakup wave disconnected from the strongest archaeology-heavy
  proof owners
- no reopening of benchmark-governance or maintainer-policy authority just to
  simplify wording elsewhere
- no mixing public cleanup, proof cleanup, and platform/install contract work
  into one fake "documentation pass"

### Day 8 Close

Sprint 70 now has one explicit future cleanup fence:

1. public-product cleanup:
   - `README.md`
   - `INSTALL.md`
2. header/reference cleanup:
   - `include/sparse_cholesky.h`
3. proof-surface cleanup:
   - `tests/test_reorder_nd.c`
   - `tests/test_chol_csc.c`
4. support-only surfaces unless later work proves otherwise:
   - tutorial / examples / benchmarks / maintainer guide
   - integration / fuzz

That gives Day 9 one exact job:

- move from surface cleanup ranking to the validation/platform contract audit
  without reopening the newly fixed public/proof cleanup fence

## Day 9 - Validation and Platform Contract I

### Goal

Reduce the broad Epic 7 validation/platform truthfulness concern to one ranked
contract-preservation map so later implementation work knows exactly which
reviewed, supplemental, staged, and excluded claims must not drift.

### Inputs Rechecked

I re-read the maintained validation, install, benchmark-report, and workflow
surfaces:

- `Makefile`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `scripts/bench_canonical_report.sh`

### Findings

#### 1. The strongest local reviewed truth surface is still the Linux-composed baseline, not any single narrower command

The live local truth hierarchy remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `make quality-review-cmake`
  - `ctest -N --test-dir build/quality-review-cmake`
- reviewed Makefile compile-quality anchor:
  - `make quality-review-compile`

This matters because later Epic 7 work can easily drift if it treats:

- `make test`
- one workflow leg
- one install script

as if they replace the stronger composed reviewed baseline.

Interpretation:

- `make quality-review-full` remains the strongest local reviewed claim
- narrower commands remain supporting proof, not replacement truth surfaces

#### 2. The strongest platform truthfulness sensitivity is still the reviewed-subset asymmetry

The current reviewed-platform contract is intentionally asymmetric:

- Linux:
  - strongest reviewed source of truth
  - reviewed Makefile compile-quality path
  - reviewed CMake parity path
  - enforced dead-code report/check path
- macOS:
  - reviewed Apple Clang lane
  - supplemental Homebrew GCC
  - supplemental static-first Make install/`pkg-config`
- Windows:
  - reviewed CMake subset only
  - explicit staged exclusions including `test_fuzz`
  - no reviewed Makefile or dead-code lane

This is the strongest current truthfulness sensitivity because:

- it is easy to over-read platform breadth from the repo's overall maturity
- the Windows surface is materially narrower than Linux
- macOS install/package confidence is intentionally supplemental, not a full
  reviewed parity claim

Interpretation:

- later Epic 7 work can move within this contract
- but must not collapse it into fake cross-platform parity claims

#### 3. The strongest install/package proof sensitivity is the split between local regression scripts and reviewed CI lanes

The install/package proof split is now explicit:

- local Unix-side proof:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- reviewed CI interpretation:
  - Linux does not claim a separate reviewed install-validation lane
  - macOS carries only a narrower supplemental Make install/`pkg-config` lane
  - Windows remains a reviewed CMake-consumer lane, not a separate reviewed
    install-validation lane

This is a high-risk drift point because install/package claims can sound
broader than the reviewed evidence if later work blurs:

- local regression scripts
- supplemental platform checks
- reviewed cross-platform proof

Interpretation:

- the scripts are real maintained proof surfaces
- but they do not by themselves widen the reviewed CI contract

#### 4. The strongest benchmark/reporting truthfulness sensitivity is still the threshold-free canonical-report reading

The canonical maintained benchmark/report surface remains intentionally narrow:

- `make bench-canonical-report`
- `scripts/bench_canonical_report.sh`
- four canonical proof benchmarks only

The key truthfulness rule is unchanged:

- this is an artifact-friendly snapshot surface
- not a pass/fail timing gate
- not portable performance proof by itself

This remains one of the strongest contract sensitivities because later Epic 7
performance or capability work could easily tempt broader timing claims than
the current benchmark-governance model justifies.

Interpretation:

- later work may extend reporting or observability
- but should preserve the current threshold-free reading unless a narrower
  justified threshold surface is explicitly introduced

#### 5. Dead-code staging remains the strongest operational quality residual

The dead-code contract remains:

- Linux:
  - enforced
- macOS:
  - staged pending fresh measurement
- Windows:
  - staged

with one operational limit still explicit:

- serialized dead-code execution because the workflow shares build/artifact
  paths

Interpretation:

- this is still a real residual
- but it is intentionally bounded and already truthfully documented
- later Epic 7 work should preserve that explicit staged reading unless it
  lands new reviewed evidence

#### 6. The current lower-risk contract surfaces are now clearer

The lower-risk or already-well-bounded contract surfaces are:

- the static-first release shape itself
  - already coherent across README, INSTALL, maintainer guide, and tests
- the canonical benchmark set definition
  - already bounded and explicit
- the local reviewed wrapper naming
  - already coherent across Makefile and docs

These still matter, but they are not the strongest current drift points
compared with:

- platform reviewed-subset asymmetry
- install/package proof ownership
- dead-code staging
- threshold-free benchmark-report interpretation

### Day 9 Close

Sprint 70 now has one ranked contract-preservation map:

1. strongest platform truthfulness sensitivity:
   - reviewed-subset asymmetry across Linux / macOS / Windows
2. strongest install/package proof sensitivity:
   - local regression scripts versus reviewed CI-lane interpretation
3. strongest benchmark/reporting sensitivity:
   - threshold-free canonical-report reading
4. strongest operational quality residual:
   - staged/serialized dead-code contract

That gives Day 10 one exact job:

- freeze the exact validation, package, benchmark, and platform non-goal fence
  for later Epic 7 implementation without widening any of the reviewed claims

## Day 10 - Validation and Platform Contract II

### Goal

Turn the Day 9 preservation map into one exact Epic 7 truthfulness fence so
later implementation sprints have an explicit non-goal and preservation
checklist before any product, capability, performance, or cleanup work lands.

### Inputs Rechecked

I re-read the live Day 9 audit and the maintained contract-authority surfaces:

- `docs/planning/EPIC_7/SPRINT_70/artifacts/day9-validation-and-platform-contract-audit.md`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `Makefile`

### Findings

#### 1. Maintained reviewed truth, supplemental proof, and deferred ambitions are now separated explicitly

The Day 10 split is now fixed:

- maintained reviewed truth:
  - `make quality-review-full`
  - Linux reviewed Makefile compile-quality path
  - Linux reviewed CMake parity path
  - Linux enforced dead-code report/check path
  - reviewed Windows CMake subset
  - reviewed macOS Apple Clang compile/CMake/wall-check/sanitize lane
- supplemental-but-truthful proof:
  - Linux direct runtime, TSan, coverage, `bench-fast`
  - macOS Homebrew GCC direct build/test/wall-check
  - macOS static-first Make install/`pkg-config`
  - local Unix-side install/package regression scripts
  - threshold-free canonical benchmark report artifacts
- explicitly deferred platform/release ambitions:
  - shared-library maturity claims
  - Windows reviewed Makefile parity
  - Windows reviewed dead-code
  - macOS reviewed dead-code
  - broad reviewed install-validation parity across every platform

This is the most useful Day 10 clarification:

- later Epic 7 work can add or improve proof
- but it should not silently upgrade supplemental or local proof into reviewed
  truth without new reviewed evidence

#### 2. The strongest contract non-goals are now explicit

Sprint 70 Day 10 confirms the following non-goals for later Epic 7 work:

- no fake shared-library or dynamic-ABI maturity claims
- no inflated Windows parity claims beyond the reviewed CMake subset
- no install/package story broader than:
  - local Unix-side proof scripts
  - supplemental macOS Make install/`pkg-config`
  - reviewed Windows CMake-consumer path
- no reinterpretation of `make bench-canonical-report` as a pass/fail or
  portable timing gate
- no benchmark or example wording that steals test-owned regression, oracle,
  or property guarantees
- no silent promotion of staged dead-code lanes into reviewed platform claims

Interpretation:

- the repo can evolve within these boundaries
- but it should not market future partial progress as if the whole reviewed
  contract had widened already

#### 3. The later-sprint preservation checklist is now explicit

Later Epic 7 implementation work should preserve the following checklist:

1. strongest local reviewed baseline still reads as:
   - `make quality-review-full`
2. reviewed CMake parity claims still route through:
   - `make quality-review-cmake`
   - `ctest -N --test-dir build/quality-review-cmake`
3. platform wording still preserves:
   - Linux strongest reviewed truth
   - macOS narrower reviewed lane plus supplemental install proof
   - Windows reviewed CMake subset only
4. install/package wording still preserves:
   - static-first release shape
   - local Unix-side script proof ownership
   - no broad reviewed install-validation claim
5. benchmark wording still preserves:
   - canonical maintained surface is bounded
   - `bench-canonical-report` is threshold-free artifact reporting
6. dead-code wording still preserves:
   - Linux enforced
   - macOS staged
   - Windows staged
   - serialized execution limit remains explicit

This is the main Day 10 deliverable:

- later sprints now have one exact contract-preservation checklist instead of
  a diffuse "be careful with docs/CI wording" warning

#### 4. The lower-value residuals should remain deferred unless a touched lane truly forces them

The following remain explicitly deferred rather than being pulled into early
Epic 7 implementation work:

- reviewed shared-library packaging claims
- Windows Makefile reviewed-wrapper parity
- Windows dead-code enablement
- macOS dead-code enablement
- broad reviewed install-validation convergence
- new timing-threshold governance beyond the currently justified surfaces

Interpretation:

- these may become future work
- but Sprint 70 closes with them outside the maintained reviewed truth fence

### Day 10 Close

Sprint 70 now has one explicit validation/platform/package truthfulness fence:

1. maintained reviewed truth:
   - strongest local reviewed baseline plus current reviewed platform lanes
2. supplemental-but-truthful proof:
   - direct runtime, extra compiler/sanitizer lanes, local install scripts,
     and threshold-free benchmark artifacts
3. explicitly deferred ambitions:
   - shared-library maturity, reviewed Windows Makefile/dead-code parity, and
     broad install-validation parity

That gives Day 11 one exact job:

- synthesize the product, capability, surface, and contract audits into one
  explicit Epic 7 state-of-the-art target and success rubric

## Day 11 - State-of-the-Art Target Synthesis

### Goal

Turn the Day 3-10 audit outputs into one explicit Epic 7 target model so the
later implementation sprints inherit a concrete success rubric instead of a
broad "modernize the library" aspiration.

### Inputs Rechecked

I re-read the live Sprint 70 audit and fence artifacts that now define the
current Epic 7 seam map:

- `docs/planning/EPIC_7/SPRINT_70/artifacts/day3-product-model-gap-inventory.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day4-first-product-model-boundary.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day5-capability-ceiling-audit.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day6-capability-modernization-fence.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day7-public-and-proof-surface-audit.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day8-support-surface-fence.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day9-validation-and-platform-contract-audit.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day10-validation-platform-fence.md`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

### Findings

#### 1. Epic 7 now has one explicit target model across six permanent axes

The Day 11 synthesis fixes the following target model:

- product model:
  - the public direct-solver story should read as a coherent owned-factor and
    repeated-run workflow with the generic mutable matrix shell no longer
    carrying the whole long-term product identity by default
- capability surface:
  - the repo should have a credible modernization path beyond the permanent
    `int32_t` ceiling, while preparing the scalar surface for later widening
    without pretending that full type-generic or complex support already
    exists
- performance maturity:
  - the fastest maintained numeric paths should read as first-class product
    lanes with explicit benchmark-governed proof, not as incidental backend
    accelerators behind a linked-list-first public story
- public/product documentation:
  - the front door, install story, and key headers should read like stable
    product/reference surfaces rather than a stitched-together sprint archive
- platform/release maturity:
  - the repo should preserve a static-first, truthful, explicitly asymmetric
    reviewed contract instead of claiming a broader release/platform story than
    the evidence supports
- assurance and proof architecture:
  - the strongest proof owners should remain explicit, with giant tests,
    benchmarks, examples, and install/package scripts each carrying a bounded
    role instead of blurred guarantees

This is the strongest Day 11 clarification:

- "state of the art" for Epic 7 is not a promise to solve every structural
  limitation in one epic
- it is the combination of coherent product boundaries, bounded capability
  modernization, trustworthy performance evidence, and truthfully documented
  platform/proof ownership

#### 2. True target qualities, strong-but-bounded qualities, and deferred ambitions are now separated explicitly

Day 11 fixes three distinct categories.

True state-of-the-art target qualities for Epic 7:

- direct-solver workflow boundaries that center the real repeated-run and
  owned-factor product lanes
- a credible index-width modernization path instead of a permanent implicit
  `int32_t` ceiling
- benchmarked compressed-backend maturity that reads as a maintained product
  capability rather than a hidden implementation detail
- public docs and key headers that give a coherent modern product story
- proof ownership that is explicit, bounded, and resistant to drift
- reviewed-platform and install/package wording that remains strictly evidence
  based

Strong-but-bounded qualities that should remain framed honestly:

- static-first packaging and release shape
- self-contained default build and dependency posture
- symmetric sparse eigensolver strength without broad unsymmetric coverage
- supplemental install/package and extra-compiler proof outside the reviewed
  core lanes
- threshold-free benchmark reporting rather than portable timing-gate claims

Intentionally deferred beyond-Epic-7 ambitions:

- full type-generic scalar support across the library
- broad complex-valued capability claims
- a full unsymmetric sparse eigensolver family
- reviewed shared-library and dynamic-ABI maturity
- reviewed Windows Makefile parity
- reviewed Windows dead-code parity
- reviewed macOS dead-code parity
- broad reviewed install-validation parity on every platform

Interpretation:

- later work can widen the product
- but the repo should not claim these deferred ambitions as if Sprint 70 had
  already made them real

#### 3. The strongest Epic 7 success criteria are now explicit

Later Epic 7 implementation should be judged against the following success
criteria:

1. the public direct-workflow story becomes more coherent without a fake
   repo-wide matrix rewrite
2. the strongest permanent capability ceiling is narrowed in a way that is
   reflected in public types, docs, and proof
3. benchmark and performance language stay bounded to maintained measured
   surfaces
4. the front-door, install, and key-header surfaces become materially more
   product-like and less archaeology-heavy
5. reviewed-platform truth stays exact even when supplemental proof expands
6. proof-owner roles remain explicit across tests, examples, benchmarks, and
   install scripts
7. later sprints reduce major permanent-source and giant-test hotspots only
   where ownership clarity or validation value justifies the proof cost

This is the core Day 11 rubric:

- Epic 7 succeeds if the library becomes more coherent, more honest, and more
  extensible on the highest-value seams
- not if it simply accumulates more surface area or more aspirational wording

#### 4. The strongest anti-goals are now fixed in writing

Sprint 70 Day 11 also fixes the following anti-goals:

- no fake "state-of-the-art" language that outruns shipped capability or proof
- no broad `SparseMatrix` rewrite disguised as workflow cleanup
- no one-sprint attempt to solve index width, scalar breadth, and eigensolver
  expansion together
- no benchmark-report wording that turns canonical snapshots into timing gates
- no platform or install-language widening without new reviewed evidence
- no giant-test breakup campaign detached from clear ownership or validation
  gain
- no permanent-surface cleanup pass that steals authority from tests, examples,
  benchmarks, or policy docs

Interpretation:

- later sprints should move on ranked seams
- not on generic modernization pressure

#### 5. The Sprint 71-79 carry-in queue is now ranked against the target model

The final carry-in queue for the remaining Epic 7 sprints is now explicit:

1. public direct-workflow boundary convergence
   - clarify the public direct-solver ownership boundary across README, key
     headers, and example handoff
2. product-model and configuration modernization
   - reduce linked-list-first identity pressure and continue env-var/default
     modernization on the highest-value public seams
3. capability modernization phase 1
   - start the bounded index-width modernization lane
4. backend and performance maturity phase 1
   - deepen maintained compressed-backend/product alignment without widening
     benchmark claims
5. benchmark and proof-surface refinement
   - preserve bounded benchmark governance while reducing the strongest proof
     archaeology seams
6. packaging/platform/install convergence phase 2
   - improve product coherence without inflating reviewed platform maturity
7. permanent-surface and header cleanup
   - reduce chronology density on the strongest public/reference surfaces
8. giant-test and large-source follow-through
   - touch only where ownership clarity or assurance value remains highest
9. final integration and validation package
   - rerun the strongest proof, benchmark, and install/package surfaces
10. Epic 7 closeout and handoff
    - freeze the final shipped reading, residual queue, and retrospective state

This ordering matters because it matches the actual target model:

- product coherence first
- capability and backend maturity next
- cleanup and closeout only after those structural seams move

### Day 11 Close

Sprint 70 now has one explicit Epic 7 target synthesis:

1. target quality:
   - coherent direct-workflow product model, bounded capability modernization,
     benchmark-governed performance maturity, product-like public surfaces, and
     truthful platform/proof ownership
2. strong-but-bounded quality:
   - static-first packaging, self-contained default build, symmetric-eigs
     strength, and supplemental proof outside the reviewed core lanes
3. deferred ambition:
   - full type-generic scalar breadth, broad unsymmetric eigs, shared-library
     maturity, and full reviewed cross-platform parity

That gives Day 12 one exact job:

- convert the target synthesis into one explicit later-sprint execution map so
  Epic 7 implementation can start from ranked contracts instead of broad themes

## Day 12 - Architecture Contract and Non-Goal Fence

### Goal

Turn the Day 3-11 audit and synthesis work into one exact Epic 7 architecture
contract so later sprints start from preserved boundaries, explicit non-goals,
and a planning map that matches both the live repo and the Epic 7 project
plan.

### Inputs Rechecked

I re-read the live Sprint 70 audit/synthesis package plus the Epic 7 project
plan section that Sprint 70 is supposed to freeze:

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day3-product-model-gap-inventory.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day4-first-product-model-boundary.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day5-capability-ceiling-audit.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day6-capability-modernization-fence.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day7-public-and-proof-surface-audit.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day8-support-surface-fence.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day9-validation-and-platform-contract-audit.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day10-validation-platform-fence.md`
- `docs/planning/EPIC_7/SPRINT_70/artifacts/day11-epic7-target-synthesis.md`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

### Findings

#### 1. Epic 7 now has one exact preserved boundary for each major implementation lane

The Day 12 contract fixes the following preserved boundaries.

Product-model convergence:

- first center:
  - public direct-workflow boundary
  - owned-factor and repeated-run workflow clarity
- explicitly not first center:
  - broad `SparseMatrix` rewrite
  - repo-wide generic matrix arithmetic redesign
  - conversion/writeback redesign across every compressed path at once

Capability modernization:

- first lane:
  - index-width modernization path
- second lane:
  - scalar-surface preparation and broadening
- later lane:
  - unsymmetric sparse eigensolver expansion
- explicitly not allowed:
  - solving width, scalar breadth, and algorithm-family expansion in one batch

Configuration modernization:

- target:
  - reduce residual env-var and compatibility-only control pressure on the
    highest-value public seams
- preserved rule:
  - typed or internal policy should widen only where precedence and ownership
    stay explicit
- explicitly not allowed:
  - another broad env-var cleanup pass disconnected from public or proof value

Backend/performance work:

- target:
  - deepen maintained compressed-backend maturity on the strongest product
    paths
- preserved rule:
  - benchmark-governed proof remains bounded and threshold free unless a new
    narrower gate is explicitly justified
- explicitly not allowed:
  - performance-story widening that outruns benchmark ownership or fallback
    proof

Benchmark governance:

- target:
  - preserve the canonical maintained surface and improve observability only in
    bounded reviewed ways
- preserved rule:
  - `make bench-canonical-report` remains artifact reporting, not a timing gate
- explicitly not allowed:
  - portable performance claims from local benchmark rows alone

Packaging/platform truthfulness:

- target:
  - preserve the static-first, evidence-based, explicitly asymmetric reviewed
    contract while tightening only what later proof actually supports
- preserved rule:
  - supplemental proof stays supplemental until new reviewed evidence exists
- explicitly not allowed:
  - shared-library maturity marketing, inflated Windows parity claims, or broad
    reviewed install-validation claims without new proof

This is the most useful Day 12 consolidation:

- every later Epic 7 lane now has an allowed center, a preserved rule, and an
  explicit "do not widen into this" boundary

#### 2. The strongest cross-cutting non-goal fence is now fixed in one place

Sprint 70 Day 12 fixes the following cross-cutting non-goals for the rest of
Epic 7:

- no broad rewrite of the library core
- no generic matrix-model redesign that tries to solve all ownership problems
  at once
- no unsupported product marketing claims
- no capability promises without end-to-end proof
- no benchmark or example wording that steals test-owned guarantees
- no drift from maintained reviewed validation truth
- no fake platform/release maturity beyond static-first, evidence-backed
  reality
- no giant-test or large-source breakup wave unless ownership clarity or
  assurance value justifies the proof cost

Interpretation:

- later sprints may land meaningful structural work
- but they should do it through ranked bounded seams rather than through
  broad-library reinvention pressure

#### 3. The planning implications for Sprints 71-79 are now explicit

The Day 12 contract fixes the later-sprint sequence in executable terms rather
than only as theme names.

Sprint 71:

- should reduce chronology and policy spill from the strongest public surfaces
- should not reopen product-model or platform-contract work unless the cleanup
  truly forces it

Sprint 72:

- should be the first real product-model convergence sprint
- should center the direct-workflow ownership seam before broader matrix-model
  ambitions

Sprint 73:

- should continue configuration modernization only on ranked residual controls
- should not become a generic "remove env vars everywhere" pass

Sprint 74:

- should land the first real capability seam on index width
- should treat scalar breadth as preparation/support, not as the same sprint's
  co-equal primary target

Sprint 75:

- should deepen backend/performance architecture only where fallback,
  observability, and benchmark proof remain bounded

Sprint 76:

- should extend benchmark and proof surfaces only where the widened product or
  backend story needs stronger evidence

Sprint 77:

- should tighten packaging/platform/install convergence without widening the
  reviewed platform contract beyond shipped proof

Sprint 78:

- should spend large-source, giant-test, and header cleanup budget only on the
  highest-value residual hotspots

Sprint 79:

- should be integration, final validation, and closeout
- not another broad feature or architecture sprint

This is the main Day 12 planning implication:

- the later Epic 7 sequence is now contract driven
- not just order-of-operations driven

#### 4. The architecture contract matches the live repo and the Epic 7 project plan

The Day 12 re-read confirms the Sprint 70 contract is aligned with the Epic 7
project plan instead of drifting from it:

- Sprint 70:
  - baseline, target, and architecture contract only
- Sprint 71:
  - public-surface cleanup first
- Sprint 72:
  - product-model convergence first
- Sprint 73:
  - residual configuration modernization
- Sprint 74:
  - bounded capability modernization led by index width
- Sprint 75+:
  - backend, benchmark, packaging/platform, cleanup, and closeout in that
    order

This matters because Sprint 70 now closes with one real planning contract:

- the project plan is still valid
- but later work now has a sharper execution fence than the raw plan text alone

### Day 12 Close

Sprint 70 now has one explicit architecture contract for Epic 7:

1. preserved implementation boundaries:
   - product-model, capability, configuration, backend/performance, benchmark,
     and packaging/platform lanes each have an allowed center and an explicit
     widening fence
2. cross-cutting non-goals:
   - no broad rewrite, no unsupported marketing, no proofless capability
     promises, and no drift from reviewed validation truth
3. planning implications:
   - Sprints 71-79 now inherit a contract-driven execution map rather than
     broad modernization themes

That gives Day 13 one exact job:

- re-read the full Sprint 70 package for coherence, gaps, and handoff quality
  before the sprint closeout days
