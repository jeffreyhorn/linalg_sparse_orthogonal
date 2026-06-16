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

## Day 3 - Product-Model Surface Audit I

### Goal

Re-rank where `SparseMatrix` still behaves like the right public owner and
where it now reads more like a compatibility shell around the stronger
compressed direct-workflow paths.

### Actions

1. Re-read the Day 3 scope in `docs/planning/EPIC_7/SPRINT_72/PLAN.md`.
2. Audit the current top product-model surfaces:
   - `include/sparse_matrix.h`
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
   - `src/sparse_matrix.c`
3. Re-read the strongest compressed direct-path support seams:
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_lu_csr.c`
4. Recheck the strongest current proof-owner surfaces for product-model drift:
   - `tests/test_sparse_matrix.c`
   - `tests/test_integration.c`
   - `tests/test_chol_csc.c`
   - `tests/test_ldlt_csc.c`
5. Classify the remaining burdens into:
   - copy/mutation surprise
   - mixed logical versus physical matrix-state semantics
   - duplicated publication or writeback ownership
   - factor/workspace ownership blur
6. Fix the first ranked contradiction map in writing.

### Findings

#### 1. The strongest exact Sprint 72 seam is still the copy-first one-shot direct workflow centered on `SparseMatrix`

The public direct-solver story still leads callers toward:

- building or loading a `SparseMatrix`
- copying the matrix when they need to preserve the coefficient view
- factoring the copy in place through the family-local one-shot entry point
- solving through the now-factored matrix shell
- relying on the matrix object itself to keep carrying permutation and
  factored-state compatibility

That remains a real strength for smaller or occasional use because it is still
low-ceremony and easy to explain.

It is also still the strongest Day 3 ceiling because the public center of
gravity remains:

- copy-first
- mutation-heavy
- matrix-state-sensitive
- centered on a linked-list compatibility shell even when the strongest
  numeric work is now clearly happening in CSC or CSR working formats

#### 2. The second strongest seam is the mixed logical/physical/permuted-state contract of the generic matrix API

`SparseMatrix` still carries too many roles at once:

- mutable construction and edit surface
- generic arithmetic surface
- permutation owner
- factored-state carrier
- interoperability shell
- storage-inspection surface for advanced callers

That shows up in the public surface through:

- distinct logical versus physical element accessors
- public permutation-array accessors
- `sparse_reset_perms(...)`
- warnings around arithmetic or mutation on non-identity-permutation matrices
- factored-state markers and state checks

This is powerful, but it also means a meaningful part of the generic matrix
API is only safe if the caller already knows which logical or physical state
the matrix is currently in.

#### 3. The third strongest seam is compressed direct-path work that still converts out of and publishes back into `SparseMatrix`

The strongest direct numeric paths now clearly live in compressed working
formats:

- CSC Cholesky
- CSC LDL^T
- CSR LU

But the current product story still routes through:

- linked-list input state
- conversion into CSC or CSR working formats
- compressed numeric factorization and solve
- publication or writeback back into a linked-list-facing result surface

The strongest exact examples are:

- `chol_csc_from_sparse_with_analysis(...)`
- `chol_csc_writeback_to_sparse(...)`
- `ldlt_csc_from_sparse_with_analysis(...)`
- `ldlt_csc_writeback_to_ldlt(...)`
- `lu_csr_from_sparse(...)`
- `lu_csr_to_sparse(...)`

So the compressed paths are already the real implementation centers, but they
still do not read like the library's dominant public product model.

#### 4. The shared repeated-run lifecycle is already a convergence step, but it still sits beside the one-shot family surfaces rather than replacing them as the dominant product center

The shared repeated-run lifecycle already aligns better with long-term
factor/workspace ownership:

- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

It already owns:

- symbolic and permutation reuse
- cross-family repeated-run direct workflow
- a cleaner factor/workspace split than the one-shot family-local paths

But it still reads as a parallel advanced surface rather than the single
dominant product model because:

- the one-shot family headers remain the easier public front door
- family-local owned-factor types still coexist beside shared
  `sparse_factors_t`
- same-pattern and identity-permutation preconditions remain fairly visible at
  the caller boundary

#### 5. The broad Sprint 72 product-model problem now reduces to four ranked seams

The ranked Day 3 product-model seams are now explicit:

1. copy-first, in-place one-shot direct workflow centered on `SparseMatrix`
2. mixed logical/physical/permuted-state semantics across the generic matrix
   API
3. compressed direct-path conversion and publication/writeback ownership split
4. shared repeated-run lifecycle versus family-local one-shot and owned-factor
   parallel surfaces

Useful lower-priority support context:

- compile-time threshold and backend-selector spill around the public direct
  workflow
- public permutation-array exposure that reinforces how much state still lives
  on the matrix object
- interoperability and Matrix Market flows that keep `SparseMatrix` as the
  universal public shell

### Day 3 Exit State

Sprint 72 Day 3 closes with one explicit current-state hotspot map:

1. the broad product-model problem is reduced to four ranked seams
2. the strongest exact target is still the copied-matrix, in-place one-shot
   direct workflow
3. the generic matrix API state-mixing seam is fixed as the strongest second
   contradiction
4. the compressed-path publication seam and repeated-run parallel-surface seam
   are both now explicit
5. Day 4 can now rerank those seams into the first true Sprint 72 convergence
   boundary instead of starting from a generic matrix-model slogan

## Day 4 - Product-Model Surface Audit II & First Landing Boundary

### Goal

Refine the Day 3 ranking and freeze the first bounded Sprint 72 convergence
fence before implementation design begins.

### Actions

1. Re-read the Day 4 scope in `docs/planning/EPIC_7/SPRINT_72/PLAN.md`.
2. Re-rank the Day 3 seams against:
   - public direct-workflow pain
   - implementation leverage
   - compatibility risk
   - bounded Sprint 72 payoff
3. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces them
   - later or explicitly deferred product-model surfaces
4. Re-test the strongest possible first fences:
   - direct one-shot workflow
   - repeated-run lifecycle handoff
   - compressed-path publication/writeback seam
   - deeper factor/workspace ownership seam
5. Fix the first Sprint 72 boundary and non-goal fence in writing.

### Findings

#### 1. The strongest first Sprint 72 fence is the public direct-workflow seam, not the deeper compressed writeback seam

The Day 4 rerank confirms the best first bounded lane is:

- direct one-shot workflow centered on `SparseMatrix`
- plus the repeated-run lifecycle handoff that already exists beside it

That lane has the strongest mix of:

- caller confusion cost
- public contract leverage
- bounded implementation payoff
- acceptable compatibility risk for a first convergence pass

By contrast, the compressed-path publication/writeback seam is real but is not
the right first landing because it quickly widens into family-specific
internals across:

- CSC Cholesky
- CSC LDL^T
- CSR LU

Interpretation:

- Sprint 72 should first make the public direct-workflow ownership boundary
  read more coherently
- it should not start by opening the deepest compressed publication machinery

#### 2. The generic matrix-state seam remains in-scope support context, but not a broad standalone rewrite target

The mixed logical/physical/permuted-state contract is still the strongest
second contradiction.

But the rerank shows it should be treated as:

- support context for the first direct-workflow landing
- not a separate repo-wide matrix-model rewrite target

This means the first batch can touch the matrix-state shell only where it
clarifies:

- one-shot direct-workflow ownership
- repeated-run handoff boundaries
- factor-state expectations visible to callers

It should not widen into:

- generic arithmetic redesign
- public permutation-accessor redesign
- broad logical-versus-physical API cleanup detached from the direct workflow

#### 3. The first-batch landing surfaces are now explicit

Required first landing:

- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `src/sparse_matrix.c`

Likely support only if the first landing forces it:

- `examples/example_analysis.c`
- `examples/example_basic_solve.c`
- `tests/test_integration.c`
- `tests/test_sparse_matrix.c`

Deferred or explicitly later surfaces:

- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- family-local proof-owner giant tests beyond support follow-through
- public capability or packaging/platform surfaces

#### 4. The strongest non-goal fence is now explicit

Sprint 72 Day 4 fixes the first-lane non-goals:

- no repo-wide `SparseMatrix` rewrite
- no capability or type widening hidden inside ownership cleanup
- no broad family-by-family redesign without a ranked center
- no compressed-path publication overhaul as the first move
- no factor/workspace abstraction campaign detached from the direct workflow

### Day 4 Exit State

Sprint 72 Day 4 closes with one explicit first convergence boundary:

1. the first landing centers on the public direct one-shot workflow and the
   repeated-run lifecycle handoff
2. `SparseMatrix` state mixing is in-scope support context, not a separate
   rewrite program
3. compressed-path publication/writeback work is explicitly deferred unless
   the first lane forces it
4. the first landing surfaces and support-only surfaces are fixed
5. Day 5 can now design a bounded implementation contract instead of debating
   where Sprint 72 should start

## Day 5 - Ownership Convergence Design

### Goal

Define the bounded implementation contract for the first Sprint 72 landing so
the code batch can improve direct-workflow ownership clarity without widening
into a broad matrix-model rewrite.

### Actions

1. Re-read the Day 5 scope in `docs/planning/EPIC_7/SPRINT_72/PLAN.md`.
2. Re-read the Sprint 70 target synthesis and architecture contract against
   the Day 4 first-batch surfaces.
3. Design the first landing around:
   - clearer direct-workflow ownership
   - reduced copy/mutation surprise
   - cleaner factor/workspace separation
   - preserved public compatibility
4. Decide what remains clearly owned by `SparseMatrix`, what should be pushed
   more explicitly toward the repeated-run lifecycle, and what must stay
   untouched in Sprint 72.
5. Fix the first-batch non-touch set and compatibility checklist in writing.

### Findings

#### 1. `SparseMatrix` remains the owner of bounded matrix-shell compatibility, not the owner of the long-term direct-solver product identity

The Day 5 design now fixes the intended ownership split:

`SparseMatrix` should remain the public owner of:

- mutable sparse construction and edit flow
- Matrix Market and generic interop shell behavior
- one-shot direct-workflow compatibility
- permutation-bearing matrix-shell publication for callers that still choose
  the one-shot lane
- factored-state compatibility markers needed by the one-shot lane

`SparseMatrix` should not keep reading like the owner of:

- reusable symbolic analysis
- long-lived factor/workspace state
- the best long-term repeated-run direct workflow
- the dominant product identity of the fastest compressed direct paths

Interpretation:

- Sprint 72 should clarify a bounded matrix-shell role
- it should not pretend the matrix object is disappearing
- it should also stop letting the matrix shell read like the full long-term
  direct-solver center

#### 2. The repeated-run lifecycle becomes the explicit long-lived owner of reusable symbolic and factor/workspace state

The Day 5 design fixes the repeated-run side as the clearer owner of:

- reusable symbolic and permutation preparation
- refactorable same-pattern numeric flow
- explicit factor/workspace lifetime separate from the matrix shell
- the strongest cross-family long-run direct workflow

That means the first batch should clarify the public relationship:

- one-shot family lanes remain supported
- repeated-run analysis/factor surfaces are the clearer reuse lane
- the matrix shell is not the best place to accumulate more long-lived solver
  ownership over time

#### 3. The first code batch should target ownership language and factor-state transitions, not deep compressed-path mechanics

The best first bounded implementation design is now explicit:

- clarify the public ownership split in:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- tighten the matrix-shell ownership mechanics in:
  - `src/sparse_matrix.c`

The strongest likely Day 6-7 implementation themes are:

- clearer invalidation/reset behavior around matrix mutation versus factored
  compatibility state
- clearer handoff wording between one-shot family APIs and repeated-run
  analysis/factor APIs
- clearer statement that compressed-path families publish back through the
  matrix shell for compatibility, not because the matrix shell is the real
  long-lived factor owner

Explicitly not in the first batch:

- CSC or CSR conversion redesign
- compressed-path publication/writeback redesign
- new family-local factor types
- removal of existing one-shot public entry points

#### 4. The first-batch non-touch set is now fixed

Sprint 72 Day 5 fixes the first-batch non-touch set:

- unrelated solver families outside the first ownership lane
- capability or type surfaces
- packaging/platform/install/workflow files
- broad public-doc cleanup spill
- giant proof-surface redesign
- deep compressed-path internal files unless the first ownership batch truly
  forces a bounded follow-through

### Day 5 Exit State

Sprint 72 Day 5 closes with one explicit implementation contract:

1. `SparseMatrix` keeps a bounded compatibility-shell role
2. repeated-run analysis/factor surfaces are fixed as the clearer long-lived
   reuse lane
3. the first code batch is aimed at ownership language and factor-state
   mechanics, not deep compressed-path redesign
4. the non-touch set and compatibility fence are fixed before code edits begin
5. Day 6 can now land the first ownership cleanup against an explicit design
