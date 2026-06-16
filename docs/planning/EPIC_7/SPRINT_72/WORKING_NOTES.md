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

## Day 6 - Ownership Convergence Batch 1

### Goal

Land the first bounded Sprint 72 implementation batch so the public
direct-workflow seam reads more coherently and the matrix shell no longer keeps
stale one-shot solve compatibility after permutation state is reset.

### Actions

1. Re-read the Day 5 implementation contract against the touched first-batch
   surfaces.
2. Tighten the ownership split wording across the first public headers:
   - `include/sparse_matrix.h`
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
3. Inspect `src/sparse_matrix.c` for the smallest matrix-shell state transition
   that still contradicts the Day 5 design.
4. Land one bounded behavior fix plus one focused regression on the live
   direct-workflow seam.
5. Run the full Day 6 validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Findings

#### 1. The public ownership split is now explicit instead of implied

The first header batch now states the intended product-model reading directly:

- `SparseMatrix` remains the mutable sparse construction and one-shot
  direct-workflow compatibility shell
- `sparse_analysis.h` now reads more directly as the clearer long-lived owner
  of reusable symbolic and factor/workspace state
- the one-shot LU / Cholesky / LDL^T headers now point back to the repeated-run
  lifecycle as the clearer reuse owner instead of reading like the only solver
  center of gravity

This keeps the existing public surface intact, but it reduces the old ambiguity
that made the copied matrix shell read like the long-term direct-solver product
identity.

#### 2. `sparse_reset_perms()` carried the strongest live state contradiction in the first batch

The strongest bounded mechanics issue in the first Sprint 72 landing turned out
to be the matrix-shell reset path.

Before the Day 6 batch:

- a copied matrix that had already gone through one-shot LU factorization could
  retain solve-ready compatibility state
- `sparse_reset_perms()` rewrote row and column permutation arrays back to
  identity
- but the function did not clearly drop the stale one-shot reordered/factored
  compatibility that depended on the old permutation shell

That left the matrix shell in an ownership state that contradicted the Day 5
design: the outward matrix permutation shell looked plain again, while the
solve path could still read like the old permuted factor shell was valid.

#### 3. The landed behavior fix now treats permutation reset as recovery of a plain matrix shell

The Day 6 implementation batch fixes that contradiction in `src/sparse_matrix.c`:

- `sparse_reset_perms()` now detects when a matrix carries either:
  - a stored reorder permutation, or
  - non-identity row / column permutation shells
- after restoring the visible matrix permutation shell to identity, it now
  clears the reorder permutation compatibility state
- when the matrix had one-shot factored/reordered compatibility tied to the old
  permutation shell, it now clears that factor compatibility too

Interpretation:

- resetting the permutation shell now means recovering a plain matrix shell
- callers can still refactor or reuse the repeated-run analysis/factor lane
- but they no longer keep a stale one-shot solve contract after destroying the
  shell that contract depended on

#### 4. The new regression proves the bounded ownership rule directly

The first implementation batch adds a focused integration regression in
`tests/test_integration.c` that proves:

- a copied matrix can still factor and solve through the one-shot LU lane
- the factored shell initially carries a non-identity row permutation
- after `sparse_reset_perms()`:
  - row and column permutation shells return to identity
  - the old one-shot LU solve compatibility is rejected with
    `SPARSE_ERR_BADARG`

That is the right Day 6 proof shape because it tests the live public seam
instead of just the internal helper mechanics.

#### 5. The first code batch stayed inside the Day 5 fence

The landed Day 6 batch touched only:

- the planned public ownership headers
- `src/sparse_matrix.c`
- one focused proof surface in `tests/test_integration.c`

It did not widen into:

- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- capability surfaces
- packaging/platform/docs truth surfaces
- giant proof-surface redesign

### Validation

The full Day 6 gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity stayed `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `test_reorder_nd` remained the dominant reviewed long-tail test at
  `229.99 sec`
- `Total Test time (real) = 324.07 sec`

### Day 6 Exit State

Sprint 72 Day 6 closes with:

1. one landed public ownership wording batch
2. one bounded matrix-shell state fix in `sparse_reset_perms()`
3. one focused integration regression proving stale permuted one-shot LU shells
   are invalidated after permutation reset
4. one full reviewed validation pass with exact parity preserved

## Day 7 - Post-Landing Audit and Rerank

### Goal

Re-rank the remaining Sprint 72 product-model seams from the live post-Day-6
state so the next implementation lane follows the strongest remaining
ownership contradiction instead of forcing a fake second matrix-shell batch.

### Actions

1. Re-read the Day 6 landing against the Day 3-5 product-model ranking.
2. Re-check the touched first-lane surfaces:
   - `include/sparse_matrix.h`
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
   - `src/sparse_matrix.c`
   - `tests/test_integration.c`
3. Re-check the strongest deferred compressed-path candidates:
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_lu_csr.c`
4. Decide whether the second lane should stay on the direct matrix shell or
   move to the strongest compressed-path publication seam.
5. Fix the exact Day 8 design target in writing.

### Findings

#### 1. The Day 6 landing closed the strongest first matrix-shell contradiction

The Day 6 batch materially closed the exact contradiction Sprint 72 ranked
first on Day 3:

- the public ownership split is no longer mostly implied
- `SparseMatrix` now reads more directly as the mutable construction and
  one-shot compatibility shell
- the repeated-run analysis/factor lane now reads more directly as the clearer
  long-lived owner of reusable symbolic and factor/workspace state
- the shell reset path no longer leaves stale one-shot solve compatibility
  behind after permutation recovery

That means the original first direct-workflow contradiction is no longer the
strongest remaining Sprint 72 seam.

#### 2. A second matrix-shell batch would now be lower-yield than the deferred compressed-path seam

The post-Day-6 state still leaves `SparseMatrix` carrying many roles, but the
strongest first-order contradiction is no longer in:

- `include/sparse_matrix.h`
- `src/sparse_matrix.c`
- the basic one-shot versus repeated-run ownership wording

What remains on the matrix-shell side is now more support/deferred pressure:

- mixed logical versus physical matrix-state semantics
- broader compatibility-shell accumulation over time
- later cleanup around generic matrix-state density and chronology

Those are real, but they are no longer the best next bounded Sprint 72
landing.

#### 3. The strongest remaining seam is now the Cholesky CSC publish-back contract

The post-Day-6 rerank shifts the strongest remaining ownership blur to the
transparent CSC-backed Cholesky path centered on:

- `src/sparse_chol_csc.c`
- support only if needed:
  - `include/sparse_cholesky.h`
  - `tests/test_chol_csc.c`
  - `tests/test_integration.c`

Why this seam is now strongest:

- the Cholesky CSC backend is the clearest live place where a compressed factor
  is still transparently transplanted back into the public matrix shell
- `chol_csc_writeback_to_sparse(...)` is doing real product-model work:
  conversion, filtering, pool/header transplant, factor-state publication, and
  permutation-state publication
- the public header still documents the path in terms of temporary reordered
  working copies and later publish-back, which means the compressed-path
  ownership seam is not merely internal

Interpretation:

- the matrix-shell side is now clearer about what it is
- the next ambiguity is how a CSC-owned factor/result gets published back
  through that shell for one-shot compatibility

#### 4. LDL^T and LU remain real compressed-path lanes, but they are weaker second targets than Cholesky

The rerank also clarifies why the other deferred files are not the best Day 8
design center:

- `src/sparse_ldlt_csc.c` is large and real, but its strongest writeback seam
  lands in a separately-owned `sparse_ldlt_t` result struct rather than
  overwriting the caller matrix shell, so the public product-model
  contradiction is weaker than on the Cholesky side
- `src/sparse_lu_csr.c` remains an important support seam, but its strongest
  current pressure is still more internal conversion/update structure than
  matrix-shell publication ownership

So the Cholesky CSC lane is the best second Sprint 72 landing because it still
connects compressed working ownership to the live public matrix-shell contract
most directly.

#### 5. The proof and support rerank is now explicit

The strongest likely Day 8-9 proof homes are now:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`

Support only if the design truly forces it:

- `include/sparse_cholesky.h`

Explicitly not the next design center:

- `tests/test_sparse_matrix.c`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`

### Day 7 Exit State

Sprint 72 Day 7 closes with:

1. the Day 6 landing confirmed as having closed the strongest first
   matrix-shell contradiction
2. the second implementation lane reranked away from another generic
   matrix-shell batch
3. the strongest remaining ownership seam fixed to the Cholesky CSC
   publish-back/publication contract
4. the exact Day 8 design target fixed to `src/sparse_chol_csc.c` with
   `tests/test_chol_csc.c` and `tests/test_integration.c` as the likely proof
   homes

## Day 8 - Compressed-Path Ownership Design

### Goal

Define the bounded second Sprint 72 implementation batch around the strongest
remaining Cholesky CSC publish-back seam so Day 9 can reduce compressed-path
ownership blur without widening into a broader backend redesign.

### Actions

1. Re-read the Day 7 rerank against the live Cholesky CSC publish-back code in
   `src/sparse_chol_csc.c`.
2. Re-read the current caller-facing Cholesky publish-back wording in
   `include/sparse_cholesky.h`.
3. Re-check the existing proof homes:
   - `tests/test_chol_csc.c`
   - `tests/test_integration.c`
4. Decide what exact ownership split the second batch should make clearer:
   - CSC factor materialization
   - matrix-shell transplant/publication
   - factor-state and reorder-state publication
5. Fix the exact Day 9 touched-file fence and the preserved compatibility
   checklist in writing.

### Findings

#### 1. The strongest remaining blur is inside `chol_csc_writeback_to_sparse(...)`, not in CSC numeric elimination itself

The Day 8 review confirms that the strongest remaining ownership contradiction
is not the CSC numeric kernel broadly. It is the publish-back seam in
`chol_csc_writeback_to_sparse(...)`.

That helper is still carrying multiple responsibilities at once:

- validate that the caller matrix shell is in original state
- copy the external permutation publication payload
- materialize a temporary linked-list shell from the CSC factor
- transplant the temporary shell storage into the caller matrix
- publish factor and reorder compatibility state back onto the caller shell

This is exactly the kind of mixed ownership bundle Sprint 72 is supposed to
separate more clearly.

#### 2. The best second batch is a publication-phase separation, not a backend-selection redesign

The right Day 9 landing is now explicit:

- keep the current public one-shot Cholesky semantics
- keep CSC dispatch and `used_csc_path` semantics unchanged
- keep the temporary reordered working-copy contract unchanged
- reduce ownership blur inside the publish-back path by separating:
  - CSC-to-temporary-shell materialization
  - caller-shell transplant
  - factor/reorder compatibility publication

Interpretation:

- the second batch should make the compressed factor own its own materialized
  temporary shell before any caller-shell mutation is committed
- the caller shell should then receive one bounded publication step instead of
  one large mixed helper body

#### 3. The exact touched-file fence is now fixed

Required second-batch design center:

- `src/sparse_chol_csc.c`

Likely proof homes:

- `tests/test_chol_csc.c`
- `tests/test_integration.c`

Support only if the exact code batch truly changes caller-facing wording:

- `include/sparse_cholesky.h`

Explicitly out of scope for the Day 9 batch:

- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- `src/sparse_matrix.c`
- `include/sparse_matrix.h`
- capability/type surfaces
- packaging/platform/docs truth surfaces
- broad benchmark or example spill

#### 4. The best proof shape is already implied by the live tests

The existing proof map suggests the right Day 9 validation targets:

- `tests/test_chol_csc.c` should remain the family-local owner for:
  - writeback preconditions
  - CSC versus linked-list path equivalence
  - writeback round-trip behavior
- `tests/test_integration.c` should remain the cross-surface owner for:
  - one-shot Cholesky path versus explicit repeated-run analysis path parity
  - public `used_csc_path` and dispatch-side behavior

That means Day 9 should prefer focused regression additions or tightening in
those existing homes rather than introducing new proof surfaces.

#### 5. The preserved compatibility checklist is now explicit

The Day 9 batch must preserve:

- one-shot Cholesky factorization still publishing a solve-ready matrix shell
  on success
- reordered one-shot attempts publishing only after successful factorization
- `used_csc_path` reporting semantics
- linked-list and CSC solve-result parity
- the Sprint 72 Day 6 matrix-shell reset rule

It must explicitly not widen into:

- new family-local factor types
- public API redesign
- backend-threshold or dispatch-policy changes
- broad compressed-path cleanup across every family

### Day 8 Exit State

Sprint 72 Day 8 closes with:

1. one exact second implementation design centered on the Cholesky CSC
   publish-back seam
2. one fixed touched-file fence for Day 9
3. one proof-home map anchored to `tests/test_chol_csc.c` and
   `tests/test_integration.c`
4. one preserved compatibility checklist that keeps the batch bounded to
   publication ownership cleanup

## Day 9 - Compressed-Path Ownership Batch

### Goal

Land the bounded second Sprint 72 implementation batch so the Cholesky CSC
publish-back seam reads as distinct materialize, transplant, and publication
phases without widening the public one-shot contract or the broader compressed
direct-family design.

### Actions

1. Re-read the Day 8 design against the live `chol_csc_writeback_to_sparse(...)`
   implementation in `src/sparse_chol_csc.c`.
2. Extract the smallest helper split that cleanly separates:
   - reorder-permutation payload copying
   - CSC-factor to temporary-shell materialization
   - caller-shell transplant
   - factor and reorder compatibility publication
3. Keep the Day 8 non-touch set intact:
   - no LDL^T follow-through
   - no LU CSR follow-through
   - no `SparseMatrix` redesign
   - no public API expansion
4. Add one focused family-local regression in `tests/test_chol_csc.c` that
   proves the writeback-produced shell is publish-ready and solve-ready.
5. Run the full Day 9 validation gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Findings

#### 1. `chol_csc_writeback_to_sparse(...)` now reads as one bounded publication pipeline instead of one mixed helper body

The Day 9 code batch landed exactly where Day 8 ranked the strongest remaining
compressed-path ownership blur:

- `src/sparse_chol_csc.c`

The old helper body mixed:

- permutation publication copying
- temporary linked-list factor materialization
- caller-shell storage transplant
- factor and reorder compatibility publication

The landed split now gives each phase a direct internal owner:

- `chol_csc_copy_reorder_perm(...)`
- `chol_csc_materialize_sparse_factor(...)`
- `chol_csc_transplant_materialized_factor(...)`
- `chol_csc_publish_materialized_factor(...)`

That keeps the public one-shot contract unchanged, but it makes the internal
publish-back seam read as one bounded pipeline instead of one large blended
implementation block.

#### 2. The batch preserves the strongest Day 8 compatibility rules

The landed Day 9 refactor preserves the exact bounded contract Sprint 72
needed:

- one-shot Cholesky factorization still publishes a solve-ready matrix shell
  on success
- reordered one-shot attempts still publish only after successful factorization
- `used_csc_path` semantics stay unchanged
- linked-list and CSC parity stay preserved
- the Day 6 matrix-shell reset rule stays intact

The helper split is intentionally internal. It does not widen into:

- new family-local factor types
- threshold or dispatch-policy changes
- public API redesign
- broad compressed-path cleanup across every direct family

#### 3. The new proof closes the exact family-local publish-back claim

The Day 9 regression lives in the planned strongest proof home:

- `tests/test_chol_csc.c`

The new test proves that a CSC factor written back through
`chol_csc_writeback_to_sparse(...)` leaves the caller shell in the intended
post-publication state:

- the matrix is factored and solve-ready
- the published reorder permutation matches the explicit permutation payload
- the internal row and column permutation shells are identity
- the writeback-produced shell solves the original SPD system correctly

That is the right proof shape for this batch because it validates the exact
publication seam without widening into unrelated public-path integration work.

#### 4. The second implementation batch stayed inside the Day 8 fence

The landed Day 9 batch touched only:

- `src/sparse_chol_csc.c`
- `tests/test_chol_csc.c`

It did not widen into:

- `src/sparse_ldlt_csc.c`
- `src/sparse_lu_csr.c`
- `src/sparse_matrix.c`
- `include/sparse_matrix.h`
- public capability/type surfaces
- packaging/platform/docs truth surfaces
- broad benchmark/example spill

### Validation

The full Day 9 gate passed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Reviewed anchors remained exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity stayed `53 vs 53`
- full reviewed CMake `ctest` passed `53 / 53`
- `test_reorder_nd` remained the dominant reviewed long-tail test at
  `227.22 sec`
- `Total Test time (real) = 325.88 sec`

### Day 9 Exit State

Sprint 72 Day 9 closes with:

1. one landed Cholesky CSC publish-back ownership split
2. one focused family-local regression proving the writeback-produced shell is
   published and solve-ready
3. one preserved one-shot Cholesky compatibility contract
4. one full reviewed validation pass with exact parity preserved

## Day 10 - Public Contract and Example Adoption Design

### Goal

Define the exact public-header, doc, and example follow-through that the Day 6
and Day 9 ownership work actually requires, while explicitly keeping already
coherent surfaces out of scope.

### Actions

1. Re-read the Day 6 and Day 9 implementation landings against the current
   public-facing contract surfaces:
   - `include/sparse_matrix.h`
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
   - `README.md`
   - `docs/tutorial.md`
   - `examples/example_analysis.c`
   - `examples/example_basic_solve.c`
2. Separate:
   - wording or example follow-through that is now genuinely required
   - surfaces that already match the landed ownership split and should not
     move
3. Check the public contract specifically against the two landed behaviors:
   - Day 6 permutation reset invalidates stale one-shot solve compatibility
   - Day 9 Cholesky CSC writeback publishes a solve-ready compatibility shell
4. Fix the exact Day 11 follow-through fence and preserved truthfulness
   checklist in writing.
5. Keep Sprint 70 truthfulness and Sprint 71 cleanup gains intact by avoiding
   generic documentation spill.

### Findings

#### 1. The strongest public contract follow-through is still header-local, not front-door or tutorial-heavy

The Day 6 and Day 9 implementation landings changed real ownership mechanics,
but the broad public story did not move enough to require another README-first
or tutorial-first cleanup pass.

The strongest live contract surfaces remain:

- `include/sparse_matrix.h`
- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`

Interpretation:

- the public ownership split is now primarily carried by the headers
- Sprint 71 already removed the strongest front-door and install-surface drift
- Day 10 therefore should not reopen those broader product surfaces unless a
  new contradiction actually appears

#### 2. The Day 6 and Day 9 public-facing story is already mostly coherent in README, tutorial, and shipped examples

The reread against `README.md`, `docs/tutorial.md`, and the two strongest
shipped direct-workflow examples shows that the current public adoption story
already matches the landed implementation direction:

- one-shot direct APIs remain first-class entry points
- callers should still use a fresh matrix or a fresh `sparse_copy()` when the
  original coefficient view must be preserved
- the explicit repeated-run direct lifecycle remains the clearer owner of
  reusable symbolic and factor/workspace state
- `example_basic_solve.c` still demonstrates the one-shot copy-then-factor
  discipline directly
- `example_analysis.c` still demonstrates the explicit same-pattern repeated-
  run lane directly
- `docs/tutorial.md` already keeps the one-shot-versus-repeated-run adoption
  split readable

That means the strongest likely Day 11 move is smaller than the original plan
pressure suggested.

#### 3. The only meaningful Day 11 follow-through center is the bounded direct-workflow contract wording, not example or README churn

The two landed behavior clarifications that could still justify follow-through
are narrow:

- the matrix shell recovers to a plain matrix after `sparse_reset_perms()`
  instead of keeping stale one-shot reordered/factored solve compatibility
- Cholesky CSC writeback is an internal compatibility-shell publication path,
  not a shift in long-lived factor ownership away from the explicit repeated-
  run direct lifecycle

Those are both header-local ownership clarifications first.

Day 11 therefore should treat these surfaces as the only likely required
follow-through center:

- `include/sparse_matrix.h`
- `include/sparse_cholesky.h`

Support only if the exact wording truly proves necessary after the final
edit pass:

- `README.md`
- `docs/tutorial.md`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

Support-first but likely non-moving:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_ldlt.h`

#### 4. The preserved truthfulness checklist is now explicit

Any Day 11 follow-through must preserve:

- `SparseMatrix` as the mutable construction and one-shot compatibility shell
- the explicit repeated-run direct lifecycle as the clearer long-lived owner
  of reusable symbolic and factor/workspace state
- one-shot direct APIs as first-class peer entry points rather than deprecated
  shims
- the copy-first discipline for one-shot factorization when callers still need
  the original coefficient view later
- Cholesky CSC writeback as an internal publish-back path that still returns a
  standard solve-ready matrix shell

It must explicitly avoid:

- reopening Sprint 71 front-door cleanup
- generic tutorial or example rewrites
- capability/platform/install spill
- new claims about long-lived factor ownership beyond what the shipped APIs
  actually guarantee

### Day 10 Exit State

Sprint 72 Day 10 closes with:

1. one explicit Day 11 follow-through fence centered on bounded
   direct-workflow contract wording
2. one clear separation between actually-required contract follow-through and
   already-coherent README/tutorial/example surfaces
3. one preserved truthfulness checklist that keeps the sprint out of generic
   documentation spill
4. one narrowed next step that lets Day 11 land only if the header-local
   wording still needs adjustment

## Day 11 - Public Contract and Example Adoption Batch

### Goal

Land only the exact public-facing follow-through still required by the Day 6
and Day 9 ownership work, and prove that the broader README/tutorial/example
surface does not need to move.

### Actions

1. Re-read the Day 10 follow-through fence against the two narrowed contract
   surfaces:
   - `include/sparse_matrix.h`
   - `include/sparse_cholesky.h`
2. Tighten wording only where the landed implementation batches still left the
   ownership rule implicit:
   - copied factored matrix-shell compatibility after `sparse_copy()`
   - Cholesky CSC publish-back as an internal solve-ready compatibility-shell
     return rather than a shift in long-lived factor ownership
3. Reconfirm that the support surfaces remain coherent and therefore should
   not move:
   - `README.md`
   - `docs/tutorial.md`
   - `examples/example_analysis.c`
   - `examples/example_basic_solve.c`
4. Run the required Day 11 gate for touched public headers:
   - `make format`
   - `make lint`
   - `make test`
5. Record the exact touched-surface result and validation notes.

### Findings

#### 1. The Day 11 follow-through stayed exactly header-local

The bounded Day 11 batch landed exactly where Day 10 said the remaining public
contract drift still lived:

- `include/sparse_matrix.h`
- `include/sparse_cholesky.h`

No broader surface had to move.

That is the right Sprint 72 outcome because the Day 6 and Day 9 implementation
changes were ownership clarifications inside the matrix shell and the Cholesky
CSC publish-back seam, not a new front-door or tutorial story.

#### 2. `SparseMatrix` copy semantics now state the Day 6 reset rule directly

The Day 11 header wording now makes one important Day 6 rule explicit in
`include/sparse_matrix.h`:

- copying a factored matrix still preserves the one-shot matrix-shell solve
  contract
- that compatibility is only a matrix-shell contract and is dropped again once
  later matrix-shell mutation or `sparse_reset_perms()` rewrites the shell

That is the right bounded clarification because it keeps the one-shot copy
discipline truthful without implying that copied matrix shells are long-lived
factor owners.

#### 3. Cholesky CSC publish-back now states the Day 9 ownership rule directly

The Day 11 wording in `include/sparse_cholesky.h` now states the exact Day 9
public contract more directly:

- the CSC lane still returns the same solve-ready `SparseMatrix`
  compatibility shell that the linked-list path returns
- the CSC publish-back step does not transfer long-lived factor ownership away
  from the explicit repeated-run direct lifecycle in `sparse_analysis.h`

This is the smallest useful clarification of the Day 9 batch because it makes
the transparent CSC lane read as an internal compatibility-shell publication
path rather than a second long-lived public factor owner.

#### 4. The README/tutorial/example surfaces remained coherent and did not move

The Day 11 reread confirms that these support surfaces already match the
landed ownership story and therefore did not need edits:

- `README.md`
- `docs/tutorial.md`
- `examples/example_analysis.c`
- `examples/example_basic_solve.c`

That preserved the Day 10 non-goal fence:

- no generic docs spill
- no example churn without a real contradiction
- no reopened Sprint 71 public-surface cleanup

### Validation

The required Day 11 gate passed:

- `make format`
- `make lint`
- `make test`

Touched-surface raw `wc -l` counts:

- `include/sparse_matrix.h` = `604`
- `include/sparse_cholesky.h` = `220`

### Day 11 Exit State

Sprint 72 Day 11 closes with:

1. one bounded public-header follow-through batch
2. one explicit restatement of the Day 6 matrix-shell reset rule in
   `include/sparse_matrix.h`
3. one explicit restatement of the Day 9 Cholesky CSC publish-back ownership
   rule in `include/sparse_cholesky.h`
4. one confirmed non-move of the broader README/tutorial/example adoption
   surfaces
5. one clean Day 11 validation pass
