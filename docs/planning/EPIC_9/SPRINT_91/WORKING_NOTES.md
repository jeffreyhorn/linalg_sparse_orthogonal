# Sprint 91 Working Notes

## Day 2 - Validation and Maintained Surface Recheck

### Goal
Refresh the implementation-day validation contract and the live maintained
reviewed, install/export, reporting, example, and workflow truth split before
Sprint 91 begins compressed-first implementation work on the direct workflow
surface.

### Actions
- Re-read the Sprint 91 Day 2 plan target in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the closest prior validation-contract artifact:
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day2-validation-baseline-and-maintained-surface-recheck.md`
- Reconfirmed the live reviewed parity anchor with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the maintained canonical benchmark-reporting owner with:
  - `make -n bench-canonical-report`
- Rechecked the presence of the strongest reviewed and maintained Sprint 91
  truth surfaces:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
  - `scripts/bench_canonical_report.sh`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
- Re-read the Linux, macOS, and Windows workflow surfaces so Sprint 91 does
  not overclaim reviewed parity or install/export breadth while touching the
  direct product model.
- Wrote the Day 2 artifact and fixed the authoritative rerun set in writing.

### Findings
- Sprint 91 continues to inherit the same strongest local reviewed baseline:
  - `make quality-review-full`
- The implementation-day and docs-day split is now fixed explicitly for
  compressed-first product work:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial product-contract, proof-owner, or support-surface batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- Reviewed CMake parity remains the primary truth anchor before any Sprint 91
  code lands:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The strongest reviewed executable truth owners for Sprint 91’s direct-product
  lane are now fixed around:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Canonical benchmark reporting remains command- and script-owned rather than
  reviewed-binary-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/export proof remains script- and fixture-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
  - `examples/cmake_example/CMakeLists.txt`
- Workflow truth remains intentionally layered rather than flattened:
  - Linux remains the strongest reviewed source of truth
  - macOS remains a narrower reviewed Apple Clang lane plus supplemental
    static-first install confidence
  - Windows remains the reviewed CMake-first consumer subset and does not
    claim reviewed Makefile or separate reviewed install-validation parity
- The highest-signal rerun set is now fixed for the rest of Sprint 91:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest Day 2 clarification is now fixed:
  - Sprint 91 should read direct-workflow product changes against the direct
    workflow proof owners, not against the reorder/runtime owners that drove
    earlier epic work
  - install/export proof and benchmark reporting remain bounded maintained
    surfaces, not generic product-parity claims
  - later Sprint 91 audit, design, and implementation days should stay
    disciplined about what is executable truth, what is script-owned proof,
    and what is only supplemental workflow evidence

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Rechecked `make -n bench-canonical-report`.
- Rechecked the representative reviewed binaries/examples and the maintained
  install/export, consumer, reporting, and workflow-owner surfaces.

### Day 2 Exit State
- Sprint 91 now has one explicit validation and maintained-surface contract
  before compressed-first implementation work begins.
- The reviewed direct-workflow truth owners, canonical reporting owner,
  install/export proof owners, and workflow-side support evidence are fixed in
  writing.
- Later Day 3-Day 11 audit, design, implementation, and follow-through work no
  longer needs to guess which surfaces are authoritative.

## Day 3 - Remaining Linked-List-First Cost Audit

### Goal
Reduce Sprint 91's broad compressed-first product-model problem to one ranked
live contradiction map centered on the remaining linked-list-first
construction, import/export, publication, and lifecycle costs.

### Actions
- Re-read the Sprint 91 Day 3 contract in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the closest prior structural audit artifact:
  - `docs/planning/EPIC_9/SPRINT_90/artifacts/day3-product-performance-capability-gap-audit.md`
- Re-scanned the live tree against the strongest Sprint 91 contradiction class:
  - linked-list-first construction
  - linked-list-first import/export and publication
  - shell-centric direct-workflow entry paths
  - lifecycle ambiguity on mutated vs solve-ready states
- Re-anchored the audit directly on the current product and implementation
  owners:
  - `README.md`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
  - `include/sparse_analysis.h`
  - `include/sparse_csr.h`
- Rechecked the strongest direct-workflow proof owners likely to matter later:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
- Captured the live hotspot map for those strongest owner surfaces.
- Wrote the Day 3 audit artifact and fixed the ranked shell-cost order in
  writing.

### Findings
- Sprint 91's broad compressed-first problem is now reduced to one ranked live
  map of the highest-value linked-list-first costs:
  - strongest first target:
    - compressed-first construction and import entry points on the public
      matrix shell
  - strongest second target:
    - shell-centric publication and export round-trips that still keep the
      linked-list shell as the default conceptual owner
  - strongest third target:
    - one-shot direct-workflow entry paths that still read as shell-first even
      when the repeated-run direct lifecycle already exists
  - strongest fourth target:
    - lifecycle ambiguity on mutated vs solve-ready shell state and on where
      the long-lived direct-workflow owner really lives
  - strongest support-only but real target:
    - README, maintainer, and public-header wording that still teaches the
      shell as conceptual center instead of bounded mutable compatibility
      surface
- The strongest current contradiction is still the public construction and
  ownership reading:
  - `README.md` still opens by describing the project as an orthogonal
    linked-list sparse matrix library
  - `include/sparse_matrix.h` still describes the public API as the orthogonal
    linked-list sparse matrix shell
  - the same header still keeps the shell as the public mutable sparse
    construction and one-shot direct-workflow compatibility owner
  - `src/sparse_matrix.c` remains a major shell, mutation, and utility owner
- The strongest current owner surfaces are now explicit from the live tree:
  - `include/sparse_matrix.h` = `622`
  - `src/sparse_matrix.c` = `1297`
  - `include/sparse_analysis.h` = `499`
  - `include/sparse_csr.h` = `109`
  - `README.md` = `1113`
  - `tests/test_sparse_matrix.c` = `1136`
  - `tests/test_integration.c` = `3197`
  - `tests/test_chol_csc.c` = `4987`
  - `tests/test_ldlt_csc.c` = `3680`
- The fix-now vs compatibility-only split is now explicit:
  - Sprint 91 should drive:
    - shell-first construction/import costs
    - shell-centric publication/export reading
    - one-shot vs repeated-run direct ownership ambiguity
  - Sprint 91 should keep compatibility-only for now:
    - broad shell removal
    - family-wide direct-API rewrites
    - fake fully compressed-first ownership claims
    - backend, runtime, capability, or packaging widening under Day 3
- The strongest Day 3 clarification is now fixed:
  - Sprint 91 does not begin with generic direct-family cleanup
  - it begins with one ranked shell-cost map
  - the best first implementation center is compressed-first construction and
    import on the public matrix-shell story
  - publication/export and lifecycle tightening follow after that

### Validation
- Re-read the Sprint 91 Day 3 plan contract.
- Re-read the closest prior Sprint 90 structural audit.
- Re-scanned the live product-model owners and strongest direct-workflow proof
  owners.
- Captured the live hotspot map for the strongest likely Sprint 91 surfaces.

### Day 3 Exit State
- Sprint 91 now has one ranked live shell-cost contradiction map grounded in
  the current post-Sprint-90 tree.
- The first compressed-first implementation center is fixed to construction and
  import entry points on the public matrix-shell story.
- Day 4 can freeze the first implementation boundary without reopening the
  ranked shell-cost order.

## Day 4 - First Implementation Boundary

### Goal
Fix one bounded first implementation fence so Sprint 91 starts with the
highest-value compressed-first seam instead of generic product churn.

### Actions
- Re-read the Sprint 91 Day 4 contract in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the Day 3 shell-cost audit against the Sprint 91 project-plan
  contract.
- Re-read the strongest current compressed conversion owner:
  - `include/sparse_csr.h`
- Rechecked the adjacent product and lifecycle wording surfaces:
  - `include/sparse_matrix.h`
  - `README.md`
  - `docs/maintainer_guide.md`
- Decided the required first landing center:
  - compressed-first construction/import seam inside the public matrix-shell
    story
- Fixed the directly forced support-only follow-through set and the explicitly
  later surfaces.
- Wrote the Day 4 boundary artifact and recorded the fence in working notes.

### Findings
- Sprint 91 now has one explicit first implementation fence:
  - required first landing:
    - `include/sparse_csr.h`
    - the matching import/construction implementation seam behind the public
      matrix-shell owner
  - directly forced support surfaces only if the first landing truly needs
    them:
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
    - `README.md`
    - `docs/maintainer_guide.md`
  - explicitly later unless the first landing truly forces movement:
    - publication/export reinterpretation
    - one-shot vs repeated-run lifecycle wording beyond the touched seam
    - `include/sparse_analysis.h`
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt_csc.c`
    - examples and install/export surfaces
- The strongest Day 4 clarification is now fixed:
  - Sprint 91 should start by improving how compressed inputs enter the public
    product model
  - it should not begin by trying to demote or remove the linked-list shell
    broadly
  - it should not reopen broad publication/export ownership or lifecycle
    wording in the first batch unless the construction/import landing actually
    forces it
- The first batch now explicitly defers:
  - broad shell removal
  - family-wide direct-API rewrites
  - public publication/export contract widening as a first-batch center
  - repeated-run direct-owner rewrites centered on `sparse_analysis.h`
  - backend, runtime, capability, or package widening
  - examples, install/export, and workflow churn detached from the first
    product seam

### Validation
- Re-read the Sprint 91 Day 4 plan contract.
- Re-read the Day 3 shell-cost audit.
- Re-scanned the current compressed conversion owner and adjacent product
  surfaces.
- Recorded the boundary and deferral set in the Day 4 artifact.

### Day 4 Exit State
- Sprint 91 has one explicit first implementation boundary.
- The first code landing is fixed to compressed-first construction/import on
  the public matrix-shell story.
- Day 5 can define the architecture contract without reopening the ranked
  first-center choice.

## Day 5 - Compressed-First Architecture Design

### Goal
Define the bounded Sprint 91 contract for compressed-first
construction/import/publication and shell containment before the first code
landing.

### Actions
- Re-read the Sprint 91 Day 5 contract in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the Day 4 boundary fence against the Day 3 shell-cost audit.
- Re-read the strongest current construction/import and shell owners:
  - `include/sparse_csr.h`
  - `include/sparse_matrix.h`
- Rechecked the repeated-run direct owner for contrast:
  - `include/sparse_analysis.h`
- Fixed the future role split for:
  - linked-list shell
  - CSR/CSC-backed construction/import
  - publication/export seams
  - repeated-run direct lifecycle
- Fixed the compatibility-shim policy and the exact Day 6 implementation
  center.
- Wrote the Day 5 architecture artifact and recorded the contract in working
  notes.

### Findings
- Sprint 91 now has one explicit compressed-first product contract:
  - linked-list shell:
    - remains the mutable sparse construction and one-shot direct-workflow
      compatibility surface
    - remains valid for pedagogy, mutation-heavy callers, and compatibility
      one-shot flows
    - stops being treated as the only natural public entry path for callers
      that already have compressed inputs
  - CSC/CSR-backed construction and import:
    - should read as first-class public entry paths for callers that already
      own compressed sparse data
    - should preserve physical-index-space truth and existing compatibility
      semantics
    - should not require broader lifecycle or publication rewrites in the
      first batch
  - public publication/export seams:
    - stay bounded behind the first batch
    - remain real Sprint 91 work, but as the second seam after entry-path
      improvement
- The useful public role split is now explicit:
  - shell-first path:
    - mutation and compatibility one-shot direct workflows
  - compressed-first path:
    - callers that already own CSR/CSC data
  - repeated-run direct path:
    - long-lived symbolic and factor/workspace state through
      `sparse_analysis.h`
- The compatibility policy is now fixed:
  - acceptable to keep:
    - shell-centered one-shot direct APIs
    - shell-centered mutation APIs
    - conversion/export helpers that preserve current behavior while wording
      and ownership are still being tightened
  - should stop being conceptual center stage:
    - the idea that every serious direct or interop workflow must begin by
      mentally adopting the linked-list shell as the primary owner
  - explicitly out of scope for the first landing:
    - broad shell deprecation
    - broad repeated-run owner rewrites
    - family-wide compressed-native API redesign
- The exact Day 6 implementation center is now fixed to:
  - `include/sparse_csr.h`
  - the matching import/construction implementation seam behind the public
    matrix-shell owner
- The strongest Day 5 clarification is now fixed:
  - Sprint 91 should promote compressed inputs to first-class public entry
    paths
  - it should not claim the whole product is already compressed-first
  - it should keep the shell as a bounded mutable compatibility surface while
    removing the strongest unnecessary shell-first conceptual detour

### Validation
- Re-read the Sprint 91 Day 5 plan contract.
- Re-read the Day 4 boundary artifact and Day 3 shell-cost audit.
- Re-scanned the current shell, compressed-import, and repeated-run owner
  surfaces.
- Recorded the architecture contract and Day 6 implementation center in the
  Day 5 artifact.

### Day 5 Exit State
- Sprint 91 now has one explicit compressed-first architecture contract.
- The shell is bounded conceptually even though compatibility remains.
- Day 6 can land the first code batch without reopening product intent.

## Day 6 - Construction / Import Batch

### Goal
Land the first bounded compressed-first implementation seam by promoting CSR
and CSC inputs to first-class public construction entry paths without
reopening broader lifecycle or publication ownership.

### Actions
- Re-read the Sprint 91 Day 6 contract in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the Day 5 architecture design against the Day 4 implementation
  fence.
- Re-scanned the current compressed conversion owner surfaces:
  - `include/sparse_csr.h`
  - `src/sparse_csr.c`
  - `tests/test_csr.c`
- Added new public compressed-first constructor entry points:
  - `sparse_create_from_csr(const SparseCsr *csr)`
  - `sparse_create_from_csc(const SparseCsc *csc)`
- Refactored the retained import implementation into shared validation and
  build helpers so the new constructor path and the legacy `sparse_from_*`
  path use the same core seam.
- Added focused proof-owner coverage for the new constructor-style entry path
  in `tests/test_csr.c`.
- Ran the required implementation-day validation queue:
  - `make format`
  - `make lint`
  - `make test`
- Fixed one local `-Wshadow` lint finding in the new shared helper seam.
- Wrote the Day 6 artifact and recorded the landed batch here.

### Findings
- Sprint 91 now has a real compressed-first public construction lane:
  - `include/sparse_csr.h` now exposes:
    - `sparse_create_from_csr(const SparseCsr *csr)`
    - `sparse_create_from_csc(const SparseCsc *csc)`
  - retained compatibility entry points:
    - `sparse_from_csr(const SparseCsr *csr, SparseMatrix **mat)`
    - `sparse_from_csc(const SparseCsc *csc, SparseMatrix **mat)`
  - the retained compatibility imports now route through the same shared
    validated builder seam as the new constructor-style entry path
- The landing stayed inside the Day 5 fence:
  - required implementation center:
    - `include/sparse_csr.h`
    - `src/sparse_csr.c`
  - directly forced proof-owner follow-through:
    - `tests/test_csr.c`
  - no support-only follow-through was needed in:
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
    - `README.md`
    - `docs/maintainer_guide.md`
- The useful technical split is now real in code:
  - callers that already own CSR/CSC data can enter through direct
    constructor-style public APIs
  - callers that still need explicit `sparse_err_t` handling can keep using
    `sparse_from_*`
  - both lanes preserve the same physical-index-space truth and linked-list
    shell compatibility semantics
- The focused proof follow-through is now explicit:
  - `tests/test_csr.c` proves direct constructor-style CSR entry
  - `tests/test_csr.c` proves direct constructor-style CSC entry
  - `tests/test_csr.c` proves null rejection on the new compressed-first APIs
  - the existing round-trip and SuiteSparse coverage still proves retained
    compatibility on the legacy `sparse_from_*` path

### Validation
- `make format` passed.
- `make lint` passed.
- `make test` passed.
- The only validation interruption was one local `-Wshadow` warning in the new
  shared helper seam; it was fixed immediately and the full queue was rerun
  from the top.

### Day 6 Exit State
- Sprint 91 now has a real first-class compressed-input construction lane.
- The linked-list shell remains the mutable compatibility owner, but it is no
  longer the only public conceptual entry path for CSR/CSC-backed callers.
- Day 7 can rerank the remaining shell-first costs from a landed validated
  implementation rather than from a pure design contract.

## Day 7 - Post-Landing Audit & Rerank

### Goal
Re-rank the remaining compressed-first work after the first code landing so
Sprint 91's second implementation center is chosen from live post-Day-6
evidence rather than from the original shell-cost audit.

### Actions
- Re-read the Sprint 91 Day 7 contract in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the Day 6 landing against:
  - the Day 3 shell-cost audit
  - the Day 5 compressed-first architecture design
- Re-scanned the strongest remaining public and lifecycle owners:
  - `README.md`
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `docs/maintainer_guide.md`
  - likely proof-owner follow-through surfaces:
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
- Re-ranked the remaining shell-first contradiction set from the live
  post-Day-6 tree.
- Fixed the exact Day 8 design center and support-only follow-through map.
- Wrote the Day 7 rerank artifact and recorded the updated order here.

### Findings
- The Day 6 landing closed the strongest first Sprint 91 contradiction:
  - compressed CSR/CSC inputs now have first-class public construction entry
    paths
  - callers that already own compressed sparse data no longer need to begin
    conceptually from `sparse_create()` plus linked-list insertion just to
    enter the matrix-shell workflow
  - the first construction/import seam is no longer the highest-value
    remaining Sprint 91 target
- The ranked remaining shell-cost map is now:
  - strongest first target now:
    - publication and public-surface clarification around the new
      compressed-first entry path
  - strongest second target now:
    - one-shot vs repeated-run direct-workflow lifecycle clarification
  - strongest third target now:
    - focused proof-owner or integration follow-through only if the
      publication and lifecycle contract truly forces it
  - strongest support-only but real target now:
    - README, maintainer, and public-header wording that still over-centers
      the linked-list shell after the Day 6 entry-path landing
- The strongest remaining contradiction is now publication/public-surface
  reading:
  - `README.md` still teaches CSR/CSC conversion as:
    - `sparse_to_csr(mat, &csr)` / `sparse_from_csr(csr, &mat)`
    - `sparse_to_csc(mat, &csc)` / `sparse_from_csc(csc, &mat)`
  - the README still presents the shell-first one-shot path as the more
    natural public center even though compressed-first construction now exists
  - `include/sparse_matrix.h` still truthfully describes the shell as the
    mutable compatibility owner, but the public adoption story around that
    owner has not yet been recalibrated against the new Day 6 entry path
- Lifecycle clarification remains real, but is now second:
  - `include/sparse_analysis.h` already gives the repo a real explicit
    repeated-run direct owner
  - `README.md` already teaches the repeated-run direct workflow
  - the remaining gap is now the relationship between the shell-first and
    compressed-first one-shot entry story and that repeated-run owner
- The exact Day 8 design center is now fixed to:
  - `README.md`
- The strongest support-only follow-through, only if the Day 8 contract truly
  forces movement, is:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `docs/maintainer_guide.md`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
- Sprint 91 no longer needs:
  - a second immediate construction/import implementation batch
  - broad linked-list-shell deprecation
  - a family-wide direct-workflow rewrite
  - proof-owner widening detached from the touched public contract

### Validation
- Re-read the Sprint 91 Day 7 plan contract.
- Re-read the Day 6 artifact against the Day 3 and Day 5 artifacts.
- Re-scanned the strongest remaining public, lifecycle, and proof-owner
  surfaces.
- This was a docs-only rerank pass, so I did not rerun `make format`,
  `make lint`, or `make test`.

### Day 7 Exit State
- The strongest remaining Sprint 91 seam is now explicit after the first
  implementation landing.
- The second implementation center is fixed first to publication/public-surface
  clarification, with lifecycle tightening ordered immediately behind it.
- Day 8 can define one exact bounded publication/lifecycle contract from the
  live post-Day-6 tree.

## Day 8 - Publication & Lifecycle Design

### Goal
Define the bounded second Sprint 91 implementation contract around
publication/public-surface clarification and one-shot vs repeated-run direct
workflow lifecycle clarity.

### Actions
- Re-read the Sprint 91 Day 8 contract in
  `docs/planning/EPIC_9/SPRINT_91/PLAN.md`.
- Re-read the Day 7 rerank against the live public workflow surfaces.
- Re-scanned the strongest current public and lifecycle owners:
  - `README.md`
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `docs/maintainer_guide.md`
  - likely proof-owner follow-through surfaces:
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
- Fixed the exact Day 9 implementation center.
- Fixed the support-only follow-through map and the explicit non-touch list.
- Wrote the Day 8 design artifact and recorded the contract here.

### Findings
- Sprint 91 now has one exact second implementation contract:
  - required Day 9 center:
    - `README.md`
  - directly forced support-only follow-through only if the Day 9 contract
    truly needs them:
    - `include/sparse_matrix.h`
    - `include/sparse_analysis.h`
    - `docs/maintainer_guide.md`
    - `tests/test_sparse_matrix.c`
    - `tests/test_integration.c`
- The key Day 8 reading is now explicit:
  - the Day 6 code already made compressed-first construction real through:
    - `sparse_create_from_csr(...)`
    - `sparse_create_from_csc(...)`
  - the strongest remaining contradiction is that the README still teaches the
    shell-first conversion story as:
    - `sparse_to_csr(mat, &csr)` / `sparse_from_csr(csr, &mat)`
    - `sparse_to_csc(mat, &csc)` / `sparse_from_csc(csc, &mat)`
  - `sparse_analysis.h` already gives the repo a real repeated-run direct
    owner
  - so the highest-value next move is to make the README teach:
    - when compressed-first entry is the right one-shot starting path
    - when the shell-first path is still the right mutable or compatibility
      path
    - when callers should move to the explicit repeated-run direct lifecycle
- The useful support-only follow-through split is now fixed:
  - `include/sparse_matrix.h`
    - only if the Day 9 README contract exposes a real mismatch in how the
      shell role is described
  - `include/sparse_analysis.h`
    - only if the Day 9 README contract exposes a real mismatch in how the
      repeated-run direct owner is described
  - `docs/maintainer_guide.md`
    - only if the Day 9 wording change alters maintainer-facing explanation of
      the public product split
  - `tests/test_sparse_matrix.c`
    - only if the touched README contract creates a real new public-behavior
      claim that needs proof
  - `tests/test_integration.c`
    - only if the touched README contract creates a real lifecycle claim that
      is not already owned by the existing integration proofs
- The explicit Day 9 non-touch list is now fixed:
  - no second construction/import code batch
  - no broad linked-list-shell deprecation
  - no family-wide direct API redesign
  - no repeated-run direct implementation changes
  - no package/install/export contract reopening
  - no iterative/eigensolver workflow rewriting
  - no examples, install scripts, benchmark docs, or CI/workflow churn
    detached from the touched direct-workflow story
- The exact intended Day 9 shape is now explicit:
  - tighten `README.md` around the direct-workflow adoption split
  - make compressed-first one-shot entry read like a real peer lane
  - keep the linked-list shell framed as:
    - mutable construction owner
    - pedagogy/compatibility owner
    - not the only natural public starting point
  - make the handoff from one-shot direct to repeated-run direct smaller and
    clearer

### Validation
- Re-read the Sprint 91 Day 8 plan contract.
- Re-read the Day 7 rerank artifact against the current README and direct
  lifecycle owners.
- Re-scanned the strongest support-only follow-through surfaces.
- This was a docs-only design pass, so I did not rerun `make format`,
  `make lint`, or `make test`.

### Day 8 Exit State
- Day 9 now has one exact bounded publication/lifecycle contract.
- The second batch is still small enough to validate cleanly.
- Broader lifecycle churn remains fenced off behind the Day 9 public-story
  landing.

## Day 9 - Publication Batch

### Intent
- Land the bounded Day 8 public-story batch by tightening `README.md` around
  compressed-first one-shot direct entry, the linked-list shell's retained
  mutable/compatibility role, and the handoff to the explicit repeated-run
  direct lifecycle.

### Actions
- Re-read the Day 8 publication/lifecycle contract and the Day 7 rerank.
- Re-read the README direct-workflow, API overview, and format-conversion
  sections against the landed Day 6 constructor-style CSR/CSC entry points.
- Tightened `README.md` so it now:
  - presents compressed-first one-shot direct entry as a first-class peer lane
  - keeps the linked-list shell framed as the mutable construction and
    compatibility owner
  - makes the repeated-run direct workflow handoff smaller and clearer
  - treats `sparse_create_from_csr(...)` / `sparse_create_from_csc(...)` as
    the primary compressed-first construction APIs while preserving
    `sparse_from_*` as compatibility wrappers when explicit `sparse_err_t`
    status is wanted
- Confirmed that no support-only follow-through was required in:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
  - `docs/maintainer_guide.md`
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
- Wrote the Day 9 artifact and recorded the validated outcome here.

### Findings
- Sprint 91 now has a bounded landed publication/public-surface batch:
  - `README.md` teaches compressed-first one-shot direct entry as a real
    public starting path for callers that already own CSR/CSC inputs
  - the linked-list shell remains explicitly framed as:
    - mutable construction owner
    - pedagogy/compatibility owner
    - not the only natural public starting point
  - the repeated-run direct workflow remains the long-lived direct owner, but
    the handoff from one-shot entry now reads smaller and more intentional
- The landed Day 9 README shape stayed inside the Day 8 fence:
  - touched center:
    - `README.md`
  - directly forced support follow-through:
    - none
- The strongest public-story changes are now explicit:
  - `Choose a Workflow` includes a direct compressed-first one-shot lane
  - `Quick Start` now points compressed-input callers toward the new
    constructor-style entry APIs before widening into repeated-run direct work
  - `Repeated-Run Direct Workflow` now makes the relationship between:
    - compressed-first one-shot entry
    - shell-first mutable compatibility entry
    - explicit repeated-run direct lifecycle
    easier to understand
  - the API overview and format-conversion sections now present:
    - `sparse_create_from_csr(...)`
    - `sparse_create_from_csc(...)`
    as the primary compressed-first construction seams
  - the legacy:
    - `sparse_from_csr(...)`
    - `sparse_from_csc(...)`
    remain documented as compatibility wrappers rather than as the only public
    compressed-input story
- The useful Day 9 no-follow-through call is now explicit:
  - the README contract changed the adoption story, not the live API contract
  - the existing headers and proof owners already remained truthful against
    that narrower public-story clarification
  - no new lifecycle claim was introduced that needed fresh proof ownership

### Validation
- Because this was a substantial public-surface batch, I ran:
  - `make quality-review-full`
- The reviewed queue passed cleanly.
- The reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - reviewed CMake `Total Test time (real)` = `363.62 sec`

### Day 9 Exit State
- Sprint 91 now teaches compressed-first one-shot direct entry as a real peer
  lane without reopening broader lifecycle or proof-surface churn.
- The linked-list shell remains real, but no longer reads like the only
  natural public entry model for compressed-input callers.
- Day 10 can rerank from a landed validated public-story batch rather than
  from a pure design contract.
