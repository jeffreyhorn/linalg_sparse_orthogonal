# Sprint 83 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 83 baseline for Epic 8 by grounding the sprint in
the validated Sprint 82 close state, the live Sprint 83 project-plan section,
and the current capability, index-width, proof-owner, benchmark, and
support-surface seams rather than another generic capability restart.

### Actions
- Re-read the Sprint 83 section of `docs/planning/EPIC_8/PROJECT_PLAN.md` and
  the full Sprint 83 day-by-day plan in
  `docs/planning/EPIC_8/SPRINT_83/PLAN.md`.
- Re-read the strongest Sprint 82 closeout context:
  - `docs/planning/EPIC_8/SPRINT_82/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_8/SPRINT_82/RETROSPECTIVE.md`
- Rechecked the maintained reviewed wrapper surface with:
  - `make -n quality-review-full`
- Re-materialized the reviewed CMake parity tree with:
  - `make quality-review-cmake-compile`
- Reconfirmed the reviewed parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 83
  touch surfaces across shared types, matrix/public seams, solver-family
  headers, implementation owners, proof-owner tests, and support surfaces.
- Opened Sprint 83 working notes and fixed the intended Day 1 and Day 2
  landing order, artifacts, and validation expectations in writing.

### Findings
- Sprint 83 starts from the same strongest local reviewed baseline Sprint 82
  closed on:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit before any Sprint 83 implementation
  work:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- Sprint 83 is not a generic “add more math types” sprint. Its highest value
  is one bounded capability-surface modernization package centered on:
  - capability re-rank
  - scalar / index architecture design
  - first scalar-surface expansion on the highest-value public seams
  - touched-path index / ABI follow-through
  - one bounded algorithm-surface widening lane
  - focused regression / docs / package alignment only where the
    implementation truly moves the contract
- The strongest likely Sprint 83 capability, proof, and support surfaces are
  explicit from the live tree:
  - `include/sparse_types.h` = `310`
  - `include/sparse_matrix.h` = `614`
  - `include/sparse_qr.h` = `385`
  - `include/sparse_svd.h` = `257`
  - `include/sparse_cholesky.h` = `227`
  - `include/sparse_ldlt.h` = `335`
  - `src/sparse_types.c` = `54`
  - `src/sparse_matrix.c` = `1295`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_svd.c` = `1319`
  - `src/sparse_chol_csc.c` = `1841`
  - `src/sparse_ldlt.c` = `1535`
  - `tests/test_sparse_matrix.c` = `1104`
  - `tests/test_qr.c` = `3197`
  - `tests/test_svd.c` = `2766`
  - `tests/test_chol_csc.c` = `4787`
  - `tests/test_ldlt.c` = `2921`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_svd.c` = `180`
  - `README.md` = `1050`
  - `docs/maintainer_guide.md` = `710`
- The strongest Day 1 clarification is now fixed:
  - Sprint 83 should not reopen Sprint 82's dense/backend ABI fence as its
    first implementation center
  - Sprint 83 should not claim repo-wide complex, mixed-precision, or package
    maturity before one bounded seam truly lands
  - it should first reduce the current real-only and compile-time-bounded
    capability ceiling on the highest-value public seams only
- The preserved Sprint 83 non-goal pressure is explicit before Day 2:
  - no broad backend-framework reopening
  - no repo-wide complex-number promise
  - no broad mixed-precision framework
  - no generic package/platform maturity claim widening
  - no algorithm-family widening before the shared scalar/index contract is
    explicit
  - no benchmark-governance drift or support-surface churn detached from a real
    landed capability seam

### Validation
- Rechecked `make -n quality-review-full`.
- Re-ran `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor at
  `ctest -N --test-dir build/quality-review-cmake` = `53`.
- Captured the live shared-type, matrix/public, solver-family, proof-owner,
  benchmark, and support-surface hotspot map from direct `wc -l` measurement.

### Day 1 Exit State
- Sprint 83 no longer starts from generic Epic 8 capability prose.
- The baseline, capability rerank, scalar/index design, first public-seam
  widening, touched-path width/ABI follow-through, bounded algorithm widening,
  focused proof, and package-alignment workstreams are fixed in writing.
- The strongest likely Sprint 83 touch surfaces are explicit before the
  validation/proof recheck begins.

## Day 2 - Validation and Proof-Surface Recheck

### Goal
Reconfirm the Sprint 83 implementation-day validation contract and the live
proof-surface split across reviewed CMake proof owners, representative
examples, benchmark/report command surfaces, and install/export proof owners
before any capability-surface batch lands.

### Actions
- Re-read the Sprint 83 Day 2 plan expectations in
  `docs/planning/EPIC_8/SPRINT_83/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor directly with:
  - `ctest -N --test-dir build/quality-review-cmake`
- Rechecked the strongest reviewed proof-owner binaries and representative
  examples most likely to matter early in Sprint 83:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the strongest reviewed benchmark follow-on binaries most likely to
  matter:
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
- Rechecked the maintained canonical report command surface with:
  - `make -n bench-canonical-report`
- Reconfirmed the maintained install/package proof owners:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 83 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 83 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial capability, width/ABI, algorithm-surface, or package/runtime
    batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the strongest early-Sprint-83 proof
  surfaces:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
- The canonical benchmark/reporting lane remains command- and script-owned
  rather than reviewed-binary-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest current proof and truth-surface split is now fixed for Sprint
  83's first lane:
  - reviewed CMake proof-owner tests and representative examples remain the
    main executable truth surfaces
  - reviewed benchmark binaries remain benchmark-side measurability surfaces
  - canonical benchmark reporting remains command/script owned
  - install/export proof remains script owned
- The highest-signal Sprint 83 rerun set is now fixed around the likely touched
  capability surfaces:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_svd`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most
  likely to matter early in Sprint 83.
- Rechecked the strongest reviewed benchmark follow-on binaries.
- Rechecked `make -n bench-canonical-report`, the root `build/` canonical
  emitters it consumes, and the maintained install/export proof scripts.

### Day 2 Exit State
- Sprint 83 now has one explicit implementation-day validation contract.
- The live proof split across reviewed binaries, command-owned canonical
  reporting, and script-owned install/export proof is fixed in writing.
- The highest-signal rerun set is explicit before the capability re-rank audit
  begins.

## Day 3 - Capability Re-rank Audit

### Goal
Reduce Sprint 83's broad capability problem to one ranked live contradiction
map grounded in the current shared type, public matrix, solver-family,
proof-owner, and support-surface seams so later boundary and architecture work
can choose one bounded capability lane instead of another generic “more types”
bucket.

### Actions
- Re-read the Sprint 83 Day 3 plan expectations in
  `docs/planning/EPIC_8/SPRINT_83/PLAN.md`.
- Re-read the strongest prior Epic 8 capability framing in:
  - `docs/planning/EPIC_8/SPRINT_80/artifacts/day3-live-competitive-gap-inventory.md`
  - `docs/planning/EPIC_8/SPRINT_82/artifacts/day14-closeout-and-handoff.md`
- Re-scanned the highest-signal public and family-local capability surfaces:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `src/sparse_matrix.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
- Re-scanned the current maintainer/doc interpretation surfaces:
  - `docs/maintainer_guide.md`
  - `README.md`
- Reconciled the scan against the already-landed bounded public scalar seam in:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

### Findings
- Sprint 83's broad capability problem is now reduced to one ranked live
  contradiction map:
  - strongest first target:
    - shared public dense-scalar owner expansion on the highest-value matrix
      shell and one-shot solver seams
  - strongest second target:
    - touched-path wider-index and package/ABI maturity on shared public paths
  - strongest third target:
    - QR / SVD algorithm-surface widening after the shared scalar/index
      contract is explicit
  - strongest fourth target:
    - true complex-scalar support
  - strongest fifth target:
    - broad mixed-precision support
  - strongest support-only but real target:
    - proof, docs, and package wording that still reflects the narrower
      current capability reading
- `include/sparse_types.h` already carries one bounded public preparation seam:
  - `sparse_scalar_t` and `SPARSE_SCALAR_BITS` exist
  - the shipped scalar contract still remains real-only `double`
  - `SPARSE_IDX_BITS` already makes width a compile-time contract rather than a
    hand-edited typedef story
- The strongest first contradiction is not “no scalar seam exists.” It is that
  the seam is still unevenly owned:
  - iterative and eigensolver public contracts already route through
    `sparse_scalar_t`
  - the highest-value shared and one-shot public seams still expose raw
    `double` buffers and result fields:
    - `include/sparse_matrix.h`
    - `include/sparse_qr.h`
    - `include/sparse_svd.h`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
  - `src/sparse_matrix.c` still acts as the shared compatibility shell owner
    beneath many of those public seams, so it remains the strongest first
    implementation center rather than a later support-only surface
- The strongest second contradiction is index-width maturity on touched public
  paths, not index-width absence:
  - the repo already has compile-time-selected `idx_t`
  - the reviewed build still defaults to the 32-bit lane
  - touched public structs, count-sensitive buffers, and package-visible width
    readings still need stronger consistency if Sprint 83 widens the shared
    capability contract at all
- The strongest third contradiction is family-local algorithm breadth on the
  QR / SVD lane:
  - `include/sparse_qr.h` and `include/sparse_svd.h` still publish owned
    factor/result buffers and helper interfaces almost entirely in raw `double`
  - that makes QR / SVD the strongest bounded algorithm-family widening lane
    once the shared scalar/index contract is explicit
  - Cholesky and LDL^T remain real follow-through surfaces, but they read more
    like support-only one-shot compatibility lanes than the best first
    algorithm-widening center
- True complex-scalar support and broad mixed precision remain lower-value
  first moves:
  - both would force much broader proof, algorithm, and package claims
  - both would outrun the current bounded maintainer reading in
    `docs/maintainer_guide.md`
  - both remain real later capability lanes, but not the first credible Sprint
    83 implementation center
- The strongest Day 3 clarification is now explicit:
  - the best first Sprint 83 move is not broad complex support
  - it is one bounded widening of the already-real `sparse_scalar_t` /
    `idx_t` ownership story across the highest-value shared and one-shot public
    seams
  - touched-path wider-index and ABI maturity follows next
  - QR / SVD capability breadth follows after the shared contract, not before

### Validation
- Re-read the Sprint 80 and Sprint 82 capability-handoff context directly.
- Re-scanned the live public/shared headers, implementation owners, and
  maintainer/doc interpretation surfaces directly.
- Reconciled the ranked Sprint 83 lane against the already-landed
  `sparse_scalar_t` public seam in iterative/eigs so the audit reflects the
  current tree rather than a generic prior-state reading.

### Day 3 Exit State
- Sprint 83 no longer has a generic “capability modernization” problem.
- The strongest first implementation center is fixed to shared public
  scalar/index ownership on the highest-value seams.
- Wider-index maturity, QR / SVD widening, and later complex/mixed-precision
  work are now clearly separated by value and risk before the Day 4 boundary
  freeze begins.

## Day 4 - First Capability Boundary Freeze

### Goal
Fix the first bounded Sprint 83 implementation fence so the next design pass
can define one real scalar/index contract instead of another broad capability
rewrite.

### Actions
- Re-read the Sprint 83 Day 4 plan expectations in
  `docs/planning/EPIC_8/SPRINT_83/PLAN.md`.
- Re-read the Day 3 capability rerank against the strongest live public/shared
  seams.
- Rechecked the prior bounded-boundary pattern in
  `docs/planning/EPIC_8/SPRINT_82/artifacts/day4-first-backend-boundary.md`
  so Sprint 83's fence stays equally explicit.
- Re-separated:
  - required first landing surfaces
  - support-only surfaces that move only if the first landing truly forces
    them
  - explicitly deferred implementation lanes and claims

### Findings
- Sprint 83 now has one explicit first implementation fence:
  - required first landing:
    - `include/sparse_types.h`
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
  - support only if the first landing truly forces it:
    - `include/sparse_qr.h`
    - `include/sparse_svd.h`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
    - `tests/test_sparse_matrix.c`
    - `tests/test_qr.c`
    - `tests/test_svd.c`
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `README.md`
    - `docs/maintainer_guide.md`
  - explicitly deferred from the first landing:
    - `src/sparse_qr.c`
    - `src/sparse_svd.c`
    - `src/sparse_chol_csc.c`
    - `src/sparse_ldlt.c`
    - broad algorithm-family widening as a first-batch center
    - true complex-scalar support
    - broad mixed-precision support
    - generic package/platform maturity widening
- The useful Day 4 clarification is now explicit:
  - the best first Sprint 83 move is the shared public scalar/index owner on
    the matrix shell and its highest-value compatibility seams
  - touched-path wider-index and ABI maturity remains the strongest second seam
  - QR / SVD family-local capability widening remains real, but explicitly
    later than the first shared contract landing
  - proof and support surfaces stay support-only unless the first landing truly
    changes behavior there
- The preserved first-batch non-goal fence is explicit now:
  - no repo-wide complex-number promise
  - no broad mixed-precision framework
  - no ABI churn detached from touched public seams
  - no algorithm-family widening before the shared contract is explicit
  - no benchmark-governance drift
  - no support-surface churn detached from a real landed capability seam

### Validation
- Re-read the Day 3 capability rerank directly.
- Rechecked the current shared/public and solver-family seam split against the
  Sprint 83 project-plan scope.
- Rechecked the prior Sprint 82 boundary artifact to keep the Sprint 83 fence
  equally bounded and explicit.

### Day 4 Exit State
- Sprint 83 now has one bounded first capability landing center.
- Day 5 can design one scalar/index architecture contract inside that fence.
- Lower-value QR/SVD family breadth, later complex/mixed-precision work, and
  broader support/package spillover are held back until later lanes.

## Day 5 - Scalar / Index Architecture Design

### Goal
Define the bounded scalar/index contract that Sprint 83 will actually land on
the shared matrix-shell and public-owner lane.

### Actions
- Re-read the Sprint 83 Day 5 plan expectations in
  `docs/planning/EPIC_8/SPRINT_83/PLAN.md`.
- Re-read the Day 4 boundary against the current shared/public owner seams in:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- Rechecked the current bounded maintainer-policy reading in:
  - `docs/maintainer_guide.md`
- Rechecked the current public/shared scalar split against:
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

### Findings
- Sprint 83 now has one explicit first implementation contract:
  - required implementation center:
    - `include/sparse_types.h`
    - `include/sparse_matrix.h`
    - `src/sparse_matrix.c`
  - support only if the first batch truly forces it:
    - `include/sparse_qr.h`
    - `include/sparse_svd.h`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
    - `tests/test_sparse_matrix.c`
    - `tests/test_qr.c`
    - `tests/test_svd.c`
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `README.md`
    - `docs/maintainer_guide.md`
- The Day 5 ownership split is now fixed:
  - shared scalar and width vocabulary owner:
    - `include/sparse_types.h`
  - public matrix-shell exposure owner:
    - `include/sparse_matrix.h`
  - compatibility-preserving implementation and publication owner:
    - `src/sparse_matrix.c`
  - family-level adoption follow-through owners, but not in the first batch:
    - `include/sparse_qr.h`
    - `include/sparse_svd.h`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
- The useful Day 5 clarification is explicit now:
  - the first landing should preserve the shipped scalar contract as real-only
    `double` even while widening ownership onto the shared public seams
  - it should widen the shared matrix-shell/public-owner reading to use the
    already-real `sparse_scalar_t` / `idx_t` vocabulary where that can be done
    without implying broad numeric genericity
  - it should keep compatibility-preserving internal representation and
    publication behavior centered in `src/sparse_matrix.c` rather than widening
    immediately into family-local algorithm code
  - it should not reopen QR, SVD, Cholesky, LDL^T, true complex support, broad
    mixed precision, or generic package/platform maturity in the same batch
- The preserved first-batch fence is explicit:
  - current callers should keep a truthful real-only reading
  - width remains a compile-time contract, not a runtime-generic claim
  - no repo-wide scalar genericity claim
  - no benchmark, install/export, or package wording drift unless the touched
    public contract truly forces it

### Validation
- Re-read the Day 4 boundary directly.
- Rechecked the current shared vocabulary, matrix-shell owner, and existing
  iterative/eigs scalar seam directly.
- Reconciled the Day 5 contract against the current maintainer-policy wording
  so the design stays inside the shipped capability claim.

### Day 5 Exit State
- Sprint 83 now has one bounded scalar/index architecture contract.
- Ownership between shared vocabulary, public matrix exposure, and
  compatibility-preserving implementation is fixed before Day 6 begins.
- Family-local capability widening remains explicitly outside the first batch.

## Day 6 - Scalar-Surface Expansion Batch

### Goal
Land the first bounded Sprint 83 capability batch by widening the shared
matrix-shell public seam to the already-real `sparse_scalar_t` vocabulary
without widening the shipped scalar contract beyond real-only `double`.

### Actions
- Updated the shared matrix-shell public contract in:
  - `include/sparse_matrix.h`
- Updated the compatibility-preserving implementation owner in:
  - `src/sparse_matrix.c`
- Added focused proof for the widened shared scalar seam in:
  - `tests/test_sparse_matrix.c`
- Reconciled the authoritative maintainer-policy reading in:
  - `docs/maintainer_guide.md`
- Preserved the first-batch fence by not widening:
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt.c`
  - `README.md`

### Findings
- The Day 6 landing stayed inside the Day 5 fence:
  - the shared matrix-shell helper/public-owner seam now uses
    `sparse_scalar_t` on its caller-facing dense-scalar paths
  - the shipped scalar contract remains real-only `double` because
    `sparse_scalar_t` remains that exact underlying type
  - no family-local QR, SVD, Cholesky, or LDL^T implementation widening was
    needed
- The highest-value shared matrix-shell public seam is now widened across:
  - insert/get/set helpers
  - symmetry tolerance input
  - norm output
  - matvec / block-matvec vectors
  - scale and add helpers
- `src/sparse_matrix.c` remains the compatibility-preserving publication owner:
  - behavior is unchanged
  - internal representation remains truthful to the shipped real-only scalar
    contract
  - the batch is vocabulary widening on the shared seam, not numeric-generic
    behavior widening
- The focused proof owner now covers the landed seam directly:
  - `tests/test_sparse_matrix.c` now proves the shared matrix-shell public
    scalar alias through `sparse_scalar_t`, `sparse_scalar_bits()`,
    `sparse_insert`, `sparse_matvec`, `sparse_norminf`, and `sparse_scale`
- The strongest required support-only follow-through was bounded:
  - `docs/maintainer_guide.md` now treats `sparse_scalar_t` as the dense
    scalar owner on the shared matrix-shell helper seam as well as the already
    landed iterative/eigs seam
  - `README.md` did not need movement because its broader capability wording
    remained truthful after the batch

### Validation
- Ran `make format`.
- Ran `make lint`.
- Ran `make test`.

### Day 6 Exit State
- Sprint 83 now has one landed shared scalar-surface expansion batch.
- The highest-value shared matrix-shell seam no longer reads as a raw `double`
  outlier relative to the already-real `sparse_scalar_t` vocabulary.
- Family-local capability widening remains explicitly deferred to later sprint
  days.

## Day 7 - Post-Landing Audit and Rerank

### Goal
Re-rank the strongest remaining Sprint 83 capability contradiction after the
Day 6 shared matrix-shell scalar-surface landing so Day 8 can design one
bounded follow-through batch instead of drifting into premature
algorithm-family widening.

### Actions
- Re-read the Day 6 landing against the Day 5 scalar/index architecture
  contract in:
  - `docs/planning/EPIC_8/SPRINT_83/artifacts/day5-scalar-index-architecture-design.md`
  - `docs/planning/EPIC_8/SPRINT_83/artifacts/day6-scalar-surface-expansion-batch.md`
- Rechecked the shared scalar/width vocabulary owner in:
  - `include/sparse_types.h`
- Rechecked the landed shared matrix-shell public seam in:
  - `include/sparse_matrix.h`
- Rechecked the current authoritative maintainer-policy reading in:
  - `docs/maintainer_guide.md`
- Rechecked the broader package-visible capability wording in:
  - `README.md`
- Re-scanned the strongest deferred family-local public surfaces:
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

### Findings
- The Day 6 landing closed the strongest first contradiction:
  - the shared matrix-shell public seam no longer reads as a raw-`double`
    outlier
  - a second immediate matrix-shell scalar batch is not the highest-value next
    move
- The strongest remaining Sprint 83 seam is now the touched-path index / ABI
  follow-through lane:
  - the shared scalar/width vocabulary owner in `include/sparse_types.h`
    still reads primarily as the iterative/eigs public seam even though Day 6
    widened the matrix-shell seam
  - that makes the strongest residual contradiction one shared-vocabulary and
    package-visible interpretation gap, not a missing second matrix-shell code
    batch
- The exact Day 8 design center is now fixed to:
  - `include/sparse_types.h`
- The strongest support-only follow-through is now:
  - `tests/test_sparse_matrix.c`
  - `README.md`
  - `docs/maintainer_guide.md`
- The strongest Day 7 clarification is explicit now:
  - QR / SVD algorithm-surface widening remains real, but it is later than the
    touched-path index / ABI follow-through
  - Cholesky / LDL^T public-family widening remains support-only and still
    does not justify reopening direct-family implementation work here
  - package/install/export mechanics still do not have their own Sprint 83
    landing absent a real touched public-contract move

### Validation
- Re-read the Day 5 and Day 6 artifacts against the live tree.
- Rechecked the shared scalar/width vocabulary owner, the landed matrix-shell
  public seam, the maintained proof-owner split, and the deferred
  family-local public headers directly.

### Day 7 Exit State
- Sprint 83's next contradiction center is now explicit after the Day 6
  landing.
- Day 8 can stay bounded to the index / ABI follow-through design lane.
- Algorithm-family widening remains intentionally deferred until after the
  touched-path shared-vocabulary seam is reconciled.
