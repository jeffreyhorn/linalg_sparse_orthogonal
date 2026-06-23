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

## Day 8 - Index / ABI Follow-Through Design

### Goal
Fix the exact shared-vocabulary and touched-path ABI follow-through contract so
Day 9 can reconcile the post-Day-6 scalar/index reading without reopening
matrix-shell implementation work or widening prematurely into
algorithm-family capability work.

### Actions
- Re-read the Day 7 rerank against the landed shared matrix-shell seam in:
  - `docs/planning/EPIC_8/SPRINT_83/artifacts/day6-scalar-surface-expansion-batch.md`
  - `docs/planning/EPIC_8/SPRINT_83/artifacts/day7-post-landing-audit-and-rerank.md`
- Rechecked the shared scalar/width vocabulary owner in:
  - `include/sparse_types.h`
- Rechecked the strongest authoritative support-only wording surfaces:
  - `README.md`
  - `docs/maintainer_guide.md`
- Rechecked the strongest direct proof-owner surface on the landed seam:
  - `tests/test_sparse_matrix.c`
- Reconfirmed the strongest deferred non-touch public headers:
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

### Findings
- Sprint 83 now has one exact second implementation contract:
  - required Day 9 center:
    - `include/sparse_types.h`
  - strongest support-only proof if the header wording truly forces movement:
    - `tests/test_sparse_matrix.c`
  - strongest support-only wording if the contract truly forces movement:
    - `README.md`
    - `docs/maintainer_guide.md`
  - lower-value non-touch surfaces for this batch:
    - `include/sparse_matrix.h`
    - `include/sparse_qr.h`
    - `include/sparse_svd.h`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
- The exact residual contradiction is now explicit:
  - `include/sparse_types.h` still describes `sparse_scalar_t` primarily as
    the iterative/eigs dense-scalar seam
  - after Day 6, that shared vocabulary owner is the strongest stale touched
    public contract because the matrix-shell helper seam now also routes
    through `sparse_scalar_t`
  - the strongest needed Day 9 move is therefore shared-vocabulary
    reconciliation, not new implementation behavior
- The useful Day 8 clarification is explicit now:
  - Day 9 can stay bounded to shared public header and support-surface
    interpretation, not matrix-shell code churn
  - QR / SVD algorithm-surface widening remains a later seam
  - Cholesky / LDL^T public-family wording remains non-touch unless Day 9
    unexpectedly forces broader scalar-owner interpretation movement
  - package/install/export mechanics remain outside the batch

### Validation
- Re-read the Day 6 and Day 7 artifacts against the live shared vocabulary
  owner.
- Rechecked the strongest proof-owner and support-only wording surfaces that
  would move only if the Day 9 header contract truly forced them.

### Day 8 Exit State
- Sprint 83 now has one exact index / ABI follow-through design contract.
- Day 9 can land one bounded shared-vocabulary reconciliation batch without
  reopening broader capability work.
- Algorithm-family widening remains explicitly after the touched-path shared
  header contract is reconciled.

## Day 9 - Index / ABI Follow-Through Batch

### Goal
Reconcile the shared scalar/width vocabulary owner with the Day 6 landed
matrix-shell scalar seam so the public contract reads consistently without
reopening matrix-shell implementation work or broader family-level capability
surfaces.

### Actions
- Updated the shared scalar/width vocabulary owner in:
  - `include/sparse_types.h`
- Rechecked the strongest direct proof owner on the landed seam:
  - `tests/test_sparse_matrix.c`
- Rechecked the strongest support-only wording surfaces:
  - `README.md`
  - `docs/maintainer_guide.md`
- Preserved the Day 8 non-touch fence by not widening:
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `src/sparse_matrix.c`

### Findings
- The Day 9 landing stayed inside the Day 8 fence:
  - `include/sparse_types.h` no longer describes `sparse_scalar_t` primarily
    as the iterative/eigs public scalar seam
  - it now treats the shared matrix-shell helper seam plus the
    iterative/eigs public scalar seams as the active public-owner surface
  - the shipped scalar contract still remains real-only `double`
- The strongest useful Day 9 clarification is now explicit:
  - the remaining contradiction after Day 6 was shared-vocabulary reading, not
    matrix-shell implementation behavior
  - `SPARSE_SCALAR_BITS` and `sparse_scalar_bits()` now describe the widened
    shared owner truthfully
  - no matrix-shell code churn was needed
- The strongest support-only follow-through was smaller than the design fence:
  - `tests/test_sparse_matrix.c` already owned the strongest direct proof and
    did not need movement
  - `docs/maintainer_guide.md` already matched the post-Day-6 owner split and
    did not need movement
  - `README.md` already remained broadly truthful and did not need movement

### Validation
- Ran `make format`.
- Ran `make lint`.
- Ran `make test`.
- Ran `make quality-review-full`.

### Day 9 Exit State
- Sprint 83 now has one landed shared-vocabulary reconciliation batch.
- The shared scalar owner reads consistently across `include/sparse_types.h`
  and the already-landed matrix-shell helper seam.
- Algorithm-family widening remains the next later Sprint 83 seam rather than
  a dependency of this batch.

## Day 10 - Algorithm-Surface Widening Design

### Goal
Fix the exact family-local capability seam that should move after the Day 6
matrix-shell scalar widening and the Day 9 shared-vocabulary reconciliation,
without reopening broader public-family churn or overstating repo-wide numeric
genericity.

### Actions
- Re-read the landed Sprint 83 scalar-owner surfaces:
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`
- Re-scanned the strongest candidate algorithm-family public seams:
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- Rechecked the strongest likely proof-owner surfaces for a family-local
  follow-through:
  - `tests/test_qr.c`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
- Rechecked the strongest support-only wording surfaces:
  - `README.md`
  - `docs/maintainer_guide.md`

### Findings
- Sprint 83 now has one exact Day 11 algorithm-surface contract:
  - required Day 11 center:
    - `include/sparse_qr.h`
  - strongest support-only proof if the header wording truly forces movement:
    - `tests/test_qr.c`
  - strongest support-only wording if the contract truly forces movement:
    - `README.md`
    - `docs/maintainer_guide.md`
  - lower-value non-touch surfaces for this batch:
    - `include/sparse_svd.h`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
    - `tests/test_svd.c`
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
- The strongest useful Day 10 clarification is now explicit:
  - `include/sparse_qr.h` is the highest-value remaining public algorithm
    seam that still reads directly in raw `double` terms across caller-owned
    vectors, residuals, and helper outputs
  - QR is the strongest next family because it is both a direct public solver
    lane and the clearest shared-algorithm follow-through after the matrix
    shell owner widening
  - `include/sparse_svd.h` is real and still narrower than the new shared
    owner story, but it is lower-value than QR because it is not the first
    caller-facing solve lane that naturally follows the Day 6 / Day 9 work
  - Cholesky and LDL^T remain non-touch because Sprint 83’s widened scalar
    reading still does not require direct-family public rewording yet
- The preserved Day 11 fence is explicit:
  - this is a public-header interpretation batch first, not a family-local
    implementation rewrite in `src/sparse_qr.c`
  - no SVD or direct-family spill should be implied unless the QR header
    wording unexpectedly forces broader contract movement
  - package/install/export mechanics remain outside the batch

### Validation
- Re-read the live Sprint 83 touched public-owner surfaces against the QR,
  SVD, and direct-family public headers.
- Rechecked the strongest proof-owner and support-only wording surfaces that
  would move only if the Day 11 QR contract truly forced them.

### Day 10 Exit State
- Sprint 83 now has one exact algorithm-surface widening design contract.
- Day 11 can land one bounded QR public-header follow-through batch without
  reopening SVD or direct-family work.
- Regression/docs/package alignment remains explicitly after the bounded QR
  follow-through, not inside it.

## Day 11 - Algorithm-Surface Widening Batch

### Goal
Land the bounded QR public-header follow-through fixed on Day 10 so the
highest-value remaining algorithm-family public seam uses the same shared
scalar owner vocabulary as the Sprint 83 matrix-shell and shared-types lanes.

### Actions
- Updated the required Day 11 public algorithm surface in:
  - `include/sparse_qr.h`
- Added focused family-local proof on the widened public seam in:
  - `tests/test_qr.c`
- Updated the authoritative proof-owner reading where the new QR proof now
  matters:
  - `docs/maintainer_guide.md`
- Rechecked and preserved the Day 10 non-touch fence:
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `src/sparse_qr.c`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `README.md`

### Findings
- The Day 11 landing stayed inside the Day 10 fence:
  - `include/sparse_qr.h` no longer exposes the strongest caller-owned QR
    vectors and dense helper outputs entirely as raw `double`
  - the QR public seam now routes the highest-value caller-owned buffers and
    helper outputs through `sparse_scalar_t`
  - the shipped scalar contract still remains real-only `double`
- The strongest useful Day 11 clarification is now explicit:
  - Sprint 83 widened a bounded QR public-owner reading, not QR numeric
    genericity
  - tolerances and condition-estimate interpretation remain real-valued
    diagnostics; Sprint 83 did not widen QR into complex or mixed-precision
    behavior
  - no `src/sparse_qr.c` implementation churn was needed because the widened
    public owner still aliases the shipped real-only `double` lane
- The strongest support-only follow-through was small and exact:
  - `tests/test_qr.c` now owns the QR public scalar seam directly
  - `docs/maintainer_guide.md` now names that proof-owner surface explicitly
  - `README.md` already remained broadly truthful and did not need movement

### Validation
- Ran `make format`.
- Ran `make lint`.
- Ran `make test`.

### Day 11 Exit State
- Sprint 83 now has one landed bounded QR public-header widening batch.
- The highest-value remaining algorithm-family public seam now reads
  consistently with the shared Sprint 83 scalar-owner story.
- SVD and direct-family public follow-through remain explicitly deferred.

## Day 12 - Regression / Docs / Package Alignment

### Goal
Fix the final Sprint 83 proof-owner and Day 13 validation reading after the
Day 11 QR header widening, while keeping support-only and package-sensitive
surfaces bounded to what the sprint actually changed.

### Actions
- Re-read the landed Sprint 83 shared scalar-owner surfaces:
  - `include/sparse_matrix.h`
  - `include/sparse_types.h`
  - `include/sparse_qr.h`
- Rechecked the strongest direct proof-owner surfaces:
  - `tests/test_sparse_matrix.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_qr.c`
- Rechecked deferred family-local proof surfaces to confirm they remain
  truthful without movement:
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`
- Rechecked representative reviewed examples and benchmark/reporting owners:
  - `build/quality-review-cmake/example_analysis`
  - `build/quality-review-cmake/example_basic_solve`
  - `build/quality-review-cmake/bench_svd`
  - `build/quality-review-cmake/bench_refactor_csc`
  - `make bench-canonical-report`
- Rechecked the reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake`

### Findings
- No new support-only edit is needed before the full sweep:
  - `README.md` already remains broadly truthful
  - `docs/maintainer_guide.md` already reflects the widened proof-owner split
  - no additional public-header correction is needed outside the landed
    matrix/types/QR surfaces
- The final Sprint 83 proof-owner map is now explicit:
  - `tests/test_sparse_matrix.c` owns the shared matrix-shell scalar seam and
    shared width contract
  - `tests/test_iterative.c` owns the iterative public scalar seam
  - `tests/test_eigs.c` owns the eigensolver public scalar seam
  - `tests/test_qr.c` owns the bounded QR public scalar seam
  - `tests/test_svd.c` remains the family-local deferred SVD proof surface,
    not a Sprint 83 widened-owner proof target
  - `tests/test_chol_csc.c` and `tests/test_ldlt.c` remain direct-family proof
    surfaces, not Sprint 83 widened-owner proof targets
  - `tests/test_integration.c` remains the cross-feature workflow owner for
    retained public behavior around direct and repeated-run flows
- The representative executable support map is explicit now:
  - reviewed CMake regression owners:
    - `test_sparse_matrix`
    - `test_qr`
    - `test_svd`
    - `test_chol_csc`
    - `test_ldlt`
    - `test_integration`
  - representative examples:
    - `example_analysis`
    - `example_basic_solve`
  - benchmark/reporting owners:
    - `bench_svd`
    - `bench_refactor_csc`
    - `make bench-canonical-report`
- Install/export proof stays explicitly out of scope for Day 13 because Sprint
  83 did not move package, install, export, or reviewed runtime-package
  mechanics.

### Validation
- Rechecked `ctest -N --test-dir build/quality-review-cmake` and confirmed the
  live reviewed parity anchor remains `53`.
- Rechecked the presence of the Day 13 focused reviewed binaries and
  representative examples/benchmarks.
- Rechecked the maintained canonical benchmark-report command surface with
  `make -n bench-canonical-report`.

### Day 12 Exit State
- Sprint 83 now has one final proof-owner and alignment map.
- The exact Day 13 queue is fixed in writing with no remaining validation
  ambiguity.
- No further docs/package follow-through is needed before the full sweep.

## Day 13 - Full Validation Sweep

### Goal
Run the complete Sprint 83 validation queue fixed on Day 12 and capture the
measured close baseline for the widened shared scalar-owner and bounded QR
public-header work.

### Actions
- Ran `make format`.
- Ran `make lint`.
- Ran `make test`.
- Ran `make quality-review-full`.
- Rechecked the reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake`
- Ran the Day 12 focused reviewed proof-owner follow-ons:
  - `./build/quality-review-cmake/test_sparse_matrix`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
  - `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `make bench-canonical-report`

### Findings
- The full Sprint 83 implementation-day gate passed:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- The maintained reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 446.47 sec`
- The Day 12 focused reruns all passed:
  - `test_sparse_matrix` -> `59 / 59`
  - `test_qr` -> `73 / 73`
  - `test_svd` -> `97 / 97`
  - `test_chol_csc` -> `149 / 149`
  - `test_ldlt` -> `87 / 87`
  - `test_integration` -> `53 / 53`
  - `example_analysis`
  - `example_basic_solve`
  - `bench_svd tests/data/suitesparse/nos4.mtx`
  - `bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `make bench-canonical-report`
- Representative retained outputs stayed clean:
  - `test_sparse_matrix` retained the new shared-owner proof:
    - `test_matrix_public_scalar_alias`
    - `test_idx_width_contract`
  - `test_qr` retained the widened QR public-owner proof:
    - `test_qr_public_scalar_alias`
    - `nos4 QR solve: rank=100`
    - `nos4 QR solve: res_norm=0.000e+00, true_res=9.415e-15`
  - `test_svd` retained deferred-family truthfulness:
    - `outer-product vs dense: ||A_off - A_on||_F / ||A_off||_F = 0.000e+00`
    - `full-mode recon: ||A - U Sigma Vt||_F / ||A||_F = 9.648e-16`
  - `test_chol_csc` retained direct-family stability:
    - `tests/data/suitesparse/bcsstk14.mtx: n=1806, rel_residual=1.080e-15`
  - `test_ldlt` retained direct-family stability:
    - `test_ldlt_dense_backend_accelerate_accepts_noperm_2x2`
    - `KKT 500x500: relres=4.465e-17, nnz(L)=1298`
  - `test_integration` retained the repeated-run/public-lifecycle surface:
    - `53 / 53`
  - `example_analysis` retained solve residual `4.44e-16`
  - `example_basic_solve` retained residual `0.00e+00`
  - `bench_svd nos4` retained:
    - `Full SVD (σ only): 6.612 ms`
    - `Partial SVD (k=5, σ): 2.170 ms`
    - `Partial/Full: 3.0x speedup`
  - `bench_refactor_csc nos4` retained:
    - `speedup_refactor = 1.40`
    - residuals `8.24e-16` / `7.06e-16`
  - `make bench-canonical-report` retained the canonical bundle write:
    - `bench_refactor_csc.csv`
    - `bench_chol_csc.csv`
    - `bench_iterative_reuse.csv`
    - `bench_eigs_reuse.csv`
    - `index.tsv`
    - `manifest.txt`
- Install/export proof remained intentionally out of scope for Day 13 because
  Sprint 83 did not move package, install, export, or runtime-package
  mechanics.
- One non-blocking runtime note is explicit in the measured baseline:
  - reviewed CMake `test_reorder_nd` still dominated runtime at `314.43 sec`
    out of the `446.47 sec` total

### Validation
- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `make quality-review-full` passed.
- `ctest -N --test-dir build/quality-review-cmake` retained `53`.
- Reviewed CMake `ctest` passed `53 / 53`.

### Day 13 Exit State
- Sprint 83 now has one measured Day 13 close baseline.
- The widened shared-owner and bounded QR public-header surfaces passed
  together with retained deferred-family and direct-family proof.
- Day 14 can close from validated evidence rather than from intermediate
  implementation state.

## Day 14 - Closeout and Handoff

### Goal
Close Sprint 83 from the validated Day 13 baseline and leave one explicit
handoff queue for Sprint 84 and the later Epic 8 capability and assurance
sprints.

### Actions
- Re-read the landed Sprint 83 package across:
  - Day 3 capability rerank
  - Day 5 scalar/index architecture design
  - Day 6 shared matrix-shell scalar-owner widening
  - Day 9 shared vocabulary reconciliation
  - Day 11 bounded QR public-header widening
  - Day 13 full validation sweep
- Rechecked the Sprint 83 section in
  `docs/planning/EPIC_8/PROJECT_PLAN.md`.
- Rechecked the immediate next Epic 8 handoff section:
  - Sprint 84: Numerical Assurance & Differential Testing Phase 2

### Findings
- Sprint 83 now closes as one coherent Epic 8 capability-surface
  modernization package across:
  - capability re-rank
  - bounded scalar/index architecture contract
  - Day 6 shared matrix-shell scalar-surface expansion
  - Day 9 shared scalar/index vocabulary reconciliation
  - Day 11 bounded QR public-header widening
  - validated Day 13 close baseline
- The preserved fence stayed intact:
  - the shipped scalar contract still remains real-only `double`
  - Sprint 83 widened the public owner reading through `sparse_scalar_t` and
    `idx_t`, not broad numeric genericity
  - no SVD public-header widening was reopened inside Sprint 83
  - no Cholesky or LDL^T public-header capability widening was reopened inside
    Sprint 83
  - no true complex-scalar or mixed-precision claim was widened
  - no package, install, export, or runtime-package claim was widened beyond
    the untouched mechanics
- `docs/planning/EPIC_8/PROJECT_PLAN.md` does not need a Sprint 83
  correction.
- The landed Sprint 83 package still supports the intended Epic 8 execution
  order:
  1. Sprint 84: stronger external differential, seeded-property, and
     failure-path assurance on the touched shared/direct lanes
  2. Sprint 85: maintainability work after the widened capability reading and
     assurance surface are stable
  3. later SVD/direct-family capability widening, true complex support, mixed
     precision, and broader package/ABI/runtime maturity only where bounded
     evidence justifies the next move

### Validation
- No new validation was required on Day 14.
- Sprint 83 closes from the Day 13 validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 446.47 sec`
  - `./build/quality-review-cmake/test_sparse_matrix` -> `59 / 59`
  - `./build/quality-review-cmake/test_qr` -> `73 / 73`
  - `./build/quality-review-cmake/test_svd` -> `97 / 97`
  - `./build/quality-review-cmake/test_chol_csc` -> `149 / 149`
  - `./build/quality-review-cmake/test_ldlt` -> `87 / 87`
  - `./build/quality-review-cmake/test_integration` -> `53 / 53`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
  - `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `make bench-canonical-report`

### Day 14 Exit State
- Sprint 83 achieved its purpose: the project now has one proof-backed shared
  scalar-owner widening on the matrix shell, one reconciled shared
  scalar/index vocabulary owner, one bounded QR public-header widening, and
  one validated close baseline.
- Sprint 84 can now expand numerical assurance on top of a clearer capability
  contract instead of reopening the same scalar-owner contradiction first.
