# Sprint 64 Retrospective

**Sprint:** 64 — Performance Backend Architecture Phase 1  
**Duration:** 14 days (Days 1-14)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 64 baseline and scope captured from the Sprint 63 validated Epic 6 state
- [x] reviewed validation/truthfulness baseline rechecked before backend-sensitive code landed
- [x] dense-kernel, supernodal, build-option, and benchmark-proof seams reduced to a ranked hotspot audit
- [x] first backend-aware landing boundary fixed before implementation changes began
- [x] bounded backend abstraction contract designed before code moved
- [x] build/option surface design fixed without widening the public runtime surface unnecessarily
- [x] exact first code-batch and proof fence fixed before kernel integration landed
- [x] first backend-aware Cholesky CSC supernodal dense-kernel seam landed behind an internal descriptor
- [x] fallback/error-path truthfulness landed through `SPARSE_ERR_BACKEND_CONTRACT`
- [x] benchmark proof surface refreshed with explicit path-identification fields
- [x] public/header/README/maintainer surfaces updated to match the landed backend-aware contract
- [x] full validation sweep completed from the final landed Sprint 64 tree
- [x] Sprint 64 closeout and handoff completed from the validated baseline

## What Went Well

1. **Sprint 64 stayed bounded to one real Phase 1 backend lane instead of drifting into a fake general framework.**
   The sprint kept the implementation center of gravity on:
   - the Cholesky CSC supernodal dense-kernel lane
   - one bounded internal descriptor seam
   - one public error-taxonomy completion
   - one benchmark proof refresh
   - one final documentation and maintainer interpretation pass
   It did not widen into:
   - repo-wide pluggable backend infrastructure
   - broad build-system redesign
   - packaging/platform work
   - QR/SVD backend layering
   - benchmark-governance sprawl
   That scope discipline is why Sprint 64 closed as a coherent Phase 1
   backend sprint instead of as another vague architecture-planning pass.

2. **The broad Epic 6 “performance/backend architecture” claim turned into a concrete landed seam.**
   Sprint 64 did not stop at hotspot discussion. It shipped a real internal
   backend-aware descriptor seam for the highest-value selected path:
   - the CSC supernodal Cholesky lane now resolves its dense helpers through
     a bounded internal descriptor
   - the builtin default descriptor remains the authoritative self-contained
     implementation
   - the selected hot path no longer assumes an always-hardwired local helper
     arrangement
   That is a meaningful architecture change, not just cleaner comments around
   existing code.

3. **The sprint chose the right first lane.**
   The Day 3-4 hotspot audit correctly fixed the first landing on:
   - `src/sparse_chol_csc_supernodal.c`
   rather than:
   - LDL^T supernodal first
   - a repo-wide `src/sparse_dense.c` rewrite
   - build-option work as the design center
   - broad QR/SVD dense unification
   That choice kept runtime relevance, touched-surface boundedness, and proof
   burden aligned.

4. **The fallback and error-path story is more truthful at sprint close than it was at sprint start.**
   Sprint 64 did not leave the new backend-aware seam hiding behind
   `SPARSE_ERR_BADARG`. It completed the public taxonomy with:
   - `SPARSE_ERR_BACKEND_CONTRACT`
   and proved the bounded contract directly for:
   - missing descriptor
   - missing factor callback
   - missing solve callback
   That makes the first backend-aware lane read like a real shipped contract
   instead of an implicit implementation assumption.

5. **The benchmark proof surface now shows what backend-aware path actually ran.**
   `bench_chol_csc` was already a useful timing surface before Sprint 64, but
   it did not identify the active dense-kernel descriptor. Sprint 64 fixed
   that by adding:
   - `csc_scalar_path`
   - `csc_supernodal_path`
   - `csc_supernodal_dense_kernel`
   The default build now reports:
   - `scalar`
   - `supernodal`
   - `builtin`
   That is a much better observability state for later backend work.

6. **The public/header/maintainer story stayed aligned with the implementation instead of drifting into broad backend marketing.**
   Sprint 64’s Day 12 follow-through was appropriately narrow:
   - `include/sparse_cholesky.h` now documents the actual
     `SPARSE_ERR_BACKEND_CONTRACT` call-site contract
   - `README.md` ties the Cholesky CSC dispatch story to the refreshed
     benchmark proof surface
   - `docs/maintainer_guide.md` now owns the bounded interpretation that
     Sprint 64 is not a general backend framework
   That is a better documentation outcome than a broad architecture narrative
   unsupported by proof.

7. **The sprint still closed from the strongest reviewed baseline.**
   Day 13 passed:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
   and preserved the maintained reviewed anchors:
   - reviewed CMake parity `53`
   - Makefile/CMake parity `53 vs 53`
   - full reviewed CMake `ctest` `53 / 53`
   - full reviewed CMake total real time `574.42 sec`
   That keeps the Sprint 64 handoff credible as real backend/productization
   work rather than only a local performance experiment.

8. **Sprint 64 ended with a smaller and more explicit backend queue.**
   The sprint did not just land one Cholesky seam. It clarified what remains:
   - LDL^T CSC supernodal backend-aware follow-through
   - bounded shared dense-kernel seam reuse only where it reduces real
     duplicate risk
   - optional build-option or pluggable-kernel widening only if the
     self-contained default-path truth surface stays explicit
   - later QR/SVD backend layering only if justified
   - broader benchmark-governance and packaging/platform work remain deferred
   That is much cleaner than handing forward a generic “more backend work”
   backlog.

## What Didn't Go Well

1. **The sprint still carried a high design/proof-to-code ratio.**
   That was the right tradeoff for the first backend-aware landing, but it
   means Sprint 64 spent significant effort on contract definition,
   interpretive alignment, and proof placement relative to the size of the
   code batch itself.

2. **The backend-aware architecture story is stronger, not comprehensive.**
   Sprint 64 intentionally landed only one bounded lane. That is the right
   Phase 1 result, but it also means the broader backend queue remains real:
   - LDL^T follow-through
   - later shared dense-kernel reuse decisions
   - any future build-option widening
   - later QR/SVD backend layering if justified

3. **The reviewed validation path is now very heavy.**
   Sprint 64 still closed cleanly, but the reviewed CMake path took:
   - `574.42 sec`
   with:
   - `test_reorder_nd` alone taking `385.06 sec`
   That does not block correctness, but it means backend-sensitive sprints now
   close against a fairly expensive reviewed baseline.

4. **The reviewed CMake rebuild still emits ordinary benchmark warning noise.**
   The recurring `bench_eigs_reuse.c` double-promotion warnings remain part of
   the background Day 13 story instead of being cleaned up here. That warning
   is non-blocking, but it is still validation noise.

## Final Metrics

### Validated closeout baseline

| Metric | Sprint 64 close state |
|---|---:|
| strongest local reviewed baseline command | `make quality-review-full` |
| reviewed CMake `ctest -N` | `53` |
| Makefile/CMake parity | `53 vs 53` |
| full reviewed CMake `ctest` | `53 / 53` |
| full reviewed CMake total real time | `574.42 sec` |

### Sprint 64 artifact package

| Metric | Sprint 64 close state |
|---|---:|
| total artifact files under `SPRINT_64/artifacts/` | `15` |
| main design/integration artifacts | `8` |
| compatibility/docs/validation/closeout artifacts | `4` |

Notes:

- main design/integration artifacts:
  - `day3-performance-hotspot-audit-part1.md`
  - `day4-performance-hotspot-rerank-and-first-landing-boundary.md`
  - `day5-backend-abstraction-contract-design.md`
  - `day6-build-option-surface-design.md`
  - `day7-kernel-integration-landing-design.md`
  - `day8-kernel-integration-batch1.md`
  - `day9-post-landing-safety-audit-and-proof-rerank.md`
  - `day10-backend-contract-error-and-fallback-truthfulness-batch.md`
- compatibility/docs/validation/closeout artifacts:
  - `day11-benchmark-proof-refresh.md`
  - `day12-docs-and-maintainer-follow-through.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Sprint 64 landed backend-aware package

| Metric | Sprint 64 close state |
|---|---:|
| backend-aware hot paths materially landed | `1` |
| highest-signal new proof/observability additions | `4` |
| targeted Day 13 follow-on commands rerun | `19` |
| explicitly deferred later backend/performance lanes | `5` |

Notes:

- backend-aware hot paths materially landed:
  - `Cholesky / CSC supernodal dense-kernel lane`
- highest-signal new proof/observability additions:
  - bounded internal dense-kernel descriptor seam for the CSC supernodal
    Cholesky lane
  - public `SPARSE_ERR_BACKEND_CONTRACT` taxonomy plus family-local missing
    descriptor/callback proof
  - benchmark-side path-identification fields:
    - `csc_scalar_path`
    - `csc_supernodal_path`
    - `csc_supernodal_dense_kernel`
  - public/header/README/maintainer alignment for the bounded backend-aware
    contract
- targeted Day 13 follow-on commands rerun:
  - `./build/test_integration`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_sparse_lu`
  - `./build/test_qr`
  - `./build/test_svd`
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_svd_lowrank`
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_chol_csc tests/data/suitesparse/bcsstk04.mtx --repeat 1`
  - `./build/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`
- explicitly deferred later backend/performance lanes:
  - LDL^T CSC supernodal backend-aware follow-through
  - bounded shared dense-kernel seam reuse where it reduces real duplicate risk
  - optional build-option or pluggable-kernel widening only if default-path
    truth stays explicit
  - later QR/SVD backend layering only if justified
  - broader benchmark-governance and packaging/platform work outside this
    immediate lane

## Residual Deferred Debt

Sprint 64 was explicitly about the first bounded backend-aware landing on the
highest-value dense-kernel and supernodal path. The main open work it
intentionally hands forward is:

- LDL^T CSC supernodal backend-aware follow-through
- bounded shared dense-kernel seam reuse only where it reduces real duplicate
  risk
- optional build-option or pluggable-kernel widening only if the
  self-contained default path and fallback truthfulness stay explicit
- later QR/SVD backend layering only if a later sprint justifies the proof
  burden
- broader benchmark-governance consolidation and packaging/platform work
  outside this immediate backend lane

Still consciously constrained rather than silently “solved”:

- no repo-wide pluggable-backend framework
- no fake platform closure beyond reviewed evidence
- no public runtime backend-surface widening for its own sake
- no broad benchmark-governance rewrite in Sprint 64
- no weakening of the self-contained default build

Not carried forward as unresolved Sprint 64 debt:

- missing hotspot ranking
- missing first backend-aware landing boundary
- missing bounded descriptor abstraction on the selected Cholesky CSC lane
- missing truthful backend-contract error taxonomy
- missing benchmark-side path-identification proof
- missing public/header/maintainer alignment for the landed lane
- missing validated Sprint 64 closeout

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day3-performance-hotspot-audit-part1.md](./artifacts/day3-performance-hotspot-audit-part1.md)
- [day4-performance-hotspot-rerank-and-first-landing-boundary.md](./artifacts/day4-performance-hotspot-rerank-and-first-landing-boundary.md)
- [day5-backend-abstraction-contract-design.md](./artifacts/day5-backend-abstraction-contract-design.md)
- [day6-build-option-surface-design.md](./artifacts/day6-build-option-surface-design.md)
- [day7-kernel-integration-landing-design.md](./artifacts/day7-kernel-integration-landing-design.md)
- [day8-kernel-integration-batch1.md](./artifacts/day8-kernel-integration-batch1.md)
- [day9-post-landing-safety-audit-and-proof-rerank.md](./artifacts/day9-post-landing-safety-audit-and-proof-rerank.md)
- [day10-backend-contract-error-and-fallback-truthfulness-batch.md](./artifacts/day10-backend-contract-error-and-fallback-truthfulness-batch.md)
- [day11-benchmark-proof-refresh.md](./artifacts/day11-benchmark-proof-refresh.md)
- [day12-docs-and-maintainer-follow-through.md](./artifacts/day12-docs-and-maintainer-follow-through.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 64 achieved its goal:

- the first Epic 6 backend-aware Phase 1 lane is now real and validated
- the self-contained default build and fallback truth surface stayed explicit
- the benchmark surface now proves not just that the CSC supernodal lane ran,
  but which dense-kernel descriptor backed it
- the public/header/maintainer story now matches the landed implementation
  without overstating Sprint 64 as a general backend framework
- the sprint closed from a fully reviewed validated baseline and handed
  forward a smaller, cleaner, and more explicit Sprint 65 queue
