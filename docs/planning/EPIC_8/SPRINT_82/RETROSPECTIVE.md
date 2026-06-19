# Sprint 82 Retrospective

**Sprint:** 82 — Dense Backend & Performance Ceiling Phase 3  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 82 fixed the dense/backend baseline, proof split, and
      implementation-day validation contract before landing code
- [x] the strongest live dense-hotspot contradiction map was reranked from the
      current tree rather than inherited generically from Sprint 81
- [x] Sprint 82 fixed one explicit first implementation fence centered on:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
- [x] Sprint 82 landed one bounded optional dense-backend batch:
  - the shipped builtin dense backend remained the default path
  - `SPARSE_CHOL_DENSE_BACKEND=accelerate` became one bounded Darwin-only
    runtime selector for the Cholesky CSC supernodal lane
- [x] Sprint 82 landed one bounded solver-adoption follow-through batch:
  - the shipped builtin LDL^T dense block factor remained the default path
  - `SPARSE_LDLT_DENSE_BACKEND=accelerate` became one bounded Darwin-only
    runtime selector for the LDL^T CSC supernodal lane
  - fallback behavior remained proof-backed through `SPARSE_ERR_PIVOT_REJECTED`
    and the existing scalar-prepass / supernodal fallback story
- [x] Sprint 82 used bounded follow-through correctly:
  - `docs/maintainer_guide.md` was reconciled with the landed backend surface
  - broader README, benchmark-doc, package, and public-header churn was
    correctly avoided where the tree already stayed truthful
- [x] Sprint 82 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- [x] Sprint 82 closed with one explicit Sprint 83-first handoff queue instead
      of another generic Epic 8 summary

## What Went Well

1. **Sprint 82 widened the backend seam without abandoning the self-contained product path.**
   The sprint added real optional acceleration work, but kept the builtin
   scalar dense path as the default shipped behavior. That preserved the
   strongest Sprint 80 non-goal fence while still moving the performance
   ceiling.

2. **The first implementation center was the right one.**
   Sprint 82 did not scatter immediately into QR, SVD, packaging, or broad
   runtime policy work. It targeted the highest-value backend seam first:
   - dense-kernel descriptor ownership
   - Cholesky CSC supernodal consumer path
   - bounded runtime selection for one optional accelerated path

3. **The second batch chose the correct adoption seam.**
   After Day 6, the strongest remaining contradiction was LDL^T parity. The
   sprint correctly moved next to the LDL^T CSC supernodal lane instead of
   spending time on weaker benchmark or support-surface churn.

4. **Fallback behavior stayed disciplined and explicit.**
   Sprint 82 did not turn optional acceleration into a risky “best effort”
   path. The LDL^T widening preserved the existing pivot/block contract and
   returned `SPARSE_ERR_PIVOT_REJECTED` when the optional path could not
   satisfy it, keeping the scalar-prepass/supernodal fallback story intact.

5. **Proof ownership stayed tight.**
   The sprint used the right proof owners:
   - `tests/test_chol_csc.c` for the bounded Cholesky optional backend lane
   - `tests/test_ldlt.c` for the bounded LDL^T optional dense-factor lane
   - `docs/maintainer_guide.md` for the authoritative support-surface reading
   Benchmarks remained measurability surfaces rather than correctness owners.

6. **Sprint 83 now has a cleaner starting point.**
   Sprint 82 reduced the builtin scalar dense/backend contradiction enough that
   capability-surface widening is now the correct next Epic 8 center instead
   of reopening the same dense-backend ceiling first.

## What Didn't Go Well

1. **Sprint 82 still landed a narrow runtime seam, not broad backend parity.**
   That was the right bounded choice, but it leaves real residual work:
   - the optional acceleration story is Darwin-only
   - the widened seam remains local to the touched Cholesky and LDL^T lanes
   - QR and SVD were explicitly deferred

2. **The branch did not widen public backend contracts.**
   Again, that was deliberate and truthful, but it means the optional backend
   work mostly lives behind internal/runtime-selection seams rather than a
   broader public capability surface.

3. **Benchmark follow-through stayed intentionally narrower than the implementation package.**
   Sprint 82 proved the widened backend surface through focused reruns and
   retained benchmark measurability, but it did not widen canonical benchmark
   logic or reporting. That preserves governance discipline, but it means
   performance observability remains intentionally conservative.

4. **The sprint did not reopen package/install proof, by design.**
   That is correct rather than a defect, but it also means Sprint 82’s final
   validation story is intentionally narrower than a package-affecting sprint:
   - install/export proof was left untouched because no package or runtime
     package mechanics moved

5. **The reviewed runtime long pole remains large.**
   `test_reorder_nd` still dominated reviewed runtime. Sprint 82 closed
   cleanly from a strong validation baseline, but it did not reduce that
   operational drag.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 82 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `611.27 sec` |
| reviewed `test_reorder_nd` time | `420.51 sec` |
| focused `test_chol_csc` follow-on | `149 / 149` |
| focused `test_ldlt` follow-on | `86 / 86` |
| focused `test_qr` follow-on | `72 / 72` |
| focused `test_svd` follow-on | `97 / 97` |
| focused `test_integration` follow-on | `53 / 53` |

### Sprint 82 artifact package

| Metric | Sprint 82 close state |
|---|---:|
| total artifact files under `SPRINT_82/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-dense-backend-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-proof-surface-recheck.md`
  - `day3-dense-hotspot-profiling-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-final-proof-alignment-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-backend-boundary.md`
  - `day5-dense-kernel-abi-design.md`
  - `day6-optional-dense-backend-integration-batch.md`
  - `day8-solver-adoption-follow-through-design.md`
  - `day9-solver-adoption-follow-through-batch.md`
  - `day10-benchmark-differential-and-runtime-alignment-design.md`
  - `day11-benchmark-differential-and-runtime-alignment-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 82 close state |
|---|---:|
| implementation `.c` files touched | `3` |
| internal header files touched | `1` |
| public header files touched | `0` |
| proof-owner test files touched | `2` |
| benchmark source files touched | `0` |
| support docs requiring follow-through | `1` |

Notes:

- implementation `.c` files touched:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc_supernodal.c`
- internal header files touched:
  - `src/sparse_chol_csc_internal.h`
- proof-owner test files touched:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
- support surface intentionally moved:
  - `docs/maintainer_guide.md`
- support surfaces intentionally left untouched after recheck:
  - `README.md`
  - `benchmarks/README.md`
  - `include/sparse_ldlt.h`

## Residual Deferred Debt

Sprint 82 deliberately stopped after the highest-value backend package. The
main open work it hands forward is:

- bounded capability-surface widening on the highest-value solver seams
- stronger external differential and property assurance on the touched direct
  families
- later QR/SVD dense-workspace and broader backend follow-through only where
  bounded evidence justifies widening
- later maintainability, runtime, package/platform, and usability work in the
  preserved Epic 8 order

Still consciously constrained rather than silently “solved”:

- no mandatory heavyweight optional-backend dependency for the default build
- no fake platform parity from a Darwin-only optional acceleration seam
- no widened public backend enum or callback contract
- no QR/SVD backend modernization claim
- no benchmark-threshold or timing-gate inflation
- no package/platform maturity claim broadening

Not carried forward as unresolved Sprint 82 debt:

- the baseline/proof-surface recheck
- the live dense-hotspot rerank
- the bounded dense-kernel ABI and runtime-selection contract
- the Day 6 optional Cholesky backend landing
- the Day 9 bounded LDL^T backend/runtime follow-through
- the bounded maintainer-policy reconciliation
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 83-first handoff queue

## Key Deliverables

1. **One proof-backed optional dense-backend seam landed on the highest-value Cholesky lane.**
   `SPARSE_CHOL_DENSE_BACKEND=accelerate` now exists as one bounded Darwin-only
   runtime selector while the builtin dense backend remains the default path.

2. **One bounded LDL^T backend/runtime follow-through package landed.**
   `SPARSE_LDLT_DENSE_BACKEND=accelerate` now exists on the LDL^T CSC
   supernodal lane with the scalar-prepass/supernodal fallback story preserved.

3. **One truthful maintainer-policy update landed.**
   `docs/maintainer_guide.md` now says directly that the Cholesky CSC lane owns
   the first optional dense-kernel runtime seam and the LDL^T CSC lane also
   owns one bounded optional dense-factor runtime seam.

4. **One strong proof-owner split was preserved and extended.**
   Sprint 82 widened backend proof where it belonged without inflating
   benchmark or support-surface responsibility.

5. **Sprint 82 closed from a measured backend-aware baseline, not just from ABI/design prose.**
   The branch ended with a full Day 13 validation sweep and an explicit Day 14
   handoff queue for Sprint 83 and the later Epic 8 lanes.

## Bottom Line

Sprint 82 succeeded because it stayed bounded where the tree most needed it.
It did not try to turn the library into a generic backend framework or a fake
cross-platform acceleration story. Instead, it raised the dense-backend ceiling
with one optional Cholesky seam, one bounded LDL^T follow-through seam, one
truthful maintainer-policy reconciliation, and one strong validated baseline.
That is enough real backend movement to make Sprint 83 the correct next
contradiction center.
