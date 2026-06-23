# Sprint 83 Retrospective

**Sprint:** 83 — Capability Surface Modernization Phase 2  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 83 fixed the capability baseline, proof split, and
      implementation-day validation contract before landing code
- [x] the strongest live capability contradiction map was reranked from the
      current tree rather than inherited generically from Sprint 82
- [x] Sprint 83 fixed one explicit first implementation fence centered on:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- [x] Sprint 83 landed one bounded shared scalar-surface expansion batch:
  - the shipped scalar contract remained real-only `double`
  - the shared matrix-shell public seam now routes caller-facing dense-scalar
    paths through `sparse_scalar_t`
- [x] Sprint 83 landed one bounded shared vocabulary reconciliation batch:
  - `include/sparse_types.h` no longer reads like `sparse_scalar_t` is mainly
    an iterative/eigs-only public seam
  - the shared matrix-shell/public-owner reading is now explicit alongside the
    iterative/eigs seams
- [x] Sprint 83 landed one bounded algorithm-surface widening batch:
  - `include/sparse_qr.h` now routes the highest-value caller-owned QR buffers
    and dense helper outputs through `sparse_scalar_t`
  - no `src/sparse_qr.c` implementation churn or broader family reopening was
    needed
- [x] Sprint 83 used bounded follow-through correctly:
  - `tests/test_sparse_matrix.c` and `tests/test_qr.c` became the direct
    proof-owner surfaces for the widened public seams
  - `docs/maintainer_guide.md` was reconciled with the landed capability
    surface
  - broader README, package, SVD, Cholesky, and LDL^T churn was correctly
    avoided where the tree already stayed truthful
- [x] Sprint 83 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- [x] Sprint 83 closed with one explicit Sprint 84-first handoff queue instead
      of another generic Epic 8 summary

## What Went Well

1. **Sprint 83 widened the public owner reading without pretending the numeric contract changed.**
   The sprint moved the highest-value public seams onto `sparse_scalar_t` and
   retained the truthful shipped constraint: the scalar lane is still
   real-only `double`. That is exactly the bounded capability move Sprint 83
   was supposed to make.

2. **The first implementation center was the right one.**
   Sprint 83 did not jump immediately into QR/SVD kernels, direct-family
   rewrites, complex arithmetic, or package churn. It first fixed the shared
   matrix-shell/public-owner contradiction where the capability story was most
   visibly stale.

3. **The second batch chose the correct residual seam.**
   After Day 6, the strongest contradiction was not another matrix-shell code
   pass. It was the shared vocabulary owner in `include/sparse_types.h`. Day 9
   resolved that without reopening lower-value implementation surfaces.

4. **QR was the right bounded family follow-through.**
   Once the shared owner reading was explicit, Sprint 83 widened one
   high-value algorithm-family public seam in `include/sparse_qr.h` and proved
   it locally in `tests/test_qr.c`. That gave the sprint one real family-local
   modernization result without pretending the whole solver surface widened.

5. **Proof ownership stayed disciplined.**
   The sprint used the right owners:
   - `tests/test_sparse_matrix.c` for the shared matrix-shell scalar and width
     contract
   - `tests/test_qr.c` for the bounded QR public scalar seam
   - `tests/test_iterative.c` and `tests/test_eigs.c` as the retained earlier
     scalar-owner proof surfaces
   - `docs/maintainer_guide.md` for the authoritative support-surface reading

6. **Sprint 84 now has a cleaner starting point.**
   Sprint 83 reduced the stale public capability reading enough that the next
   highest-value Epic 8 move is stronger external/differential/property
   assurance, not another round of owner-seam contradiction cleanup.

## What Didn't Go Well

1. **Sprint 83 widened ownership, not true numeric breadth.**
   That was the correct bounded result, but it leaves real residual work:
   - no true complex-scalar support
   - no mixed-precision contract
   - no broader direct-family or SVD public capability widening

2. **The implementation package stayed intentionally narrow.**
   Most of the sprint’s real movement lives in public headers, one shared
   matrix implementation file, proof-owner tests, and maintainer policy. That
   is truthful, but it means Sprint 83 did not yet force broader algorithm
   implementation movement.

3. **The widened scalar owner still aliases the shipped real-only lane.**
   This keeps compatibility intact, but it also means the widened capability
   surface is still preparatory in one important sense: it modernizes the
   public seam before broadening the actual numeric domain.

4. **The sprint did not reopen package/install proof, by design.**
   That is correct rather than a defect, but it also means Sprint 83’s final
   validation story is intentionally narrower than a package-affecting sprint:
   - install/export proof was left untouched because no package or runtime
     package mechanics moved

5. **The reviewed runtime long pole remains large.**
   `test_reorder_nd` still dominated reviewed runtime. Sprint 83 closed
   cleanly from a strong validation baseline, but it did not reduce that
   operational drag.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 83 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `446.47 sec` |
| reviewed `test_reorder_nd` time | `314.43 sec` |
| focused `test_sparse_matrix` follow-on | `59 / 59` |
| focused `test_qr` follow-on | `73 / 73` |
| focused `test_svd` follow-on | `97 / 97` |
| focused `test_chol_csc` follow-on | `149 / 149` |
| focused `test_ldlt` follow-on | `87 / 87` |
| focused `test_integration` follow-on | `53 / 53` |

### Sprint 83 artifact package

| Metric | Sprint 83 close state |
|---|---:|
| total artifact files under `SPRINT_83/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-capability-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-proof-surface-recheck.md`
  - `day3-capability-rerank-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-final-proof-alignment-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-capability-boundary.md`
  - `day5-scalar-index-architecture-design.md`
  - `day6-scalar-surface-expansion-batch.md`
  - `day8-index-abi-follow-through-design.md`
  - `day9-index-abi-follow-through-batch.md`
  - `day10-algorithm-surface-widening-design.md`
  - `day11-algorithm-surface-widening-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 83 close state |
|---|---:|
| implementation `.c` files touched | `1` |
| internal header files touched | `0` |
| public header files touched | `3` |
| proof-owner test files touched | `2` |
| benchmark source files touched | `0` |
| support docs requiring follow-through | `1` |

Notes:

- implementation `.c` files touched:
  - `src/sparse_matrix.c`
- public header files touched:
  - `include/sparse_types.h`
  - `include/sparse_matrix.h`
  - `include/sparse_qr.h`
- proof-owner test files touched:
  - `tests/test_sparse_matrix.c`
  - `tests/test_qr.c`
- support surface intentionally moved:
  - `docs/maintainer_guide.md`
- support surfaces intentionally left untouched after recheck:
  - `README.md`
  - `include/sparse_svd.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

## Residual Deferred Debt

Sprint 83 deliberately stopped after the highest-value capability-surface
package. The main open work it hands forward is:

- stronger external differential, seeded-property, and failure-path assurance
  on the touched shared/direct lanes
- later SVD and direct-family public capability widening only where bounded
  evidence justifies widening
- later true complex support and mixed precision
- later maintainability, package/ABI/runtime maturity, and broader usability
  work in the preserved Epic 8 order

Still consciously constrained rather than silently “solved”:

- no claim that `sparse_scalar_t` is now numerically generic in practice
- no true complex-scalar support
- no mixed-precision framework
- no SVD public-header widening
- no direct-family public-header capability widening
- no package/platform maturity claim broadening

Not carried forward as unresolved Sprint 83 debt:

- the baseline/proof-surface recheck
- the live capability rerank
- the bounded scalar/index architecture contract
- the Day 6 shared matrix-shell scalar-surface expansion
- the Day 9 shared scalar/index vocabulary reconciliation
- the Day 11 bounded QR public-header widening
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 84-first handoff queue

## Key Deliverables

1. **One proof-backed shared matrix-shell scalar-owner widening landed.**
   The highest-value shared caller-facing dense-scalar public seam now routes
   through `sparse_scalar_t` while preserving the shipped real-only `double`
   contract.

2. **One bounded shared scalar/index vocabulary reconciliation landed.**
   `include/sparse_types.h` now says directly that the shared matrix-shell
   seam, together with the iterative/eigs seams, is the active public-owner
   surface for `sparse_scalar_t`.

3. **One bounded QR public-header widening package landed.**
   `include/sparse_qr.h` now routes the highest-value caller-owned QR buffers
   and helper outputs through `sparse_scalar_t` without forcing immediate QR
   implementation churn.

4. **One truthful maintainer-policy update landed.**
   `docs/maintainer_guide.md` now reflects the widened shared and QR
   proof-owner split rather than the earlier narrower capability reading.

5. **Sprint 83 closed from a measured capability-aware baseline, not just from header/design prose.**
   The branch ended with a full Day 13 validation sweep and an explicit Day 14
   handoff queue for Sprint 84 and the later Epic 8 lanes.

## Bottom Line

Sprint 83 succeeded because it stayed bounded where the tree most needed it.
It did not pretend that header alias widening alone solved true numeric
breadth, but it did remove the strongest stale public-owner contradictions on
the matrix shell, the shared vocabulary layer, and one high-value QR seam.
That is enough real capability-surface movement to make Sprint 84 the correct
next contradiction center.
