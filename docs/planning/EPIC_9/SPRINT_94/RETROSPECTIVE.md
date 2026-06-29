# Sprint 94 Retrospective

**Sprint:** 94 — Capability Surface Modernization Phase 3  
**Duration:** 14 days  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 94 fixed the implementation-day validation and maintained-surface
      contract before capability work widened
- [x] the sprint reduced the broad capability problem to one ranked live
      contradiction map instead of reopening generic feature-expansion churn
- [x] Sprint 94 froze one explicit scalar/index capability contract:
  - bounded scalar-contract debt first
  - touched index/ABI maturity second
  - solver-family breadth only where the widened scalar/index seam truly
    needed it
- [x] Sprint 94 froze one explicit first implementation fence centered on:
  - `include/sparse_types.h`
  - the matching shared matrix-shell scalar implementation seam
- [x] Sprint 94 froze one explicit scalar-widening implementation contract:
  - shared matrix-shell helper plus storage/build seam first
  - no broad solver-family widening in the first batch
  - proof follow-through bounded to directly forced owners
- [x] Sprint 94 landed one bounded matrix-shell scalar widening batch:
  - `Node.value` now follows `sparse_scalar_t`
  - builder/storage seams now follow `sparse_scalar_t`
  - public scalar-alias proof widened through matrix-shell operations
- [x] Sprint 94 landed one bounded matrix-shell index/ABI maturity batch:
  - checked save/print writes now fail closed with `SPARSE_ERR_IO`
  - malformed Matrix Market dimension and coordinate cases now reject with
    `SPARSE_ERR_PARSE`
  - touched width-aware parse/diagnostic proof was added
- [x] Sprint 94 explicitly retired the final solver-family implementation batch
      by evidence instead of widening into generic numeric-family churn
- [x] Sprint 94 landed one bounded support-only alignment batch:
  - maintainer wording now matches the real matrix-shell scalar seam
  - README scalar limitation now matches the actual widened public seams
  - proof-owner mapping now includes the touched Matrix Market parse-width
    lane
- [x] Sprint 94 ran the full final validation sweep and closed from one
      explicit validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
  - focused scalar/index/capability reruns
  - `make bench-canonical-report`
- [x] Sprint 94 closed with one explicit Sprint 95-first handoff queue instead
      of reopening bounded capability work as another broad modernization reset

## What Went Well

1. **Sprint 94 chose the right first capability seam.**
   The sprint did not begin with generic solver-family widening. It targeted
   the highest-value shared scalar seam first: the public matrix-shell helper
   and matching storage/build owner.

2. **The scalar widening became real where it mattered most.**
   Day 7 moved `sparse_scalar_t` from a bounded preparation story to a real
   matrix-shell implementation seam, which is a stronger and more defensible
   product move than widening a lower-value family-local surface first.

3. **The index/ABI follow-through stayed sharply bounded.**
   Day 10 improved width-aware save/load and diagnostic truth without
   pretending the repo now has broad 64-bit or ABI modernization everywhere.

4. **The sprint retired unnecessary implementation work by evidence.**
   Day 11 correctly concluded that a further solver-family code batch was not
   needed to make the bounded Sprint 94 capability claim truthful.

5. **Support wording caught up to the code and proof cleanly.**
   Day 12 aligned maintainer and public wording to the actual landed scalar
   and index seams without broadening the non-claim fence.

6. **The sprint closed from a strong validated baseline.**
   Sprint 94 closed from the implementation-day queue, the full reviewed path,
   exact Makefile/CMake parity, focused scalar/index/capability reruns, and
   canonical benchmark reporting.

## What Didn't Go Well

1. **The widened capability claim is still intentionally narrow.**
   Sprint 94 materially improved the capability surface, but it did not create
   broad complex support, mixed-precision maturity, or dense/SVD-family scalar
   widening.

2. **The index-width story is stronger on touched paths than across the whole repo.**
   Day 10 sharpened the matrix-shell consumer and diagnostic seam, but it did
   not attempt to claim full 64-bit maturity across every product and proof
   owner.

3. **The final solver-family item collapsed rather than landing as code.**
   That was the correct bounded decision, but it means Sprint 94’s visible code
   movement is concentrated in matrix-shell scalar/index surfaces more than in
   every solver-family owner named in the original broad contradiction set.

4. **The reviewed long pole is still elsewhere.**
   Sprint 94 closed cleanly, but `test_reorder_nd` remained the dominant
   reviewed runtime owner even though this sprint correctly did not reopen the
   Sprint 93 runtime lane.

5. **Install/export proof stayed intentionally untouched.**
   That kept the sprint disciplined, but it also means the capability closeout
   story is concentrated in code/proof/docs rather than in any broadened
   packaging contract.

## Final Metrics

### Validation and close anchors

| Metric | Sprint 94 close state |
|---|---:|
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `446.86 sec` |
| reviewed `test_reorder_nd` time | `217.46 sec` |
| reviewed `test_fuzz` time | `79.28 sec` |
| canonical reporting follow-through | `make bench-canonical-report` passed |

### Focused Sprint 94 scalar/index/capability anchors

| Metric | Sprint 94 close state |
|---|---:|
| touched matrix-shell proof owner | `test_sparse_matrix` |
| `test_sparse_matrix` result | `63 / 63` |
| touched Matrix Market proof owner | `test_sparse_io` |
| `test_sparse_io` result | `26 / 26` |
| retained iterative scalar proof owner | `test_iterative` |
| `test_iterative` result | `80 / 80` |
| retained QR scalar proof owner | `test_qr` |
| `test_qr` result | `73 / 73` |
| retained eigensolver scalar proof owner | `test_eigs` |
| `test_eigs` result | `31 / 31` |
| `example_analysis` residual | `4.44e-16` |
| `example_basic_solve` residual | `0.00e+00` |

### Sprint 94 artifact package

| Metric | Sprint 94 close state |
|---|---:|
| total artifact files under `SPRINT_94/artifacts/` | `15` |
| baseline/setup artifacts | `3` |
| audit/design/fence artifacts | `7` |
| implementation/follow-through artifacts | `3` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/setup artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-scope-and-capability-baseline.md`
  - `day2-validation-baseline-and-maintained-surface-recheck.md`
- audit/design/fence artifacts:
  - `day3-capability-rerank-audit.md`
  - `day4-scalar-and-index-capability-contract-design.md`
  - `day5-first-implementation-boundary.md`
  - `day6-scalar-family-widening-design.md`
  - `day8-post-landing-audit-and-rerank.md`
  - `day9-index-and-abi-maturity-design.md`
  - `day11-solver-family-breadth-and-alignment-design.md`
- implementation/follow-through artifacts:
  - `day7-scalar-family-widening-batch.md`
  - `day10-index-and-abi-maturity-batch.md`
  - `day12-solver-family-breadth-and-support-alignment-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed change class

| Metric | Sprint 94 close state |
|---|---:|
| public type/header surfaces touched | `1` |
| shared implementation source owners touched | `1` |
| internal matrix-shell headers touched | `1` |
| proof-owner C test files touched | `2` |
| support/public docs touched | `2` |
| build-system surfaces touched | `0` |
| install/export proof scripts touched | `0` |
| workflow files touched | `0` |

Notes:

- landed implementation/proof/support surfaces:
  - `include/sparse_types.h`
  - `src/sparse_matrix_internal.h`
  - `src/sparse_matrix.c`
  - `tests/test_sparse_matrix.c`
  - `tests/test_sparse_io.c`
  - `docs/maintainer_guide.md`
  - `README.md`
- no build-system, install/export, or workflow surface had to move for Sprint
  94 to land

## Residual Deferred Debt

Sprint 94 intentionally widened the capability surface on the highest-value
bounded seams without pretending the repo now has broad numeric genericity.

Most important carry-forward work:

- narrative and workflow coherence on permanent product/support surfaces
- chronology residue and public/support simplification
- remaining large-source and proof-owner maintainability concentration
- build/package/workflow convergence
- broader comparison depth
- final Epic 9 integration and closeout

Still consciously constrained rather than silently "solved":

- no broad complex support claim
- no broad mixed-precision maturity claim
- no dense/SVD-family scalar widening claim
- no claim that wider index maturity is uniformly complete across the repo
- no broader package/platform reinterpretation

Not carried forward as unresolved Sprint 94 debt:

- the capability rerank and scalar/index contract freeze
- the Day 7 bounded matrix-shell scalar widening
- the Day 10 bounded matrix-shell index/ABI maturity landing
- the Day 11 retirement of unnecessary solver-family implementation work
- the Day 12 support-only wording/proof-owner alignment
- the Day 13 validated close baseline

## Key Deliverables

1. **One real matrix-shell scalar widening landed.**
   Sprint 94 made the shared matrix-shell storage/build seam match the public
   `sparse_scalar_t` contract instead of leaving that alias as preparation-only
   wording on the highest-value touched owner.

2. **One touched index/ABI maturity seam landed.**
   The matrix-shell load/save and diagnostic path now rejects malformed Matrix
   Market inputs more truthfully and treats write failures as real I/O errors.

3. **One unnecessary solver-family implementation batch was retired.**
   Sprint 94 proved that its bounded capability claim did not need another
   numeric-family code landing to remain truthful.

4. **One aligned capability close state landed.**
   The maintainer and public wording now match the actual bounded scalar/index
   seams and proof owners instead of reading like stale preparation language.

5. **One exact validated Sprint 94 baseline landed.**
   The sprint closes from the full implementation-day queue, reviewed parity,
   focused scalar/index/capability reruns, and canonical benchmark reporting.

## Bottom-Line Closeout

Sprint 94 succeeded because it made one bounded capability improvement real on
the highest-value shared owner and then stopped widening when the evidence no
longer justified more code. The matrix-shell scalar seam is now real, the
touched index/ABI consumer path is sharper, the support wording matches the
landed proof, and the sprint closes from a strong validated baseline. It did
not solve broad numeric genericity, and it did not claim to, which is why the
close state is credible.
