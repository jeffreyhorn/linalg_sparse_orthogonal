# Sprint 96 Retrospective

**Sprint:** 96 - Large-Source & Giant-Test Maintainability Phase 6
**Duration:** 14 days (Days 1-14 landed on this branch)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 96 started from the Epic 9 project-plan section and live
      post-Sprint-95 hotspot evidence
- [x] implementation and proof-owner candidates were reranked by review cost,
      ownership ambiguity, extraction risk, and validation blast radius
- [x] source extraction boundaries were designed before code movement
- [x] the direct-family cleanup landed as a bounded LDLT dense/backend
      extraction without public API changes
- [x] the solver-family cleanup landed as a bounded iterative block-wrapper
      extraction without public API changes
- [x] the giant-test cleanup split the Cholesky CSC proof owner into core and
      supernodal/writeback owners
- [x] touched internal comments and helper names now describe durable
      ownership and invariants instead of stale implementation chronology
- [x] Makefile and CMake registrations include all new source and test owners
- [x] the full branch-level validation chain passed before closeout:
  - `make format`
  - `make lint`
  - `make test`
- [x] Sprint 96 closed with an explicit Sprint 97 residual maintainability
      queue

## What Went Well

1. **The sprint picked real hotspots instead of broad cleanup themes.**
   Day 1 and Day 2 measured the live source and proof-owner surfaces before
   selecting work. That kept the sprint centered on `src/sparse_ldlt_csc.c`,
   `src/sparse_iterative.c`, and `tests/test_chol_csc.c`.

2. **The direct-family extraction found a clean internal boundary.**
   `src/sparse_ldlt_dense.c` now owns dense LDLT primitive behavior and
   backend selection details that were embedded in `src/sparse_ldlt_csc.c`.
   The move stayed behind existing internal declarations and avoided public
   API churn.

3. **The solver cleanup reduced mixed ownership without changing callers.**
   `src/sparse_iterative_block.c` now owns block CG/GMRES wrappers while
   `src/sparse_iterative.c` keeps scalar, handle, and matrix-free solver
   ownership. That gives future solver work a clearer file boundary.

4. **The giant-test split improved proof ownership directly.**
   `tests/test_chol_csc.c` no longer carries every Cholesky CSC proof group.
   Supernodal, dense-backend, panel, parametrised cross-check, and writeback
   proofs now live in `tests/test_chol_csc_supernodal.c` with shared helper
   ownership in `tests/test_chol_csc_supernodal_helpers.h`.

5. **Day 12 removed confusing local chronology from touched owners.**
   Helper names such as `day8_count_supernodes` and `day10_roundtrip_check`
   were replaced with intent-based names, and touched comments now explain
   current invariants and compatibility behavior.

6. **The sprint closed from a full validation anchor.**
   Day 13 ran `make format && make lint && make test`, including strict
   warning compilation, benchmark/example binary builds, `clang-tidy`,
   `cppcheck`, and the full test suite.

## What Didn't Go Well

1. **The total line count did not simply shrink.**
   The main improvement is ownership clarity. Extracted code and split proof
   groups moved into new owners, so some total touched-area line counts are
   roughly preserved or slightly higher even though individual review surfaces
   are clearer.

2. **Several large proof owners remain.**
   `tests/test_ldlt_csc.c`, `tests/test_integration.c`, `tests/test_qr.c`,
   and `tests/test_iterative.c` remain large enough to deserve dedicated
   follow-up work.

3. **The cleanup correctly avoided public API redesign.**
   That kept Sprint 96 bounded, but it also means broader package, public
   header, and generated-documentation modernization remain outside this
   sprint's claim.

4. **Some chronology remains outside touched owners.**
   Day 12 intentionally avoided a repo-wide sprint/day text purge. Historical
   names and comments remain in unrelated legacy tests, build comments, and
   planning history.

5. **Validation cost was high.**
   Because Sprint 96 changed `.c` and `.h` files, each implementation day had
   to be backed by focused checks and full quality gates. That was necessary,
   but it made the closeout runtime heavier than a docs-only sprint.

## Final Metrics

### Validation

| Metric | Sprint 96 close state |
|---|---:|
| standard branch-level gate | `make format && make lint && make test` passed |
| final test summary | `All tests passed.` |
| new source/test registration scan | passed for Makefile and CMake |
| removed Cholesky helper-name scan | no matches in split proof-owner files |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 96 docs and touched source/test/build files |

### Sprint 96 Artifact Package

| Metric | Sprint 96 close state |
|---|---:|
| total artifact files under `SPRINT_96/artifacts/` | `15` |
| baseline/rerank/design artifacts | `4` |
| source cleanup artifacts | `6` |
| proof/comment/validation/closeout artifacts | `5` |

Notes:

- baseline/rerank/design artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-scope-and-hotspot-baseline.md`
  - `day2-hotspot-rerank.md`
  - `day3-source-extraction-design.md`
- source cleanup artifacts:
  - `day4-direct-family-boundary-freeze.md`
  - `day5-direct-family-source-cleanup-batch1.md`
  - `day6-direct-family-cleanup-closeout.md`
  - `day7-solver-algorithm-boundary-freeze.md`
  - `day8-solver-source-cleanup-batch1.md`
  - `day9-solver-cleanup-closeout.md`
- proof/comment/validation/closeout artifacts:
  - `day10-giant-test-architecture-design.md`
  - `day11-giant-test-cleanup-batch1.md`
  - `day12-internal-comment-rationale-cleanup.md`
  - `day13-validation-and-residual-queue.md`
  - `day14-sprint96-closeout.md`

### Landed Cleanup Package

| Metric | Sprint 96 close state |
|---|---:|
| implementation source owners added | `2` |
| implementation source owners reduced in place | `2` |
| internal headers touched | `2` |
| proof-owner test files added | `1` |
| proof-owner helper headers touched | `1` |
| build registration surfaces touched | `2` |

Notes:

- implementation owners:
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_ldlt_dense.c`
  - `src/sparse_iterative.c`
  - `src/sparse_iterative_block.c`
- internal headers:
  - `src/sparse_ldlt_csc_internal.h`
  - `src/sparse_iterative_internal.h`
- proof owners:
  - `tests/test_chol_csc.c`
  - `tests/test_chol_csc_supernodal.c`
  - `tests/test_chol_csc_supernodal_helpers.h`
- build registration surfaces:
  - `Makefile`
  - `CMakeLists.txt`

### Final Hotspot Shape

| Owner | Sprint 96 close state |
|---|---:|
| `src/sparse_ldlt_csc.c` | `2174` lines |
| `src/sparse_ldlt_dense.c` | `590` lines |
| `src/sparse_iterative.c` | `1495` lines |
| `src/sparse_iterative_block.c` | `375` lines |
| `tests/test_chol_csc.c` | `2611` lines |
| `tests/test_chol_csc_supernodal.c` | `2464` lines |
| `tests/test_chol_csc_supernodal_helpers.h` | `244` lines |

## Residual Deferred Debt

Sprint 96 deliberately stopped after the highest-value direct source,
iterative source, and Cholesky CSC proof-owner cleanup lanes.

Most important carry-forward work:

- split or helper cleanup for `tests/test_ldlt_csc.c`
- QR-focused cleanup across `src/sparse_qr.c` and `tests/test_qr.c`
- eigensolver lifecycle cleanup in `src/sparse_eigs.c`
- design-first lifecycle/progress cleanup for `tests/test_integration.c`
- later cleanup for `src/sparse_lu_csr.c`, `src/sparse_ldlt.c`,
  `src/sparse_chol_csc.c`, `src/sparse_matrix.c`, `src/sparse_svd.c`, and
  remaining broad proof owners

Still consciously constrained rather than silently solved:

- no public-header redesign
- no benchmark command or harness rename
- no generated API documentation edit
- no repo-wide sprint/day chronology purge
- no simultaneous split of multiple giant proof owners
- no claim that all large owners are now small

Not carried forward as unresolved Sprint 96 debt:

- live hotspot rerank
- source extraction design
- LDLT dense/backend extraction
- iterative block-wrapper extraction
- Cholesky CSC supernodal/writeback proof-owner split
- touched-owner comment and helper-name cleanup
- final branch-wide validation
- explicit Sprint 97 residual queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-hotspot-baseline.md](./artifacts/day1-scope-and-hotspot-baseline.md)
- [day2-hotspot-rerank.md](./artifacts/day2-hotspot-rerank.md)
- [day3-source-extraction-design.md](./artifacts/day3-source-extraction-design.md)
- [day5-direct-family-source-cleanup-batch1.md](./artifacts/day5-direct-family-source-cleanup-batch1.md)
- [day6-direct-family-cleanup-closeout.md](./artifacts/day6-direct-family-cleanup-closeout.md)
- [day8-solver-source-cleanup-batch1.md](./artifacts/day8-solver-source-cleanup-batch1.md)
- [day9-solver-cleanup-closeout.md](./artifacts/day9-solver-cleanup-closeout.md)
- [day11-giant-test-cleanup-batch1.md](./artifacts/day11-giant-test-cleanup-batch1.md)
- [day12-internal-comment-rationale-cleanup.md](./artifacts/day12-internal-comment-rationale-cleanup.md)
- [day13-validation-and-residual-queue.md](./artifacts/day13-validation-and-residual-queue.md)
- [day14-sprint96-closeout.md](./artifacts/day14-sprint96-closeout.md)

## Bottom Line

Sprint 96 achieved its goal:

- the largest direct-family source owner has a clearer dense/backend boundary
- the largest iterative source owner has a clearer block-wrapper boundary
- the largest Cholesky CSC proof owner is split by proof responsibility
- touched comments and helper names now explain current ownership instead of
  implementation-day history
- the new owners are registered in both supported build systems
- the branch validates cleanly under the full quality chain
- Sprint 97 receives a bounded maintainability queue instead of a broad
  refactor backlog

Future maintainability work can now start from explicit owner seams and a
ranked residual queue instead of rediscovering the same large-file map.
