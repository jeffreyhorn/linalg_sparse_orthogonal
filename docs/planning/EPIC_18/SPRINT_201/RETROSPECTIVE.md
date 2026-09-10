# Sprint 201 Retrospective

**Sprint:** 201 - Additional Review-Surface Reduction
**Duration:** 14 days (Days 1-14 landed on branch `sprint-201`)
**Status:** Closed with selected SVD rank, pseudoinverse, and dense low-rank
helper review-surface reduction completed; broad SVD, public API/ABI,
performance, package, platform, and repository-wide review-surface claims
remain unclaimed

## Source Artifact Note

Sprint 201 was executed from the Epic 18 project-plan section for Sprint 201
and lives under `docs/planning/EPIC_18/SPRINT_201/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint selected exactly one large review surface: the rank, pseudoinverse,
and dense low-rank test cluster in `tests/test_svd.c`. It records the candidate
ranking, selected-cluster boundary, preservation invariants, extraction design,
helper extraction passes, registration alignment, ownership guard, focused
regression, maintainer alignment, integrated validation, review hardening, and
final closeout.

## Definition Of Done Checklist

- [x] Created Sprint 201 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Ranked large source and test surfaces by review risk, ownership clarity,
      and extraction feasibility.
- [x] Selected exactly one bounded SVD test cluster instead of spreading work
      across several large files.
- [x] Recorded no-behavior-change, no-public-API, and no-ABI boundaries before
      code movement.
- [x] Moved selected SVD rank, pseudoinverse, and dense low-rank test bodies
      into `tests/test_svd_selected_helpers.h`.
- [x] Kept `tests/test_svd.c` as the proof-owner binary with selected
      `RUN_TEST(...)` registrations retained.
- [x] Added `make svd-helper-guard` and fixture-based Python guard regression
      coverage for ownership drift.
- [x] Ran focused SVD behavior checks, source-list/CMake/docs checks, adjacent
      helper guard checks, formatting, lint, full tests, and whitespace
      validation.
- [x] Preserved explicit residuals for remaining SVD clusters,
      partial-SVD corpus ownership, broader large-surface cleanup, public
      API/ABI, performance, package, platform, release, and state-of-the-art
      claims.

## What Went Well

1. **The selected cluster stayed bounded.** The sprint reduced one SVD test
   surface and did not absorb partial-SVD, graph, direct-solver, or broader
   large-file cleanup into the same claim.

2. **Behavior-preservation invariants came before code movement.** Day 4
   captured test names, expected values, tolerances, null/error behavior,
   proof-owner registration, and helper-only boundaries before extraction.

3. **The proof-owner binary remained stable.** `tests/test_svd.c` still owns
   the selected `RUN_TEST(...)` registrations while the helper header carries
   the moved implementations.

4. **The guard is drift-sensitive.** `scripts/check_svd_helper_guard.sh` and
   `tests/test_svd_helper_guard.py` catch missing helper includes, missing
   dependency includes, duplicate moved ownership, missing registrations, and
   accidental helper registration.

5. **Validation matched the changed surface.** Since `.c` and `.h` files
   changed, Day 12 ran `make format`, `make lint`, and `make test` in addition
   to focused and documentation checks.

6. **Documentation used selected-scope vocabulary.** Maintainer and planning
   surfaces describe one no-behavior-change SVD helper review-surface
   reduction without promoting broad correctness or support claims.

## What Didn't Go Well

1. **Evidence links needed late hardening.** Day 13 found tracking rows that
   referenced Sprint 201 evidence only through Day 11 even after integrated
   validation existed. Those rows were corrected through Day 14 before
   closeout.

2. **The selected surface still leaves a large SVD owner file.** The sprint
   reduced one meaningful cluster, but `tests/test_svd.c` still contains other
   large clusters that require separate ranking and invariants.

3. **Header-only helper ownership requires explicit guard wording.** Because
   `tests/test_svd_selected_helpers.h` is intentionally included only by the
   `test_svd` proof-owner binary, while `tests/test_svd_helpers.h` remains
   shared fixture support, the guard has to protect dependencies, registration
   order, explicit Makefile prerequisites, and CMake/library-manifest
   boundaries.

4. **Broad review-surface cleanup remains incomplete.** Sprint 201 closes one
   selected reviewability gap, not a repository-wide maintainability program.

## Final Metrics

### Validation

| Metric | Sprint 201 close state |
| --- | --- |
| selected SVD focused binary | passed on Days 10 and 12; Day 12 reported 114 tests, 0 failures, 0 skipped, and 2067 assertions |
| SVD helper ownership guard | passed on Days 9, 12, 13, and 14 |
| SVD helper guard regression test | passed on Days 9, 12, 13, and 14 |
| source-list check | passed on Day 12 with 49 library sources |
| CMake configure check | passed on Day 12 under ignored `build/sprint201-day12-validation-check` |
| docs check | passed on Days 12, 13, and 14 with 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages |
| adjacent QR helper guard | passed on Day 12 |
| format | passed on Day 12 |
| lint | passed on Day 12 |
| full test suite | passed on Day 12 with `All tests passed.` |
| final `git diff --check` | passed |
| final `make format && make lint && make test` | passed on Day 12 because `.c` and `.h` files changed during the sprint |

### Changed Surface

| Metric | Sprint 201 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Epic residual queue files changed | 1 |
| Public documentation files changed | 0 |
| Maintainer documentation files changed | 1 |
| Makefile targets added | 1 |
| Shell guard files added | 1 |
| Python guard files added | 1 |
| C implementation files changed | 0 |
| C test files changed | 1 |
| Test helper header files changed | 1 |
| Public or internal library header files changed | 0 |
| Public API/ABI declarations changed | 0 |
| CI workflow files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Candidate-ranking items completed | 1 |
| Cluster-selection items completed | 1 |
| Helper-extraction items completed | 1 |
| Ownership-guard items completed | 1 |
| Focused-regression items completed | 1 |
| Validation-and-docs items completed | 1 |
| Broad SVD behavior claims promoted | 0 |
| Public API/ABI claims promoted | 0 |
| Repository-wide review-surface claims promoted | 0 |

The count covers Sprint 201 items 201.1 through 201.6.

## Closed Claim

Sprint 201 closes this bounded claim:

The current branch reduces one selected SVD test review surface by moving the
rank, pseudoinverse, and dense low-rank test implementations from
`tests/test_svd.c` into `tests/test_svd_selected_helpers.h`. The shared fixture
helpers remain in `tests/test_svd_helpers.h`; the proof-owner binary remains
`tests/test_svd.c`; selected test registrations remain there and in their
existing order; both helper headers are Makefile prerequisites for
`build/test_svd`; the helper boundary is protected by `make svd-helper-guard`
and `tests/test_svd_helper_guard.py`; focused and full validation show selected
cluster behavior is preserved. Assertion source locations now follow the
helper-owned implementation file; the preservation claim is limited to selected
test names, status/error behavior, and emitted diagnostic text.

This claim does not include broad SVD correctness, new SVD algorithm
capability, partial-SVD corpus ownership, public API or ABI changes, library
implementation behavior changes, numerical tolerance changes, performance
improvements, package-manager support, platform support, release readiness,
state-of-the-art status, or repository-wide review-surface cleanup.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-large-surface-intake.md](./artifacts/day1-large-surface-intake.md);
- [day2-candidate-ranking.md](./artifacts/day2-candidate-ranking.md);
- [day3-selected-cluster-boundary.md](./artifacts/day3-selected-cluster-boundary.md);
- [day4-preservation-invariants.md](./artifacts/day4-preservation-invariants.md);
- [day5-extraction-design.md](./artifacts/day5-extraction-design.md);
- [day6-first-extraction-pass.md](./artifacts/day6-first-extraction-pass.md);
- [day7-cohesion-pass.md](./artifacts/day7-cohesion-pass.md);
- [day8-registration-alignment.md](./artifacts/day8-registration-alignment.md);
- [day9-ownership-guard.md](./artifacts/day9-ownership-guard.md);
- [day10-focused-regression.md](./artifacts/day10-focused-regression.md);
- [day11-maintainer-alignment.md](./artifacts/day11-maintainer-alignment.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-review-hardening.md](./artifacts/day13-review-hardening.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Remaining `tests/test_svd.c` review surfaces remain large | Future selected SVD cluster | Rank remaining clusters, select one bounded owner, record invariants, move only behavior-preserved helper bodies, add/update guards, and run focused plus full validation. |
| `tests/test_svd_partial_corpus.c` ownership remains unclaimed | Future selected partial-SVD owner | Record partial-SVD fixture/corpus invariants, define proof-owner registrations, add guard coverage, and run focused partial-SVD regression. |
| Shared helper dependency tracking remains local-only | Future helper dependency guard owner | Add a shared guard pattern only when more helper extractions justify common dependency tracking. |
| Other large graph/direct-solver/test surfaces remain unclaimed | Future selected review-surface sprint | Select one owner binary or module, record no-behavior-change boundaries, and close with guard and focused regression evidence. |
| Broad review-surface cleanup remains unclaimed | Future multi-sprint maintainability owner | Close enough selected surfaces with evidence before changing broad maintainability claims. |

## Next-Sprint Readiness

Sprint 201 leaves one selected SVD helper surface in a closed, guarded,
validated state.

| Future need | Sprint 201 handoff |
| --- | --- |
| Additional SVD test cleanup | Reuse the Day 1-Day 4 ranking and invariant pattern, but select a new bounded cluster. |
| Partial-SVD reviewability | Treat partial-SVD corpus evidence as separate from the selected Sprint 201 rank/pseudoinverse/dense-low-rank cluster. |
| Helper guard maintenance | Keep `scripts/check_svd_helper_guard.sh`, `tests/test_svd_helper_guard.py`, and `make svd-helper-guard` synchronized with selected moved markers. |
| Broader review-surface reduction | Continue closing one owner at a time rather than making broad cleanup claims. |
| Claim docs | Keep maintainer guide as the proof-owner ledger and planning docs as selected-sprint evidence, with public docs unchanged unless user-facing behavior changes. |

## Final Assessment

Sprint 201 is complete as a selected SVD helper review-surface reduction
sprint. It improves local test maintainability for one bounded cluster,
protects the new boundary with guards, validates behavior preservation, and
keeps all broader correctness, support, performance, and state-of-the-art
claims out of scope.

The branch is ready for review as test helper extraction, focused guard,
registration, maintainer documentation, and planning evidence work for the
selected SVD rank, pseudoinverse, and dense low-rank test cluster.
