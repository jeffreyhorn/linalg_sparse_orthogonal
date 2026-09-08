# Sprint 200 Day 14 Closeout Review

## Scope

Day 14 finalizes Sprint 200 evidence, item status, residuals, and retrospective
inputs for the additional allocation-failure owner proof. The selected owner
is exactly `sparse_symbolic_lu()`.

## Final Item Status

| Item | Final status | Evidence |
| --- | --- | --- |
| 200.1 Owner Selection | Complete | `day1-candidate-intake.md`; `day2-owner-selection.md`; selected owner is `sparse_symbolic_lu()` and rejected candidates are deferred explicitly. |
| 200.2 Invariant Record | Complete | `day3-lifecycle-trace.md`; `day4-invariant-record.md`; `day13-review-hardening.md` maps invariants to tests and guards. |
| 200.3 Harness Integration | Complete | `day5-harness-design.md`; `day6-harness-integration.md`; selected direct allocations use existing private allocation wrappers, and hook reset behavior is tested. |
| 200.4 Regression Tests | Complete | `day7-failed-allocation-tests.md`; `day8-cleanup-proof.md`; `day9-retry-proof.md`; `tests/test_etree.c` covers selected failure status, cleanup, stale-output suppression, caller-owned input preservation, and retry. |
| 200.5 Focused Gate | Complete | `day10-focused-gate.md`; `Makefile`; `tests/test_symbolic_lu_allocation_failure_gate_registration.py`; selected gate is `make symbolic-lu-allocation-failure-gate`. |
| 200.6 Docs And Validation | Complete | `day11-claim-documentation.md`; `day12-integrated-validation.md`; `day13-review-hardening.md`; README, INSTALL, maintainer guide, project-plan, residual queue, and Epic retrospective wording are claim-safe. |

## Completed Work

Sprint 200 completed one additional selected allocation-failure owner proof:

- selected `sparse_symbolic_lu()` as the owner after ranking reliability
  candidates;
- documented cleanup, publication, stale-output, retry, caller-owned input,
  hook, and scope invariants before implementation;
- converted selected owner-owned allocation sites to existing private
  allocation wrappers so deterministic failure injection reaches them;
- added deterministic regression tests for allocation failure, cleanup,
  stale-output suppression, caller-owned input preservation, repeated cleanup,
  and retry-after-reset behavior;
- added `make symbolic-lu-allocation-failure-gate` and a Python registration
  guard;
- calibrated README, INSTALL, maintainer, project-plan, residual queue, and
  Epic retrospective wording to the earned selected-owner claim;
- recorded integrated validation and review-hardening evidence.

## Final Validation Ledger

| Command | Result | Evidence |
| --- | --- | --- |
| `make symbolic-lu-allocation-failure-gate` | PASS | Day 12 and Day 13 runs reported 3 tests, 0 failures, 0 skipped, and 3054 assertions. |
| `make symbolic-allocation-failure-gate` | PASS | Day 12 run reported 104 tests, 0 failures, 0 skipped, and 4316 assertions. |
| `make source-list-check` | PASS | Day 12 source-list guard reported 49 library sources. |
| `make docs-check` | PASS | Day 12 and Day 13 runs verified Doxygen generation and API docs coverage for 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages. |
| `make format` | PASS | Day 12 formatting completed across source, test, benchmark, example, and public header surfaces. |
| `make lint` | PASS | Day 12 strict compile, clang-tidy, and cppcheck completed successfully. |
| `make test` | PASS | Day 12 full test suite completed with `All tests passed.` |
| `python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py` | PASS | Day 14 rerun reported `symbolic-lu-allocation-failure-gate-registration: passed`. |
| `git diff --check` | PASS | Day 12, Day 13, and Day 14 whitespace checks passed. |

## Known Residuals

The Sprint 200 proof does not claim:

- broad allocation-failure coverage across the library;
- `sparse_analyze()` lifecycle cleanup;
- standalone etree, postorder, or colcount helper allocation-failure coverage;
- direct solver, eigensolver, graph, SVD, sparse matrix construction,
  conversion, IO, package/install, generated-tooling, or platform allocation
  reliability;
- operating-system OOM behavior;
- concurrent allocation-hook behavior;
- hosted CI proof for this selected owner;
- state-of-the-art reliability support.

## Follow-Up Candidates

| Candidate | Recommended disposition |
| --- | --- |
| `sparse_analyze()` lifecycle cleanup | Keep as a future selected-owner reliability sprint; scope must be separate from symbolic LU. |
| Standalone etree/postorder/colcount helpers | Split by helper family before proof work to avoid broad symbolic-analysis claims. |
| Direct-solver output publication | Treat as a separate owner with its own stale-output and retry contract. |
| Matrix construction/insertion allocation failures | Keep separate from symbolic LU because `sparse_create()` and `sparse_insert()` are shared construction surfaces. |
| Hosted allocation-failure evidence | Defer until a workflow explicitly runs selected reliability gates and claim docs are updated together. |

## Retrospective Inputs

| Area | Input |
| --- | --- |
| Completed work | One selected symbolic LU allocation-failure owner proof is complete and locally validated. |
| Validation | Focused gate, broader symbolic gate, source-list, docs, format, lint, full tests, and whitespace checks passed before closeout. |
| Deviations | A separate selected symbolic LU gate was added rather than folding the selected-only path into the broader symbolic gate, keeping review vocabulary precise. |
| Deferred breadth | Analysis lifecycle, helper-level symbolic routines, direct solvers, matrix construction, platform, package/install, generated tooling, and hosted CI proof remain out of scope. |
| Recommendation | Keep future allocation-failure sprints selected-owner scoped and require a Day 4 invariant record before code edits. |

## Intended Worktree Surface

The intended Sprint 200 change surface is:

- `src/sparse_etree.c`;
- `tests/test_etree.c`;
- `tests/test_symbolic_lu_allocation_failure_gate_registration.py`;
- `tests/test_symbolic_allocation_failure_gate_registration.py`;
- `Makefile`;
- README, INSTALL, maintainer guide, Epic 18 planning/status files, and
  `SPRINT_200` artifacts.

No unrelated worktree surface is intentionally part of Sprint 200.

## Day 14 Validation

Commands run after the closeout artifact and status updates:

```sh
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
make symbolic-lu-allocation-failure-gate
make docs-check
git diff --check
```

Results:

- `python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py`:
  PASS.
- `make symbolic-lu-allocation-failure-gate`: PASS; selected gate reported 3
  tests, 0 failures, 0 skipped, and 3054 assertions.
- `make docs-check`: PASS; Doxygen and API docs coverage completed with 18
  checked-in public headers, 18 generated reference pages, and 18 generated
  source pages.
- `git diff --check`: PASS.
