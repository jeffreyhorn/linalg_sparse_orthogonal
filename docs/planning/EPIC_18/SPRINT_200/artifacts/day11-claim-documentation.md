# Sprint 200 Day 11 Claim Documentation

## Scope

Day 11 calibrates user-facing, maintainer-facing, and planning claim surfaces
for the selected `sparse_symbolic_lu()` allocation-failure proof added during
Sprint 200. The documentation now records one additional selected owner without
promoting broad allocation-failure, OS OOM, platform, package/install,
generated-tooling, or state-of-the-art reliability claims.

## Claim Surfaces Updated

| File | Day 11 update |
| --- | --- |
| `README.md` | Adds `sparse_symbolic_lu()` to selected allocation-failure proof wording, documents `make symbolic-lu-allocation-failure-gate`, and keeps broad reliability non-claims. |
| `INSTALL.md` | Updates the support-readiness matrix row for local selected allocation-failure proof to include the selected symbolic LU gate and owner boundary. |
| `docs/maintainer_guide.md` | Extends the reliability proof-owner ledger with Sprint 200 symbolic LU tests, gate, guard, artifacts, and retained non-claims. |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Updates the interim Sprint 200 status from pending future execution to in progress through Day 11. |
| `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` | Updates the additional allocation-failure residual from pending future execution to in-progress selected symbolic LU proof. |
| `docs/planning/EPIC_18/EPIC_18_RETROSPECTIVE.md` | Updates the Sprint 200 status row and status metrics so the current branch no longer says no Sprint 200 artifacts exist. |

## Earned Claim

The earned claim is:

`sparse_symbolic_lu()` has focused local deterministic allocation-failure proof
for selected bounded fixtures covering allocation-failure status,
requested-output cleanup, stale-output suppression, caller-owned
matrix/permutation preservation, repeated cleanup after failure, and
retry-after-reset behavior.

## Retained Non-Claims

The Day 11 docs retain these non-claims:

- broad allocation-failure coverage across the library;
- `sparse_analyze()` lifecycle allocation-failure coverage;
- standalone etree, postorder, or colcount helper allocation-failure coverage;
- direct solver, eigensolver, graph, SVD, sparse matrix construction,
  conversion, IO, package/install, or generated-tooling allocation-failure
  proof;
- operating-system OOM behavior;
- platform parity or hosted CI proof for this selected owner;
- concurrent allocation-hook behavior;
- state-of-the-art reliability support.

## Evidence Links

| Evidence | Path |
| --- | --- |
| Owner selection | `docs/planning/EPIC_18/SPRINT_200/artifacts/day2-owner-selection.md` |
| Invariant record | `docs/planning/EPIC_18/SPRINT_200/artifacts/day4-invariant-record.md` |
| Failed-allocation tests | `docs/planning/EPIC_18/SPRINT_200/artifacts/day7-failed-allocation-tests.md` |
| Cleanup proof | `docs/planning/EPIC_18/SPRINT_200/artifacts/day8-cleanup-proof.md` |
| Retry proof | `docs/planning/EPIC_18/SPRINT_200/artifacts/day9-retry-proof.md` |
| Focused gate | `docs/planning/EPIC_18/SPRINT_200/artifacts/day10-focused-gate.md` |

## Validation

Day 11 changed documentation only. No `.c` or `.h` files were modified by the
Day 11 claim-calibration edits, so the full C quality gate is deferred to Day
12 integrated validation.

Commands run:

```sh
make docs-check
python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py
git diff --check
```

Results:

| Command | Result | Evidence |
| --- | --- | --- |
| `make docs-check` | PASS | Doxygen generation and API docs coverage completed; 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages were verified. |
| `python3 tests/test_symbolic_lu_allocation_failure_gate_registration.py` | PASS | `symbolic-lu-allocation-failure-gate-registration: passed`. |
| `git diff --check` | PASS | Whitespace validation completed after documentation updates. |
