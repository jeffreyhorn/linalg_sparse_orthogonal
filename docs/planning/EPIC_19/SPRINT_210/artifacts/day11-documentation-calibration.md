# Sprint 210 Day 11: Documentation Calibration

## Scope

Day 11 calibrates user-facing, support/readiness, and maintainer documentation
for the selected no-reorder linked-list LDLT allocation-failure proof added
during Sprint 210. The documentation records one additional selected owner
without promoting broad allocation-failure, direct-solver, package/install,
platform, performance, release, external-library parity, or state-of-the-art
reliability claims.

## Claim Surfaces Updated

| File | Day 11 update |
| --- | --- |
| `README.md` | Adds the linked-list LDLT focused gate to selected allocation-failure proof wording, the command list, and repeated-run reliability boundary notes. |
| `INSTALL.md` | Updates the support-readiness matrix row for local selected allocation-failure proof to include the linked-list LDLT gate and retained non-claims. |
| `docs/maintainer_guide.md` | Extends the reliability proof-owner ledger with Sprint 210 linked-list LDLT tests, gate, guard, artifacts, and selected-only interpretation. |
| `docs/planning/EPIC_19/SPRINT_210/WORKING_NOTES.md` | Records the earned claim, retained non-claims, changed surfaces, and validation commands. |

## Earned Claim

The earned claim is:

Selected no-reorder linked-list LDLT numeric factorization has focused local
deterministic allocation-failure proof for bounded known fixtures covering 25
injected allocation-failure sites, cleanup, stale-output suppression,
caller-input preservation, free-safe output state, repeated cleanup after
failure, and retry-after-reset behavior.

## Retained Non-Claims

The Day 11 docs retain these non-claims:

- broad allocation-failure coverage across the library;
- CSC LDLT allocation-failure proof;
- reordered LDLT allocation-failure proof;
- Cholesky, broad direct solver, QR, SVD, eigensolver, sparse matrix
  construction, conversion, IO, package/install, or generated-tooling
  allocation-failure proof;
- operating-system OOM behavior;
- platform parity or hosted CI proof for this selected owner;
- package-manager proof, shared-library ABI proof, performance proof, release
  readiness, or state-of-the-art reliability support;
- concurrent allocation-hook behavior.

## Evidence Links

| Evidence | Path |
| --- | --- |
| Owner selection | `docs/planning/EPIC_19/SPRINT_210/artifacts/day2-owner-ranking.md` |
| Lifecycle baseline | `docs/planning/EPIC_19/SPRINT_210/artifacts/day3-lifecycle-baseline.md` |
| Harness design | `docs/planning/EPIC_19/SPRINT_210/artifacts/day4-harness-design.md` |
| Harness implementation | `docs/planning/EPIC_19/SPRINT_210/artifacts/day5-harness-implementation.md` |
| Failure sweep | `docs/planning/EPIC_19/SPRINT_210/artifacts/day6-failure-sweep.md` |
| Cleanup proof | `docs/planning/EPIC_19/SPRINT_210/artifacts/day7-cleanup-proof.md` |
| Stale-output preservation | `docs/planning/EPIC_19/SPRINT_210/artifacts/day8-stale-output-preservation.md` |
| Retry proof | `docs/planning/EPIC_19/SPRINT_210/artifacts/day9-retry-proof.md` |
| Focused gate | `docs/planning/EPIC_19/SPRINT_210/artifacts/day10-focused-gate.md` |

## Validation

Day 11 changed documentation only. No `.c` or `.h` files were modified by the
Day 11 claim-calibration edits; the full C quality gate already passed after
the Day 10 code and build-surface updates.

Commands run:

```sh
make ldlt-linked-list-allocation-failure-gate
python3 tests/test_ldlt_allocation_failure_gate_registration.py
make docs-check
make support-docs-guard
git diff --check
```

Results:

| Command | Result | Evidence |
| --- | --- | --- |
| `make ldlt-linked-list-allocation-failure-gate` | PASS | `95` LDLT tests, `0` failures, `0` skips, and `7781` assertions. |
| `python3 tests/test_ldlt_allocation_failure_gate_registration.py` | PASS | `ldlt-allocation-failure-gate-registration: passed`. |
| `make docs-check` | PASS | Doxygen generation and API docs coverage completed; 18 checked-in public headers, 18 generated reference pages, 18 generated source pages, and `sparse_version.h` kept under its separate installed-header policy. |
| `make support-docs-guard` | PASS | `test-support-quick-reference-docs: ok`. |
| `git diff --check` | PASS | Whitespace validation completed after documentation updates. |
