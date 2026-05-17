# Sprint 30 Day 6 Compile-Hygiene Playbook

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Turn the Sprint 30 warning baseline and cleanup observations into an explicit compile-hygiene policy that later Epic 3 work can apply consistently.

## Deliverable

Created:

- `docs/planning/EPIC_3/SPRINT_30/COMPILE_HYGIENE_PLAYBOOK.md`

The playbook defines:

- which build path is authoritative for the current full-tree baseline
- which warning classes are must-fix now versus temporarily backlog-acceptable
- area-specific expectations for `src/`, `tests/`, `benchmarks/`, `examples`, and generated artifacts
- how to treat compiler-specific warnings versus portable warnings
- what evidence is required before claiming a warning class is fully addressed
- the day-end validation standard for future Sprint 30 commits

## Key Decisions Captured

### Repository-wide baseline authority

- The clean Apple Clang CMake build remains the authoritative full-tree warning inventory.
- The Makefile `all` path remains a useful library-only cross-check, but not a substitute for full-tree warning accounting.

### Immediate versus deferred warning policy

- `src/` warnings are not acceptable.
- New warnings anywhere are not acceptable.
- Pre-existing auxiliary warnings may remain temporarily only when they are already measured and explicitly queued.
- Global `-Werror` remains out of scope until the baseline debt is genuinely reduced.

### Area-specific quality bar

- Core library code must stay warning-free.
- Tests, benchmarks, and examples may carry documented debt temporarily, but touched files should not regress.
- Public-facing examples and benchmark tools must stay aligned with current enums and public option-struct style expectations.

### Evidence standard

Any future claim that a warning class is fixed must include:

- before/after counts
- named build-path scope
- captured logs
- proportional regression validation
- updated Sprint 30 notes

## Links

- [Compile-Hygiene Playbook](../COMPILE_HYGIENE_PLAYBOOK.md)
- [Sprint 30 Working Notes](../WORKING_NOTES.md)
- [Day 1 warning baseline](./day1-warning-baseline.md)
- [Day 2 warning taxonomy](./day2-warning-taxonomy.md)

## Validation Performed

End-of-day validation sequence completed successfully:

- `make format`
- `make lint`
- `make test`

## Day 6 Conclusion

Day 6 did not change library behavior. It converted the Sprint 30 baseline work into an explicit decision framework so later cleanup and eventual warning gating can be defended with the same rules across compilers, build paths, and code areas.
