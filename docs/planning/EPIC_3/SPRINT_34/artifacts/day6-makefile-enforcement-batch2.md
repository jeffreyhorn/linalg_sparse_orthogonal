# Sprint 34 Day 6 Makefile Enforcement Batch II

**Date:** 2026-05-19  
**Branch:** `sprint-34`

## Objective

Finish the local Sprint 34 phase-1 reviewed-quality path by tightening the
Makefile graph around the serial dead-code constraint, documenting the wrapper
targets in the maintained README, and revalidating the full reviewed path.

## Changes

### Makefile

Added explicit non-parallel handling:

- `.NOTPARALLEL: quality-review-compile quality-review deadcode deadcode-report deadcode-check`

Current relevant anchors:

- `.NOTPARALLEL`: line `454`
- `quality-review-compile`: line `456`
- `quality-review`: line `463`

Purpose:

- keep the reviewed wrapper path serial even under `make -j`
- make the Sprint 33 shared dead-code build/artifact constraint visible in the
  target graph itself, not only in sprint notes

### README

Updated the maintained Makefile usage docs to include:

- `make quality-review-compile`
- `make quality-review`

Also added a dedicated "Reviewed Local Quality Path" section that explains:

- the exact wrapper sequence for each target
- that the wrappers are additive
- that `check`, `lint`, `test`, and `deadcode-check` retain their existing
  meanings

## Validation

### Dry-run graph check

- `make -n -j2 quality-review`

Result:

- still showed the intended serial banner sequence:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`

### Live reviewed-path check

- `make -j2 quality-review`

Result:

- passed

Important observations:

- the reviewed path remained bannered and attributable
- the wrapper still flowed through the inherited `lint` tooling-build gate
- `deadcode-check` remained the terminal step
- because the dead-code report stamp was already current, the final phase did
  not rerun the whole raw workflow and completed directly with:
  - `deadcode-check: report completeness checks passed.`

## Remaining Limitation

The Makefile graph is tighter, but the Sprint 33 shared-path constraint still
exists at the process level:

- `build/deadcode-cmake`
- `build/deadcode/`

So separate concurrent shell invocations of `deadcode*` can still race. Sprint
34's local reviewed wrapper no longer introduces a parallel sibling path for
them, but the broader build-tree isolation work remains a later-sprint concern.

## Day 6 Conclusion

Sprint 34's local reviewed-quality path is now complete enough to hand forward:

- the wrapper targets exist
- the serial dead-code constraint is encoded in the Makefile
- the README documents the new local contract
- the full reviewed path still passes end to end on the updated graph
