# Sprint 39 Day 2 Artifact: Final Warning Audit

## Purpose

Map the final Epic 3 warning-clean contract to the actual authoritative build
surfaces so Sprint 39 does not overstate what the maintained reviewed wrappers
prove.

## Day 2 Bottom Line

Sprint 39 Day 2 did **not** surface a known warning regression. The dominant
remaining warning-closeout work is about preserving the correct evidence model.

The key distinction remains:

- **authoritative repository-wide warning proof**
  - Sprint 30 Apple Clang CMake full-tree inventory
  - reproducible through `make warning-workflow WARNING_WORKFLOW_LABEL=label`
- **narrower secondary cross-check**
  - Makefile `all` library-only build
- **routine maintained local reviewed baseline**
  - `make quality-review-full`
  - plus its underlying `make quality-review` and `make quality-review-cmake`

## Authoritative vs Supporting Warning Surfaces

### Authoritative

- Sprint 30 compile-hygiene playbook
- Sprint 30 rebuild/warning workflow
- Apple Clang CMake full-tree warning inventory

Use this tier when claiming:

- repository-wide warning cleanliness
- closure of a warning class across the full tree
- before/after warning counts by area/class/file

### Supporting but narrower

- `make lint`
  - strict `src/*.c` compile with `-Werror`
  - `clang-tidy`
  - `cppcheck`
  - `tooling-build`
- `make quality-review-full`
  - reviewed Makefile path
  - reviewed CMake parity path
- CI platform matrix

Use this tier when claiming:

- routine local regression protection
- reviewed local baseline integrity
- current direct/reviewed paths still pass

Do **not** use this tier alone when claiming:

- repository-wide warning inventory is zero
- the whole tree has been re-audited at warning-class granularity

## Current Day 2 Assessment

### What is already in good shape

- No new stale top-level warning-count claims were found in the current README
  command map.
- `warning-workflow` is still explicitly labeled in `Makefile` as the
  reproducible Epic 3 warning-capture workflow.
- Sprint 38’s inherited final validation already records the strongest local
  reviewed baseline separately from dead-code/reporting semantics.

### What remains risky

- Final Epic 3 closeout language could accidentally collapse:
  - full-tree warning proof
  - narrower Makefile cross-checks
  - strongest local reviewed baseline

That would be a truthfulness bug even if the code itself has no new warnings.

## Day 5 Likely Implementation Shape

Unless a stronger warning rerun surfaces a real regression, the expected
warning-closeout batch is narrow:

1. preserve the Sprint 30 authority model explicitly
2. align final Epic 3 standards/summary wording with that model
3. avoid broad warning-doc churn or fake simplification

## Immediate Guidance For Later Sprint 39 Work

- Use the Sprint 30 warning workflow as the authoritative reference for final
  warning claims.
- Treat `make quality-review-full` as the strongest routine local reviewed
  baseline, not as a replacement for full-tree warning inventory.
- Keep Makefile `all` framed as a narrower cross-check.
