# Sprint 59 Day 4 - quality follow-through design

Date: 2026-06-08
Branch: `sprint-59`

## Scope

Freeze the first bounded Sprint 59 quality/platform landing boundary by
selecting the highest-value residual seam from Day 3, defining the exact
touched surfaces, and recording the preserved invariants and non-goal fence
before any maintained contract edits land.

## Selected first landing

### Primary target: residual-disposition reconciliation

The first Sprint 59 follow-through batch should land on the residual seam that
already needs bounded follow-through now:

- cross-surface residual-disposition wording reconciliation

This batch should make the current final-sprint quality/platform story easier
to read without pretending staged limits are gone.

### Touched surfaces

The first landing should cover:

- `README.md`
  - compact operator-facing residual summary
  - concise cross-platform staged/enforced wording
- `docs/maintainer_guide.md`
  - maintainer-facing interpretation of the remaining staged/excluded limits
- `Makefile`
  - executable/help wording only where the current residual interpretation or
    rerun guidance can be made clearer without changing the targets

Workflow comments should only be touched if they materially disagree with the
three maintained contract surfaces above:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

## Deferred from the first landing

The first Sprint 59 follow-through batch should intentionally defer:

- running macOS dead-code in CI
- expanding Windows to full Makefile reviewed-wrapper parity
- expanding Windows dead-code execution
- dead-code path/topology redesign
- coverage-policy rewrite

Those items either:

- still need measurement first
- or are already acceptable deferred residuals rather than truthfulness defects

## Invariants

The Day 5 batch must preserve:

- reviewed baseline wording
  - `make quality-review-full` remains the strongest local reviewed baseline
- Makefile/CMake parity truthfulness
  - `ctest -N --test-dir build/quality-review-cmake` remains the main parity
    anchor
- platform-story honesty
  - Linux remains the enforced reviewed source-of-truth path
  - macOS dead-code remains staged unless fresh evidence changes that
  - Windows reviewed CMake subset remains enforced while wrappers/dead-code
    stay staged
- stable local developer workflow
  - no target renames
  - no new target taxonomy
  - no command-surface redesign

## Cleanup policy

For the first quality/platform batch:

- minimize blast radius
- prefer measurement-backed wording or already-proven live CI facts
- remove ambiguity before adding explanation
- keep repo-wide interpretation in `docs/maintainer_guide.md`
- keep executable command detail in `Makefile`
- keep operator-facing platform snapshots concise in `README.md`
- avoid broad platform abstraction redesign

## Day 5 landing checklist

1. Tighten residual-disposition wording across the selected maintained
   surfaces.
2. Preserve serialized dead-code execution as an explicit current deferred
   operational limit.
3. Preserve macOS dead-code as staged pending measurement.
4. Preserve Windows reviewed-wrapper/dead-code as staged rather than silently
   incomplete.
5. Remove coverage from any wording that still implies it is an unresolved
   active residual.
6. Keep workflow comments aligned only where needed for agreement.
7. Avoid any behavior change that would widen the batch into platform
   implementation work.

## Conclusion

Sprint 59 now has an exact first follow-through boundary:

- first target:
  - residual-disposition reconciliation across maintained quality/platform
    surfaces
- likely touched files:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - workflow comments only if needed
- preserved invariants:
  - reviewed baseline wording
  - parity truthfulness
  - platform-story honesty
  - stable local developer workflow
- explicit non-goals:
  - no macOS dead-code enablement yet
  - no Windows wrapper/dead-code expansion
  - no coverage-policy rewrite
  - no topology redesign

That is enough to move to Day 5 implementation from one bounded residual seam
instead of a generic final-sprint cleanup bucket.
