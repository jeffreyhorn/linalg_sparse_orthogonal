# Sprint 71 Day 2: Validation Baseline and Truth-Surface Recheck

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Reconfirm the docs-only validation contract and the exact truth surfaces that
Sprint 71 cleanup must preserve before the sprint starts rewriting public or
reference-facing wording.

## Reviewed Baseline

Sprint 71 still starts from:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

## Validation Split

Sprint 71 now fixes the following validation split:

- docs-only days:
  - targeted sanity checks only
- bounded `*.c` / `*.h` days, if they appear later:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial architecture, capability, backend, or
  platform work in later sprints:
  - `make quality-review-full`

## Preserved Truth-Surface Checklist

`README.md` must preserve:

- the orthogonal linked-list public center as the shipped current product
  reading
- examples vs benchmarks vs tests ownership
- the threshold-free reading of `make bench-canonical-report`
- the current platform-confidence summary

`INSTALL.md` must preserve:

- static-first install/release shape
- reviewed Linux/macOS/Windows lane asymmetry
- local install/package regression ownership without promoting it to a broad
  reviewed install-validation claim

`docs/maintainer_guide.md` must remain:

- the main policy authority
- the home for deeper rationale and deferred-queue reading that should not
  stay duplicated in user-facing docs

`benchmarks/README.md` must preserve:

- benchmarks as workflow/performance proof surfaces
- tests as regression/oracle/property owners

`examples/README.md` must preserve:

- examples as adoption and workflow-teaching surfaces
- no benchmark- or test-owned guarantee widening

## Docs-Only Sanity Set

The maintained Sprint 71 docs-only sanity set is now:

1. diff review on touched public/reference/support surfaces
2. terminology/alignment scans on:
   - workflow ownership
   - benchmark/test/example authority
   - static-first packaging and reviewed-platform wording
3. touched-surface `wc -l` checks where snapshot measurements are recorded
4. branch-state rechecks after each landing batch

## Confirmed Day 3 Audit Targets

The Day 2 reread confirms the strongest Day 3 public-audit targets remain:

- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

`docs/maintainer_guide.md` remains support-first unless the later cleanup
truly forces it to move.

## Exit State

Sprint 71 Day 2 closes with one explicit docs-only validation and
truth-surface contract:

1. strongest local reviewed baseline remains unchanged
2. docs-only, code-day, and substantial-day validation expectations are all
   explicit
3. preserved public/install/benchmark/example/policy truth surfaces are fixed
   in writing
4. the targeted Sprint 71 sanity set is defined before deeper audit begins
