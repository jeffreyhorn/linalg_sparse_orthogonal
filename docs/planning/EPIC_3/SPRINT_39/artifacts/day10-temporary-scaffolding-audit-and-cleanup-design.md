# Sprint 39 Day 10: Temporary Scaffolding Audit & Cleanup Design

## Purpose

Identify which remaining transitional notes, helper comments, or closeout-era
allowances should not survive Epic 3 closeout as permanent operator-facing
clutter, while preserving the comments and helper structure that still carry
real maintenance value.

## Audit Result

Sprint 39 does **not** end with a broad temporary-scaffolding deletion queue.
The main residual cleanup target is much narrower:

- sprint-day provenance comments inside permanent operator-facing files

The audit does **not** support removing:

- `docs/planning/EPIC_3/**` sprint plans, handoffs, retrospectives, or
  artifacts
- README feature-history/performance provenance that still gives useful
  technical context
- behavior-bearing comments about toolchain quirks, platform exclusions, or
  runtime/coverage limitations

## Current Candidate Surfaces

### Highest-value cleanup candidates

- `Makefile`
  - `Sprint 31 Day 11` provenance in compile-only tooling comments
  - `Sprint 37 Day 4/7` provenance in quality-target ownership comments
  - `Sprint 30 Day 7` provenance in warning-workflow comments
  - `Sprint 33 Day 5` provenance in dead-code workflow comments
- `.github/workflows/ci.yml`
  - `Sprint 29 Day 13` provenance in supplemental bench-fast comments
  - `Sprint 34 Day 10-12` provenance in dead-code serial-job comments

### Explicit non-candidates for Day 11

- `docs/planning/EPIC_3/**`
  - intended historical evidence, not clutter
- `README.md` feature-history sections
  - still useful measured/provenance context
- dead-code serialized-execution contract
  - real workflow limitation, not mere wording residue

## Keep / Remove / Defer

### Keep

- sprint artifacts in `docs/planning/EPIC_3/**`
- comments that explain:
  - Xcode / ld64 behavior
  - Apple Clang vs `lcov` / `gcovr` coverage behavior
  - TSan / libomp limitations
  - current dead-code serialized execution contract
  - current Windows/macOS staged or excluded surfaces

### Remove or compress in Day 11

- permanent-file sprint-day provenance prefixes where the surrounding comment
  already explains the behavior well enough without “Sprint XX Day YY” wording

### Defer

- any behavior change to dead-code topology or platform coverage
- any broad README historical rewrite
- any attempt to purge useful sprint provenance from non-operator technical
  narrative

## Chosen Day 11 Batch

Apply a narrow permanent-comment cleanup batch:

- in `Makefile`
  - compress sprint-day provenance comments into stable behavior-oriented
    wording
- in `.github/workflows/ci.yml`
  - compress the same class of sprint-day provenance comments where the job/step
    intent is already clear
- preserve all behavior, targets, workflow steps, and current enforced/staged
  contracts

## Validation Plan For Day 11

Because the chosen batch is comment-only in permanent files, the validation
surface should be direct and lightweight:

- `make -n quality-review-compile`
- `make -n quality-review-full`
- YAML parse for touched workflow files

If the batch remains comments-only, the full `make format && make lint && make
test` gate will not be required.

## Expected Post-Day-11 State

If the batch lands cleanly:

- permanent operator-facing files will carry less sprint-implementation residue
- the repo will keep its historical engineering evidence in `docs/planning/`
  where it belongs
- the final Epic 3 closeout can talk about stable contracts rather than
  lingering day-by-day provenance in the operator surfaces
