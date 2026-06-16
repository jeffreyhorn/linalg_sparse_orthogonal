# Sprint 73 Day 11: Docs / Maintainer / Header Follow-Through Batch

Date: 2026-06-16
Branch: `sprint-73`

## Purpose

Align the maintained policy wording with the landed Sprint 73 configuration
contract while staying inside the Day 10 bounded touch set.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/PLAN.md`
- `docs/planning/EPIC_7/SPRINT_73/artifacts/day10-docs-maintainer-header-follow-through-design.md`
- `docs/maintainer_guide.md`
- `include/sparse_analysis.h`

## Day 11 Batch Result

### 1. The maintainer guide now states the landed ownership split directly

Touched maintained surface:

- `docs/maintainer_guide.md`

The residual configuration queue now distinguishes:

- compatibility-first FM policy overrides:
  - `SPARSE_FM_FINEST_STRATEGY`
  - `SPARSE_FM_ENSEMBLE_STRATEGIES`
  - `SPARSE_FM_FINEST_PASSES`
  - `SPARSE_FM_INTERMEDIATE_PASSES`
  - `SPARSE_FM_ANNEALING_SCHEDULE`
  - `SPARSE_FM_THICK_RESTART_PERTURB`
  - `SPARSE_FM_GAIN_NOISE_SCHEDULE`
- developer-only FM debug flags:
  - `SPARSE_FM_ENSEMBLE_DEBUG`
  - `SPARSE_FM_THICK_RESTART_DEBUG`
  - `SPARSE_FM_ANNEALING_DEBUG`
  - `SPARSE_FM_GAIN_NOISE_DEBUG`
- the explicitly deferred debug/profile lane:
  - `SPARSE_ND_PROFILE`
  - `SPARSE_QG_PROFILE`
  - `SPARSE_HCC_DEBUG`

The landed interpretation is now direct:

- recognized FM compatibility env vars parse once at the graph orchestration
  boundary
- they lower into one internal typed FM policy/runtime contract
- the refinement subsystem is no longer a second independent FM parser
- that narrower internal ownership still does not create a public typed FM
  option family

### 2. The support-only public header remained accurate

Untouched support surface:

- `include/sparse_analysis.h`

It already remained truthful because it still says:

- lower-level FM tuning and debug/profile env vars remain internal or
  compatibility-only for now

That wording continues to match the landed code, so no header churn was
necessary.

### 3. The Day 10 fence stayed intact

The batch did not widen into:

- `src/sparse_reorder_amd_qg.c`
- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `examples/README.md`
- `benchmarks/README.md`

## Sanity Checks

This was a docs-only batch, so I did not run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

I used the targeted docs-only sanity set instead:

- touched-surface diff review
- terminology/alignment check
- touched-surface `wc -l`
- branch-status recheck

Raw `wc -l` counts after the landing:

- `docs/maintainer_guide.md` = `601`
- `include/sparse_analysis.h` = `499`

## Exit State

Sprint 73 Day 11 closes with:

1. one maintained-surface follow-through batch in `docs/maintainer_guide.md`
2. no required `include/sparse_analysis.h` follow-through
3. the FM compatibility-vs-internal-vs-debug split stated directly
4. the `SPARSE_QG_PROFILE` lane still explicitly deferred as support-only
