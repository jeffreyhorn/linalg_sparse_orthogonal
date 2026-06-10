# Sprint 61 Day 12 - Docs & Maintainer Story Update

Date: 2026-06-09
Branch: sprint-61

## Objective

Align the highest-value public and maintainer-facing surfaces with the landed
Sprint 61 Phase 1 typed configuration model:

- make the preferred typed path explicit
- keep the precedence story coherent
- name the residual deferred env-var queue directly

## Landed Surface

Touched files:

- `README.md`
- `include/sparse_analysis.h`
- `docs/maintainer_guide.md`

This stayed inside the planned Day 12 fence. No implementation, test logic,
example code, or benchmark driver behavior changed.

## Main Result

### 1. `README.md` now points callers at the shipped typed analysis/reorder path

The fill-reducing reordering summary now:

- names `sparse_analysis_opts_t.reorder_opts` as the public typed
  analysis-time control surface
- describes legacy `SPARSE_SUPERNODAL_POSTORDER` / `SPARSE_ND_*` env vars as
  compatibility overrides only when the typed field is unspecified
- removes the remaining sprint-local framing from that reordering paragraph

This makes the top-level product story match the actual landed control plane
from Days 6-11.

### 2. `include/sparse_analysis.h` now states the Phase 1 fence more directly

The public header now says the quiet part explicitly:

- the typed public surface is intentionally limited to caller-meaningful
  analysis-time routing and ND policy
- lower-level FM tuning and debug/profile env vars remain internal or
  compatibility-only for now
- `sparse_analysis_opts_t.reorder_opts` now states the shipped precedence rule
  directly:
  1. explicit typed value
  2. legacy compatibility env var when unspecified
  3. internal default fallback

That keeps the public header aligned with the Day 11 proof surface instead of
making callers infer the full model from sprint-local notes or implementation
details.

### 3. `docs/maintainer_guide.md` now owns the residual deferred configuration queue

The maintainer guide now has an explicit configuration-surface ownership
section covering:

- preferred typed path
- precedence
- interpretation of legacy env vars
- residual deferred queue

The queue is now named directly:

- compatibility-only alias:
  - `SPARSE_ND_SUPERNODAL_POSTORDER`
- internal/default-policy-only control:
  - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
- deferred debug/profile controls:
  - `SPARSE_ND_PROFILE`
  - `SPARSE_QG_PROFILE`
  - `SPARSE_HCC_DEBUG`
- deferred FM-family controls:
  - all `SPARSE_FM_*`

That gives future Epic 6 configuration work one maintained policy checkpoint
instead of scattering the residual queue across sprint artifacts only.

## Compatibility Read

After Day 12, the highest-value configuration surfaces tell one coherent story:

1. use `sparse_analysis_opts_t.reorder_opts` for the shipped advanced
   analysis/reorder controls
2. explicit typed values win
3. env vars are compatibility overrides only when the typed field is left
   unspecified
4. FM tuning and debug/profile controls remain outside the public Phase 1
   surface

## Validation

Because `include/sparse_analysis.h` changed, I ran:

- `make format`
- `make lint`
- `make test`

All passed.

Focused retained proof points:

- the Phase 1 precedence/default coverage in `test_reorder_nd` stayed clean
- the graph/reorder-sensitive normal test surface also stayed clean:
  - `test_graph`
  - `test_graph_fm_buckets`
  - `test_reorder_nd`
  - `test_reorder_amd_qg`

Day 12 note:

- I did not rerun `make quality-review-full` on this batch because it was a
  docs/header narrative follow-through with no implementation or behavior
  change; the required `*.h` gate still passed cleanly

## Close

Sprint 61 Day 12 completes the planned docs and maintainer-story update:

- caller-facing surfaces now steer users toward the shipped typed path
- the public header now states the real precedence rule directly
- the deferred env-var queue is explicit in the maintainer-policy home
- no caller-facing contradiction remains on the touched Phase 1
  configuration surfaces
