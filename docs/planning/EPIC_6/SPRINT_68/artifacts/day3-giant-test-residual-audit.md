# Sprint 68 Day 3: Giant-Test Residual Audit

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Re-rank the remaining giant-test hotspots after Sprint 67 so Sprint 68 can
land bounded high-value test refactors and assurance work instead of broad
generic cleanup.

## Audit Inputs

- `docs/planning/EPIC_6/SPRINT_68/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_68/WORKING_NOTES.md`
- direct live-file measurements across:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_graph.c`
  - `tests/test_iterative.c`
  - `tests/test_ldlt.c`
  - `tests/test_svd.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_eigs.c`

## Measured Giant-Test Field

| File | Lines | `test_` count | `RUN_TEST` count | Sprint mentions | Day mentions |
|---|---:|---:|---:|---:|---:|
| `tests/test_chol_csc.c` | `4751` | `144` | `145` | `35` | `60` |
| `tests/test_ldlt_csc.c` | `3680` | `96` | `96` | `30` | `51` |
| `tests/test_qr.c` | `3197` | `72` | `72` | `10` | `26` |
| `tests/test_graph.c` | `2900` | `60` | `60` | `53` | `96` |
| `tests/test_iterative.c` | `2802` | `76` | `79` | `7` | `17` |
| `tests/test_ldlt.c` | `2798` | `84` | `84` | `5` | `34` |
| `tests/test_svd.c` | `2766` | `74` | `97` | `25` | `38` |
| `tests/test_integration.c` | `2371` | `43` | `47` | `6` | `4` |
| `tests/test_reorder_nd.c` | `2262` | `34` | `34` | `81` | `99` |
| `tests/test_eigs.c` | `1522` | `30` | `30` | `14` | `40` |

## Audit Conclusions

### 1. The remaining field is not one generic giant-test bucket

The current hotspots separate into three real classes:

- strongest first-lane refactor candidates:
  - `tests/test_chol_csc.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_ldlt_csc.c`
- strongest second-layer oracle/assurance candidates:
  - `tests/test_integration.c`
  - `tests/test_eigs.c`
  - `tests/test_iterative.c`
  - `tests/test_svd.c`
- large but more internally coherent or lower-priority follow-through surfaces:
  - `tests/test_qr.c`
  - `tests/test_graph.c`
  - `tests/test_ldlt.c`

So Sprint 68 should not chase every large test equally. The highest-value work
is concentrated in a smaller set of files where size, chronology, helper
sprawl, and mixed proof ownership still collide.

### 2. `tests/test_chol_csc.c` is the strongest first target

Why it ranks first:

- it is the largest remaining giant test by a clear margin
- it combines multiple distinct proof roles in one permanent file:
  - family-local CSC factorization behavior
  - dense primitive checks
  - supernodal extract/writeback plumbing
  - dispatch and backend-contract proof
  - large corpus and regression lanes
- it still carries meaningful sprint-history layering, but the larger problem
  is that multiple ownership layers coexist in one file

That makes it the strongest first Sprint 68 landing if the sprint wants one
bounded helper extraction or split with the best maintenance payoff.

### 3. `tests/test_reorder_nd.c` is the strongest second target for chronology and proof-layer reasons

Why it ranks second:

- it is smaller than `test_chol_csc.c`, but its chronology density is the
  highest in the field
- it still mixes:
  - public ND behavior
  - compatibility/env-policy proof
  - post-Sprint-27 follow-through contracts
  - enum-dispatch and supernodal-postorder validation

So its maintenance pressure is real, but it is more about chronology and proof
layering than about one obvious first helper split. That makes it slightly
worse than `test_chol_csc.c` as the first bounded landing, but still the
strongest competitor.

### 4. `tests/test_ldlt_csc.c` is real pressure, but cleaner than the first two

Why it ranks third:

- it is large and helper-heavy
- it carries substantial Sprint-history layering
- but it reads more consistently as a family-local owner than
  `test_chol_csc.c`
- and it carries less cross-family or compatibility layering than
  `test_reorder_nd.c`

So it remains a strong later giant direct-family target, not the best first
Sprint 68 landing.

### 5. The strongest assurance owner is not the strongest first refactor target

`tests/test_integration.c` remains the strongest shared oracle/parity owner:

- it is the natural home for cross-family public-path comparisons
- it has lower chronology density than the giant family-local files
- the highest-value next move there is stronger second-layer assurance, not
  first-wave file splitting

That distinction matters because Sprint 68 should not confuse oracle value with
refactor-first value.

## Day 3 Exit State

Sprint 68’s broad giant-test claim is now reduced to one ranked live seam map:

1. strongest first target:
   - `tests/test_chol_csc.c`
2. strongest second target:
   - `tests/test_reorder_nd.c`
3. strongest later giant direct-family target:
   - `tests/test_ldlt_csc.c`
4. strongest oracle/assurance owner:
   - `tests/test_integration.c`
5. strongest later assurance/follow-through owners:
   - `tests/test_eigs.c`
   - `tests/test_iterative.c`
   - `tests/test_svd.c`

Day 4 should now fix one explicit first-landing boundary instead of carrying a
generic hotspot shortlist.
