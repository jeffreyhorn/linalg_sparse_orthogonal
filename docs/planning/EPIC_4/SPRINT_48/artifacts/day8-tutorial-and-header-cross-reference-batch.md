# Sprint 48 Day 8: Tutorial and Header Cross-Reference Batch

## Objective

Land the bounded tutorial/header cross-reference cleanup identified on Day 7 so
the remaining lifecycle and original-matrix caveats stay locally useful while
repeating less policy text and acknowledging the new maintainer-guide
ownership boundary.

## Commands Run

1. Re-read the Sprint 48 Day 8 plan section:
   - `sed -n '285,360p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the Day 7 post-guide audit:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_48/artifacts/day7-post-guide-audit.md`
3. Re-read the exact tutorial and header target blocks:
   - `sed -n '138,170p' docs/tutorial.md`
   - `sed -n '216,270p' docs/tutorial.md`
   - `sed -n '120,170p' include/sparse_types.h`
   - `sed -n '57,115p' include/sparse_lu.h`
   - `sed -n '97,145p' include/sparse_cholesky.h`

## Changes

#### 1. Added the first tutorial-to-guide policy boundary

The QR tutorial section now keeps the user-facing original-matrix rule and adds
the first direct guide-aware reference for the broader documentation-ownership
and lifecycle-policy interpretation.

Interpretation:

- the tutorial still teaches the workflow locally
- it no longer acts as if it owns the full policy explanation

#### 2. Reduced repeated original-matrix wording in later tutorial sections

The ILU/ILUT/IC(0) and SVD sections now refer back to the QR section’s
original-matrix rule instead of re-explaining the same guidance from scratch.

Interpretation:

- user-facing workflow guidance stays intact
- repeated wording is lower without turning the tutorial into a reference maze

#### 3. Added the generic guide-aware boundary at the shared callback contract

`include/sparse_types.h` now states that the repository-wide documentation and
maintainer-policy interpretation of lifecycle/cancellation rules lives in
`docs/maintainer_guide.md`.

Interpretation:

- the generic callback contract still owns the shared behavior statement
- the broader policy ownership is now explicit

#### 4. Kept LU and Cholesky routine-specific details local while tightening the shared-policy boundary

`include/sparse_lu.h` and `include/sparse_cholesky.h` now:

- preserve their routine-specific pre-iteration mutation details
- point back to `sparse_progress_cb_t` for the generic callback contract
- point to `docs/maintainer_guide.md` for the broader maintainer-policy
  interpretation

Interpretation:

- local truth stayed local
- generic policy repetition is lower

## Bottom Line

Sprint 48 Day 8 completes the first cross-reference pass exactly where Day 7
said it should:

- touched:
  - `docs/tutorial.md`
  - `include/sparse_types.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
- preserved:
  - local workflow and routine-specific caveats
- reduced:
  - repeated policy wording
- added:
  - the first guide-aware boundary for tutorial and header surfaces

That is the right bounded cross-reference batch before the Day 9
quality-contract audit.
