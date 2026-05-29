# Sprint 48 Day 7: Post-Guide Audit

## Objective

Audit the post-Day-6 documentation state so Sprint 48 can separate the
remaining tutorial/header cross-reference cleanup from the later
quality-contract ownership batch instead of treating both as one generic
follow-on queue.

## Commands Run

1. Re-read the Sprint 48 Day 7 plan section:
   - `sed -n '245,330p' docs/planning/EPIC_4/SPRINT_48/PLAN.md`
2. Re-read the new maintainer guide:
   - `sed -n '1,260p' docs/maintainer_guide.md`
3. Re-read the touched README maintainer-facing tail:
   - `sed -n '700,790p' README.md`
4. Refresh duplication markers across tutorial, headers, README, and guide:
   - `rg -n "factored|factor|cancel|cancellation|original/unfactored|original matrix|maintainer guide|Maintainer Guide|quality-review-full|deadcode-check" docs/tutorial.md include README.md docs/maintainer_guide.md`
5. Read the strongest remaining tutorial and header caveat blocks:
   - `sed -n '138,170p' docs/tutorial.md`
   - `sed -n '216,270p' docs/tutorial.md`
   - `sed -n '57,115p' include/sparse_lu.h`
   - `sed -n '97,145p' include/sparse_cholesky.h`
   - `sed -n '120,170p' include/sparse_types.h`
6. Confirm where the new guide is already referenced:
   - `rg -n "maintainer_guide|Maintainer Guide" docs/tutorial.md include README.md`

## Findings

#### 1. The remaining queue is now mostly local-behavior duplication, not maintainer-policy homelessness

Day 6 solved the main policy-home problem:

- `docs/maintainer_guide.md` now exists
- `README.md` now points to it

What remains is narrower:

- repeated tutorial wording about original/unfactored matrix expectations
- repeated cancellation/lifecycle wording across callback docs and routine
  option comments
- a small number of local README behavioral reminders that should stay local

Interpretation:

- Sprint 48 is no longer choosing where policy belongs
- it is now deciding where concise local caveats should stay and where a
  reference is enough

#### 2. The strongest Day 8 tutorial target is the repeated “original matrix view” guidance

The live tutorial still repeats the same original/unfactored-state guidance in
multiple sections:

- QR factorization block
- ILU(0) / ILUT / IC(0) preconditioner block
- SVD block

Those passages are still useful because they are user-facing workflow guidance,
but they now repeat the same state rule more than necessary.

Interpretation:

- Day 8 should target these tutorial passages first
- the goal should be consistent phrasing and lighter repetition, not removing
  the guidance

#### 3. The strongest Day 8 header target is the cancellation/lifecycle seam across `sparse_types.h`, LU, and Cholesky

The current cancellation contract still appears at three levels:

- generic callback contract in `include/sparse_types.h`
- LU-specific pre-iteration mutation details in `include/sparse_lu.h`
- Cholesky-specific pre-iteration mutation details in
  `include/sparse_cholesky.h`

This is mostly the right ownership shape already:

- generic shared callback semantics belong in `sparse_types.h`
- routine-specific mutation details belong in the routine headers

What is still missing is the guide-aware reference boundary.

Interpretation:

- Day 8 should not delete these local caveats
- it should tighten them so the generic policy explanation lives in the guide
  and the headers focus on the local behavioral truth

#### 4. README no longer looks like the main Day 8 target

The post-Day-6 README still has local behavioral notes around:

- original matrix copies
- in-place factorization
- factored-state validation
- thread-safety constraints

Those are user/operator-facing and still belong close to the main API and
limitations overview.

Interpretation:

- README should not be the main Day 8 landing zone
- further README quality-contract simplification belongs to Day 9/10 instead

#### 5. There are not yet any tutorial or header references to the new guide

Current guide references appear in README only.

Interpretation:

- Day 8 is a first cross-reference pass for tutorial/header surfaces, not a
  refinement pass
- the batch should stay small and deliberate

#### 6. The bounded Day 8 target set is now explicit

The highest-value Day 8 targets are:

- `docs/tutorial.md`
  - unify the repeated “original matrix view” guidance across QR,
    preconditioners, and SVD
- `include/sparse_types.h`
  - keep the generic callback contract, but add a stable guide-aware reference
    boundary for the broader maintainer-policy interpretation
- `include/sparse_lu.h`
  - preserve LU-specific cancellation details while avoiding generic policy
    repetition
- `include/sparse_cholesky.h`
  - preserve Cholesky-specific cancellation details while avoiding generic
    policy repetition

Interpretation:

- this is a bounded cross-reference batch
- it is not a broad tutorial rewrite or a header comment overhaul

#### 7. Content that should stay local is also explicit now

The following should stay local and not move again in Sprint 48:

- tutorial user-facing workflow guidance about when to use `sparse_copy()`
- routine-specific LU/Cholesky cancellation details
- README limitations and thread-safety notes
- API-local caveats in QR / ILU / IC / SVD headers

Interpretation:

- Day 8 should reduce duplication by tightening wording and adding references,
  not by stripping away local truth

#### 8. No maintainer-guide scope expansion is needed before the quality-contract batch

The new guide already covers the right policy classes for Sprint 48:

- reviewed baseline interpretation
- warning authority
- dead-code meaning
- documentation ownership
- lifecycle/cancellation maintainer expectations
- stable repo norms

Interpretation:

- Day 9 should focus on quality-contract ownership simplification, not on
  growing the guide into a broader handbook first

## Bottom Line

Sprint 48 Day 7 makes the remaining documentation queue concrete:

- Day 8 direct targets:
  - `docs/tutorial.md`
  - `include/sparse_types.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
- Day 8 goal:
  - keep local behavioral truth
  - reduce repetition
  - add the first guide-aware cross-reference boundary
- content that should stay local:
  - tutorial workflow guidance
  - routine-specific cancellation details
  - README limitations and thread-safety notes
- Day 9/10 remains the right place for:
  - README/guide/Makefile/script quality-contract ownership cleanup

That is the right post-guide state before the Day 8 cross-reference batch.
