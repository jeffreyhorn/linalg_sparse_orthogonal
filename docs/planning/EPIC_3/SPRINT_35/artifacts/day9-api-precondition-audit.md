# Sprint 35 Day 9: API Precondition Language Audit

## Scope

Audit the rewritten public docs against the installed headers and isolate the
remaining user-facing usage assumptions that are still implicit or
underspecified.

Day 9 is an audit day. The point is to narrow Day 10 to a small wording pass,
not to reopen the full README/tutorial rewrite.

## Main Result

The remaining Sprint 35 debt is no longer stale API naming. It is a smaller
"headers precise, prose implicit" problem.

The installed headers already describe several routine-level preconditions
accurately, but the tutorial and README still leave some of those assumptions
unstated in the places where users are most likely to make the wrong call.

## Findings

### 1. Matrix-state assumptions are the strongest remaining gap

Several public APIs require an original or identity-permutation matrix view:

- ILU(0) / ILUT
- IC(0)
- QR factorization
- LDL^T
- analyze-once symbolic analysis
- SVD-family routines

The headers say this clearly. The user-facing docs do not yet say it clearly
enough.

Why it matters:

- a reader may reasonably try to reuse a matrix after factorization or
  reordering without realizing that some APIs require the original physical
  layout instead
- the tutorial's current ILUT example already uses `sparse_copy(A)`, but the
  reason is not spelled out

This is the highest-value Day 10 cleanup item.

### 2. Matrix-class guidance is still somewhat implicit

The public docs already say:

- CG is for SPD systems
- Cholesky is for SPD systems
- GMRES is for general systems

The remaining gap is preconditioner selection language:

- IC(0) belongs naturally with SPD workflows
- ILU(0) / ILUT belong naturally with general or indefinite workflows
- preconditioned CG still assumes an SPD operator path

That guidance belongs mainly in `docs/tutorial.md`, with only short signposts
in `README.md`.

### 3. QR routine selection needs one more user-facing clarification

The header contract is explicit:

- `sparse_qr_solve()` is the least-squares path
- for underdetermined systems it gives a basic solution, not the minimum-norm
  one
- `sparse_qr_solve_minnorm()` is the minimum-norm path

The tutorial and README still leave this distinction too implicit for a
first-pass reader.

This is a real usage-selection issue, not just wording polish.

### 4. SVD and quality-command wording are no longer the leading risk

After Day 8:

- SVD examples are aligned with the installed header contract
- reviewed-quality command names in README are current

These areas may still get small Day 11 cleanup if needed, but they are not the
main Day 10 target.

## Day 10 Cleanup Queue

### Primary tutorial pass

- explain fresh/original matrix expectations where the examples currently leave
  them implicit
- make solver/precondition matrix-class guidance more direct
- clarify the QR least-squares vs minimum-norm split

### Secondary README pass

- add only short signposts where a meaningful usage assumption is easy to miss
- do not duplicate the full header or tutorial safety narrative

### Conditional header pass

- only if Day 10 exposes a genuine header ambiguity rather than a prose-only
  omission

## Surface Mapping

- `docs/tutorial.md` = primary fix surface
- `README.md` = brief signposts only
- installed headers = keep as the precise contract surface
- support docs = follow later only if the Day 10 wording baseline changes

## Bottom Line

Sprint 35's remaining public-doc queue is now small and concrete:

1. matrix-state assumptions
2. matrix-class / preconditioner guidance
3. QR least-squares vs minimum-norm selection

That is the right shape for Day 10: a focused wording-tightening pass rather
than another open-ended rewrite.
