# Sprint 217 Day 5: Fixture Closure And Scaled-Norm Design

**Scope:** [Day 5 plan](../PLAN.md) and the [Day 2 contract](day2-numerical-behavior-contract.md).
This is a pre-implementation review. Production arithmetic and solver code
remain unchanged; failing tests below are expected, not waived.

## Fixture Review

The Day 3-4 cases already cover ordinary finite vectors, `1e200` and
`1e-200` pairs, mixed scales, near-equal signed entries, infinities, NaN in
both orders relative to infinity, four tiny-RHS solver false successes, and
the observational `west0067` factor error. Day 5 adds three vector tests:
`DBL_MAX` singleton versus pair, smallest-subnormal singleton versus pair,
and positive-zero/empty controls for the infinity norm. It also adds a CG
nonfinite-initial-guess/zero-RHS precedence test and a block-CG invalid
second-column test. No existing red expectation was relaxed. The only
accuracy oracle newly added is `hypot` for a two-element subnormal vector;
the singleton and final-overflow controls use exact `DBL_MAX`/smallest-
subnormal values and IEEE classifications.

| Planned branch or behavior | Pre-change fixture | Baseline |
| --- | --- | --- |
| NULL and `n <= 0` -> `+0`, no read | Existing norm2 NULL/zero-length, Day 3 negative-length; existing norminf NULL, Day 5 negative-length | Pass; positive sign explicit for new and Day 3 cases |
| All `+0`/`-0` -> `+0` | Day 3 norm2 and Day 5 norminf | Pass |
| Finite ordinary and finite signed singleton | Existing 3-4-5 and -7; Day 3 -1e100 | Pass |
| Finite large representable norm, including maximal singleton | Day 3 `1e200`/`1e154`, Day 5 `DBL_MAX` singleton | Fail: intermediate squares overflow |
| Finite small normal and subnormal nonzero result | Day 3 `1e-200`, Day 5 smallest subnormal | Fail: intermediate squares underflow |
| Mixed finite magnitudes | Day 3 `[1e200,1e-200,-1e200,-1e-200]` | Fail: intermediate overflow |
| True final 2-norm overflow | Day 5 `[DBL_MAX,-DBL_MAX]` | Pass: `+inf` is legitimate |
| Infinity without NaN | Day 4 positive and negative infinity for both norms | Pass |
| NaN with finite or infinity, either order | Day 4 norm2 and norminf cases | Norm2 pass; norminf fail (NaN skipped) |
| Selected scalar solver tiny nonzero RHS | Day 4 CG, GMRES, MINRES, BiCGSTAB with independently measured true residual | Four false-success failures |
| Caller nonfinite RHS/initial guess | Day 4 GMRES/MINRES NaN RHS; Day 5 CG NaN guess with zero RHS; Day 5 block-CG second-column NaN | Fail; status or precedence wrong |
| Nonfinite computed residual | Day 4 CG/BiCGSTAB finite-input overflow | CG fail; BiCGSTAB pass |
| ILUT factor-system diagnostic | Day 4 `west0067` independent `L(Uz)-Pr` measurement | Pass as observation only; enforcement belongs to Days 8-10 |

This is representative branch coverage, not the full Cartesian product of
each scalar solver, matrix-free adapter, input class, and iteration state.
The matrix-free/handle variants share the selected scalar kernels and need
adapter-specific verification during Days 8-10. Block CG has its own norm
and max-column reporting loops, so it receives an explicit invalid-column
fixture now. Exact-zero residual, zero RHS, ordinary nonconvergence, and
breakdown have existing solver tests; Day 8-10 must inspect their status
diagnostics after hardening. A reliably constructed finite-input *ratio*
overflow or zero Krylov norm without earlier scalar arithmetic failure is
not yet available; these are implementation-branch obligations for Days 8-10,
not a reason to weaken the Day 2 contract. The ILUT negative controls and
enforced backward-error threshold are likewise intentionally deferred to
the ILUT implementation phase.

## Scaled 2-Norm Design

1. Return `+0.0` before reading memory when `v == NULL` or `n <= 0`.
2. Initialize `scale=0`, `ssq=1`, and a flag for whether any infinity was
   seen. For each component, take its magnitude. If it is NaN, return NaN;
   if it is infinity, set the flag but continue scanning so a later NaN wins.
   For finite nonzero `a`, if `a > scale`, update
   `ssq = 1 + ssq*(scale/a)*(scale/a)` then `scale=a`; otherwise update
   `ssq += (a/scale)*(a/scale)`. Zero components do nothing.
3. After the scan, return `+inf` if the infinity flag is set; otherwise
   return `+0` if `scale==0`, or `scale*sqrt(ssq)`. A genuinely
   unrepresentable final norm may be infinite. Preserve the public signature.

Each division ratio is in `[0,1]`; its square cannot overflow. In exact
arithmetic, `ssq` is the sum of squared components relative to the largest
magnitude so far and is in `[1,n]` after the first nonzero value. The finite
`idx_t` length therefore cannot drive `ssq` anywhere near `DBL_MAX` on a
feasible allocation. A tiny ratio may underflow to zero, but its contribution
is then below the resolution of the larger-scale result; a vector made only
of subnormal entries still has a positive `scale`, and at least one term is
one. The final product is the only place a large finite-input norm may
overflow. Day 6 should test `DBL_MAX` singleton and pair specifically so
the former stays finite while the latter legitimately overflows. The Day 2
normal/subnormal tolerance rules govern numerical acceptance, not an
unqualified promise of correct rounding for every length.

`vec_norminf` needs a separate simple loop: return `+0` for NULL/empty,
inspect all magnitudes, return NaN upon seeing one, otherwise retain the
maximum (possibly infinity). A shared internal helper would add indirection
without removing meaningful duplicate arithmetic; the two kernels share
only input guards and classification concepts. Keep both public APIs and
their source ownership in `src/sparse_vector.c` unchanged.

## Day 6 Handoff

Implement only the two norm kernels first, then rerun the focused vector
binary. Do not hide the solver failures with skipped tests: the scaled norm
fix alone may remove tiny-RHS false success but does not establish the
Day 2 caller-input/status contract. Days 8-10 own those status paths and
ILUT gate enforcement.
