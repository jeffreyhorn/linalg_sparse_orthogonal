# Sprint 35 Day 8: README & Tutorial Implementation

## Scope

Apply the Day 7 rewrite plan to the two primary public-doc surfaces:

- `docs/tutorial.md`
- `README.md`

The goal is to close the stale API-usage drift first, then leave only the
narrower precondition-language cleanup for Days 9 and 10.

## Main Result

Day 8 closes the strongest remaining public truthfulness issues from the Day 6
audit.

The tutorial now uses the current public iterative and ILUT types, and the SVD
snippets now match the installed-header contract. The README needed only a
smaller consistency pass, not a second large tutorial rewrite.

## What Changed

### 1. `docs/tutorial.md` now teaches the current iterative public surface

The iterative and matrix-free sections now use:

- `sparse_iter_opts_t` for CG
- `sparse_gmres_opts_t` for GMRES
- `sparse_ilut_opts_t` for ILUT

The old stale names:

- `sparse_cg_opts_t`
- `sparse_ilu_opts_t`

are no longer taught in the tutorial.

This is the most important Day 8 fix because it closes the highest-signal
public API falsehood identified on Day 6.

### 2. Tutorial snippets now follow the Sprint 35 public example style

The touched tutorial examples now use:

- multi-line designated initializers for non-default options
- `NULL` only for the pure-default singular-values-only SVD path
- self-contained snippets instead of relying on implied surrounding state

This aligns the tutorial with the installed headers and avoids teaching a
different public style from the rest of the repo.

### 3. The tutorial's SVD section now matches the installed-header contract

The rewritten SVD examples now state the current behavior directly:

- `sparse_svd_compute(A, NULL, &svd)` for singular values only
- `compute_uv = 1, economy = 1` for thin/economy singular vectors
- `economy = 0` for full `U` and `V^T`
- partial SVD singular vectors only in the economy/thin path

That removes the risk that the tutorial might lag behind the Day 4
`include/sparse_svd.h` fix.

### 4. `README.md` only needed a small reconciliation pass

The README's command names and SVD feature wording were already current. Its
main residual inconsistency was snippet presentation.

Day 8 therefore kept the README edit narrow:

- the public GMRES example now uses the same multi-line designated-initializer
  style as the tutorial and headers

This preserves the Day 7 ownership split:

- README = concise entrypoint
- tutorial = fuller teaching surface

## Residual Queue for Day 9

The remaining Sprint 35 public-doc debt is now mostly precondition language:

- which examples should say "SPD only" more explicitly
- where fresh-copy / identity-permutation assumptions should be stated in user
  prose versus left to headers
- how much ILU / ILUT safety guidance belongs in tutorial text versus API
  contract comments

That is a much narrower queue than the Day 6 starting point, which still
included stale type names and stale option examples.

## Bottom Line

Day 8 completes the main README/tutorial rewrite successfully:

- tutorial truthfulness drift is closed
- README snippet style is reconciled
- the remaining work is now the intended Day 9 / Day 10 precondition pass,
  not unfinished API-name cleanup
