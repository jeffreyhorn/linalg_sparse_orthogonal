# Sprint 53 Day 10: Dispatch Reconciliation Batch

## Purpose

Day 10 lands the smallest high-signal public-story patch identified by the Day
9 audit. The goal is to tighten the top-level Cholesky / LDL^T CSC wording so
it matches the landed Sprint 53 code and benchmark proof surfaces without
reopening implementation work or broad documentation churn.

## Main Day 10 Result

Sprint 53 now has a materially cleaner top-level CSC dispatch story:

- the Cholesky CSC section now says the repeated-run story is intentionally
  simple
- the LDL^T CSC section no longer reads as if the analysis-aware indefinite
  follow-through is still future work
- the new indefinite factor-many benchmark proof is now visible at the README
  layer instead of only in benchmark-local docs and working notes

This stayed inside the Day 9 fence:

- only `README.md` changed
- `benchmarks/README.md` did not need more edits
- no header, test, or implementation changes were needed

## Touched Surface

### `README.md`

The Day 10 patch did two bounded things.

### 1. Tightened the Cholesky repeated-run / CSC dispatch summary

The README now says explicitly that Cholesky's CSC story is the simpler one:

- AUTO picks linked-list vs CSC by size
- forcing CSC means the CSC backend directly
- the strongest repeated-run proof surfaces are:
  - `bench_refactor`
  - default SPD mode in `bench_refactor_csc`

That keeps the Cholesky section compact while making its dispatch model
distinguishable from LDL^T's layered CSC story.

### 2. Replaced the stale LDL^T "future follow-up" wording with the current Sprint 53 contract

Before Day 10, the LDL^T CSC section still said the analysis-aware `_with_analysis`
follow-through was effectively a later-sprint idea. That was stale after the
Sprint 53 Days 4-8 work.

The README now states the current bounded contract:

- forcing CSC means the CSC pipeline, not a blanket promise that the batched
  path wins every indefinite input
- the scalar Bunch-Kaufman pre-pass remains the authoritative indefinite
  permutation-resolution step
- after CSC selection, completion may:
  - retain the batched path
  - or fall back to the resolved scalar-prepass factor when the batched path
    rejects the cached pivot pattern

It also now mentions the new bounded indefinite proof surface:

- `bench_refactor_csc --indefinite-kkt`
  - public repeated-run LDL^T path vs direct resolved-analysis CSC completion
    path
  - same-pattern KKT workload
  - round-off residuals on both sides after the Sprint 53 permutation fix

## Why This Was the Right Scope

Day 9 already showed that the remaining drift was not spread evenly across the
repo:

- `include/sparse_ldlt.h` was already the strongest family-local source of
  truth
- `benchmarks/README.md` was already aligned enough after Day 8
- CSC-specific tests already described the layered LDL^T pipeline accurately

So the real remaining problem was the top-level README compressing:

- Cholesky CSC dispatch
- LDL^T CSC dispatch

into a story that was too symmetric and too coarse.

Day 10 fixes that without duplicating the full local benchmark or header
contracts.

## Targeted Sanity Checks

Because this was docs-only, Day 10 used targeted wording checks instead of the
code-day validation gate:

- `rg -n "bench_refactor_csc|indefinite-kkt|Bunch-Kaufman|CSC pipeline|batched path|resolved scalar-prepass factor|Cholesky CSC dispatch|LDL\\^T CSC dispatch" README.md benchmarks/README.md include/sparse_ldlt.h include/sparse_cholesky.h`
- `sed -n '548,648p' README.md`
- `wc -l README.md benchmarks/README.md include/sparse_ldlt.h include/sparse_cholesky.h`

Results:

- the new README wording matches the benchmark-local README and LDL^T header
- no README-driven contradiction forced a header follow-on
- the patch stayed bounded to the primary Day 9 target

## Operational Result

Sprint 53 now has a cleaner late-sprint public story:

1. Cholesky CSC dispatch is described as the intentionally simpler family
2. LDL^T CSC dispatch is described as the intentionally layered family
3. the indefinite repeated-run proof is visible at the top-level README layer
4. the remaining Day 11 work can stay focused on proof gaps instead of more
   generic documentation reconciliation
