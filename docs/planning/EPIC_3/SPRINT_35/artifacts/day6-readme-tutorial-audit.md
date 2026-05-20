# Sprint 35 Day 6: README & Tutorial Consistency Audit

## Scope

Audit the user-facing doc layer after the Day 5 header cleanup, identify where
public docs still drift from the current shipped API, and convert the residual
work into named rewrite batches before broad prose edits begin.

Day 6 is intentionally an audit day, not a rewrite day.

## Main Result

The remaining Sprint 35 public-doc debt is not evenly distributed.

The dominant truthfulness issue is `docs/tutorial.md`, which still teaches
stale iterative and ILU-related option types that are no longer part of the
installed public API. `README.md` is mostly current on behavior and workflow
names, but it still carries a smaller consistency/structure queue.

That means Day 8 should not start with a generic README refresh. It should
start with a tutorial-first rewrite, then reconcile the README around that
canonical public wording.

## Audit Findings

### 1. The tutorial is the strongest remaining API-truthfulness problem

`docs/tutorial.md` still contains stale public type names:

- `sparse_cg_opts_t`
- `sparse_ilu_opts_t`

The current shipped public surface instead uses:

- `sparse_iter_opts_t`
- `sparse_gmres_opts_t`
- `sparse_ilut_opts_t`

This is the highest-signal Sprint 35 drift because it can mis-teach real API
usage to downstream users even if the examples are otherwise readable.

### 2. The README is mostly current, but still needs a cleanup pass

`README.md` already names the maintained Sprint 34 quality flow correctly:

- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`
- `make deadcode-check`

Its SVD feature summary is also current about the `economy = 0` /
`compute_uv = 1` full-output path.

The README queue is therefore narrower:

- normalize snippet style to the Day 3 public example contract
- decide where operator guidance should be concise vs detailed
- reduce duplicated command explanations where the same workflow is described
  multiple times

### 3. Example-facing support docs are not the leading problem

`examples/README.md` and `INSTALL.md` are comparatively healthy:

- `examples/README.md` is short and current
- `INSTALL.md` still points to the maintained build/test flows and does not
  show the stale iterative/ILU type drift found in the tutorial

They remain valid follow-on cleanup surfaces, but they should not lead the
Sprint 35 rewrite sequence.

## Conflict Map

### Iterative solver guidance

Currently split across:

- `README.md`
- `docs/tutorial.md`
- `include/sparse_iterative.h`
- `include/sparse_ilu.h`
- shipped iterative examples

Risk:

- headers are current
- tutorial is stale
- README is mostly current

This makes iterative setup the most important cross-surface reconciliation
topic for Day 7 and Day 8.

### SVD usage guidance

Currently split across:

- `README.md`
- `docs/tutorial.md`
- `include/sparse_svd.h`

Risk:

- the installed header is now current after Day 4
- README behavior wording is current
- tutorial snippets still need style alignment with the Day 3 public example
  rule

### Quality/build workflow guidance

Currently split across:

- `README.md`
- `INSTALL.md`

Risk:

- the commands themselves are current
- the remaining issue is responsibility split, not falsehood

Day 7 should decide whether README stays the operator command map while
`INSTALL.md` remains installation/platform guidance only.

## Named Rewrite Queue

### Primary rewrite batch

- `docs/tutorial.md`
- update stale iterative/ILUT type names
- apply the designated-initializer vs `NULL` teaching split consistently
- keep SVD usage aligned with the Day 4 `include/sparse_svd.h` contract

### Secondary rewrite batch

- `README.md`
- normalize public snippet presentation
- trim or consolidate duplicated workflow explanation
- keep it as the concise entrypoint instead of letting it absorb tutorial-level
  detail

### Support-doc follow-on batch

- `examples/README.md`
- `INSTALL.md`

Only touch these if the Day 8 rewrite changes the canonical wording enough to
require follow-on cleanup.

## Bottom Line

Sprint 35's residual public-doc queue is now clearly prioritized:

1. `docs/tutorial.md` for API truthfulness
2. `README.md` for consistency and ownership
3. `examples/README.md` / `INSTALL.md` only as follow-on cleanup

That gives Day 7 a clean job: decide the ownership split once, so Day 8 can
rewrite the public docs without re-litigating where each explanation belongs.
