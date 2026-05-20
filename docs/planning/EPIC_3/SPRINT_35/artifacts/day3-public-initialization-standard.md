# Sprint 35 Day 3: Public Initialization Standard

## Scope

Define the maintainer-facing rule for how public options-struct usage should be shown across:

- installed headers in `include/`
- `README.md`
- `docs/tutorial.md`
- example-facing public snippets
- explanatory tests when they are used as source material for public examples

This note does **not** change the API. It defines the documentation and example contract that later Sprint 35 edits should follow.

## Main Result

Sprint 35 should use one explicit public-example rule:

- **Use designated initializers whenever the example teaches non-default options.**
- **Use `NULL` when the point of the example is to teach the pure-default path.**

That is the stable contract already implied by the current codebase and by prior cleanup sprints. Sprint 35 should make it explicit and apply it consistently.

## Why This Rule Fits the Current Repo

### 1. It matches the current implementation and prior cleanup work

Recent production-facing surfaces already lean this way:

- installed headers such as `include/sparse_lu.h`, `include/sparse_iterative.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_analysis.h`, and `include/sparse_svd.h`
- `README.md`
- `docs/algorithm.md`
- shipped examples using current public option types

Sprint 31 and Sprint 34 also already established designated-initializer practice across reviewed benchmark/test/example code. Sprint 35 should keep public docs aligned with that reviewed implementation pattern instead of drifting back to positional or stale examples.

### 2. It is more truthful than forcing every example through a local struct declaration

Many public APIs already document:

- `opts == NULL` means “use defaults”

If a snippet is only trying to show the default path, an explicit local options struct is extra noise and can easily drift from real defaults later. Passing `NULL` is shorter and more accurate for that teaching goal.

### 3. It minimizes future breakage

Public structs already carry several trailing-field compatibility notes. Designated initializers make that back-compat story readable:

- examples mention only the fields they are intentionally overriding
- added trailing fields do not force public snippet rewrites
- readers do not infer that every omitted field needs to be set manually

## Cross-Surface Contract

### Rule 1: Non-default examples use designated initializers

When a snippet is teaching solver/factorization tuning, it should use designated initializers and name only the relevant fields.

Preferred shape:

```c
sparse_gmres_opts_t opts = {
    .max_iter = 500,
    .restart = 30,
    .tol = 1e-10,
};
```

Not preferred:

- positional struct literals
- “set every field” boilerplate when most fields are defaults
- stale or hypothetical option types

### Rule 2: Pure-default examples use `NULL`

When a snippet is only showing the basic call path and does not need to teach tuning, pass `NULL` for the options parameter.

Preferred shape:

```c
sparse_err_t err = sparse_solve_gmres(A, b, x, NULL, NULL, NULL, &result);
```

This is especially appropriate in:

- parameter docs
- minimal quick-start examples
- places where the surrounding prose already explains that defaults are acceptable

### Rule 3: Public snippets use current shipped names only

Public docs must name:

- the current installed option types
- the current installed fields
- the current supported behavior

This means Sprint 35 should eliminate stale names like:

- `sparse_cg_opts_t`
- `sparse_ilu_opts_t`

when the current public surface instead uses other types or call patterns.

### Rule 4: Snippets stay minimal

Public docs should not restate defaults unless the default itself is the point. Show only the fields needed to communicate the example’s intent.

Good examples:

- `.reorder = SPARSE_REORDER_AMD`
- `.compute_uv = 1, .economy = 1`
- `.max_iter = 1000, .tol = 1e-10`

Avoid:

- repeating obvious defaults “for completeness”
- full-struct pseudo-templates that look like required boilerplate

### Rule 5: Public wording must match the actual API behavior

Syntax cleanup is not enough. Public examples and nearby prose must describe what the API actually does today.

That matters immediately for:

- `include/sparse_svd.h`, where the docs still contradict current full-SVD and partial-SVD behavior
- `docs/tutorial.md`, where example type names no longer match the installed API

## Exception Policy

Acceptable public-facing exceptions are intentionally narrow.

### Allowed exceptions

- **Pure-default paths:** use `NULL` instead of an options struct.
- **Compact one-line snippets:** designated initializers may stay on one line when readability is still good.
- **Implementation-dense tests:** tests may remain denser than README/header prose, but if a test snippet is promoted into public docs it should be rewritten to follow the public rule.

### Not acceptable as exceptions

- positional options-struct literals in public docs
- zero-init sentinels presented as recommended public style
- stale historical type names
- examples that imply unsupported reorder or precondition behavior

## Where The Rule Lives During Sprint 35

For this sprint, the authoritative maintainer rule lives in:

- this Day 3 artifact
- `docs/planning/EPIC_3/SPRINT_35/WORKING_NOTES.md`

Later Sprint 35 edits should reference and apply this rule implicitly by making the touched public surfaces converge on it. If the rule still feels necessary after implementation, Day 12 or later documentation cleanup can decide whether a shorter maintainer-facing note belongs in `README.md` or another stable contributor-facing location.

## Immediate Consequences For Day 4+

### Day 4 / Day 5

Targeted header cleanup should be:

- behavior-truthfulness first
- syntax cleanup only where a touched public snippet still violates the rule

Highest-priority header file:

- `include/sparse_svd.h`

### Day 6 / Day 7 / Day 8

README/tutorial cleanup should:

- replace stale type names with current public ones
- apply the designated-init vs `NULL` split consistently
- keep user-facing snippets short and current

## Bottom Line

Sprint 35 does not need a broad “convert every public snippet” campaign. It needs one stable public-example contract:

- designated initializers for non-default behavior
- `NULL` for pure-default behavior
- current type names and current behavior wording only

That rule is concrete enough to drive the remaining header, README, tutorial, and example-facing cleanup without ad hoc decisions per file.
