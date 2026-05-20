# Sprint 35 Day 7: README & Tutorial Rewrite Design

## Scope

Turn the Day 6 audit into a concrete public-doc rewrite plan before any broad
README/tutorial editing begins.

The Day 7 goal is not to start rewriting prose. It is to prevent Day 8 through
Day 11 from re-litigating ownership, wording, and file order while editing.

## Main Result

Sprint 35 now has a single public-doc architecture:

- `README.md` is the concise public entrypoint
- `docs/tutorial.md` is the fuller usage-teaching surface
- installed headers remain the authoritative API contract
- `INSTALL.md` and `examples/README.md` are support docs, not competing API
  teaching layers

That split is the key Day 7 decision. Without it, the Sprint 35 rewrite would
keep moving the same guidance between files instead of reducing drift.

## Ownership Decisions

### `README.md`

Keep it responsible for:

- short capability overview
- maintained quality/build command map
- compact, current public snippets
- first-stop orientation for new users

Do not let it become:

- a second long-form tutorial
- the full home for iterative/precondition explanation
- a duplicate of `INSTALL.md`

### `docs/tutorial.md`

Make it responsible for:

- step-by-step usage teaching
- multi-step solver workflows
- iterative and precondition explanations
- matrix-free and SVD walkthroughs

This is the file that should absorb the strongest public teaching burden once
the stale type names are removed.

### Installed headers

Keep them as the authoritative contract for:

- current public type names
- option fields and accepted values
- routine behavior
- per-routine preconditions

README and tutorial should follow the headers, not reinterpret them.

### `INSTALL.md` and `examples/README.md`

Keep them narrow:

- `INSTALL.md` = platform/build/install guidance
- `examples/README.md` = catalog of shipped examples

Neither should become a second canonical API tutorial.

## Canonical Public Wording

### Initialization examples

- show designated initializers for non-default usage
- use `NULL` only for the pure-default path
- avoid public zero-init sentinel style

### Iterative / ILU naming

- use `sparse_iter_opts_t` for CG
- use `sparse_gmres_opts_t` for GMRES
- use `sparse_ilut_opts_t` for ILUT
- describe ILU(0) as the default no-options incomplete factorization path

### Reorder wording

- `NONE`, `RCM`, `AMD`, and `ND` are the normal symmetric-analysis reorder set
- `COLAMD` is accepted in analysis, but not the normal symmetric-analysis
  recommendation
- QR-specific public examples may teach the column-oriented `COLAMD` path

### Quality/build wording

- README owns the concise reviewed-quality command map
- `INSTALL.md` may reference the build/test flows, but should not duplicate the
  full operator explanation
- tutorial should mention these commands only where they support user examples

## Precondition-Guidance Structure

Precondition language should live at one level per document type:

- headers = authoritative routine-specific requirements
- tutorial = user-facing operational explanation
- README = short signposts only
- support docs = no independent safety narrative unless directly required

This is important for Day 9 and Day 10: precondition auditing will go much
faster if the expected destination for each class of wording is already fixed.

## Implementation Order

### Day 8

Primary rewrite:

- `docs/tutorial.md`

Secondary reconciliation:

- `README.md`

Reason:

- tutorial contains the strongest remaining public falsehoods
- README is mostly current and should be normalized against the tutorial rewrite
  rather than edited first

### Day 9

Audit residual precondition-language debt across:

- rewritten `README.md`
- rewritten `docs/tutorial.md`
- installed headers
- any user-facing example comments touched by the rewrite

### Day 10

Implement the precondition wording fixes surfaced by Day 9.

### Day 11

Run the support-doc cleanup batch:

- `INSTALL.md`
- `examples/README.md`
- `benchmarks/README.md`

Only after the main public wording is stable.

### Day 12

Use example-build and snippet validation to catch any remaining doc/code drift.
Only touch shipped example source files if that validation exposes a real
mismatch.

## Bottom Line

Day 7 removes the remaining planning ambiguity from Sprint 35.

The rewrite no longer has to answer "which file should own this explanation?"
while editing. Day 8 can now apply one stable structure:

- tutorial first for truthfulness
- README second for consistency
- support docs later for cleanup
