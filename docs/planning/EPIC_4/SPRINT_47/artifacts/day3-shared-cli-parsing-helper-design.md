# Sprint 47 Day 3 Artifact: Shared CLI Parsing Helper Design

## Purpose

Define a bounded internal parsing-helper seam for benchmark and example CLIs so
Sprint 47 can modernize `bench_main` and smaller peer auxiliary surfaces
without creating a broad public-facing CLI framework.

## Design Summary

Sprint 47 should add one small internal helper layer for benchmark/example CLI
parsing with four responsibilities only:

1. parse positive integers
2. parse bounded integers
3. parse finite doubles
4. parse enum-like string modes

The helper seam should be:

- internal-only
- reusable across benchmark and example binaries
- narrow enough to preserve file-local usage/help text and benchmark-specific
  semantics

This is intentionally smaller than a full "CLI library". The goal is to remove
the current ad hoc numeric and mode parsing drift, especially in
`bench_main.c`, while keeping command-specific behavior local.

## Main Drivers From the Live Code

### `bench_main.c` current problems

The live `bench_main.c` parser still owns:

- `atoi(...)` for:
  - `--spmv-iters`
  - `--size`
  - `--repeat`
- inline string matching for:
  - pivot mode
  - reorder mode
- hand-written malformed-input handling dispersed inside the main argument loop

This creates three specific Day 3 design requirements:

1. integer parsing must reject trailing junk and overflow
2. range checks must be built into the helper call contract, not left to every
   caller
3. enum-like parsing should separate shared string-to-token recognition from
   benchmark-specific semantic policy

### `bench_eigs.c` current strengths

The live `bench_eigs.c` parser already demonstrates the stronger intended
shape:

- checked `strtol(...)`
- checked `strtod(...)`
- helper-based diagnostics
- explicit mode parsing functions

That means Sprint 47 should not design a new system from scratch. It should
generalize only the small reusable core that `bench_main` lacks, while keeping
backend- or eigensolver-specific semantics local to `bench_eigs.c`.

## Proposed Helper Scope

## 1. Positive integer parsing helper

Purpose:

- parse a CLI value as an integer
- reject empty, non-numeric, overflowed, and trailing-junk inputs
- enforce `value >= min`

Primary Day 5 consumers:

- `--spmv-iters`
- `--size`
- `--repeat`

Proposed contract:

- input:
  - flag name
  - raw string
  - minimum value
  - typed output pointer
- behavior:
  - parse plus range-check in one step
  - print a caller-facing diagnostic on failure
  - return success/failure as a simple boolean/int result

Rationale:

- `bench_main.c` should not have to separately call parse, then remember to
  enforce `> 0`
- Sprint 47's main improvement is consistency, so the helper should own both
  syntactic validity and the common lower-bound check

## 2. Bounded integer parsing helper

Purpose:

- parse integer CLI values that allow zero or a bounded minimum other than one
- keep the same checked parse behavior as the positive-integer helper

Primary expected consumers:

- fields like block sizes or optional numeric knobs on peer surfaces
- possible later example/helper adoption

Proposed contract:

- same shape as positive-integer parsing
- explicit minimum argument defines whether zero is legal

Rationale:

- Day 3 should avoid two separate parsing styles for "positive int" and "other
  checked int" if the underlying semantics are the same
- the actual distinction is not parse mechanism; it is allowed minimum value

## 3. Finite double parsing helper

Purpose:

- parse floating-point CLI values
- reject empty, non-numeric, trailing-junk, overflow/underflow, NaN, and Inf
- optionally enforce a lower bound

Likely peer consumers:

- tolerance-like options
- shift/sigma-like options
- later example or benchmark configuration surfaces

Proposed contract:

- input:
  - flag name
  - raw string
  - minimum allowed value
  - output pointer
- behavior:
  - parse and range-check in one step
  - require finite result
  - print a clear diagnostic on failure

Rationale:

- `bench_eigs.c` already proves this is a useful shared pattern
- keeping "finite" inside the helper avoids repeated caller mistakes

## 4. Enum-like mode parsing helper

Purpose:

- parse small mode strings consistently
- support explicit accepted-string sets
- separate shared matching logic from command-specific semantic handling

Primary early consumers:

- reorder mode parsing in `bench_main.c`
- possibly pivot mode parsing

Proposed contract:

- shared layer should provide one bounded string-matching helper shape rather
  than centralizing every benchmark's mode semantics
- callers should still own:
  - the actual enum type they map to
  - help/usage wording
  - any benchmark-specific exclusions or aliases

Rationale:

- `bench_eigs.c` already has several local `parse_*` mode functions
- Day 3 should not over-generalize those into a fragile generic "any enum" API
- the right shared boundary is stable string-matching and diagnostic style, not
  all command-specific meaning

## Ownership and Usage Rules

### Internal-only placement

The helper layer should live in a benchmark/example auxiliary internal seam,
not in `include/` and not as part of the core library API.

Good properties of this placement:

- reusable by benchmark/example binaries
- does not become a supported public API
- can evolve during Sprint 47 without lifecycle guarantees beyond internal use

### Return contract

The helper layer should use a simple caller contract:

- return success/failure as `int` or equivalent small status
- print the user-facing diagnostic itself on failure
- leave the caller responsible only for:
  - exiting with the right command status
  - printing usage/help when appropriate

This matches the existing `bench_eigs.c` pattern closely enough to keep Day 5
small and avoids forcing every caller to duplicate error strings.

### Parse-plus-range-check rule

The helpers should do parse plus basic range validation in one call.

They should not be split into:

- "string to numeric"
- then separate caller-side "is this allowed?"

because that would recreate the current inconsistency drift in a new place.

### Diagnostics rule

The helper-generated diagnostics should be:

- flag-specific
- concrete about the failure class
- short enough to remain CLI-friendly

Examples of intended classes:

- missing value
- not a valid integer
- not a valid finite number
- out of range
- unknown mode

### Shared vs local boundary

Shared helper responsibilities:

- checked parse mechanics
- lower-bound / finite checks
- consistent diagnostic style
- stable string matching for small mode sets

Keep local to each binary:

- help/usage text
- command-specific defaults
- solver- or benchmark-specific semantics
- mode aliases that are unique to a specific tool
- reporting / output formatting

This keeps Sprint 47 from accidentally creating a general-purpose CLI layer.

## Reorder-Mode Parsing Design Note

The reorder-mode seam needs a small explicit rule because Day 2 showed that
this is not only a parser cleanup issue.

The Day 3 design rule should be:

- the parser helper should recognize a bounded set of candidate strings
- the command-level caller should own the final allowed-mode policy for that
  binary

For `bench_main`, this is important because Sprint 47 still needs a Day 6
parity decision on:

- which reorder modes are truly intended to be exposed there
- how emitted labels should align with the broader library surface

So the Day 3 design should not bake today's `bench_main` mode list into the
shared helper API. It should leave the mode set caller-supplied.

## File/Layer Shape

Sprint 47 should prefer one small internal helper seam rather than many
ad hoc local copies.

Recommended shape:

- one small internal helper header for benchmark/example CLI parsing
- optional paired implementation file if the helper bodies are large enough to
  justify it
- keep benchmark-specific wrappers local when they are just thin enum mappers

This keeps:

- shared numeric parsing centralized
- benchmark-specific mode policy local

## Non-Goals

Sprint 47 Day 3 should explicitly avoid designing:

- a public CLI parsing API
- a generalized shell/workflow parsing system
- a benchmark framework abstraction layer
- a broad replacement for `argparse`-style script interfaces
- a shared help/usage text renderer

Those would expand the sprint well beyond the bounded auxiliary modernization
goal.

## Main Day 3 Conclusions

### 1. The right shared seam is small and mechanical

Sprint 47 should share checked parse mechanics and diagnostics, not full
command semantics.

### 2. `bench_main` should be the first consumer because it has the clearest drift

The helper design is justified directly by the live `atoi(...)` and inline mode
parsing remaining there.

### 3. `bench_eigs.c` should remain partly local even after Sprint 47 helper adoption

Its backend/preconditioner/mode semantics are too command-specific to push down
into a generic shared parser layer.

### 4. Reorder-mode parsing should stay caller-configured

That preserves space for the Day 6 parity cleanup without hard-coding
today's potentially incomplete `bench_main` mode surface into the helper API.
