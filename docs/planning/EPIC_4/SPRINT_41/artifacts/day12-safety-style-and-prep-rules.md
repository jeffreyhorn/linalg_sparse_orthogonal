# Sprint 41 Day 12 Artifact: Safety-Style Documentation & Prep Rules

## Purpose

Capture the stable implementation rules Sprint 40 handed forward and Sprint 41
exercised in live code so later Epic 4 work has:

- a concrete shared-helper usage rule
- an explicit internal-first execution rule
- a clear exception boundary for local specialization
- a validation rule that matches the Sprint 40 anchor

This note is not a replacement for the Sprint 40 architecture contract. It is
the compact execution guide derived from that contract and from Sprint 41's
landed batches.

## 1. Shared-Helper Usage Rule

### Default rule

When new internal code needs generic array-size arithmetic or array allocation
based on `idx_t`/`size_t` counts, it should use the Sprint 41 shared helper
layer instead of creating new file-local arithmetic helpers.

Primary helper surface:

- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

Primary helper vocabulary:

- `sparse_size_mul_overflow(...)`
- `sparse_size_add_overflow(...)`
- `sparse_count_bytes_overflow(...)`
- `sparse_idx_count_bytes_overflow(...)`
- `sparse_size_to_idx_checked(...)`
- `sparse_malloc_array(...)`
- `sparse_calloc_array(...)`

### Use the shared helper layer when:

1. the code is internal `src/` implementation code
2. the logic is generic count/bytes arithmetic rather than algorithm-specific
   symbolic reasoning
3. the allocation pattern is a normal array allocation
4. the intended failure contract is the existing:
   - overflow -> `SPARSE_ERR_ALLOC`
   - allocation failure -> `SPARSE_ERR_ALLOC`

### Do not create new local clones of:

- `size_mul_overflow(...)`
- `count > SIZE_MAX / sizeof(T)` guards
- repeated `malloc((size_t)n * sizeof(T))` / `calloc((size_t)n, sizeof(T))`
  wrappers that only restate the shared helper semantics

Interpretation:

- Sprint 41 turned these into common internal infrastructure
- later Epic 4 work should reuse that infrastructure instead of reintroducing
  ad hoc local arithmetic vocabulary

## 2. Internal-First Prep Rule

### Default posture

Early Epic 4 structural work should stay internal-first whenever the same
maintainability/safety gain can be landed without changing public semantics.

That means:

- prefer internal payload/helper/workspace cleanup before public API reshaping
- prefer wrapper-preserving refactors before compatibility-narrowing work
- prefer internal ownership normalization before public documentation churn

### In practice, Sprint 41 exercised this as:

- internal helper layer in `src/`
- behavior-preserving source migration in hotspot modules
- public examples aligned only through an example-local helper seam
- no private `src/` headers leaked into public examples

Interpretation:

- "internal-first" is not just an architectural preference
- it is the default execution rule unless a task is explicitly public-surface
  work

## 3. Public-Semantics Preservation Rule

Sprint 41 reaffirmed that helper/safety cleanup is not permission to rewrite
public meaning.

Preserve exactly unless the task explicitly says otherwise:

- function semantics
- lifecycle meaning
- example teaching flow
- command/CI contract wording
- error-class behavior
- printed operator/example output structure when it is part of the example's
  teaching value

Examples from Sprint 41:

- iterative, matrix-free, and COLAMD examples changed allocation style only
- no solver-flow or teaching-semantics rewrite was bundled into the helper work

Interpretation:

- helper consolidation should be mechanically narrow unless a later sprint
  explicitly owns a public-facing rewrite

## 4. Validation-Anchor Rule

Sprint 40's validation anchor remains the default proof model for later Epic 4
implementation work.

### Minimum gate for `*.c` / `*.h` changes

```bash
make format
make lint
make test
```

### Stronger default for substantial refactors

```bash
make quality-review-full
```

### Targeted follow-on validation when justified

Use targeted checks when the touched surface warrants them:

- examples:
  - run the touched example binaries
- benchmarks/tooling:
  - run the touched benchmark/tooling path if the sprint changes them
- support/dead-code surfaces:
  - use the authoritative serial dead-code path
- wider refactors:
  - add the stronger reviewed baseline and any relevant follow-on checks

### Preserve these truthfulness rules

- reviewed CMake count truth remains `53`
- Makefile/CMake parity remains explicit
- dead-code execution remains serialized
- `deadcode-check` remains a completeness gate, not a zero-findings claim
- cross-platform enforced/staged/excluded wording remains honest

Interpretation:

- later Epic 4 work should start from the validation anchor rather than
  improvising per-sprint proof rules

## 5. Local-Specialization Exception Rule

The shared helper layer is **not** supposed to erase legitimate local
specialization.

Local helpers / local logic may remain acceptable when the code is genuinely
specialized rather than a clone of the generic arithmetic seam.

### Main accepted exception classes

#### A. Symbolic accumulation and representability choreography

Keep local when the logic is bound to symbolic structure-building semantics,
for example:

- prefix-sum / cumulative structure counts
- algorithm-specific representability checks
- structure-dependent sentinel or empty-shape handling

Sprint 41 example:

- `src/sparse_etree.c`

#### B. File-specific cleanup choreography

Keep local when the hard part is:

- ownership choreography
- sibling-buffer cleanup ordering
- algorithm/result-state interaction

not generic count arithmetic itself.

#### C. Specialized algorithm/harness surfaces

Defer or keep local when the maintainability problem is larger than helper
substitution alone, for example:

- `src/sparse_graph.c`
- large benchmark harnesses like `bench_main.c` / `bench_eigs.c`

#### D. Public example boundary

Do not pull private `src/` helper headers into public examples.

If public examples need bounded alignment:

- use a public-safe example-local helper seam
- or leave the example unchanged until a later public-facing pass owns it

Sprint 41 example:

- `examples/example_alloc_helpers.h`

Interpretation:

- "use the shared helper layer" is the default rule
- these exceptions make the keep-local/defer boundary explicit instead of
  leaving it implicit

## 6. How To Decide On A New Call Site

Use this decision order:

1. Is the code internal `src/` implementation code?
   - yes -> continue
   - no -> do not pull in private `src/` helper headers
2. Is the logic generic count/bytes arithmetic or simple array allocation?
   - yes -> use the shared helper layer
   - no -> continue
3. Is the logic dominated by symbolic accumulation, specialized cleanup, or
   larger algorithm/harness ownership?
   - yes -> keep local or defer
   - no -> use the shared helper layer
4. Does the change risk widening into public semantics/doc/example churn?
   - yes -> stop and re-scope the batch
   - no -> proceed
5. Does the touched surface change `*.c` / `*.h`?
   - yes -> run the required gate
   - then add targeted checks justified by the touched surface

## 7. Day 12 Conclusion

Sprint 41 now leaves behind a stable execution note for later Epic 4 work:

- reuse the shared helper layer for generic internal arithmetic/allocation
- keep early work internal-first
- preserve public semantics unless the sprint explicitly owns public churn
- treat symbolic/specialized/public-example cases as explicit exceptions
- validate against the Sprint 40 anchor rather than inventing new proof rules

That is the practical prep-rule layer later Epic 4 sprints need before moving
from helper consolidation into the larger lifecycle-handle refactor work.
