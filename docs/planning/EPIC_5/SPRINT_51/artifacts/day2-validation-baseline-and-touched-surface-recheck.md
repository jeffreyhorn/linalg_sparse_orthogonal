# Sprint 51 Day 2 Artifact: Validation Baseline and Touched-Surface Recheck

## Purpose

Reconfirm the maintained reviewed baseline and truthfulness anchors Sprint 51
must preserve, then define the exact rerun set and validation boundary for the
later public direct-solver lifecycle code days.

## Main Day 2 Conclusion

Sprint 51’s validation contract is already strong enough and specific enough
that the implementation sprint does not need any custom quality policy.

The maintained baseline remains:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The correct Sprint 51 move is therefore to preserve and reuse that contract
exactly rather than reinterpret it.

## Reviewed Baseline Truthfulness

The maintained wrapper wording remains exact:

- `quality-review-full: strongest local reviewed baseline`
- `quality-review-full: rerun failing phases directly with 'make quality-review' or 'make quality-review-cmake'`

The README and maintainer guide remain aligned with that same language:

- `README.md` still treats `make quality-review-full` as the strongest local
  reviewed baseline
- `docs/maintainer_guide.md` still treats that wording as the authoritative
  maintainer close state

Interpretation:

- Sprint 51 should preserve the same baseline words and meaning
- meaningful public API batches should keep that wording visible in validation
  reporting and closeout notes

## Reviewed CMake Parity Anchor

The maintained reviewed CMake path still resolves to:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The most relevant direct-solver tests inside that reviewed suite remain:

- `test_cholesky`
- `test_ldlt`
- `test_etree`
- `test_chol_csc`
- `test_ldlt_csc`

Interpretation:

- Sprint 51 should continue to treat `53` as a truthfulness anchor
- the direct-lifecycle landing must preserve both the count and the
  Makefile/CMake parity contract

## Quality-Contract Authority Split

The live authority split remains:

- `make quality-review-full`:
  - strongest local reviewed baseline
- `make quality-review`:
  - reviewed Makefile local path
  - `format-check + lint + test + deadcode-check`
- `make quality-review-cmake`:
  - reviewed CMake parity path with full suite execution
- `make deadcode-check`:
  - report-completeness gate, not a zero-findings or removal-ready gate

Interpretation:

- Sprint 51 should use the same layered quality contract the repo already uses
- public direct-lifecycle work does not need a sprint-local alternative

## Validation Boundary For Later `*.c` / `*.h` Work

### Mandatory gate for any Sprint 51 code change

Later implementation days that modify `*.c` or `*.h` files must run:

- `make format`
- `make lint`
- `make test`

### Stronger default for substantial public API batches

When a batch changes public direct lifecycle headers, core direct integration
code, or one-shot wrapper routing, it should also run:

- `make quality-review-full`

Interpretation:

- Sprint 51 already has a fixed code-day validation boundary before header and
  source work begins
- the sprint should not blur docs-only notes with code-touch validation claims

## Targeted Touched-Surface Follow-Ons

The highest-signal later rerun set remains:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

### Why these are the right touched follow-ons

- `example_analysis` is the strongest shipped repeated-run direct caller
  surface
- `bench_refactor` and `bench_refactor_csc` are the strongest factor-many
  benchmark surfaces tied to the same analysis/refactor contract
- the direct regression binaries cover the family-level and structural seams
  most likely to matter to the phase-1 lifecycle landing

## Day 2 Operational Result

By the end of Day 2, Sprint 51 has:

- exact reviewed-baseline wording
- exact reviewed CMake count
- fixed mandatory code-day gate
- fixed stronger reviewed default
- fixed high-signal touched-surface rerun list

## Highest-Value Day 2 Conclusions

### 1. Sprint 51 does not need a custom validation story

The existing repo quality contract is already explicit and sufficient.

### 2. The direct-lifecycle landing should preserve visible truthfulness anchors

The `53`-test reviewed CMake count and strongest-local-baseline wording remain
important for substantial public API batches.

### 3. The touched rerun set is already concrete before implementation begins

Sprint 51 can move to header mapping and code landing without any remaining
validation-boundary ambiguity.
