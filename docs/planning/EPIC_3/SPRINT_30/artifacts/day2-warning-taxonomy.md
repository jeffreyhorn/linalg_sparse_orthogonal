# Sprint 30 Day 2 Warning Taxonomy

**Date:** 2026-05-16  
**Branch:** `sprint-30`  
**Input baseline:** `artifacts/day1-cmake-build.stderr.txt`

## Objective

Turn the Day 1 raw warning stream into a usable work inventory:

- classify every warning by area and warning class,
- separate library-proper warnings from auxiliary-code warnings,
- and rank warning classes by Sprint 30 urgency versus later-sprint urgency.

## Classification Summary

### By area

- `tests`: `98`
- `benchmarks`: `13`
- `src`: `11`
- `examples`: `1`

### By warning class

- `-Wmissing-field-initializers`: `72`
- `-Wdouble-promotion`: `45`
- `-Wunused-function`: `3`
- `-Wimplicit-function-declaration`: `2`
- `-Wswitch`: `1`

### By area and warning class

- `src`
  - `-Wdouble-promotion`: `11`
- `tests`
  - `-Wdouble-promotion`: `33`
  - `-Wmissing-field-initializers`: `62`
  - `-Wunused-function`: `3`
- `benchmarks`
  - `-Wdouble-promotion`: `1`
  - `-Wimplicit-function-declaration`: `2`
  - `-Wmissing-field-initializers`: `9`
  - `-Wswitch`: `1`
- `examples`
  - `-Wmissing-field-initializers`: `1`

## Ownership Split

### Library-proper warnings

Definition for Day 2: warnings originating from `src/`.

- total library warnings: `11`
- warning classes present:
  - `-Wdouble-promotion`: `11`

Affected files:

- `src/sparse_svd.c`: `5`
- `src/sparse_qr.c`: `4`
- `src/sparse_lu.c`: `1`
- `src/sparse_ldlt.c`: `1`

### Auxiliary-code warnings

Definition for Day 2: warnings originating from `tests/`, `benchmarks/`, or `examples/`.

- total auxiliary warnings: `112`
- warning classes present:
  - `-Wmissing-field-initializers`: `72`
  - `-Wdouble-promotion`: `34`
  - `-Wunused-function`: `3`
  - `-Wimplicit-function-declaration`: `2`
  - `-Wswitch`: `1`

Top auxiliary files:

- `tests/test_ldlt.c`: `18`
- `tests/test_sprint20_integration.c`: `9`
- `tests/test_colamd.c`: `8`
- `tests/test_chol_csc.c`: `8`
- `tests/test_reorder_nd.c`: `7`
- `benchmarks/bench_main.c`: `5`

## Ranked Warning Queue

### Rank 1: `src` `-Wdouble-promotion` — Sprint 30 must-fix

Why this ranks first:

- It is the only warning class currently present in the core library.
- It appears in the exact four files identified by the Epic 3 review as Sprint 30’s highest-signal cleanup target.
- It is low-risk to fix without changing library behavior.
- Cleaning it up creates an immediate, measurable reduction in library warning debt.

Sprint placement:

- Sprint 30 Day 3-5

### Rank 2: auxiliary `-Wmissing-field-initializers` — later sprint, high volume

Why this ranks second:

- It is the largest warning class numerically (`72`).
- It is almost entirely outside the library proper.
- It reflects real maintainability drift around evolving option structs.
- It is important, but not the right first code-edit target for Sprint 30 because it does not affect the core library implementation.

Sprint placement:

- Primary target for Sprint 31 and Sprint 34 follow-through

### Rank 3: auxiliary `-Wunused-function` — later sprint, structurally important

Why this ranks third:

- It is low-volume (`3`) but high-signal.
- It points to dormant or commented-out test scaffolding rather than ordinary compile noise.
- It directly supports the Epic 3 review finding about test-suite honesty and dead scaffolding.

Sprint placement:

- Primary target for Sprint 32

### Rank 4: benchmark `-Wimplicit-function-declaration` — later sprint, portability-focused

Why this ranks fourth:

- It is low-volume (`2`) but indicates concrete portability drift in benchmark code.
- It affects developer tooling rather than the core library.
- It aligns directly with Sprint 31’s benchmark portability cleanup.

Sprint placement:

- Primary target for Sprint 31

### Rank 5: benchmark `-Wswitch` — later sprint, API/tooling drift

Why this ranks fifth:

- It is only one warning, but it signals stale enum handling in an auxiliary benchmark tool.
- It is a usability/tooling drift issue rather than a core compile-hygiene blocker.

Sprint placement:

- Primary target for Sprint 31

### Rank 6: auxiliary `-Wdouble-promotion` outside `src` — later sprint, bulk cleanup

Why this ranks sixth:

- It is worth fixing eventually, but the Sprint 30 review goal is to start with the library’s own warning debt.
- Most of these occur in tests and benchmark helpers, so they are lower-priority than the library-proper cluster.

Sprint placement:

- Later cleanup after Sprint 30 core fixes

## Day 2 Conclusion

The Day 2 taxonomy confirms that Sprint 30 should focus first on the `src/` `-Wdouble-promotion` cluster, while later sprints handle the much larger auxiliary-code warning volume. The warning backlog is therefore not “one big cleanup”; it separates cleanly into:

1. core-library compile hygiene,
2. benchmark/example tooling drift,
3. and test-structure / dormant-scaffold debt.
