# Sprint 101 Retrospective

**Sprint:** 101 - Compressed-First Product Model & Storage Front Door
**Duration:** 14 days (Days 1-14 landed on this branch)
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 101 started from the Epic 10 project-plan scope and Sprint 100
      claim/evidence handoff.
- [x] public storage and solver-entry surfaces were audited before API or
      documentation changes landed.
- [x] compressed-first API design was bounded to CSR/CSC constructor/front-door
      behavior instead of broad solver replacement.
- [x] implementation boundaries were frozen before the constructor/import
      batch.
- [x] public CSR/CSC constructor comments now describe validation and
      copy-ownership behavior.
- [x] focused CSR/CSC tests cover simple constructor success, diagnostic
      invalid input, and caller-owned buffer copy semantics.
- [x] representative solver-entry proof covers CSR-built matrices entering LU
      and CSC-built matrices entering Cholesky.
- [x] mutable `SparseMatrix` construction remains documented as supported
      compatibility rather than removed or deprecated.
- [x] README, tutorial, and examples wording now expose compressed-input
      construction as the front-door route for callers that already have
      CSR/CSC data.
- [x] a new executable compressed-input example was added and registered with
      CMake.
- [x] final validation passed:
  - `make format`
  - `make examples`
  - `make lint`
  - `make test`
  - focused CMake configure/build for `example_compressed_input` and
    `test_csr`
  - `./build/example_compressed_input`
  - `git diff --check`
  - trailing-whitespace scans
- [x] earned, deferred, and non-claim states were recorded before closeout.
- [x] Sprint 102 direct-solver oracle prerequisites and residual queue were
      handed forward explicitly.

## What Went Well

1. **The sprint started with surface audits instead of immediate edits.**
   Day 2 and Day 3 separated public construction, import, mutation, publication,
   and solver-entry paths before implementation. That kept the branch focused
   on the product-model front door rather than broad storage rewrites.

2. **The API design stayed bounded.**
   Day 4 and Day 5 chose to improve the existing CSR/CSC constructor route and
   public wording without inventing direct CSR/CSC solver APIs, no-copy
   constructors, or a replacement matrix ownership model.

3. **Ownership semantics are now testable and documented.**
   The branch documents that compressed constructors copy caller-owned arrays,
   and `tests/test_csr.c` verifies that caller mutations after construction do
   not change the created matrix.

4. **Regression proof covers both formats and representative solver entry.**
   Sprint 101 covers CSR and CSC constructor success, invalid-input handling,
   copy ownership, CSR-to-LU entry, and CSC-to-Cholesky entry. That is enough
   evidence for the bounded compressed-first product-model claim.

5. **The public workflow now has an executable example.**
   `examples/example_compressed_input.c` demonstrates diagnostic CSR
   construction, copy ownership, normal LU solve, cleanup, and expected output.

6. **Compatibility wording avoided breaking existing users.**
   README, tutorial, and example documentation now route compressed-input
   callers more clearly while preserving mutable insertion as a supported
   compatibility path.

7. **The closeout separates earned claims from future proof.**
   Day 13 and Day 14 explicitly leave direct-solver oracle work, broad solver
   parity, performance evidence, and state-of-the-art claims for later Epic 10
   sprints.

## What Didn't Go Well

1. **The public matrix shell remains central to solver entry.**
   Sprint 101 improved the compressed-input front door, but matrices still
   enter solvers through `SparseMatrix`. That is honest compatibility for this
   sprint, not a full compressed-native product model.

2. **The implementation proof is intentionally narrow.**
   The new tests prove constructor behavior and representative solver entry,
   but they do not prove every solver family or direct compressed-object solver
   path.

3. **The example is source-level rather than data-file driven.**
   The new compressed-input example is useful for API adoption, but Matrix
   Market compressed-object publication and fixture-driven examples remain
   future work.

4. **Direct-solver oracle work is still ahead.**
   Sprint 102 must still choose external fixtures, tolerances, and comparison
   rules before any broader solver claim can be made.

5. **Claim wording still needs active discipline.**
   The sprint earned a stronger front-door claim, but it would be easy for
   later public docs to overstate this as direct CSR/CSC solver support,
   performance superiority, or broad state-of-the-art status.

## Final Metrics

### Validation

| Metric | Sprint 101 close state |
|---|---:|
| full branch-level gate | `make format`, `make lint`, and `make test` passed |
| example build gate | `make examples` passed |
| example binaries built | `13` |
| focused CMake configure/build | `example_compressed_input` and `test_csr` passed |
| compressed-input example run | passed; retained copied CSR value, all-ones solution, zero residual |
| focused CSR/CSC test binary | `19` tests, `0` failures, `585` assertions |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on touched docs, examples, planning artifacts, header, and test file |

### Sprint 101 Artifact Package

| Metric | Sprint 101 close state |
|---|---:|
| total artifact files under `SPRINT_101/artifacts/` | `16` |
| baseline and audit artifacts | `4` |
| design and boundary artifacts | `2` |
| implementation and lifecycle artifacts | `4` |
| docs, examples, regression, validation, and closeout artifacts | `6` |

Notes:

- baseline and audit artifacts:
  - `day1-authoritative-inputs.txt`
  - `day1-scope-baseline.md`
  - `day2-public-storage-surface-audit.md`
  - `day3-solver-entry-path-audit.md`
- design and boundary artifacts:
  - `day4-compressed-first-api-design.md`
  - `day5-implementation-boundary-freeze.md`
- implementation and lifecycle artifacts:
  - `day6-constructor-import-batch1.md`
  - `day7-post-batch-audit-and-rerank.md`
  - `day8-lifecycle-and-ownership-design.md`
  - `day9-lifecycle-ownership-batch.md`
- docs, examples, regression, validation, and closeout artifacts:
  - `day10-compatibility-documentation-design.md`
  - `day11-docs-and-examples-follow-through.md`
  - `day12-regression-proof-expansion.md`
  - `day13-validation-and-reconciliation.md`
  - `day14-closeout-and-handoff.md`
  - `day14-artifact-index.md`

### Landed Product Surface

| Metric | Sprint 101 close state |
|---|---:|
| public constructor header updated | `include/sparse_csr.h` |
| focused test file updated | `tests/test_csr.c` |
| new executable example | `examples/example_compressed_input.c` |
| build-system example registration | `CMakeLists.txt` |
| public documentation files updated | `README.md`, `docs/tutorial.md`, `examples/README.md` |
| new representative solver-entry tests | `2` |
| new executable adoption example routes | `1` |

## Residual Deferred Debt

Most important carry-forward work:

- external oracle fixture taxonomy and tolerances for Sprint 102
- broader direct-solver comparison evidence
- solver-family-specific success and failure criteria
- decision on whether direct CSR/CSC solver entry points are justified
- Matrix Market compressed-object publication or fixture-driven examples
- broad solver-selection public guidance
- benchmark, runtime, backend, and performance sentinel evidence
- support-tier and ABI publication decisions

Still consciously constrained rather than silently solved:

- no direct solver APIs that accept `SparseCsr` or `SparseCsc` directly
- no no-copy or adopt constructor contract
- no mutable `SparseMatrix` deprecation
- no broad compressed parity across every solver family
- no portable performance superiority claim
- no broad state-of-the-art replacement claim

Not carried forward as unresolved Sprint 101 debt:

- public storage surface audit
- solver-entry path audit
- compressed-first API design
- constructor/import implementation boundary
- CSR/CSC constructor ownership documentation
- CSR/CSC invalid-input diagnostic coverage
- CSR/CSC copy-ownership regression coverage
- representative CSR-to-LU and CSC-to-Cholesky solver-entry proof
- compressed-input executable example
- compatibility-path documentation
- Sprint 102 handoff requirements

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-baseline.md](./artifacts/day1-scope-baseline.md)
- [day2-public-storage-surface-audit.md](./artifacts/day2-public-storage-surface-audit.md)
- [day3-solver-entry-path-audit.md](./artifacts/day3-solver-entry-path-audit.md)
- [day4-compressed-first-api-design.md](./artifacts/day4-compressed-first-api-design.md)
- [day5-implementation-boundary-freeze.md](./artifacts/day5-implementation-boundary-freeze.md)
- [day6-constructor-import-batch1.md](./artifacts/day6-constructor-import-batch1.md)
- [day8-lifecycle-and-ownership-design.md](./artifacts/day8-lifecycle-and-ownership-design.md)
- [day10-compatibility-documentation-design.md](./artifacts/day10-compatibility-documentation-design.md)
- [day11-docs-and-examples-follow-through.md](./artifacts/day11-docs-and-examples-follow-through.md)
- [day12-regression-proof-expansion.md](./artifacts/day12-regression-proof-expansion.md)
- [day13-validation-and-reconciliation.md](./artifacts/day13-validation-and-reconciliation.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)
- [day14-artifact-index.md](./artifacts/day14-artifact-index.md)

## Bottom Line

Sprint 101 achieved its goal:

- compressed CSR/CSC workflows are now a clearer public product entry point
- constructor ownership and diagnostics are documented and regression-tested
- compressed-built matrices are proven to enter representative existing solver
  workflows
- mutable matrix-shell construction remains supported compatibility
- public docs and examples now match the bounded product model
- final validation passed before closeout
- Sprint 102 receives direct-solver oracle work as explicit future evidence,
  not as an implied Sprint 101 claim
