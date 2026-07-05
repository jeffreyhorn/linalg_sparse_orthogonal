# Sprint 106 Day 9 - Giant-Test Fixture Boundary

## Goal

Define behavior-preserving fixture and helper extraction boundaries for the
largest test owners before editing test code. Day 9 intentionally does not move
test code; it separates reusable setup and assertion mechanics from test intent
so Day 10 can make small, reviewable changes.

## Inputs

- Sprint 106 plan Day 9 fixture-boundary tasks.
- Day 2 extraction target re-rank and giant-test inventory.
- Current line counts and helper-density samples from:
  - `tests/test_ldlt_csc.c`
  - `tests/test_integration.c`
  - `tests/test_qr.c`
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_lu_csr.c`
- Existing helper-header patterns:
  - `tests/test_solver_helpers.h`
  - `tests/test_iterative_handle_helpers.h`
  - `tests/test_svd_partial_helpers.h`
  - `tests/test_chol_csc_supernodal_helpers.h`

## Inventory Snapshot

| owner | lines | local helper/macro blocks | test cases | reusable helper pressure |
|---|---:|---:|---:|---|
| `tests/test_ldlt_csc.c` | 3,884 | 137 | 100 | direct CSC fixture builders, row-adjacency assertions, residual/oracle helpers |
| `tests/test_integration.c` | 3,421 | 68 | 53 | lifecycle matrix builders, progress callbacks, refactor-preservation helpers |
| `tests/test_qr.c` | 3,234 | 84 | 73 | reconstruction, residual, dense/sparse comparison, SuiteSparse oracle helpers |
| `tests/test_graph.c` | 2,925 | 79 | 61 | synthetic graph builders, partition invariants, SuiteSparse smoke helpers |
| `tests/test_reorder_nd.c` | 2,340 | 56 | 35 | synthetic graph builders, cached SuiteSparse fixture copies, ND residual helpers |
| `tests/test_lu_csr.c` | 1,899 | 59 | 53 | matrix comparison, LU factorization verification, residual helpers |

The line and helper counts confirm that fixture pressure is not isolated to one
test family. Day 10 should still move only narrow helper seams and preserve all
test names and assertions.

## Candidate Helper Classes

### Graph And Reorder Fixtures

Likely files:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Reusable mechanics:

- 2D grid builders
- 1D path builders
- 3D mesh builders
- two-clique and complete/bipartite graph builders
- partition-side counters and invariant checks
- SuiteSparse load/copy cache helpers for heavy reorder fixtures

Recommended owner:

- `tests/test_graph_fixtures.h`

Rationale:

- This is the clearest Day 10 starting point because the helpers are fixture
  builders and invariant checks, not solver algorithms.
- Both direct graph partition tests and nested-dissection tests need similar
  generated graph shapes.
- A header-only helper can preserve CTest registration and avoid test target
  membership changes.

### Direct-Solver Assertions And Residual Helpers

Likely files:

- `tests/test_lu_csr.c`
- `tests/test_ldlt_csc.c`
- `tests/test_direct_csc_regression.c`

Reusable mechanics:

- matrix equality helpers
- LU/LDLT factorization verification helpers
- norm/residual helpers
- direct CSC fixture builders for small SPD, KKT, tridiagonal, and indefinite
  fixtures

Recommended owner:

- `tests/test_direct_solver_helpers.h`

Rationale:

- Day 4-8 extracted direct-solver source seams; Day 10 should tighten the proof
  layer around those seams.
- The first move should be assertion and residual helpers, not test bodies.
- Direct CSC and LU CSR validation can share explicit helper names while keeping
  each test's headline intent local.

### Integration Lifecycle Helpers

Likely file:

- `tests/test_integration.c`

Reusable mechanics:

- lifecycle matrix builders such as tridiagonal SPD, unsymmetric, KKT, CSR, and
  CSC constructor fixtures
- progress callback counters and cancellation helpers
- refactor-preservation assertion helpers

Recommended owner:

- Defer to Day 11, with a candidate name of `tests/test_lifecycle_helpers.h`.

Rationale:

- `tests/test_integration.c` has the highest recent churn and broadest
  cross-family behavior surface.
- Moving helpers before the direct/graph helper pattern is proven would raise
  review risk.

### QR Oracle Helpers

Likely file:

- `tests/test_qr.c`

Reusable mechanics:

- QR reconstruction error helpers
- true residual helpers
- dense-vs-sparse QR comparison helpers

Recommended owner:

- Defer until after direct/graph extraction; candidate name
  `tests/test_qr_helpers.h`.

Rationale:

- Day 7 already extracted QR implementation helpers.
- QR test helpers are valuable but less urgent than graph/reorder duplication
  and direct-solver proof ownership after Day 4-8 source work.

## Ownership And Naming Rules

- Prefer header-only helper owners for test helpers unless a helper needs
  mutable global state or a large compiled implementation.
- Name helper headers by proof family, not by sprint:
  - good: `test_graph_fixtures.h`, `test_direct_solver_helpers.h`
  - avoid: `test_s106_helpers.h`
- Keep helper names specific:
  - good: `tf_make_grid_2d(...)`, `tf_assert_partition_invariant(...)`
  - avoid: broad names like `make_matrix(...)` or `check(...)`
- Use the `tf_` prefix for reusable test-fixture helpers.
- Do not move test bodies into helper headers unless the file already follows
  that pattern for a tightly scoped repeated-run proof family.
- Keep test names, `RUN_TEST(...)` order, and reviewed CTest target counts
  unchanged unless a later artifact explicitly approves a registration change.
- Preserve printed fixture labels where CI or review artifacts use them as
  evidence.
- Keep SuiteSparse fixture cache helpers scoped to the test family unless they
  are needed by at least two owners.

## Build And Registration Mapping

### Header-Only Helper Extraction

Expected updates:

- Add `tests/<helper>.h`.
- Include it from affected `tests/*.c` owners.
- No Makefile test-target update required because tests compile through their
  existing `.c` owner.
- No CMake test-target update required for normal compilation.
- No CTest registration change.
- `make format && make lint && make test` is required because `.c` and `.h`
  files changed.

### Compiled Helper Extraction

Use only if helper code becomes too large or needs state that should not live in
a header.

Expected updates:

- Add `tests/<helper>.c` and `tests/<helper>.h`.
- Update each affected Makefile test target link line.
- Update each affected CMake executable source list.
- Run CMake registration/count checks because target membership changed.
- Preserve test target names and `RUN_TEST(...)` names unless separately
  approved.

## Selected Day 10 Targets

### Target 1: Graph/Reorder Fixture Header

Proposed file:

- `tests/test_graph_fixtures.h`

Candidate helpers:

- `tf_make_grid_2d(...)`
- `tf_make_path_1d(...)`
- `tf_make_mesh_3d(...)`
- `tf_make_two_cliques_with_bridge(...)`
- `tf_assert_partition_invariant(...)` or a non-asserting predicate plus
  explicit call-site assertion

Affected tests:

- `tests/test_graph.c`
- `tests/test_reorder_nd.c`

Focused validation:

- `make build/test_graph build/test_reorder_nd`
- `./build/test_graph`
- `./build/test_reorder_nd`
- `make large-matrix-guardrails` only if guardrail-owned fixtures or policies
  are touched

### Target 2: Direct-Solver Assertion Header

Proposed file:

- `tests/test_direct_solver_helpers.h`

Candidate helpers:

- matrix equality helper currently local to LU CSR tests
- residual norm helper currently local to LU CSR tests
- direct-solver factorization verification helpers that are assertion-only and
  do not obscure test-specific fixture intent

Affected tests:

- `tests/test_lu_csr.c`
- `tests/test_ldlt_csc.c`, only if the moved helper is actually shared
- `tests/test_direct_csc_regression.c`, only if the moved helper is actually
  shared

Focused validation:

- `make build/test_lu_csr build/test_ldlt_csc build/test_direct_csc_regression`
- `./build/test_lu_csr`
- `./build/test_ldlt_csc`
- `./build/test_direct_csc_regression`

## Deferred Targets

| target | reason |
|---|---|
| integration lifecycle helpers | high value but high blast radius; Day 11 should own this after Day 10 proves naming/validation |
| QR oracle helper extraction | useful, but lower immediate value than graph/reorder and direct-solver proof helpers |
| broad direct CSC fixture split | risky if it moves test intent; start with assertion helpers only |
| compiled test helper objects | unnecessary unless header-only extraction becomes too large or stateful |

## Validation Checklist Before Day 10 Edits

- Confirm the helper to move is behavior-preserving setup, assertion, or
  residual code, not the test's headline behavior.
- Confirm call sites remain readable after extraction.
- Confirm no `RUN_TEST(...)` names, order, or counts change.
- Confirm whether helper extraction is header-only or compiled.
- For header-only extraction, run focused tests plus the full C quality gate.
- For compiled extraction, also run CMake build/registration parity checks.
- For graph/reorder fixture changes, run large-matrix guardrails only if the
  extraction touches guardrail-owned fixture/policy paths.

## Exit Criteria

Day 9 satisfies the fixture-boundary criteria:

- direct, graph/reorder, and integration test owners are represented;
- fixture extraction targets are separated from behavior changes;
- helper naming and ownership rules are defined;
- Make/CMake/test-registration follow-through is mapped;
- validation commands are known before Day 10 test edits.
