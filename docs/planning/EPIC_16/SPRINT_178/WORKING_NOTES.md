# Sprint 178 Working Notes

**Sprint:** 178 - Allocation-Failure Proof Batch 2
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_178/`
**Status:** Complete

## Source Artifact Note

The Sprint 178 source section lives in
`docs/planning/EPIC_16/PROJECT_PLAN.md` under "Sprint 178:
Allocation-Failure Proof Batch 2". Sprint 178 artifacts in this directory
follow the Epic 16 scope.

## Sprint Goal

Add deterministic allocation-failure cleanup evidence for one additional
high-risk subsystem beyond iterative repeated-run handles.

## Baseline Inputs

- `docs/planning/EPIC_16/PROJECT_PLAN.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day8-gate-templates.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day10-quality-surface-map.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day11-claim-boundary-freeze.md`
- `docs/planning/EPIC_16/SPRINT_177/artifacts/day12-handoff-package.md`
- `docs/planning/EPIC_16/SPRINT_177/RETROSPECTIVE.md`
- allocation hook owners: `src/sparse_alloc_internal.c` and
  `src/sparse_alloc_internal.h`
- current focused proof owners: `tests/test_iterative.c`,
  `tests/test_iterative_handle_helpers.h`, `Makefile`, and `CMakeLists.txt`
- candidate subsystem surfaces: matrix construction/conversion, direct solver
  setup/factorization, and decomposition workspace owners
- public wording owners: `README.md` and `docs/maintainer_guide.md`

## Starting Branch Snapshot

- Branch: `sprint-178`
- Starting commit: `3907e7545a58c462e24eb3d0d4df1ef7a75589bf`
- Recent base context:
  - `3907e754` Merge pull request #197 from `sprint-177`
  - `4bca0a10` Address PR #197 review comments
  - `aad776d9` Move Sprint 177 planning artifacts to Epic 16
  - `6ca0e39a` Add Sprint 177 planning artifacts
  - `8e5a759f` Merge pull request #196 from `planning/epic-16`

## Current Allocation-Failure Proof Baseline

| Surface | Current state |
| --- | --- |
| Private allocation hook | `sparse_alloc_test_fail_after()` and `sparse_alloc_test_reset()` live in `src/sparse_alloc_internal.*`; hook state is private/internal and not public API. |
| Countdown semantics | `remaining == 0` fails the next wrapped allocation once, then resets to normal allocation behavior; positive values count down before the single injected failure. |
| Focused Make gate | `make iterative-allocation-failure-gate` runs the existing iterative allocation-failure proof. |
| Focused CTest selector | `test_iterative` carries the `allocation_failure` CTest label. |
| Current covered family | CG, GMRES, and MINRES repeated-run handle prepare/growth cleanup. |
| Current public boundary | README and maintainer guide state this is selected, family-local allocation-failure evidence, not broad coverage. |

## Sprint 178 Project-Plan Items

| Item | Name | Status | Notes |
| --- | --- | --- | --- |
| 178.1 | Subsystem Selection Detail | Complete | Day 3 selects `sparse_matmul()` workspace allocation and freezes entry points, failure sites, out-of-scope paths, and non-claims. |
| 178.2 | Cleanup Invariant Record | Complete | Day 4 defines `sparse_matmul()` workspace cleanup, no-publication, retry, and unsupported-breadth invariants. |
| 178.3 | Harness Extension | Complete | Day 6 adds the `test_matmul` helper harness for the selected `sparse_matmul()` workspace failure sites without public API changes. |
| 178.4 | Regression Tests | Complete | Day 7 adds the first-site stale-output regression for `acc`; Day 8 expands the same no-stale-public-state and retry assertion to `nz_flag` and `touched`. |
| 178.5 | Focused Gate | Complete | Day 10 adds `make matmul-allocation-failure-gate`, CTest labels for `test_matmul`, and a registration guard. |
| 178.6 | Claim Documentation and Validation | Complete | Day 11 documents the scoped `sparse_matmul()` allocation-failure claim and focused gate; Day 12 completes integrated focused, CMake/CTest, docs hygiene, and full C quality validation. |

## Daily Log

### Day 1 - Sprint Intake And Gate Baseline

Status: Complete

Completed:

- Re-read the Sprint 178 project-plan section.
- Reviewed Sprint 177 Gate 1 and Day 12 Sprint 178 handoff.
- Created Sprint 178 working notes and artifact directory structure.
- Recorded current Sprint 176 allocation-failure proof status and owners.
- Recorded protected non-claims for broad allocation-failure coverage.
- Created the Day 1 sprint-intake artifact.

Validation:

- `git diff --check`

## Open Risks

- Sprint 178 must not select more than one additional subsystem.
- The selected subsystem must have deterministic failure sites and observable
  cleanup/retry behavior without adding public test API.
- Any C or header edits require `make format && make lint && make test`.
- Documentation must continue to use "allocation-failure" consistently and
  avoid "allocator-failure" drift.
- Public docs must not imply broad allocation-failure coverage for direct
  solvers, eigensolvers, matrix construction, package/install flows,
  generated tooling, or unrelated allocation paths unless the selected
  subsystem earns that exact evidence.

## Handoff Notes

- Day 2 should inventory at least three candidate allocation-heavy subsystems
  and compare closure fitness before selection.
- Day 3 should select exactly one subsystem and record in-scope entry points,
  ownership paths, failure sites, and explicit non-claims.

### Day 2 - Allocation Surface Inventory

Status: Complete

Completed:

- Inspected current private allocation-failure hook semantics and current
  iterative proof ownership.
- Inventoried candidate matrix construction and conversion paths.
- Inventoried candidate direct solver setup and factorization paths.
- Inventoried candidate decomposition workspace ownership paths.
- Compared candidates by public state exposure, cleanup observability, retry
  feasibility, hook reachability, and implementation risk.
- Recorded that many direct/decomposition paths still use raw `malloc` and
  `calloc`, so deterministic proof may require conversion to wrapped helpers
  or a local harness.
- Created the Day 2 allocation-surface inventory artifact.

Validation:

- `git diff --check`

Handoff:

- Day 3 should choose exactly one subsystem. The strongest low-risk candidate
  is matrix shell allocation and matrix multiply workspace because these paths
  already use wrapped allocation helpers and have observable public-state and
  retry behavior. Higher-value direct/decomposition paths should be selected
  only if the sprint is willing to convert a narrow set of raw allocations or
  add a carefully scoped private harness.

### Day 3 - Subsystem Selection Detail

Status: Complete

Completed:

- Reviewed the Day 2 allocation-surface inventory.
- Selected exactly one Sprint 178 subsystem:
  `sparse_matmul()` workspace allocation.
- Defined the in-scope public entry point: `sparse_matmul(const SparseMatrix
  *A, const SparseMatrix *B, SparseMatrix **C)`.
- Froze the selected deterministic failure sites: allocation of `acc`,
  `nz_flag`, and `touched` after output matrix shell creation succeeds.
- Defined ownership paths for input matrices, local workspace arrays, output
  matrix cleanup, and `*C` no-publication behavior.
- Recorded adjacent out-of-scope allocation paths and retained non-claims.
- Created the Day 3 subsystem-selection artifact.

Validation:

- `git diff --check`

Handoff:

- Day 4 should convert the Day 3 selection into cleanup invariants: `*C`
  remains `NULL` on injected workspace allocation failure, partially allocated
  workspace is freed, the temporary output matrix is freed, inputs remain
  reusable, and a retry after hook reset succeeds.

### Day 4 - Cleanup Invariant Record

Status: Complete

Completed:

- Traced selected `sparse_matmul()` ownership from validation through output
  matrix creation, workspace allocation, cleanup, and final publication.
- Defined public-state invariants for `C`, `*C`, `A`, and `B`.
- Defined internal cleanup invariants for `acc`, `nz_flag`, `touched`, and
  the temporary output matrix `out`.
- Defined retry expectations after `sparse_alloc_test_reset()`.
- Defined unsupported breadth and wording limits for future docs.
- Created the Day 4 cleanup-invariant artifact.

Validation:

- `git diff --check`

Handoff:

- Day 5 should decide whether the existing fail-at-count hook is sufficient
  for the selected `acc`, `nz_flag`, and `touched` allocation sites. The
  invariant record expects no public API changes and no broad allocation hook
  redesign.

### Day 5 - Harness Design

Status: Complete

Completed:

- Compared the existing Sprint 176 fail-at-count hook against the selected
  `sparse_matmul()` workspace allocation sites.
- Confirmed no new public API or broad hook redesign is needed.
- Defined the selected fail-at counts for the current `sparse_matmul()` path:
  `acc` after 6 prior wrapped allocations, `nz_flag` after 7, and `touched`
  after 8.
- Designed test helper behavior for fixture creation, injected failure,
  `*C == NULL` assertion, hook reset, successful retry, and numeric product
  verification.
- Defined reset expectations that keep the one-shot hook semantics from
  Sprint 176 intact.
- Created the Day 5 harness-design artifact.

Validation:

- `git diff --check`

Handoff:

- Day 6 should implement only test-side helper constants/functions for the
  selected `sparse_matmul()` workspace sites. It should not change public
  headers or add new product API. If implementation edits are needed, the
  sprint must run `make format && make lint && make test`.

### Day 6 - Harness Implementation

Status: Complete

Completed:

- Added `tests/test_matmul.c` coverage that includes the private
  allocation-failure hook.
- Added fixed fail-after constants for the selected `sparse_matmul()`
  workspace allocations: `acc`, `nz_flag`, and `touched`.
- Added local fixture builders that construct `A` and `B` before enabling the
  failure hook.
- Added a helper that injects each selected workspace failure, verifies
  `SPARSE_ERR_ALLOC`, asserts `*C == NULL`, resets the hook, retries, and
  verifies the expected numeric product.
- Preserved Sprint 176 one-shot countdown semantics and avoided public API or
  product-code changes.
- Created the Day 6 harness-implementation artifact.

Validation:

- `make build/test_matmul`
- `./build/test_matmul`
- `make format`
- `make lint`
- `make test`

Handoff:

- Day 7 should use the new harness as the base for any remaining regression
  assertions and decide whether the selected `sparse_matmul()` coverage is
  sufficient before adding a focused gate in the later sprint days.

### Day 7 - First Failure Regression

Status: Complete

Completed:

- Added the first dedicated deterministic failure regression for the selected
  `sparse_matmul()` subsystem.
- Targeted the first selected workspace allocation site: `acc`.
- Initialized the caller output pointer with a separate stale matrix before
  invoking `sparse_matmul()`.
- Asserted `SPARSE_ERR_ALLOC` for the injected `acc` allocation failure.
- Asserted the public output pointer is cleared to `NULL`, proving no stale
  output publication is observable on the first selected failure path.
- Asserted the separate stale matrix remains caller-owned and unchanged.
- Reset the hook and verified retry success with the expected product.
- Created the Day 7 first-regression artifact.

Validation:

- `make build/test_matmul && ./build/test_matmul`
- `make format`
- `make lint`
- `make test`

Handoff:

- Day 8 should expand the stale-output regression pattern to the remaining
  selected workspace allocation sites (`nz_flag` and `touched`) while keeping
  adjacent allocation paths out of scope.

### Day 8 - Failure Coverage Expansion

Status: Complete

Completed:

- Added a shared stale-output regression helper for selected `sparse_matmul()`
  workspace allocation failures.
- Preserved the Day 7 `acc` stale-output regression and reused the helper for
  the first selected site.
- Added regression coverage for the remaining selected workspace allocation
  sites: `nz_flag` and `touched`.
- Asserted `SPARSE_ERR_ALLOC`, `C == NULL`, caller-owned stale matrix
  preservation, hook reset, successful retry, and numeric product correctness
  for the remaining selected ownership paths.
- Kept `sparse_create()` shell allocation, `sparse_insert()` product-flush
  allocation, and unrelated subsystem allocation paths out of scope.
- Created the Day 8 coverage-expansion artifact.

Validation:

- `make build/test_matmul && ./build/test_matmul`
- `make format`
- `make lint`
- `make test`

Handoff:

- Day 9 should confirm no product cleanup changes are needed for the selected
  workspace failures, preserve public error-ordering contracts, and prepare
  the focused gate without broadening the allocation-failure claim.

### Day 9 - Cleanup And Error Contracts

Status: Complete

Completed:

- Reviewed the selected `sparse_matmul()` workspace allocation cleanup path
  after the Day 7-8 regressions.
- Confirmed no product-code cleanup fix is required for the selected `acc`,
  `nz_flag`, and `touched` allocation failures.
- Added an error-precedence regression for `sparse_matmul()` that asserts
  `C == NULL` returns `SPARSE_ERR_NULL` before dereference.
- Asserted non-`NULL` output pointers are cleared before null-input and
  shape-mismatch rejection paths.
- Asserted the caller-owned stale matrix remains unchanged after rejected
  public calls.
- Verified retry success with the selected fixture product.
- Created the Day 9 cleanup/error-contract artifact.

Validation:

- `make build/test_matmul && ./build/test_matmul`
- `make format`
- `make lint`
- `make test`

Handoff:

- Day 10 should add focused Make/CTest registration for the selected
  `sparse_matmul()` allocation-failure proof without broadening the claim.

### Day 10 - Focused Gate Registration

Status: Complete

Completed:

- Added `make matmul-allocation-failure-gate` as the focused maintained
  command for the selected `sparse_matmul()` allocation-failure proof.
- Added `matmul;allocation_failure` CTest labels to `test_matmul`.
- Added `tests/test_matmul_allocation_failure_gate_registration.py` to guard
  Makefile, CMake, and `test_matmul` registration drift.
- Kept the focused gate scoped to `test_matmul` because the selected
  allocation-failure regressions share fixture and retry helpers with the
  existing matrix multiply tests.
- Preserved non-claims for matrix shell allocation, product-flush allocation,
  unrelated sparse matrix operations, solvers, package/install flows, and
  generated tooling.
- Created the Day 10 focused-gate artifact.

Validation:

- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `make matmul-allocation-failure-gate`
- CMake configure/build for `test_matmul`
- `ctest --test-dir build-sprint178-day10 -N -L allocation_failure`
- `ctest --test-dir build-sprint178-day10 --output-on-failure -L matmul`
- `make format`
- `make lint`
- `make test`

Handoff:

- Day 11 should update README and maintainer guidance with the scoped
  `make matmul-allocation-failure-gate` command while preserving broad
  allocation-failure non-claims.

### Day 11 - Scoped Claim Documentation

Status: Complete

Completed:

- Updated README quality wording to name both maintained selected
  allocation-failure proofs: the Sprint 176 iterative repeated-run handle
  proof and the Sprint 178 `sparse_matmul()` workspace proof.
- Added `make matmul-allocation-failure-gate` to the README command list.
- Added README repeated-run guidance that keeps the matrix multiply proof
  separate from iterative handle semantics and names the excluded matrix shell,
  insertion/product flush, conversion, solver, package/install, and generated
  tooling surfaces.
- Updated maintainer guidance with the `tests/test_matmul.c` owner, exact
  selected regression names, Make gate, CTest selector, registration guard, and
  non-claim boundary.
- Preserved consistent "allocation-failure" terminology across touched docs.
- Created the Day 11 scoped-claim artifact.

Validation:

- `make matmul-allocation-failure-gate`
- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `rg -n "allocator-failure" README.md docs/maintainer_guide.md docs/planning/EPIC_16/SPRINT_178 || true`
- `git diff --check`

Handoff:

- Day 12 should run integrated validation and reconcile the Sprint 178 evidence
  table before closeout. The public claim remains scoped to selected
  `sparse_matmul()` workspaces, not broad allocation-failure coverage.

### Day 12 - Integrated Validation

Status: Complete

Completed:

- Ran the focused `sparse_matmul()` allocation-failure Make gate.
- Ran the standalone registration guard for Makefile, CMake, and
  `test_matmul` registration drift.
- Ran CMake configure/build for `test_matmul` and `test_iterative` in an
  isolated `build-sprint178-day12` tree.
- Confirmed CTest allocation-failure registration discovers exactly the
  `test_matmul` and `test_iterative` focused lanes.
- Ran the `matmul` CTest selector and the full `allocation_failure` selector.
- Ran documentation hygiene for the disallowed `allocator-failure` spelling;
  only intentional command/evidence references and the Day 1 anti-drift note
  remain.
- Ran the required full C quality gate because Sprint 178 modified C code.
- Created the Day 12 integrated-validation artifact.

Validation:

- `make matmul-allocation-failure-gate`
- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `cmake -S . -B build-sprint178-day12`
- `cmake --build build-sprint178-day12 --target test_matmul test_iterative --parallel 1`
- `ctest --test-dir build-sprint178-day12 -N -L allocation_failure`
- `ctest --test-dir build-sprint178-day12 --output-on-failure -L matmul`
- `ctest --test-dir build-sprint178-day12 --output-on-failure -L allocation_failure`
- `rg -n "allocator-failure" README.md docs/maintainer_guide.md docs/planning/EPIC_16/SPRINT_178 || true`
- `git diff --check`
- `make format && make lint && make test`

Handoff:

- Day 13 should reconcile the evidence and public claims against the retained
  non-claims. The validated positive claim remains limited to selected
  `sparse_matmul()` workspace allocation failure cleanup, no stale output, and
  retry-after-reset behavior.

### Day 13 - Claim Recalibration And Residuals

Status: Complete

Completed:

- Compared Sprint 178 evidence against Sprint 177 Gate 1.
- Confirmed the earned claim names exactly one additional subsystem:
  `sparse_matmul()` workspace allocation.
- Confirmed evidence covers deterministic injected failures for accumulator,
  nonzero-flag, and touched-column workspaces.
- Confirmed tests assert stale-output suppression and retry-after-reset
  behavior for the selected workspace failures.
- Confirmed Make and CTest focused validation are present and bounded:
  `make matmul-allocation-failure-gate`, `ctest -L matmul`, and
  `ctest -L allocation_failure`.
- Confirmed README and maintainer wording preserve broad allocation-failure
  non-claims.
- Recorded retained residuals for unselected allocation paths.
- Created the Day 13 claim-recalibration artifact.

Validation:

- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `rg -n "Selected allocation-failure proofs|matmul-allocation-failure-gate|tests/test_matmul\\.c.*owns|matrix multiply allocation-failure proof" README.md docs/maintainer_guide.md`
- `rg -n "broad allocation-failure|not broad allocation-failure|does not establish broad allocation-failure" README.md docs/maintainer_guide.md docs/planning/EPIC_16/SPRINT_178/artifacts`
- `git diff --check`

Residuals:

- Matrix shell construction, insertion/product flush, matrix copy, transpose,
  CSR/CSC conversion, and build-helper allocation remain unproven.
- Direct solvers, QR, LDLT, Cholesky, SVD, eigensolvers, graph routines, and
  reorder routines remain outside the Sprint 178 allocation-failure proof.
- Package/install flows and generated-report tooling remain outside this proof.
- The allocation-failure hook remains private/internal and is not public API.

Handoff:

- Day 14 should close Sprint 178 by finalizing artifact inventory, working
  notes, and retrospective inputs. No additional claim widening should occur
  unless new evidence is added and validated.

### Day 14 - Sprint Closeout

Status: Complete

Completed:

- Finalized the Sprint 178 artifact inventory from Day 1 through Day 14.
- Confirmed all Sprint 178 items are complete:
  - subsystem selection;
  - cleanup invariant record;
  - harness extension;
  - regression tests;
  - focused Make/CTest gate;
  - scoped claim documentation and integrated validation.
- Confirmed the selected allocation-failure proof has focused evidence and no
  documented blocker.
- Confirmed broad allocation-failure non-claims remain protected.
- Prepared retrospective inputs for the Sprint 178 retrospective.
- Recorded the Sprint 179 handoff confirmation for generated API HTML
  publication decision work.
- Created the Day 14 sprint-closeout artifact.

Validation:

- `find docs/planning/EPIC_16/SPRINT_178 -maxdepth 2 -type f | sort`
- `python3 tests/test_matmul_allocation_failure_gate_registration.py`
- `rg -n "broad allocation-failure|not broad allocation-failure|does not establish broad allocation-failure" README.md docs/maintainer_guide.md docs/planning/EPIC_16/SPRINT_178/artifacts`
- `git diff --check`

Retrospective inputs:

- Sprint 178 selected one additional allocation-heavy subsystem,
  `sparse_matmul()`, and completed deterministic allocation-failure proof for
  selected workspace allocations.
- The proof remained private-hook-backed and did not add public test-injection
  API.
- The focused gate and CTest labels give maintainers a single command and a
  discoverable selector for the new proof.
- Full integrated validation passed on Day 12.
- Remaining allocation-failure gaps are explicitly residual and should be
  selected one subsystem at a time.

Sprint 179 handoff:

- Sprint 179 should move to generated API HTML publication decision work using
  `docs/planning/EPIC_16/PROJECT_PLAN.md` lines 98-132 as the source section.
- Sprint 178 leaves no generated API HTML implementation changes.
- Allocation-failure claim wording should remain unchanged unless future
  evidence broadens it.
