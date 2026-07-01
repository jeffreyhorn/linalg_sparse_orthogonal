# Sprint 101 Working Notes

## Day 1 - Sprint 101 Scope & Baseline Setup

### Goal

Open Sprint 101 from the completed Sprint 100 evidence contract and the Epic
10 project-plan scope. Day 1 does not audit every storage surface or start
compressed-first implementation. It creates the bounded execution frame,
authoritative input list, workstream inventory, artifact structure, and
validation expectations that Days 2-14 should follow.

### Actions

- Re-read the Sprint 101 section of
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read Sprint 101 Day 1 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read the Sprint 100 handoff inputs required for Sprint 101:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day6-state-of-the-art-target.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day8-claim-dependency-model.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day12-public-claim-audit.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-sprint100-handoff-package.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-claim-non-goal-register.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day14-closeout-and-validation.md`
- Created the Sprint 101 artifact directory:
  `docs/planning/EPIC_10/SPRINT_101/artifacts/`.
- Recorded authoritative inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Day 1 scope baseline, workstream inventory, day ownership,
  guardrails, and validation expectations in
  `artifacts/day1-scope-baseline.md`.

### Workstreams

Sprint 101 is organized around seven compressed-first product-model
workstreams:

1. Storage surface audit.
2. Compressed-first API design.
3. Constructor and import implementation.
4. Lifecycle and ownership clarification.
5. Compatibility path documentation.
6. Regression proof.
7. Validation and Sprint 102 handoff.

### Findings

- Sprint 100 marks compressed-first workflows as a candidate product-maturity
  claim, not an earned claim. Sprint 101 must earn it with audit, design,
  implementation, tests, and docs/examples.
- Sprint 100 also marks the mutable matrix shell as compatibility-supported.
  Sprint 101 can make CSR/CSC workflows more central, but it must not claim a
  full replacement of the linked-list shell.
- The Sprint 101 plan needs a two-stage opening: audit first, then design,
  then implementation. This prevents API or docs changes from outrunning the
  ownership and validation model.
- If Sprint 101 touches `.c` or `.h` files, the required quality chain is
  `make format && make lint && make test`.
- The most important Day 1 guardrail is product-language discipline:
  compressed-first should become the obvious front door, while mutable
  matrix-shell workflows remain supported compatibility.

### Validation Expectations

- Day 1 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, package, or test
  files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only setup pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Day 1 Exit State

Sprint 101 now has a scoped execution baseline, authoritative input list,
seven workstreams, artifact structure, validation expectations, and landing
order. Day 2 can audit the public storage surface without reopening Sprint
101 scope or starting implementation prematurely.

## Day 2 - Public Storage Surface Audit

### Goal

Audit public construction, compressed import/export, mutation, and publication
paths for remaining linked-list-first product costs. Day 2 does not change
APIs, code, tests, or public docs. It classifies the current storage surface so
Day 4 can design the highest-value compressed-first refinements.

### Actions

- Re-read Sprint 101 Day 2 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Inspected current public storage headers and implementation:
  - `include/sparse_matrix.h`
  - `include/sparse_csr.h`
  - `src/sparse_matrix.c`
  - `src/sparse_csr.c`
- Inspected current public storage narrative:
  - `README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
- Traced constructor, mutation, import, export, Matrix Market, and publication
  surfaces using `rg`.
- Recorded the Day 2 public storage surface audit in
  `artifacts/day2-public-storage-surface-audit.md`.

### Findings

- The project already exposes real compressed-first public constructors:
  `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)`.
- README already names the compressed-first one-shot workflow for callers whose
  data arrives as CSR or CSC.
- The longer tutorial and example set still begin from mutable matrix-shell
  insertion, which keeps the learning path more linked-list-first than the
  product target wants.
- `sparse_from_csr(...)` and `sparse_from_csc(...)` preserve explicit
  `sparse_err_t` diagnostics, but their names and surrounding guidance read
  more like compatibility conversion wrappers than diagnostic front-door
  constructors.
- The implementation validates CSR/CSC structure carefully, but still builds
  the public `SparseMatrix` shell through `sparse_create` plus per-entry
  `sparse_insert`.
- The strongest Day 4 candidates are diagnostic constructor clarity,
  compressed-input examples/tutorial guidance, focused bad-input tests, and
  possibly a bounded build-path improvement if Day 3 shows solver-entry value.

### Validation Expectations

- Day 2 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, package, public
  docs, examples, or test files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only audit pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Day 2 Exit State

Sprint 101 now has a public storage surface audit and a ranked
compressed-first candidate list. Day 3 can audit solver entry paths to decide
whether constructor/import refinements are sufficient or whether solver-facing
compressed workflows need additional ownership or documentation work.

## Day 3 - Solver Entry Path Audit

### Goal

Audit direct, iterative, eigensolver, SVD, and analysis entry paths for
compressed-first adoption costs. Day 3 keeps solver-path risks separate from
Day 2 storage-constructor risks and does not introduce any broad solver-parity
claim.

### Actions

- Re-read Sprint 101 Day 3 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Inspected public solver headers:
  - `include/sparse_lu.h`
  - `include/sparse_lu_csr.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_analysis.h`
  - `include/sparse_qr.h`
  - `include/sparse_svd.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `include/sparse_ilu.h`
  - `include/sparse_ic.h`
- Inspected current solver workflow narrative:
  - `README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
- Traced current references to CSR/CSC construction, solver entry, direct
  repeated-run lifecycle, iterative handles, eigensolver handles, and
  matrix-free solve paths.
- Recorded the Day 3 solver entry path audit in
  `artifacts/day3-solver-entry-path-audit.md`.

### Findings

- Most public solver families still accept `SparseMatrix` as the public
  coefficient object, even when their implementation uses compressed or
  compressed-like internals.
- The repeated direct lifecycle is the strongest solver-facing reuse story:
  `sparse_analyze(...)`, `sparse_factor_numeric(...)`,
  `sparse_factor_solve(...)`, and `sparse_refactor_numeric(...)` already give
  callers an explicit analyze-once / factor-many owner model.
- Cholesky and LDLT have meaningful CSC-backed internal paths, and LU exposes
  a public CSR working-format specialist surface, but none of those eliminates
  the need for a public `SparseMatrix` shell in the normal user workflow.
- Iterative matrix-free APIs let advanced callers wrap compressed-native
  matvecs without building a matrix shell, but the library does not yet
  promote a built-in CSR/CSC adapter as the obvious compressed-input solver
  route.
- QR, SVD, eigensolver, ILU, ILUT, and IC remain public `SparseMatrix`-first
  entry points. Their ownership models are valid, but they should not be used
  to claim broad direct CSR/CSC solver parity.
- The strongest Day 4 direction is to make the existing compressed
  constructors plus solver entry flow feel coherent, then prove that workflow
  with focused tests and documentation rather than adding broad solver-family
  APIs.

### Validation Expectations

- Day 3 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, package, public
  docs, examples, or test files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only audit pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Day 3 Exit State

Sprint 101 now has separate storage-surface and solver-entry audits. Day 4
can reconcile those audits into a bounded compressed-first API design that
promotes existing CSR/CSC constructors, preserves `SparseMatrix` compatibility,
and avoids unearned solver-parity claims.

## Day 4 - Compressed-First API Design

### Goal

Reconcile the Day 2 storage audit and Day 3 solver audit into a bounded
CSR/CSC-first API design. Day 4 does not implement the design. It selects the
highest-value refinements, defines ownership and error semantics, records
compatibility behavior, and sets up Day 5's implementation boundary freeze.

### Actions

- Re-read Sprint 101 Day 4 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read:
  - `artifacts/day2-public-storage-surface-audit.md`
  - `artifacts/day3-solver-entry-path-audit.md`
- Re-inspected current compressed constructor contracts in:
  - `include/sparse_csr.h`
  - `src/sparse_csr.c`
- Re-checked current tests and public references to:
  - `sparse_create_from_csr(...)`
  - `sparse_create_from_csc(...)`
  - `sparse_from_csr(...)`
  - `sparse_from_csc(...)`
  - `sparse_to_csr(...)`
  - `sparse_to_csc(...)`
- Recorded the Day 4 API design in
  `artifacts/day4-compressed-first-api-design.md`.

### Selected Design

- Treat `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)` as the
  simple compressed-first public front door.
- Treat `sparse_from_csr(...)` and `sparse_from_csc(...)` as the diagnostic
  compressed-first constructors for callers that need explicit
  `sparse_err_t` results.
- Preserve copy/build ownership: caller-owned CSR/CSC arrays are read during
  construction, and a successful constructor returns a new caller-owned
  `SparseMatrix`.
- Preserve the mutable `SparseMatrix` shell as the compatibility object and
  normal solver coefficient object.
- Prove compressed-input workflows through focused constructor diagnostics,
  ownership tests, and bounded solver smoke tests before updating the longer
  docs/example narrative.

### Deferred or Rejected Scope

- No broad direct CSR/CSC solver entry family in Sprint 101.
- No replacement of `SparseMatrix` as the public solver coefficient object.
- No renaming of existing constructors.
- No adopt/no-copy CSR/CSC constructor until a separate ownership and lifetime
  design exists.
- No Matrix Market compressed-object publication work.
- No internal CSR/CSC build-path optimization unless Day 5 can bound it as a
  very small, low-risk patch with clear validation.

### Validation Expectations

- Day 4 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, package, public
  docs, examples, or test files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only design pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Day 4 Exit State

Sprint 101 now has a bounded compressed-first API design. Day 5 can translate
that design into a file-level implementation plan, focused validation matrix,
and quality gate before any `.c`, `.h`, example, or test files are changed.

## Day 5 - Implementation Boundary Freeze

### Goal

Freeze the first implementation batch before code, header, test, example, or
public documentation edits begin. Day 5 converts the Day 4 design into a
file-level owner map, focused validation plan, compatibility/rollback notes,
and explicit quality gates.

### Actions

- Re-read Sprint 101 Day 5 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read `artifacts/day4-compressed-first-api-design.md`.
- Inspected the existing CSR/CSC constructor declarations in
  `include/sparse_csr.h`.
- Inspected the existing CSR/CSC constructor implementations in
  `src/sparse_csr.c`.
- Inspected the existing CSR/CSC tests in `tests/test_csr.c`.
- Checked current focused test registration for `test_csr` in `CMakeLists.txt`
  and the Makefile build path.
- Recorded the Day 5 implementation boundary freeze in
  `artifacts/day5-implementation-boundary-freeze.md`.

### Frozen Scope

- Day 6 may update `include/sparse_csr.h` comments to clarify simple versus
  diagnostic compressed constructor roles.
- Day 6 may update `tests/test_csr.c` with focused bad-input, output-nulling,
  ownership-independence, and bounded solver-smoke coverage.
- Day 6 should record implementation evidence in a Day 6 artifact.
- Day 6 should not add broad direct CSR/CSC solver APIs, adopt/no-copy
  constructors, Matrix Market compressed-object publication, build-system
  changes, benchmark changes, or solver-family parity claims.

### Validation Expectations

- Day 5 changed planning documentation only.
- No `.c`, `.h`, build-system, workflow, benchmark, script, package, public
  docs, examples, or test files were modified.
- Full `make format && make lint && make test` is not required for this
  docs-only boundary pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`
- Day 6 must run `make format && make lint && make test` if it modifies any
  `.c` or `.h` file.

### Day 5 Exit State

Sprint 101 now has a frozen implementation boundary for Batch 1. Day 6 can
implement the selected constructor/import refinements with clear file owners,
focused checks, rollback notes, and the full required quality gate for any C
or header changes.

## Day 6 - Constructor and Import Batch 1

### Goal

Implement the first bounded compressed-first constructor/import refinement
from the Day 5 boundary. Day 6 preserves existing ABI and mutable-shell
compatibility while strengthening the public CSR/CSC constructor contract and
focused regression coverage.

### Actions

- Updated `include/sparse_csr.h` comments to clarify:
  - `sparse_create_from_csr(...)` and `sparse_create_from_csc(...)` as simple
    compressed-first constructors;
  - `sparse_from_csr(...)` and `sparse_from_csc(...)` as diagnostic
    compressed-first constructors;
  - CSR/CSC arrays are validated and copied into an independent caller-owned
    `SparseMatrix`, not adopted or modified.
- Updated `tests/test_csr.c` with focused coverage for:
  - CSR and CSC bad input diagnostics;
  - strict pointer-array and index validation;
  - duplicate and unsorted entry rejection;
  - simple constructor `NULL` behavior for representative invalid structures;
  - caller-owned array copy independence after successful construction;
  - a bounded CSR-built matrix entering one-shot LU factor/solve.
- Recorded implementation evidence in
  `artifacts/day6-constructor-import-batch1.md`.

### Validation

- `make format` passed.
- Focused validation passed:
  - `make build/test_csr`
  - `./build/test_csr`
  - result: 18 tests, 0 failures, 580 assertions.
- First `make lint` run failed on two cppcheck `intToPointerCast` warnings
  from non-portable test sentinel pointer casts.
- Replaced the sentinel casts with a portable null-output assertion pattern.
- Required full gate then passed:
  - `make format`
  - `make lint`
  - `make test`

### Findings

- The existing implementation already enforced the strict CSR/CSC validation
  rules selected on Day 4; the batch mainly made those rules explicit in
  focused tests.
- The constructors copy caller-owned arrays into an independent
  `SparseMatrix`, which supports the Day 4 ownership contract without adding
  adopt/no-copy semantics.
- A CSR-built matrix can enter the ordinary one-shot LU path through the
  public matrix shell, which supports the front-door workflow claim without
  implying direct CSR solver parity.

### Day 6 Exit State

Sprint 101 now has its first implemented compressed-first constructor/import
batch. Day 7 can audit the landed header and test changes against the Day 4
design and Day 5 boundary, then decide whether a second implementation batch
is justified.

## Day 7 - Post-Batch Audit and Rerank

### Goal

Audit the Day 6 constructor/import batch before any new edits. Day 7 checks
boundary compliance, ownership and mutation drift, validation status,
remaining candidate priority, and the second-batch decision.

### Actions

- Re-read Sprint 101 Day 7 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read:
  - `artifacts/day4-compressed-first-api-design.md`
  - `artifacts/day5-implementation-boundary-freeze.md`
  - `artifacts/day6-constructor-import-batch1.md`
- Inspected the landed implementation changes in:
  - `include/sparse_csr.h`
  - `tests/test_csr.c`
- Compared the landed changes against the Day 4 selected API refinements and
  Day 5 frozen file-level implementation plan.
- Recorded the post-batch audit and rerank in
  `artifacts/day7-post-batch-audit-and-rerank.md`.

### Findings

- Day 6 stayed within the frozen implementation boundary: only
  `include/sparse_csr.h` and `tests/test_csr.c` changed for implementation.
- Header comments now clearly distinguish simple compressed constructors from
  diagnostic compressed constructors without changing ABI.
- Focused tests now cover CSR/CSC invalid structures, duplicate/unsorted
  structural entries, copy ownership, and one bounded one-shot LU workflow
  proof from a CSR-built matrix.
- No ownership drift was introduced: caller-owned CSR/CSC arrays are still
  copied, not adopted, and returned matrices remain caller-owned
  `SparseMatrix` shells.
- No immediate second constructor/import implementation batch is justified.
  The next highest-value work is Day 8 lifecycle and ownership design.

### Validation Expectations

- Day 7 changed planning documentation only.
- Day 6's code/header/test changes have already passed:
  - `make format`
  - `make lint`
  - `make test`
- Full `make format && make lint && make test` is not required for this
  docs-only audit pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Day 7 Exit State

Sprint 101 has reconciled the first implementation batch. Remaining work is
reranked toward lifecycle/ownership design, public docs/examples, and later
regression proof rather than another immediate constructor/import patch.

## Day 8 - Lifecycle and Ownership Design

### Goal

Clarify ownership, lifetime, mutation, and repeated-run rules for
compressed-first callers after the Day 6 constructor/import batch. Day 8 is a
design pass and does not change APIs, code, tests, or public docs.

### Actions

- Re-read Sprint 101 Day 8 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read `artifacts/day7-post-batch-audit-and-rerank.md`.
- Inspected lifecycle and ownership wording in:
  - `include/sparse_csr.h`
  - `include/sparse_analysis.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
  - `README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
- Recorded the lifecycle and ownership design in
  `artifacts/day8-lifecycle-and-ownership-design.md`.

### Findings

- The compressed constructor ownership contract is now clear in
  `include/sparse_csr.h`: caller-owned CSR/CSC arrays are validated and copied,
  not adopted, and the returned matrix is an independent caller-owned
  `SparseMatrix`.
- Existing direct repeated-run ownership is clear in `include/sparse_analysis.h`:
  `sparse_analysis_t` owns symbolic/permutation setup and `sparse_factors_t`
  owns numeric factor state.
- Iterative and eigensolver handle ownership is already clear: handles preserve
  allocation capacity only and do not preserve numerical iteration state.
- README still describes `sparse_from_csr/csc` as retained compatibility
  wrappers. After Day 6, those should be described as diagnostic compressed
  constructors instead.
- The next likely Day 9 work is a narrow lifecycle wording batch, not a broad
  code batch.

### Validation Expectations

- Day 8 changed planning documentation only.
- Day 6's code/header/test changes have already passed:
  - `make format`
  - `make lint`
  - `make test`
- Full `make format && make lint && make test` is not required for this
  docs-only design pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Day 8 Exit State

Sprint 101 now has a lifecycle and ownership design for compressed-first
callers. Day 9 can land a focused follow-through batch, most likely public
wording that aligns README/tutorial lifecycle language with the Day 6 header
contract and Day 8 ownership map.

## Day 9 - Lifecycle and Ownership Batch

### Goal

Land the focused lifecycle, ownership, and error-handling follow-through from
the Day 8 design. Day 9 checked for behavior gaps first, then selected a
documentation batch because Day 6 already covered constructor invalid-input,
copy-ownership, and solver-entry behavior.

### Actions

- Re-read Sprint 101 Day 9 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read the Day 8 lifecycle and ownership design artifact.
- Inspected compressed-first and lifecycle wording in:
  - `README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
- Updated `README.md` to:
  - state that CSR/CSC constructor input arrays are copied, not adopted;
  - state that callers keep ownership of compressed arrays;
  - describe `sparse_from_csr(...)` and `sparse_from_csc(...)` as diagnostic
    compressed-first constructors rather than compatibility wrappers;
  - state that successful diagnostic construction returns a caller-owned
    `SparseMatrix`.
- Updated `docs/tutorial.md` to:
  - mention compressed-first construction in the workflow-selection list;
  - include `sparse_csr.h` in the relevant public header list;
  - add a compact compressed-input ownership paragraph near matrix creation.
- Recorded the Day 9 lifecycle batch evidence in
  `artifacts/day9-lifecycle-ownership-batch.md`.

### Findings

- No new C behavior gap was identified for Day 9.
- Day 6 already provides the focused ownership and bad-input tests requested
  by the Day 9 plan.
- The remaining Day 9 gap was public wording: README still called diagnostic
  constructors retained compatibility wrappers, and the tutorial's first
  creation section still began only from insertion-based construction.
- The docs can now describe compressed-input ownership without claiming direct
  CSR/CSC solver APIs or replacing the mutable matrix shell.

### Validation Expectations

- Day 9 changed public documentation and planning documentation only.
- No additional `.c` or `.h` files were modified on Day 9.
- Full `make format && make lint && make test` is not required for this
  docs-only lifecycle batch.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `README.md`, `docs/tutorial.md`, and
    `docs/planning/EPIC_10/SPRINT_101`

### Day 9 Exit State

Sprint 101 now has implementation evidence, header comments, README wording,
and tutorial guidance aligned around one ownership model: CSR/CSC input arrays
remain caller-owned, constructors validate and copy into a caller-owned
`SparseMatrix`, and `sparse_from_csr/csc` provide explicit diagnostic status.
Day 10 can design the broader compatibility-path documentation pass without
reopening constructor semantics.

## Day 10 - Compatibility Path Documentation Design

### Goal

Design public wording that presents mutable matrix-shell workflows as
supported compatibility rather than the only product center. Day 10 is a
design pass for Day 11 docs/examples follow-through; it does not change APIs,
code, tests, examples, or public docs.

### Actions

- Re-read Sprint 101 Day 10 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read `artifacts/day9-lifecycle-ownership-batch.md`.
- Audited compressed-first and compatibility-shell wording in:
  - `README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
  - `include/sparse_csr.h`
  - `include/sparse_matrix.h`
- Checked shipped examples for an existing compressed-input construction
  example.
- Recorded the Day 10 compatibility documentation design in
  `artifacts/day10-compatibility-documentation-design.md`.

### Findings

- `include/sparse_csr.h`, `include/sparse_matrix.h`, README workflow
  guidance, README API reference, and the tutorial workflow chooser are now
  aligned with the compressed-first ownership model.
- README Quick Start still opens with insertion-based construction. That is
  acceptable for a tiny hand-written matrix, but Day 11 should add a short
  route for callers whose input already exists as CSR/CSC arrays.
- The tutorial creation section now includes compressed-input wording, but the
  section framing still reads insertion-first. Day 11 should make insertion
  one construction option rather than the conceptual center.
- `examples/README.md` has no compressed-input route, and there is no dedicated
  compressed-input example program.
- A small `example_compressed_input.c` would be useful only if Day 11 can add
  normal build registration and run the required C quality gate.

### Validation Expectations

- Day 10 changed planning documentation only.
- No additional `.c` or `.h` files were modified on Day 10.
- Full `make format && make lint && make test` is not required for this
  docs-only design pass.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Day 10 Exit State

Sprint 101 now has a scoped Day 11 docs/example edit plan. The next day should
prioritize `examples/README.md`, README Quick Start framing, and tutorial
creation-section framing. If an executable compressed-input example is added,
Day 11 must run the full C quality gate because a `.c` file would change.

## Day 11 - Docs and Examples Follow-Through

### Goal

Update public docs and examples to reflect the earned compressed-first product
model while preserving mutable matrix-shell workflows as supported
compatibility paths.

### Actions

- Re-read Sprint 101 Day 11 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read the Day 10 compatibility documentation design artifact.
- Inspected Make and CMake example registration:
  - `Makefile` picks up `examples/*.c` through `EX_SRCS`.
  - `CMakeLists.txt` registers each example explicitly.
- Added `examples/example_compressed_input.c`.
- Registered `example_compressed_input` in `CMakeLists.txt`.
- Updated `README.md` Quick Start framing to tell CSR/CSC callers to skip
  incremental insertion and start from compressed constructors.
- Updated `docs/tutorial.md` so the first matrix section is framed as choosing
  a construction path, with insertion as the small hand-written matrix path.
- Updated `examples/README.md` with a compressed-input route and program
  description.
- Recorded the Day 11 docs/example follow-through evidence in
  `artifacts/day11-docs-and-examples-follow-through.md`.

### Findings

- Adding the example is justified because Day 10 found no shipped executable
  compressed-input teaching reference.
- Make example registration required no manual source-list edit because the
  Makefile uses a wildcard.
- CMake required explicit registration for the new example.
- The example demonstrates ownership and workflow entry only. It does not
  claim direct CSR/CSC solver APIs or no-copy construction.

### Validation Expectations

- Day 11 added a `.c` example and touched `CMakeLists.txt`.
- Full `make format && make lint && make test` is required.
- Focused example validation should include `make examples`.
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `README.md`, `docs/tutorial.md`, `examples`,
    and `docs/planning/EPIC_10/SPRINT_101`

### Validation Results

- `make format`: passed.
- `make examples`: passed; built 13 example binaries including
  `example_compressed_input`.
- `make lint`: passed.
- `make test`: passed.
- `cmake -S . -B build/cmake-sprint101-day11 && cmake --build
  build/cmake-sprint101-day11 --target example_compressed_input`: passed.
- `./build/example_compressed_input`: passed; it printed unchanged
  `A(0,0)` after caller-owned CSR mutation, the expected all-ones solution,
  and zero residual.
- `git diff --check`: passed.
- Trailing-whitespace scan on `README.md`, `docs/tutorial.md`, `examples`,
  and `docs/planning/EPIC_10/SPRINT_101`: passed.

### Day 11 Exit State

Sprint 101 now has public README/tutorial framing, an examples map route, and
an executable compressed-input example aligned with the Day 6 constructor
behavior and Day 8 ownership model. Day 12 can review the accumulated tests
and decide whether any additional regression proof is needed beyond the Day 6
constructor tests and the Day 11 example build.

## Day 12 - Regression Proof Expansion

### Goal

Complete focused regression proof for compressed-first construction,
ownership, error handling, and solver entry behavior without broadening Sprint
101 into direct CSR/CSC solver APIs or full solver-family parity.

### Actions

- Re-read Sprint 101 Day 12 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read the Day 11 docs/examples follow-through artifact.
- Reviewed existing Day 6 coverage in `tests/test_csr.c`:
  - CSR/CSC simple constructor success;
  - CSR/CSC diagnostic invalid-input handling;
  - CSR/CSC copy ownership;
  - CSR-built matrix entering one-shot LU.
- Identified the remaining narrow solver-entry gap: no CSC-built matrix
  entered a representative solver path in the focused CSR/CSC test suite.
- Added `test_csc_constructed_matrix_enters_cholesky_solve` to
  `tests/test_csr.c`.
- Confirmed no Make/CMake test registration change is needed because the new
  test lives in the existing `test_csr` binary.
- Recorded Day 12 regression proof evidence in
  `artifacts/day12-regression-proof-expansion.md`.

### Findings

- Day 6 already covers invalid input and copy ownership for both CSR and CSC.
- Day 11 adds an executable public workflow example, but examples are adoption
  proof rather than the primary regression suite.
- Adding a CSC-to-Cholesky smoke test balances the existing CSR-to-LU smoke
  test without implying broad compressed solver parity.
- Because no new test executable was added, reviewed CTest test counts should
  not change.

### Validation Expectations

- Day 12 modified `tests/test_csr.c`.
- Full `make format && make lint && make test` is required.
- Focused validation should include:
  - `make build/test_csr`
  - `./build/test_csr`
- Documentation hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101` and
    `tests/test_csr.c`

### Validation Results

- `make format`: passed.
- `make build/test_csr`: passed.
- `./build/test_csr`: passed; `test_csr` ran 19 tests, 0 failures, and 585
  assertions.
- `make lint`: passed.
- `make test`: passed.

### Day 12 Exit State

Sprint 101 now has focused regression proof for CSR and CSC construction,
diagnostics, copy ownership, and representative solver entry. Remaining gaps
are explicit non-goals or deferred performance/product topics rather than
missing Sprint 101 regression coverage.

## Day 13 - Full Validation and Product-Model Reconciliation

### Goal

Run final required validation before closeout and reconcile Sprint 101 public
wording, implementation, tests, and examples against Sprint 100 claim
boundaries.

### Actions

- Re-read Sprint 101 Day 13 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Re-read the Day 12 regression proof artifact.
- Re-read Sprint 100 claim-boundary inputs:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day12-public-claim-audit.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-sprint100-handoff-package.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-claim-non-goal-register.md`
- Reconciled Sprint 101 public wording across:
  - `include/sparse_csr.h`
  - `include/sparse_matrix.h`
  - `README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
  - `examples/example_compressed_input.c`
- Recorded earned, deferred, and non-claim states in
  `artifacts/day13-validation-and-reconciliation.md`.

### Findings

- Sprint 101 has enough evidence to earn the bounded compressed-first product
  model claim: constructor contracts, ownership tests, invalid-input tests,
  solver-entry smoke tests, public docs, and an executable example.
- Sprint 101 does not earn direct CSR/CSC solver APIs, no-copy/adopt
  constructors, replacement of `SparseMatrix`, broad solver-family parity, or
  portable performance superiority.
- No Make/CMake test registration drift is expected from Day 12 because the
  new regression lives inside existing `test_csr`.
- Sprint 102 can rely on compressed input entering the public matrix shell,
  but must still own broader direct-solver oracle work.

### Validation Expectations

- This branch contains `.c` and `.h` changes, so Day 13 must rerun:
  - `make format`
  - `make lint`
  - `make test`
- Focused surface checks should also include:
  - `make examples`
  - CMake configure/build for `example_compressed_input` and `test_csr`
  - `./build/example_compressed_input`
  - `git diff --check`
  - trailing-whitespace scan across touched docs/examples/planning/test/header
    surfaces

### Validation Results

- `make format`: passed.
- `make examples`: passed; all 13 example binaries built.
- `make lint`: passed.
- `make test`: passed.
- `cmake -S . -B build/cmake-sprint101-day13`: passed.
- `cmake --build build/cmake-sprint101-day13 --target example_compressed_input test_csr`:
  passed.
- `./build/example_compressed_input`: passed; the example reported retained
  copied CSR values, the all-ones solution, and zero residual.
- `git diff --check`: passed.
- Trailing-whitespace scan across touched docs, examples, planning artifacts,
  header, and test file: passed.

### Day 13 Exit State

Sprint 101 is reconciled and validated for the bounded compressed-first
product model. Day 14 can focus on closeout, artifact indexing, and ensuring
deferred claims are handed forward without expanding Sprint 101's public
claim.

## Day 14 - Sprint 101 Closeout and Handoff

### Goal

Close Sprint 101 with complete artifacts, an explicit Sprint 102 handoff, and
retrospective inputs that preserve the bounded compressed-first product-model
claim.

### Actions

- Re-read Sprint 101 Day 14 in
  `docs/planning/EPIC_10/SPRINT_101/PLAN.md`.
- Reconciled every Sprint 101 project-plan item against a deliverable.
- Created `artifacts/day14-closeout-and-handoff.md`.
- Created `artifacts/day14-artifact-index.md`.
- Recorded Sprint 102 direct-solver oracle prerequisites and residual queue.
- Captured retrospective inputs and final validation posture.

### Findings

- Every Sprint 101 project-plan item has a corresponding artifact, code
  change, documentation update, example, or validation record.
- Sprint 101 earned the bounded compressed-first front-door claim but did not
  earn direct CSR/CSC solver APIs, no-copy/adopt constructors, broad solver
  parity, performance superiority, or broad state-of-the-art replacement
  claims.
- Sprint 102 can start from stable compressed-input ownership and lifecycle
  rules, but must own external oracle fixtures, tolerances, and direct-solver
  evidence.

### Validation Expectations

- Day 14 changed planning documentation only.
- The required full C quality chain for Sprint 101's `.c` and `.h` changes
  was already rerun and recorded on Day 13:
  - `make format`
  - `make examples`
  - `make lint`
  - `make test`
  - focused CMake configure/build
  - `./build/example_compressed_input`
- Day 14 hygiene should include:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_101`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_101`: passed; no matches.

### Day 14 Exit State

Sprint 101 is closed from a complete artifact set. The sprint leaves a stable
compressed-first ownership and lifecycle baseline for Sprint 102, with
direct-solver oracle work and broader solver evidence explicitly deferred.
