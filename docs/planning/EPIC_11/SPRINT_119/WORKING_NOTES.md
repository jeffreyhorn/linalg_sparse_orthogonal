# Sprint 119 Working Notes

## Sprint Goal

Sprint 119 converts the safest Epic 10 eigensolver residual movements into
validated source-boundary improvements without widening public eigensolver
claims.

## Starting Constraints

- Treat Sprint 118 as the current baseline for product truth, explicit
  non-claims, source/test hotspot rankings, and evidence-template usage.
- Do not move eigensolver code before feasibility, focused consumer proof,
  source-list/CMake impact, expected CTest count, and rollback expectations
  are documented.
- Preserve public eigensolver support as symmetric eigensolver workflows with
  bounded evidence. Do not claim ARPACK, SciPy, LAPACK, broad nonsymmetric
  eigensolver, or state-of-the-art parity.
- Treat implementation days as code-touching unless proven otherwise. If `.c`
  or `.h` files change, run `make format && make lint && make test`.
- If Makefile, CMake, workflow, package, benchmark, script, or install
  surfaces change, run the relevant focused validation lane and record whether
  it is reviewed, supplemental, or local.
- If documentation only changes, run `git diff --check` and a focused
  trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_119`.

## Input Artifact Inventory

| Input | Sprint 119 use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 119 section | Authoritative project-plan items, estimates, deliverables, and sprint goal. |
| `docs/planning/EPIC_11/SPRINT_119/PLAN.md` | Day-by-day execution plan and completion criteria. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day6-residual-owner-map.md` | Eigensolver residual owners, dependency order, and proof gates. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day8-product-truth-map.md` | Current eigensolver truth, candidate claims, and explicit non-claims. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day10-hotspot-owner-handoff.md` | Eigensolver source/test hotspot handoff and movement prerequisites. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day12-evidence-template-refresh.md` | Refreshed evidence-template index and future-sprint usage rules. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day14-sprint-closeout-handoff.md` | Sprint 119 handoff requirements and residual deferred debt. |
| `docs/planning/EPIC_11/SPRINT_118/templates/source-movement-evidence-template.md` | Required source-movement, proof, validation, drift, non-claim, and handoff fields. |

## Day-Level Ownership

| Day | Planned Focus | Project Plan Item |
|---:|---|---|
| 1 | Sprint intake, artifact skeleton, input inventory, validation boundaries, and owner map. | Items 1-7 intake |
| 2 | Eigensolver movement candidate inventory and consumer map. | Item 1 |
| 3 | Movement feasibility ranking and first movement recommendation. | Item 1 |
| 4 | Source boundary design for the first movement batch. | Item 2 |
| 5 | Focused consumer proof design and source-movement evidence draft. | Items 2, 3 |
| 6 | First movement batch implementation. | Item 3 |
| 7 | First movement batch focused validation. | Items 3, 6 |
| 8 | Selection/lifting proof audit. | Item 4 |
| 9 | Selection/lifting movement or explicit deferral. | Item 4 |
| 10 | Selection/lifting validation. | Items 4, 6 |
| 11 | Shift-invert boundary decision. | Item 5 |
| 12 | Shift-invert movement or deferral validation. | Items 5, 6 |
| 13 | Full validation and parity package. | Item 6 |
| 14 | Closeout, residuals, non-claims, and Sprint 120 handoff. | Item 7 |

## Validation Expectations

| Touched Surface | Required Checks |
|---|---|
| Documentation-only planning artifacts | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_119`. |
| `.c` or `.h` source/header changes | `make format && make lint && make test`. |
| Source-list or Makefile membership | `make source-list-check` or the relevant reviewed wrapper. |
| CMake membership | CMake configure/build and `ctest -N` count proof as affected. |
| Eigensolver behavior | Focused eigensolver tests for grow-m, thick restart, LOBPCG, shift-invert, and repeated-handle consumers touched by the movement. |
| Public claim or support wording | Check against Sprint 118 Day 8 product truth and explicit non-claims. |

## Source-Movement Evidence Fields Required

Every movement or explicit deferral artifact should record:

- scope and touched surfaces;
- baseline owner metrics and current product truth references;
- behavior boundary;
- old/new file plan;
- internal header and private API contract;
- source-list, Makefile, and CMake impact;
- focused consumer proof;
- CTest membership and expected count;
- validation commands and results;
- rollback or defer plan;
- non-claims preserved;
- residual handoff.

## Scope Boundaries

Sprint 119 may inspect, rank, design, move, validate, or explicitly defer the
eigensolver residual candidates named in the project plan. It should not:

- broaden public eigensolver claims;
- add new solver-family oracle architecture reserved for Sprint 120-122;
- alter package, ABI, platform, benchmark, adoption, or public API surfaces
  unless required by the bounded movement and explicitly validated;
- perform broad source cleanup outside eigensolver movement candidates;
- hide movement deferrals without residual owners.

## Day 1 Notes

- Created the Sprint 119 working-notes baseline and artifact directory.
- Re-read the Sprint 119 project-plan section and Sprint 119 day-by-day plan.
- Re-read Sprint 118 closeout handoff and identified the required Sprint 118
  inputs for Sprint 119:
  - residual owner map;
  - product truth map and non-claims;
  - hotspot owner handoff;
  - evidence template refresh;
  - closeout handoff package;
  - source-movement evidence template.
- Mapped all Sprint 119 project-plan items to day-level owners.
- Recorded validation expectations for documentation-only, C/header,
  source-list, CMake, focused eigensolver, and public-claim touched surfaces.
- Recorded required source-movement evidence fields before any code movement
  begins.
- Added Day 1 sprint intake artifact:
  `artifacts/day1-sprint-intake.md`.
- Kept Day 1 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 2 Notes

- Inspected Sprint 119 Item 1 requirements and the Sprint 118 residual owner,
  hotspot handoff, and closeout artifacts.
- Inspected current eigensolver source boundaries around `src/sparse_eigs.c`,
  `src/sparse_eigs_internal.h`, `src/sparse_eigs_thick_restart.c`,
  `src/sparse_eigs_lobpcg.c`, and eigensolver workspace owners.
- Inventoried the named movement candidates:
  - `s20_select_indices`;
  - `s20_lift_ritz_vectors`;
  - shift-invert setup/conversion;
  - `lanczos_iterate_op`;
  - the broader eigensolver private-owner movement bucket.
- Mapped grow-m, thick-restart, LOBPCG, shift-invert, repeated-handle, and
  focused-test consumers for each candidate.
- Recorded Makefile, CMake, public-header, and internal-header touch points.
- Added Day 2 artifact:
  `artifacts/day2-eigensolver-movement-candidate-inventory.md`.
- Deferred ranking until Day 3; Day 2 only records audit inputs.
- Kept Day 2 documentation-only; no C source, header, build, workflow,
  package, benchmark, or test surfaces were modified.

## Day 3 Notes

- Ranked Sprint 119 eigensolver movement candidates using the Day 2 consumer
  map, Sprint 118 proof gates, source/build touch points, and rollback costs.
- Identified the recommended first movement batch as the paired private helper
  extraction for `s20_select_indices` and `s20_lift_ritz_vectors`.
- Recorded why the paired selection/lift movement is lower risk than moving
  `lanczos_iterate_op`, shift-invert setup/conversion, or the broad
  eigensolver private-owner bucket.
- Marked `lanczos_iterate_op` as a medium-high-risk candidate that needs a
  recurrence-specific proof plan before movement.
- Deferred shift-invert setup/conversion to the Day 11 boundary decision
  unless later proof finds a smaller safe split.
- Added move/defer conditions, proof-risk notes, rollback-risk notes, and the
  Day 4 design gate.
- Added Day 3 artifact:
  `artifacts/day3-movement-feasibility-ranking.md`.
- Completed Sprint 119 Item 1. Kept Day 3 documentation-only; no C source,
  header, build, workflow, package, benchmark, or test surfaces were modified.

## Day 4 Notes

- Converted the Day 3 first movement recommendation into a concrete source
  boundary design for the paired `s20_select_indices` and
  `s20_lift_ritz_vectors` movement.
- Chose `src/sparse_eigs_selection_internal.c` as the proposed new private
  source owner for the first movement batch.
- Kept `src/sparse_eigs_internal.h` as the private declaration owner for this
  batch to avoid unnecessary header churn.
- Explicitly excluded `lanczos_iterate_op`, shift-invert setup/conversion,
  grow-m retry logic, thick-restart state management, LOBPCG RR logic,
  refinement, and workspace allocation from the first movement batch.
- Recorded Makefile and CMake source-list updates that implementation will
  require, with CTest membership expected to remain unchanged.
- Recorded public API, ABI/package, docs, benchmark, and public-claim impact
  as none.
- Recorded rollback and partial-move handling. The two helper functions should
  move together or both remain in `src/sparse_eigs.c`.
- Added Day 4 artifact:
  `artifacts/day4-source-boundary-design.md`.
- Completed the Day 4 portion of Sprint 119 Item 2. Kept Day 4
  documentation-only; no C source, header, build, workflow, package,
  benchmark, or test surfaces were modified.

## Day 5 Notes

- Defined the focused consumer proof package for the planned
  `s20_select_indices` and `s20_lift_ritz_vectors` movement.
- Mapped focused proof across grow-m, thick-restart, LOBPCG, shift-invert, and
  repeated-handle public surfaces.
- Recorded behavior invariants for largest/smallest/nearest-sigma selection,
  bounded take handling, column-major vector publication, shift-invert vector
  publication, LOBPCG selection adjacency, public-header stability, and CTest
  membership stability.
- Recorded expected reviewed POSIX CMake CTest count as `54`, inherited from
  the Sprint 118 reviewed baseline.
- Filled the source-movement evidence draft for the planned private-owner
  extraction.
- Recorded Day 6 implementation checklist and validation commands, including
  full `make format && make lint && make test` once `.c` files and build
  metadata are changed.
- Added Day 5 artifact:
  `artifacts/day5-focused-consumer-proof-design.md`.
- Completed the Day 5 portion of Sprint 119 Items 2 and 3. Kept Day 5
  documentation-only; no C source, header, build, workflow, package,
  benchmark, or test surfaces were modified.

## Day 6 Notes

- Implemented the first movement batch by moving `s20_select_indices` and
  `s20_lift_ritz_vectors` from `src/sparse_eigs.c` into the new private source
  owner `src/sparse_eigs_selection_internal.c`.
- Kept private declarations in `src/sparse_eigs_internal.h` unchanged and made
  no public header, public API, package/ABI, benchmark, documentation, or
  claim-surface changes.
- Updated build membership in `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt`.
- Built the focused eigensolver binaries with
  `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg`;
  the focused compile/link proof passed.
- Ran focused eigensolver behavior tests:
  - `./build/test_eigs`: pass, 43 tests, 0 failed, 955 assertions.
  - `./build/test_eigs_thick_restart`: pass, 23 tests, 0 failed,
    384 assertions.
  - `./build/test_eigs_lobpcg`: pass, 29 tests, 0 failed, 287 assertions.
- Ran `make source-list-check`; the first run caught the missing
  `build-metadata/library_sources.txt` entry for the new source. Added the
  manifest entry and reran the check successfully with 49 library sources.
- Ran CMake membership proof with
  `cmake -S . -B build-cmake-review && cmake --build build-cmake-review &&
  ctest --test-dir build-cmake-review -N`; the proof passed and reported
  `Total Tests: 54`.
- Removed the generated local `build-cmake-review` directory after the proof.
- Ran the required full C quality chain because `.c` and build metadata were
  modified: `make format && make lint && make test`; the chain passed and all
  tests passed.
- Added Day 6 artifact:
  `artifacts/day6-first-movement-implementation.md`.
- Deferred `lanczos_iterate_op` movement, shift-invert setup/conversion
  movement, and broader eigensolver private-owner cleanup to their documented
  later proof gates.

## Day 7 Notes

- Revalidated the Day 6 first movement batch against the focused consumer
  proof designed on Day 5.
- Confirmed focused eigensolver binaries were build-current with
  `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg`.
- Reran focused eigensolver tests:
  - `./build/test_eigs`: pass, 43 tests, 0 failed, 955 assertions.
  - `./build/test_eigs_thick_restart`: pass, 23 tests, 0 failed,
    384 assertions.
  - `./build/test_eigs_lobpcg`: pass, 29 tests, 0 failed, 287 assertions.
- Reran `make source-list-check`; it passed with 49 library sources.
- Reran CMake membership and CTest registration proof with
  `cmake -S . -B build-cmake-review && cmake --build build-cmake-review &&
  ctest --test-dir build-cmake-review -N`; CMake compiled the new
  `src/sparse_eigs_selection_internal.c` source and `ctest -N` reported
  `Total Tests: 54`.
- Removed the generated local `build-cmake-review` directory after the proof.
- Reran the required full C quality chain for the branch's `.c` movement:
  `make format && make lint && make test`; the chain passed and ended with
  `All tests passed.`
- Added Day 7 validation artifact:
  `artifacts/day7-first-movement-validation.md`.
- Recorded that Day 8 should treat selection/lifting as a post-movement proof
  audit, and Day 9 should become an explicit no-op or evidence consolidation
  unless Day 8 finds a corrective follow-up.

## Day 8 Notes

- Audited the moved `s20_select_indices` and `s20_lift_ritz_vectors` helper
  dependencies after the successful Day 6 movement and Day 7 validation.
- Confirmed both helper declarations remain private in
  `src/sparse_eigs_internal.h` and both implementation bodies now live in
  `src/sparse_eigs_selection_internal.c`.
- Mapped `s20_select_indices` consumers:
  - grow-m backend in `src/sparse_eigs.c`;
  - thick-restart backend in `src/sparse_eigs_thick_restart.c`;
  - LOBPCG Rayleigh-Ritz step in `src/sparse_eigs_lobpcg.c`;
  - direct selector contract tests in `tests/test_ldlt_backend_dispatch.c`.
- Mapped `s20_lift_ritz_vectors` consumers:
  - grow-m vector and partial-result publication in `src/sparse_eigs.c`;
  - thick-restart locked-block and result-vector publication in
    `src/sparse_eigs_thick_restart.c`.
- Confirmed LOBPCG is a direct selection consumer but not a direct lifting
  consumer because it owns its vector-publication path.
- Documented public-result invariants for largest/smallest ordering,
  nearest-sigma transformed ordering, bounded partial publication,
  column-major vector layout, selected projected columns, and public claim
  stability.
- Recorded the move-together decision: the helpers should remain paired in
  `src/sparse_eigs_selection_internal.c`; no separate movement or deferral is
  needed after Day 7 validation.
- Recorded Day 9 as a no-op/evidence consolidation by default, unless a
  corrective follow-up is found before then.
- Added Day 8 artifact:
  `artifacts/day8-selection-lifting-proof-audit.md`.
- Kept Day 8 documentation-only; no additional C source, header, build,
  workflow, package, benchmark, test, public docs, or claim surfaces were
  modified.

## Day 9 Notes

- Applied the Day 8 move-together decision as an evidence-consolidation day
  rather than duplicate code movement.
- Verified `src/sparse_eigs_selection_internal.c` contains both moved helper
  bodies:
  - `s20_select_indices`;
  - `s20_lift_ritz_vectors`.
- Verified `src/sparse_eigs_internal.h` remains the private declaration owner.
- Verified grow-m, thick-restart, and LOBPCG call sites still consume the
  helpers through private declarations:
  - `src/sparse_eigs.c`;
  - `src/sparse_eigs_thick_restart.c`;
  - `src/sparse_eigs_lobpcg.c`.
- Verified build membership still includes the new private source in
  `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt`.
- Recorded that neither helper has residual movement debt after Day 9:
  both moved together and Day 7 validation remains authoritative.
- Recorded Day 10 as the final selection/lifting validation refresh before
  shift-invert boundary work begins.
- Added Day 9 artifact:
  `artifacts/day9-selection-lifting-movement-consolidation.md`.
- Kept Day 9 documentation-only; no additional C source, header, build,
  workflow, package, benchmark, test, public docs, or claim surfaces were
  modified.

## Day 10 Notes

- Performed the final selection/lifting validation refresh before
  shift-invert boundary work begins.
- Confirmed focused eigensolver binaries were build-current with
  `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg`.
- Reran focused eigensolver tests:
  - `./build/test_eigs`: pass, 43 tests, 0 failed, 955 assertions.
  - `./build/test_eigs_thick_restart`: pass, 23 tests, 0 failed,
    384 assertions.
  - `./build/test_eigs_lobpcg`: pass, 29 tests, 0 failed, 287 assertions.
- Reran `make source-list-check`; it passed with 49 library sources.
- Reran CMake membership and CTest registration proof with
  `cmake -S . -B build-cmake-review && cmake --build build-cmake-review &&
  ctest --test-dir build-cmake-review -N`; CMake compiled
  `src/sparse_eigs_selection_internal.c` and `ctest -N` reported
  `Total Tests: 54`.
- Removed the generated local `build-cmake-review` directory before the full
  quality chain.
- Reran the required full C quality chain for the branch's `.c` movement:
  `make format && make lint && make test`; the chain passed and ended with
  `All tests passed.`
- Confirmed Sprint 119 Item 4 selection/lifting validation is complete with no
  remaining helper movement residuals.
- Added Day 10 artifact:
  `artifacts/day10-selection-lifting-validation.md`.

## Day 11 Notes

- Inspected shift-invert setup and conversion ownership in `src/sparse_eigs.c`
  and the private contracts in `src/sparse_eigs_internal.h`.
- Mapped current shift-invert ownership across:
  - `s20_op_shift_invert`;
  - shifted matrix construction in `s46_sparse_eigs_sym_impl`;
  - `sparse_ldlt_factor_opts` lifecycle and `sparse_ldlt_free` cleanup;
  - `result->used_csc_path_ldlt` telemetry;
  - operator callback selection;
  - transformed eigenvalue conversion in grow-m and thick-restart result
    publication;
  - one-shot, handle, and internal workspace entry paths.
- Reviewed focused shift-invert tests in `tests/test_eigs.c`,
  `tests/test_ldlt_backend_dispatch.c`, and LOBPCG nearest-sigma coverage.
- Decided not to split shift-invert setup/conversion in Sprint 119. The setup,
  factor lifetime, telemetry, transformed-value conversion, backend dispatch,
  and cleanup paths are too coupled to move safely without a dedicated private
  context/lifecycle design.
- Recorded Day 12 as an explicit deferral-validation day rather than a source
  movement day unless a small corrective issue is found.
- Added Day 11 artifact:
  `artifacts/day11-shift-invert-boundary-decision.md`.
- Kept Day 11 documentation-only; no C source, header, build, workflow,
  package, benchmark, test, public docs, or claim surfaces were modified.

## Day 12 Notes

- Validated the Day 11 explicit deferral of shift-invert setup/conversion
  source movement.
- Reviewed `git diff --name-only`; the current tracked code/build changes are
  the earlier selection/lifting movement surfaces, not a Day 12 shift-invert
  split:
  - `CMakeLists.txt`;
  - `Makefile`;
  - `build-metadata/library_sources.txt`;
  - `src/sparse_eigs.c`.
- Reviewed the `src/sparse_eigs.c` diff for shift-invert-related terms. The
  only matches are from earlier removed selection/lift helper comments near
  `NEAREST_SIGMA`; `s20_op_shift_invert` and the
  `s46_sparse_eigs_sym_impl` setup/cleanup flow remain intact.
- Reran focused shift-invert and adjacent eigensolver tests:
  - `./build/test_eigs`: pass, 43 tests, 0 failed, 955 assertions.
  - `./build/test_eigs_lobpcg`: pass, 29 tests, 0 failed, 287 assertions.
- Reran `make source-list-check`; it passed with 49 library sources.
- Did not rerun CMake/CTest on Day 12 because no Day 12 build metadata changed;
  Day 10 CMake proof remains current for the branch's source movement and
  reported `Total Tests: 54`.
- Did not rerun full `make format && make lint && make test` on Day 12 because
  no Day 12 `.c` or `.h` changes were made; Day 10 full quality remains
  current for the branch's `.c` movement and ended with `All tests passed.`
- Added Day 12 artifact:
  `artifacts/day12-shift-invert-deferral-validation.md`.
- Kept Day 12 documentation-only; no additional C source, header, build,
  workflow, package, benchmark, test, public docs, or claim surfaces were
  modified.

## Day 13 Notes

- Packaged the Sprint 119 validation evidence for the branch's selection/lift
  helper movement and the explicit shift-invert setup/conversion deferral.
- Reran `make source-list-check`; it passed with 49 library sources.
- Confirmed focused eigensolver binaries were build-current with
  `make build/test_eigs build/test_eigs_thick_restart build/test_eigs_lobpcg`.
- Reran focused eigensolver tests:
  - `./build/test_eigs`: pass, 43 tests, 0 failed, 955 assertions.
  - `./build/test_eigs_thick_restart`: pass, 23 tests, 0 failed,
    384 assertions.
  - `./build/test_eigs_lobpcg`: pass, 29 tests, 0 failed, 287 assertions.
- Reran CMake membership and CTest registration proof with
  `cmake -S . -B build-cmake-review && cmake --build build-cmake-review &&
  ctest --test-dir build-cmake-review -N`; CMake compiled
  `src/sparse_eigs_selection_internal.c` and `ctest -N` reported
  `Total Tests: 54`.
- Removed the generated local `build-cmake-review` directory before the full
  quality chain.
- Reran the required full C quality chain for the branch's `.c` movement:
  `make format && make lint && make test`; the chain passed and ended with
  `All tests passed.`
- Recorded skipped supplemental lanes:
  - Windows CTest count remains CI-owned.
  - Benchmarks were not run because this is not a performance-claim change.
  - Package/install validation was not run because package/export surfaces did
    not change.
  - Public documentation claim validation was limited to diff review because no
    public docs or claims changed.
- Added Day 13 artifact:
  `artifacts/day13-validation-parity-package.md`.

## Day 14 Notes

- Closed Sprint 119 with a movement summary covering:
  - completed selection/lifting helper extraction into
    `src/sparse_eigs_selection_internal.c`;
  - build registration in `Makefile`, `CMakeLists.txt`, and
    `build-metadata/library_sources.txt`;
  - explicit shift-invert setup/conversion deferral;
  - validation lanes from Days 7, 10, 12, and 13.
- Recorded deferred residuals:
  - shift-invert setup/conversion private-owner extraction;
  - transformed eigenvalue conversion helper extraction;
  - `lanczos_iterate_op` movement after shift-invert lifecycle ownership is
    clearer.
- Recorded non-claim boundaries:
  - no broad eigensolver parity claim;
  - no ARPACK, SciPy, LAPACK, state-of-the-art, performance, or public API
    claim.
- Recorded Sprint 120 handoff guidance for direct/iterative oracle work:
  - keep proof-owner movement small and source-list aware;
  - preserve solver-specific tolerances and failure modes;
  - pair any new source owner with Make, CMake, and source-list registration;
  - continue documenting non-claims separately from validation evidence.
- Added the Sprint 119 artifact index and closeout artifact:
  `artifacts/day14-movement-closeout.md`.
- Kept Day 14 documentation-only; no additional C source, header, build,
  workflow, package, benchmark, test, public docs, or claim surfaces were
  modified.
