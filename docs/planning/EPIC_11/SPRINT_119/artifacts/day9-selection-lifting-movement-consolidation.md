# Sprint 119 Day 9 Selection and Lifting Movement Consolidation

## Purpose

Day 9 was planned as the implementation or deferral day for
`s20_select_indices` and `s20_lift_ritz_vectors` after the Day 8 proof audit.
Because both helpers already moved safely as the Day 6 paired movement and
passed Day 7 focused validation, Day 9 performs evidence consolidation rather
than duplicate code movement.

## Day 8 Decision Applied

| Question | Day 9 result |
|---|---|
| Move `s20_select_indices`? | Already complete. The body lives in `src/sparse_eigs_selection_internal.c`. |
| Move `s20_lift_ritz_vectors`? | Already complete. The body lives in `src/sparse_eigs_selection_internal.c`. |
| Defer either helper? | No. Day 8 found no current defer condition for either helper. |
| Split the helpers into separate owners? | No. They remain paired to preserve a single selection/vector-publication proof and rollback boundary. |
| Make additional C source changes on Day 9? | No. No corrective issue was found. |

## Implementation State Verification

| Surface | Verification |
|---|---|
| Private implementation owner | `src/sparse_eigs_selection_internal.c` contains both `s20_select_indices` and `s20_lift_ritz_vectors`. |
| Private declarations | `src/sparse_eigs_internal.h` still declares both helpers. |
| Grow-m consumers | `src/sparse_eigs.c` still calls both helpers through the private declarations. |
| Thick-restart consumers | `src/sparse_eigs_thick_restart.c` still calls both helpers through the private declarations. |
| LOBPCG consumer | `src/sparse_eigs_lobpcg.c` still calls `s20_select_indices`; it does not consume `s20_lift_ritz_vectors`. |
| Makefile membership | `Makefile` includes `$(SRCDIR)/sparse_eigs_selection_internal.c`. |
| CMake membership | `CMakeLists.txt` includes `src/sparse_eigs_selection_internal.c`. |
| Source-list metadata | `build-metadata/library_sources.txt` includes `src/sparse_eigs_selection_internal.c`. |

## Focused Compile And Source-List Notes

No new Day 9 code, header, build, CMake, workflow, package, benchmark, test,
public documentation, or claim surfaces changed. Day 9 therefore did not rerun
the full C quality chain.

The current movement remains covered by the Day 7 validation package:

- focused eigensolver binaries built and linked;
- `./build/test_eigs` passed with 43 tests, 0 failed, 955 assertions;
- `./build/test_eigs_thick_restart` passed with 23 tests, 0 failed,
  384 assertions;
- `./build/test_eigs_lobpcg` passed with 29 tests, 0 failed, 287 assertions;
- `make source-list-check` passed with 49 library sources;
- CMake build and `ctest -N` passed with `Total Tests: 54`;
- `make format && make lint && make test` passed.

## Residual Helper Movement List

| Residual | Day 9 status |
|---|---|
| `s20_select_indices` | No residual movement; already moved and validated. |
| `s20_lift_ritz_vectors` | No residual movement; already moved and validated. |
| Selection/lifting validation refresh | Still scheduled for Day 10 as an evidence refresh and final confirmation before shift-invert work. |
| `lanczos_iterate_op` | Still deferred; requires recurrence-specific proof before any movement. |
| Shift-invert setup/conversion | Still deferred to the Day 11 boundary decision. |

## Updated Source-Movement Evidence

Day 9 closes Sprint 119 Item 4 implementation for the selection/lifting helper
pair:

- movement was performed only after Day 4-5 design and proof setup;
- Day 6 made the paired private source movement;
- Day 7 validated focused consumers, source-list metadata, CMake/CTest
  membership, and full quality gates;
- Day 8 audited the dependency and invariant proof;
- Day 9 confirms no duplicate move, split, or deferral is needed.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Selection/lifting movement or deferral decision exists. | Complete: moved together. |
| Code and build metadata changes exist where cleared. | Complete from Day 6. |
| Focused compile/source-list notes exist. | Complete; Day 7 validation remains authoritative. |
| Residual helper movement list exists. | Complete. |
| Updated source-movement evidence exists. | Complete. |
| Item 4 implementation or deferral is complete. | Complete. |
| Movement did not proceed without proof. | Complete; proof setup and validation are recorded in Days 4-8. |
| Explicit residuals exist for any deferred helper. | Complete; neither selection/lifting helper is deferred. |
