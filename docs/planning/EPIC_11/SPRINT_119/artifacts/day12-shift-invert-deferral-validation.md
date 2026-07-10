# Sprint 119 Day 12 Shift-Invert Deferral Validation

## Purpose

Day 12 validates the Day 11 decision to defer shift-invert setup/conversion
source movement. No shift-invert split was applied. This artifact records the
deferral proof, focused shift-invert validation, source-list evidence, and
residual owner for future private lifecycle extraction.

## Movement Or Deferral Decision

| Question | Day 12 result |
|---|---|
| Apply shift-invert split? | No. Day 11 explicitly deferred the split. |
| Move `s20_op_shift_invert`? | No. Moving only the thin solve wrapper would add churn without addressing lifecycle ownership. |
| Move shifted-matrix construction or LDLT setup? | No. This requires a dedicated private context and cleanup contract. |
| Move transformed eigenvalue conversion? | No. Conversion remains at backend result-publication sites for converged and partial paths. |
| Public claim or API change? | None. |

## Deferral Proof

| Check | Result |
|---|---|
| `git diff --name-only` review | Current tracked code/build changes are the earlier selection/lifting movement surfaces: `CMakeLists.txt`, `Makefile`, `build-metadata/library_sources.txt`, and `src/sparse_eigs.c`. |
| Shift-invert lifecycle split review | No new Day 12 source/build movement was made for shift-invert setup, LDLT lifecycle, telemetry, operator dispatch, or transformed-value conversion. |
| `src/sparse_eigs.c` diff review | The shift-invert-related diff text appears only because the earlier removed selection/lift helper comments referenced nearest-sigma and shift-invert; the `s20_op_shift_invert` body and `s46_sparse_eigs_sym_impl` setup/cleanup flow remain intact. |
| CMake/CTest count requirement | Not rerun on Day 12 because no Day 12 build metadata changed. Day 10 CMake proof remains current for the branch's source movement and reported `Total Tests: 54`. |
| Full C quality requirement | Not rerun on Day 12 because no Day 12 `.c` or `.h` changes were made. Day 10 full quality remains current for the branch's `.c` movement and ended with `All tests passed.` |

## Focused Shift-Invert Validation

| Command | Result |
|---|---|
| `./build/test_eigs` | Pass: 43 tests, 0 failed, 955 assertions. |
| `./build/test_eigs_lobpcg` | Pass: 29 tests, 0 failed, 287 assertions. |
| `make source-list-check` | Pass: 49 library sources. |

`./build/test_eigs` covers the focused shift-invert surfaces named in Day 11:

- diagonal nearest-sigma selection;
- symmetric-indefinite shift-invert setup;
- singular sigma error propagation;
- original-space eigenvector publication;
- wide-spectrum interior convergence;
- Sprint 114 vector-publication and grow-m conversion boundaries;
- LDLT CSC/linked-list telemetry through `used_csc_path_ldlt`.

`./build/test_eigs_lobpcg` covers LOBPCG nearest-sigma adjacency through the
shared operator/selection path.

## Source-List And Build Metadata Evidence

`make source-list-check` passed with 49 library sources. No Day 12 source-list,
Makefile, or CMake edits were made. The branch's existing build metadata still
reflects the Day 6 selection/lifting source owner:

- `src/sparse_eigs_selection_internal.c` in `Makefile`;
- `src/sparse_eigs_selection_internal.c` in `CMakeLists.txt`;
- `src/sparse_eigs_selection_internal.c` in
  `build-metadata/library_sources.txt`.

## Residual List

| Residual | Status | Future owner |
|---|---|---|
| Shift-invert setup/conversion private-owner extraction | Deferred. | Future sprint after Sprint 119. |
| Private shift-invert context type and cleanup helper | Deferred. | Same future shift-invert lifecycle extraction. |
| Transformed eigenvalue conversion helper | Deferred; keep conversion at backend publication sites until a dedicated design proves both converged and partial paths. | Future lifecycle extraction or backend publication cleanup. |
| `lanczos_iterate_op` recurrence movement | Still deferred; not part of Day 12. | Future recurrence-specific proof sprint. |

## Future Handoff Requirements

Any future shift-invert movement should begin from the Day 11 decision package
and prove all of the following before code movement:

1. private context setup/cleanup contract;
2. zeroed and partially initialized cleanup safety;
3. exact `SPARSE_ERR_SINGULAR`, allocation, mutation, factor, and solve error
   propagation;
4. `used_csc_path_ldlt` telemetry preservation;
5. one-shot, handle, and internal workspace lifetime preservation;
6. grow-m converged and partial-result conversion proof;
7. thick-restart converged and partial-result conversion proof;
8. LOBPCG nearest-sigma adjacency proof;
9. CMake/CTest/source-list and full C quality evidence if `.c` or `.h` files
   change.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Shift-invert movement or explicit deferral exists. | Complete: explicit deferral. |
| Focused shift-invert proof results exist. | Complete. |
| Source-list evidence exists. | Complete. |
| CMake/CTest count evidence rationale exists. | Complete; not rerun because no Day 12 build metadata changed. |
| Required quality-check summary exists. | Complete; no Day 12 `.c`/`.h` change, Day 10 full quality remains current. |
| Updated residual list exists. | Complete. |
| Item 5 validation is complete. | Complete. |
| Required checks pass before closeout work begins. | Complete. |
