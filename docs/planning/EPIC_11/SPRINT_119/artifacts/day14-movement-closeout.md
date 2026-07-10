# Sprint 119 Day 14 Movement Closeout

## Purpose

Close Sprint 119 by publishing exactly what moved, what stayed in place, what
validation supports the work, which claims were intentionally not made, and
what Sprint 120 should inherit from the eigensolver source-boundary effort.

## Movement Summary

| Candidate | Outcome | Evidence |
| --- | --- | --- |
| Eigensolver movement inventory | Completed | Day 2 catalogued private eigensolver owners, dependency risk, and likely movement candidates. |
| Movement feasibility ranking | Completed | Day 3 ranked candidates and selected the lowest-risk movement path. |
| Source-boundary design | Completed | Day 4 defined new/old file ownership, build registrations, rollback expectations, and proof surfaces. |
| Focused consumer proof design | Completed | Day 5 defined grow-m, thick-restart, LOBPCG, source-list, and CMake validation expectations. |
| Selection/lifting helper movement | Completed | Day 6 moved `s20_select_indices` and `s20_lift_ritz_vectors` into `src/sparse_eigs_selection_internal.c`. |
| Build-system registration | Completed | Day 6 registered the new source in `Makefile`, `CMakeLists.txt`, and `build-metadata/library_sources.txt`. |
| Focused movement validation | Completed | Days 7, 10, and 13 reran focused eigensolver tests, source-list checks, CMake CTest count proof, and full Make quality. |
| Shift-invert setup/conversion movement | Deferred | Day 11 documented lifecycle coupling; Day 12 validated the explicit deferral. |

## Files Moved Or Changed

| File | Change |
| --- | --- |
| `src/sparse_eigs_selection_internal.c` | New private source owning selection/lifting helpers. |
| `src/sparse_eigs.c` | Removed the extracted selection/lifting helper implementations while keeping shift-invert setup/conversion local. |
| `Makefile` | Added the new private source to the library source list. |
| `CMakeLists.txt` | Added the new private source to the CMake library target. |
| `build-metadata/library_sources.txt` | Added the new private source to source-list parity metadata. |
| `docs/planning/EPIC_11/SPRINT_119/` | Added Sprint 119 plan, working notes, and day-by-day evidence artifacts. |

## Deferred Debt And Residuals

| Residual | Why Deferred | Follow-Up Requirement |
| --- | --- | --- |
| Shift-invert setup/conversion private-owner extraction | The setup flow owns shifted matrix construction, LDLT factor lifetime, backend telemetry, operator selection, transformed eigenvalue conversion, error propagation, and cleanup. Moving it without a context/lifecycle owner would increase risk. | Future sprint should design a private shift-invert context with setup/cleanup helpers, one-shot and reusable-handle proof, LOBPCG nearest-sigma adjacency proof, CMake/CTest count evidence, and full quality evidence. |
| Transformed eigenvalue conversion helper extraction | Conversion currently occurs at result publication boundaries for converged and partial results. Moving it separately risks drifting grow-m and thick-restart behavior. | Extract only with a shared publication contract or document why backend-local conversion remains the clearer boundary. |
| `lanczos_iterate_op` movement | Lower priority than selection/lifting and shift-invert lifecycle ownership for Sprint 119. | Reconsider only after shift-invert ownership is explicit enough that operator callback lifetimes are easy to audit. |

## Validation Summary

| Validation Lane | Status | Evidence |
| --- | --- | --- |
| Source-list parity | Pass | `make source-list-check` reported 49 library sources. |
| Focused grow-m eigensolver behavior | Pass | `./build/test_eigs`: 43 tests, 0 failed, 955 assertions. |
| Focused thick-restart behavior | Pass | `./build/test_eigs_thick_restart`: 23 tests, 0 failed, 384 assertions. |
| Focused LOBPCG behavior | Pass | `./build/test_eigs_lobpcg`: 29 tests, 0 failed, 287 assertions. |
| CMake build and CTest registration | Pass | Clean CMake build compiled `src/sparse_eigs_selection_internal.c`; `ctest -N` reported `Total Tests: 54`. |
| Required full Make quality | Pass | `make format && make lint && make test` ended with `All tests passed.` |

## Non-Claim Register

Sprint 119 deliberately creates a source-boundary and proof-owner improvement,
not a new numerical capability claim.

| Non-Claim | Reason |
| --- | --- |
| No broad eigensolver parity claim | The work moved private helper ownership; it did not broaden numerical algorithms, matrix corpus scope, convergence guarantees, or external oracle coverage. |
| No ARPACK parity claim | Sprint 119 did not compare against ARPACK, duplicate ARPACK semantics, or add ARPACK-backed acceptance criteria. |
| No SciPy parity claim | Sprint 119 did not run SciPy comparisons or publish SciPy-compatible behavior claims. |
| No LAPACK parity claim | The moved helpers are sparse eigensolver selection/lifting internals, not dense LAPACK-backed eigensolver or SVD parity work. |
| No state-of-the-art eigensolver claim | The sprint reduced source-boundary risk and preserved behavior; it did not establish benchmark, robustness, scalability, or external-comparison evidence sufficient for a state-of-the-art claim. |
| No performance claim | Benchmarks were not run because the completed movement should be behavior-preserving and ownership-focused. |
| No public API claim | Public headers and API contracts were not changed. |

## Sprint 120 Handoff

Sprint 120 should use Sprint 119 as a validation-pattern precedent, not as a
direct eigensolver continuation.

| Handoff Area | Guidance |
| --- | --- |
| Proof-owner movement discipline | Keep movement batches small, source-list aware, and paired with focused consumer tests before full quality. |
| Oracle architecture | Apply the Day 5/Day 13 evidence packaging pattern to direct/iterative oracle splits. |
| Giant-test split risk | Preserve solver-specific tolerances and failure modes when extracting shared fixtures. Do not hide behavior behind generic helpers. |
| Build parity | Any new source owner must be registered in Make, CMake, and source-list metadata in the same change. |
| Non-claims | Continue to state explicitly when a source-boundary or fixture split does not create solver parity, benchmark, or state-of-the-art claims. |
| Shift-invert residual | Do not mix the shift-invert private-owner extraction into Sprint 120 unless Sprint 120 scope is intentionally changed; it remains a future eigensolver-specific lifecycle task. |

## Artifact Index

| Day | Artifact | Purpose |
| --- | --- | --- |
| 1 | `day1-sprint-intake.md` | Sprint scope, prerequisites, and validation baseline. |
| 2 | `day2-eigensolver-movement-candidate-inventory.md` | Candidate inventory and initial ownership map. |
| 3 | `day3-movement-feasibility-ranking.md` | Movement-risk ranking and candidate selection. |
| 4 | `day4-source-boundary-design.md` | Source-boundary design and rollback plan. |
| 5 | `day5-focused-consumer-proof-design.md` | Focused proof design for eigensolver consumers. |
| 6 | `day6-first-movement-implementation.md` | Selection/lifting movement implementation evidence. |
| 7 | `day7-first-movement-validation.md` | First post-movement validation evidence. |
| 8 | `day8-selection-lifting-proof-audit.md` | Selection/lifting proof audit. |
| 9 | `day9-selection-lifting-movement-consolidation.md` | Consolidation of movement boundaries and residuals. |
| 10 | `day10-selection-lifting-validation.md` | Final selection/lifting validation refresh. |
| 11 | `day11-shift-invert-boundary-decision.md` | Shift-invert split/defer decision. |
| 12 | `day12-shift-invert-deferral-validation.md` | Explicit shift-invert deferral validation. |
| 13 | `day13-validation-parity-package.md` | Full validation and parity package. |
| 14 | `day14-movement-closeout.md` | Closeout, residuals, non-claims, and Sprint 120 handoff. |

## Completion Criteria

| Criterion | Status |
| --- | --- |
| Sprint 119 Item 7 complete | Complete |
| Every moved candidate has evidence | Complete |
| Every deferred candidate has evidence | Complete |
| Validation outcomes are recorded | Complete |
| Non-claim boundaries are recorded | Complete |
| Sprint 120 handoff risks and prerequisites are recorded | Complete |
