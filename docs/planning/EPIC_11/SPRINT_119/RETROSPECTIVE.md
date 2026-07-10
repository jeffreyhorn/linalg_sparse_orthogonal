# Sprint 119 Retrospective

**Sprint:** 119 - Eigensolver Source Boundary & Proof-Owner Follow-Through
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 119 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read the Epic 11 Sprint 119 project-plan scope and prior residual
      eigensolver movement notes.
- [x] Inventoried eigensolver private-owner movement candidates, including
      `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert setup,
      transformed-value conversion, and `lanczos_iterate_op`.
- [x] Ranked candidates by dependency risk and selected the lowest-risk
      selection/lifting movement path.
- [x] Designed exact source-boundary ownership, build registration, rollback,
      and focused consumer proof expectations before moving code.
- [x] Moved `s20_select_indices` and `s20_lift_ritz_vectors` into the new
      private source `src/sparse_eigs_selection_internal.c`.
- [x] Registered the new private source in `Makefile`, `CMakeLists.txt`, and
      `build-metadata/library_sources.txt`.
- [x] Validated the movement with focused grow-m, thick-restart, and LOBPCG
      eigensolver tests.
- [x] Validated source-list and CMake/CTest membership parity after the new
      source registration.
- [x] Ran the required full C quality chain for the branch's `.c` movement:
      `make format && make lint && make test`.
- [x] Audited shift-invert setup/conversion ownership and explicitly deferred
      that movement because lifecycle, telemetry, operator dispatch, cleanup,
      and transformed-value conversion remain tightly coupled.
- [x] Validated the shift-invert deferral with focused adjacent eigensolver
      tests and source-list checks.
- [x] Published non-claim boundaries for eigensolver parity, ARPACK, SciPy,
      LAPACK, performance, public API, and state-of-the-art claims.
- [x] Published Sprint 120 handoff guidance for proof-owner movement,
      direct/iterative oracle work, source-list parity, and non-claim hygiene.
- [x] Finalized this retrospective and ran focused documentation hygiene.

## What Went Well

1. **The sprint moved the safest eigensolver owner instead of the most tempting
   one.**
   The movement inventory and feasibility ranking kept Sprint 119 focused on
   selection/lifting helpers, where ownership was private and consumer proof was
   compact. That avoided mixing a source-boundary improvement with the more
   fragile shift-invert lifecycle.

2. **Build-system parity stayed coupled to the source move.**
   `src/sparse_eigs_selection_internal.c` was added to Make, CMake, and
   `build-metadata/library_sources.txt` in the same movement batch. Day 13 then
   proved that the source-list check and clean CMake build both recognized the
   new owner.

3. **Focused consumer proof covered the relevant eigensolver paths.**
   The validation suite included grow-m Lanczos, thick-restart, and LOBPCG
   tests. That was the right scope for moved selection/lifting behavior because
   the helpers influence value ordering and vector publication across those
   consumers.

4. **The shift-invert decision was explicit rather than accidental.**
   Day 11 documented why setup/conversion should not move in Sprint 119:
   shifted matrix construction, LDLT lifetime, backend telemetry, operator
   selection, transformed-value conversion, and cleanup still share one public
   eigensolver flow. Day 12 then validated that deferral instead of leaving it
   as an untracked loose end.

5. **The validation package preserved the claim boundary.**
   Day 13 recorded source-list, focused eigensolver, CMake CTest count, and
   full Make quality evidence while also listing skipped supplemental lanes.
   That made Day 14 closeout a factual summary rather than a new claim pass.

6. **Non-claims were documented next to the evidence.**
   The closeout explicitly says this sprint did not create ARPACK, SciPy,
   LAPACK, performance, public API, or state-of-the-art eigensolver parity
   claims. That keeps the source-boundary win from being overstated.

## What Did Not Go Well

1. **The original eigensolver file still owns important lifecycle-heavy code.**
   Moving selection/lifting reduced one private-owner cluster, but
   shift-invert setup/conversion and adjacent operator lifecycle logic remain
   in `src/sparse_eigs.c`. That is correct for Sprint 119, but still real
   residual maintainability debt.

2. **Shift-invert movement needs a design sprint, not a helper extraction.**
   The future work needs a private context with setup/cleanup contracts,
   telemetry publication rules, and one-shot/reusable-handle proof. That is
   larger than a safe end-of-sprint movement batch.

3. **The sprint needed repeated validation because the touched surface is
   small but central.**
   The source move was mechanically small, but eigensolver selection and vector
   lifting affect several public behaviors. The repeated focused test and full
   quality runs were necessary, even though they consumed a large share of the
   final sprint days.

4. **Performance and external-comparison evidence remain out of scope.**
   This was an ownership sprint. It did not add external eigensolver oracle
   comparisons, benchmark refreshes, or broader matrix corpus evidence.

## Final Metrics

### Validation

| Metric | Sprint 119 close state |
|---|---:|
| library source-list count | 49 |
| CMake registered tests | 54 |
| focused grow-m eigensolver tests | 43 passed, 0 failed |
| focused grow-m eigensolver assertions | 955 |
| focused thick-restart tests | 23 passed, 0 failed |
| focused thick-restart assertions | 384 |
| focused LOBPCG tests | 29 passed, 0 failed |
| focused LOBPCG assertions | 287 |
| required full Make quality | `make format && make lint && make test` passed |
| full Make test final result | `All tests passed.` |
| clean CMake membership proof | configured, built, and `ctest -N` reported 54 tests |
| diff hygiene | `git diff --check` passed |
| trailing-whitespace scan | passed on Sprint 119 docs |
| temporary CMake review build | removed before closeout |

### Sprint Artifact Package

| Metric | Sprint 119 close state |
|---|---:|
| artifact files under `SPRINT_119/artifacts/` | 14 |
| sprint plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| moved private source files | 1 |
| modified existing source files | 1 |
| modified build/source-list files | 3 |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Selection/lifting source ownership | Completed private source extraction into `src/sparse_eigs_selection_internal.c`. |
| Grow-m behavior | Preserved by focused `test_eigs` evidence. |
| Thick-restart behavior | Preserved by focused `test_eigs_thick_restart` evidence. |
| LOBPCG behavior | Preserved by focused `test_eigs_lobpcg` evidence. |
| Make/CMake/source-list parity | Preserved; new source is registered in all three inventories. |
| Shift-invert setup/conversion | Explicitly deferred with lifecycle rationale and focused validation. |
| Public API | Unchanged. |
| Public documentation claims | Unchanged; closeout records non-claims. |
| Benchmarks/performance | Not claimed and not refreshed. |
| External eigensolver parity | Not claimed. |

## Residual Deferred Debt

Most important carry-forward work:

- Design and implement a private shift-invert context owner only with:
  - shifted matrix construction ownership;
  - LDLT factor lifecycle and cleanup contracts;
  - backend telemetry publication rules;
  - exact error propagation;
  - one-shot, handle, workspace, grow-m, thick-restart, and LOBPCG proof.
- Decide whether transformed eigenvalue conversion should be extracted into a
  shared publication helper or remain backend-local with documented boundaries.
- Reconsider `lanczos_iterate_op` movement after shift-invert callback
  lifetime ownership is clearer.
- Preserve the Day 13 validation pattern for future eigensolver movement:
  source-list proof, focused eigensolver tests, CMake CTest count evidence, and
  full quality when `.c` or `.h` files change.

Still consciously constrained rather than silently solved:

- no broad eigensolver parity claim;
- no ARPACK parity claim;
- no SciPy parity claim;
- no LAPACK parity claim;
- no state-of-the-art eigensolver claim;
- no performance claim;
- no public API claim;
- no package/install claim;
- no Windows local validation claim.

Not carried forward as unresolved Sprint 119 debt:

- eigensolver movement candidate inventory;
- selection/lifting feasibility ranking;
- source-boundary design for the selected movement;
- focused consumer proof design;
- selection/lifting private source extraction;
- Make/CMake/source-list registration for the new private source;
- final validation and closeout package.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake.md](./artifacts/day1-sprint-intake.md)
- [day2-eigensolver-movement-candidate-inventory.md](./artifacts/day2-eigensolver-movement-candidate-inventory.md)
- [day3-movement-feasibility-ranking.md](./artifacts/day3-movement-feasibility-ranking.md)
- [day4-source-boundary-design.md](./artifacts/day4-source-boundary-design.md)
- [day5-focused-consumer-proof-design.md](./artifacts/day5-focused-consumer-proof-design.md)
- [day6-first-movement-implementation.md](./artifacts/day6-first-movement-implementation.md)
- [day7-first-movement-validation.md](./artifacts/day7-first-movement-validation.md)
- [day8-selection-lifting-proof-audit.md](./artifacts/day8-selection-lifting-proof-audit.md)
- [day9-selection-lifting-movement-consolidation.md](./artifacts/day9-selection-lifting-movement-consolidation.md)
- [day10-selection-lifting-validation.md](./artifacts/day10-selection-lifting-validation.md)
- [day11-shift-invert-boundary-decision.md](./artifacts/day11-shift-invert-boundary-decision.md)
- [day12-shift-invert-deferral-validation.md](./artifacts/day12-shift-invert-deferral-validation.md)
- [day13-validation-parity-package.md](./artifacts/day13-validation-parity-package.md)
- [day14-movement-closeout.md](./artifacts/day14-movement-closeout.md)
