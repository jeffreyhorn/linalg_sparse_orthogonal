# Sprint 114 Day 10: Eigensolver Movement Decision

## Purpose

Day 10 reviews the Sprint 114 eigensolver proof stack from Days 2-9 and
decides whether one narrow private source movement is safe. The decision must
avoid broad eigensolver source splits, unsupported build/source-list drift,
public API changes, install-header drift, helper-target drift, and reviewed
CTest membership changes.

## Evidence Reviewed

| Evidence | Artifact | Result |
|---|---|---|
| Lanczos behavior proof design | `day2-lanczos-iterate-behavior-design.md` | Defined basic grow-m, thick-restart, and LOBPCG-adjacent proof targets. |
| Lanczos behavior proof implementation | `day3-lanczos-iterate-behavior-proof.md` | Added grow-m public behavior, thick-restart recurrence parity, and LOBPCG-adjacent parity tests. |
| Repeated/clustered Ritz selection design | `day4-ritz-selection-proof-design.md` | Separated exact selector proof from public clustered-spectrum proof. |
| Repeated/clustered Ritz selection proof | `day5-ritz-selection-proof.md` | Proved repeated values, nearest-sigma ties, and clustered public values. |
| Vector publication design | `day6-ritz-vector-publication-design.md` | Inventoried grow-m, shift-invert, thick-restart, and LOBPCG publication paths. |
| Vector publication proof | `day7-ritz-vector-publication-proof.md` | Proved full publication, vector residuals, orthogonality, and sentinel boundaries across current paths. |
| Partial-result publication proof | `day8-partial-result-publication-proof.md` | Proved bounded grow-m `m_cap` exhaustion publication shape. |
| Shift-invert grow-m conversion proof | `day9-shift-invert-growm-conversion-proof.md` | Proved original-space conversion, nearest-sigma order, backend fields, basis fields, and vector residuals. |

## Movement Candidates Considered

| Candidate | Current owner | Evidence now available | Decision |
|---|---|---|---|
| Move `s20_select_indices` into a separate Ritz-selection unit | `src/sparse_eigs.c`, declared in `src/sparse_eigs_internal.h`, consumed by grow-m, thick-restart, and LOBPCG | Day 5 proves selector behavior directly. | Defer. The helper is shared across three backend files, and movement would require source-list/build metadata edits plus focused Make/CMake/Windows reviewed-count proof not yet scheduled in this sprint. |
| Move `s20_lift_ritz_vectors` into a publication helper unit | `src/sparse_eigs.c`, consumed by grow-m and thick-restart | Days 7-8 prove public publication behavior and grow-m partial publication. | Defer. Grow-m and thick-restart still have different partial-state fallthroughs, and LOBPCG publishes from `X` rather than from a Lanczos basis/reduced-vector pair. |
| Move shift-invert setup/conversion into a private unit | `src/sparse_eigs.c` public entry path and grow-m publication branches | Days 7 and 9 prove original-space vector publication and `lambda = sigma + 1 / theta` conversion. | Defer. Setup is coupled to LDLT factor lifecycle, `used_csc_path_ldlt`, operator selection, public error propagation, and cleanup ownership in the public entry function. |
| Move `lanczos_iterate_op` into a private recurrence unit | `src/sparse_eigs.c`, declared in `src/sparse_eigs_internal.h`, consumed by thick-restart | Days 2-3 prove basic, thick-restart, and adjacent behavior. | Defer. The recurrence is central enough that movement would require an explicit source-list/compile-unit proof and focused tests for all current consumers after the split. |
| Move the entire grow-m backend into a separate unit | `src/sparse_eigs.c` | Days 2-9 prove many grow-m public behaviors. | Reject for Sprint 114. This is broader than a narrow movement and would couple selection, publication, shift-invert, workspace, progress, and public result semantics in one risky source split. |

## Decision

Continue the eigensolver no-move contract for Sprint 114 Day 10.

The Sprint 114 proof stack materially improves the safety case, but no
candidate is narrow enough to move without introducing build/source-list risk
or hiding still-distinct proof-owner semantics. The safest outcome is to leave
the current source ownership intact and hand a precise movement checklist to a
future sprint.

## Future Proof Requirements

Before a future sprint moves any eigensolver owner, it should provide:

1. A single-owner movement plan naming exact old and new files.
2. Makefile and CMake source-list updates with no unreviewed target drift.
3. Focused tests for all current consumers of the moved helper.
4. `ctest -N` evidence for reviewed CMake paths if test registration or
   source membership changes affect reviewed CI lanes.
5. Windows reviewed CMake count review when source-list changes touch the
   enforced Windows consumer subset.
6. A rollback plan that restores the helper to its current owner without
   changing public behavior.
7. Full `make format && make lint && make test` after movement.

## Non-Claims

- No eigensolver source file was split.
- No public API, install header, or ABI claim changed.
- No helper target, Make target, CMake target, source-list entry, or reviewed
  CTest membership changed.
- No package, platform, Windows, or CMake parity claim is added by this
  decision.
- Day 11 direct/iterative cleanup does not depend on future eigensolver
  movement.

## Validation

Day 10 changes documentation only. The required validation is:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_114
```

The prior Day 9 C changes already passed the required full gate:

```text
make format && make lint && make test
```

## Completion Criteria

- Days 2-9 evidence has been reviewed.
- A clear no-move decision is published.
- Candidate-specific blockers and future proof requirements are documented.
- Direct/iterative cleanup can proceed without depending on eigensolver source
  movement.
