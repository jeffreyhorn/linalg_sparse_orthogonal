# Sprint 105 Day 10 Graph/Reorder Ownership Cleanup

## Purpose

Day 10 starts the graph/reorder ownership cleanup with a narrow,
behavior-preserving source pass. The selected owner is
`tests/test_reorder_amd_qg.c` because it is small, directly participates in
the Day 9 large-matrix guardrail target, and carried stale day-by-day comments
around the maintained qg-AMD proof surface.

## Cleanup Scope

Touched file:

```text
tests/test_reorder_amd_qg.c
```

Changes made:

- rewrote the file header around current ownership:
  - qg-AMD internal argument validation;
  - public wrapper delegation;
  - symbolic fill equality;
  - large regular banded structural guardrail;
- replaced the obsolete stub-retirement comment with the current argument
  validation contract;
- rewrote the wrapper/helper comparison comment to describe current delegation
  behavior;
- replaced the old day-labeled banded stress comment with a maintained
  large-regular-input guardrail description;
- rewrote the timing ceiling comment to clarify that the test is a structural
  guard, not a portable timing benchmark;
- renamed the test suite banner from sprint-history wording to current
  ownership wording.

## Helper Extraction Decision

No helper extraction was made.

Reason:

- `is_valid_permutation` and `symbolic_cholesky_nnz_with_perm` are small,
  local, and scoped to this proof owner;
- extracting them would add cross-file surface area without removing
  meaningful duplication;
- Day 10's cleanup goal is better served by clarifying current ownership and
  preserving behavior.

## Preserved Behavior

The same tests still run:

- `test_amd_qg_null_args`;
- `test_amd_qg_rejects_rectangular`;
- `test_amd_qg_singleton`;
- `test_amd_qg_delegation_nos4`;
- `test_amd_qg_delegation_bcsstk04`;
- `test_amd_qg_delegation_bcsstk14`;
- `test_amd_stress_10k_banded`.

The qg-AMD large generated guardrail remains the same:

- fixture: `banded-n10000-bw5`;
- structural check: valid permutation;
- broad ceiling: `secs < 30.0`;
- interpretation: structural regression guard, not portable timing evidence.

## Focused Validation

Command:

```sh
make build/test_reorder_amd_qg && ./build/test_reorder_amd_qg
```

Result:

```text
Tests run:    7
Tests failed: 0
Tests skipped: 0
Assertions:   2068
ALL TESTS PASSED
```

## Residual Cleanup Queue

| owner | residual issue | recommendation |
|---|---|---|
| `tests/test_graph.c` | very large proof owner with many sprint-history comments | defer to a dedicated cleanup pass; avoid touching while guardrails are still being stabilized |
| `tests/test_reorder_nd.c` | very large ND policy proof owner with many historical rationale blocks | defer to a dedicated cleanup pass with narrower sub-owner selection |
| `src/sparse_graph.c` | large design block and many historical policy notes | clean only when touching graph implementation behavior or extracting docs |
| `src/sparse_reorder_nd.c` | threshold and policy comments mix current rationale with history | clean only with ND policy artifact alignment |
| `benchmarks/bench_amd_qg.c` | historical bitset benchmark comments remain intentionally descriptive | keep until benchmark docs absorb the bitset-foil context |

## Completion Check

| criterion | status |
|---|---|
| touched comments describe current ownership | complete |
| helper extraction considered | complete |
| behavior preserved | complete |
| focused validation passed | complete |
| residual cleanup queue recorded | complete |
