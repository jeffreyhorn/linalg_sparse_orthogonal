# Sprint 44 Day 3 Artifact: FM Refinement Module Boundary Design

## Purpose

Define the concrete FM refinement extraction boundary before Sprint 44 begins
moving code out of the residual `src/sparse_graph.c`.

## Day 3 Design Goal

Sprint 44 should extract the FM refinement subsystem as one real owned module
without pulling `graph_uncoarsen(...)` or broader partition orchestration into
the same batch.

The correct Day 5 extraction target is therefore:

- `src/sparse_graph_refine.c`

This is an algorithm-owned module, not a generic graph-helper file.

## Proposed Module Ownership

### The FM module should own

#### 1. FM-local thread state

- `fm_pop_use_tail`
- `fm_use_annealing`
- `fm_anneal_schedule`
- `fm_use_thick_restart`
- `fm_thick_restart_perturb`
- `fm_gain_noise_schedule`
- `fm_anneal_pass_idx`
- `fm_anneal_total_passes`

These are part of the FM behavior seam, not general graph orchestration state.

#### 2. FM-local parser/helpers

- `parse_fm_anneal_schedule(...)`
- `parse_fm_thick_restart_perturb(...)`
- `parse_fm_gain_noise_schedule(...)`
- `thick_restart_perturb(...)`

These describe refinement behavior itself and belong with the FM module.

#### 3. FM score / update helper

- `compute_cut_weight(...)`

Even though `graph_uncoarsen(...)` currently uses it for thick-restart and
ensemble bookkeeping, the helper is more conceptually tied to refinement-state
evaluation than to top-level orchestration.

#### 4. FM bucket implementation

- `fm_bucket_array_init(...)`
- `fm_bucket_array_free(...)`
- `fm_bucket_insert(...)`
- `fm_bucket_remove(...)`
- `fm_bucket_pop_max(...)`
- `fm_bucket_pop_max_tail(...)`

#### 5. FM algorithm entry point

- `graph_refine_fm(...)`

## What Should Remain Outside the FM Module

### Keep in residual orchestration for Sprint 44

- `graph_uncoarsen(...)`
- finest-level pass-count parsing
- finest-level strategy selection parsing
- ensemble selector-list parsing
- intermediate-pass parsing
- pass-to-pass orchestration around FM calls

Reason:

- those responsibilities still compose:
  - extracted coarsening hierarchy
  - extracted coarse bisection
  - extracted FM
  - residual separator lifting

They are not pure FM ownership.

## Header and Interface Decisions

### `src/sparse_graph_internal.h`

Keep as the main behavior-level graph seam.

FM-related exports should remain minimal:

- `graph_refine_fm(...)`

Possible additional declaration only if Day 5 requires a cross-file call:

- `compute_cut_weight(...)`

Day 3 default:

- do not expand this header with FM parser helpers
- do not expose FM thread-local state

### `src/sparse_graph_fm_buckets.h`

Keep as the dedicated FM-support header.

It should continue to own:

- `fm_bucket_array_t`
- bucket API declarations

Reason:

- this header already has focused unit-test coverage
- it is narrower and cleaner than folding the bucket API into
  `src/sparse_graph_internal.h`

## Shared vs Local Ownership Map

### Shared behavior seam

- `graph_refine_fm(...)`

### Shared support seam

- bucket API through `src/sparse_graph_fm_buckets.h`

### Translation-unit local by default

- FM thread-local controls
- FM parser helpers
- perturbation helper
- debug-print helpers implicit in the current implementation

### Conditional shared helper

- `compute_cut_weight(...)`

Only promote it if Day 5 actually needs `graph_uncoarsen(...)` to call across
files after the move. Otherwise prefer to keep it local.

## Naming and Ownership Rules

### Naming

Use:

- `src/sparse_graph_refine.c`

Avoid:

- `src/sparse_graph_phase2.c`
- `src/sparse_graph_runtime.c`
- `src/sparse_graph_helpers.c`

Reason:

- the extracted ownership is refinement-specific
- broader names would blur the boundary Sprint 44 is trying to make clearer

### Ownership rules

- FM move logic, gain recomputation, rollback, and perturbation behavior belong
  to the FM module
- orchestration of when/how many FM passes to run stays with
  `graph_uncoarsen(...)`
- bucket abstractions remain a separate FM-support seam with their existing
  dedicated header/tests

## Day 5 Implementation Target

### Do

- create `src/sparse_graph_refine.c`
- move the FM bucket implementation
- move the FM parser/helpers
- move FM thread-local state
- move `compute_cut_weight(...)`
- move `graph_refine_fm(...)`
- update build wiring
- update only the minimum headers/includes needed

### Do not

- move `graph_uncoarsen(...)`
- move separator lifting
- redesign env-var semantics
- broaden into runtime-strategy cleanup
- redesign graph tests beyond what the extraction itself requires

## Bottom Line

The FM extraction boundary is now explicit:

- the right extracted file is `src/sparse_graph_refine.c`
- the module owns FM-local state, FM parser/helpers, the bucket
  implementation, `compute_cut_weight(...)`, and `graph_refine_fm(...)`
- `graph_uncoarsen(...)` remains outside the extraction because it is still
  composition/orchestration glue
- shared-header expansion should stay minimal, with `compute_cut_weight(...)`
  the only plausible new cross-file declaration candidate
