# Day 12 Eigensolver Source Deferral

## Purpose

Day 12 closes the Sprint 107 eigensolver source workstream without moving
`src/sparse_eigs.c`. Day 11 established that no split candidate is low-risk
enough for this sprint because the remaining source groups cross shared
spectral kernels, public dispatch behavior, workspace handle behavior,
shift-invert/refinement semantics, or Sprint 103 comparison evidence.

## Decision

No eigensolver source extraction is performed in Sprint 107.

The Day 12 implementation path is the deferral path from the sprint plan:

- no new eigensolver `.c` file;
- no new eigensolver internal header;
- no Makefile or CMake source-list update;
- no public API or install-header movement;
- no CTest registration change;
- no reviewed source-count or Windows CTest-count implication.

## Deferral Rationale

The remaining `src/sparse_eigs.c` responsibilities are tightly coupled:

- Shared Lanczos and selection kernels are consumed by grow-m, thick restart,
  LOBPCG, and focused tests.
- Shift-invert and refinement helpers protect grow-m residual behavior and
  Sprint 29 integration evidence.
- Workspace handle glue is tied to public handle API behavior, while the
  backing workspace storage is already split into
  `src/sparse_eigs_workspace_internal.c`.
- Public defaults, validation, backend selection, and dispatch form one
  behavioral boundary.
- Dense symmetric Jacobi is a plausible future helper owner, but it is shared
  by thick restart and LOBPCG and should move only with dedicated source-list,
  CMake, and cross-backend validation work.

For Sprint 107, preserving comparison evidence and reviewed build surfaces is
more valuable than creating a small source split with broad follow-through
requirements.

## Residual Queue

Future eigensolver maintainability work should proceed in this order:

1. Dense helper split feasibility
   - Candidate: isolate `s21_dense_sym_jacobi` into a private dense spectral
     helper source.
   - Prerequisites: Makefile/CMake/source-list parity plan, focused thick
     restart and LOBPCG validation, no public header movement.
2. Grow-m refinement boundary audit
   - Candidate: document or separate shift-invert/refinement helpers only
     after residual comparison coverage is strong enough to catch semantic
     drift.
   - Prerequisites: explicit residual/oracle validation for closed-form,
     shift-invert, and refined eigenpair behavior.
3. Dispatch and handle boundary audit
   - Candidate: improve local naming and comments before any public dispatch
     extraction.
   - Prerequisites: reviewed backend-selection tests and handle-workspace
     behavior evidence.
4. Cross-backend shared kernel audit
   - Candidate: evaluate whether Lanczos selection and lifting helpers need a
     shared kernel owner.
   - Prerequisites: proof that thick restart, LOBPCG, grow-m, and focused
     tests can validate the same helper movement without obscuring comparison
     semantics.

## Validation Scope

Because Day 12 performs no source extraction and changes planning artifacts
only, the required validation is documentation hygiene:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_107 docs/planning/EPIC_10/PROJECT_PLAN.md
```

No focused eigensolver binary, source-list check, CMake check, or full
`make format && make lint && make test` gate is required for Day 12 because no
`.c`, `.h`, Makefile, or CMake file is changed by this day.

## Handoff

Day 13 can proceed to the central matrix shell deferral contract. The
eigensolver handoff is intentionally conservative:

- Sprint 107 has a current source boundary for `src/sparse_eigs.c`.
- Sprint 107 has an explicit no-split decision for Day 12.
- Future eigensolver source movement is fenced behind build-system parity and
  focused comparison validation.
