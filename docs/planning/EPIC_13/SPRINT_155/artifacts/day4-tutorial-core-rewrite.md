# Sprint 155 Day 4 Tutorial Core Rewrite

## Purpose

Day 4 implements the core tutorial rewrite from the Day 3 design. The changes
focus on build-tree first use, maintained example anchors, data-input routing,
local link/install boundaries, and solver-choice alignment. Deeper diagnostics,
SVD/eigensolver finishing, advanced controls, report handoff, and API
reference handoff remain Day 5 work.

## Files Changed

| File | Change Role |
| --- | --- |
| `docs/tutorial.md` | Reframed the tutorial opening, documentation map, local build/link path, input routing, and compact solver workflow table. |
| `docs/planning/EPIC_13/SPRINT_155/WORKING_NOTES.md` | Recorded Day 4 implementation summary and Day 5 handoff. |
| `docs/planning/EPIC_13/SPRINT_155/artifacts/day4-tutorial-core-rewrite.md` | Captures this Day 4 rewrite record. |

No `.c` or public `.h` files were modified.

## Implemented Core Tutorial Changes

### First-Use Opening

The tutorial now starts from the maintained first-use route:

1. build locally;
2. run the first maintained solve;
3. start from CSR, CSC, Matrix Market, or hand-written input;
4. choose the solver family by problem shape;
5. inspect workflow-local diagnostics;
6. install only when a downstream consumer needs it;
7. move to advanced controls, benchmarks, reports, public headers, or API
   reference only after the first workflow works.

This aligns the tutorial with the README, examples README, cookbook, and
solver-selection surfaces without duplicating their full policy content.

### Documentation Map

The documentation map now names owner surfaces more directly:

- README for the short project front door;
- examples for runnable first-use examples;
- cookbook for data-first recipes;
- solver-selection for the compact problem-shape decision tree;
- INSTALL for installed consumers;
- benchmarks README for benchmark/report interpretation;
- public headers and generated API reference for exact declarations;
- maintainer guide for maintainer quality policy.

### Build-Tree First Solve

The tutorial now gives one concrete first success path:

```sh
make
make examples
./build/example_basic_solve
```

It also clarifies:

- `example_basic_solve` is the smallest maintained first success path;
- `make examples-build` is the compile-only examples route;
- `make test` is validation for code changes rather than the first learning
  step.

### Local Link And Install Boundary

The tutorial keeps the local build-tree compile command:

```sh
cc -O2 -Iinclude -o my_program my_program.c -Lbuild -lsparse_lu_ortho -lm
```

Installed consumers are delegated to `INSTALL.md`:

- `INSTALL.md#start-here`;
- `INSTALL.md#using-via-pkg-config`;
- `INSTALL.md#using-from-a-cmake-project`.

The tutorial explicitly leaves the static-first package story owned by
`INSTALL.md`.

### Include Guidance

The include list now covers the high-use families needed by the current
tutorial flow:

- core matrix and CSR/CSC conversion;
- LU, Cholesky, LDLT, and QR;
- iterative solvers;
- ILU and IC preconditioners;
- SVD;
- symmetric eigensolver.

Exact declarations, option structs, result fields, and ownership contracts
remain delegated to public headers and the API reference surface.

### Data-Input Routing

The `Start From Your Matrix` section now appears before solver detail and
routes:

- small hand-written matrices to `sparse_create(...)` /
  `sparse_insert(...)`;
- CSR arrays to `sparse_create_from_csr(...)` or `sparse_from_csr(...)`;
- CSC arrays to `sparse_create_from_csc(...)` or `sparse_from_csc(...)`;
- Matrix Market files to `sparse_load_mm(...)`.

It also points to:

- `example_compressed_input`;
- `example_matrix_market`;
- `docs/cookbook.md#start-from-your-data`;
- `docs/matrix_market.md`.

### Solver Workflow Table

The tutorial now has a compact `Choose the Solver Workflow` table that maps
needs to first workflows and runnable anchors:

- LU and `example_basic_solve`;
- Cholesky and solver-selection direct guidance;
- LDLT and `example_ldlt`;
- QR and `example_least_squares` / `example_minnorm`;
- iterative solvers and `example_iterative` / `example_ic_minres`;
- `sparse_eigs_sym(...)` and `example_eigs`;
- SVD APIs and `example_svd_lowrank` / `example_condition`;
- matrix-free iterative workflows and `example_matrix_free`.

## Claim Boundary Check

Day 4 preserved these boundaries:

- no shared-library support claim;
- no dynamic ABI support claim;
- no package-manager support claim;
- no broad Windows parity claim;
- no Windows Makefile or Windows `pkg-config` parity claim;
- no portable performance claim;
- no broad QR, SVD, partial-SVD, or external-library parity claim;
- no state-of-the-art claim;
- no generated report freshness claim.

The tutorial now links install consumers to `INSTALL.md` instead of restating
package support tiers in the tutorial.

## Day 5 Handoff

Day 5 should finish the remaining tutorial alignment work:

1. Rewrite the preconditioning wording that still says ILU "dramatically
   reduces" iteration counts.
2. Refresh or delegate the partial-SVD evidence text to the current Sprint 151
   fixture set.
3. Add a compact symmetric eigensolver section if the current workflow table is
   not enough.
4. Add workflow-local diagnostics guidance aligned with
   `docs/solver_selection.md#diagnostics-handoff`.
5. Add concise advanced-control, report, install, and API-reference handoffs.
6. Re-check tutorial wording for unsupported package, ABI, platform,
   performance, external-parity, generated-report, and state-of-the-art
   claims.

## Day 4 Completion Check

- Core tutorial opening was rewritten around the first-use ladder.
- Maintained first-solve path points to `example_basic_solve`.
- Data-input routing now precedes solver depth.
- Local build-tree linking is separated from installed-consumer setup.
- Solver-choice table aligns with current owner docs and examples.
- Remaining Day 5 work is bounded to diagnostics, advanced handoffs, and stale
  evidence/claim wording.
