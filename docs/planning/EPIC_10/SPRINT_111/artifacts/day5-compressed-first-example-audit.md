# Day 5 Compressed-First Example Audit

## Purpose

Day 5 selects the bounded example work for Sprint 111 before editing example
source files. The sprint needs examples that reinforce the new
`docs/solver_selection.md` guide, especially compressed-first CSR/CSC
construction, solver workflow selection, Matrix Market load/use, and example
versus benchmark handoff. This artifact records the current example inventory,
update/add decisions, validation requirements, and public-API guardrails for
Days 6-8.

## Source Inputs

- `docs/solver_selection.md`
- `examples/README.md`
- `examples/*.c`
- `examples/cmake_example/`
- `Makefile`
- `CMakeLists.txt`
- `docs/matrix_market.md`
- public headers under `include/`

## Build Registration Baseline

| Build Surface | Current Behavior | Day 5 Implication |
|---|---|---|
| Makefile | `EX_SRCS = $(wildcard examples/*.c)` and `EX_BINS = $(patsubst ...)`. | New `examples/*.c` files are automatically included in `make examples`. |
| CMake | Each example is registered explicitly with `add_executable(...)` and `target_link_libraries(...)`. | New compiled examples must be added to `CMakeLists.txt`. |
| Example docs | `examples/README.md` describes selected example binaries and points to install/tutorial/benchmark docs. | README needs alignment with the new solver-selection guide and any new Matrix Market route. |

## Existing Example Inventory

| Example | Current Workflow | Matrix Setup | Day 5 Decision |
|---|---|---|---|
| `example_basic_solve` | Smallest one-shot LU solve. | Mutable insertion. | Keep; use as first direct-solve reference. |
| `example_compressed_input` | Caller-owned CSR arrays into one-shot LU. | CSR compressed-first only. | Update on Day 6 to include CSC construction or conversion. |
| `example_analysis` | Analyze-once / factor-many direct lifecycle. | Mutable same-pattern SPD fixtures. | Keep; align README wording with solver guide on Day 7. |
| `example_iterative` | One-shot GMRES with and without ILU(0). | Mutable generated matrix. | Keep; align README wording with solver guide on Day 7. |
| `example_eigs` | Symmetric eigensolver, Matrix Market fixture load, shift-invert, LOBPCG. | Matrix Market for SuiteSparse fixtures plus generated KKT. | Keep; reference as eigensolver example and avoid treating it as the dedicated Matrix Market teaching path. |
| `example_svd_lowrank` | SVD, rank, condition, low-rank output. | Mutable generated matrix. | Keep; align with SVD section of solver guide. |
| `example_colamd` | COLAMD ordering and QR usage. | Mutable generated matrix. | Keep; use for reorder/fill handoff. |
| `example_ldlt` | Symmetric indefinite LDLT/KKT solve. | Mutable generated KKT. | Keep; make discoverable from examples README if needed. |
| `example_ic_minres` | IC(0), CG, MINRES, KKT, block MINRES. | Mutable generated matrices. | Keep; make discoverable from iterative/preconditioner guide text if needed. |
| `example_least_squares` | Overdetermined QR least-squares. | Mutable generated matrix. | Keep; align with QR guidance. |
| `example_minnorm` | Underdetermined minimum-norm QR. | Mutable generated matrix. | Keep; align with QR guidance. |
| `example_condition` | Condition number and ill-conditioning. | Mutable generated matrices. | Keep; align with SVD/condition handoff. |
| `example_matrix_free` | Matrix-free GMRES callback. | No explicit matrix. | Keep; document as advanced iterative route if examples README is updated. |
| `examples/cmake_example` | Installed downstream CMake consumer. | Small local matrix. | Keep separate from local build-tree examples. |

## Selected Sprint 111 Workflows

| Workflow | Existing Coverage | Sprint 111 Action | Planned Day |
|---|---|---|---|
| CSR compressed-first construction | `example_compressed_input` | Keep and preserve caller-owned-array demonstration. | Day 6 |
| CSC compressed-first construction | Header/docs mention it; no dedicated example path. | Add a compact CSC construction or conversion lane to `example_compressed_input`. | Day 6 |
| Direct solve | `example_basic_solve`, `example_analysis`, `example_ldlt`, QR examples. | Align examples README with solver guide. | Day 7 |
| Iterative solve | `example_iterative`, `example_ic_minres`, `example_matrix_free`. | Align examples README with solver guide and reuse boundaries. | Day 7 |
| Eigensolver | `example_eigs`. | Keep as eigensolver route; avoid overclaiming benchmark/performance proof. | Day 8 |
| SVD | `example_svd_lowrank`, `example_condition`. | Align examples README with SVD/condition route. | Day 8 |
| Reorder/fill | `example_colamd`, `example_analysis`. | Align examples README with symmetric-vs-COLAMD guidance. | Day 7 |
| Matrix Market load/use | `example_eigs` uses Matrix Market fixtures, docs have a snippet. | Add or identify a dedicated Matrix Market teaching route, preferably a small `example_matrix_market` if the Day 8 edit budget permits. | Day 8 |

## Update/Add Decision List

| Decision | Rationale | Guardrail |
|---|---|---|
| Update `example_compressed_input.c` instead of creating a second compressed example. | CSR and CSC construction are one concept; keeping both in one small example reduces example sprawl. | Keep the example copyable and avoid private headers. |
| Do not rewrite solver examples to start from compressed input yet. | Day 6 should first establish a clean compressed construction example; broad solver example rewrites would blur Day 7 scope. | Day 7 may update docs, not solver behavior, unless a narrow mismatch is found. |
| Add or identify a dedicated Matrix Market route on Day 8. | `example_eigs` uses Matrix Market but teaches eigensolvers, not Matrix Market ownership/error behavior. | If adding a compiled example, update `CMakeLists.txt` and validate with `make examples`. |
| Update `examples/README.md` after example decisions stabilize. | The README should point users to `docs/solver_selection.md` and the chosen workflow examples. | Keep maintainer proof language out of example docs. |

## Validation Plan

| Change Type | Required Validation |
|---|---|
| `example_compressed_input.c` update only | `make examples`; run `./build/example_compressed_input` if feasible; `git diff --check`. |
| New `examples/example_matrix_market.c` | Add CMake registration; `make examples`; run `./build/example_matrix_market` if feasible; `git diff --check`. |
| `examples/README.md` documentation-only update | `git diff --check`; trailing-whitespace scan. |
| Mixed example source and docs | `make examples`; focused example execution where feasible; `git diff --check`; trailing-whitespace scan. |
| CMake example registration change | Review `CMakeLists.txt`; configure/build CMake if the registration change is broad or if `make examples` is not sufficient for confidence. |

## Public-API Guardrails

- Examples must include only public headers from `include/` plus local example
  helper headers where already conventional.
- Examples must not include private `src/*_internal.h` headers.
- Examples must not describe a public Matrix I/O module or public builder API.
- Examples must not treat benchmark output as correctness proof.
- Examples must keep input ownership and cleanup visible where the workflow
  allocates or imports data.
- Examples must remain small teaching programs, not test or benchmark
  substitutes.

## Completion Criteria Status

- Each planned example supports a real user workflow.
- Example decisions are scoped to public API behavior.
- Validation expectations are clear before Day 6-8 edits begin.
- Example work avoids maintainer-only proof scaffolding.
