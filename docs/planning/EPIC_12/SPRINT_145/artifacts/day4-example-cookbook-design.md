# Sprint 145 Day 4 Example And Cookbook Design

## Purpose

Specify the maintained example and cookbook updates for the Sprint 145
front-door workflow before implementation.

## Maintained Example Selection Matrix

| Priority | Workflow rung | Selected example or doc | Current role | Day 5 action |
| ---: | --- | --- | --- | --- |
| 1 | Smallest local success | `examples/example_basic_solve.c`, `examples/README.md` | Smallest one-shot LU solve and residual example. | Make this the first rung in the example ladder and keep command/output expectations obvious. |
| 2 | Data already in CSR/CSC | `examples/example_compressed_input.c`, `docs/cookbook.md` | Demonstrates caller-owned compressed input copied into the public matrix shell. | Tighten cookbook and example routing so compressed input becomes the second front-door rung. |
| 3 | Problem-shape branch | `docs/solver_selection.md`, `docs/cookbook.md`, `examples/README.md` | Solver choice exists in multiple places with different depth. | Link the same compact branch table to maintained examples without duplicating full evidence text. |
| 4 | Diagnostics | `examples/example_compressed_input.c`, `examples/example_iterative.c`, `docs/cookbook.md` | Diagnostics are present but not a named first-use step. | Add a short diagnostics handoff covering `sparse_err_t`, NULL-return constructors, residuals, convergence, and status fields. |
| 5 | Repeated direct lifecycle | `examples/example_analysis.c` | Strong adoption example for stable-pattern repeated direct solves. | Keep as opt-in after first one-shot success; do not make it the first example. |
| 6 | QR least-squares/minimum-norm | `examples/example_least_squares.c`, `examples/example_minnorm.c`, `docs/cookbook.md` | Maintained QR teaching examples with bounded corpus evidence language. | Route rectangular/rank-sensitive users here while preserving fixture-local QR claims. |
| 7 | SVD/low-rank/condition | `examples/example_svd_lowrank.c`, `examples/example_condition.c`, `docs/cookbook.md` | Maintained SVD and conditioning examples. | Route rank/condition/low-rank users here without implying broad SVD or performance parity. |
| 8 | Installed downstream consumer | `examples/cmake_example/`, `INSTALL.md` | Maintained installed CMake consumer path. | Keep separate from build-tree examples and link from README/INSTALL as the installed-consumer proof. |

## Expected Command And Output Plan

| Example path | Build command | Run command | Expected output shape | Validation owner |
| --- | --- | --- | --- | --- |
| `examples/example_basic_solve.c` | `make examples-build` or `make examples` | `./build/example_basic_solve` | Prints a successful solve with residual/solution values and exits `0`. | Day 5 example smoke check; `make examples-build` if source changes. |
| `examples/example_compressed_input.c` | `make examples-build` or `make examples` | `./build/example_compressed_input` | Shows CSR/CSC construction and a successful one-shot solve. | Day 5 example smoke check; cookbook link owner. |
| `examples/example_analysis.c` | `make examples-build` or `make examples` | `./build/example_analysis` | Shows analyze/factor/solve/refactor lifecycle success. | Day 5 or Day 11 focused run if docs change its front-door role. |
| `examples/example_iterative.c` | `make examples-build` or `make examples` | `./build/example_iterative` | Shows iterative convergence with and without preconditioning. | Day 8 diagnostics/solver handoff owner. |
| `examples/example_least_squares.c` | `make examples-build` or `make examples` | `./build/example_least_squares` | Shows QR least-squares solution and residuals. | Day 5 or Day 8 QR route validation. |
| `examples/example_minnorm.c` | `make examples-build` or `make examples` | `./build/example_minnorm` | Shows underdetermined QR minimum-norm solve. | Day 5 or Day 8 QR route validation. |
| `examples/example_svd_lowrank.c` | `make examples-build` or `make examples` | `./build/example_svd_lowrank` | Shows singular values, rank/condition, and low-rank summaries. | Day 5 or Day 8 SVD route validation. |
| `examples/example_condition.c` | `make examples-build` or `make examples` | `./build/example_condition` | Shows conditioning comparison and small solve result. | Day 8 diagnostics/SVD route validation. |
| `examples/cmake_example/` | CMake install/downstream flow from `INSTALL.md` | built installed example executable | Prints installed package example output including `OK`. | Install/downstream validation only if package commands change. |

The Day 5 implementation should avoid changing example source unless the
front-door wording cannot be made clear from `examples/README.md` and
`docs/cookbook.md`. If any `.c` file changes, the full C quality gate applies.

## Cookbook Update Plan

| Cookbook section | Current role | Planned update |
| --- | --- | --- |
| `Start From Your Data` | Good data-first table. | Keep as the cookbook front door and add explicit link back to the example ladder if needed. |
| `Direct Solves From Compressed Input` | Strong CSR/CSC direct path. | Make it the canonical compressed-input rung after `example_basic_solve`. |
| `Iterative Solves From Compressed Input` | Good iterative route. | Add concise diagnostics handoff to solver result fields if needed. |
| `Matrix Market Load/Use` | Good file-input route. | Keep as a data-first branch rather than first local success. |
| `SVD and Low-Rank Workflows` | Accurate SVD/partial-SVD route and non-claims. | Preserve bounded partial-SVD wording and route to `example_svd_lowrank`. |
| `Symmetric Eigensolver Workflows` | Advanced but useful. | Keep after SVD and direct/iterative paths; do not promote to first rung. |
| `Measure After Choosing the API Workflow` | Correct benchmark/report boundary. | Keep advanced and link from README only after API workflow choice. |

## README And INSTALL Link Plan

| Source | Link target | Purpose |
| --- | --- | --- |
| `README.md` Start Here | `examples/README.md#start-here` | Send first-use readers to the maintained runnable ladder. |
| `README.md` workflow chooser | `docs/solver_selection.md#choose-the-smallest-workflow` | Route solver choice to detailed problem-shape guidance. |
| `README.md` CSR/CSC/Matrix Market note | `docs/cookbook.md#start-from-your-data` | Route data-first users to cookbook recipes. |
| `README.md` install summary | `INSTALL.md#start-here` | Keep install detail out of the first local solve path. |
| `INSTALL.md` Start Here | `examples/README.md#installed-consumer-example-examplescmake_example` | Route installed-consumer users to the CMake example after install detail. |
| `INSTALL.md` CMake consumer section | `examples/cmake_example/` | Keep downstream CMake proof concrete and static-first scoped. |

## Advanced-Only Example List

These remain valuable but should not become the first-use ladder:

- `example_eigs`: symmetric eigensolver, shift-invert, and preconditioned
  LOBPCG are advanced solver workflows.
- `example_matrix_free`: callback operators are useful after the normal matrix
  shell path is understood.
- `example_colamd`: ordering/fill comparison belongs after solver choice.
- `example_ic_minres`: preconditioner-assumption matching is a second-step
  iterative workflow.
- benchmark binaries under `benchmarks/`: measurement surfaces, not adoption
  examples.

Advanced-only does not mean hidden. These examples should remain linked from
solver-selection, cookbook, examples README, and benchmark handoff sections
where their assumptions are already named.

## Example Claim Boundaries

| Example family | Keep | Do not imply |
| --- | --- | --- |
| Basic direct examples | One-shot public API adoption and residual checks. | Broad direct-solver optimality or performance. |
| Compressed input | CSR/CSC arrays are validated and copied into a public matrix shell. | Zero-copy ownership or package/platform proof. |
| Repeated direct lifecycle | Same-pattern analyze/factor/refactor workflow. | Hidden structural rebuild or universal speedup. |
| QR examples | Rectangular least-squares and minimum-norm teaching paths. | Broad QR, external-library parity, platform, or performance claims. |
| SVD examples | Full SVD, condition, and low-rank workflows. | Broad partial-SVD performance, repeated-spectrum, or external-library parity. |
| Installed CMake example | Static-first CMake downstream consumer proof. | Package-manager, shared-library, ABI, or Windows Makefile/`pkg-config` parity. |

## Day 5 Implementation Checklist

1. Keep the first-use ladder in `examples/README.md` concise and ordered:
   basic solve, compressed input, solver branch, diagnostics, installed
   consumer.
2. Add or adjust cookbook links only where they support that ladder.
3. Avoid editing example `.c` files unless the current runnable examples lack
   required output or diagnostics.
4. Preserve QR, partial-SVD, package, platform, benchmark, and report
   non-claims.
5. Run `git diff --check` for docs-only changes.
6. If example `.c` files change, run `make format && make lint && make test`
   before proceeding.
7. If package install commands change, run `bash tests/test_install.sh` and
   `bash tests/test_cmake_install.sh`.

## Day 4 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected example changes are scoped to first-use adoption. | Complete | Selection matrix chooses a compact ladder and marks advanced-only examples separately. |
| Every new or revised example has a validation owner. | Complete | Expected command/output plan maps each selected example to build/run and validation owner. |
| Advanced examples do not make the front door dense again. | Complete | Advanced-only list keeps eigensolver, matrix-free, ordering, preconditioner, and benchmarks behind solver-specific links. |

## Day 5 Handoff

Day 5 should implement the example/cookbook batch with the smallest docs-first
change set. The likely implementation surfaces are `examples/README.md` and
`docs/cookbook.md`; avoid touching `.c` examples unless a runnable command or
diagnostic expectation is actually missing.
