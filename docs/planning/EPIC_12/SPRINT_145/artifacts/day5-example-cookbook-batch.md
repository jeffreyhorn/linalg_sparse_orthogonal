# Sprint 145 Day 5 Example And Cookbook Batch

## Purpose

Implement the maintained example and cookbook front-door updates selected on
Day 4, keeping the change docs-first and avoiding example source changes unless
necessary.

## Changed Surfaces

| Surface | Change | Proof owner |
| --- | --- | --- |
| `examples/README.md` | Added a first-use ladder, expected first commands, problem-shape branch table, advanced-only example routing, and diagnostics handoff. | `make examples-build`; focused example smoke runs. |
| `docs/cookbook.md` | Added a first-use ladder, linked build-tree and installed-consumer examples, clarified data-to-solver routing, and added simple construction/direct diagnostics guidance. | Markdown scans; linked maintained examples. |

No example `.c` files, public headers, package scripts, CMake files, or
Makefile targets were changed.

## Implemented First-Use Ladder

| Rung | Public route | Notes |
| --- | --- | --- |
| 1. Build examples | `make examples` or `make examples-build` | Maintains existing build target ownership. |
| 2. Smallest local success | `./build/example_basic_solve` | First direct solve with residual/solution output. |
| 3. Data already in CSR/CSC | `./build/example_compressed_input` | Confirms arrays are validated and copied, not adopted. |
| 4. Solver branch | examples README table plus `docs/solver_selection.md` | Routes to repeated direct, iterative, QR, SVD, Matrix Market, and installed consumer paths. |
| 5. Diagnostics | examples README and cookbook diagnostics handoff | Keeps diagnostics local to the workflow that produced them. |
| 6. Installed consumer | `examples/cmake_example/` plus `INSTALL.md` | Static-first downstream CMake route, not a package-manager or shared-library claim. |

## Advanced-Only Routing Preserved

| Surface | Current disposition |
| --- | --- |
| `example_eigs` | Maintained symmetric eigensolver example, but not first local success. |
| `example_matrix_free` | Maintained callback-operator example after normal matrix-shell adoption. |
| `example_colamd` | Maintained ordering/fill example after solver choice. |
| `example_ic_minres` | Maintained preconditioner-assumption example after iterative basics. |
| benchmark binaries | Measurement surfaces after API workflow choice, not adoption examples. |

## Claim Boundary Review

| Area | Day 5 result |
| --- | --- |
| QR | Wording still treats QR examples as teaching paths and preserves external-library/performance non-claims. |
| SVD and partial-SVD | Wording routes rank/condition/low-rank users to examples without broad parity or performance claims. |
| Package/install | Installed CMake consumer remains static-first and separate from local build-tree examples. |
| Platform | No Linux/macOS/Windows support-tier wording changed. |
| Runtime/backend | Advanced controls remain behind links; no portable performance claim added. |
| Reports/benchmarks | Benchmarks remain measurement surfaces, not example or portable evidence. |

## Validation

| Check | Result |
| --- | --- |
| `git diff --check` | Passed |
| `.c` / `.h` changed-file scan | Passed: no paths |
| example/cookbook section scan | Passed |
| `make examples-build` | Passed: built 14 example binaries, no execution |

`make format && make lint && make test` was not required because no `.c` or
`.h` files changed. Day 5 changed only documentation, but the maintained
example build owner was still run because the touched docs route users through
the example ladder.

## Day 5 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Examples are runnable or explicitly documented as docs-only snippets. | Complete | Day 5 changed documentation only and points to existing maintained runnable examples. |
| Example wording does not widen numerical, package, or platform claims. | Complete | Claim boundary review preserves QR, SVD, package, platform, runtime, and benchmark limits. |
| Focused validation for touched examples passes. | Complete | Documentation checks passed; no example source changed, so build ownership remains unchanged. |

## Day 6 Handoff

Day 6 should restructure README around the implemented example/cookbook ladder:
first local build, `example_basic_solve`, compressed input, solver branch,
diagnostics, install handoff, and advanced controls. Keep dense capability,
performance, report, and maintainer evidence below the front door or behind
links.
