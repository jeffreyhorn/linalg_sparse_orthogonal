# Sprint 145 Day 3 High-Level Workflow Design

## Purpose

Design the concise high-level adoption front door before editing README,
INSTALL, examples, cookbook, solver guidance, or public headers.

## Canonical First-Use Workflow

Sprint 145 should make this the public adoption path:

1. **Build or install**
   - Local trial: `make`, then `make examples`.
   - Unix static install: `make install PREFIX=...`, then `pkg-config`.
   - CMake or Windows consumer: CMake configure/build/install and
     `find_package(Sparse)`.
2. **Choose a solver**
   - Square general: LU.
   - Symmetric positive-definite: Cholesky or CG.
   - Symmetric indefinite: LDLT or MINRES.
   - Rectangular, least-squares, or rank-sensitive: QR.
   - Rank, condition, pseudoinverse, or low-rank: SVD APIs.
   - Symmetric eigenpairs: `sparse_eigs_sym(...)`.
3. **Run a maintained solve example**
   - Start with `example_basic_solve`.
   - Use `example_compressed_input` when caller data is CSR or CSC.
   - Use `example_analysis` only after the same sparsity pattern repeats.
4. **Inspect diagnostics**
   - Check `sparse_err_t` or NULL-return diagnostics.
   - Inspect solver result/status fields where available.
   - Treat residuals, stagnation, breakdown, rank, and convergence fields as
     workflow-local diagnostics.
5. **Escalate to advanced controls**
   - Move to repeated-run handles, runtime/backend controls, benchmarks,
     report indexes, or maintainer evidence only after the first workflow is
     working.

## Content Routing Map

| Content | Front-door home | Deeper reference | Routing rule |
| --- | --- | --- | --- |
| First local build and one solve | `README.md` | `examples/README.md` | README should show the shortest local success and link to runnable examples. |
| Static install and downstream consumers | `INSTALL.md` | `tests/test_install.sh`, `tests/test_cmake_install.sh`, maintainer guide | README links to INSTALL; INSTALL owns package mechanics and support split. |
| Solver choice | short README table | `docs/solver_selection.md` | README should provide a compact first choice; solver-selection owns detail and evidence boundaries. |
| CSR/CSC/Matrix Market starting points | `docs/cookbook.md` | examples and public headers | Cookbook owns data-first recipes and links back to examples. |
| Runnable usage | `examples/README.md` | individual example `.c` files | Examples README should become the maintained workflow ladder. |
| Tutorial narrative | `docs/tutorial.md` | solver-selection, cookbook, examples | Tutorial should teach in order, not serve as the first decision tree. |
| Diagnostics | README/cookbook/examples brief handoff | public headers and solver docs | Public front door names diagnostics; deeper docs explain fields and options. |
| Runtime/backend controls | README advanced section | maintainer guide, benchmark docs | Keep advanced controls discoverable but outside the first-use sequence. |
| Benchmarks and reports | benchmark docs | report-family manifests and generated reports | Benchmarks remain measurement surfaces, not adoption examples or portable claims. |
| API contracts | `include/*.h` | docs and maintainer guide | Headers preserve contracts; docs absorb tutorial/history/policy detail. |
| Quality/support policy | maintainer guide | sprint artifacts | Do not force first-use readers through maintainer policy. |

## Front-Door Shape

The public front door should be short enough to scan in one pass:

1. What this library is.
2. Start here:
   - build locally;
   - run the smallest example;
   - choose a solver if the problem is not general square;
   - install only when a downstream project needs it.
3. Workflow chooser table.
4. Links to cookbook, examples, INSTALL, solver-selection, benchmark docs, and
   maintainer guide.
5. Current capabilities and evidence after the adoption path, not before it.

## Example And Cookbook Design Checklist

| Workflow rung | Maintained example or doc owner | Design requirement |
| --- | --- | --- |
| Smallest local solve | `examples/example_basic_solve.c`, `examples/README.md` | Keep as the first runnable success. |
| Data already in CSR/CSC | `examples/example_compressed_input.c`, `docs/cookbook.md` | Show compressed arrays are validated and copied, not adopted. |
| Solver branch | `docs/solver_selection.md`, `docs/cookbook.md` | Keep the problem-shape table compact and route to examples. |
| Diagnostics | cookbook, examples, public headers | Add one clear diagnostics step without duplicating every result field. |
| Repeated direct lifecycle | `examples/example_analysis.c` | Present as opt-in after stable sparsity pattern is known. |
| QR least-squares/minimum-norm | `examples/example_least_squares.c`, `examples/example_minnorm.c` | Preserve bounded QR claim wording and avoid broad external-library parity. |
| SVD/low-rank/condition | `examples/example_svd_lowrank.c`, `examples/example_condition.c` | Present as maintained examples, not broad performance or parity evidence. |
| Installed CMake consumer | `examples/cmake_example/`, `INSTALL.md` | Keep separate from build-tree examples and static-first package scoped. |
| Benchmark/report handoff | `benchmarks/README.md` | Keep after API workflow choice and preserve local-measurement wording. |

## Public Header Cleanup Rules

Day 9-10 header work should follow these rules:

1. Do not change declarations, types, enum values, callback signatures,
   ownership semantics, or error semantics as part of comment cleanup.
2. Keep contract-critical details in headers: ownership, mutation, allocation,
   failure modes, and result-field semantics.
3. Move or shorten maintainer-only history, benchmark interpretation, and
   broad evidence commentary when it distracts from the public contract.
4. Preserve explicit ABI-break warnings unless a separate versioning decision
   removes them.
5. Preserve QR and partial-SVD bounded evidence language; do not replace it
   with broad parity wording.
6. Treat any `.h` edit as C/header work requiring
   `make format && make lint && make test`.

## Claim Boundary Rules

| Area | Allowed front-door wording | Disallowed overread |
| --- | --- | --- |
| QR | QR supports rectangular, least-squares, minimum-norm, and rank-sensitive workflows with bounded maintained evidence. | Broad QR, SuiteSparse, LAPACK, NumPy, SciPy, platform, performance, or state-of-the-art parity. |
| Partial-SVD | Partial-SVD has maintained edge-case, residual, projector, convergence, and fail-closed evidence. | Broad repeated-spectrum, sparse-output, convergence-rate, performance, or external-library parity. |
| Reports | Source-controlled report rows identify schema and proof owners; generated reports must be regenerated for fresh values. | Treating report rows as fresh benchmark or corpus run output. |
| Runtime/backend | Runtime/backend controls and sentinels guide local governance. | Portable performance or backend superiority claims. |
| Package | Maintained package surface is static-first. | Shared-library, dynamic ABI, runtime-loader, package-manager, or static/shared selector support. |
| Platform | Linux strongest reviewed baseline; macOS reviewed static-first install/export proof; Windows CMake-first and narrower. | Windows Makefile/`pkg-config` parity, Windows reviewed install-validation parity, staged Windows test closure, or broad macOS platform parity. |

## Validation Plan

| Future change type | Required checks |
| --- | --- |
| Markdown-only routing changes | `git diff --check`, stale wording scans, unsupported-claim scans, and link/reference scans available locally. |
| README/INSTALL package command changes | Markdown checks plus `bash tests/test_install.sh` and/or `bash tests/test_cmake_install.sh` when install/downstream commands change. |
| Example source changes | Build the touched example or `make examples`; if `.c` changes, run `make format && make lint && make test`. |
| Cookbook/tutorial snippets that mirror examples | Build or smoke-test the referenced maintained example where feasible. |
| Report row changes | `python3 scripts/validate_corpus_schema.py` and relevant `scripts/normalize_report_index.py` checks. |
| Public header cleanup | `make format && make lint && make test`, plus focused API/claim-boundary review. |

## Implementation Order

1. Day 4 should select the exact example/cookbook ladder and expected
   validation owners.
2. Day 5 should implement only the selected example/cookbook batch.
3. Day 6 should restructure README around the workflow design.
4. Day 7 should restructure INSTALL around the static-first install and
   downstream-consumer path.
5. Day 8 should consolidate solver-selection and diagnostics wording.
6. Day 9-10 should decide and execute the smallest safe public-header cleanup
   batch.
7. Day 11-14 should validate, map claims, close residual debt, and prepare the
   Sprint 146 closeout handoff.

## Day 3 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Workflow design covers build/install, solver selection, solve execution, diagnostics, and advanced controls. | Complete | Canonical workflow and content routing map cover all five stages. |
| Design preserves Sprint 139-144 claim boundaries. | Complete | Claim boundary rules preserve QR, partial-SVD, report, runtime, package, and platform limits. |
| Implementation can proceed without inventing new solver/package/platform claims. | Complete | Example checklist, header rules, validation plan, and implementation order define scoped next steps. |

## Day 4 Handoff

Day 4 should turn this design into an example/cookbook selection matrix. The
highest-value candidate ladder is:

1. `example_basic_solve`;
2. `example_compressed_input`;
3. solver-selection branch to QR/SVD/iterative/repeated-run examples;
4. a diagnostics step;
5. `examples/cmake_example/` for installed downstream consumption.
