# Sprint 95 Day 9: Example Narrative Cleanup

## Purpose

Day 9 keeps examples focused on reusable public workflows. Example docs and
source comments should help readers choose and run the right binary without
pulling in sprint history, broad proof policy, or benchmark interpretation.

## Cleanup Batch

| Surface | Cleanup | Behavior preserved |
|---|---|---|
| `examples/README.md` | Replaced proof-style wording with measurement and owner-link language. | Example list, build command, executable names, and workflow routing unchanged. |
| `examples/README.md` | Changed eigensolver and installed-consumer descriptions to current workflow language. | `example_eigs` and `examples/cmake_example/` remain the same entry points. |
| `examples/example_analysis.c` | Reworded the analyze-once / factor-many header comment to describe reuse as a caller-visible contract. | Source code, API calls, and output unchanged. |
| `examples/example_eigs.c` | Reworded the header and residual-check comment to remove proof-style emphasis. | Source code, solver options, fixtures, and output unchanged. |

## User-Workflow Cross-Reference Map

| Reader need | Example surface | Owner for deeper context |
|---|---|---|
| Smallest direct solve | `example_basic_solve` | README solver chooser, tutorial direct-solve walkthrough |
| Stable-pattern direct reuse | `example_analysis` | Tutorial repeated-run section, public analysis headers |
| One-shot iterative solve | `example_iterative` | Tutorial iterative section, public iterative headers |
| Symmetric eigensolver usage | `example_eigs` | README eigensolver summary, public eigensolver headers |
| Rectangular QR / least squares | `example_least_squares`, `example_minnorm` | README QR section, `sparse_qr.h` |
| SVD and low-rank approximation | `example_svd_lowrank` | README SVD section, `sparse_svd.h` |
| Installed CMake consumer path | `examples/cmake_example/` | `INSTALL.md` |
| Measurement after adoption | benchmark binaries referenced from examples | `benchmarks/README.md` |
| Quality-policy interpretation | short owner link only | `docs/maintainer_guide.md` |

## Rename And Residual Notes

- No example files were renamed on Day 9.
- Current example names are already product-oriented enough for the public map:
  `example_basic_solve`, `example_analysis`, `example_iterative`,
  `example_eigs`, and `examples/cmake_example/` describe caller workflows.
- `example_analysis` is slightly abstract, but renaming it would churn Makefile
  targets, docs links, and user muscle memory. Defer unless a later proof-owner
  naming pass chooses a broader example target update.
- Keep benchmark names unchanged. They are measurement surfaces owned by
  `benchmarks/README.md`, not example teaching binaries.

## Validation Plan

Day 9 changed `.c` files, so the required quality chain is:

```bash
make format
make lint
make test
```

## Validation Result

- `make format && make lint && make test` passed.
- Example narrative scan passed: no `Sprint`, `sprint`, `Day`, `SPRINT_`,
  `bench_day`, `proof`, `proves`, `high-signal`, `dramatically`, `critical`,
  or `self-validating` matches remain in `examples/`.
- Trailing-whitespace scan passed for `examples/` and Sprint 95 planning
  artifacts.

## Day 9 Result

Examples now reinforce the cleaned README/tutorial adoption path: run a small
binary for the workflow, use the tutorial or headers for deeper API contracts,
and move to benchmark or maintainer docs only when that owner is needed.
