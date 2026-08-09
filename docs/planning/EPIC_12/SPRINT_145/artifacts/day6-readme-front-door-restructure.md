# Sprint 145 Day 6 README Front-Door Restructure

## Purpose

Simplify the README adoption path so a first-time user can build locally, run a
maintained solve, choose a solver, inspect diagnostics, and move to install or
advanced controls without reading maintainer-level evidence first.

## Changed Surface

| Surface | Change | Owner |
| --- | --- | --- |
| `README.md` `Start Here` | Replaced broad question bullets with a seven-step first-use path. | README front door |
| `README.md` `Adoption Map` | Routed first solve, data-first input, diagnostics, and install to maintained anchors. | README front door plus linked docs |
| `README.md` `Current Capabilities` | Added a short note that the capability inventory is reference material after first use. | README reference section |
| `README.md` `Quality` | Shortened CI/support-tier wording and routed details to `INSTALL.md` and the maintainer guide. | Install/platform and maintainer docs |
| `README.md` `Choose a Workflow` | Added `example_compressed_input`, examples ladder, and cookbook first-use links. | Solver-selection front door |
| `README.md` `Quick Start` | Clarified that the inline snippet is pasteable while the maintained runnable ladder lives in `examples/README.md`. | First local solve |
| `README.md` `Installation` | Routed platform-specific install detail to `INSTALL.md#start-here` and kept the README static-first summary compact. | Static-first install |

No source files, public headers, examples, build scripts, package metadata, or
CI workflow files were changed.

## Front-Door Shape

| Step | README route | Deeper owner |
| --- | --- | --- |
| Build locally | `make`, `make examples` | `examples/README.md#start-here` |
| First maintained solve | `./build/example_basic_solve` | `examples/README.md` |
| Data-first input | `example_compressed_input` | `docs/cookbook.md#first-use-ladder` |
| Solver choice | `README.md#choose-a-workflow` | `docs/solver_selection.md#choose-the-smallest-workflow` |
| Diagnostics | local return code, residual, convergence, rank, and benchmark context | `examples/README.md#diagnostics-handoff` |
| Install/downstream | README installation summary | `INSTALL.md#start-here` |
| Advanced controls | runtime/backend, benchmarks, reports, maintainer policy | `docs/maintainer_guide.md` and benchmark docs |

## Claim Boundary Review

| Area | Day 6 result |
| --- | --- |
| QR | Existing fixture-local QR corpus language remains intact; README still rejects broad QR and external-library parity claims. |
| Partial-SVD | Existing clustered/repeated diagonal fixture language remains intact; README still rejects broad repeated-spectrum, parity, performance, and state-of-the-art claims. |
| Runtime/backend | Advanced controls remain typed-option/local-diagnostics guidance, not API, ABI, package, platform parity, or portable performance claims. |
| Package/ABI | Installation text remains static-first and preserves shared-library deferral plus dynamic-loader/package-manager non-claims. |
| Platform | Linux, macOS, and Windows support tiers remain differentiated and are routed to `INSTALL.md` plus the maintainer guide for detail. |
| Reports/benchmarks | Report and benchmark rows remain local bounded evidence, not release proof or portable performance gates. |

## Validation

| Check | Result |
| --- | --- |
| README front-door anchor scan | Passed |
| Unsupported-claim scan for touched README wording | Passed: matches are explicit non-claims or support boundaries |
| `git diff --check` | Passed |
| `.c` / `.h` changed-file scan | Passed: no paths |

`make format && make lint && make test` was not required because Day 6 changed
only documentation.

## Day 6 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| README has a clear first-use path without burying users in maintainer detail. | Complete | `Start Here` now provides a seven-step build, solve, data, solver, diagnostics, install, and advanced-control path. |
| README still preserves earned support tiers and non-claims. | Complete | CI/support, partial-SVD, QR, runtime/backend, package, and install non-claims remain present or are routed to deeper owners. |
| Docs scans pass for touched README sections. | Complete | Anchor, unsupported-claim, whitespace, and changed-file scans passed. |

## Day 7 Handoff

Day 7 should apply the same simplification pattern to `INSTALL.md`: make the
static-first first-use install and downstream-consumer path easier to scan,
then route platform support tiers and package evidence into their deeper
sections without changing the static-only ABI posture.
