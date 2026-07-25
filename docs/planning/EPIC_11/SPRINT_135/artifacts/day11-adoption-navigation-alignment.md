# Sprint 135 Day 11 - Adoption Navigation Alignment

## Purpose

Align README, tutorial, install, cookbook, reference, report, and maintainer
navigation after the algorithm split and cookbook/report-index additions. The
goal is a predictable first-use path without burying historical or maintainer
surfaces.

## Navigation Owner Map

| Reader need | Primary surface | Secondary surface |
|---|---|---|
| Smallest local build and solve | `README.md` quick start | `examples/README.md` |
| Problem-shape decision tree | `README.md` Choose a Workflow | `docs/solver_selection.md` |
| CSR, CSC, or Matrix Market first-use recipes | `docs/cookbook.md` | maintained examples linked from cookbook |
| Installed consumer setup | `README.md` Installation summary | `INSTALL.md` |
| Benchmark/report interpretation | `benchmarks/README.md` | generated index/manifest artifacts |
| Current algorithm behavior | `docs/algorithm.md` | public headers |
| Historical measurement notes | `docs/algorithm_history.md` | Sprint planning artifacts |
| Maintainer quality policy | `docs/maintainer_guide.md` | Sprint planning artifacts |

## Public Documentation Changes

### `README.md`

Added an `Adoption Map` near the front door. The map keeps first-use routing in
one place and points to:

- quick start
- solver selection
- compressed-first cookbook
- installation
- benchmark/report interpretation
- algorithm reference
- algorithm history
- maintainer guide

Also updated the Start Here cookbook description so it covers the full Day 9
cookbook scope, not just direct/iterative/Matrix Market workflows.

### `docs/tutorial.md`

Added a `Documentation Map` after the getting-started handoff. This keeps the
tutorial as a learning path while making neighboring owner docs easy to find:

- solver selection
- cookbook
- examples
- install
- benchmarks/report indexes
- algorithm reference
- algorithm history
- maintainer guide

### `docs/solver_selection.md`

Updated the front-matter handoff order:

1. compressed-first cookbook
2. runnable examples
3. public headers
4. install docs
5. benchmark/report docs

### `examples/README.md`

Updated start-here bullets so compressed-input readers see cookbook handoff
coverage for direct, iterative, SVD, eigensolver, and benchmark workflows. Also
added a benchmark/report interpretation handoff for readers who have already
chosen an API workflow.

### `docs/cookbook.md`

Added install, current algorithm reference, and historical appendix handoffs at
the top. The cookbook still remains task-oriented and does not become the
algorithm reference or install guide.

### `benchmarks/README.md`

Added a cookbook handoff in the introduction so benchmark readers can return
to compressed-first workflow guidance before measuring.

### `INSTALL.md`

Added a cookbook handoff for CSR, CSC, or Matrix Market first-use recipes before
installation. Install and downstream-consumer details remain owned by
`INSTALL.md`.

## Maintainer And Historical Placement

Historical and maintainer surfaces remain findable, but they are no longer
default first-use routes:

- `docs/algorithm.md` points to `docs/algorithm_history.md` for historical
  measurements.
- README and tutorial route current behavior to `docs/algorithm.md` first.
- quality-policy interpretation remains in `docs/maintainer_guide.md`.
- benchmark/report interpretation remains in `benchmarks/README.md`.

## Package And Platform Support Alignment

Day 11 did not change package, ABI, or platform support wording. The navigation
edits preserve Sprint 133-134 boundaries:

- installed consumer details stay in `INSTALL.md`
- static-first install truth remains unchanged
- no package-manager availability claim was added
- no shared-library or dynamic-ABI support claim was added
- no Windows, macOS, or Linux support-tier claim was expanded

## Validation Plan

Documentation-only validation for this batch:

- `git diff --check`
- trailing-whitespace scan on touched docs and Sprint 135 artifacts
- navigation-link scan for cookbook, tutorial, solver selection, install,
  benchmarks, algorithm reference/history, examples, and maintainer guide
- unsupported-claim scan for package, ABI, platform, and performance wording
- `git diff --name-only -- '*.c' '*.h'` to confirm no code-day quality gate is
  required

## Completion Criteria

- README and tutorial entry points lead to simplified adoption paths
- historical and maintainer material remains findable without being the default
  first-use path
- package/platform support wording remains aligned with Sprint 133-134 truth
