# Sprint 155 Day 1 Documentation Baseline

## Purpose

Day 1 establishes the Sprint 155 working baseline for tutorial alignment,
public-header cleanup, and API reference coherence. The baseline identifies the
source documents, previous-sprint evidence, earned claims, explicit non-claims,
artifact structure, and stop conditions that later days must preserve.

## Project-Plan Scope

Sprint 155 is `Tutorial, Header Cleanup & API Reference Coherence` from
`docs/planning/EPIC_13/PROJECT_PLAN.md`. Its project-plan deliverables are:

- aligned tutorial;
- cleaned selected public headers;
- API reference publication plan;
- declaration-preservation evidence;
- Sprint 156 closeout handoff.

Day 1 does not rewrite tutorial prose or edit public headers. It prepares the
evidence map that Day 2 and later implementation days must use.

## Prerequisite Evidence

| Source | Relevant Evidence For Sprint 155 |
| --- | --- |
| `docs/planning/EPIC_12/SPRINT_145/RETROSPECTIVE.md` | First-use ladder, README/INSTALL/examples/cookbook/solver-selection alignment, four cleaned high-impact public headers, and residual tutorial/header debt. |
| `docs/planning/EPIC_12/SPRINT_146/RETROSPECTIVE.md` | Epic 12 residual IDs `R8` and `R9`, final claim boundaries, and conservative state-of-the-art decision. |
| `docs/planning/EPIC_13/SPRINT_148/RETROSPECTIVE.md` | Windows staged test portability closure and platform support-tier carry-forward. |
| `docs/planning/EPIC_13/SPRINT_149/RETROSPECTIVE.md` | Windows install-validation parity decision and remaining Windows package-surface boundaries. |
| `docs/planning/EPIC_13/SPRINT_150/RETROSPECTIVE.md` | QR maintained corpus family expansion and fixture-local QR evidence boundaries. |
| `docs/planning/EPIC_13/SPRINT_151/RETROSPECTIVE.md` | Partial-SVD maintained corpus family expansion and fixture-local partial-SVD evidence boundaries. |
| `docs/planning/EPIC_13/SPRINT_152/RETROSPECTIVE.md` | Generated report freshness publication and generated-row interpretation. |
| `docs/planning/EPIC_13/SPRINT_153/RETROSPECTIVE.md` | Shared-library ABI product decision and static-first package contract. |
| `docs/planning/EPIC_13/SPRINT_154/artifacts/day14-closeout-sprint155-handoff.md` | First narrow `qr-minnorm` comparison lane and its explicit non-parity boundary. |

## Documentation Owner Inventory

| Surface | Current Role | Sprint 155 Use |
| --- | --- | --- |
| `README.md` | Short first-use front door, capability summary, workflow chooser, support-tier summary, and local command map. | Day 2 source-of-truth for tutorial routing and high-level claims. |
| `INSTALL.md` | Static-first install contract, downstream consumer setup, support split, and platform support details. | Source-of-truth for install/package/platform wording in tutorial and API docs. |
| `examples/README.md` | Runnable example ladder and diagnostics handoff. | Source-of-truth for maintained first-solve and example references. |
| `docs/cookbook.md` | Data-first CSR, CSC, Matrix Market, solver, benchmark, and report recipes. | Source-of-truth for data-input tutorial flow. |
| `docs/solver_selection.md` | Solver-family decision tree and diagnostics interpretation. | Source-of-truth for solver-choice tutorial text. |
| `docs/tutorial.md` | Fuller learning path after README. | Primary Day 2 audit and Days 3-5 rewrite target. |
| `docs/maintainer_guide.md` | Maintainer policy, report interpretation, package/ABI boundaries, and validation commands. | Source-of-truth for claim boundaries, report evidence, and API/reference maintenance guidance. |
| `benchmarks/README.md` | Benchmark/report commands and local measurement interpretation. | Reference surface for diagnostics/report sections only. |
| `docs/matrix_market.md` | Matrix Market format, ownership, duplicate-entry, pattern, and errno contract. | Source-of-truth for Matrix Market tutorial references. |
| `docs/api/html/` | Current generated Doxygen API reference output. | API reference baseline; Days 10-11 decide guidance/publication plan. |

## Public Header Inventory

The installed public header surface currently contains `19` headers/templates
under `include/`:

- `include/sparse_analysis.h`
- `include/sparse_bidiag.h`
- `include/sparse_cholesky.h`
- `include/sparse_csr.h`
- `include/sparse_dense.h`
- `include/sparse_eigs.h`
- `include/sparse_ic.h`
- `include/sparse_ilu.h`
- `include/sparse_iterative.h`
- `include/sparse_ldlt.h`
- `include/sparse_lu.h`
- `include/sparse_lu_csr.h`
- `include/sparse_matrix.h`
- `include/sparse_qr.h`
- `include/sparse_reorder.h`
- `include/sparse_svd.h`
- `include/sparse_types.h`
- `include/sparse_vector.h`
- `include/sparse_version.h.in`

Sprint 145 already cleaned these high-impact public headers:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_qr.h`
- `include/sparse_svd.h`

Day 6 should therefore select the next cleanup batch from the remaining
headers, unless the Day 2-5 tutorial work identifies a stronger user-facing
reason to revisit one of the previously cleaned headers for a small,
declaration-preserving correction.

## Earned Claims To Preserve

The tutorial, selected public headers, and API reference plan may rely on these
bounded claims:

- local build-tree first-use workflow is documented through README and
  examples;
- static-first install and downstream consumption are maintained through
  Make/CMake install surfaces and package metadata;
- Linux is the strongest reviewed platform source of truth;
- macOS has reviewed static-first install/export proof;
- Windows has reviewed CMake-first support, promoted staged test coverage, and
  reviewed CMake install/downstream validation for the maintained static-first
  package surface;
- QR maintained corpus evidence is fixture-local and owned by tests and corpus
  reports;
- partial-SVD maintained corpus evidence is fixture-local and owned by tests
  and corpus reports;
- generated report-index freshness is command- and family-specific;
- the `qr-minnorm` comparison lane is a narrow local external-process dense
  reference comparison for `qr_underdetermined_minnorm_2x4`;
- public backend/runtime controls are local workflow controls unless a later
  sprint earns wider claims.

## Non-Claims And Stop Conditions

Sprint 155 must not introduce or imply:

- unqualified state-of-the-art sparse linear algebra status;
- broad QR, SVD, partial-SVD, eigensolver, or direct-solver correctness beyond
  reviewed fixtures and tests;
- NumPy, SciPy, LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, CHOLMOD, ARPACK,
  or ecosystem parity from the narrow Sprint 154 comparison;
- portable performance superiority or backend superiority from local benchmark,
  sentinel, or comparison rows;
- package-manager support;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader compatibility;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad Windows parity;
- generated report freshness without the explicit selected freshness command.

Public header edits have additional stop conditions:

- no declaration spelling changes;
- no signature changes;
- no typedef, enum, macro, or struct-field changes;
- no installed-header add/remove/rename behavior;
- no exported symbol changes;
- no include-guard churn unless explicitly justified and preserved;
- no generated API reference refresh treated as proof without a publication
  policy.

## Day 2 Audit Inputs

Day 2 should audit `docs/tutorial.md` against these source documents:

- `README.md`
- `INSTALL.md`
- `examples/README.md`
- `docs/cookbook.md`
- `docs/solver_selection.md`
- `docs/maintainer_guide.md`
- `docs/matrix_market.md`
- `benchmarks/README.md`
- selected public headers under `include/`
- generated API reference entry points under `docs/api/html/`
- Sprint 154 comparison handoff if tutorial text mentions comparison or report
  evidence

The audit should classify findings into:

- stale build/link/install guidance;
- duplicated content better owned by README, INSTALL, examples, cookbook, or
  solver-selection docs;
- missing first-use flow steps;
- unclear diagnostics or return-code guidance;
- stale solver/support-tier claims;
- stale report/comparison wording;
- advanced-control content that belongs in API reference rather than tutorial;
- public-header wording that should be considered during Day 6 selection.

## Day 1 Completion Check

- Sprint 155 working notes exist.
- Sprint 155 artifact directory exists.
- Day 1 documentation baseline exists.
- Documentation owner surfaces are inventoried.
- Public-header cleanup surface is inventoried.
- Earned claims and non-claims are recorded.
- Day 2 audit inputs are explicit.
