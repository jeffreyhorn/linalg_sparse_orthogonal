# Day 4: Quick Reference Design

**Sprint:** 205 - Support Matrix and Adoption Quick-Reference Consolidation  
**Theme:** Design the compact problem-shape to workflow table for common local
and installed use cases.  
**Time estimate:** 12 hours  
**Branch:** `sprint-205`  
**Base commit:** `9a2b4ff6`

## Purpose

Day 4 turns the Day 2 public-doc audit and Day 3 maintainer/report audit into
an implementation-ready quick-reference design. The quick reference is a
compact user routing table. It must help users pick the first maintained
workflow while linking to authoritative support/readiness and proof surfaces
instead of copying long evidence or non-claim paragraphs.

## Placement Decision

| Candidate | Decision | Rationale |
| --- | --- | --- |
| README | Link to the quick reference, but do not host the full table. | README is already the front door and contains repeated proof-owner detail. Adding the full table there would increase its length and duplicate cookbook/solver-selection routing. |
| `docs/cookbook.md` | Selected host. | The cookbook already owns data-first adoption and first-use recipes. A compact "problem-shape quick reference" near the first-use ladder fits naturally and can route to examples, solver selection, INSTALL, API reference, and benchmarks. |
| `docs/solver_selection.md` | Detailed owner, not compact host. | It already has the full solver decision tree and selected-evidence caveats. The quick reference should link into it for detail. |
| `INSTALL.md` | Support truth owner, not quick-reference host. | It owns install/support readiness and should not become the first solver workflow guide. |
| New document | Not selected for Sprint 205. | A new doc would add another routing surface and increase link maintenance. The audit found the existing cookbook can absorb the compact table. |

## Implementation Route

Day 6 should implement the quick reference as:

1. A new compact section in `docs/cookbook.md` near `## First-Use Ladder`.
2. Short links from README Start Here/Adoption Map, `docs/tutorial.md`, and
   `examples/README.md`.
3. No copied selected-target row lists, benchmark methodology details,
   residual queue details, or maintainer-only proof vocabulary.
4. Link-only routing to `INSTALL.md#support-readiness-matrix` for current
   support/readiness status.

## Row Vocabulary

Use user-facing labels:

- **First workflow**: the initial API family or setup path to try.
- **Run or read first**: the concrete example or owner doc.
- **Support/readiness route**: where current support status lives.
- **Boundary**: short retained non-claim or caution.

Avoid raw maintainer/report labels in the quick-reference table:

- `support_tier`
- `claim_boundary`
- `freshness_policy`
- `non_claims`
- selected target IDs
- generated report row IDs
- residual queue IDs

Those terms may appear in linked maintainer, benchmark, or manifest surfaces,
but the quick reference should stay user-oriented.

## Problem-Shape Categories

| Category | First workflow | Run or read first | Support/readiness route | Boundary |
| --- | --- | --- | --- | --- |
| Small local build and first solve | Local build-tree example | `examples/README.md#start-here`, `example_basic_solve` | `INSTALL.md#support-readiness-matrix` for support status after first solve | Not an install, package-manager, performance, or broad platform claim. |
| Caller-owned CSR/CSC arrays | Compressed input constructors, then solver selection | `docs/cookbook.md#start-from-your-data`, `example_compressed_input` | `INSTALL.md#support-readiness-matrix` for platform/package status | Storage format does not decide solver support by itself. |
| Matrix Market file | Matrix Market load, then solver selection | `docs/cookbook.md#start-from-your-data`, `example_matrix_market` | `docs/matrix_market.md` plus INSTALL support matrix | File parsing support is not broad solver or benchmark proof. |
| General square solve | LU | `docs/solver_selection.md#direct-solvers`, `example_basic_solve` | INSTALL support matrix for platform/install status | Fixture-local comparison evidence is not broad external-library parity. |
| Symmetric positive-definite solve | Cholesky | `docs/solver_selection.md#direct-solvers`, Cholesky examples/tests as linked from solver selection | INSTALL support matrix; selected comparison detail stays in solver selection/report docs | Cholesky is not a general fallback and does not imply broad SPD or Windows freshness promotion. |
| Symmetric indefinite solve | LDLT | `docs/solver_selection.md#direct-solvers`, `example_ldlt` | INSTALL support matrix | No broad KKT, package, ABI, or platform claim. |
| Rectangular, least-squares, or rank-sensitive solve | QR | `docs/solver_selection.md#direct-solvers`, `example_least_squares`, `example_minnorm` | INSTALL support matrix; selected report detail stays in solver selection/report docs | No broad QR parity, raw basis identity, Windows QR selected freshness, or external-library parity. |
| Many solves with same sparsity pattern | Explicit analysis/factor/refactor lifecycle | README repeated-run direct workflow and solver selection | INSTALL support matrix for install/platform status | Reuse support is workflow-specific and not a package/ABI claim. |
| Large solve where direct cost is the issue | Iterative solver with diagnostics | `docs/solver_selection.md#iterative-solvers`, iterative examples | INSTALL support matrix | Iteration count and residual behavior are local diagnostics, not portable performance proof. |
| Matrix-free solve | Matrix-free iterative workflow | `example_matrix_free`, solver selection iterative section | INSTALL support matrix | Matrix-free support does not imply backend superiority or package support. |
| Symmetric eigenpairs | `sparse_eigs_sym` | `docs/solver_selection.md#eigensolver-workflows`, `example_eigs` | INSTALL support matrix | No nonsymmetric eigensolver or state-of-the-art parity claim. |
| Rank, condition, pseudoinverse, or low-rank work | SVD APIs | `docs/solver_selection.md#svd-and-low-rank-workflows`, `example_svd_lowrank` | INSTALL support matrix | No broad SVD parity, raw singular-vector identity, or package/ABI claim. |
| Local benchmark or report interpretation | Benchmark docs after workflow selection | `benchmarks/README.md#reading-benchmark-results` | `benchmarks/README.md` and selected report manifest owners | Local and selected hosted freshness do not prove portable performance. |
| Installed downstream consumer | Static Make/pkg-config or CMake consumer | `INSTALL.md#start-here`, installed consumer tutorial | `INSTALL.md#support-readiness-matrix` | Static-first install only; no shared-library, dynamic ABI, or package-manager support. |
| API declarations and local generated docs | Source-controlled API reference | `docs/api_reference.md` | `docs/api_reference.md` plus INSTALL support matrix | Generated HTML is local-only, ignored output; no hosted or release evidence. |
| Threading/OpenMP controls | Build/runtime controls after first workflow works | README runtime/backend controls and algorithm docs | INSTALL support matrix for platform/install status | OpenMP controls are not portable performance or broad platform proof. |

## Link Target And Ownership Map

| Quick-reference link target | Owner role | Use in table |
| --- | --- | --- |
| `docs/cookbook.md#start-from-your-data` | Data-first workflow owner | CSR, CSC, Matrix Market, first-use data routes. |
| `docs/solver_selection.md#choose-the-smallest-workflow` | Detailed solver decision owner | Direct, iterative, eigensolver, QR, SVD workflow details. |
| `examples/README.md#start-here` | Runnable example owner | First build-tree success and example discovery. |
| `examples/README.md#diagnostics-handoff` | Runnable diagnostics handoff | Local diagnostic interpretation after examples. |
| `INSTALL.md#support-readiness-matrix` | Public support truth | Every row that could imply package, platform, install, ABI, or support status. |
| `docs/api_reference.md` | API declaration and generated API policy owner | API/reference row. |
| `benchmarks/README.md#reading-benchmark-results` | Benchmark interpretation owner | Benchmark/report row. |
| `docs/maintainer_guide.md` | Maintainer proof interpretation owner | Link indirectly from support matrix or detailed docs, not from every quick-reference row. |

## Support/Readiness Interpretation

The quick reference should use these compact interpretations:

| Label | Meaning | Owner |
| --- | --- | --- |
| Supported workflow | Normal public workflow documented by examples, solver selection, headers, and tests. | Workflow docs plus public headers. |
| Validated install path | Install/downstream path covered by install validation evidence. | `INSTALL.md`. |
| Local-only evidence | Generated/local proof that must not be read as hosted, release, package, or broad support evidence. | Owner doc for the local surface. |
| Hosted selected evidence | A named hosted selected lane or artifact for a selected target only. | Manifest, benchmark/report docs, maintainer guide. |
| Guarded workflow | A workflow path guarded or reviewed without support/freshness promotion. | INSTALL and maintainer guide. |
| Deferred / not claimed | Explicitly unsupported or residualized until closure evidence exists. | INSTALL and residual/maintainer docs. |

## Acceptance Criteria

Day 6 implementation must satisfy these criteria:

1. The quick reference fits in one compact table or one short table plus a
   short lead-in.
2. Every row links to a source-controlled owner doc rather than planning
   artifacts.
3. Every row that could imply support status links to
   `INSTALL.md#support-readiness-matrix`.
4. Solver-family rows link to `docs/solver_selection.md` for detailed evidence
   instead of copying selected report/fixture caveats.
5. Benchmark/report rows link to `benchmarks/README.md` and must not claim
   portable performance, timing thresholds, release evidence, backend
   superiority, or state-of-the-art status.
6. API rows preserve Sprint 204 local-only generated API boundaries and link to
   `docs/api_reference.md`.
7. Packaging rows do not claim Homebrew, package-manager distribution,
   shared-library support, dynamic ABI compatibility, runtime-loader behavior,
   or broad platform parity.
8. Windows wording uses short `validated`, `guarded workflow`, or `deferred`
   labels only when backed by the support matrix and must not promote Windows
   selected freshness.
9. The quick reference does not use raw manifest/schema vocabulary unless it
   is inside a link label to an owner document.
10. Public wording remains short enough that later Day 7 consolidation can
    remove duplicated caveats rather than adding another long caveat block.

## Guard Implications For Later Days

Day 11-Day 12 should consider guard coverage for:

- quick-reference presence and link targets in `docs/cookbook.md`;
- README/tutorial/examples routing to the selected quick-reference anchor;
- retained `INSTALL.md#support-readiness-matrix` support-truth link;
- absence of package-manager, Homebrew/core, shared-library, dynamic ABI,
  broad Windows, portable performance, release, and state-of-the-art claims in
  the quick reference;
- preservation of generated API local-only routing to `docs/api_reference.md`;
- preservation of benchmark interpretation routing to `benchmarks/README.md`.

## Non-Changes

Day 4 does not edit public documentation, guards, scripts, source files,
headers, workflows, generated outputs, or support claims. It records the design
that Day 6-Day 8 should implement after Day 5 selects the support-truth
architecture.

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 205.2 has a concrete table design before implementation. | Met. The row design, placement, link targets, vocabulary, and acceptance criteria are recorded here. |
| Every row has an intended support/readiness interpretation. | Met. Each row maps to a support/readiness route and retained boundary. |
| Compact wording cannot imply unearned package, ABI, platform, or performance support. | Met by design. Acceptance criteria require INSTALL routing and explicit package, ABI, Windows, benchmark, generated API, release, and state-of-the-art boundaries. |

## Day 4 Disposition

Day 4 is complete. Day 5 should define the central support/readiness ownership
model and duplicate-caveat conversion plan so Day 6-Day 8 can implement the
quick reference and related routes without weakening claim boundaries.
