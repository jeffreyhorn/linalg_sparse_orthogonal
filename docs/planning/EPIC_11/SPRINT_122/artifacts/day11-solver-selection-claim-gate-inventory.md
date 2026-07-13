# Sprint 122 Day 11 Solver-Selection Claim Gate Inventory

## Purpose

Day 11 audits current solver-selection, README, tutorial, example, benchmark,
install, and maintainer wording against Sprint 121 and Sprint 122 evidence. The
goal is to define what evidence is required before public solver-selection or
support-level wording can expand on Day 12 or in later adoption/corpus sprints.

This artifact does not edit public docs. It creates the claim gate for any
future wording change.

## Audited Surfaces

| Surface | Current Role | Day 11 Disposition |
| --- | --- | --- |
| `README.md` | Public front door and capability overview. | Mostly aligned; workflow wording is supported, but detailed external/support claims must remain delegated to docs and tests. |
| `docs/solver_selection.md` | Primary public solver workflow chooser. | Aligned; it routes by problem shape and explicitly fences benchmark/local/state-of-the-art claims. |
| `docs/tutorial.md` | Fuller learning path after README. | Aligned; it teaches workflow selection and keeps repeated-run support narrow. |
| `examples/README.md` | Runnable example map. | Aligned; examples are teaching/adoption surfaces, not oracle or benchmark owners. |
| `benchmarks/README.md` | Benchmark command and CSV/report semantics. | Aligned; benchmark rows are local measurement, not portable performance guarantees. |
| `INSTALL.md` | Install, consumer, and platform-tier wording. | Aligned; static-first install and reviewed-platform scope remain explicit. |
| `docs/maintainer_guide.md` | Quality, oracle, platform, package, and support interpretation. | Aligned but contains older Sprint 102/103 oracle tables that need Day 12 reconciliation if public wording references Sprint 122 evidence. |
| Sprint 121 artifacts | Prior audit/helper/external-lane baseline. | Used as input; not reopened. |
| Sprint 122 Days 2-10 artifacts | Current SVD, QR, partial-SVD, helper, and non-claim decisions. | Source of truth for new claim gates. |

## Sprint 122 Evidence Summary

| Evidence Area | Sprint 122 Outcome | Claim Boundary |
| --- | --- | --- |
| Additional SVD external fixture | Added `svd_rankdef_duplicate_5x4` bounded external singular-value lane. | Supports one additional rank-deficient singular-value fixture only; no broad LAPACK/NumPy/SciPy parity. |
| QR external dense reference | Added `qr_overdetermined_incompatible_4x2` bounded least-squares lane. | Supports one incompatible full-column-rank tall LS fixture; no broad QR, rank-deficient, minimum-norm, Q-basis, or LAPACK parity. |
| Partial-SVD external reference | Added `partial_svd_diag6_k2` bounded top-k singular-value lane. | Supports one top-k value fixture only; no vector, subspace, convergence-budget, repeated-spectrum, or broad partial-SVD parity. |
| Minimum-norm helper ownership | Deferred migration; preserved QR, COLAMD/reordering, SVD pseudoinverse, refinement, fallback, and SuiteSparse owners. | Public docs can mention minimum-norm APIs as supported workflows but must not claim consolidated proof ownership or external minimum-norm parity. |
| Bidiagonal/Golub-Kahan helpers | Deferred general consolidation; future extraction limited to bidiagonal/GK-specific measurement helpers. | Public docs must not imply low-level Bidiagonal/GK helper consolidation or broad external dense-library proof. |
| Cross-solver oracle baseline | Sprint 120 added a small LU/Cholesky/QR/CG generated-RHS pilot. | Useful bounded internal oracle; not broad direct/iterative parity or performance evidence. |

## Public/Support Wording Inventory

| Wording Area | Current Public Wording | Evidence Status | Gate |
| --- | --- | --- | --- |
| Direct solver selection | LU for general square, Cholesky for SPD, LDLT for symmetric indefinite, QR for rectangular/rank-sensitive. | Supported by long-standing tests plus bounded direct external lanes from earlier sprints. | Keep. Any stronger comparative wording needs family-specific evidence. |
| QR least-squares | Solver-selection and examples route rectangular/rank-sensitive systems to QR. | Supported by internal QR tests plus Sprint 122 one-fixture external LS lane. | Keep workflow wording; do not claim LAPACK/NumPy/SciPy parity or all LS cases. |
| Minimum-norm QR | Examples/tutorial mention `sparse_qr_solve_minnorm`. | Supported by internal QR/COLAMD/SVD-pinv checks; helper ownership remains split. | Keep API/workflow wording; avoid external parity or consolidated oracle wording. |
| SVD rank/pseudoinverse/low-rank | README and solver-selection route singular values, rank, condition, pseudoinverse, and low-rank to SVD APIs. | Internal SVD evidence is broad; external evidence is still only bounded singular-value fixtures. | Keep workflow wording; no broad dense-library parity, vector parity, or global optimality claim. |
| Partial SVD | README lists full and partial SVD. | Internal tests are broad; Sprint 122 adds one top-k external value fixture. | Keep availability wording; no vector/subspace/convergence or broad parity wording. |
| Eigensolver selection | README and solver-selection present symmetric eigensolver workflows and AUTO backend. | Supported by internal exact/cross-backend tests and prior proof-owner work. | Keep symmetric-only wording; no nonsymmetric eigensolver or ARPACK parity claim. |
| Cross-solver agreement | Maintainer/test artifacts own bounded generated-RHS comparison. | One small SPD fixture compares LU/Cholesky/QR/CG. | Do not publish as broad parity; can be referenced only as bounded maintainer evidence. |
| Benchmarks/performance | README/solver-selection/benchmarks docs treat benchmarks as local measurement. | Aligned. | Keep; no portable performance, scalability, or state-of-the-art claim. |
| Platform/install/package | README/INSTALL/maintainer guide keep Linux strongest, macOS narrower, Windows CMake subset. | Aligned. | Keep; no package-manager, shared-library, dynamic ABI, Windows Makefile, or Windows install-validation claim. |
| Public API/support level | README and headers describe shipped APIs. | Aligned for availability; support breadth varies by family. | Do not infer API expansion from test helper or oracle work. |

## Evidence-To-Wording Matrix

| Candidate Wording | Required Evidence Before Publication | Current Status |
| --- | --- | --- |
| "QR has external least-squares validation." | Multiple QR external fixtures covering compatible/incompatible, rank-deficient, underdetermined/minimum-norm, Q-basis/economy semantics, skip behavior, and failure interpretation. | Not met; only one incompatible 4x2 LS fixture exists. |
| "SVD matches dense library behavior." | Maintained external dense-library/reference suite covering singular values, vectors, rank thresholds, pseudoinverse, low-rank, rectangular, rank-deficient, and tolerance policy. | Not met; bounded pure-Python singular-value fixtures only. |
| "Partial SVD has external parity." | Top-k values, vector/subspace semantics, convergence budgets, degenerate spectra, rectangular/rank-deficient fixtures, and external reference trust boundary. | Not met; one top-k diagonal value fixture only. |
| "Minimum-norm proof ownership is consolidated under QR." | Helper migration with behavior-specific names, unchanged QR/COLAMD/SVD-pinv tests, focused validation, and preserved tolerance ownership. | Not met; migration deferred on Day 9. |
| "Bidiagonal/GK helpers are generalized SVD helpers." | Dedicated bidiagonal/GK helper owner, extracted measurement helpers, wide-transpose proof, SVD/GK focused tests, and no loss of scenario labels. | Not met; general consolidation rejected on Day 10. |
| "Solvers have broad cross-family parity." | Corpus-spanning direct/iterative/eigensolver/SVD oracle matrix with solver-specific assumptions, tolerance bands, and unsupported-case handling. | Not met; Sprint 120 pilot is one small SPD generated-RHS fixture. |
| "Benchmarks prove performance superiority." | Portable benchmark protocol with platform/compiler/thread/backend controls, statistical treatment, and reviewed thresholds. | Not met; benchmarks remain local measurement artifacts. |
| "Cross-platform support is equivalent." | Reviewed Linux, macOS, and Windows parity over the same build/test/install surfaces with staged exclusions retired. | Not met; current platform tiers remain explicit. |

## Non-Claim List

The following must remain fenced in public and support wording:

- LAPACK, NumPy, SciPy, PETSc, Trilinos, Eigen, SuiteSparse, ARPACK, or broad
  ecosystem parity;
- broad external dense-library parity for SVD, QR, partial SVD, direct solvers,
  iterative solvers, or eigensolvers;
- singular-vector, Q-basis, Ritz-vector, or subspace external parity unless a
  basis-invariant metric is explicitly implemented and validated;
- broad minimum-norm global optimality beyond current bounded internal tests;
- broad partial-SVD convergence-budget or vector/subspace parity;
- broad cross-solver equivalence or solver superiority;
- portable performance, scalability, memory, fill-reduction, or state-of-the-art
  claims;
- package-manager distribution support;
- shared-library or dynamic ABI stability;
- equal Linux/macOS/Windows reviewed support;
- Windows Makefile, install-validation, thread/fuzz/property, or full CTest
  parity;
- public API expansion from test-helper, external-reference, or maintainer-only
  proof work.

## Required Evidence Categories

| Category | Required Before Wording Expands |
| --- | --- |
| Fixture breadth | At least one fixture per named behavior class, with shape, rank, conditioning, and compatibility documented. |
| Reference trust | External helper or oracle source, dependency policy, skip behavior, and failure interpretation must be explicit. |
| Tolerance ownership | Every tolerance must be tied to fixture scale and behavior, not hidden inside broad assertion helpers. |
| Vector/subspace semantics | Sign, basis, projection, or principal-angle semantics must be defined before vector/subspace claims. |
| Platform impact | CMake/CTest membership and Windows/macOS/Linux reviewed surfaces must be counted before support wording changes. |
| Public/API boundary | Public wording must distinguish shipped API availability from internal proof-owner or helper movement. |
| Performance governance | Benchmark wording needs branch, compiler, platform, backend, thread, fixture, repeat, and artifact context. |
| Residual owner | Any unearned candidate wording must map to a future sprint, not remain implicit. |

## Day 12 Claim-Gate Checklist

Day 12 should choose one of two outcomes.

| Outcome | Checklist |
| --- | --- |
| No public wording update | Confirm current README, solver-selection, tutorial, examples, benchmarks, INSTALL, and maintainer wording already fences Sprint 122 evidence; write explicit no-update rationale and residual owner table. |
| Limited maintainer/support wording update | Update only maintainer evidence tables or solver-selection caveats that are stale relative to Sprint 122; avoid README capability expansion; run docs checks; record exact evidence link and non-claim preserved. |

Any Day 12 wording update must answer:

1. Which exact validated artifact supports the wording?
2. Which unsupported interpretations remain fenced?
3. Does the change affect README, solver-selection, tutorial, examples,
   benchmarks, install, platform, or maintainer-only wording?
4. Does it imply new Makefile, CMake, CTest, CI, package, or public API support?
5. What validation is required for the touched surface?

## Validation Notes

Day 11 changed documentation only. Required validation is `git diff --check`
and a focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_122`.
The branch passed `make format`, `make lint`, and `make test` after the Day 8
C/header changes; Day 11 adds no new code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Candidate public wording is tied to evidence requirements. | Complete | See evidence-to-wording matrix and required evidence categories. |
| Unsupported external, support-level, performance, platform, and API claims remain fenced. | Complete | See non-claim list. |
| Adoption and corpus/report sprints receive clear prerequisites. | Complete | Day 12 checklist and required evidence categories define the prerequisites. |
