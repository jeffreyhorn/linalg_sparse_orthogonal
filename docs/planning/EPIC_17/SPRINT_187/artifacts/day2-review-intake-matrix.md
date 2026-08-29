# Sprint 187 Day 2: Review Intake Matrix

## Purpose

Convert the Codex Epic 17 review findings into a structured initial gap ledger
with owner surfaces, current evidence, claim risks, user value, closure
candidacy, candidate sprint, required validation, and non-goals.

## Source Inputs

| Source | Use |
| --- | --- |
| `docs/planning/EPIC_17/reviews/review-codex-2026-08-28.md` | Primary source for strengths, gaps, state-of-the-art assessment, and prioritized gap list. |
| `docs/planning/EPIC_17/reviews/todo-codex-2026-08-28.md` | Source for step-by-step closure phases and candidate validation commands. |
| `docs/planning/EPIC_16/EPIC_16_RESIDUAL_QUEUE.md` | Source for inherited residual IDs and closure targets. |
| `docs/planning/EPIC_17/PROJECT_PLAN.md` | Source for planned Sprint 188-196 closure sequence. |
| `docs/planning/EPIC_17/SPRINT_187/WORKING_NOTES.md` | Day 1 source map, owner-surface map, risks, and ledger schema. |

## Initial Gap Ledger

| Gap ID | Source | Area | Finding | Owner surfaces | Current evidence | Claim risk | User value | Closure candidate | Candidate sprint | Required validation | Non-goals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E17-GAP-001 / R186-PKG-LICENSE | Epic 16 residual; review package findings | Packaging | Homebrew proof is blocked by missing standalone license metadata. | Root license metadata; `packaging/homebrew/`; `scripts/homebrew_local_formula_proof.sh`; `README.md`; `INSTALL.md`; package guards. | Homebrew local formula proof exists but exits before full install proof because no root `LICENSE`, `COPYING`, or `NOTICE` exists. | Accidental Homebrew/provider support claim before proof passes. | Package adoption and support clarity. | Closure candidate | Sprint 188 | `bash scripts/homebrew_local_formula_proof.sh`; package guards; install checks selected by implementation; docs checks. | Homebrew/core, bottles, Linuxbrew, public tap, binary package, and broad package-manager support. |
| E17-GAP-002 / R186-WIN-PWSH | Epic 16 residual; todo Phase 3 | Platform | PowerShell validation ownership is missing locally and not separated from report freshness promotion. | `.github/workflows/windows-ci.yml`; PowerShell snippets; selected workflow guards; report docs; maintainer guide. | Windows CI exists for CMake/MSVC and static-first install, but local `pwsh` was unavailable during Epic 16. | Treating absent local PowerShell proof as Windows report workflow confidence. | Platform validation confidence. | Closure candidate | Sprint 189 | Future PowerShell parse/workflow command; selected workflow tests; schema tests; docs checks. | Broad Windows parity, Windows Makefile parity, Windows `pkg-config` execution parity. |
| E17-GAP-003 / R186-WIN-REPORT-FRESHNESS | Epic 16 residual; review coverage gaps | Platform and reports | Windows selected report freshness remains formally deferred. | Windows CI; selected report target manifest; report generator/normalizer scripts; report docs; maintainer guide. | Formal guarded deferral from Sprint 182 with no promoted Windows freshness lane. | Claiming Windows generated report freshness or broad Windows report parity without hosted evidence. | Cross-platform report credibility. | Closure candidate | Sprint 190 | Selected Windows freshness command or renewed deferral guard; manifest/schema tests; normalizer tests; workflow guard tests. | Broad generated-report freshness and broad Windows platform/package parity. |
| E17-GAP-004 / R186-BROAD-COMPARISON | Epic 16 residual; review state-of-the-art gaps | Comparison | External comparison evidence remains selected-fixture-only. | `scripts/run_external_comparison.py`; comparison tests; selected manifest; report family docs; solver docs. | Selected QR, partial-SVD, LU, and Cholesky comparison rows exist with explicit non-parity wording. | Broad SuiteSparse, Eigen, SciPy, LAPACK, PETSc, Trilinos, or ecosystem parity claim. | Numerical credibility for one selected family. | Closure candidate | Sprint 191 | Comparison runner tests; selected freshness; manifest/schema tests; focused C tests where relevant. | Broad external-library parity. |
| E17-GAP-005 | Codex review efficiency and performance findings | Performance | Performance evidence is mostly local, threshold-free, or sentinel-scoped. | `benchmarks/`; benchmark scripts; report index normalization; CI freshness jobs; benchmark docs; maintainer guide. | Benchmark drivers, canonical reports, performance sentinels, and selected freshness paths exist but avoid portable performance claims. | Portable performance superiority or backend superiority claim. | Performance credibility for one selected workflow. | Closure candidate | Sprint 192 | `make bench-canonical-report-freshness`; `make performance-sentinels`; normalizer tests; docs checks; hosted artifact checks. | Portable performance superiority and broad state-of-the-art performance. |
| E17-GAP-006 / R186-REVIEW-SURFACE-NEXT | Epic 16 residual; review maintainability hotspots | Maintainability | Large test/source files still concentrate review risk. | Candidate large tests and source files; helper guard scripts; `Makefile`; `CMakeLists.txt`; maintainer guide. | Sprint 185 reduced only the LDLT CSC review surface; many large files remain. | Claiming broad review-surface cleanup or behavior preservation without focused proof. | Maintainability and defect reviewability. | Closure candidate | Sprint 193 | Focused tests; future helper guard; `make source-list-check`; `make format && make lint && make test`; CMake parity if source lists change. | Behavior changes, API expansion, broad cleanup claims. |
| E17-GAP-007 | Codex review usability findings | Usability | Adoption path is improving but still heavy for new users. | `README.md`; `INSTALL.md`; `docs/tutorial.md`; `docs/cookbook.md`; `docs/solver_selection.md`; examples. | README has adoption map and examples exist, but workflows and caveats remain dense. | Implying broader support by simplifying away caveats. | Faster first successful user workflow. | Closure candidate | Sprint 194 | Docs link checks; example builds; install checks; Doxygen checks if headers/docs change. | Removing necessary support-tier caveats or historical evidence needed by maintainers. |
| E17-GAP-008 | Codex review documentation/coherence findings | Documentation and coherence | Claim boundaries are correct but verbose and distributed. | README; INSTALL; API reference; maintainer guide; benchmark docs; package docs; planning docs. | Epic 16 calibrated claims, but support truth is spread across many docs. | Drift between public docs, maintainer docs, package docs, and planning artifacts. | Support clarity and review speed. | Closure candidate | Sprint 194 | Local markdown link checks; claim scans; API docs checks; docs-specific guards. | Over-centralizing by deleting necessary owner-specific contracts. |
| E17-GAP-009 | Codex review coverage/reliability findings | Reliability and test coverage | Allocation-failure proof remains selected-lane-only. | Allocation hooks; selected owner source/tests; focused Make/CTest gates; README; maintainer guide. | Iterative handle and `sparse_matmul()` allocation-failure proofs exist, not broad coverage. | Broad allocation-failure cleanup claim. | Higher reliability confidence for one more high-risk owner. | Closure candidate | Sprint 195 | Future focused allocation/failure gate; `make format && make lint && make test`; docs checks. | Broad allocation-failure coverage across all solvers. |
| E17-GAP-010 | Codex review state-of-the-art findings | State-of-the-art readiness | Representative matrix corpus and broad numerical robustness evidence are insufficient for broad claims. | Tests corpus; SuiteSparse fixtures; comparison reports; solver docs; maintainer guide. | Selected fixtures and corpus rows exist with explicit bounded claims. | Broad solver correctness or ecosystem parity claim. | More credible numerical evidence if narrowed. | Long-horizon | None by default | Future corpus/comparison gates if selected. | General state-of-the-art solver claim inside Epic 17. |
| E17-GAP-011 | Codex review maintainability findings | Maintainability | Several implementation files mix algorithm logic, workspace ownership, dispatch, telemetry, and cleanup. | `src/sparse_ldlt_csc.c`; `src/sparse_lu_csr.c`; `src/sparse_ldlt.c`; `src/sparse_iterative.c`; `src/sparse_qr.c`; `src/sparse_eigs.c`; `src/sparse_svd.c`. | Source-list guard and tests exist; only one selected review-surface cluster was reduced in Epic 16. | Hidden behavior changes during broad refactor. | Easier code review and safer changes. | Closure candidate only through one selected cluster | Sprint 193 | Focused tests and full C gate for selected cluster only. | Broad multi-module refactor. |
| E17-GAP-012 | Codex review efficiency findings | Efficiency | Orthogonal linked-list storage is useful but not a high-performance default representation for modern sparse kernels. | Matrix shell; CSR/CSC import/export; direct solver backends; benchmark docs. | CSR/CSC paths exist for selected kernels; linked-list remains core identity. | State-of-the-art performance or memory-locality claim. | Better future performance direction. | Long-horizon | None by default | Future backend/corpus/performance gates if selected. | Replacing core storage model inside Epic 17. |
| E17-GAP-013 | Codex review test coverage findings | Test coverage | Coverage is supplemental/tree-mutating and fuzz/property/differential evidence remains bounded. | `Makefile` coverage targets; CI coverage job; fuzz tests; corpus tests; external comparison scripts. | Coverage target exists; fuzzing and selected comparisons exist. | Treating test volume as broad numerical proof. | Better evidence planning. | Long-horizon except selected reliability proof | Sprint 195 for one owner | Focused failure-path gate; optional coverage if selected. | Broad property-based campaign across all solvers. |
| E17-GAP-014 | Codex review platform/package findings | Packaging and ABI | Shared-library and dynamic ABI support remain intentionally unsupported. | `CMakeLists.txt`; install docs; package tests; CMake package metadata; `sparse.pc.in`. | Static-first package contract is validated; shared-library requests are rejected. | Accidentally implying ABI stability or shared-library readiness. | Product clarity. | Retained non-claim | None by default | Static package guards and install tests. | Shared-library support and dynamic ABI support. |
| E17-GAP-015 | Codex review platform findings | Platform | Broad Windows parity remains unsupported beyond reviewed CMake/MSVC and static-first paths. | Windows CI; INSTALL; README; maintainer guide. | Windows CMake/MSVC and install/downstream validation exist; Makefile and `pkg-config` execution parity remain non-claims. | Broad Windows package/platform parity claim. | Platform-support clarity. | Retained non-claim except selected report freshness | Sprint 189-190 for selected slices | Windows CI, PowerShell validation, selected report gates. | Broad Windows parity. |
| E17-GAP-016 | Codex review executive assessment | State-of-the-art readiness | Epic 17 should end with calibrated claims, not an unqualified state-of-the-art claim. | README; maintainer guide; Epic 17 closeout docs; project plan. | Epic 16 explicitly rejected broad state-of-the-art claims. | Marketing language outruns proof. | Credible public positioning. | Retained non-claim until closeout calibration | Sprint 196 | Final claim audit; focused/full validation according to changed surfaces. | Unqualified state-of-the-art sparse linear algebra status. |

## Initial Status Summary

| Status | Count | Gap IDs |
| --- | ---: | --- |
| Closure candidate | 9 | E17-GAP-001 through E17-GAP-009. |
| Long-horizon | 4 | E17-GAP-010, E17-GAP-011 as broad multi-module refactor, E17-GAP-012, and E17-GAP-013 as broad coverage campaign. |
| Retained non-claim | 3 | E17-GAP-014, E17-GAP-015 broad parity, E17-GAP-016. |

## Long-Horizon Non-Goal Split

These findings are real, but they should not become broad Epic 17 work unless
a later sprint narrows them to one complete closure:

- general replacement of the orthogonal linked-list storage model;
- broad external ecosystem parity;
- broad numerical robustness proof across all solver families;
- shared-library and dynamic ABI productization;
- broad Windows package/platform parity;
- unqualified state-of-the-art sparse linear algebra positioning.

## Day 3 Follow-Up Needs

Day 3 should reconcile this ledger with the Epic 16 residual queue by:

1. preserving inherited residual IDs for package, Windows, hosted API,
   comparison, and review-surface work;
2. merging duplicate review findings into those residuals where appropriate;
3. deciding whether hosted API publication remains a selected Epic 17 closure
   candidate or stays long-horizon;
4. confirming candidate sprint assignments for Sprint 188 through Sprint 195;
5. recording any residuals that should be explicitly retained as non-claims.

## Validation

Day 2 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
