# Sprint 121 Day 1 Sprint Intake

## Purpose

Establish the Sprint 121 execution baseline before SVD/QR audits, fixture
taxonomy design, helper extraction, rank-deficient evidence expansion, or
dense-reference pilot work begins. Day 1 records authoritative scope, required
inputs, day-level ownership, validation rules, evidence fields, and non-claim
boundaries.

## Authoritative Sprint Scope

Sprint 121 implements the Epic 11 project-plan section:
`Sprint 121: SVD, QR & Rank-Deficient Numerical Oracle Expansion`.

| Field | Value |
|---|---|
| Duration | 14 days |
| Estimate | 168 hours |
| Goal | Strengthen SVD, QR, rank, pseudoinverse, and least-squares evidence with reusable helpers while keeping LAPACK/SciPy parity as a non-claim. |
| Primary deliverables | SVD/QR/rank fixture taxonomy; reusable SVD/QR proof helpers; expanded rank-deficient evidence; bounded dense-reference pilot; updated trust-boundary docs. |

## Project-Plan Item Map

| Item # | Item | Planned Days | Day 1 Intake Notes |
|---:|---|---|---|
| 1 | SVD/QR Evidence Audit | Days 2-3 | Split into SVD and QR audit days so SVD, partial-SVD, QR, least-squares, pseudoinverse, low-rank, and rank-deficient owners stay visible. |
| 2 | Matrix Taxonomy Design | Days 4-5, 8-10 | Taxonomy must preserve deterministic rank, conditioning, shape, sparsity, scaling, and expected-failure semantics. |
| 3 | SVD Helper Extraction | Days 5-6, 10 | SVD helpers must preserve reconstruction, orthogonality, rank, low-rank, and pseudoinverse tolerance ownership. |
| 4 | QR/Least-Squares Expansion | Days 5, 7-9 | QR/least-squares work must keep residual, rank-deficient, rectangular, and generated-RHS behavior visible. |
| 5 | External/Dense Reference Pilot | Days 11-12 | Pilot must be bounded to named fixtures and cannot imply LAPACK/SciPy parity. |
| 6 | Validation | Days 6-13 | Focused SVD/QR tests first; full quality required if `.c` or `.h` files change. |
| 7 | Docs and Non-Claims | Days 13-14 | Trust-boundary docs must explicitly avoid broad external parity or state-of-the-art claims. |

## Input Artifact Inventory

| Input | Required Use |
|---|---|
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 121 section | Defines authoritative item scope and estimates. |
| `docs/planning/EPIC_11/SPRINT_121/PLAN.md` | Defines day-by-day execution sequence. |
| `docs/planning/EPIC_11/SPRINT_120/WORKING_NOTES.md` | Provides current validation expectations, evidence fields, and non-claim style. |
| `docs/planning/EPIC_11/SPRINT_120/RETROSPECTIVE.md` | Provides recent closeout lessons and residual queue framing. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day4-shared-fixture-architecture.md` | Provides helper-boundary and fixture architecture rules. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day11-cross-solver-oracle-pilot-design.md` | Provides bounded comparison pilot design pattern. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day12-cross-solver-oracle-pilot-implementation.md` | Provides pilot implementation and focused proof pattern. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day13-validation-package.md` | Provides latest focused/full validation packaging pattern. |
| `docs/planning/EPIC_11/SPRINT_120/artifacts/day14-oracle-closeout.md` | Provides residual handoff and non-claim closeout pattern. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day2-validation-inventory.md` | Provides validation lane categories and command expectations. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day8-product-truth-map.md` | Provides product truth and support boundaries. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day11-evidence-template-design.md` | Provides evidence template structure. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day13-public-claim-drift-audit.md` | Provides claim-drift and non-claim checks. |
| `docs/planning/EPIC_11/SPRINT_118/templates/oracle-expansion-evidence-template.md` | Provides required fields for bounded oracle expansion artifacts. |

## Validation Rules

| Change Type | Required Day-Level Validation |
|---|---|
| Planning documentation only | `git diff --check`; focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_121`. |
| Test-only `.c` or helper `.h` edits | `make format && make lint && make test`; also run focused touched tests before full quality where useful. |
| Source-list or Makefile membership edits | `make source-list-check`; focused build of touched target; full quality if `.c` or `.h` changed. |
| CMake or CTest membership edits | CMake configure/build and `ctest -N` count proof; full quality if `.c` or `.h` changed. |
| SVD helper or proof changes | Focused full-SVD, partial-SVD, low-rank, pseudoinverse, rank, reconstruction, and orthogonality tests as touched. |
| QR helper or proof changes | Focused QR, QR solve, least-squares, rank-deficient, reconstruction, generated-RHS, and QR-vs-reference tests as touched. |
| Dense-reference or external pilot | Focused pilot test plus adjacent SVD/QR tests that share fixtures, helpers, or assertions. |
| Public/support wording | Claim scan against Sprint 118 product truth, Sprint 118 public-claim drift audit, Sprint 120 non-claim framing, and current README/docs wording. |

## Non-Claim Boundaries

Sprint 121 may improve bounded SVD/QR/rank evidence and validation hygiene. It
does not, by itself, claim:

- broad LAPACK parity;
- broad SciPy parity;
- broad SuiteSparse, PETSc, Trilinos, or Eigen parity;
- full external-oracle coverage;
- state-of-the-art numerical completeness or superiority;
- portable benchmark or performance superiority;
- package/install support expansion;
- public API expansion;
- symmetric platform validation beyond the lanes actually run.

## Day-Level Owner Map

| Day | Owner Focus | Output |
|---:|---|---|
| 1 | Intake setup | Working notes and intake artifact. |
| 2 | SVD audit | SVD, partial-SVD, low-rank, rank, and pseudoinverse owner inventory. |
| 3 | QR audit | QR, least-squares, rank-deficient, and rectangular owner inventory. |
| 4 | Matrix taxonomy | Fixture taxonomy and metadata design. |
| 5 | Helper plan | Exact SVD/QR helper extraction checklist. |
| 6 | SVD implementation | SVD helper extraction and focused proof. |
| 7 | QR implementation | QR helper extraction and focused proof. |
| 8 | Rank fixtures | Rank-deficient and near-dependent fixture expansion. |
| 9 | Least-squares/pseudoinverse | Rectangular, least-squares, and pseudoinverse evidence. |
| 10 | Low-rank/partial-SVD | Low-rank and partial-SVD proof expansion. |
| 11 | Reference design | Bounded dense-reference or external pilot design. |
| 12 | Reference implementation | Pilot implementation and focused proof. |
| 13 | Validation/docs | Focused/full validation package and trust-boundary docs. |
| 14 | Closeout | Residuals, non-claims, artifact index, and retrospective inputs. |

## Completion Criteria

| Criterion | Status |
|---|---|
| Every Sprint 121 project-plan item has a day-level owner | Complete |
| Prior oracle patterns and reusable validation lanes are identified | Complete |
| Validation requirements are recorded before implementation | Complete |
| Non-claim boundaries are recorded before implementation | Complete |
| No helper extraction, fixture expansion, or pilot work begins before proof scope, non-claims, and validation expectations are recorded | Complete |
