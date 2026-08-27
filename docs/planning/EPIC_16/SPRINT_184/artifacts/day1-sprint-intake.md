# Sprint 184 Day 1: Sprint Intake and Prior-Art Review

**Sprint:** 184 - Public Header Coherence Batch 3
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Sprint path:** `docs/planning/EPIC_16/SPRINT_184/`
**Status:** Complete

## Purpose

Day 1 establishes Sprint 184 scope, inherited declaration-preserving
constraints, candidate family context, documentation surfaces, and open
questions before any public header family is selected or edited.

## Active Planning Source

| Field | Value |
| --- | --- |
| Requested sprint | Sprint 184: Public Header Coherence Batch 3 |
| Active project-plan path | `docs/planning/EPIC_16/PROJECT_PLAN.md` |
| Active section | `Sprint 184: Public Header Coherence Batch 3` |
| Sprint goal | Normalize one more high-impact public header family without declaration drift, improving API usability and generated documentation input. |
| Prior handoff | Sprint 177 selected this work as closure target `S177-R09`. |

## Project-Plan Item Interpretation

| Item | Name | Day 1 interpretation |
| --- | --- | --- |
| 184.1 | Header Family Selection | Start from QR/SVD/LDLT candidates, inherit prior selection criteria, and prepare for Day 2 baseline capture. |
| 184.2 | Contract Cleanup | Track lifecycle, ownership, error-code, tolerance, workspace, option/result, and cancellation wording as cleanup categories. |
| 184.3 | Declaration Organization | Treat organization as allowed only after declaration-preserving guardrails are available. |
| 184.4 | Example and Docs Alignment | Map examples, tutorial, solver-selection, cookbook, and API reference surfaces before edits. |
| 184.5 | Mechanical Guard | Reuse or extend declaration checksum, docs coverage, or unsupported-claim guard patterns for the selected family. |
| 184.6 | Validation | Require full C quality gates after `.c` or `.h` edits and focused docs/guard checks for documentation changes. |

## Prior Evidence Reviewed

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_16/SPRINT_177/artifacts/day2-residual-audit.md` | Residual `S177-R09` identifies public-header coherence breadth as open, with QR, SVD, LDLT, IC, ILU, reorder, and analysis still uneven. |
| `docs/planning/EPIC_16/SPRINT_177/artifacts/day7-target-selection.md` | Sprint 184 is selected to normalize one high-impact public header family, preserve declarations, and align API reference/user docs. |
| `docs/planning/EPIC_15/SPRINT_172/WORKING_NOTES.md` | Prior cleanup guardrails require one selected family, declaration baselines, scoped docs alignment, and full C gates after public header edits. |
| `docs/planning/EPIC_15/SPRINT_172/RETROSPECTIVE.md` | Sprint 172 successfully cleaned `include/sparse_lu.h`, added focused LU docs/header guard coverage, and kept package/ABI/platform/performance non-claims intact. |
| `docs/planning/EPIC_14/SPRINT_164/artifacts/day2-header-selection.md` | Selection criteria should weigh user impact, documentation ambiguity, claim risk, option/result complexity, downstream visibility, and declaration-preserving cleanup feasibility. |

## Candidate Family Snapshot

Day 1 keeps all three project-plan candidates open. The purpose is to make Day
2 baseline capture concrete, not to make the final family decision early.

| Candidate | Header | Header lines | Primary documentation surfaces | Test and evidence surfaces | Day 1 read |
| --- | --- | ---: | --- | --- | --- |
| QR | `include/sparse_qr.h` | 373 | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md`, `examples/README.md` | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_corpus.c`, `tests/qr_external_dense_reference.py` | Strong candidate because rank, nullspace, least-squares, and minimum-norm wording is high visibility and claim-sensitive. |
| SVD / partial SVD | `include/sparse_svd.h` | 243 | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md`, `examples/README.md` | `tests/test_svd.c`, `tests/test_svd_partial_corpus.c`, `tests/svd_external_dense_reference.py` | Strong candidate because rank, condition, vector, convergence, and low-rank output wording is evidence-sensitive. |
| LDLT | `include/sparse_ldlt.h` | 315 | `README.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `docs/api_reference.md`, `examples/README.md` | `tests/test_ldlt.c`, `tests/test_ldlt_csc.c`, `tests/test_ldlt_backend_dispatch.c`, `tests/ldlt_external_dense_reference.py` | Strong candidate because direct-solver lifecycle, backend dispatch, inertia, and symmetric-indefinite semantics need careful contract wording. |

## Documentation And Example Surface Map

| Surface | Candidate relevance |
| --- | --- |
| `README.md` | All three candidates appear in capability and public API sections; QR and SVD also carry explicit fixture-local evidence boundaries. |
| `docs/api_reference.md` | Summary rows exist for `sparse_qr.h`, `sparse_svd.h`, and `sparse_ldlt.h`. |
| `docs/tutorial.md` | QR and SVD have dedicated walkthrough sections; LDLT appears in direct-solver routing and include guidance. |
| `docs/cookbook.md` | QR and SVD have broader workflow guidance; LDLT appears in direct routing and benchmark context. |
| `docs/solver_selection.md` | QR, SVD, and LDLT are part of user-facing solver routing; QR and SVD include explicit evidence-boundary wording. |
| `examples/example_least_squares.c` | QR overdetermined least-squares path. |
| `examples/example_minnorm.c` | QR underdetermined minimum-norm path. |
| `examples/example_colamd.c` | QR with COLAMD ordering. |
| `examples/example_svd_lowrank.c` | SVD rank and low-rank output path. |
| `examples/example_condition.c` | SVD condition workflow. |
| `examples/example_ldlt.c` | LDLT factor, solve, refine, inertia, and condition-estimate workflow. |

## Inherited Declaration-Preservation Controls

| Control | Sprint 184 use |
| --- | --- |
| Select one family only | Avoids broad public-header churn and keeps review focused. |
| Baseline before edit | Day 2 should capture declarations and declaration-like surfaces before comment or organization changes. |
| Comment cleanup first | Contract wording can improve without changing signatures, types, macros, enum values, or include guards. |
| Organization only under guard | Declaration reordering should wait until a guard or checksum strategy can make drift visible. |
| Docs alignment after header cleanup | Examples and docs should follow the selected public-header contract, not lead it. |
| Family-specific guard precedent | `scripts/check_lu_header_docs_guard.sh` can inform a QR/SVD/LDLT guard if a similar lightweight check is selected. |

## Claim Boundaries

Sprint 184 may clarify API-local contracts for:

- lifecycle and cleanup responsibilities;
- caller-owned and callee-owned memory;
- option/result structure meanings;
- tolerance defaults and interpretation;
- workspace and reusable-state expectations;
- error-code and failure behavior;
- cancellation or callback behavior where supported by the selected family.

Sprint 184 must not imply new support for:

- dynamic ABI stability;
- shared-library builds or runtime-loader behavior;
- package-manager providers;
- broad Windows or platform parity;
- portable performance superiority;
- broad external-library parity;
- broad generated API HTML publication;
- state-of-the-art sparse linear algebra coverage.

## Day 2 Handoff

Day 2 should capture baselines for the QR, SVD, and LDLT candidates before
narrowing the family further:

- public declarations, typedefs, structs, enums, macros, and function
  signatures;
- declaration ordering and section groupings;
- option/result structures and lifecycle/free helpers;
- comments that feed generated API documentation;
- existing guard coverage and gaps;
- docs/example mismatches visible before edits.

## Validation

Day 1 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The sprint scope is traceable to project-plan items 184.1 through 184.6. | Complete | Project-plan item interpretation is recorded above and in `WORKING_NOTES.md`. |
| Prior header-cleanup patterns are identified and ready to reuse. | Complete | Sprint 172 and Sprint 164 precedents are captured with reusable controls. |
| Candidate family evidence is sufficient to begin baseline capture. | Complete | QR, SVD, and LDLT header, docs, examples, and test surfaces are inventoried for Day 2. |
