# Sprint 172 Day 1: Sprint Intake And Header Baseline

## Purpose

Day 1 establishes Sprint 172 scope, inherited public-header cleanup
constraints, package/ABI/adoption claim boundaries, and the starting candidate
set before any header family is selected or edited.

## Active Planning Source

| Field | Value |
| --- | --- |
| Requested sprint | Sprint 172: Public Header Coherence Batch 2 |
| Prompt path | `docs/planning/EPIC_12/PROJECT_PLAN.md` |
| Active project-plan path | `docs/planning/EPIC_15/PROJECT_PLAN.md` |
| Active section | `Sprint 172: Public Header Coherence Batch 2` |
| Sprint goal | Normalize one additional high-impact public header family and add lightweight guardrails against documentation drift. |

The prompt path is stale. `docs/planning/EPIC_12/PROJECT_PLAN.md` lines
176-206 describe an older report-index sprint, not the requested Sprint 172
header-coherence work. Sprint 172 proceeds from the active Epic 15 section.

## Project-Plan Items

| Item | Name | Sprint 172 interpretation |
| --- | --- | --- |
| 172.1 | Header Family Selection | Select one high-impact public header family based on adoption risk and API complexity. |
| 172.2 | Contract Cleanup | Normalize ownership, lifecycle, error-code, tolerance, workspace, and threading language. |
| 172.3 | Declaration Organization | Reorder declarations into coherent sections without changing behavior. |
| 172.4 | Example Alignment | Update examples or docs so the cleaned header has a clear workflow. |
| 172.5 | Mechanical Guard | Add or extend lightweight declaration/docs drift coverage where practical. |
| 172.6 | Validation | Run full C quality gates when `.c` or `.h` files change, plus focused docs/guard checks. |

## Prior Evidence Reviewed

| Source | Day 1 finding |
| --- | --- |
| `docs/planning/EPIC_14/SPRINT_164/artifacts/day2-header-selection.md` | Prior header-risk criteria used line counts, signal density, adoption visibility, option/result complexity, claim risk, and declaration-preserving cleanup feasibility. |
| `docs/planning/EPIC_14/SPRINT_158/artifacts/day3-public-header-coverage-map.md` | Checked-in `include/*.h` headers are the generated API source set; generated `sparse_version.h` remains separate package/install policy. |
| `docs/planning/EPIC_15/SPRINT_167/RETROSPECTIVE.md` | Epic 15 claim-gate conventions require unsupported package, ABI, platform, performance, generated-doc, and state-of-the-art claims to stay explicit. |
| `docs/planning/EPIC_15/SPRINT_170/RETROSPECTIVE.md` | Static-first-only package posture remains selected; shared-library and dynamic ABI support are deferred. |
| `docs/planning/EPIC_15/SPRINT_171/RETROSPECTIVE.md` | Package-manager support remains formally deferred and guarded. |

## Starting Header Candidate Set

Day 1 confirms 18 checked-in public headers under `include/`:

| Header | Day 1 role |
| --- | --- |
| `include/sparse_analysis.h` | Candidate for direct-solver analysis/lifecycle cleanup. |
| `include/sparse_bidiag.h` | Candidate, low current surface area. |
| `include/sparse_cholesky.h` | Candidate, direct-solver cleanup follow-up if needed. |
| `include/sparse_csr.h` | Candidate for compressed-storage helper cleanup. |
| `include/sparse_dense.h` | Candidate, helper-surface cleanup. |
| `include/sparse_eigs.h` | Prior high-impact header; may already have received cleanup, but remains a complexity reference. |
| `include/sparse_ic.h` | Candidate, preconditioner lifecycle surface. |
| `include/sparse_ilu.h` | Candidate, preconditioner lifecycle surface. |
| `include/sparse_iterative.h` | Prior high-impact header; may already have received cleanup, but remains a complexity reference. |
| `include/sparse_ldlt.h` | Candidate for symmetric-indefinite/direct lifecycle cleanup. |
| `include/sparse_lu.h` | Candidate for direct-solver one-shot/repeated-use cleanup. |
| `include/sparse_lu_csr.h` | Candidate for specialized CSR LU working-format cleanup. |
| `include/sparse_matrix.h` | Prior high-impact core header; may already have received cleanup, but remains first-use reference. |
| `include/sparse_qr.h` | Candidate with sensitive rank/nullspace/minimum-norm claim boundaries. |
| `include/sparse_reorder.h` | Candidate for ordering/analysis workflow cleanup. |
| `include/sparse_svd.h` | Candidate with sensitive partial-SVD evidence and convergence wording. |
| `include/sparse_types.h` | Candidate with shared type/error/macro wording and ABI-risk sensitivity. |
| `include/sparse_vector.h` | Candidate, low current surface area. |

Generated `sparse_version.h` is not a Day 1 cleanup candidate because it is
produced from `include/sparse_version.h.in` during build/install and remains
owned by package/version metadata policy.

## Inherited Claim Boundaries

Header cleanup may clarify:

- caller/callee ownership;
- allocation and free responsibilities;
- lifecycle and invalidation rules;
- error-code and failure semantics;
- tolerance defaults and interpretation;
- workspace and reusable-handle expectations;
- threading and callback behavior;
- recommended local workflow links.

Header cleanup must not imply:

- dynamic ABI stability;
- shared-library build/install support;
- runtime-loader support;
- package-manager support;
- provider registry, tap, recipe, or binary package readiness;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- portable performance superiority;
- broad external-library parity;
- state-of-the-art sparse linear algebra coverage.

## Quality-Gate Baseline

| Change type | Required validation |
| --- | --- |
| Planning/docs only | `git diff --check` |
| Shell guard changed | `bash -n <script>` plus focused script validation |
| Python guard/script changed | focused Python command or script check plus `git diff --check` |
| Public `.h` changed | `make format && make lint && make test` |
| C `.c` changed | `make format && make lint && make test` |
| Package/adoption wording changed | package-manager/static-package deferral guards as applicable |

## Day 1 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Sprint 172 working-notes baseline | Complete | `WORKING_NOTES.md` created. |
| Artifact directory structure | Complete | `artifacts/` now contains this Day 1 artifact. |
| Source artifact note | Complete | Epic 12 prompt mismatch recorded. |
| Header/package/ABI/adoption boundary summary | Complete | Boundaries and stop conditions are recorded above and in working notes. |
| Day 1 header-intake artifact | Complete | This file. |

## Validation

Day 1 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Sprint 172 scope is tied to the active Epic 15 project plan. | Complete | Active planning source is recorded. |
| Public-header cleanup constraints are explicit before selecting a family. | Complete | Candidate set, stop conditions, and quality gates are defined. |
| Package-manager, shared-library, ABI, runtime-loader, and broad platform non-claims remain protected. | Complete | Inherited claim boundaries are listed explicitly. |
