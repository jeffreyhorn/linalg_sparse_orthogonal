# Sprint 185 Day 1: Review-Surface Intake

## Purpose

Establish Sprint 185 scope, carry forward the Sprint 177 review-surface
inventory, identify current candidate clusters, and record guardrails before
any extraction design or cluster selection begins.

## Project-Plan Boundaries

| Item | Day 1 interpretation |
| --- | --- |
| 185.1 Cluster Selection | Day 1 prepares the candidate set; it does not select the cluster. |
| 185.2 Extraction Design | Future extraction must define helper boundaries, fixture ownership, and no-behavior-change validation before moving code. |
| 185.3 Mechanical Extraction | Extraction should be limited to the selected cluster and should update build/test registration only as needed. |
| 185.4 Drift Guard Update | Existing source-list and CMake/Make parity checks should be reused or extended after the selected cluster is known. |
| 185.5 Maintenance Note | The selected cluster needs contribution guidance after extraction. |
| 185.6 Validation | Header/source changes require the full C gate; planning-only days require `git diff --check`. |

## Prior Evidence Reviewed

| Evidence | Sprint 185 use |
| --- | --- |
| `SPRINT_177/artifacts/day4-surface-inventory.md` | Provides the inherited large review-surface list and duplicated registration risks. |
| `SPRINT_177/artifacts/day7-target-selection.md` | Confirms Sprint 185 should select exactly one cluster and preserve behavior/build registration. |
| `SPRINT_177/RETROSPECTIVE.md` | Carries the risk that large review surfaces slow review and should be reduced one bounded cluster at a time. |
| `SPRINT_139/artifacts/day14-closeout-validation-summary.md` | Shows a prior pattern for adding a focused proof-owner test and Make/CMake registration without weakening broad coverage. |

## Current Large-File Snapshot

| File | Current lines | Candidate interpretation |
| --- | ---: | --- |
| `tests/test_qr.c` | 3970 | Largest current test. High review cost, but recent QR work and existing `test_qr_corpus` proof owner may lower urgency. |
| `tests/test_ldlt_csc.c` | 3915 | Very large direct-solver test surface with possible helper seams around row adjacency, native pivots, solve, and dense-reference fixtures. |
| `tests/test_integration.c` | 3279 | Broad lifecycle/progress/refactor integration owner. High review cost and high scope-expansion risk. |
| `tests/test_svd.c` | 3029 | Large SVD proof owner with existing helper-header precedent for partial-SVD areas. |
| `tests/test_ldlt.c` | 3006 | Large LDLT family-local proof surface with KKT, reorder, backend, refinement, and dense-helper tests. |
| `tests/test_etree.c` | 2962 | Large ordering/analysis test surface with possible fixture/helper extraction candidates. |
| `tests/test_iterative.c` | 2929 | Large iterative proof owner, but allocation-failure gates make behavior-preservation risk higher. |
| `tests/test_graph.c` | 2764 | Large graph/FM/partition surface with existing graph fixture helper ownership. |
| `tests/test_chol_csc.c` | 2554 | Large Cholesky CSC proof surface, though lower by line count than QR/LDLT CSC/integration/SVD. |
| `src/sparse_ldlt_csc.c` | 2095 | Large implementation file; source extraction has higher behavior and registration risk. |
| `scripts/run_external_comparison.py` | 2094 | Large report tooling owner; candidate only if tooling review-surface reduction is selected. |
| `tests/test_normalize_report_index.py` | 1861 | Large report-index test owner with many drift diagnostics. |
| `docs/maintainer_guide.md` | 1761 | Large documentation surface; lower fit for this sprint unless maintenance-note extraction becomes the selected scope. |

## Registration And Guard Inventory

| Guard or registration surface | Current owner |
| --- | --- |
| Library source order and membership | `build-metadata/library_sources.txt`, `Makefile` `LIB_SRCS`, `CMakeLists.txt` `add_library(...)` |
| Library source-list drift check | `scripts/check_library_sources.py`, exposed as `make source-list-check` |
| Make test binary registration | `Makefile` `TEST_SRCS` and derived `TEST_BINS` |
| CMake test binary registration | `CMakeLists.txt` `add_sparse_test(...)` |
| Make/CMake test-count parity | `make quality-review-cmake-compile` |
| Formatting and lint coverage | `make format`, `make format-check`, `make lint` |
| Full behavior regression | `make test` |
| Focused cluster validation | `make build/<test>` followed by `./build/<test>` for selected C test binaries |

## Candidate Selection Guardrails

- Select one cluster only.
- Prefer low-risk helper or fixture extraction over solver implementation
  extraction unless Day 2-3 evidence shows implementation extraction is safer.
- Preserve existing test semantics, fixture data, tolerance values, public API
  contracts, and generated evidence boundaries.
- Treat build/test registration as part of the extraction, not a follow-up.
- Use source-list and test-count parity checks whenever registration changes.
- Keep generated build/report/API output unstaged.

## Day 2 Handoff

Day 2 should turn the Day 1 candidate list into a scored baseline:

1. group each candidate by responsibilities, helper seams, registration needs,
   and focused validation commands;
2. distinguish review-cost score from refactor-risk score;
3. identify one or two strongest candidates for the Day 3 selection decision;
4. avoid selecting a cluster solely because it has the highest line count.

## Validation

Day 1 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.
