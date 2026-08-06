# Day 1 Partial-SVD Residual Intake

## Purpose

Day 1 establishes Sprint 140 scope and records the inherited evidence before
fixture, oracle, test, or documentation implementation begins. The sprint
must close one selected partial-SVD residual completely while preserving broad
SVD and partial-SVD non-claims.

This is an intake and planning artifact. It does not change SVD source, tests,
corpus rows, oracle commands, public documentation, or support claims.

## Project-Plan Scope

Sprint 140 implements "Partial-SVD Edge-Case & Convergence Residual Closure"
from `docs/planning/EPIC_12/PROJECT_PLAN.md`.

The sprint goal is to completely close the selected partial-SVD residual with:

- edge-case fixtures;
- comparison and oracle semantics;
- convergence-budget proof;
- proof-owner cleanup;
- focused validation and full gates when code changes require them;
- updated SVD docs, non-claims, and Sprint 141 report-index handoff.

## Handoff Inputs

| Input | Day 1 use |
| --- | --- |
| Sprint 138 corpus architecture | Provides fixture manifests, expected-result rows, oracle/report schema, skip/defer semantics, and generated-output boundaries. |
| Sprint 139 QR closure | Provides the fixture-local closure pattern, generated-reference versus solver-backed row split, proof-owner discipline, residual-safe comparison style, and stale-report guidance. |
| Sprint 139 retrospective | Defines Sprint 140 requirements: partial-SVD-specific fixture keys, expected rows, oracle rows, proof owner, tolerances, ambiguity rules, support tiers, and public wording. |
| `tests/test_svd.c` | Current SVD/partial-SVD test registration and broad proof surface. |
| `tests/test_svd_partial_helpers.h` | Main source of partial-SVD fixture patterns, vector residual metrics, range-projector checks, and convergence-budget behavior. |
| `tests/svd_external_dense_reference.py` | Bounded dense-reference helper for named partial-SVD fixtures. |
| `scripts/run_corpus_oracle.py` | Pattern for maintained corpus/oracle/report generation and local manifest metadata. |
| SVD public and maintainer docs | Claim surfaces to update only after evidence lands. |

## Initial Evidence Inventory

| Evidence family | Existing examples | Intake conclusion |
| --- | --- | --- |
| Singular-value fixtures | `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3`, `partial_svd_nonsym_rect10x8_k3` | Useful bounded patterns, but not yet a maintained Sprint 140 corpus lane. |
| Vector residuals | Helper metrics for `A*v ~= sigma*u` and `A^T*u ~= sigma*v` | Candidate comparison metric for selected residual. |
| Projector/subspace checks | `partial_svd_rankdef_diag6x4_k2_range_projector` | Candidate pattern for rank-deficient range-space closure. |
| Low-rank optimality | `partial_svd_lowrank_diag6x4_k2_frobenius_optimality` and sparse/dense low-rank tests | Existing evidence, but broad low-rank optimality remains a non-claim unless selected. |
| Convergence-budget behavior | `partial_svd_max_iter_fail_closed_diag6_k2` | Strong candidate proof pattern for fail-closed behavior and recovery semantics. |
| Repeated/clustered spectra | Full SVD repeated-value tests and Sprint 138/139 handoff notes | Partial-SVD clustered/repeated fixture closure remains open. |

## Day-Level Ownership Map

| Item | Day owner(s) |
| --- | --- |
| Item 1: Partial-SVD Residual Reaudit | Days 1-3 |
| Item 2: Edge-Case Fixture Batch | Days 4-5 |
| Item 3: Comparison Semantics | Days 6-7 |
| Item 4: Convergence-Budget Tests | Days 8-9 |
| Item 5: Proof-Owner Cleanup | Day 10 |
| Item 6: Validation | Day 12, plus focused checks on implementation days |
| Item 7: Docs and Closeout | Days 11, 13, 14 |

## Initial Closure Criteria

The selected residual is closed only when:

- fixture-local success and failure semantics are documented;
- deterministic fixture rows and expected rows are source controlled;
- solver-backed oracle/report rows are reproducible and parseable;
- focused partial-SVD proof ownership exists;
- convergence-budget behavior is tested without masking non-convergence;
- public and maintainer wording matches the earned evidence;
- generated outputs remain ignored/untracked;
- all required validation passes.

## Initial Non-Claims

Sprint 140 does not start with any claim for:

- broad SVD or partial-SVD correctness;
- raw singular-vector identity;
- broad repeated or clustered singular-value behavior;
- broad rank-deficient null-space or range-space behavior;
- convergence-rate or performance behavior;
- external-library parity;
- optional SuiteSparse or external-data pass evidence;
- platform, package, ABI, shared-library, loader, package-manager, or
  state-of-the-art status.

## Validation Expectations

- Documentation/planning-only days require `git diff --check`,
  trailing-whitespace scans, and focused Markdown link/path validation.
- Corpus/TSV days require schema validation and TSV consistency checks.
- Python oracle/reference changes require `python3 -m py_compile` and focused
  command validation.
- `.c` or `.h` changes require focused SVD/partial-SVD tests and then
  `make format && make lint && make test`.

## Day 1 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 140 project-plan item has a day-level owner. | Complete | Item-to-day map in this artifact and `WORKING_NOTES.md`. |
| Inherited SVD/partial-SVD/corpus evidence is visible before implementation begins. | Complete | Handoff and evidence inventory tables above. |
| Closure criteria distinguish earned fixture-local evidence from broad partial-SVD non-claims. | Complete | Closure criteria and non-claim register above. |
