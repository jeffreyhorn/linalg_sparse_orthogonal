# Sprint 205 Day 9: Diagnostics Vocabulary Design

**Date:** 2026-09-19  
**Sprint item:** 205.4 Diagnostics Vocabulary  
**Theme:** Define consistent result, residual, convergence, and status wording
before editing public and maintainer documentation.

## Purpose

Day 9 defines the vocabulary model for diagnostics wording across solver docs,
examples, benchmark/report docs, and maintainer-only validation notes. This is
a design artifact only: it does not change API behavior, status enums, report
schemas, generated artifacts, or validation commands.

## Term Groups

| Group | Public surfaces | Maintainer surfaces | Current risk |
| --- | --- | --- | --- |
| Direct solver return codes | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `examples/README.md` | `docs/maintainer_guide.md`, selected comparison reports | Users may read residual examples as broad correctness proof. |
| Iterative convergence | `docs/solver_selection.md`, `examples/README.md`, `benchmarks/README.md` | iterative test evidence and benchmark reports | `converged`, `stagnated`, and `breakdown` can sound like support tiers unless scoped to a run. |
| QR/SVD rank and residuals | `README.md`, `docs/solver_selection.md`, `examples/README.md`, `docs/api_reference.md` | selected oracle/comparison manifests, maintainer guide | Rank/residual wording can imply broad parity, raw basis identity, or external-library proof. |
| Eigensolver diagnostics | `docs/solver_selection.md`, `examples/README.md`, `benchmarks/README.md` | maintainer guide and eigensolver tests | Ritz residuals and backend fields can be misread as portable backend superiority. |
| Benchmark/report metadata | `README.md`, `benchmarks/README.md` | `docs/maintainer_guide.md`, manifest/schema docs | `status`, `support_tier`, `claim_boundary`, and `freshness` are too dense for first-use docs. |
| Validation guards and deferrals | `INSTALL.md`, `docs/api_reference.md`, `benchmarks/README.md` | guard scripts, workflow docs, residual queues | `skip`, `defer`, `local_only`, and `hosted_selected` can be mistaken for pass/fail status. |

## Preferred Vocabulary Table

| Concept | Public preferred wording | Maintainer/report wording | Avoid in public docs unless quoted | Example use |
| --- | --- | --- | --- | --- |
| Successful local solve | "the local solve completed" plus the checked return code or residual | `SPARSE_OK`, passing fixture row, `status=pass` where schema-owned | "proved solver correctness" | "Check the factorization return code and problem-local residual." |
| Direct residual | "problem-local residual" | `residual_norm`, fixture tolerance, comparison row residual | "accuracy guarantee" | "Residuals describe the shown system only." |
| Iterative convergence | "this run converged" or "the result reports convergence" | `sparse_iter_result_t.converged`, iteration count, final relative residual | "solver support passed" | "Inspect convergence, stagnation, breakdown, and final residual together." |
| Iterative non-convergence | "did not converge within the configured budget" | `SPARSE_ERR_NOT_CONVERGED`, result fields, fail-closed fixture | "failure" without context | "Tight-budget non-convergence is a configured-budget diagnostic." |
| Stagnation/breakdown | "run-local stagnation/breakdown field" | `stagnated`, `breakdown`, residual history | "unsupported solver" | "Use these fields before changing tolerance or preconditioner." |
| QR rank/nullspace | "QR-local rank/nullity/nullspace diagnostic" | selected QR row, tolerance-local rank/nullspace residual | "broad rank-deficient solve claim" | "The row is fixture-local and tolerance-local." |
| SVD rank/condition | "SVD-local rank, condition, triplet residual, or low-rank diagnostic" | selected SVD/partial-SVD rows, projector residuals | "raw singular-vector identity" | "Use rank/condition output as workflow diagnostics." |
| Eigensolver residual | "Ritz residual for the requested eigenpairs" | `result.residual_norm`, backend, peak basis size | "backend superiority" | "Backend fields are diagnostics, not performance claims." |
| Benchmark measurement | "local measurement artifact" | `status=measurement`, `local_threshold_free` | "performance proof" | "Benchmark rows are configuration-sensitive." |
| Selected hosted evidence | "reviewed selected hosted freshness for the named target" | `support_tier=hosted_selected`, selected workflow/job/artifact metadata | "platform support" or "portable speed" | "Hosted selected evidence applies only to the selected target." |
| Fresh generated report | "current generated output for the selected gate" | `fresh`, source commit, artifact path, manifest row | "release proof" | "Freshness means generated outputs match current inputs." |
| Stale/missing report | "out-of-date or missing generated output" | `stale`, `missing`, `duplicate`, `error` diagnostics | "unsupported" | "Regenerate or inspect the selected freshness command." |
| Skip | "optional data or prerequisite unavailable by policy" | `skip` row | "pass" | "Skips explain absence of optional evidence." |
| Defer | "intentionally handed off to future work or residual queue" | `defer`, residual queue entry | "fail" or "pass" | "Defers are scope notes, not passing evidence." |
| Local-only generated docs | "local-only ignored generated HTML" | `local_only`, `make api-docs-freshness` | "published docs" | "Use source-controlled `docs/api_reference.md` for public routing." |
| Unsupported package/ABI claim | "not claimed" or "unsupported by current support matrix" | support matrix non-claim, guard marker | "failed" | "No shared-library or dynamic ABI support is claimed." |

## Public Versus Maintainer Wording

Public docs should use user-centered workflow words:

- "local solve", "problem-local residual", "this run", "chosen workflow",
  "runnable example", "local measurement", "selected target", "not claimed";
- "support/readiness matrix" for current package/platform/API/support status;
- "benchmark/report interpretation" for measurement and generated report rows.

Maintainer docs may use schema and guard vocabulary when the surrounding
section explains it:

- `status`, `support_tier`, `claim_boundary`, `freshness_policy`;
- `local_only`, `hosted_selected`, `local_threshold_free`,
  `hosted_selected_threshold_free`;
- `skip`, `defer`, `pass`, `fail`, `fresh`, `stale`, `missing`,
  `duplicate`, `error`.

Do not copy raw manifest/schema vocabulary into README, tutorial, cookbook, or
examples unless the text is explicitly teaching report-index interpretation.

## Replacement Target List For Day 10

| Surface | Target wording to review | Preferred Day 10 action |
| --- | --- | --- |
| `README.md` | First-use diagnostics, benchmark caveats, selected report wording | Replace broad "status" or "freshness" wording with local/selected evidence phrasing where needed. |
| `docs/tutorial.md` | diagnostics and benchmark handoffs | Keep workflow-local diagnostics and support-matrix routing; avoid schema terms. |
| `docs/cookbook.md` | quick reference boundaries and measurement handoff | Ensure "local diagnostics" and "not portable performance proof" stay consistent. |
| `docs/solver_selection.md` | diagnostics handoff, QR/SVD/eigs evidence boundaries, selected comparison rows | Apply the preferred residual/convergence/rank/freshness terms without weakening named evidence. |
| `examples/README.md` | diagnostics handoff and example sections | Keep output descriptions example-local and move broad interpretation to solver selection. |
| `benchmarks/README.md` | reading benchmark results, report index handoff, sentinel status wording | Distinguish measurement, threshold gates, selected freshness, skips, and defers. |
| `docs/api_reference.md` | local generated API freshness wording | Keep local-only generated HTML and source-controlled route wording. |
| `docs/maintainer_guide.md` | normalized report diagnostics and support truth routing | Preserve schema vocabulary, but keep it scoped as maintainer/report terminology. |

## Risks And Guardrails

| Risk | Why it matters | Guardrail |
| --- | --- | --- |
| Changing "failure" to "unsupported" in solver docs | Could imply the API does not support a workflow when a run merely failed or did not converge. | Use return-code or configured-budget language. |
| Calling selected freshness a "pass" | Could imply broad support or performance proof. | Use "selected freshness for the named target" unless the schema row is explicitly a pass/fail row. |
| Removing raw schema terms from maintainer docs | Could make report interpretation less precise. | Keep schema terms in maintainer/report sections with definitions. |
| Using "fresh" in public docs without context | Could sound like release or support proof. | Say "current generated output for the selected gate" or route to report docs. |
| Treating skip/defer as pass/fail | Could overstate optional or residual evidence. | Define skip/defer as scope states, not success states. |
| Over-normalizing residual wording | Could hide solver-specific meanings. | Preserve direct, iterative, QR, SVD, and Ritz residual distinctions. |

## Completion Criteria Review

- Item 205.4 has a concrete vocabulary table for Day 10 implementation.
- Preferred terms remain consistent with public return-code and result-field
  APIs.
- Public and maintainer vocabulary are separated so first-use docs do not
  inherit raw manifest/schema terminology.
- The design preserves existing evidence scope and does not imply new behavior,
  stronger validation evidence, package support, platform support, portable
  performance, release proof, or state-of-the-art claims.
