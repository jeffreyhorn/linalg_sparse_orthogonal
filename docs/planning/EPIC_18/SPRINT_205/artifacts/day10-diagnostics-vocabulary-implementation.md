# Sprint 205 Day 10: Diagnostics Vocabulary Implementation

**Date:** 2026-09-19  
**Sprint item:** 205.4 Diagnostics Vocabulary  
**Theme:** Apply the Day 9 diagnostics vocabulary to selected public and
maintainer documentation.

## Purpose

Day 10 applied the Day 9 vocabulary decisions to user-facing and maintainer
documentation without changing implementation behavior, API contracts, report
schemas, generated artifacts, or validation commands.

## Changed Surfaces

| Surface | Change | Boundary retained |
| --- | --- | --- |
| `README.md` | Reworded first-use diagnostics and benchmark/report summaries to use problem-local residuals, run-local convergence fields, local measurement artifacts, and selected target language. | Raw report schema fields remain owned by benchmark/report docs; no portable performance or support claim added. |
| `docs/cookbook.md` | Reworded first-use diagnostics and normalized report-index handoff toward workflow-local diagnostics and current generated-output diagnostics. | Cookbook stays a routing aid, not a report schema or support ledger. |
| `docs/solver_selection.md` | Reworded diagnostics handoff and selected evidence paragraphs for direct, iterative, QR, SVD, eigensolver, and selected comparison rows. | Named evidence rows and commands are preserved; no broad correctness, parity, package, platform, performance, or state-of-the-art claim added. |
| `examples/README.md` | Reworded diagnostics handoff to use run-local convergence fields and QR/SVD-local output descriptions. | Examples remain runnable workflow references, not broad diagnostic or support evidence. |
| `docs/tutorial.md` | Reworded diagnostics table and advanced report handoff around problem-local, run-local, QR-local, SVD-local, and generated-output scope. | Tutorial remains a learning path; report checks remain maintainer/advanced evidence. |
| `benchmarks/README.md` | Clarified selected hosted metadata, local measurement artifacts, skip semantics, backend-context rows, and pass/fail scope. | Benchmark rows remain local or selected evidence, not portable performance proof. |
| `docs/api_reference.md` | Reworded generated API freshness language as current-output diagnostics while preserving the `make api-docs-freshness` command. | Generated HTML remains local-only ignored output, not hosted or release evidence. |
| `docs/maintainer_guide.md` | Added diagnostics vocabulary routing guidance for future doc edits. | Maintainer/report schema vocabulary remains allowed where it is defined and scoped. |

## Retained Exceptions

| Exception | Reason |
| --- | --- |
| Exact API names, enum names, field names, and commands remain unchanged. | Day 10 is a documentation vocabulary pass, not an API or behavior change. |
| `status`, `support_tier`, `claim_boundary`, `local_only`, and `hosted_selected` remain in benchmark and maintainer report sections. | These are raw schema/manifest fields and are appropriate where docs explain report interpretation. |
| `pass`, `fail`, and `skip` remain where report or sentinel rows explicitly own those states. | Day 10 clarifies their scope instead of hiding schema-owned status values. |
| "freshness" remains in command names and report-index sections. | Commands and report tools use that term; public-facing interpretation now explains it as current generated-output or selected-target evidence. |

## Vocabulary Applied

- "problem-local residual" for direct and repeated direct examples.
- "run-local convergence fields" for iterative examples.
- "QR-local" and "SVD-local" for rank, residual, condition, and low-rank
  diagnostics.
- "Ritz residual for the requested eigenpairs" for eigensolver diagnostics.
- "local measurement artifact" for benchmark rows.
- "current generated output for the selected gate" for freshness meaning when
  raw report-index vocabulary is not needed.
- "`skip` means optional data or prerequisites were unavailable by policy" for
  report/sentinel scope states.

## Non-Changes

Day 10 did not change:

- production C source or public headers;
- Makefile, CMake, CI workflow, manifest, schema, or generated-output files;
- validation commands or report generation behavior;
- support/readiness status, package/ABI/platform claims, hosted API policy,
  release proof, portable performance claims, or state-of-the-art claims.

## Completion Criteria Review

- Item 205.4 is implemented across the selected documentation surfaces.
- User-facing and maintainer-facing terms are coherent and intentionally
  separated.
- Exact API/report vocabulary is preserved where docs quote implementation or
  schema surfaces.
- No implementation behavior is implied to have changed.
