# Sprint 213 Day 5: Product Policy Decision

## Decision

Sprint 213 will implement **stronger local-only generated API closure**.

Generated Doxygen HTML remains ignored local output under `docs/api/html/`.
Sprint 213 will not publish hosted generated API HTML, retain generated-doc CI
artifacts, or commit generated HTML. The remaining implementation work should
strengthen the local-only policy with guard coverage, documentation, and
future reopening criteria.

## Selected Scope

| Field | Selected policy |
| --- | --- |
| Policy | Stronger local-only generated API closure |
| Source-controlled API route | `docs/api_reference.md` plus checked-in public headers under `include/` |
| Generated output root | `docs/api/` |
| Generated HTML path | `docs/api/html/` |
| Support tier | `local_only` |
| Freshness command | `make api-docs-freshness` |
| Coverage scope | Checked-in public headers selected by `Doxyfile` |
| Publication status | No hosted API publication, retained generated-doc artifact, or committed generated HTML |
| Routing policy | User-facing docs route through source-controlled API docs and headers |

## Rationale

Day 2 proved the current local-only chain passes:

- `make api-docs-freshness` passes;
- Doxygen generated local HTML for 18 checked-in public headers;
- coverage found 18 generated reference pages and 18 generated source pages;
- `docs/api/` remains ignored output only;
- routing checks keep the source-controlled API entry point at
  `docs/api_reference.md`;
- current workflows do not publish generated API output paths.

Day 4 criteria show that publication paths are not yet safe enough to select:

- hosted publication lacks a selected hosted URL, deployment permission model,
  stale-site rollback policy, and route allowlist;
- retained artifacts lack a selected retention/audience policy and exact
  upload exception;
- committed generated HTML would require accepting persistent generated output
  churn for a current 214-file, about 3.1 MB generated tree.

Stronger local-only closure resolves the generated API publication ambiguity
without creating a hosted, retained, or committed generated-output surface that
the branch cannot fully govern.

## Rejected Alternatives

| Alternative | Reason rejected |
| --- | --- |
| Hosted generated API HTML | Requires hosted URL, deployment settings, permissions, stale-site handling, route allowlist, and publication-specific docs that are not selected for this sprint. |
| Retained generated-doc artifact | Requires exact artifact audience, retention, upload path, retrieval workflow, and non-release wording that are not selected for this sprint. |
| Committed generated HTML | Conflicts with current ignore/local-only policy and adds persistent generated review noise for the current generated tree. |
| Broad generated API publication | Out of scope and would imply unsupported API stability, package, ABI, release, or platform claims. |

## Required Non-Claims

Sprint 213 implementation must not claim:

- hosted API publication;
- retained generated-doc artifacts;
- committed generated HTML;
- release evidence;
- package-manager distribution;
- package, ABI, shared-library, dynamic-loader, or runtime compatibility;
- broad Windows or platform parity;
- external-library parity;
- portable performance;
- state-of-the-art coverage;
- completeness beyond checked-in public headers selected by `Doxyfile`;
- generated installed-header Doxygen coverage for `sparse_version.h`.

## Implementation Direction

| Surface | Direction |
| --- | --- |
| Automation | Preserve local-only validation and identify any missing fail-closed bypass fixtures. |
| Local-only guard | Keep `docs/api/` ignored/untracked/unstaged and reject workflow publication/staging semantics. |
| Routing guard | Keep user-facing routes on `docs/api_reference.md`, public headers, `Doxyfile`, workflow guides, and INSTALL. |
| Documentation | Explain that generated HTML remains local-only by policy and identify evidence needed before future hosted/artifact/committed output. |
| Maintainer guide | Add repair workflow and future reopening criteria for generated API publication. |
| Residuals | Close Sprint 213 as local-only policy closure while leaving hosted, retained, and committed output as future options only if Day 4 criteria are satisfied. |

## Day 5 Outcome

Item 213.2 is complete with a documented product policy. Days 6-12 should
design, implement, test, and document stronger local-only generated API closure
rather than publication exceptions.

## Validation

Day 5 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

