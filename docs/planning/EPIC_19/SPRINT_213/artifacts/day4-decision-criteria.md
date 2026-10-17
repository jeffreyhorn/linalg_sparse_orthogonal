# Sprint 213 Day 4: Decision Criteria And Stop Conditions

## Summary

Day 4 defines the acceptance gates for the Sprint 213 generated API policy
decision. The criteria intentionally separate a safe local-only closure from
three publication paths: hosted generated API HTML, retained generated-doc
artifacts, and committed generated HTML.

No policy is selected on Day 4. Day 5 must choose exactly one path using these
criteria.

## Decision Rule

Day 5 must choose one of:

1. stronger local-only closure;
2. hosted generated API publication;
3. retained generated-doc artifact publication;
4. committed generated HTML.

If no publication path satisfies all required criteria, Day 5 should select
stronger local-only closure and explicitly record why hosted, retained, and
committed generated output remain rejected.

## Stronger Local-Only Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Validation passes | `make api-docs-freshness` passes with current coverage/routing/local-only checks. |
| Ignored generated output | `docs/api/` remains ignored, untracked, and unstaged. |
| Workflow publication rejection | Workflow scans continue rejecting generated API publication, broad docs paths, archives, dynamic paths, and upload/deploy bypasses. |
| Source-controlled route | User docs route to `docs/api_reference.md` and checked-in headers, not generated HTML. |
| Non-claims | Docs preserve no hosted publication, no retained generated-doc artifact, no committed generated HTML, no release evidence, no package/ABI claim, and no completeness beyond Doxyfile-selected headers. |
| Residual closure | Docs explain why generated HTML remains local-only and what would be needed to reopen publication. |

## Hosted Publication Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Hosted URL | A single project-owned generated API URL is selected. |
| Deployment ordering | Deployment occurs only after Doxygen generation, coverage/freshness, adapted staging checks, and routing validation. |
| Permissions | Workflow permissions and deployment settings are explicit and least-privilege. |
| Stale-output handling | Maintainer docs define rollback, disablement, or stale warning behavior. |
| Routing allowlist | Routing guard permits only the selected hosted route and rejects near-misses, release artifact URLs, generic generated-doc hosts, and `docs/api/` links. |
| Claim boundary | Docs/tests prevent hosted docs from implying API stability, release readiness, package-manager distribution, ABI support, broad platform parity, or completeness beyond Doxyfile input. |

## Retained Artifact Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Exact artifact scope | Artifact name and paths are narrow and exclude broad `docs/` or repository root uploads. |
| Retention | `retention-days` is explicit and documented. |
| Upload ordering | Upload follows Doxygen generation and freshness checks. |
| Retrieval wording | Docs state who should use the artifact, when it expires, and why it is not hosted docs or release evidence. |
| Guard replacement | Local-only/workflow guard allows only the selected artifact shape and keeps broad publication bypasses rejected. |

## Committed Generated HTML Acceptance Criteria

| Criterion | Required proof |
| --- | --- |
| Review burden accepted | The sprint accepts committing generated output currently measured at 214 files and about 3.1 MB. |
| Freshness enforcement | A guard proves committed generated pages match current checked-in headers and `Doxyfile`. |
| Ignore policy revised | `.gitignore`, staging checks, and docs wording are intentionally updated. |
| Ownership clear | Docs explain source/header precedence and generated-output review workflow. |
| Non-claims retained | Committed generated HTML still does not imply package, ABI, release, broad platform, performance, or state-of-the-art support. |

## Stop Conditions

Stop and ask for user direction if:

- the selected publication path requires repository settings, DNS, secrets, or
  external state unavailable on the branch;
- publication requires weakening existing guards without a replacement guard;
- artifact retention or audience is ambiguous;
- committed generated HTML is requested without accepting generated-output
  review noise and repository growth;
- Doxygen generation or `make api-docs-freshness` fails before implementation;
- docs cannot preserve unsupported package, ABI, release, platform,
  performance, or state-of-the-art non-claims;
- required validation fails.

## Evidence-To-Test Mapping

| Criteria family | Enforcement path |
| --- | --- |
| Coverage and freshness | `scripts/check_api_docs_coverage.py`, `tests/test_api_docs_coverage.py`, `make docs-check`. |
| Local-only and workflow staging | `scripts/check_api_docs_local_only.sh`, `tests/test_api_docs_local_only_guard.py`, `make api-docs-freshness`. |
| Routing and hosted/publication links | `scripts/check_api_docs_routing.py`, `tests/test_api_docs_routing.py`, `make api-docs-freshness`. |
| Publication-specific exceptions | New fixtures for exact hosted URL, exact artifact path, or committed-output freshness depending on Day 5 decision. |
| Documentation boundary | Required text and forbidden-claim/link markers in routing/local-only/docs guard tests. |

## Day 4 Outcome

Item 213.2 now has explicit decision criteria before selection. Day 5 can
evaluate each publication option against testable gates and select the only
policy the branch can safely support.

## Validation

Day 4 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

