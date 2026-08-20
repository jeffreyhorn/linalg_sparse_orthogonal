# Day 4: Publication Decision Record

## Purpose

Select the supported generated API HTML publication path for Sprint 173 and
define the claims, non-claims, checks, and implementation scope that follow
from that decision.

## Final Decision

Sprint 173 selects **guarded local-only generated API HTML with strengthened
freshness/staging enforcement**.

| Decision field | Sprint 173 decision |
| --- | --- |
| Generated HTML tracking | Keep `docs/api/` ignored. |
| Generated HTML source-control status | Do not commit `docs/api/html/`. |
| Hosted generated HTML | Not selected. |
| CI artifact-only generated HTML | Not selected. |
| Supported user entry point | `docs/api_reference.md` plus checked-in public headers under `include/`. |
| Local generated view | `make docs-check` generates and checks `docs/api/html/` for the active checkout. |
| Freshness claim | Local generated HTML is current only for the branch/checkout where the selected checks just passed. |
| Installed version header policy | `sparse_version.h` remains install/version owned, not an expected Doxygen page under current input policy. |

This decision continues the Sprint 158 local-only product posture. Sprint 173
does not add a new generated API publication surface; it tightens enforcement
so the inherited local-only surface is harder to overclaim or stage
accidentally.

## Supported Claims

Sprint 173 may support these claims after the required checks pass:

- checked-in public headers under `include/` remain the exact declaration and
  call-site contract source of truth;
- `docs/api_reference.md` remains the source-controlled API reference entry
  point;
- `make docs-check` is the selected local command for regenerating Doxygen HTML
  and checking generated page coverage for checked-in public headers;
- local generated HTML under `docs/api/html/` is a convenience navigation view
  for the active checkout after `make docs-check` passes;
- `scripts/check_api_docs_coverage.py` checks expected Doxygen reference and
  source pages for the configured checked-in public-header input set;
- `include/sparse_version.h` remains a generated installed header governed by
  install/version validation, not current Doxygen page coverage.

## Unsupported Claims

Sprint 173 does not support these claims:

- generated API HTML is hosted;
- generated API HTML is committed or source-controlled;
- generated API HTML is available as a maintained CI artifact;
- ignored local generated HTML is release evidence;
- generated API HTML is fresh without a just-passed selected local check;
- generated API HTML covers generated installed headers such as
  `sparse_version.h`;
- Doxygen page coverage proves broader API completeness beyond configured
  checked-in headers;
- generated docs imply package-manager provider availability;
- generated docs imply shared-library build/install support;
- generated docs imply dynamic ABI stability;
- generated docs imply runtime-loader behavior support;
- generated docs imply Windows Makefile parity or Windows `pkg-config`
  execution parity;
- generated docs imply broad platform parity;
- generated docs imply portable performance guarantees;
- generated docs imply external-library parity;
- generated docs imply state-of-the-art sparse linear algebra coverage.

## Required Checks For Selected Mode

| Check | Required behavior | Owner surface |
| --- | --- | --- |
| Generator command check | `make docs-check` remains the selected local generator plus page-coverage command. | `Makefile`, `Doxyfile`, `scripts/check_api_docs_coverage.py` |
| Page coverage check | Expected checked-in `include/*.h` reference and source pages must exist after generation. | `scripts/check_api_docs_coverage.py` |
| Ignored-output check | `docs/api/` must remain ignored unless a future sprint selects committed output. | `.gitignore`, focused staging guard |
| Staging check | Generated files under `docs/api/` must not be staged or tracked by accident. | focused staging guard |
| Freshness wording check | Docs must say local generated HTML is current only after the selected local check passes in the active checkout. | `docs/api_reference.md`, `docs/maintainer_guide.md`, README if touched |
| Non-claim scan | Documentation changes must not promote hosted, committed, artifact-only, package, ABI, platform, performance, external-parity, or state-of-the-art claims. | docs and focused guards |
| Deferral guards | Package or ABI wording changes require the Sprint 170/Sprint 171 deferral guards. | `scripts/static_package_deferral_check.sh`, `scripts/package_manager_deferral_check.sh` |

## Day 5 Through Day 9 Implementation Scope

| Day | Scope authorized by this decision | Explicitly out of scope |
| --- | --- | --- |
| Day 5 | Design generator command normalization and staging/freshness behavior for local-only output. | Hosted docs, committed generated HTML, artifact upload. |
| Day 6 | Implement selected command/enforcement changes if needed. | Changing Doxygen input policy without a new decision. |
| Day 7 | Design source-to-local-output freshness and ignored-output staging checks. | Treating ignored output as source-controlled evidence. |
| Day 8 | Implement freshness/staging guard and local check integration. | Publishing generated HTML. |
| Day 9 | Design docs navigation updates for the selected local-only surface. | Adding hosted/generated artifact links. |

Any future move to hosted, committed, or artifact-only generated API HTML needs
a new decision record with deployment, retention, staging, and claim-boundary
evidence.

## Publication Mode Dispositions

| Mode | Day 4 disposition | Reason |
| --- | --- | --- |
| Hosted site | Deferred | No hosted Doxygen lane, URL ownership, deployment permission, or retention policy exists. |
| Committed HTML | Rejected for Sprint 173 | Current generated tree is ignored and large enough to create review churn without improving source API truth. |
| CI artifact-only | Deferred | No workflow lane exists and artifact retention/discovery semantics are undefined. |
| Guarded local-only | Selected | Matches repository policy, keeps reviews source-focused, and supports stronger local freshness/staging checks. |

## Validation Requirements By Touched Surface

| Touched surface | Required validation |
| --- | --- |
| Planning docs only | `git diff --check`. |
| Public docs/navigation | `git diff --check`; targeted claim scan; deferral guards if package/ABI/platform wording changes. |
| Make/Doxygen/script generated-doc checks | Run the focused generated-doc command/check and `git diff --check`. |
| Public headers | `make format && make lint && make test`; `make docs-check`; generated-output staging check. |
| Generated HTML output | Must remain ignored and unstaged unless a future decision selects committed output. |

## Decision Rationale

Guarded local-only remains the best fit because:

- the repository already treats `docs/api/` as generated ignored output;
- source-controlled truth already lives in `docs/api_reference.md` and
  checked-in public headers;
- `make docs-check` already gives maintainers a reproducible local generated
  view and page-coverage check;
- Day 2 found no CI or hosted publication infrastructure;
- committing 214 local generated files would add review churn without closing a
  broader product claim;
- artifact-only and hosted modes need retention, URL, deployment, and
  support-tier work outside the evidence currently present.

## Completion Check

Day 4 completion criteria are met:

- exactly one publication path is selected: guarded local-only generated API
  HTML;
- the selected path has clear validation requirements;
- unselected hosted, committed, and artifact-only publication modes remain
  non-claims.

No `.c` or `.h` files changed on Day 4, so the full C quality gate is not
required for this day.
