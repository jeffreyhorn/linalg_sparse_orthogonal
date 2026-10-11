# Sprint 213 Day 6: Automation Design

## Summary

Day 6 converts the Day 5 stronger local-only generated API decision into an
implementation-ready automation design. The design keeps generated Doxygen
HTML ignored under `docs/api/`, preserves the source-controlled API route, and
targets guard/test hardening rather than publication exceptions.

No code changes are made on Day 6. Days 7-8 should implement only the missing
fixtures or guard markers identified by this design.

## Automation Layers

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Doxygen generation | `Doxyfile`; `make docs` | Generate local HTML from checked-in public headers under `include/`. |
| Coverage/freshness | `scripts/check_api_docs_coverage.py`; `tests/test_api_docs_coverage.py`; `make docs-check` | Validate generated reference/source pages for checked-in public headers; keep `sparse_version.h` separate. |
| Local-only staging/workflow | `scripts/check_api_docs_local_only.sh`; `tests/test_api_docs_local_only_guard.py` | Keep generated API output ignored, untracked, unstaged, and absent from publication workflows. |
| Routing | `scripts/check_api_docs_routing.py`; `tests/test_api_docs_routing.py` | Keep user-facing routes on source-controlled API docs and reject generated/hosted publication links. |
| Validation wiring | `Makefile` | Preserve serialized `api-docs-freshness` chain. |
| Documentation | README, INSTALL, API reference, maintainer guide | Explain local-only policy, non-claims, repair workflow, and reopening criteria. |

## Path Policy

| Path | Policy |
| --- | --- |
| `docs/api/` | Ignored local generated output; not staged, tracked, hosted, retained, or committed. |
| `docs/api/html/` | Local generated HTML current only after `make api-docs-freshness` passes. |
| `docs/api_reference.md` | Source-controlled API reference entry point. |
| `include/*.h` | Source of truth for public declarations and Doxygen input. |
| `include/sparse_version.h` | Generated installed header outside current Doxygen page expectations. |
| Hosted generated API URL | Rejected for Sprint 213. |
| Generated-doc CI artifact | Rejected for Sprint 213. |
| Committed generated HTML | Rejected for Sprint 213. |

## Fixture Plan

| Fixture | Expected behavior |
| --- | --- |
| Hosted generated API link | Routing guard rejects it unless a future decision adds an exact allowlist. |
| GitHub release or Actions artifact generated-doc URL | Routing guard rejects it as retained/release publication evidence. |
| `actions/upload-artifact` with `docs/api` or `docs/api/html` | Local-only workflow guard fails. |
| Broad `docs/` upload with publication semantics | Local-only workflow guard fails. |
| Tracked or non-ignored generated output | Local-only guard fails. |
| Missing local-only/non-publication wording | Docs guard fails. |
| Source-controlled API routes | Routing guard continues accepting `docs/api_reference.md`, checked-in headers, `Doxyfile`, workflow guides, and INSTALL. |

## Implementation Priorities

1. Inspect current local-only and routing regression suites before adding new
   code.
2. Add only missing high-priority fixtures for hosted links, retained artifact
   links, local-only residual wording, or workflow publication paths.
3. Preserve existing source-controlled API routing.
4. Do not add publication allowlists or workflow upload/deploy paths.
5. Validate with focused API docs tests and `make api-docs-freshness` when
   automation changes are made.

## Validation Order

The selected policy depends on this chain:

```text
docs
  -> api-docs-coverage
  -> docs-check
  -> api-docs-local-only
  -> api-docs-routing
  -> api-docs-validate
  -> api-docs-freshness
```

Generated output must exist before the local-only and routing guards prove it
is fresh, ignored, unstaged, untracked, and not treated as publication.

## Day 6 Outcome

Item 213.3 now has an implementation-ready design. Workflow, routing,
freshness, staging, documentation, and validation responsibilities are
explicitly owned, and Days 7-8 have a bounded fixture plan.

## Validation

Day 6 changed planning documentation only. No `.c` or `.h` files were modified,
so `make format && make lint && make test` is not required.

