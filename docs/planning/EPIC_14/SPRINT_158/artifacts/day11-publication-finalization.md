# Day 11 Publication Finalization

## Final Decision

Sprint 158 keeps generated API HTML local-only and ignored.

The finalized publication path is:

- do not commit `docs/api/html/`;
- do not add hosted Doxygen HTML publication in this sprint;
- use `make docs-check` as the maintained local freshness and page-coverage
  guard;
- keep checked-in public headers and `docs/api_reference.md` as the
  source-controlled API reference authority.

## Implemented Guard

| Surface | Final state |
| --- | --- |
| `Makefile` | `docs-check` runs `docs` and then `api-docs-coverage`. |
| `scripts/check_api_docs_coverage.py` | Checks local generated Doxygen output for reference and source pages for checked-in public headers. |
| `docs/api_reference.md` | Describes `make docs-check`, local-only generated HTML, and source-header-first ownership. |
| `docs/maintainer_guide.md` | Defines the maintainer interpretation for local freshness, ignored output, and generated `sparse_version.h` policy. |
| `README.md` | Lists `make docs-check` beside `make docs` in the local command inventory. |

## Tracking And Ignore Evidence

`.gitignore` contains:

```text
# Generated documentation
docs/api/
```

Current status evidence:

```text
!! docs/api/
```

The generated Doxygen output remains ignored local output. It is not staged and
is not part of the source-controlled diff.

## Generated Path Agreement

| Path | Owner | Day 11 interpretation |
| --- | --- | --- |
| `docs/api/html/` | Doxygen output from `make docs` / `make docs-check` | Ignored local convenience HTML. |
| `docs/api_reference.md` | Source-controlled API reference index | Public entry point for exact declaration routing. |
| `include/*.h` | Checked-in public headers | Exact declaration and call-site contract source of truth. |
| generated `sparse_version.h` | Build/install output from `VERSION` and `include/sparse_version.h.in` | Installed-header policy row; not an expected Doxygen page under current input. |

## Support-Claim Boundary

The finalized path does not claim:

- hosted generated API documentation;
- source-controlled generated HTML;
- release evidence from ignored local output;
- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad platform parity;
- external-library parity;
- portable performance;
- completeness beyond the configured Doxygen input set.

## Validation

Day 11 changed documentation only. Validation focused on generated-doc
freshness and policy consistency:

```text
make docs-check
git diff --check
```

Results:

- `make docs-check` passed with complete generated page coverage for the 18
  checked-in public headers.
- `git diff --check` passed.
- Trailing-whitespace scan over README, API reference, maintainer guide, and
  Sprint 158 planning artifacts passed.
- `git status --ignored=matching docs/api` still reports `!! docs/api/`.

## Day 12 Handoff

Day 12 should run the broader validation evidence pass for the touched sprint
surface, including `make docs-check`, docs hygiene checks, and the full
`make format && make lint && make test` gate because Sprint 158 has edited
public headers earlier in the branch.
