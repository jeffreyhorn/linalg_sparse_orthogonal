# Day 12: CI And Maintenance Surface Review

## Purpose

Reconcile CI, local commands, maintainer guidance, report/freshness ownership,
and residuals for the selected generated API HTML path.

## Owner Map

| Surface | Owner | Current role |
| --- | --- | --- |
| `Doxyfile` | Doxygen configuration | Defines checked-in public-header input and `docs/api/html/` output. |
| `make docs` | Makefile | Raw local Doxygen generation. |
| `make api-docs-coverage` | Makefile plus `scripts/check_api_docs_coverage.py` | Generated page coverage for checked-in public headers. |
| `make docs-check` | Makefile | Local generation plus page coverage. |
| `make api-docs-local-only` | Makefile plus `scripts/check_api_docs_local_only.sh` | Local-only ignore/tracking/staging boundary check. |
| `make api-docs-validate` | Makefile | `docs-check` plus local-only guard. |
| `make api-docs-freshness` | Makefile | Selected generated API freshness command. |
| `docs/api_reference.md` | API docs entry point | Source-controlled API reference and generated HTML interpretation. |
| `docs/maintainer_guide.md` | Maintainer policy | Generated API freshness/local-only guidance and non-claims. |
| `.gitignore` | Generated-output policy | Keeps `docs/api/` ignored. |

## Local Command Chain

The selected local generated API freshness command is:

```bash
make api-docs-freshness
```

It expands through:

```text
api-docs-freshness -> api-docs-validate -> docs-check + api-docs-local-only
docs-check -> docs + api-docs-coverage
docs -> doxygen Doxyfile
api-docs-coverage -> python3 scripts/check_api_docs_coverage.py
api-docs-local-only -> bash scripts/check_api_docs_local_only.sh
```

This command chain matches the Day 4 local-only decision:

- generated HTML is regenerated locally;
- expected public-header pages are checked;
- generated HTML remains ignored, untracked, unstaged, and not visible as
  non-ignored untracked output.

## CI Surface

Day 12 reviewed `.github` references through the generated-doc search surface.
No current workflow publishes, uploads, deploys, or otherwise promotes Doxygen
HTML under `docs/api/html/`.

Sprint 173 therefore does not add:

- hosted generated API HTML;
- artifact-only generated API HTML;
- generated API HTML retention policy;
- generated API HTML deployment permissions;
- hosted generated API freshness claims.

A future CI docs-check lane may run `make api-docs-freshness`, but that would
still be a check unless a separate publication decision selects upload or
hosting.

## Report And Freshness Metadata Surface

Day 12 found no report-family metadata row that treats generated API HTML as a
hosted, committed, artifact-only, package, ABI, platform, performance,
external-parity, or state-of-the-art proof surface.

That is correct for Sprint 173. Generated API HTML is local documentation
output, not a report-family evidence artifact.

## Documentation Surface

The public and maintainer docs now say:

- `README.md` lists `make api-docs-freshness` in the command surface;
- `docs/api_reference.md` says generated HTML is current only after
  `make api-docs-freshness` passes;
- `docs/maintainer_guide.md` says `make api-docs-freshness` runs `docs-check`
  plus local-only generated-output staging enforcement;
- generated HTML remains not source-controlled, hosted, artifact-published, or
  release evidence.

## Residuals

| Residual | Status | Rationale |
| --- | --- | --- |
| Hosted generated API HTML | Deferred | No hosted lane, URL ownership, deployment policy, or retention policy selected. |
| CI artifact-only generated API HTML | Deferred | No artifact upload/retention policy selected. |
| Committed generated API HTML | Rejected for Sprint 173 | `docs/api/` remains ignored to avoid generated-review churn. |
| CI docs-check lane | Deferred | Local freshness command exists; CI integration would be a separate workflow decision. |
| `sparse_version.h` Doxygen page | Deferred/non-goal | Installed generated header remains owned by install/version validation. |
| Persisted generated API freshness metadata | Deferred/non-goal | Local freshness regenerates before checking; source-controlled metadata for ignored output risks overclaim. |

## Claim Boundary Review

Day 12 confirms no Sprint 173 maintenance surface adds support for:

- package-manager provider availability;
- shared-library support;
- dynamic ABI stability;
- runtime-loader behavior;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad platform parity;
- portable performance guarantees;
- external-library parity;
- state-of-the-art sparse linear algebra coverage.

## Day 13 Handoff

Day 13 should run an integrated claim review over:

- Day 4 publication decision;
- Day 6 implementation;
- Day 8 freshness target;
- Day 10 docs updates;
- Day 11 validation;
- this Day 12 owner/residual map.

The likely Sprint 174 handoff is: keep generated API HTML local-only unless a
future sprint explicitly funds hosted or CI artifact publication.

## Completion Check

Day 12 completion criteria are met:

- maintainers know `make api-docs-freshness` owns generated API status;
- CI/local behavior matches the local-only publication decision;
- residuals are explicit and bounded.

No `.c` or `.h` files changed on Day 12, so the full C quality gate is not
required for this day.
