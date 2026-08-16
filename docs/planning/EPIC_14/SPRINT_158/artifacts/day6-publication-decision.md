# Day 6 Publication Decision

## Scope

Day 6 converts the Day 5 recommendation into a concrete Sprint 158 product
decision and implementation checklist.

This artifact finalizes the generated API HTML policy for Sprint 158. It does
not edit `.gitignore`, CI workflows, Doxygen configuration, public docs, public
headers, or generated HTML.

## Final Decision

| Decision field | Sprint 158 decision |
| --- | --- |
| Generated HTML tracking | Keep `docs/api/` ignored. |
| Generated HTML publication | Do not commit `docs/api/html/` in Sprint 158. |
| CI publication | Do not add Doxygen HTML artifact upload, hosted pages, or deployment in Sprint 158. |
| Product policy | Guarded local-only generated API HTML. |
| Source-controlled API truth | `docs/api_reference.md` plus checked-in public headers under `include/`. |
| Freshness evidence | A local `make docs` run plus warning closure and page-coverage guard result. |
| Generated version header | Separate installed-header/version metadata policy row; not an expected checked-in public header page. |

Sprint 158 therefore closes the generated API HTML residual with an explicit
no-commit/local-only product decision plus recurring guard, not with committed
or hosted generated HTML.

## File-Change Checklist

| Surface | Expected action | Rationale |
| --- | --- | --- |
| `.gitignore` | Leave unchanged. | Current `docs/api/` ignore policy matches the selected local-only decision. |
| `Doxyfile` | Leave output path unchanged unless Day 7/8 guard work needs a focused input/check adjustment. | Current `INPUT = include/`, `FILE_PATTERNS = *.h`, and `OUTPUT_DIRECTORY = docs/api` support local generation. |
| `Makefile` | Add a page-coverage guard target only if Day 7/8 selects Make as the guard owner. | Existing `docs` target generates HTML but does not validate coverage. |
| `scripts/` | Add a focused coverage script only if Day 7/8 selects script ownership. | Script ownership may keep coverage logic portable and testable. |
| `.github/workflows/*.yml` | Leave unchanged for Doxygen HTML publication. | Hosted Doxygen publication is deferred, not selected. |
| `docs/api_reference.md` | Update generated HTML wording to say local-only output is reproducible and guarded, not source-controlled freshness evidence. | Public docs must match the no-commit decision. |
| `docs/maintainer_guide.md` | Replace committed-output freshness requirement with local-only guard policy. | Current maintainer text still assumes committed generated output for freshness. |
| Public headers | Edit only for selected Day 4 warning closure. | Header comments may need Doxygen cleanup; any public-header edit triggers full C/header gates. |
| `docs/api/html/` | Do not stage or commit. | Generated output remains ignored local context. |

## Stale-Output Prevention Rules

| Rule | Required behavior |
| --- | --- |
| Local-only output is not checked-in evidence | `docs/api/html/` may be inspected locally, but it must not be cited as source-controlled freshness evidence. |
| Regeneration is explicit | A claim that local generated HTML is current must name the branch/run and `make docs` result. |
| Warnings block freshness | Doxygen warnings must be fixed or explicitly triaged before generated HTML is described as current for the configured input set. |
| Page coverage is required | The coverage guard must compare generated pages against the Day 3 checked-in public header source set. |
| Generated version header is separate | `sparse_version.h` generation/installation is documented separately from checked-in `include/*.h` page coverage. |
| Public headers remain authoritative | Exact declarations and call-site contracts remain owned by checked-in public headers. |
| No hosted claim without hosted lane | Public docs must not imply hosted/generated HTML publication unless a future workflow or deployment explicitly provides it. |

## Draft Support-Tier Wording

The following wording should guide Day 10 documentation alignment:

> Generated API HTML is local-only Doxygen output from the configured public
> header input set. It is useful for local navigation after running `make docs`,
> but it is not committed or hosted release evidence. Treat it as current only
> for the branch/run where `make docs`, Doxygen warning checks, and public-header
> page coverage have passed. The checked-in public headers and
> `docs/api_reference.md` remain the source-controlled API reference.

For the maintainer guide:

> Do not require `docs/api/html/` to be committed for freshness under the
> Sprint 158 policy. Instead, require the local guard result: `make docs`,
> warning closure, page coverage for checked-in `include/*.h`, and explicit
> generated `sparse_version.h` treatment.

## Days 7-11 Implementation Checklist

| Day | Implementation focus | Required outcome |
| --- | --- | --- |
| Day 7 | Page coverage check design | Choose script, Make target, documented manual procedure, or combination; define inputs, outputs, pass/fail behavior, and generated version-header treatment. |
| Day 8 | Page coverage check implementation | Implement or document the guard and run it locally or record the blocker. |
| Day 9 | Selected warning fixes | Fix W158-01 through W158-03 or reclassify them; rerun `make docs`; run full C/header gate if public headers changed. |
| Day 10 | Documentation policy alignment | Update `docs/api_reference.md` and `docs/maintainer_guide.md`; adjust README/tutorial only if routing or user-facing commands change. |
| Day 11 | Publication finalization | Confirm `docs/api/` remains ignored, guard behavior is implemented, generated HTML is not staged, and support-tier wording matches the local-only decision. |

## Validation Plan

| Touched surface | Required validation |
| --- | --- |
| Planning docs only | `git diff --check`; direct trailing-whitespace scan for untracked Sprint 158 docs. |
| Public docs | `git diff --check`; claim-sensitive scan for generated, hosted, package, ABI, platform, parity, performance, and state-of-the-art wording. |
| Coverage script or Make target | Focused guard run plus docs hygiene; additional script checks if implemented in Python or shell. |
| Public headers, even comment-only | `make format && make lint && make test`; `make docs`; page-coverage guard; docs hygiene. |
| Generated HTML | Keep ignored; verify `git status --ignored=matching --short docs/api` reports ignored output and no staged generated files. |

## Non-Claims Preserved

The Day 6 decision does not claim:

- committed generated API HTML;
- hosted generated API HTML;
- generated API HTML as release evidence;
- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad platform parity;
- external-library parity;
- portable performance;
- state-of-the-art coverage.

## Day 7 Handoff

Day 7 should design the page-coverage guard from the final policy:

1. expected inputs are the 18 checked-in public headers under `include/*.h`;
2. generated output path is `docs/api/html/`;
3. each checked-in header should have a generated `*_8h.html` reference page;
4. source pages may be a supplemental check;
5. generated `sparse_version.h` remains a separate policy row, not a missing
   page;
6. the guard must not treat ignored generated HTML as source-controlled
   evidence.

## Completion Check

- The selected publication path has a concrete implementation checklist.
- Repository tracking policy for generated HTML is unambiguous.
- Unsupported hosted and source-controlled freshness claims remain blocked.
- Days 7-11 can proceed without reopening the publication decision.
