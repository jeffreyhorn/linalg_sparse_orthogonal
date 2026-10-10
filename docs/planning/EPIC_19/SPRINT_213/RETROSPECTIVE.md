# Sprint 213 Retrospective

**Sprint:** 213 - Generated API Publication Decision  
**Duration:** 14 days (Days 1-14 landed on branch `sprint-213`)  
**Status:** Closed with stronger local-only generated API closure

## Source Artifact Note

Sprint 213 was executed from the Epic 19 project-plan section for Sprint 213
and lives under `docs/planning/EPIC_19/SPRINT_213/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint evaluated whether generated Doxygen HTML under `docs/api/html/`
should remain local-only, be hosted, be retained as a generated-doc CI
artifact, or be committed. It selected stronger local-only closure. The branch
therefore hardens workflow/staging detection, routing/link validation,
user-facing documentation, maintainer repair guidance, and Epic 19 residual
status without publishing generated API HTML.

## Definition Of Done Checklist

- [x] Created Sprint 213 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Inventoried generated API evidence surfaces, including `Doxyfile`,
      ignored `docs/api/` output, API docs coverage, local-only workflow
      guards, routing guards, user docs, maintainer docs, and prior Sprint 204
      local-only policy evidence.
- [x] Captured current local-only baseline data with `make api-docs-freshness`
      and standalone API docs regression suites.
- [x] Compared four policy options: stronger local-only closure, hosted Pages,
      retained generated-doc artifact, and committed generated HTML.
- [x] Defined publication and non-publication decision criteria before the
      product policy decision.
- [x] Selected stronger local-only generated API closure and rejected hosted,
      retained-artifact, and committed generated HTML for this sprint.
- [x] Hardened workflow/staging detection for folded staging/archive commands,
      quote-aware YAML comment stripping, and `.yaml` workflow coverage.
- [x] Hardened routing/link validation for generated/hosted publication links,
      required source-controlled route markers, encoded links, and maintainer
      residual markers.
- [x] Added focused regression coverage in `tests/test_api_docs_local_only_guard.py`
      and `tests/test_api_docs_routing.py`.
- [x] Updated README, INSTALL, API reference, maintainer guide, and Epic 19
      project-plan wording for stronger local-only generated API closure.
- [x] Ran generated API docs checks, focused regression suites, aggregate
      freshness validation, whitespace validation, and no-C/header checks.

## What Went Well

1. **The sprint made a policy decision before implementation.** Days 1-5
   captured baseline behavior, compared options, defined acceptance criteria,
   and selected stronger local-only closure before changing guards or docs.

2. **The selected policy stayed narrow.** The branch improves generated API
   evidence integrity without introducing hosted docs, retained artifacts, or
   committed generated HTML.

3. **Workflow and routing hardening target practical bypasses.** Folded shell
   staging/archive commands, quoted `#` YAML scalars, `.yaml` workflows,
   encoded generated-output links, and hosted/release/artifact link routes now
   have focused regression coverage.

4. **User and maintainer docs now say where the durable API route is.** README,
   INSTALL, and `docs/api_reference.md` direct users to source-controlled API
   docs and checked-in public headers. The maintainer guide records repair
   workflow and residual publication criteria.

5. **The Epic 19 status table no longer leaves Sprint 213 ambiguous.** The
   project plan marks Sprint 213 closed and keeps Sprints 214-216 as the
   remaining future execution range.

## What Didn't Go Well

1. **The sprint closed publication by deferral, not by adding a hosted surface.**
   This is the right evidence-based outcome for the branch, but it does not
   make generated HTML easier to consume outside a local checkout.

2. **The local-only guards continue to rely on textual workflow scanning.**
   The scanner is stronger than before, but future workflow syntax changes may
   still require additional fixtures or a more structured workflow parser.

3. **Documentation guard coupling increased.** Required wording markers prevent
   accidental claim drift, but future docs rewrites must update guard markers
   and tests deliberately.

4. **Generated output remains current only after a local command.** Users still
   need `make api-docs-freshness` to inspect current generated HTML; there is
   no hosted or retained generated-doc artifact to inspect.

## Final Metrics

### Validation

| Metric | Sprint 213 close state |
| --- | --- |
| `make docs-check` | passed |
| `python3 tests/test_api_docs_coverage.py` | passed |
| `python3 tests/test_api_docs_local_only_guard.py` | passed |
| `python3 tests/test_api_docs_routing.py` | passed |
| `make api-docs-freshness` | passed |
| final `git diff --check` | passed |
| final `git diff --name-only -- '*.c' '*.h'` | no C/header changes |

No `.c` or `.h` files changed during Sprint 213, so the full C quality gate
`make format && make lint && make test` was not required by the sprint rule.

### Generated API Policy Metrics

| Metric | Sprint 213 close state |
| --- | --- |
| selected policy | stronger local-only generated API closure |
| source-controlled API route | `docs/api_reference.md` and checked-in public headers |
| generated output root | `docs/api/` |
| generated HTML path | `docs/api/html/` |
| freshness command | `make api-docs-freshness` |
| checked-in public headers covered | 18 |
| generated reference pages covered | 18 |
| generated source pages covered | 18 |
| hosted generated API HTML promoted | 0 |
| retained generated-doc artifacts promoted | 0 |
| committed generated HTML promoted | 0 |

### Changed Surface

| Metric | Sprint 213 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Epic project-plan files changed | 1 |
| Public documentation files changed | 3 |
| Maintainer documentation files changed | 1 |
| Local-only guard implementation files changed | 2 |
| Routing guard files changed | 1 |
| Local-only guard test files changed | 1 |
| Routing guard test files changed | 1 |
| Public API/ABI declarations changed | 0 |
| C/header files changed | 0 |

### Line Snapshot

| Path | Lines |
| --- | ---: |
| `README.md` | 1137 |
| `INSTALL.md` | 624 |
| `docs/api_reference.md` | 106 |
| `docs/maintainer_guide.md` | 2258 |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 449 |
| `scripts/check_api_docs_local_only.sh` | 925 |
| `scripts/api_docs_workflow_yaml_common.py` | 242 |
| `scripts/check_api_docs_routing.py` | 609 |
| `tests/test_api_docs_local_only_guard.py` | 2910 |
| `tests/test_api_docs_routing.py` | 1678 |
| `docs/planning/EPIC_19/SPRINT_213/WORKING_NOTES.md` | 1232 |
| `docs/planning/EPIC_19/SPRINT_213/artifacts/day14-closeout-review.md` | 101 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Publication option review items completed | 1 |
| Policy decision items completed | 1 |
| Automation implementation items completed | 1 |
| Routing and guard test items completed | 1 |
| User and maintainer docs items completed | 1 |
| Validation and closeout items completed | 1 |
| Generated API publication paths promoted | 0 |
| Package, ABI, platform, performance, release, external parity, or state-of-the-art claims promoted | 0 |

The count covers Sprint 213 items 213.1 through 213.6.

## Closed Claim

Sprint 213 closes this bounded claim:

Generated API HTML remains local-only ignored output under `docs/api/html/`,
and the repository now has stronger workflow/staging, routing/link,
documentation, maintainer, planning, and validation guards preserving that
policy.

This claim does not include hosted generated API HTML, retained generated-doc
CI artifacts, committed generated HTML, generated API release evidence,
generated API package-manager evidence, broad API completeness beyond
checked-in public headers selected by `Doxyfile`, package support, ABI support,
shared-library support, runtime-loader support, broad platform support,
portable performance, external-library parity, or state-of-the-art sparse
linear algebra status.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-generated-api-evidence-intake.md](./artifacts/day1-generated-api-evidence-intake.md);
- [day2-current-local-only-baseline.md](./artifacts/day2-current-local-only-baseline.md);
- [day3-publication-option-inventory.md](./artifacts/day3-publication-option-inventory.md);
- [day4-decision-criteria.md](./artifacts/day4-decision-criteria.md);
- [day5-product-policy-decision.md](./artifacts/day5-product-policy-decision.md);
- [day6-automation-design.md](./artifacts/day6-automation-design.md);
- [day7-automation-implementation-batch-one.md](./artifacts/day7-automation-implementation-batch-one.md);
- [day8-automation-implementation-batch-two.md](./artifacts/day8-automation-implementation-batch-two.md);
- [day9-routing-and-link-validation.md](./artifacts/day9-routing-and-link-validation.md);
- [day10-workflow-and-staging-validation.md](./artifacts/day10-workflow-and-staging-validation.md);
- [day11-user-documentation-update.md](./artifacts/day11-user-documentation-update.md);
- [day12-maintainer-documentation-and-residuals.md](./artifacts/day12-maintainer-documentation-and-residuals.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Hosted generated API HTML | Future generated API publication sprint | Exact hosted URL, deployment workflow, access control, freshness ordering, stale-output rollback, source/generated route policy, user docs, maintainer docs, and guard allowlist. |
| Retained generated-doc CI artifacts | Future artifact-publication sprint | Exact artifact name, retention period, freshness proof, artifact routing policy, release-evidence non-claims, and workflow guard coverage. |
| Committed generated HTML | Future generated-output ownership sprint | Repository size/noise assessment, generated-output ownership policy, freshness enforcement, review workflow, stale-output repair path, and docs/guard updates. |
| Structured workflow parser | Future guard-maintenance owner | Replace or augment textual workflow scanning with structured YAML/data-flow validation while preserving all current bypass fixtures. |
| Broader API completeness | Future API documentation owner | Explicit Doxygen input expansion, generated-header policy, checked-in/generated header boundary, page coverage evidence, and user documentation updates. |
| Package, ABI, platform, performance, release, external parity, or state-of-the-art evidence | Future Epic 19/20 owner | Add exact proof, docs, and guards before promoting any broad support claim. |

## Next-Sprint Readiness

Sprint 213 leaves generated API publication status closed as local-only and
keeps future publication options explicit.

| Future need | Sprint 213 handoff |
| --- | --- |
| Current Epic 19 status | Start from `docs/planning/EPIC_19/PROJECT_PLAN.md`, which marks Sprint 213 closed and Sprints 214-216 pending. |
| Generated API freshness | Run `make api-docs-freshness`; it regenerates local HTML and runs coverage, local-only, and routing checks. |
| Local-only workflow changes | Run `python3 tests/test_api_docs_local_only_guard.py` and `make api-docs-freshness`. |
| API route or documentation changes | Run `python3 tests/test_api_docs_routing.py` and `make api-docs-freshness`. |
| Source or header changes | Run `make format && make lint && make test` before closeout. |
| Retrospective source material | Use `WORKING_NOTES.md` and Day 1-Day 14 artifacts under `SPRINT_213/artifacts/`. |

## Final Assessment

Sprint 213 closes the generated API publication decision end to end by choosing
stronger local-only closure and backing that choice with guards, regressions,
documentation, planning status, and integrated validation. The repository now
has a clearer generated API route: source-controlled docs and checked-in public
headers are the durable API surface, while Doxygen HTML remains a local,
ignored, current-checkout inspection view.
