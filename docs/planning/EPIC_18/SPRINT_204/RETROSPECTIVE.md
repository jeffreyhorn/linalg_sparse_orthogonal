# Sprint 204 Retrospective

**Sprint:** 204 - Generated API Publication Decision  
**Duration:** 14 days (Days 1-14 landed on branch `sprint-204`)  
**Status:** Closed for product decision, stronger local-only generated API
policy implementation, validation, claim calibration, and closeout

## Source Artifact Note

Sprint 204 was executed from the Epic 18 project-plan section for Sprint 204
and lives under `docs/planning/EPIC_18/SPRINT_204/` with its plan, working
notes, daily artifacts, closeout review, and retrospective in one package.

The sprint evaluated whether generated Doxygen API HTML should become hosted
documentation, a retained CI artifact, committed source-controlled output, or
a stronger local-only generated-output path. The final decision was to keep
generated API HTML local-only and strengthen the guard, freshness, routing,
and claim-boundary evidence around that policy.

## Definition Of Done Checklist

- [x] Created Sprint 204 plan, working notes, artifact directory, daily
      artifacts, closeout review, and retrospective.
- [x] Mapped Sprint 204 items 204.1 through 204.6 to owner surfaces, evidence
      requirements, validation commands, and explicit non-goals.
- [x] Reproduced the current local Doxygen baseline and confirmed
      `docs/api/`, `docs/api/html/`, and `docs/api/html/index.html` are
      ignored generated output.
- [x] Compared hosted publication, retained CI artifacts, committed generated
      HTML, and stronger local-only policy options.
- [x] Selected the stronger local-only generated API policy and documented why
      hosted, retained-artifact, and committed-output paths were not selected.
- [x] Strengthened local-only staging and workflow publication guards for
      generated API paths.
- [x] Added generated-page coverage and stale-page freshness checks for
      checked-in public headers selected by `Doxyfile`.
- [x] Added source-controlled API routing validation for README, INSTALL,
      `docs/api_reference.md`, and maintainer guidance.
- [x] Calibrated README, INSTALL, API reference, and maintainer-guide wording
      so generated API HTML does not imply hosted docs, retained artifacts,
      committed output, release, ABI, package, platform, performance, or
      state-of-the-art evidence.
- [x] Hardened Makefile routing wiring so `api-docs-validate` must retain
      `api-docs-routing`.
- [x] Ran focused generated API docs validation, Python regression, syntax,
      whitespace, and generated-output tracking checks.
- [x] Confirmed no `.c` or `.h` files changed, so the full C quality gate was
      not required by the sprint instruction.

## What Went Well

1. **The product decision stayed explicit.** The sprint did not drift into
   hosted publication or committed generated output by accident. It selected
   one policy path and kept unselected publication paths as explicit residuals.

2. **The local-only policy now has executable evidence.** `make
   api-docs-freshness` runs Doxygen generation, coverage/freshness checks,
   local-only staging checks, workflow non-publication checks, routing checks,
   and guard regressions.

3. **Stale generated pages are now caught.** The coverage checker compares
   generated reference/source page mtimes against checked-in public headers,
   closing a practical freshness gap in the prior local-only proof.

4. **User routing is source-controlled.** Public docs now point users to
   `docs/api_reference.md`, checked-in headers, `Doxyfile`, INSTALL, and
   maintainer guidance instead of treating generated HTML as hosted or release
   documentation.

5. **Claim boundaries moved with the implementation.** README, INSTALL,
   `docs/api_reference.md`, and `docs/maintainer_guide.md` use aligned
   vocabulary for unsupported hosted-docs, release, ABI, package, platform,
   performance, and state-of-the-art claims.

6. **Review hardening found a real wiring gap.** Day 13 added guard and
   regression coverage proving the aggregate validation path still includes
   `api-docs-routing`.

## What Didn't Go Well

1. **Generated API publication remains unresolved as a product feature.** The
   sprint intentionally selected stronger local-only behavior because hosted,
   retained-artifact, and committed-output paths needed more infrastructure
   and claim ownership than this closure justified.

2. **The policy spans many surfaces.** README, INSTALL, API reference,
   maintainer guide, Makefile targets, shell guards, Python guards, and tests
   all had to stay synchronized to avoid contradictory API support wording.

3. **Workflow-publication detection is still policy-specific.** The guard now
   rejects generated API publication semantics for the selected local-only
   policy, but a future hosted-docs sprint would need deliberate workflow,
   link, retention, and freshness semantics rather than reusing this policy.

4. **Local generated output still exists after validation.** Doxygen output is
   correctly ignored, but reviewers must remember that `docs/api/` and Python
   `__pycache__` directories are generated local artifacts, not PR content.

## Final Metrics

### Validation

| Metric | Sprint 204 close state |
| --- | --- |
| `make docs-check` | passed on Days 2, 12, and 14 |
| `make api-docs-freshness` | passed on Days 2, 7, 8, 9, 10, 11, 12, 13, and 14 |
| generated API coverage regressions | passed on Days 8, 12, and 14 |
| local-only generated API guard regressions | passed on Days 7, 12, and 14 |
| API routing guard regressions | passed on Days 9, 11, 12, 13, and 14 |
| Makefile routing-wiring guard | added and passed on Day 13; passed again on Day 14 |
| direct API routing guard | passed on Days 9, 10, 11, 12, 13, and 14 |
| Python syntax check | passed on Days 8, 9, 10, 11, 12, 13, and 14 for edited docs tooling |
| generated-output git tracking check | passed on Days 2, 12, 13, and 14 |
| final `git diff --check` | passed |
| final full C quality gate | not required because no `.c` or `.h` files changed |

### Changed Surface

| Metric | Sprint 204 close state |
| --- | ---: |
| Sprint plan files added | 1 |
| Working notes files added | 1 |
| Sprint daily artifacts added | 14 |
| Sprint retrospective files added | 1 |
| Public documentation files changed | 3 |
| Maintainer documentation files changed | 1 |
| Makefile targets changed | 3 |
| Shell guard files changed | 1 |
| Python validation or guard scripts changed | 2 |
| Python validation or guard test files changed | 3 |
| Epic project-plan files changed | 1 |
| CI workflow files changed | 0 |
| Doxyfile files changed | 0 |
| `.gitignore` files changed | 0 |
| Production C implementation files changed | 0 |
| Public or internal C header files changed | 0 |

### Project-Plan Status Metrics

| Status family | Final count |
| --- | ---: |
| Product decision completed | 1 |
| Stronger local-only guard implemented | 1 |
| Generated-page freshness and coverage checks completed | 1 |
| Source-controlled API routing docs completed | 1 |
| Claim-boundary guard completed | 1 |
| Final validation completed | 1 |
| Hosted generated API publication implemented | 0 |
| Retained generated-doc artifact implemented | 0 |
| Committed generated HTML implemented | 0 |
| Package, ABI, platform, performance, release, or state-of-the-art claims promoted | 0 |

The count covers Sprint 204 items 204.1 through 204.6.

## Closed Claim

Sprint 204 closes this bounded generated API policy claim:

Generated Doxygen API HTML remains local-only ignored output under
`docs/api/html/`. The source-controlled API documentation route is
`docs/api_reference.md`, checked-in public headers under `include/`,
`Doxyfile`, README, INSTALL, and maintainer guidance. The aggregate
`make api-docs-freshness` path now validates Doxygen generation, generated
page coverage and freshness for checked-in public headers, generated-output
staging/tracking boundaries, workflow non-publication semantics,
source-controlled API routing, maintainer claim-boundary wording, and Makefile
routing wiring.

This claim does not include hosted generated API documentation, retained
generated-doc artifacts, committed generated HTML, API completeness beyond
checked-in public headers selected by `Doxyfile`, dynamic ABI compatibility,
shared-library support, package-manager distribution, broad platform parity,
release evidence, performance evidence, external-library parity, or
state-of-the-art sparse linear algebra evidence.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-generated-api-intake.md](./artifacts/day1-generated-api-intake.md);
- [day2-current-doxygen-baseline.md](./artifacts/day2-current-doxygen-baseline.md);
- [day3-publication-option-inventory.md](./artifacts/day3-publication-option-inventory.md);
- [day4-decision-criteria-acceptance-gate.md](./artifacts/day4-decision-criteria-acceptance-gate.md);
- [day5-product-decision.md](./artifacts/day5-product-decision.md);
- [day6-workflow-tracking-design.md](./artifacts/day6-workflow-tracking-design.md);
- [day7-policy-implementation-batch.md](./artifacts/day7-policy-implementation-batch.md);
- [day8-freshness-and-coverage-checks.md](./artifacts/day8-freshness-and-coverage-checks.md);
- [day9-link-and-routing-validation.md](./artifacts/day9-link-and-routing-validation.md);
- [day10-user-facing-api-docs-update.md](./artifacts/day10-user-facing-api-docs-update.md);
- [day11-maintainer-and-claim-boundary-docs.md](./artifacts/day11-maintainer-and-claim-boundary-docs.md);
- [day12-integrated-validation.md](./artifacts/day12-integrated-validation.md);
- [day13-review-hardening.md](./artifacts/day13-review-hardening.md);
- [day14-closeout-review.md](./artifacts/day14-closeout-review.md).

## Residuals

| Residual | Owner condition | Evidence required to close |
| --- | --- | --- |
| Hosted generated API HTML | Future generated-docs publication owner | Select hosted publication deliberately; add hosting workflow, freshness metadata, link validation, publication URL ownership, retention policy, and claim-boundary docs. |
| Retained generated-doc artifacts | Future CI artifact owner | Add artifact upload, retention, discoverability, freshness, and stale-output semantics before treating generated HTML as retained evidence. |
| Committed generated HTML | Future docs-source owner | Replace the ignored-output policy with checked-in generated-output ownership, regeneration discipline, review-noise controls, and stale-output checks. |
| Broad API completeness | Future API documentation owner | Define API completeness scope beyond checked-in public headers selected by `Doxyfile` and add coverage evidence for that broader scope. |
| ABI/shared-library support | Future ABI/package owner | Add ABI policy, shared-library build/install evidence, compatibility tests, and public support wording. |
| Package-manager distribution | Future packaging owner | Add package recipes, package validation, installation evidence, and explicit support tiers. |
| Release, platform, performance, or state-of-the-art evidence | Future release/evidence owner | Provide dedicated release, platform, benchmark, external comparison, and claim-reviewed evidence before promoting these claims. |

## Next-Sprint Readiness

Sprint 204 leaves a stronger local generated API guard path without increasing
the public support surface.

| Future need | Sprint 204 handoff |
| --- | --- |
| Routine generated API validation | Use `make api-docs-freshness` as the aggregate local-only generated API guard. |
| Documentation routing maintenance | Keep README, INSTALL, `docs/api_reference.md`, and maintainer guidance aligned with source-controlled API routes. |
| Guard maintenance | Update coverage, local-only, routing, and Makefile-wiring tests together whenever generated API policy changes. |
| Hosted docs reconsideration | Start with a new product decision and do not infer hosted support from local Doxygen freshness. |
| Review hygiene | Treat `docs/api/`, `scripts/__pycache__/`, and `tests/__pycache__/` as ignored local output after validation runs. |

## Final Assessment

Sprint 204 is complete as a generated API publication decision sprint. It
chooses stronger local-only generated API evidence, adds freshness and routing
guards, aligns public and maintainer wording, hardens Makefile validation
wiring, and records final validation without changing C implementation or
header behavior. It does not publish generated API HTML or promote package,
ABI, platform, release, performance, or state-of-the-art claims.
