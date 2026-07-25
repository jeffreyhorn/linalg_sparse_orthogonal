# Sprint 135 Retrospective

**Sprint:** 135 - Adoption Surface Simplification & Documentation Productization
**Duration:** 14 days (Days 1-14 landed on branch `sprint-135`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 135 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Sprint 135 Epic 11 project-plan scope and inherited Sprint 131,
      Sprint 133, and Sprint 134 boundaries before editing public docs.
- [x] Audited README, INSTALL, tutorial, solver-selection, algorithm, Matrix
      Market, maintainer, examples, and benchmark adoption surfaces.
- [x] Designed the split between current algorithm reference and historical
      measurement appendix content.
- [x] Implemented the bounded algorithm split:
  - `docs/algorithm.md` remains the current algorithm reference.
  - `docs/algorithm_history.md` preserves historical measurements,
    sprint-era implementation decisions, benchmark rationale, and planning
    links.
- [x] Created `docs/cookbook.md` as the compressed-first adoption owner.
- [x] Added compressed-first cookbook paths for:
  - direct solves;
  - iterative solves;
  - Matrix Market load/use;
  - SVD and low-rank workflows;
  - symmetric eigensolver workflows;
  - benchmark/report handoff after API workflow selection.
- [x] Surfaced generated report-index and freshness metadata paths for:
  - canonical benchmark reports;
  - performance sentinel reports;
  - large-matrix guardrail reports.
- [x] Aligned first-use navigation across README, tutorial, solver-selection,
      cookbook, examples, benchmarks, install docs, algorithm reference,
      algorithm history, and maintainer guide.
- [x] Preserved Sprint 133-134 package/platform truth:
  - static-first package support remains the install baseline;
  - shared-library packaging, dynamic ABI compatibility, runtime-loader
    behavior, and package-manager support remain non-claims;
  - Linux remains the strongest reviewed package-contract source of truth;
  - macOS and Windows package install/downstream confidence remains
    supplemental where documented.
- [x] Preserved Sprint 131 report-index truth:
  - generated indexes are artifact maps and freshness context;
  - cross-report normalized indexing remains deferred;
  - report rows do not become broad correctness, coverage, release, or
    performance claims.
- [x] Documentation validation passed:
  - `git diff --check`;
  - focused trailing-whitespace scans;
  - local markdown link-target check;
  - package/platform claim scan;
  - performance/report claim scan;
  - cookbook workflow discoverability scan.
- [x] No `.c` or `.h` files changed, so the full
      `make format && make lint && make test` gate was not required.
- [x] Closeout artifact, residual queue, Sprint 136 handoff, and retrospective
      input queue were written.

## What Went Well

1. **The adoption owner surfaces are now explicit.**
   README now gives a compact Adoption Map. Tutorial has a Documentation Map.
   Cookbook owns compressed-first recipes. Benchmark docs own measurement and
   generated report interpretation. INSTALL owns installed-consumer setup.
   Algorithm docs now split current reference from historical measurement
   context.

2. **The algorithm split kept the public path stable.**
   `docs/algorithm.md` stayed in place for existing links, but high-friction
   historical measurement blocks moved or collapsed into references to
   `docs/algorithm_history.md`.

3. **Compressed-first workflows have one natural home.**
   Before this sprint, CSR/CSC, Matrix Market, SVD, eigensolver, and benchmark
   handoffs were spread across README, tutorial, solver selection, examples,
   Matrix Market docs, and benchmark docs. `docs/cookbook.md` now gives those
   users a single task-oriented route.

4. **Report-index discovery improved without changing report semantics.**
   The sprint surfaced `index.tsv`, `sentinels.tsv`, and `manifest.txt`
   locations for canonical, sentinel, and large-matrix guardrail reports while
   preserving Sprint 131's generated-versus-curated boundaries.

5. **Validation caught real wording drift.**
   Day 12 and Day 13 found and fixed front-door wording that could read like
   portable speedup evidence, sprint-era phrasing in current algorithm
   reference text, and an `Algorithm Description` title that no longer matched
   the README's `Algorithm Reference` label.

6. **The sprint stayed honestly docs-only.**
   No implementation files changed. Validation stayed proportional to the
   touched surface while still checking links, whitespace, report wording, and
   package/platform claims.

## What Didn't Go Well

1. **The algorithm reference is still dense.**
   The sprint moved the highest-friction historical blocks, but
   `docs/algorithm.md` remains a long technical reference. Further shortening
   would require a dedicated reference-architecture pass, not another small
   validation cleanup.

2. **Link validation is still ad hoc.**
   Day 12 used a focused local markdown-link script. It worked, but a
   maintained repository target would make future documentation review easier
   and less dependent on one-off command construction.

3. **Historical artifacts need final-context reading.**
   Early Sprint 135 artifacts intentionally describe pre-decision state. The
   closeout and retrospective now carry final truth, but readers need to treat
   Day 14 and this retrospective as the current summary.

4. **Generated report indexing remains intentionally fragmented.**
   The sprint made index locations discoverable but did not normalize report
   schemas. That is the right constraint for now, but it leaves future readers
   with multiple row models to understand.

## Final Metrics

### Validation

| Metric | Sprint 135 close state |
|---|---:|
| tracked `.c`/`.h` changes | 0 |
| public docs created | 2 |
| existing public docs edited | 7 |
| total public docs touched | 9 |
| public adoption/owner docs in validation sweep | 11 |
| maintained example/header link targets checked | 11 |
| generated report families surfaced | 3 |
| cookbook workflow families covered | 6 |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |
| local markdown link-target check | passed |
| package/platform claim scan | passed |
| performance/report claim scan | passed |
| full C quality gate | not required; no `.c`/`.h` changes |

### Sprint 135 Artifact Package

| Metric | Sprint 135 close state |
|---|---:|
| total artifact files under `SPRINT_135/artifacts/` | 14 |
| intake and audit artifacts | 2 |
| algorithm split design/prep artifacts | 2 |
| implementation artifacts | 5 |
| validation/review/closeout artifacts | 5 |

Notes:

- intake and audit artifacts:
  - `day1-adoption-intake.md`
  - `day2-adoption-surface-audit.md`
- algorithm split design/prep artifacts:
  - `day3-algorithm-doc-split-design.md`
  - `day4-algorithm-split-preparation.md`
- implementation artifacts:
  - `day5-algorithm-split-batch1.md`
  - `day6-algorithm-split-batch2.md`
  - `day7-compressed-first-cookbook-design.md`
  - `day8-compressed-first-cookbook-batch1.md`
  - `day9-compressed-first-cookbook-batch2.md`
- validation/review/closeout artifacts:
  - `day10-benchmark-report-index-docs.md`
  - `day11-adoption-navigation-alignment.md`
  - `day12-link-and-claim-validation.md`
  - `day13-integrated-adoption-review.md`
  - `day14-adoption-closeout-handoff.md`

## Residual Deferred Debt

Most important carry-forward work:

- continue shrinking and reorganizing long current-reference sections in
  `docs/algorithm.md`;
- add a maintained docs link-check target if documentation work continues to
  grow;
- revisit cross-report normalized indexing only if report-family row meanings,
  support tiers, freshness fields, and claim boundaries can be preserved;
- keep `docs/cookbook.md` current as new examples, workflows, or report
  surfaces land;
- keep README and tutorial navigation aligned with the cookbook and algorithm
  split as future sprints add features.

Still consciously constrained rather than silently solved:

- no code behavior change;
- no expanded package support;
- no shared-library support;
- no dynamic ABI compatibility claim;
- no runtime-loader behavior claim;
- no package-manager support claim;
- no platform parity expansion;
- no portable benchmark performance claim;
- no normalized cross-report index claim;
- no coverage-completeness claim from report-index wording.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-adoption-intake.md](./artifacts/day1-adoption-intake.md)
- [day2-adoption-surface-audit.md](./artifacts/day2-adoption-surface-audit.md)
- [day3-algorithm-doc-split-design.md](./artifacts/day3-algorithm-doc-split-design.md)
- [day4-algorithm-split-preparation.md](./artifacts/day4-algorithm-split-preparation.md)
- [day5-algorithm-split-batch1.md](./artifacts/day5-algorithm-split-batch1.md)
- [day6-algorithm-split-batch2.md](./artifacts/day6-algorithm-split-batch2.md)
- [day7-compressed-first-cookbook-design.md](./artifacts/day7-compressed-first-cookbook-design.md)
- [day8-compressed-first-cookbook-batch1.md](./artifacts/day8-compressed-first-cookbook-batch1.md)
- [day9-compressed-first-cookbook-batch2.md](./artifacts/day9-compressed-first-cookbook-batch2.md)
- [day10-benchmark-report-index-docs.md](./artifacts/day10-benchmark-report-index-docs.md)
- [day11-adoption-navigation-alignment.md](./artifacts/day11-adoption-navigation-alignment.md)
- [day12-link-and-claim-validation.md](./artifacts/day12-link-and-claim-validation.md)
- [day13-integrated-adoption-review.md](./artifacts/day13-integrated-adoption-review.md)
- [day14-adoption-closeout-handoff.md](./artifacts/day14-adoption-closeout-handoff.md)

Public documentation deliverables:

- [`docs/cookbook.md`](../../../cookbook.md)
- [`docs/algorithm_history.md`](../../../algorithm_history.md)
- [`docs/algorithm.md`](../../../algorithm.md)
- [`benchmarks/README.md`](../../../../benchmarks/README.md)
- [`README.md`](../../../../README.md)
- [`INSTALL.md`](../../../../INSTALL.md)
- [`docs/tutorial.md`](../../../tutorial.md)
- [`docs/solver_selection.md`](../../../solver_selection.md)
- [`examples/README.md`](../../../../examples/README.md)

## Sprint 136 Handoff

- Treat `docs/cookbook.md` as the compressed-first adoption owner.
- Treat `docs/algorithm.md` as the current algorithm reference.
- Treat `docs/algorithm_history.md` as the historical measurement appendix.
- Treat `benchmarks/README.md` as the benchmark/report-index interpretation
  owner.
- Treat `INSTALL.md` as the install and downstream-consumer support owner.
- Treat `docs/maintainer_guide.md` as the support-tier and maintainer-policy
  owner.
- Preserve Sprint 131, Sprint 133, and Sprint 134 claim boundaries unless a
  future sprint explicitly implements and validates a changed product claim.
