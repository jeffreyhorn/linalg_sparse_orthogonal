# Sprint 141 Retrospective

**Sprint:** 141 - Report Index Normalization & Freshness Gates
**Duration:** 14 days (Days 1-14 landed on branch `sprint-141`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 141 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Re-read Sprint 138 corpus architecture, Sprint 139 QR closure handoff,
      Sprint 140 partial-SVD closure handoff, report-producing scripts,
      corpus manifests, benchmark/report docs, package docs, and maintainer
      guidance.
- [x] Inventoried maintained and generated report families: corpus, oracle,
      benchmark, sentinel, guardrail, coverage, dead-code, package, install,
      CI lane definitions, documentation guidance, report-index diagnostics,
      and runtime/backend governance.
- [x] Added source-controlled report-family metadata in
      `tests/corpus/manifests/report_families.tsv`.
- [x] Added normalized report-index field documentation in
      `tests/corpus/schemas/report_index_fields.md`.
- [x] Extended `scripts/validate_corpus_schema.py` with report-family schema
      validation, vocabulary checks, duplicate identity checks, support-tier
      checks, and false-pass guardrails.
- [x] Added `scripts/normalize_report_index.py` as the maintained normalized
      report index generator.
- [x] Added deterministic normalized rows for source-controlled corpus
      fixture, generator, optional-data, expected-result, report-family,
      package proof-owner, CI lane, documentation advisory, missing-generated,
      and runtime/backend defer surfaces.
- [x] Added generated report ingestion for corpus/oracle, benchmark, sentinel,
      guardrail, coverage, and dead-code artifacts while preserving native row
      meanings.
- [x] Added freshness validation behind explicit `--check-freshness`, with
      deterministic diagnostics, `--require-generated`, `--strict-generated`,
      and `--advisory-ok`.
- [x] Added `tests/test_normalize_report_index.py` covering deterministic
      output, family filtering, generated-artifact ingestion, missing rows,
      required-generated failures, stale generated rows, advisory rows,
      sentinel hard-gate failures, optional-data skips, and runtime/backend
      defers.
- [x] Updated README, cookbook, maintainer guide, benchmark docs, corpus docs,
      and install docs with normalized-index commands, freshness
      interpretation, support-tier boundaries, and non-claims.
- [x] Published final validation evidence and generated-output hygiene.
- [x] Published the Sprint 142 runtime/backend governance handoff.
- [x] Ran Sprint 141 focused validation:
  - `python3 -m py_compile scripts/validate_corpus_schema.py
    scripts/normalize_report_index.py tests/test_normalize_report_index.py`;
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `python3 scripts/normalize_report_index.py --no-generated --output
    build/report-index/normalized-index.tsv`;
  - `python3 scripts/normalize_report_index.py --no-generated --check`;
  - `python3 scripts/normalize_report_index.py --check`;
  - `python3 scripts/normalize_report_index.py --check-freshness`;
  - `python3 scripts/normalize_report_index.py --family runtime_backend
    --check-freshness`;
  - `python3 scripts/normalize_report_index.py --family coverage --family
    deadcode --family package --check-freshness`;
  - `python3 scripts/normalize_report_index.py --family benchmark --family
    sentinel --family guardrail --check-freshness`;
  - expected-failure probes for required generated coverage and oracle rows;
  - generated-artifact ignored checks;
  - `git diff --check`;
  - trailing-whitespace scans.
- [x] No C or header files changed, so the sprint did not require
      `make format && make lint && make test`.

## What Went Well

1. **The sprint turned report interpretation into a maintained contract.**
   Report families now have source-controlled row meanings, support tiers,
   freshness policies, commands, artifact patterns, claim scopes, owners, and
   non-claim boundaries.

2. **The normalized index preserves native row meaning.** Corpus fixtures,
   expected-result rows, generated oracle rows, solver-backed proof rows,
   benchmark measurements, sentinel hard/advisory rows, guardrail lanes,
   coverage rows, dead-code rows, package proof-owner rows, CI lane
   definitions, and documentation advisories keep their family-specific
   semantics instead of being flattened into generic pass/fail evidence.

3. **Missing generated reports became visible without becoming failures by
   default.** Advisory report families emit deterministic `not_generated`
   diagnostics, while `--require-generated <family>` lets a focused review
   promote a selected missing family to a hard error.

4. **Freshness behavior is explicit and reproducible.** The diagnostic format
   `freshness: <severity>: <row_id>: <state>: <reason>` gives maintainers a
   stable way to distinguish fresh, stale, source-controlled, skipped,
   deferred, advisory, and required-generated states.

5. **The documentation now matches the tooling.** README, cookbook,
   maintainer, benchmark, corpus, and install docs describe how to run and
   interpret the normalized index without implying release proof or broader
   product claims.

## What Didn't Go Well

1. **The row semantics were inherently heterogeneous.** A single index had to
   carry source-controlled contracts, local generated reports, external CI
   lane definitions, package proof owners, optional-data skips, and deferred
   governance rows. The explicit row-meaning taxonomy was necessary to avoid
   false uniformity.

2. **Local generated artifacts can confuse freshness review.** Existing
   ignored oracle outputs may be stale relative to current `HEAD`, so default
   freshness mode warns rather than failing. Strict review flows need to use
   `--require-generated oracle` when current generated oracle evidence is
   required.

3. **Runtime/backend rows could not be honestly closed in Sprint 141.** The
   index can preserve backend-related row context, but policy decisions about
   precedence, typed controls, environment overrides, backend fallback, and
   sentinel expansion belong to Sprint 142.

4. **The docs surface is broad.** Aligning user and maintainer interpretation
   required changes across multiple documents. The Day 12 and Day 14 artifacts
   are important because they explain why these docs all changed together.

## Final Metrics

### Validation

| Metric | Sprint 141 close state |
| --- | --- |
| tracked `.c`/`.h` changes | no |
| full C quality gate required | no |
| Python compile checks | passed |
| `python3 scripts/validate_corpus_schema.py` | passed |
| `python3 tests/test_normalize_report_index.py` | passed |
| deterministic source-controlled normalized index | passed: `47` rows |
| generated-aware normalized index | passed: `59` rows |
| default freshness check | passed: `59` rows |
| runtime/backend freshness check | passed: `1` defer row |
| coverage/dead-code/package freshness check | passed: `10` rows |
| benchmark/sentinel/guardrail freshness check | passed: `8` rows |
| required-generated coverage probe | expected failure: missing generated coverage promoted to `freshness: error` |
| required-generated oracle probe | expected failure: missing generated oracle rows promoted to `freshness: error` |
| generated normalized-index file tracked | no |
| generated normalized-index file ignored | yes: `build/report-index/normalized-index.tsv` |
| `git diff --check` | passed |
| trailing-whitespace scan | passed |

### Artifact Package

| Metric | Sprint 141 close state |
| --- | ---: |
| daily artifacts under `SPRINT_141/artifacts/` | 14 |
| final retrospective files | 1 |
| new Python generator scripts | 1 |
| new Python test files | 1 |
| validator scripts changed | 1 |
| report-family manifest files added | 1 |
| report-index schema docs added | 1 |
| public/maintainer documentation surfaces changed | 6 |
| source-controlled generated report files | 0 |

## Closed Claim

Sprint 141 closes this claim:

The project now has a maintained normalized report-index and freshness
diagnostic surface for honestly normalizable report families across corpus,
oracle, benchmark, sentinel, guardrail, coverage, dead-code, package, CI lane
definition, documentation, report-index diagnostic, and runtime/backend defer
rows.

This claim is supported by:

- `tests/corpus/manifests/report_families.tsv`;
- `tests/corpus/schemas/report_index_fields.md`;
- `scripts/validate_corpus_schema.py`;
- `scripts/normalize_report_index.py`;
- `tests/test_normalize_report_index.py`;
- `README.md`;
- `docs/cookbook.md`;
- `docs/maintainer_guide.md`;
- `benchmarks/README.md`;
- `tests/corpus/README.md`;
- `INSTALL.md`;
- Day 13 and Day 14 validation evidence.

## Sprint 142 Readiness

Sprint 142 should consume the `runtime_backend` defer row as its starting
point:

| Handoff field | Sprint 142 requirement |
| --- | --- |
| runtime control audit | Audit OpenMP, backend dispatch, dense helper selection, eigensolver backend selection, direct-solver dispatch, environment variables, and typed options. |
| precedence contract | Define maintained precedence for typed options, compile-time flags, environment compatibility overrides, backend fallback, and deterministic behavior. |
| typed-control decisions | Promote high-value implicit or environment-only controls into typed options, or explicitly classify them as maintainer-only. |
| sentinel expansion | Add normalized local sentinel rows only where they provide useful regression evidence without portable timing claims. |
| documentation | Update README, benchmark docs, maintainer guide, and examples for any runtime/backend contract changes. |
| validation | Run focused runtime/backend tests, sentinels, freshness checks, and full quality gates if `.c` or `.h` files change. |

## Residual Deferred Debt

Most important carry-forward work:

- Sprint 142 runtime/backend governance and precedence policy;
- typed runtime/backend control decisions for environment-only or implicit
  controls;
- sentinel expansion where backend decisions need maintained local regression
  visibility;
- hosted CI log normalization, if a future sprint creates a source-controlled
  summary format;
- optional generated-report promotion rules, if a later review requires
  source-controlled evidence artifacts rather than ignored local outputs.

Still consciously constrained rather than silently solved:

- no portable performance claim;
- no broad solver, QR, partial-SVD, corpus, external-library, or
  state-of-the-art correctness claim;
- no package-manager, shared-library ABI, dynamic-linking, or broad platform
  support claim;
- no hosted CI proof from local generated rows;
- no zero-dead-code claim;
- no coverage completeness claim;
- no Sprint 141 closure claim for runtime/backend governance.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-report-family-intake.md](./artifacts/day1-report-family-intake.md)
- [day2-report-family-inventory.md](./artifacts/day2-report-family-inventory.md)
- [day3-shared-metadata-contract.md](./artifacts/day3-shared-metadata-contract.md)
- [day4-normalized-index-generator-design.md](./artifacts/day4-normalized-index-generator-design.md)
- [day5-metadata-contract-implementation.md](./artifacts/day5-metadata-contract-implementation.md)
- [day6-normalized-index-generator-implementation.md](./artifacts/day6-normalized-index-generator-implementation.md)
- [day7-corpus-oracle-index-integration.md](./artifacts/day7-corpus-oracle-index-integration.md)
- [day8-benchmark-sentinel-guardrail-index-integration.md](./artifacts/day8-benchmark-sentinel-guardrail-index-integration.md)
- [day9-quality-package-index-integration.md](./artifacts/day9-quality-package-index-integration.md)
- [day10-freshness-gate-design.md](./artifacts/day10-freshness-gate-design.md)
- [day11-freshness-gate-implementation.md](./artifacts/day11-freshness-gate-implementation.md)
- [day12-documentation-alignment.md](./artifacts/day12-documentation-alignment.md)
- [day13-validation-and-quality-gates.md](./artifacts/day13-validation-and-quality-gates.md)
- [day14-closeout-and-sprint142-handoff.md](./artifacts/day14-closeout-and-sprint142-handoff.md)

## Closeout

Sprint 141 is complete. It closes the report-index normalization and
freshness-gate sprint with a maintained metadata contract, deterministic
normalized index generator, explicit freshness diagnostics, focused tests,
updated documentation, final validation evidence, and a narrow Sprint 142
runtime/backend governance handoff. It does not promote generated local
reports into source control or widen public claims beyond report navigation,
freshness diagnostics, and source-controlled ownership metadata.
