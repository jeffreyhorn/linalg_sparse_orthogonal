# Sprint 135 Day 10 - Benchmark and Report Index Docs

## Purpose

Surface generated report indexes and local-measurement interpretation in
concise adoption language. Day 10 builds on Sprint 131's report-index decisions
without changing benchmark commands, generated schemas, CI policy, package
support, or performance claims.

## Sprint 131 Inputs

Reviewed inputs:

- `docs/planning/EPIC_11/SPRINT_131/artifacts/day6-report-index-requirements.md`
- `docs/planning/EPIC_11/SPRINT_131/artifacts/day10-first-index-implementation.md`
- `docs/planning/EPIC_11/SPRINT_131/artifacts/day11-index-validation-freshness-policy.md`
- `docs/planning/EPIC_11/SPRINT_131/artifacts/day14-closeout-report-index-handoff.md`
- current `benchmarks/README.md`

Key retained decisions:

- canonical benchmark reports already generate `index.tsv` and `manifest.txt`
  under `build/bench-reports/canonical/`
- performance sentinels generate `sentinels.tsv` and `manifest.txt` under
  `build/bench-reports/sentinels/`
- large-matrix guardrails generate `index.tsv` and `manifest.txt` under
  `build/bench-reports/large-matrix-guardrails/`
- Sprint 131 accepted the existing large-matrix guardrail `index.tsv` as the
  first generated report/index path
- cross-report normalized indexing remains deferred because row meanings differ
  across benchmark, sentinel, guardrail, coverage, dead-code, and oracle
  families

## Public Documentation Changes

### `benchmarks/README.md`

Added a `Report index handoff` section and quick-navigation entry.

The new section identifies:

| Target | Directory | Index | Context |
|---|---|---|---|
| `make bench-canonical-report` | `build/bench-reports/canonical/` | `index.tsv` | `manifest.txt` |
| `make performance-sentinels` | `build/bench-reports/sentinels/` | `sentinels.tsv` | `manifest.txt` |
| `make large-matrix-guardrails` | `build/bench-reports/large-matrix-guardrails/` | `index.tsv` | `manifest.txt` |

It also adds concise interpretation rules:

- start with generation command and report directory
- check manifest freshness context before comparing rows
- use lane id, sentinel id, command, artifact, and category before interpreting
  status
- treat skips, `n/a`, fallback, and supplemental rows as scope information
- keep CSV timing rows tied to the recorded environment
- regenerate reports rather than editing generated indexes by hand

### `docs/cookbook.md`

Updated the benchmark/report handoff with:

- `make large-matrix-guardrails`
- generated index/manifest locations for canonical, sentinel, and
  large-matrix guardrail reports
- a first-use reminder that indexes are artifact maps and freshness context

### `README.md`

Updated benchmark command guidance with:

- `index.tsv` / `manifest.txt` context for `make bench-canonical-report`
- a visible `make large-matrix-guardrails` command line in the build command
  list

## Claim Boundary Review

This batch does not claim:

- portable performance
- broad pass/fail timing behavior beyond the existing wall-check lane
- broad scalability or memory proof from large-matrix guardrails
- coverage completeness
- package-manager availability
- shared-library or dynamic-ABI support
- platform support-tier expansion
- normalized cross-report schema availability

The added wording describes report indexes as artifact maps, freshness context,
and local measurement interpretation aids.

## Residual Queue

Remaining report-doc work:

- Day 11 navigation alignment should verify that README, tutorial, cookbook,
  benchmark docs, algorithm reference, install docs, and maintainer guide have
  predictable first-use versus maintainer routing.
- Day 12 validation should re-run unsupported-claim scans across the expanded
  adoption surface.
- Future work may design a normalized cross-report index only if it preserves
  report-family row meaning instead of collapsing everything into pass/fail
  status.

## Validation Plan

Documentation-only validation for this batch:

- `git diff --check`
- trailing-whitespace scan on touched docs and Sprint 135 artifacts
- report-index path scan for canonical, sentinel, and large-matrix guardrail
  paths
- unsupported-claim scan for package, ABI, platform, performance, coverage, and
  cross-report schema wording
- `git diff --name-only -- '*.c' '*.h'` to confirm no code-day quality gate is
  required

## Completion Criteria

- generated report indexes are discoverable from first-use docs
- benchmark interpretation remains concise and evidence-bounded
- report docs do not duplicate maintainer history or imply unsupported claims
