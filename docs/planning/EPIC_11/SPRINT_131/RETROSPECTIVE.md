# Sprint 131 Retrospective

**Sprint:** 131 - Numerical Corpus, Coverage Architecture & Report Indexes
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 131 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 11 Sprint 131 scope and Sprint 120-130 oracle, residual,
      optional-corpus, helper, and solver-selection claim gates.
- [x] Established duplicate fences around checked-in smoke fixtures,
      generated matrix families, benchmark timing rows, coverage percentages,
      dead-code findings, optional corpus paths, and guardrail reports.
- [x] Inventoried checked-in Matrix Market fixtures, SuiteSparse-derived
      corpus files, generated test families, external-reference helpers,
      expected failures, skips, optional gates, benchmark outputs, coverage
      outputs, dead-code outputs, and large-matrix guardrails.
- [x] Defined corpus taxonomy for structure, numerical properties, evidence
      class, oracle provenance, ownership, availability, support tier,
      expected failures, promotion, demotion, and non-claims.
- [x] Applied the taxonomy to representative parser, direct solver,
      Cholesky CSC, external LU, QR projector, partial-SVD, eigensolver,
      large-matrix guardrail, and integration rows.
- [x] Designed report-index requirements for benchmark, sentinel, coverage,
      dead-code, large-matrix, oracle, planning, and benchmark-local report
      families.
- [x] Selected the existing large-matrix guardrail `index.tsv` as the first
      generated report/index candidate.
- [x] Re-ranked coverage gaps by user-facing workflow, numerical risk,
      platform risk, corpus availability, claim impact, and owner readiness.
- [x] Preserved coverage as tree-mutating supplemental evidence, not reviewed
      behavioral completeness.
- [x] Defined dead-code and guardrail actionability, false-positive, waiver,
      stale-report, and index-eligibility policies.
- [x] Accepted the existing `make large-matrix-guardrails` output as Sprint
      131's first generated index path without schema changes.
- [x] Validated the large-matrix guardrail index path and defined freshness,
      drift, missing-input, optional-data, and residual validation policy.
- [x] Published recurring owners, orphaned-output status, supplemental-to-
      reviewed promotion criteria, no-update maintainer rationale, residual
      assurance queue, closeout package, and Sprint 132 handoff.
- [x] Ran `make large-matrix-guardrails` for the accepted first generated
      index path.
- [x] Ran final documentation hygiene with `git diff --check` and the Sprint
      131 markdown trailing-whitespace scan.

## What Went Well

1. **The sprint separated fixture evidence from report evidence.** Matrix
   Market fixtures, generated families, external helpers, benchmarks,
   coverage, dead-code, guardrails, and planning artifacts now have distinct
   roles instead of sharing one ambiguous evidence bucket.

2. **The taxonomy made promotion rules explicit.** Rows now need stable keys,
   owners, oracle/output matches, support tiers, validation commands,
   freshness rules, and non-claim boundaries before they can become reviewed
   recurring assurance.

3. **The first generated index decision avoided premature normalization.** The
   existing large-matrix guardrail `index.tsv` already had stable lane IDs,
   reviewed/supplemental categories, skip rows, artifacts, commands, and a
   manifest, so Sprint 131 accepted it without forcing a broader schema.

4. **Coverage risk was ranked by impact instead of percentage alone.** Direct
   solver, iterative/preconditioner, and SVD/bidiag gaps now have higher
   reviewed-risk priority when they affect public solve correctness,
   convergence, or failure semantics.

5. **Dead-code and guardrail meanings stayed bounded.** Dead-code reports
   remain triage and report-completeness evidence; large-matrix guardrails
   remain structural and bounded CSV-shape evidence.

## What Did Not Go Well

1. **Most corpus rows still lack generated index metadata.** The taxonomy is
   ready, but broad SuiteSparse, integration, product-observed, and
   expected-error rows still need row-level owner and oracle packages before a
   generated corpus index is safe.

2. **Cross-report normalization remains deferred.** Coverage, dead-code,
   benchmark, guardrail, oracle, and planning rows have different freshness
   and failure semantics, so Sprint 131 correctly stopped short of one schema.

3. **Coverage remains operationally expensive.** Coverage reports are useful,
   but still tree-mutating, backend-specific, and supplemental.

4. **Dead-code freshness is incomplete.** `report.tsv` has useful bucket
   semantics but lacks manifest-style branch, commit, and timestamp metadata.

5. **Supplemental guardrail lanes are not recurring evidence yet.** `S1` and
   `S2` remain opt-in, threshold-free reports until runtime and support-tier
   policy exist.

## Final Metrics

| Metric | Sprint 131 close state |
|---|---:|
| Sprint 131 artifact files | 14 |
| retrospective files | 1 |
| checked-in fixture and generated-family inventory artifacts | 2 |
| external-reference/expected-failure inventory artifacts | 1 |
| taxonomy and tagging artifacts | 2 |
| report-index design artifacts | 2 |
| coverage architecture artifacts | 1 |
| dead-code/guardrail architecture artifacts | 1 |
| first generated index accepted | 1 |
| source/schema changes for first index | 0 |
| maintainer-guide wording updates | 0 |
| residual queue artifacts | 2 |
| `make large-matrix-guardrails` validation | passed |
| final diff hygiene | passed |
| final Sprint 131 markdown whitespace scan | passed |
| full C quality gate | not required; documentation-only sprint |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake and duplicate fencing | Completed in working notes and Day 1 artifact. |
| Numerical fixture inventory | Completed for checked-in Matrix Market files, SuiteSparse-derived files, and generated matrix families. |
| External-reference and expected-failure inventory | Completed for five helper scripts, helper outputs, skips, expected failures, and optional gates. |
| Corpus taxonomy | Completed with structural, numerical, evidence, oracle, ownership, availability, support-tier, promotion, demotion, and non-claim rules. |
| Corpus tagging dry run | Completed for representative parser, direct solver, QR, SVD, eigensolver, guardrail, and integration rows. |
| Report-index requirements | Completed for benchmark, sentinel, coverage, dead-code, large-matrix, oracle, planning, and benchmark-local report families. |
| First index design | Large-matrix guardrail index selected because reviewed/supplemental semantics were already stable. |
| Coverage architecture | Completed with risk ranking, reviewed/supplemental split, owner labels, report-index gates, and residual coverage queue. |
| Dead-code and guardrail architecture | Completed with output inventory, actionability policy, false-positive policy, waiver policy, stale-report policy, and index eligibility. |
| First generated index implementation | Existing large-matrix guardrail `index.tsv` accepted without schema changes. |
| Index validation and freshness | Completed with schema inspection, freshness labels, drift responsibilities, and missing/optional behavior. |
| Ownership map | Completed with recurring owners, orphaned-output register, promotion criteria, no-update maintainer rationale, and future-owner queue. |
| Validation and residual queue | Completed with final docs checks, residual support-tier classification, and closeout inputs. |
| Sprint closeout | Completed with project-plan reconciliation, validation package, public/maintainer claim review, and Sprint 132 handoff. |

## Residual Deferred Debt

Most important carry-forward work:

- Generate a corpus index only after row-level SuiteSparse, integration,
  product-observed, expected-error, oracle, tolerance, runtime, and support
  metadata exist.
- Add a coverage index only after preserving backend, threshold,
  tree-mutating, freshness, source-filter, reset, support-tier, and claim
  boundary fields.
- Add dead-code freshness metadata without weakening bucket classification or
  turning reports into cleanup proof.
- Build a stale-report scanner after a common metadata contract exists across
  report families.
- Create an external-reference helper index that preserves helper-specific
  output classes and skip behavior.
- Decide whether supplemental large-matrix lanes need recurring validation or
  should remain opt-in threshold-free reports.
- Resolve primary owners for integration fixtures before promoting them to
  reviewed corpus rows.
- Revisit maintainer-guide wording only if a future sprint changes target
  behavior, schema, support tier, CI role, or public claim.

Still consciously constrained rather than silently solved:

- no broad Matrix Market or SuiteSparse corpus coverage claim;
- no LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  vendor-backend parity claim;
- no raw basis-vector, sign, orientation, eigenvector, or singular-vector
  parity claim when only projector, residual, value, or rank evidence exists;
- no coverage percentage as reviewed behavioral completeness;
- no dead-code report as removal-ready proof;
- no benchmark row as correctness or portable performance proof;
- no large-matrix guardrail as broad scalability, memory, timing, or corpus
  proof;
- no freshness label as a CI, release, platform, or support guarantee.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-assurance-intake.md](./artifacts/day1-assurance-intake.md)
- [day2-numerical-fixture-inventory.md](./artifacts/day2-numerical-fixture-inventory.md)
- [day3-external-reference-expected-failure-inventory.md](./artifacts/day3-external-reference-expected-failure-inventory.md)
- [day4-corpus-taxonomy-policy.md](./artifacts/day4-corpus-taxonomy-policy.md)
- [day5-corpus-tagging-dry-run.md](./artifacts/day5-corpus-tagging-dry-run.md)
- [day6-report-index-requirements.md](./artifacts/day6-report-index-requirements.md)
- [day7-report-index-design.md](./artifacts/day7-report-index-design.md)
- [day8-coverage-gap-architecture.md](./artifacts/day8-coverage-gap-architecture.md)
- [day9-deadcode-guardrail-architecture.md](./artifacts/day9-deadcode-guardrail-architecture.md)
- [day10-first-index-implementation.md](./artifacts/day10-first-index-implementation.md)
- [day11-index-validation-freshness-policy.md](./artifacts/day11-index-validation-freshness-policy.md)
- [day12-coverage-report-ownership-map.md](./artifacts/day12-coverage-report-ownership-map.md)
- [day13-validation-residual-assurance-queue.md](./artifacts/day13-validation-residual-assurance-queue.md)
- [day14-closeout-report-index-handoff.md](./artifacts/day14-closeout-report-index-handoff.md)

## Final Status

Sprint 131 is complete. It delivered a numerical corpus and report assurance
architecture, accepted the existing large-matrix guardrail `index.tsv` as the
first generated report/index path, preserved coverage and dead-code as bounded
supplemental/triage signals, published recurring owners and residual queues,
and handed Sprint 132 a clear set of generated-index and freshness candidates.
