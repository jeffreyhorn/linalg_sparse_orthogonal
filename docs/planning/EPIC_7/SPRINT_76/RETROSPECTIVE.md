# Sprint 76 Retrospective

**Sprint:** 76 — Benchmark Governance, Profiling & Longitudinal Reporting  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 76 scope, benchmark-governance hotspot map, and validation
      baseline were fixed before any landing work began
- [x] the strongest live contradiction was re-ranked to the canonical report
      workflow/schema lane rather than treated as generic benchmark churn
- [x] the first landing stayed bounded to:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- [x] the canonical report surface is now stronger without widening what is
      canonical:
  - the same four maintained benchmark emitters remain canonical
  - one CSV per canonical emitter remains the numeric artifact surface
  - the report remains threshold-free
- [x] the canonical bundle now carries bounded longitudinal metadata:
  - `report_label`
  - `git_commit`
  - `git_branch`
  - explicit artifact inventory
  - `index.tsv`
  - `manifest.txt`
- [x] support-surface reconciliation landed in the right owners:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- [x] `README.md` was correctly left support-only rather than forced into a
      redundant top-level rewrite
- [x] Sprint 76 preserved the narrower benchmark-policy truth:
  - `make bench-canonical-report` is threshold-free canonical reporting
  - `bench-fast` stays the bounded runtime lane
  - `wall-check` stays the narrow thresholded regression gate
  - `bench_reorder` and `bench_amd_qg` stay runtime/reporting context only
- [x] no new threshold machinery or portable pass/fail timing gate was added
- [x] the final docs/proof alignment closed as an explicit bounded no-op
- [x] the full Sprint 76 branch passed the standard code-day gate, the
      strongest reviewed baseline, and the focused benchmark/example/install
      follow-ons
- [x] Sprint 76 closed with one explicit validated benchmark-governance
      package and a ranked carry-forward queue

## What Went Well

1. **Sprint 76 strengthened the canonical reporting surface without changing what counts as canonical.**
   The main landing improved report comparability and artifact readability
   without reopening benchmark taxonomy. The four maintained canonical emitters
   stayed fixed, one CSV per emitter stayed fixed, and the gain came from
   bounded metadata rather than from a new benchmark regime.

2. **The workflow landing stayed properly bounded.**
   The code-bearing batch stayed on:
   - `scripts/bench_canonical_report.sh`
   - `Makefile`
   and did not widen into:
   - benchmark-driver rewrites
   - new timing thresholds
   - runtime or exploratory benchmark capture
   - widened backend or platform claims

3. **The support surfaces now explain the stronger bundle cleanly.**
   `benchmarks/README.md` and `docs/maintainer_guide.md` now name the stronger
   bundle directly, including `index.tsv`, `manifest.txt`, explicit artifact
   inventory, timestamp, label support, and git metadata support.

4. **Sprint 76 avoided a fake threshold-policy landing.**
   The sprint reranked correctly: after the reporting and support-surface work,
   the strongest remaining action was to preserve the current threshold split
   rather than force a new threshold batch. That kept the sprint truthful.

5. **The validated close state is strong.**
   Sprint 76 ended with:
   - `make format` passed
   - `make lint` passed
   - `make test` passed
   - `make quality-review-full` passed
   - reviewed CMake parity still exact at `53`
   - Makefile/CMake parity still `53 vs 53`
   - reviewed CMake `ctest` still `53 / 53`
   - focused reviewed benchmark/example/install follow-ons still clean

## What Didn't Go Well

1. **Sprint 76 improves governance, not benchmark verdict power.**
   That was the right bounded outcome, but it means the sprint does not deliver:
   - portable performance thresholds
   - historical pass/fail comparison bands
   - any stronger machine-independent “state of the art” timing claim

2. **The threshold lane remains intentionally narrow.**
   `wall-check` remains a justified narrow regression gate, but Sprint 76 does
   not solve broader questions around timing stability or how much thresholded
   performance governance should ever expand.

3. **Exploratory and regression-sensitive lanes remain deliberately outside canonical truth.**
   That is correct, but it leaves ongoing interpretive tension around:
   - `bench-fast`
   - `bench_reorder`
   - `bench_amd_qg`
   Later work still needs to preserve that separation.

4. **The reviewed validation path still has a dominant runtime hotspot.**
   Sprint 76 closed cleanly, but `test_reorder_nd` still dominated reviewed
   CMake runtime. That is operational friction for future proof-heavy sprints
   even though Sprint 76 itself was not a reorder sprint.

5. **The sprint depended on disciplined non-moves.**
   Success required not turning longitudinal reporting into threshold policy,
   not turning benchmark docs into product overclaim, and not widening runtime
   or exploratory surfaces into canonical proof. That discipline held, but the
   deferred pressure remains real.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 76 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `346.44 sec` |
| reviewed `test_reorder_nd` time | `244.00 sec` |
| install regression | `11 / 11` |
| CMake install regression | `13 / 13` |

### Sprint 76 artifact package

| Metric | Sprint 76 close state |
|---|---:|
| total artifact files under `SPRINT_76/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/landing artifacts | `6` |
| review/closeout artifacts | `3` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-benchmark-governance-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-truth-surface-recheck.md`
  - `day3-benchmark-governance-reaudit.md`
  - `day4-first-governance-boundary.md`
  - `day7-post-landing-audit-and-rerank.md`
- design/landing artifacts:
  - `day5-longitudinal-report-design.md`
  - `day6-canonical-reporting-batch.md`
  - `day8-support-surface-reconciliation-design.md`
  - `day9-support-surface-reconciliation-batch.md`
  - `day10-threshold-and-comparison-policy-design.md`
  - `day11-threshold-and-comparison-recheck.md`
- review/closeout artifacts:
  - `day12-docs-alignment-and-final-validation-queue.md`
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed governance/reporting package

| Metric | Sprint 76 close state |
|---|---:|
| workflow/reporting sources touched | `2` |
| maintained benchmark docs touched | `1` |
| maintained policy docs touched | `1` |
| canonical benchmark drivers touched | `0` |
| public headers touched | `0` |
| implementation `.c` files touched | `0` |

Notes:

- workflow/reporting sources touched:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- maintained benchmark docs touched:
  - `benchmarks/README.md`
- maintained policy docs touched:
  - `docs/maintainer_guide.md`
- intentionally untouched:
  - canonical benchmark driver sources
  - reviewed proof-owner tests
  - `README.md`

## Residual Deferred Debt

Sprint 76 deliberately stopped after the bounded governance/reporting landing.
The main open work it intentionally hands forward is:

- eigensolver backend/runtime parity as the strongest remaining backend-aware
  second lane after Sprint 75
- QR and SVD backend-aware follow-through only where a bounded proof-backed
  seam justifies movement
- later packaging, ABI, or platform convergence only where maintained
  evidence supports a stronger claim
- later permanent-surface cleanup only after the higher-value backend and
  capability lanes move

Still consciously constrained rather than silently “solved”:

- no new threshold machinery
- no portable pass/fail timing gate on the canonical report surface
- no widened benchmark claim detached from retained measured evidence
- no silent promotion of runtime or exploratory lanes into canonical truth

Not carried forward as unresolved Sprint 76 debt:

- the benchmark-governance rerank
- the Day 6 canonical reporting batch
- the Day 9 benchmark/policy support-surface reconciliation batch
- the Day 10/11 threshold-policy recheck and bounded no-op conclusion
- the Day 12 docs/proof alignment pass
- the full Day 13 validation sweep
- the Day 14 closeout and ranked Sprint 77 handoff queue

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-scope-and-benchmark-governance-baseline.md](./artifacts/day1-scope-and-benchmark-governance-baseline.md)
- [day1-authoritative-inputs.txt](./artifacts/day1-authoritative-inputs.txt)
- [day2-validation-baseline-and-truth-surface-recheck.md](./artifacts/day2-validation-baseline-and-truth-surface-recheck.md)
- [day3-benchmark-governance-reaudit.md](./artifacts/day3-benchmark-governance-reaudit.md)
- [day4-first-governance-boundary.md](./artifacts/day4-first-governance-boundary.md)
- [day5-longitudinal-report-design.md](./artifacts/day5-longitudinal-report-design.md)
- [day6-canonical-reporting-batch.md](./artifacts/day6-canonical-reporting-batch.md)
- [day7-post-landing-audit-and-rerank.md](./artifacts/day7-post-landing-audit-and-rerank.md)
- [day8-support-surface-reconciliation-design.md](./artifacts/day8-support-surface-reconciliation-design.md)
- [day9-support-surface-reconciliation-batch.md](./artifacts/day9-support-surface-reconciliation-batch.md)
- [day10-threshold-and-comparison-policy-design.md](./artifacts/day10-threshold-and-comparison-policy-design.md)
- [day11-threshold-and-comparison-recheck.md](./artifacts/day11-threshold-and-comparison-recheck.md)
- [day12-docs-alignment-and-final-validation-queue.md](./artifacts/day12-docs-alignment-and-final-validation-queue.md)
- [day13-full-validation-sweep.md](./artifacts/day13-full-validation-sweep.md)
- [day14-closeout-and-handoff.md](./artifacts/day14-closeout-and-handoff.md)

## Bottom Line

Sprint 76 accomplished the bounded benchmark-governance landing it was
supposed to accomplish.

It did not pretend to solve benchmark policy in the abstract. It made the
canonical report bundle materially stronger for longitudinal comparison, kept
the benchmark taxonomy honest, preserved the right threshold boundaries in the
right owners, and closed from a fully validated reviewed baseline.

That leaves Sprint 77 and later Epic 7 work in a better position: they can
start from a clearer evidence-based reporting contract rather than from a
loose benchmark backlog or a drift-prone threshold story.
