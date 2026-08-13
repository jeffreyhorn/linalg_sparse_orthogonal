# Sprint 155 Retrospective

**Sprint:** 155 - Tutorial, Header Cleanup & API Reference Coherence
**Duration:** 14 days (Days 1-14 landed on branch `sprint-155`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 155 day-by-day plan, working notes, artifact directory,
      closeout artifact, Sprint 156 handoff, and retrospective.
- [x] Audited `docs/tutorial.md` against README, INSTALL, examples, cookbook,
      solver-selection, maintainer/report guidance, public headers, generated
      API reference, and Sprint 154 comparison evidence.
- [x] Reworked `docs/tutorial.md` around build, first maintained solve,
      data-input routing, solver choice, diagnostics, install handoff,
      advanced controls, benchmarks/reports, public headers, and API reference.
- [x] Selected a high-impact public-header cleanup batch:
      `sparse_ldlt.h`, `sparse_ic.h`, `sparse_eigs.h`, and
      `sparse_analysis.h`.
- [x] Defined public-header cleanup guardrails before editing headers.
- [x] Cleaned selected public-header comments while preserving declarations,
      signatures, typedefs, enum values, struct fields, macros, include guards,
      installed header names, and exported names.
- [x] Captured before/after/current declaration-preservation evidence and
      normalized empty diffs.
- [x] Added `docs/api_reference.md` as the user-facing API reference entry
      point and linked it from README, tutorial, and cookbook.
- [x] Added generated Doxygen API-reference freshness and publication guidance
      to `docs/maintainer_guide.md`.
- [x] Explicitly deferred generated API HTML refresh after documenting that the
      current `docs/api/html/` surface is partial.
- [x] Ran the full required C quality gate because public headers changed:
      `make format && make lint && make test`.
- [x] Ran whitespace, link-target, stale-phrase, declaration-preservation, and
      unsupported-claim scans.
- [x] Prepared the Sprint 156 final validation and claim-recalibration handoff.

## What Went Well

1. **The tutorial became a real adoption path.** It now starts with the
   smallest build-tree solve, routes CSR/CSC/Matrix Market users through the
   cookbook and maintained examples, and defers install/package/report policy
   to the docs that own those details.

2. **The header cleanup was bounded before edits began.** Day 6 selected a
   focused four-header batch, and Day 7 defined allowed edits, disallowed
   declaration drift, claim boundaries, and validation commands before any
   public-header comments changed.

3. **Declaration preservation is visible.** Days 8, 9, 12, and 13 produced
   before/after/current scans. The normalized declaration diffs are all empty,
   giving reviewers a direct way to see that the header work was comment-only.

4. **The API reference story stopped being implicit.** The new
   `docs/api_reference.md` page gives users a stable index into public headers
   and generated Doxygen HTML without pretending that the generated HTML is
   complete or fresh.

5. **Generated documentation freshness is now policy, not folklore.**
   `docs/maintainer_guide.md` now says when generated API HTML can be treated
   as fresh, how to review generated output, and when to call it stale or
   partial.

6. **The branch was validated at the right level.** Because public headers
   changed, Sprint 155 ran the full `make format && make lint && make test`
   gate and refreshed declaration evidence after formatting.

## What Didn't Go Well

1. **Generated API HTML remains partial.** The sprint found that
   `docs/api/html/` has generated pages for `13` of `18` checked-in public
   headers, but intentionally did not refresh the generated tree because that
   would create a large generated-output diff.

2. **The generated version header remains outside Doxygen input.**
   `sparse_version.h` is installed from the build include directory, while the
   current `Doxyfile` reads checked-in `include/*.h`. That needs a separate
   build-aware documentation decision.

3. **Only selected headers were cleaned.** The sprint cleaned the highest-value
   batch, but remaining headers outside the Day 6 selection still need the
   same selection/contract/declaration-preservation process if they are
   touched later.

4. **The API reference index is intentionally lightweight.** It improves
   discoverability, but it is not a generated reference refresh and does not
   replace exact header comments or future Doxygen publication work.

5. **Claim hygiene still needs Epic-wide closeout.** Sprint 155 preserved
   local non-claims, but Sprint 156 still needs the final Epic 13 public
   claim/non-claim audit across all support surfaces.

## Final Metrics

### Validation

| Metric | Sprint 155 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked public `.h` changes | yes: `sparse_ldlt.h`, `sparse_ic.h`, `sparse_eigs.h`, `sparse_analysis.h` |
| full C quality gate required | yes |
| full C quality gate | passed: `make format && make lint && make test` |
| final test output | `All tests passed.` |
| declaration preservation | passed: Day 8, Day 9, and Day 12 normalized diffs are `0` bytes |
| API-reference link-target checks | passed |
| stale phrase scan | passed: `API reference surface` and `generated API reference` absent |
| focused unsupported-claim scan | passed; active hits are explicit non-claim wording |
| `git diff --check` | passed |

### Artifact Package

| Metric | Sprint 155 close state |
| --- | ---: |
| daily artifacts under `SPRINT_155/artifacts/` | 22 |
| plan files | 1 |
| working notes files | 1 |
| sprint retrospective files | 1 |
| new user-facing docs | 1 |
| public/support docs changed | 4 |
| public header files changed | 4 |
| C source files changed | 0 |
| generated API HTML refreshed | 0 |
| declaration scan files | 7 |
| normalized declaration diffs | 3 |

## Closed Claim

Sprint 155 closes this adoption/documentation coherence claim:

The project now has an aligned tutorial, a compact API reference entry point,
selected public-header comment cleanup, maintainer-owned generated API
reference freshness policy, and declaration-preservation evidence showing that
the edited public-header batch preserved declarations and installed API shape.

This claim is supported by:

- [PLAN.md](./PLAN.md);
- [WORKING_NOTES.md](./WORKING_NOTES.md);
- [day1-documentation-baseline.md](./artifacts/day1-documentation-baseline.md);
- [day2-tutorial-audit.md](./artifacts/day2-tutorial-audit.md);
- [day3-tutorial-flow-design.md](./artifacts/day3-tutorial-flow-design.md);
- [day4-tutorial-core-rewrite.md](./artifacts/day4-tutorial-core-rewrite.md);
- [day5-tutorial-alignment-summary.md](./artifacts/day5-tutorial-alignment-summary.md);
- [day6-header-cleanup-selection.md](./artifacts/day6-header-cleanup-selection.md);
- [day7-header-cleanup-contract.md](./artifacts/day7-header-cleanup-contract.md);
- [day8-header-cleanup-summary.md](./artifacts/day8-header-cleanup-summary.md);
- [day9-header-cleanup-summary.md](./artifacts/day9-header-cleanup-summary.md);
- [day10-api-reference-publication-plan.md](./artifacts/day10-api-reference-publication-plan.md);
- [day11-api-reference-guidance-implementation.md](./artifacts/day11-api-reference-guidance-implementation.md);
- [day12-preservation-and-reconciliation.md](./artifacts/day12-preservation-and-reconciliation.md);
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md);
- [day14-closeout-sprint156-handoff.md](./artifacts/day14-closeout-sprint156-handoff.md).

## Next-Sprint Readiness

Sprint 156 can begin from this baseline:

| Starting item | Required posture |
| --- | --- |
| Tutorial and adoption docs | Treat Sprint 155 tutorial/API-reference changes as the current adoption baseline. |
| Public header cleanup | Use the Day 6/Day 7 selection and cleanup contract before touching additional headers. |
| Declaration preservation | Keep Day 8/Day 9/Day 12 scan pattern for any further public-header edits. |
| API reference | `docs/api_reference.md` is the user-facing index; generated HTML is partial until refreshed under the maintainer policy. |
| Generated API HTML | Do not cite `docs/api/html/` as complete or fresh unless `make docs` is run, warnings are triaged, and coverage is checked. |
| Final claim audit | Rescan public/support docs for unsupported ABI, package, platform, performance, external-parity, generated-reference completeness, or state-of-the-art claims. |
| Epic closeout | Fold Sprint 155 evidence into final Epic 13 evidence inventory, residual queue, and retrospective. |

## Residual Deferred Debt

Still explicitly unresolved at Sprint 155 close:

- generated API HTML refresh for the current public-header set;
- missing generated pages for `sparse_analysis.h`, `sparse_eigs.h`,
  `sparse_ic.h`, `sparse_ldlt.h`, and `sparse_lu_csr.h`;
- generated installed `sparse_version.h` Doxygen input decision;
- public-header cleanup outside the selected Sprint 155 batch;
- final Epic 13-wide claim/non-claim audit;
- final Epic 13 residual queue with owners, blockers, prerequisites, and
  promotion gates.

Still consciously constrained rather than silently solved:

- `docs/api_reference.md` is an index, not generated reference completeness;
- generated Doxygen HTML is a convenience view of configured inputs, not a
  package, ABI, platform, or completeness claim;
- selected header comment cleanup is not public API redesign;
- declaration preservation is local to the edited header batch;
- fixture-local QR, partial-SVD, and comparison evidence remains bounded;
- full local quality-gate success is not hosted cross-platform proof;
- no dynamic ABI, shared-library, package-manager, broad Windows parity,
  external-library parity, portable performance, or state-of-the-art claim was
  earned.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [RETROSPECTIVE.md](./RETROSPECTIVE.md)
- [day1-documentation-baseline.md](./artifacts/day1-documentation-baseline.md)
- [day2-tutorial-audit.md](./artifacts/day2-tutorial-audit.md)
- [day3-tutorial-flow-design.md](./artifacts/day3-tutorial-flow-design.md)
- [day4-tutorial-core-rewrite.md](./artifacts/day4-tutorial-core-rewrite.md)
- [day5-tutorial-alignment-summary.md](./artifacts/day5-tutorial-alignment-summary.md)
- [day6-header-cleanup-selection.md](./artifacts/day6-header-cleanup-selection.md)
- [day7-header-cleanup-contract.md](./artifacts/day7-header-cleanup-contract.md)
- [day8-header-cleanup-summary.md](./artifacts/day8-header-cleanup-summary.md)
- [day8-header-declarations-before.txt](./artifacts/day8-header-declarations-before.txt)
- [day8-header-declarations-after.txt](./artifacts/day8-header-declarations-after.txt)
- [day8-header-declarations-normalized-diff.txt](./artifacts/day8-header-declarations-normalized-diff.txt)
- [day9-header-cleanup-summary.md](./artifacts/day9-header-cleanup-summary.md)
- [day9-header-declarations-before.txt](./artifacts/day9-header-declarations-before.txt)
- [day9-header-declarations-after.txt](./artifacts/day9-header-declarations-after.txt)
- [day9-header-declarations-normalized-diff.txt](./artifacts/day9-header-declarations-normalized-diff.txt)
- [day10-api-reference-publication-plan.md](./artifacts/day10-api-reference-publication-plan.md)
- [day11-api-reference-guidance-implementation.md](./artifacts/day11-api-reference-guidance-implementation.md)
- [day12-header-declarations-current.txt](./artifacts/day12-header-declarations-current.txt)
- [day12-header-declarations-normalized-diff.txt](./artifacts/day12-header-declarations-normalized-diff.txt)
- [day12-preservation-and-reconciliation.md](./artifacts/day12-preservation-and-reconciliation.md)
- [day13-integrated-validation.md](./artifacts/day13-integrated-validation.md)
- [day14-closeout-sprint156-handoff.md](./artifacts/day14-closeout-sprint156-handoff.md)
