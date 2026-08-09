# Sprint 145 Retrospective

**Sprint:** 145 - Adoption Surface Simplification & High-Level Workflow Front Door
**Duration:** 14 days (Days 1-14 landed on branch `sprint-145`)
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 145 day-by-day plan, working notes, artifact directory,
      and closeout artifact.
- [x] Consumed Sprint 139-144 claim boundaries as prerequisite evidence:
  - QR claims remain bounded to fixture-local corpus proof;
  - partial-SVD claims remain bounded to fixture-local residual and
    convergence proof;
  - report semantics and freshness policy remain source-controlled;
  - runtime/backend governance remains local-control and sentinel scoped;
  - package/ABI posture remains static-first;
  - platform tiers remain Linux strongest reviewed source of truth, macOS
    reviewed static-first install/export proof, and Windows reviewed CMake
    subset plus supplemental CMake install/downstream confidence.
- [x] Audited README, INSTALL, cookbook, tutorial, solver-selection docs,
      examples, benchmark/report docs, maintainer guidance, report metadata,
      and public headers for first-use friction and stale support-tier wording.
- [x] Designed and implemented a concise high-level first-use route:
  - local build;
  - maintained first solve;
  - data-first input path;
  - solver choice;
  - diagnostics handoff;
  - static-first install/downstream handoff;
  - advanced controls after the first workflow is understood.
- [x] Updated the maintained example and cookbook ladder without changing
      example source.
- [x] Simplified README and INSTALL around first-use adoption while preserving
      support-tier and non-claim boundaries.
- [x] Aligned solver-selection and diagnostics handoff wording across README,
      examples, cookbook, and solver-selection docs.
- [x] Cleaned the four selected highest-impact public headers:
  - `include/sparse_matrix.h`;
  - `include/sparse_iterative.h`;
  - `include/sparse_qr.h`;
  - `include/sparse_svd.h`.
- [x] Preserved public API contracts during header cleanup:
  - no intentional signature, typedef, enum, macro, or struct-field changes;
  - caller ownership and NULL/error semantics remain documented;
  - QR and partial-SVD evidence remains bounded;
  - backend/runtime controls remain local workflow controls, not ABI,
    package, platform, or portable performance claims.
- [x] Repaired report-family coherence for runtime/backend policy rows by
      documenting `runtime_backend_governance_policy` as a source-controlled
      advisory row while keeping generated sentinel measurements under the
      `sentinel` family.
- [x] Published the adoption claim map, residual documentation debt ledger,
      public-header cleanup status, support-tier consistency check, and Sprint
      146 closeout handoff.
- [x] Ran Sprint 145 validation:
  - `python3 scripts/validate_corpus_schema.py`;
  - `python3 tests/test_normalize_report_index.py`;
  - `python3 scripts/normalize_report_index.py --family documentation
    --family package --family ci --family runtime_backend --check`;
  - `python3 scripts/normalize_report_index.py --family documentation
    --family package --family ci --family runtime_backend --check-freshness`;
  - `make examples-build`;
  - `bash tests/test_install.sh`;
  - `bash tests/test_cmake_install.sh`;
  - public-surface support-tier and unsupported-claim scans;
  - header declaration-preservation scans;
  - `git diff --check`;
  - `make format && make lint && make test`.

## What Went Well

1. **The sprint improved adoption without weakening proof boundaries.** The
   README, INSTALL, examples, cookbook, and solver-selection docs now route a
   first-use reader through a shorter path while still linking deeper evidence
   and support-tier owners.

2. **The workflow ladder gave each surface a clear job.** README owns the
   shortest path, INSTALL owns static-first install/downstream behavior,
   examples own runnable teaching workflows, cookbook owns data-first recipes,
   solver-selection owns problem-shape routing, and report/benchmark docs own
   local measurement interpretation.

3. **Header cleanup stayed appropriately narrow.** The sprint cleaned comments
   in four high-impact public headers and validated that declarations did not
   drift. Broader header cleanup was left as explicit residual debt instead of
   expanding the blast radius.

4. **Validation matched the changed surfaces.** Because public headers changed,
   the sprint ran the full local C gate. Because install/downstream and report
   metadata changed, it also ran install, CMake install, report schema,
   normalization, and freshness checks.

5. **The final claim map is useful input for Sprint 146.** Sprint 146 can now
   audit public claims against explicit source evidence instead of rebuilding
   the adoption history from scattered daily artifacts.

## What Didn't Go Well

1. **The tutorial was audited but not fully aligned.** Sprint 145 prioritized
   the higher-traffic adoption surfaces, leaving tutorial alignment as a
   source-owned residual.

2. **Report metadata drift still needed a late coherence repair.** Day 11 found
   stale runtime/backend row meaning. The fix was small and covered by tests,
   but it shows that source-controlled advisory rows can drift unless schema
   and normalization tests stay close to the docs.

3. **Header cleanup remains incomplete outside the selected surfaces.** The
   sprint intentionally avoided broad header churn, so several public headers
   still need their own cleanup design and validation.

4. **Hosted platform evidence is still external to the local closeout.** Local
   validation passed, but final Linux, macOS, and Windows support-tier
   reconciliation still depends on hosted CI results and belongs in Sprint
   146.

5. **Adoption work cannot close product gaps by itself.** The improved front
   door does not create shared-library ABI support, Windows parity, package
   manager distribution, portable performance proof, or state-of-the-art
   evidence.

## Final Metrics

### Validation

| Metric | Sprint 145 close state |
| --- | --- |
| tracked `.c` changes | no |
| tracked `.h` changes | yes, four selected public headers |
| full C quality gate required | yes |
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |
| maintained example build | passed: 14 example binaries built |
| Make install/downstream validation | passed: 23 passed, 0 failed |
| CMake install/downstream validation | passed: 26 passed, 0 failed, 0 skipped |
| corpus schema validation | passed |
| report normalization check | passed: 9 selected rows |
| report freshness check | passed: 9 selected source-controlled rows |
| public-header declaration-preservation scan | passed |
| support-tier and non-claim scans | passed with matches limited to explicit non-claims, support boundaries, or bounded evidence |
| `git diff --check` | passed |

### Artifact Package

| Metric | Sprint 145 close state |
| --- | ---: |
| daily artifacts under `SPRINT_145/artifacts/` | 14 |
| plan files | 1 |
| working notes files | 1 |
| retrospective files | 1 |
| public adoption docs changed | 5 |
| public headers changed | 4 |
| report metadata/schema/test files changed | 4 |
| example source files changed | 0 |
| generated report files committed | 0 |

## Closed Claim

Sprint 145 closes this claim:

The project now has a clearer first-use adoption surface that routes users from
local build to first solve, data import, solver choice, diagnostics,
static-first install/downstream consumption, and advanced controls while
preserving the bounded QR, partial-SVD, report, runtime/backend, package/ABI,
and platform support claims earned in Sprints 139-144.

This claim is supported by:

- `README.md`;
- `INSTALL.md`;
- `examples/README.md`;
- `docs/cookbook.md`;
- `docs/solver_selection.md`;
- `include/sparse_matrix.h`;
- `include/sparse_iterative.h`;
- `include/sparse_qr.h`;
- `include/sparse_svd.h`;
- `tests/corpus/manifests/report_families.tsv`;
- `tests/corpus/schemas/report_index_fields.md`;
- `scripts/validate_corpus_schema.py`;
- `tests/test_normalize_report_index.py`;
- Day 3 workflow front-door design;
- Day 5 example/cookbook batch;
- Day 6 README restructure;
- Day 7 INSTALL restructure;
- Day 8 solver front door;
- Day 10 public-header cleanup;
- Day 11 cross-surface coherence pass;
- Day 12 validation gate;
- Day 13 adoption claim map and residual-debt ledger;
- Day 14 closeout handoff.

## Sprint 146 Readiness

Sprint 146 should consume Sprint 145's adoption closeout packet as the public
claim baseline:

| Handoff field | Sprint 146 requirement |
| --- | --- |
| Evidence inventory | Start from the Day 13 adoption claim map and Day 14 changed-file inventory when assembling the final Epic 12 evidence list. |
| CI reconciliation | Reconcile hosted Linux, macOS, and Windows runs against the support tiers preserved in README, INSTALL, report metadata, and sprint evidence. |
| Claim audit | Recheck README, INSTALL, examples, cookbook, solver-selection, benchmark/report docs, public headers, and report-family metadata for unsupported state-of-the-art, package, ABI, platform, performance, report, or external-parity wording. |
| Residual queue | Carry forward tutorial alignment, broader header cleanup, generated report refreshes, Windows staged parity, shared-library/dynamic ABI productization, and competitive positioning. |
| State-of-the-art decision | Keep state-of-the-art status as a non-claim unless Sprint 146 finds direct comparative evidence that supports a narrower earned claim. |

## Residual Deferred Debt

Most important carry-forward work:

- tutorial alignment with the new first-use ladder;
- broader public-header cleanup beyond the four selected headers;
- hosted Linux, macOS, and Windows CI reconciliation after final PR runs;
- generated benchmark, coverage, dead-code, and sentinel report refreshes when
  corresponding measurement rows are needed;
- Windows reviewed install-validation parity;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows staged pthread/POSIX test portability;
- shared-library build/install/export support;
- dynamic ABI compatibility and ABI version policy;
- runtime-loader compatibility;
- package-manager distribution;
- portable performance evidence;
- final state-of-the-art competitive-positioning decision.

Still consciously constrained rather than silently solved:

- no broad QR, SVD, partial-SVD, LAPACK, NumPy, SciPy, or SuiteSparse parity;
- no broad repeated-spectrum, rank-deficient, sparse-output, convergence-rate,
  or performance guarantee beyond bounded evidence;
- no source-controlled generated-report freshness proof;
- no runtime/backend portable performance parity;
- no shared-library support;
- no dynamic ABI compatibility;
- no runtime-loader compatibility;
- no package-manager availability;
- no static/shared package selector support;
- no Windows Makefile or `pkg-config` parity;
- no Windows reviewed install-validation parity;
- no Windows staged test closure;
- no broad macOS platform parity beyond reviewed static-first install/export
  proof;
- no portable performance claim;
- no state-of-the-art claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-adoption-surface-intake.md](./artifacts/day1-adoption-surface-intake.md)
- [day2-adoption-friction-audit.md](./artifacts/day2-adoption-friction-audit.md)
- [day3-high-level-workflow-design.md](./artifacts/day3-high-level-workflow-design.md)
- [day4-example-cookbook-design.md](./artifacts/day4-example-cookbook-design.md)
- [day5-example-cookbook-batch.md](./artifacts/day5-example-cookbook-batch.md)
- [day6-readme-front-door-restructure.md](./artifacts/day6-readme-front-door-restructure.md)
- [day7-install-front-door-restructure.md](./artifacts/day7-install-front-door-restructure.md)
- [day8-solver-front-door.md](./artifacts/day8-solver-front-door.md)
- [day9-public-header-cleanup-design.md](./artifacts/day9-public-header-cleanup-design.md)
- [day10-public-header-cleanup.md](./artifacts/day10-public-header-cleanup.md)
- [day11-cross-surface-coherence.md](./artifacts/day11-cross-surface-coherence.md)
- [day12-validation-gate.md](./artifacts/day12-validation-gate.md)
- [day13-adoption-claim-map-residual-debt.md](./artifacts/day13-adoption-claim-map-residual-debt.md)
- [day14-closeout-handoff.md](./artifacts/day14-closeout-handoff.md)

## Closeout

Sprint 145 is complete. It simplified the first-use adoption path, aligned
README/INSTALL/examples/cookbook/solver-selection guidance, cleaned selected
public headers without API contract drift, repaired report-family coherence,
validated the cumulative header and documentation changes, and handed Sprint
146 a claim map plus residual-debt ledger for final Epic 12 closeout. It does
not implement or imply broad numerical parity, shared-library ABI support,
package-manager support, Windows parity, portable performance, or
state-of-the-art status.
