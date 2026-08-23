# Day 14: Sprint And Epic Closeout

## Purpose

Day 14 finalizes Sprint 176 records, verifies the repository closeout state,
and prepares the Sprint 176 and Epic 15 handoff for review.

## Closeout Review

| Area | Closeout state |
| --- | --- |
| Day-by-day evidence trail | Complete: Day 1 through Day 14 artifacts are present under `SPRINT_176/artifacts/`. |
| Working notes | Updated with final validation, retrospective finalization, closeout decisions, and daily log entries. |
| Sprint retrospective | Created at `docs/planning/EPIC_15/SPRINT_176/RETROSPECTIVE.md`. |
| Epic retrospective | Created at `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md`. |
| Allocation-failure evidence | Selected iterative repeated-run handle proof is documented and guarded. |
| Claim boundaries | Broad allocation-failure, state-of-the-art, package-manager, shared-library, ABI, platform, report, generated API hosting, and release non-claims are retained. |
| Generated output staging | No generated outputs under `build/`, `coverage/`, or `docs/api/` were staged. |

## Final Validation Status

The final integrated validation baseline remains the Day 12 record:

```sh
make iterative-allocation-failure-gate &&
bash scripts/package_manager_deferral_check.sh &&
bash scripts/static_package_deferral_check.sh &&
python3 tests/test_normalize_report_index.py &&
python3 tests/test_selected_comparison_workflow.py &&
python3 tests/test_bench_canonical_freshness.py
```

Result: passed.

Required source/header gate:

```sh
make format && make lint && make test
```

Result: passed on Day 12.

Day 14 changed planning documentation only, so the full C gate was not rerun.

Final closeout hygiene:

```sh
git diff --check
```

Result: passed.

## Sprint 176 Handoff

Sprint 176 closes the selected allocation-failure evidence gap for the
iterative repeated-run handle family:

- private deterministic allocation-failure hooks are available through the
  internal allocation helper layer;
- selected owner allocation, CG workspace, GMRES growth, MINRES growth, and
  invalid prepare state-publication cases are tested;
- `make iterative-allocation-failure-gate` provides the maintained focused
  proof;
- `ctest -L allocation_failure` can select the same proof from CMake;
- public and maintainer docs describe the cleanup invariant and proof scope.

## Epic 15 Handoff

Epic 15 is closed with explicit residuals in
`docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md`.

Highest-priority residuals:

1. broader allocation-failure coverage;
2. hosted generated API HTML;
3. package-manager provider support;
4. shared-library and dynamic ABI support;
5. Windows report freshness;
6. selected oracle freshness beyond Linux;
7. bounded external comparison expansion;
8. portable performance publication;
9. broader public-header coherence;
10. workflow target-list deduplication.

## Review Notes

The branch contains source/header changes from Days 5-8 and planning/docs
changes from the remaining closeout days. The source/header changes were
validated by the full C gate on Day 12. Day 14 adds only planning closeout
artifacts and therefore uses `git diff --check` as the final local hygiene
gate.
