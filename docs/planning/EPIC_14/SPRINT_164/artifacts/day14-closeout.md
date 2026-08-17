# Sprint 164 Day 14: Closeout And Retrospective Prep

## Purpose

Day 14 finalized Sprint 164 closeout evidence for the public-header/API
coherence batch and prepared the retrospective input set.

## Final Changed Surface

Source and public documentation changes:

- `include/sparse_matrix.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `README.md`
- `docs/solver_selection.md`
- `docs/tutorial.md`

Sprint evidence:

- `docs/planning/EPIC_14/SPRINT_164/PLAN.md`
- `docs/planning/EPIC_14/SPRINT_164/WORKING_NOTES.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day1-sprint-intake.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day2-header-selection.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day3-declaration-baseline-design.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day4-declaration-baseline.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day5-ownership-cleanup.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day6-error-output-cleanup.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day7-options-backend-cleanup.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day8-cross-link-cleanup.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day9-generated-reference-check.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day10-declaration-preservation.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day11-documentation-coherence.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day12-focused-validation.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day13-evidence-review.md`
- `docs/planning/EPIC_14/SPRINT_164/artifacts/day14-closeout.md`

## Final Decisions

- Sprint 164 selected a bounded public-header cleanup batch:
  `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`.
- Header edits stayed declaration-preserving and focused on comments,
  ownership/lifetime, status/error interpretation, backend-routing wording, and
  workflow navigation.
- Public docs were updated only where they contradicted the selected headers:
  README eigensolver backend summary and tutorial eigensolver result type.
- Generated API HTML remains local-only ignored output under the Sprint 158
  policy.
- Static-first package, shared-library, dynamic ABI, runtime-loader, and
  package-manager work remains a Sprint 165 package-boundary hardening topic.

## Final Validation Record

### Declaration Preservation

Day 14 re-ran the normalized declaration capture for the selected headers.

Final checksum:

```text
513db6c806353ea8d54deb7b9eef7c23e1444e4c0d59d0a979a0dd1fec8e1b41
```

This matches the Day 4 baseline, Day 10 recapture, and Day 12 post-format
recapture. The Day 14 before/final declaration diff produced no output:

```sh
diff -u \
  build/sprint164/declarations/selected-public-headers.before.normalized.txt \
  build/sprint164/declarations/selected-public-headers.day14.normalized.txt
```

### Generated API Reference

Day 14 re-ran:

```sh
make docs-check
```

Result:

```text
api-docs-coverage: PASS
  checked-in public headers: 18
  generated reference pages: 18
  generated source pages:    18
  generated sparse_version.h: separate installed-header policy row; not an expected page
```

### Full Public-Header Gate

Day 12 ran the full required public-header quality chain after the accumulated
header/doc edits:

```sh
make format && make lint && make test
```

Result: passed. `make test` ended with `All tests passed.`

Day 13 and Day 14 made planning-artifact-only changes, so Day 14 re-ran
targeted closeout checks rather than duplicating the full repository-wide
quality chain.

### Stale Wording And Claim Checks

Day 14 stale wording scan found no remaining hits for:

- `sparse_eigs_result_t`
- `sparse_eigs_sym(A, k, &opts, &result)` described as only grow-m Lanczos
- stale `via Lanczos (growing-m)` wording
- unbounded backend-superiority wording in the selected header/doc surfaces

Day 14 unsupported-claim scan found only explicit disclaimers, local
dispatch-policy wording, or maintainer policy boundaries for:

- package-manager support;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- Windows Makefile or Windows `pkg-config` parity;
- external-library parity;
- portable runtime or performance claims;
- hosted generated documentation;
- source-controlled generated HTML;
- backend superiority;
- state-of-the-art evidence.

### Generated And Local Evidence Churn

Day 14 checked:

```sh
git status --short -- docs/api/html build/sprint164/declarations
```

Result: no tracked generated HTML or local declaration-evidence churn.

### Diff Hygiene

Day 14 checked:

```sh
git diff --check
```

Result: passed.

## Sprint 165 Handoff

Sprint 165 should start from the following state:

- selected public-header cleanup is declaration-preserving and validated;
- public API docs now match the selected headers for eigensolver backend/result
  wording;
- generated API HTML policy remains local-only and covered by `make docs-check`;
- static-first package and ABI non-claims remain explicit;
- package metadata, install scripts, CMake package files, `sparse.pc`, version
  docs, and CI checks are ready for static-first boundary hardening.

Suggested Sprint 165 first checks:

- inspect CMake package files and `sparse.pc` for unsupported package, ABI,
  shared-library, runtime-loader, or package-manager wording;
- verify install/package tests still reject unsupported shared-library requests
  and do not imply dynamic ABI support;
- align README, INSTALL, maintainer guide, CMake docs, and package comments
  with the static-first contract;
- keep Sprint 164's public declarations frozen unless a future sprint
  explicitly approves API drift.

## Residual Queue

Deferred work after Sprint 164:

- broader non-selected-header public-comment cleanup:
  `sparse_ldlt.h`, `sparse_qr.h`, `sparse_svd.h`, `sparse_ilu.h`, and
  `sparse_ic.h`;
- table-wide README/API index reshaping;
- generated API HTML publication beyond local ignored output;
- package/ABI product changes or shared-library support;
- backend threshold retuning or new performance claims;
- exhaustive tutorial expansion for every option/result field;
- maintained helper script for declaration-preservation capture.

## Retrospective Inputs

Use these Sprint 164 artifacts when writing the retrospective:

- Day 1 for intake, scope, and quality constraints.
- Day 2 for selected-header rationale.
- Day 3 and Day 4 for declaration-preservation design and baseline.
- Days 5-8 for the actual header cleanup sequence.
- Day 9 for generated-reference policy alignment.
- Day 10 for before/after declaration preservation.
- Day 11 for public-doc coherence fixes.
- Day 12 for full validation.
- Day 13 for evidence review, non-claim trace, residuals, and Sprint 165
  handoff.
- Day 14 for final closeout status.

## Outcome

Sprint 164 deliverables are complete and traceable. The selected public-header
cleanup is declaration-preserving, generated-reference compatible, validated,
and handed off to Sprint 165 with a clear static-first package/API boundary.
