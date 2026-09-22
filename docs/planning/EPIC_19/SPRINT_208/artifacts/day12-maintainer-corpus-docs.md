# Sprint 208 Day 12 Maintainer And Corpus Documentation

## Scope

Day 12 aligned maintainer-facing, corpus, schema, and Epic 19 planning
documentation with the Sprint 208 selected Windows Cholesky re-deferral state.
The documentation now records that Sprint 208 reviewed current hosted evidence,
strengthened guard surfaces, and kept selected Windows freshness unpromoted.

## Changed Files

- `docs/maintainer_guide.md`
- `tests/corpus/README.md`
- `tests/corpus/schemas/report_index_fields.md`
- `docs/planning/EPIC_19/PROJECT_PLAN.md`

## Maintainer Documentation Updates

`docs/maintainer_guide.md` now:

- includes Sprint 208 artifacts in the Windows/PowerShell ownership row;
- states that Sprint 208 reviewed the current Sprint 190 Cholesky hosted path;
- names the strengthened manifest, workflow/PowerShell, and normalizer guard
  surfaces;
- keeps local missing `pwsh` as environment residual evidence, not pass
  evidence;
- keeps selected Windows freshness re-deferred until selected metadata,
  generated support tier, generated non-claim wording, and the claim contract
  are promoted together.

## Corpus And Schema Updates

`tests/corpus/README.md` now records Sprint 208 as the current review of the
bounded Windows Cholesky path and retains the distinction between guarded
workflow evidence and selected manifest promotion.

`tests/corpus/schemas/report_index_fields.md` now states that Sprint 208 keeps
`windows` absent from selected target `workflow_platforms` as source-controlled
authority and guards that absence with manifest, workflow, and normalizer
regressions.

## Epic 19 Status Update

`docs/planning/EPIC_19/PROJECT_PLAN.md` now describes Sprint 208 as in
progress on this branch rather than pending future execution. The status row is
limited to evidence present through Day 12 and does not close Sprint 208 before
integrated validation and closeout.

## Validation

Commands run:

```sh
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
python3 scripts/validate_corpus_schema.py
rg -n "Sprint 199 reviewed|Sprints 208 through 216|208-216 \\| Pending|Windows selected (Cholesky|comparison|report) freshness is promoted|PowerShell validation proves Windows report freshness" docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md docs/planning/EPIC_19/PROJECT_PLAN.md README.md INSTALL.md
```

Results:

- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`
- `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok`
- The stale/broad-claim search returned no matches.

## Retained Boundaries

Day 12 did not promote:

- selected Windows Cholesky freshness;
- broad Windows report freshness;
- Windows selected oracle or benchmark freshness;
- Windows QR incompatible selected freshness;
- package-manager support;
- shared-library or dynamic ABI support;
- performance, release, external-library parity, or state-of-the-art evidence.

## Completion Notes

Item 208.5 now has both public and maintainer-facing documentation aligned with
the Sprint 208 implementation state. Day 13 should run the integrated validation
matrix before closeout.
