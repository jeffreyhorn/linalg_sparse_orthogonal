# Day 10 Normalizer Semantics Implementation

## Scope

Day 10 implements the Day 9 semantics audit for selected hosted report
freshness rows. The implementation is limited to
`scripts/normalize_report_index.py` and focused normalizer tests.

No C or public-header files were modified.

## Changed Files

| File | Change |
| --- | --- |
| `scripts/normalize_report_index.py` | Treat selected required current-commit generated oracle/comparison rows as `fresh` instead of generic `generated_present_unchecked` warnings, and rename comparison skip/defer diagnostics from optional-row wording to selected-row wording. |
| `tests/test_normalize_report_index.py` | Added synthetic selected comparison fixtures and tests for valid, missing, stale, duplicate, unexpected, failed, and deferred selected comparison rows. |
| `docs/planning/EPIC_13/SPRINT_159/WORKING_NOTES.md` | Recorded Day 10 implementation notes and Day 11 handoff. |
| `docs/planning/EPIC_13/SPRINT_159/artifacts/day10-normalizer-semantics-implementation.md` | Captured this implementation artifact. |

## Implementation Details

The normalizer now recognizes selected required generated rows before emitting
freshness diagnostics:

- oracle rows with `row_origin=generated_local`, `row_id` starting with
  `oracle_`, and selected oracle policy enabled;
- comparison rows with `row_origin=generated_local`, `row_id` starting with
  `comparison_`, and selected comparison policy enabled.

When those rows have a current `source_commit`, they are reported as:

```text
freshness: advisory: <row_id>: fresh: generated row source_commit matches current HEAD
```

This removes the previous successful-gate ambiguity where selected current
rows could still print:

```text
generated row exists but strict freshness comparison is pending
```

Stale rows still become `freshness: error` before this fresh-state override,
because `evaluate_freshness_state()` returns `stale` when the generated
`source_commit` differs from current `HEAD`.

## Comparison Selected-Row Tightening

Focused tests now cover selected comparison behavior directly:

- complete six-row selected comparison set passes;
- missing selected row fails with `comparison_selected_rows` and
  `row_set_mismatch`;
- unexpected selected row fails with `unexpected=<row_id>`;
- duplicate row ID fails through the global duplicate normalized row guard;
- stale selected row fails with recorded/current commit remediation;
- failed selected row fails through generated comparison failure and selected
  status diagnostics;
- deferred selected row fails as non-pass evidence and uses selected-row
  wording.

The comparison skip/defer diagnostic now says:

```text
freshness: defer: comparison_selected_rows: skip_or_defer_not_proof: ...
```

This keeps selected proof rows distinct from optional NumPy/SciPy dependency
defers, which remain contextual comparison dependency evidence rather than
selected pass evidence.

## Boundaries Preserved

- Missing selected oracle/comparison artifacts still fail required freshness.
- Stale selected oracle/comparison rows still fail required freshness.
- Failed selected oracle/comparison rows still fail required freshness.
- Oracle row-count, solver-family, and fixture-key checks remain hard errors.
- Comparison row-set and selected status checks remain hard errors.
- Source-controlled rows remain advisory.
- Unpromoted local-only families keep advisory/local semantics.
- Workflow commands and artifact uploads are unchanged.

## Validation

Passed locally:

```sh
python3 tests/test_normalize_report_index.py
make report-index-oracle-freshness
make report-index-comparison-freshness
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
make lint
git diff --check -- scripts/normalize_report_index.py tests/test_normalize_report_index.py .github/workflows/ci.yml docs/planning/EPIC_13/SPRINT_159
```

Observed selected freshness behavior after the change:

- oracle selected rows report `fresh` and no longer emit strict unchecked
  warnings on a passing required gate;
- comparison selected rows report `fresh` and no longer emit strict unchecked
  warnings on a passing required gate.

## Completion Check

- Promoted rows fail clearly when stale, missing, invalid, or incomplete.
- Non-promoted rows retain documented advisory/local behavior.
- Focused semantics validation passes locally.
- Hosted workflow surface remains unchanged from Day 8.
