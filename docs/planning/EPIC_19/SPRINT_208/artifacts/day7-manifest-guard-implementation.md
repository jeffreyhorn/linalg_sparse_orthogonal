# Sprint 208 Day 7 Manifest Guard Implementation

## Scope

Day 7 implemented the Day 6 re-deferral design for the selected Windows
Cholesky freshness manifest surface. The source-controlled selected target
manifest remains Linux/macOS-only for positive selected metadata; Day 7 changed
the manifest contract tests so this state cannot drift silently and so a
future Windows promotion must update claim fields coherently.

## Implementation

Changed file:

- `tests/test_selected_report_targets_manifest.py`

Implemented guard coverage:

- freezes the current `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` row identity,
  target key, family/subfamily, artifact pattern, generator command, support
  tier, claim scope, required files, expected row IDs, workflow files,
  workflow jobs, workflow artifacts, workflow platforms, and non-claims;
- rejects accidental Windows artifact metadata on the current re-deferred
  Cholesky row;
- keeps the generic no-Windows selected-platform deferral guard in place;
- extends the future Windows Cholesky allowlist so adding `windows` requires
  a non-`local_only` support tier, a claim scope that names Windows, and
  removal of stale `no Windows report freshness` / `no hosted CI proof`
  re-deferral wording;
- preserves existing future-promotion checks for target ID, workflow file,
  workflow job, workflow artifact, row count, required files, and artifact
  reuse.

## Manifest Decision

`tests/corpus/manifests/selected_report_targets.tsv` was intentionally not
changed. The selected Cholesky row still records:

- `support_tier=local_only`;
- Linux/macOS workflow metadata only;
- `workflow_platforms=linux;macos`;
- `no Windows report freshness` in `non_claims`.

This matches the Sprint 208 Day 5 decision: the latest hosted Windows evidence
is valid input evidence, but source-controlled selected Windows freshness
promotion remains re-deferred until generated support tier, claim scope,
non-claims, and documentation all promote the claim together.

## Validation

Commands run:

```sh
python3 tests/test_selected_report_targets_manifest.py
python3 -m py_compile tests/test_selected_report_targets_manifest.py
python3 scripts/validate_corpus_schema.py
```

Results:

- `test-selected-report-targets-manifest: ok`
- Python compilation completed without diagnostics.
- `validate-corpus-schema: /Users/jeff/experiments/linalg_sparse_orthogonal/tests/corpus ok`

## Completion Notes

Day 7 closes the manifest-side implementation for Item 208.3. Remaining Sprint
208 work should continue with workflow/PowerShell guard alignment and
normalizer regression review, not with manifest promotion.
