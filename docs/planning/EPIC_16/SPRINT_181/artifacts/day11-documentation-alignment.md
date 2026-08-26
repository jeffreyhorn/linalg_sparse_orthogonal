# Sprint 181 Day 11: Documentation Alignment

## Purpose

Day 11 aligns maintainer and report-index documentation with the selected
report target manifest. The goal is to make
`tests/corpus/manifests/selected_report_targets.tsv` the documented authority
for selected oracle, comparison, and benchmark target metadata while
preserving existing support and non-claim boundaries.

## Documentation Updates

Updated `tests/corpus/schemas/report_index_fields.md` to document both report
manifests:

- `report_families.tsv` remains the broad report-family authority for row
  meanings, freshness-policy vocabulary, default support tiers, family-level
  claims, and family-level non-claims.
- `selected_report_targets.tsv` owns selected target identity, exact selected
  commands, expected rows, expected row IDs, artifact patterns, required files,
  workflow scope, upload artifact names, support tiers, freshness policies,
  claim scopes, non-claims, owners, and introduction provenance.

Updated `docs/maintainer_guide.md` so selected oracle, comparison, and
canonical benchmark freshness sections name the selected-target manifest as
the target-list authority. The guide now tells maintainers to update
`SRT-ORACLE-QR-PSVD-LOCAL`, `SRT-COMP-*`, or
`SRT-BENCH-REFACTOR-CSC-NOS4` instead of copying selected target lists into
the guide.

Updated `README.md` so public maintainer-facing command guidance says the
selected target list, row counts, required artifacts, workflow upload names,
support tiers, freshness policies, claim scopes, and non-claims live in
`selected_report_targets.tsv`.

Updated `benchmarks/README.md` so selected benchmark and selected comparison
handoff wording points at the selected-target manifest instead of treating
duplicated paths and target names as authoritative.

## Reduced Duplication

Reduced duplicated selected target facts in docs:

| Former doc-owned detail | New authority |
| --- | --- |
| Selected oracle expected total and fixture-key set | `SRT-ORACLE-QR-PSVD-LOCAL.expected_rows` and `expected_row_ids` |
| Selected comparison target keys | `SRT-COMP-* target_key` |
| Selected comparison expected row counts and row IDs | `SRT-COMP-* expected_rows` and `expected_row_ids` |
| Selected comparison artifact paths and required files | `SRT-COMP-* artifact_pattern` and `required_files` |
| Selected workflow upload artifact names | `workflow_artifact` with `workflow_platforms` |
| Selected benchmark artifact/support/freshness metadata | `SRT-BENCH-REFACTOR-CSC-NOS4` |

## Remaining Documentation Exceptions

Some details remain duplicated intentionally:

| Detail | Reason |
| --- | --- |
| Make target names | They are user entry points and should remain visible. |
| High-level non-claim wording | Public and maintainer docs need readable boundaries even though the manifest owns row-level non-claims. |
| Selected oracle output path examples | They help maintainers inspect generated local artifacts; the manifest still owns the selected target contract. |
| Benchmark workload and methodology hints | The current manifest schema does not yet model typed workload, matrix-size, repeat, warmup, variance, baseline, or threshold fields. |
| Comparison generated row-name summaries | They explain how to read generated `study.tsv` rows, not which targets are selected. |

## Boundary Preservation

Day 11 keeps the existing claim boundaries:

- local generated rows do not become release proof;
- hosted Linux/macOS lanes promote only their selected manifest-owned lanes;
- Windows report freshness remains a non-claim;
- package-manager support and shared-library ABI support remain non-claims;
- selected benchmark freshness remains threshold-free metadata freshness, not
  timing superiority;
- selected comparison rows remain fixture-local and do not prove broad
  external-library parity or state-of-the-art status.

## Validation

Validation run:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Maintainers know where selected target authority lives. | Complete | README, maintainer guide, benchmark docs, and report-index schema docs now name `selected_report_targets.tsv`. |
| Docs and guard behavior describe the same manifest contract. | Complete | Docs describe the same fields consumed by normalizer, benchmark, schema, and workflow guard tests. |
| Public support wording does not widen report, platform, package, or performance claims. | Complete | Updated docs preserve local-only, hosted-selected, Windows, package/ABI, performance, and external-parity non-claims. |
