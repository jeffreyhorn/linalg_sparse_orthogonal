# Sprint 169 Day 9: Documentation Indexing Design

## Purpose

Day 9 designs the documentation path for selected performance evidence. The
goal is to make the selected `bench_refactor_csc` evidence discoverable from
the README, benchmark documentation, maintainer guide, and report-index
workflow without treating generated `build/` artifacts as checked-in evidence
or broad performance claims.

## Current Surfaces Reviewed

| Surface | Current role | Day 9 finding |
| --- | --- | --- |
| `README.md` first-use boundary | Explains when examples, benchmarks, and tests own different evidence types. | Good top-level location, but the sentinel text still mentions only S5 plus S2/S3 and should include S6. |
| `README.md` build command list | Lists `make bench-canonical-report-freshness`, `make bench-canonical-report`, and `make performance-sentinels`. | Good command discoverability, but selected evidence path should link from these commands to benchmark docs. |
| `README.md` normalized report-index note | Explains normalized index as a navigation/freshness aid. | Correct non-claim framing; should mention selected performance evidence remains rooted in the focused freshness checker. |
| `benchmarks/README.md` canonical report section | Owns selected-row methodology, support tier, claim boundary, warmup, variance, baseline, and threshold semantics. | Best detailed source for selected performance evidence interpretation. |
| `benchmarks/README.md` sentinel section | Owns S5/S6 hard-gate and S2/S3 threshold-free context interpretation. | Now includes S6 after Day 8 and should remain the detailed runtime-sentinel reference. |
| `benchmarks/README.md` report-index handoff | Maps report targets to generated indexes and interpretation rules. | Best cross-report documentation hub for selected benchmark and sentinel evidence. |
| `docs/maintainer_guide.md` performance section | Owns authoritative maintainer rules, stop conditions, non-claims, and generated-output handling. | Correct location for stale-report and source-control policy. |
| `scripts/normalize_report_index.py` | Generates a cross-family local navigation index. | Useful downstream view, but should remain secondary to focused selected freshness checks. |

## Selected Evidence Path

Day 10 should implement this path:

1. `README.md`
   - Keep the top-level summary short.
   - Point users to `make bench-canonical-report-freshness` for the selected
     performance freshness check.
   - Point users to `make performance-sentinels` for local S5/S6 hard gates
     and S2/S3 threshold-free context.
   - Link to `benchmarks/README.md#report-index-handoff` for detailed report
     interpretation.
2. `benchmarks/README.md#report-index-handoff`
   - Remain the main documentation hub for generated performance and sentinel
     indexes.
   - Explain how to find the selected canonical row in
     `build/bench-reports/canonical/index.tsv`.
   - Explain how to find the selected local S6 sentinel row in
     `build/bench-reports/sentinels/sentinels.tsv`.
   - Keep generated output under ignored `build/` paths.
3. `docs/maintainer_guide.md`
   - Own stale-report handling and source-control policy.
   - State that the focused selected freshness checker is authoritative for
     selected performance freshness, while the normalized report index is a
     navigation aid.
   - Preserve the S6 non-claim boundary.

## README Evidence-Index Design

The README should expose a compact table or short bullet list with these
entries:

| Need | Command | Detailed reference | Read as |
| --- | --- | --- | --- |
| Selected performance freshness | `make bench-canonical-report-freshness` | `benchmarks/README.md#canonical-benchmark-reports` and `benchmarks/README.md#report-index-handoff` | Selected `bench_refactor_csc` row freshness and methodology validation only. |
| Local selected regression smoke gate | `make performance-sentinels` | `benchmarks/README.md#report-index-handoff` | S6 local smoke ceiling, not portable performance proof. |
| Cross-report navigation | `python3 scripts/normalize_report_index.py --check-freshness` | `docs/maintainer_guide.md#normalized-report-index-workflow` | Generated/local navigation and diagnostics, not replacement validation. |

The top-level wording should avoid expanding into every column name. The
benchmark docs already own detailed schema explanations.

## Benchmark-Doc Link Design

`benchmarks/README.md` should remain the detailed selected-performance landing
page because it already explains:

- selected artifact identity;
- selected command and fixture;
- support tier and claim boundary;
- repeat semantics;
- warmup and variance policy;
- `baseline=n/a` and `threshold=n/a` for canonical publication rows;
- S5/S6 hard-gate versus S2/S3 threshold-free sentinel boundaries;
- generated artifact locations.

Day 10 should add one short "Selected performance evidence path" subsection
near the canonical report and report-index sections, or tighten the existing
report-index handoff so a reader can move from the README command list to the
exact generated row identity.

## Report-Index Ownership Decision

The focused checker remains authoritative for selected performance freshness:

```sh
make bench-canonical-report-freshness
```

The normalized report index is secondary:

```sh
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel \
  --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel \
  --check-freshness
```

Rationale:

- the selected freshness checker owns exact selected-row validation and
  manifest agreement;
- the normalized report index combines report families and should not become
  a hidden replacement for focused validation;
- generated normalized rows should help maintainers find evidence, not create
  new claims.

## Stale-Report And Generated-Output Handling

Generated outputs remain local and ignored:

- `build/bench-reports/canonical/index.tsv`;
- `build/bench-reports/canonical/manifest.txt`;
- `build/bench-reports/sentinels/sentinels.tsv`;
- `build/bench-reports/sentinels/manifest.txt`;
- `build/report-index/normalized-index.tsv`.

Day 10 documentation should say:

- regenerate reports before interpreting them;
- use manifest branch, commit, timestamp, platform, compiler, build mode, and
  thread settings before citing a row;
- do not hand-edit generated report files;
- do not commit generated report files unless a later sprint creates a stable
  checked-in sample with a freshness contract;
- treat stale diagnostics as a prompt to regenerate or rerun the focused
  checker, not as performance proof.

## Claim-Safe Wording

Use:

- "selected performance freshness";
- "selected `bench_refactor_csc` row";
- "threshold-free publication row";
- "local selected-lane smoke ceiling";
- "generated local report";
- "navigation and freshness aid".

Avoid:

- "performance guarantee";
- "portable speed";
- "hosted benchmark result" for local artifacts;
- "state-of-the-art performance";
- "external-library parity";
- "release benchmark proof";
- "regression guarantee" for threshold-free canonical rows.

## Day 10 Implementation Plan

1. Update `README.md` first-use boundary so `make performance-sentinels`
   mentions S6 and points to the benchmark report-index handoff.
2. Update the README build command comment for `make performance-sentinels`
   to include S5/S6 hard gates and S2/S3 threshold-free context.
3. Add or tighten the README selected-performance paragraph so the selected
   evidence path is discoverable without reading the maintainer guide first.
4. Add a concise selected-performance evidence path note in
   `benchmarks/README.md` if the existing report-index handoff remains too
   implicit.
5. Add a maintainer-guide note that `make bench-canonical-report-freshness`
   is authoritative for selected performance freshness and normalized report
   index output is secondary navigation.
6. Run `git diff --check` and any focused Markdown/link scans needed for the
   touched docs.

## Day 9 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected performance evidence has a discoverable documentation path. | Complete | The design defines README -> benchmark report-index handoff -> maintainer guide ownership. |
| Generated output is not accidentally treated as checked-in evidence. | Complete | Generated artifacts remain under ignored `build/` paths with explicit stale-report handling. |
| Documentation wording remains claim-safe. | Complete | The design separates threshold-free selected publication from local S6 smoke-gate wording and lists terms to use/avoid. |
