# Sprint 166 Day 8: Public Claim Audit Part 1

## Purpose

Day 8 audits public state-of-the-art, external-library parity, portable
performance, hosted-publication, generated-report, and release-proof wording.
It follows the Day 7 hosted CI reconciliation, which corrected the Linux
hosted comparison lane so selected comparison artifacts now cover QR min-norm,
QR compatible least-squares, and partial-SVD diag6 k2 after hosted CI passes.

## Audited Surfaces

| Surface | Role |
| --- | --- |
| `README.md` | Top-level public claim and command surface. |
| `docs/tutorial.md` | User workflow and benchmark/report handoff surface. |
| `docs/cookbook.md` | Practical workflow and generated-report interpretation surface. |
| `docs/solver_selection.md` | Solver-choice and bounded evidence surface. |
| `docs/algorithm.md` | Algorithm explanation and historical measurement links. |
| `docs/algorithm_history.md` | Historical benchmark/report interpretation surface. |
| `docs/maintainer_guide.md` | Maintainer proof-owner and report-index interpretation surface. |
| `benchmarks/README.md` | Benchmark command, measurement, and report-index handoff surface. |
| `tests/corpus/README.md` | Corpus and generated-report evidence boundary surface. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index schema and freshness semantics surface. |
| `.github/workflows/*.yml` | Hosted reviewed/supplemental evidence wording. |

## Scan Terms

The audit scanned for wording around:

- state-of-the-art status;
- broad external-library parity;
- LAPACK, SuiteSparse, Eigen, SciPy, and NumPy;
- portable performance, performance superiority, and backend superiority;
- hosted publication or hosted proof;
- release proof or release-quality wording;
- generated reports, local-only rows, oracle freshness, and comparison
  freshness.

## Classification Summary

| Wording class | Classification | Result |
| --- | --- | --- |
| State-of-the-art wording | Explicit non-claim | Existing public docs use state-of-the-art language to reject broad claims. |
| External-library parity wording | Explicit non-claim or fixture-scoped reference | References to LAPACK, SuiteSparse, Eigen, SciPy, and NumPy are bounded as non-parity, optional dependency context, dense helper context, or named fixture evidence. |
| Portable performance wording | Explicit non-claim | Benchmark docs, README, tutorial, cookbook, algorithm history, and maintainer guide keep benchmark rows local and methodology-bound. |
| Backend superiority wording | Explicit non-claim | Performance/sentinel wording remains local, threshold-scoped, or descriptive. |
| Hosted report wording | Cleanup required and applied | Several selected-comparison references still described rows as local-only after Day 7 promoted the selected comparison lane to reviewed Linux hosted evidence when CI runs. |
| Release proof wording | Explicit non-claim | Normalized report indexes, generated rows, benchmark reports, coverage, dead-code, and package metadata are not release proof by themselves. |

## Documentation Cleanup Applied

| File | Cleanup |
| --- | --- |
| `README.md` | Clarified that the reviewed Linux hosted report-freshness lane runs the selected oracle gate and the selected comparison gate only, with comparison scope limited to QR min-norm, QR compatible least-squares, and partial-SVD diag6 k2 generated rows/artifacts. |
| `README.md` | Replaced "narrow local comparison freshness gate" with selected comparison freshness wording and added that the same gate is mirrored by reviewed Linux hosted CI for selected comparison artifacts only. |
| `docs/solver_selection.md` | Replaced the stale absolute `local_only` selected partial-SVD comparison wording with local-by-default plus reviewed-Linux-hosted-only-when-uploaded wording. |
| `docs/maintainer_guide.md` | Updated selected comparison freshness guidance so generated comparison outputs are local by default, while the reviewed Linux hosted lane promotes only the selected gate and uploaded selected artifacts. |
| `docs/maintainer_guide.md` | Adjusted solver evidence-table wording from selected local generated comparisons to selected generated comparisons and changed the non-claim boundary to reject broad hosted-CI claims. |
| `tests/corpus/README.md` | Updated selected comparison freshness semantics from local-only rows to local-by-default plus reviewed-hosted-only-after-CI wording. |
| `tests/corpus/schemas/report_index_fields.md` | Updated selected comparison schema wording from generated-local rows to generated rows that are local by default and reviewed hosted only through the selected uploaded artifact lane. |

## Replacement Wording Pattern

Use this wording pattern for generated selected comparison evidence:

> Selected comparison rows are local generated evidence by default. They become
> reviewed Linux hosted evidence only when the hosted report-freshness lane
> runs the selected comparison gate and uploads the selected artifacts.

Pair that with these boundaries:

- selected artifacts only;
- named fixtures only;
- no broad QR, SVD, or partial-SVD correctness;
- no external-library parity;
- no portable performance or backend superiority;
- no platform/package/ABI proof;
- no release proof;
- no state-of-the-art status.

## Stale Wording Check

The current public docs and workflows no longer contain these stale current
surface phrases:

- `selected local generated comparisons`
- `selected local generated comparison`
- `It remains local_only evidence`
- `QR minimum-norm comparison rows only`
- `sprint159-comparison-qr-minnorm`

Historical sprint artifacts still contain earlier-sprint QR-minnorm-only
wording where it describes the state of those prior sprints. Those historical
records were not rewritten.

## Non-Claims Preserved

Day 8 did not add or imply:

- broad state-of-the-art sparse linear algebra status;
- broad LAPACK, SuiteSparse, Eigen, SciPy, NumPy, or ecosystem parity;
- portable performance superiority;
- backend superiority;
- broad generated-report freshness;
- release proof;
- package-manager support;
- shared-library support;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad platform parity.

## Validation

| Check | Result | Notes |
| --- | --- | --- |
| Focused claim scan | Pass | Hits are explicit non-claims, bounded fixture references, or corrected selected-comparison hosted evidence wording. |
| Stale selected-comparison wording scan | Pass | No stale QR-minnorm-only or local-only selected-comparison wording remains in current public docs/workflows. |
| `git diff --check` | Pass | No whitespace errors reported after artifact creation. |

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public performance and external-parity claims are evidence-bounded. | Complete | Scan results show non-claim or fixture/methodology-bound wording; benchmark docs explicitly reject portable performance claims. |
| Generated-report wording does not exceed hosted proof. | Complete | Selected comparison wording now distinguishes local generated rows from reviewed Linux hosted evidence after CI upload. |
| Stale claim wording is fixed or explicitly deferred with owner. | Complete | Stale selected-comparison local-only wording was fixed; historical sprint artifacts remain historical records. |
