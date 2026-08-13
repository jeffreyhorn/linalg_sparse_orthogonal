# Sprint 155 Day 14 Closeout And Sprint 156 Handoff

## Purpose

Day 14 closes Sprint 155 by summarizing delivered documentation coherence work,
recording residuals, and handing Sprint 156 a concrete final-validation and
claim-recalibration queue.

## Project-Plan Traceability

| Sprint 155 project-plan item | Completed work | Evidence |
| --- | --- | --- |
| Tutorial Audit | Audited `docs/tutorial.md` against README, examples, cookbook, solver-selection, install/package, support-tier, maintainer, public-header, and API-reference surfaces. | `day1-documentation-baseline.md`, `day2-tutorial-audit.md` |
| Tutorial Alignment | Reworked tutorial flow around build, first solve, input formats, solver choice, diagnostics, install, advanced controls, benchmarks/reports, and API reference handoff. | `docs/tutorial.md`, `day3-tutorial-flow-design.md`, `day4-tutorial-core-rewrite.md`, `day5-tutorial-alignment-summary.md` |
| Header Cleanup Selection | Selected a high-impact public-header batch and documented deferred headers. | `day6-header-cleanup-selection.md` |
| Header Cleanup Batch | Cleaned selected public-header comments without declaration, install-surface, package, or ABI drift. | `include/sparse_ldlt.h`, `include/sparse_ic.h`, `include/sparse_eigs.h`, `include/sparse_analysis.h`, `day7-header-cleanup-contract.md`, `day8-header-cleanup-summary.md`, `day9-header-cleanup-summary.md` |
| API Reference Plan | Inventoried Doxygen/API surfaces, found generated HTML coverage gaps, and defined publication/freshness rules. | `day10-api-reference-publication-plan.md` |
| Declaration Preservation | Captured before/after/current declaration scans and normalized diffs for edited public headers. | `day8-header-declarations-*.txt`, `day9-header-declarations-*.txt`, `day12-header-declarations-current.txt`, `day12-header-declarations-normalized-diff.txt` |
| Validation And Closeout | Ran integrated validation, full C gate, link checks, stale-phrase checks, and claim scans. | `day13-integrated-validation.md`, this closeout |

## Delivered User-Facing Improvements

- `docs/tutorial.md` now follows the current adoption ladder instead of mixing
  first-use guidance with maintainer policy.
- `README.md`, `docs/tutorial.md`, and `docs/cookbook.md` now point to
  `docs/api_reference.md` for exact declarations and ownership contracts.
- `docs/api_reference.md` gives users a compact public-header index and
  explains the generated Doxygen HTML freshness boundary.
- `docs/maintainer_guide.md` now owns public-header cleanup rules plus
  generated API-reference publication/freshness policy.

## Public Header Cleanup

Edited selected headers:

- `include/sparse_ldlt.h`
- `include/sparse_ic.h`
- `include/sparse_eigs.h`
- `include/sparse_analysis.h`

The cleanup shortened call-site comments, clarified ownership/error/default
contracts, removed long sprint-history style prose, and preserved:

- public declarations;
- signatures;
- typedefs;
- enum values;
- struct fields;
- macros;
- include guards;
- installed header names;
- exported names;
- static-first package and ABI boundaries.

## Validation Summary

Day 13 ran:

```sh
make format && make lint && make test
```

Result: passed. The final output ended with `All tests passed.`

Additional checks:

- `git diff --check` passed.
- API-reference link-target checks passed.
- stale phrase scan for `API reference surface` and `generated API reference`
  returned no matches.
- unsupported-claim scan returned only explicit non-claim wording.
- Day 8, Day 9, and Day 12 normalized declaration diffs are all `0` bytes.

## Residual And Deferred Work Register

| Residual | Status | Reason | Promotion gate |
| --- | --- | --- | --- |
| Generated API HTML refresh | Deferred | `docs/api/html/` is partial for the current checked-in public header set, and refreshing it would create a large generated-output diff. | Run `make docs`, capture Doxygen version/warnings, check page coverage, and commit generated output in a dedicated reference-refresh change or alongside the header/comment edits that justify it. |
| Missing generated API pages | Deferred | Current generated HTML has `13` checked-in public-header pages for `18` checked-in public headers. Missing pages are analysis, eigs, IC, LDLT, and LU CSR. | Same generated API HTML refresh gate. |
| Generated installed `sparse_version.h` in Doxygen | Deferred | Current `Doxyfile` reads checked-in `include/*.h`; installed `sparse_version.h` is generated under the build include directory. | Decide whether a build-aware Doxygen input path is acceptable; otherwise keep version macro behavior owned by installed headers, `VERSION`, and install tests. |
| Remaining public-header cleanup outside selected batch | Deferred | Sprint 155 intentionally cleaned only selected high-impact headers and documented the rest as out of scope. | Re-run Day 6/Day 7 selection and declaration-preservation process for any new header batch. |
| Broad documentation claim recalibration | Deferred to Sprint 156 | Sprint 155 preserved non-claims locally but did not perform the final Epic 13-wide claim audit. | Sprint 156 Item 4 public claim/non-claim audit. |

## Earned Improvements Versus Non-Claims

Earned in Sprint 155:

- tutorial alignment with current adoption surfaces;
- selected public-header comment cleanup with declaration-preservation evidence;
- API reference entry point;
- generated API-reference freshness policy;
- maintainer guidance for future header/API-comment cleanup;
- full local quality-gate validation after public header edits.

Still not claimed:

- dynamic ABI compatibility;
- shared-library support;
- package-manager distribution;
- broad Windows Makefile or Windows `pkg-config` parity;
- broad platform parity;
- external-library parity;
- portable performance;
- broad generated-reference completeness;
- state-of-the-art sparse linear algebra status.

## Sprint 156 Handoff

Sprint 156 should start from this Sprint 155 state and execute the Epic 13
final closeout plan:

1. Inventory final Epic 13 evidence across platform, corpus, report, ABI,
   package, comparison, adoption, documentation, and validation artifacts.
2. Run the strongest feasible local validation baseline and any package,
   report, corpus, comparison, or generated-reference checks selected for the
   final closeout.
3. Reconcile Linux, macOS, Windows, supplemental, staged, reviewed, local-only,
   and deferred evidence after final CI runs.
4. Perform the public claim/non-claim audit across README, install docs,
   tutorial, cookbook, solver-selection, benchmark/report docs, API reference,
   maintainer guide, package metadata comments, and selected public headers.
5. Publish residual queue entries with owners, blockers, prerequisites, and
   promotion gates.
6. Write the Epic 13 retrospective with earned claims, non-claims, validation
   evidence, competitive assessment, and next-epic handoff.

## Closeout Decision

Sprint 155 is complete. The branch is ready for Sprint 155 retrospective/PR
packaging after Day 14, with generated API HTML refresh explicitly deferred to
a future reviewable action unless Sprint 156 chooses to promote it into final
closeout validation.
