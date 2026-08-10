# Sprint 149 Day 14: Closeout Handoff

## Purpose

Close Sprint 149 by publishing the final Windows CMake install/downstream
validation outcome, tying local evidence to the scoped support claim, recording
hosted-Windows proof as PR-time evidence, and handing Sprint 150 a clean QR
corpus starting boundary.

## Final Outcome

Sprint 149 promotes the Windows install/downstream package lane from
confidence-only wording to a reviewed, scoped CMake install/downstream
validation lane for the maintained static-first package surface.

The closed local claim is:

> Windows has reviewed CMake install/downstream validation for the maintained
> static-first package surface, subject to hosted MSVC proof in PR CI.

This claim is intentionally narrow. It covers the Windows CMake
install/downstream lane only, not broad package or platform parity.

## Support Boundary After Sprint 149

The reviewed Windows install/downstream support statement is:

- hosted Windows MSVC 2022 via CMake;
- CMake configure, build, and install;
- installed static `sparse_lu_ortho.lib`;
- installed public headers, including `sparse_version.h`;
- installed CMake package metadata;
- installed `sparse.pc` metadata as text/package metadata;
- absence of installed DLLs;
- absence of shared/module imported CMake metadata;
- positive `STATIC IMPORTED` CMake target metadata;
- install-prefix include metadata;
- installed `.lib` imported-location metadata;
- source/build path leak rejection;
- maintained installed CMake example;
- generated basic installed CMake consumer;
- exact-version installed CMake consumer;
- mismatch-version rejection.

The following remain explicit non-claims:

- Windows Makefile install or uninstall parity;
- Windows `pkg-config` execution parity;
- Windows `pkg-config` downstream compile/link/run parity;
- package-manager installation or resolver behavior;
- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad Windows parity beyond hosted MSVC CMake lanes.

## Artifact Package

| Artifact | Purpose | Status |
| --- | --- | --- |
| `PLAN.md` | 14-day Sprint 149 execution plan | Complete |
| `WORKING_NOTES.md` | Day-by-day implementation and validation notes | Complete |
| `artifacts/day1-install-intake.md` | Install-lane intake and evidence inventory | Complete |
| `artifacts/day2-windows-package-audit.md` | Existing Windows package-lane audit | Complete |
| `artifacts/day3-promotion-criteria.md` | Promotion/defer/reject criteria | Complete |
| `artifacts/day4-product-decision.md` | Conditional promotion product decision | Complete |
| `artifacts/day5-workflow-design.md` | Workflow implementation design | Complete |
| `artifacts/day6-workflow-implementation.md` | Workflow implementation summary | Complete |
| `artifacts/day7-metadata-check-design.md` | CMake/pkg-config metadata check design | Complete |
| `artifacts/day8-metadata-implementation.md` | Metadata implementation summary | Complete |
| `artifacts/day9-consumer-proof-design.md` | Downstream consumer proof design | Complete |
| `artifacts/day10-consumer-implementation.md` | Downstream consumer implementation summary | Complete |
| `artifacts/day11-docs-alignment.md` | README/INSTALL/maintainer/report alignment | Complete |
| `artifacts/day12-local-validation.md` | Local validation and syntax review | Complete |
| `artifacts/day13-integrated-evidence-review.md` | Integrated evidence review and hosted-CI status | Complete |
| `artifacts/day14-closeout-handoff.md` | Closeout, residuals, and Sprint 150 handoff | Complete |

## Validation Summary

Day 12 completed the local package/install validation pass:

- `git diff --check`: passed;
- targeted trailing-whitespace scan: passed;
- workflow YAML parse: passed;
- `python3 scripts/validate_corpus_schema.py`: passed;
- `python3 scripts/normalize_report_index.py --family ci --check`: passed;
- `python3 scripts/normalize_report_index.py --family package --check`:
  passed;
- `bash tests/test_cmake_install.sh`: passed with 26 checks, 0 failed,
  0 skipped;
- `bash tests/test_install.sh`: passed with 23 checks, 0 failed;
- `bash scripts/static_package_deferral_check.sh`: passed.

Day 13 completed the integrated evidence review:

- final workflow job and step names match the Day 4 decision;
- public docs and maintainer guidance use reviewed Windows CMake
  install/downstream wording;
- unsupported Windows package/platform surfaces remain explicit non-claims;
- no PR or branch run existed yet for `sprint-149`, so hosted Windows evidence
  remains PR-time pending.

Day 14 closeout hygiene:

- artifact links and day coverage reviewed;
- Sprint 150 project-plan scope reviewed;
- final whitespace, YAML, stale-reference, and `git diff --check` checks are
  recorded in `WORKING_NOTES.md`.

No repository `.c` or `.h` files changed during Sprint 149, so the full
`make format && make lint && make test` gate is not required by the
review-comment rule. The package/install gates above are the affected local
validation surface.

## Hosted Windows Evidence

No pull request or hosted branch run exists for `sprint-149` at Day 14
closeout. The PR must provide the final hosted evidence for:

- `Windows enforced reviewed CMake consumer subset (MSVC)`;
- `Windows reviewed CMake install/downstream validation path`.

For the install/downstream job, the required hosted proof is:

- CMake install configure/build/install passes;
- installed static `.lib` exists;
- no installed DLLs exist;
- installed header count remains intentional;
- installed CMake package files exist;
- `SparseTargets.cmake` declares `Sparse::sparse_lu_ortho` as
  `STATIC IMPORTED`;
- exported include metadata uses the install prefix;
- imported Release target points to the installed `.lib`;
- installed CMake package metadata does not leak source/build paths;
- installed CMake package metadata has no shared/module imported target;
- installed `sparse.pc` metadata has the expected name, version, static archive
  description, `Cflags`, and `Libs` rows;
- installed `sparse.pc` metadata has no unsupported package or ABI wording;
- generated basic installed CMake consumer configures, builds, and runs;
- maintained installed CMake example configures, builds, and runs;
- exact-version installed CMake consumer configures, builds, and runs;
- lower same-major mismatch version is rejected.

If hosted Windows fails, treat the failure as a Sprint 149 PR fix. Either fix
the failing criterion or roll the public wording back to pending before merge.

## Residual Deferred Debt

Still explicitly unresolved at Sprint 149 close:

- hosted Windows proof for the reviewed CMake install/downstream lane until PR
  CI runs;
- Windows Makefile install/uninstall parity;
- Windows `pkg-config` execution parity;
- Windows `pkg-config` downstream compile/link/run parity;
- package-manager installation or resolver behavior;
- shared-library packaging;
- dynamic ABI compatibility;
- runtime-loader behavior;
- broad Windows platform parity beyond hosted MSVC CMake lanes.

Still consciously constrained rather than silently solved:

- no broad Windows ecosystem parity claim;
- no package-manager availability claim;
- no shared-library or dynamic ABI support claim;
- no local claim from absent hosted CI logs;
- no unqualified package parity claim from CMake-only proof.

## Sprint 150 QR Corpus Handoff

Sprint 150 is planned as `QR Maintained Corpus Family Expansion`. Its goal is
to close a broader but still bounded QR corpus family beyond the Sprint 139
fixture-local closure.

Sprint 150 should start from this Sprint 149 boundary:

| Starting item | Required posture |
| --- | --- |
| Windows package lane | Treat reviewed CMake install/downstream validation as PR-time pending until hosted Windows CI proves it. |
| Package non-claims | Do not depend on Windows Makefile, Windows `pkg-config`, package-manager, shared-library, dynamic ABI, runtime-loader, or broad Windows parity. |
| Public headers | If QR work adds public headers, update the fixed Windows installed-header count intentionally. |
| Report rows | Keep source-controlled report rows distinct from hosted CI logs; do not infer freshness from manifest rows alone. |
| QR fixture selection | Select two or three QR families for complete closure: rank-deficient rectangular, underdetermined minimum-norm, and reorder/COLAMD paths. |
| QR oracle semantics | Use residual, rank, nullspace, minimum-norm, and subspace-safe comparisons; avoid raw-basis identity claims. |
| QR proof ownership | Prefer focused QR corpus proof-owner tests rather than expanding the largest monolithic QR test file. |
| Validation | Run corpus schema, focused QR tests, oracle/report checks, and full C gates if QR code/header files change. |

Suggested Sprint 150 opening checks:

1. Inspect merged Sprint 149 hosted Windows CI results if the PR has already
   run.
2. Confirm the Windows expected CTest count remains stable after merge.
3. Select the QR fixture families and claim scopes before generating report
   rows.
4. Define QR oracle semantics before writing tests, especially for rank,
   nullspace, and minimum-norm comparisons.
5. Keep QR corpus claims independent from Windows package-lane residuals.

## Retrospective Input Notes

- What worked: the Day 4 conditional-promotion decision kept the implementation
  tied to explicit evidence requirements instead of broad Windows package
  parity.
- What worked: metadata checks and consumer checks were separated but retained
  in one reviewed CMake install/downstream lane, reducing ambiguity for users.
- What worked: docs and report rows preserve package and platform non-claims in
  the same places that name the reviewed Windows CMake validation.
- Watch item: hosted Windows proof is still pending because the branch has no
  PR or branch run yet.
- Watch item: fixed installed-header counts remain intentionally brittle and
  must be updated when public headers change.
- Follow-through: PR review should verify that hosted Windows logs, workflow
  wording, README, INSTALL, maintainer guide, and report manifest agree on the
  same scoped claim.

## Completion Criteria Status

| Completion Criteria | Status | Evidence |
| --- | --- | --- |
| Sprint 149 product decision and evidence are ready for retrospective. | Complete | Day 4 decision, Day 13 evidence review, and this closeout artifact align. |
| Residuals are explicit and assigned to later sprint candidates. | Complete | Residual table and Sprint 150 handoff separate package residuals from QR work. |
| Branch is clean except for intentional Sprint 149 changes. | Complete | Final status is recorded in `WORKING_NOTES.md`; changes are Sprint 149 workflow, docs, report row, and artifacts. |
