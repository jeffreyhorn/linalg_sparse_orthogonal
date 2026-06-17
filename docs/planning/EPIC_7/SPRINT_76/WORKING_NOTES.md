# Sprint 76 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 76 benchmark-governance, profiling, and longitudinal-reporting baseline grounded in the live tree, the maintained reviewed validation contract, and the current canonical benchmark/reporting surfaces.

### Actions
- Re-read the Sprint 76 plan in `docs/planning/EPIC_7/SPRINT_76/PLAN.md` and the Sprint 76 section in `docs/planning/EPIC_7/PROJECT_PLAN.md`.
- Rechecked the maintained reviewed wrapper surface with `make -n quality-review-full`.
- Re-materialized the reviewed CMake parity tree with `make quality-review-cmake-compile`.
- Reconfirmed the reviewed CMake parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Re-read the strongest maintained benchmark-governance and reporting surfaces:
  - `README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `scripts/bench_canonical_report.sh`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
  - `benchmarks/bench_reorder.c`
  - `benchmarks/bench_amd_qg.c`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 76 touch surfaces.

### Findings
- Sprint 76 starts from the same strongest local reviewed baseline as Sprint 75:
  - `make quality-review-full`
- Reviewed CMake parity remains explicit and exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The strongest Sprint 76 pressure is now clearly narrowed to:
  - benchmark-governance re-audit
  - canonical reporting and longitudinal-comparison design
  - maintained benchmark workflow clarification
  - profiling and threshold-policy truthfulness
  - benchmark/proof-owner alignment
  - final validation and closeout
- The strongest maintained benchmark-governance and reporting surfaces are now explicit from the live tree:
  - `README.md` = `1045`
  - `benchmarks/README.md` = `377`
  - `docs/maintainer_guide.md` = `677`
  - `Makefile` = `897`
  - `scripts/bench_canonical_report.sh` = `56`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `423`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`
  - `benchmarks/bench_reorder.c` = `321`
  - `benchmarks/bench_amd_qg.c` = `332`
- The maintained Sprint 76 benchmark-governance fence is already clear and must be preserved:
  - `make bench-canonical-report` is the threshold-free canonical reporting surface
  - canonical maintained performance proof centers on:
    - `bench_refactor_csc`
    - `bench_chol_csc`
    - `bench_iterative_reuse`
    - `bench_eigs_reuse`
  - benchmark artifacts remain reporting and interpretation surfaces, not portable pass/fail timing gates
  - narrower thresholded or exploratory lanes such as `bench-fast`, `wall-check`, `bench_reorder`, and `bench_amd_qg` must not silently broaden into the canonical proof contract

### Validation
- Rechecked `make -n quality-review-full`.
- Rebuilt the reviewed CMake tree with `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Captured the live benchmark-governance hotspot map from direct reads plus targeted terminology scans.

### Day 1 Exit State
- Sprint 76 no longer starts from a generic “benchmark cleanup” prompt.
- The maintained benchmark/reporting owners, truthfulness fence, and strongest likely touch surfaces are fixed in writing.
- The branch is clean after the Day 1 baseline commit.

## Day 2 - Validation Baseline

### Goal
Reconfirm the Sprint 76 implementation-day validation contract and the live proof-surface split across reviewed benchmark binaries, workflow/report-generation entry points, representative examples, and install/package proof.

### Actions
- Re-read the Sprint 76 Day 2 plan expectations in `docs/planning/EPIC_7/SPRINT_76/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed benchmark/test/example binaries in `build/quality-review-cmake`.
- Rechecked the report-generation workflow entry point with `make -n bench-canonical-report`.
- Reconfirmed the maintained install/package proof scripts:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

### Findings
- Sprint 76 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 76 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial benchmark, workflow, or governance batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the key Sprint 76 benchmark-governance proof surfaces:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_svd`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_refactor_csc`
  - `./build/quality-review-cmake/bench_chol_csc`
  - `./build/quality-review-cmake/bench_iterative_reuse`
  - `./build/quality-review-cmake/bench_eigs_reuse`
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_amd_qg`
- The canonical report-generation workflow remains source and command owned rather than reviewed-binary owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters consumed by that script:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- Maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the reviewed benchmark/test/example binary set in `build/quality-review-cmake`.
- Rechecked the canonical report-generation command surface with `make -n bench-canonical-report`.
- Reconfirmed the maintained install/package proof scripts exist and remain callable.

### Day 2 Exit State
- Sprint 76 now has one explicit implementation-day validation contract.
- The live proof split across reviewed binaries, canonical-report workflow ownership, and install/package scripts is fixed in writing.
- The strongest likely Sprint 76 rerun set is explicit before governance and reporting design work begins.

## Day 3 - Governance Re-audit

### Goal
Re-rank the live benchmark-governance surface by actual reporting value, proof leverage, and maintenance clarity so Sprint 76 starts from the strongest contradiction center rather than from a generic “benchmark reporting overhaul” idea.

### Actions
- Re-read the Sprint 76 Day 3 plan expectations in `docs/planning/EPIC_7/SPRINT_76/PLAN.md`.
- Re-read the benchmark-local governance and schema surface in `benchmarks/README.md`.
- Re-read the compact front-door benchmark/reporting summary in `README.md`.
- Re-read the authoritative benchmark-governance policy section in `docs/maintainer_guide.md`.
- Re-read the current canonical report generator in `scripts/bench_canonical_report.sh`.
- Re-read the current `bench-canonical-report`, `bench-fast`, and `wall-check` command ownership in `Makefile`.
- Rechecked the strongest benchmark-governance/reporting terminology seams with targeted `rg`.

### Findings
- Sprint 76's broad benchmark-governance pressure is now reduced to one ranked contradiction map instead of one generic “benchmark modernization” bucket:
  - strongest first target:
    - canonical reporting workflow and longitudinal-comparison schema
  - strongest second target:
    - benchmark-local role and interpretation surface
  - strongest third target:
    - authoritative threshold and category policy surface
  - strongest support-surface contradiction:
    - compact front-door benchmark summary
  - strongest adjacent but not first-batch lane:
    - regression-sensitive runtime surfaces around `bench-fast`, `wall-check`, `bench_reorder`, and `bench_amd_qg`
- The strongest first contradiction center is not the canonical benchmark binaries themselves.
- It is the reporting/orchestration layer across:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
  - the canonical maintained emitters the script drives
- That lane ranks first because:
  - the maintained canonical benchmark surface is already explicit
  - the report script is intentionally threshold-free and cheap
  - but longitudinal comparison still depends on a very small manifest and manually interpreted CSV bundles
  - so the strongest remaining Sprint 76 leverage is schema, metadata, and workflow governance rather than new timing surfaces
- `benchmarks/README.md` is the strongest second target because it currently carries the densest user-facing ownership split across:
  - canonical maintained proof
  - regression-sensitive runtime lane
  - exploratory comparison lane
  - `bench-fast`
  - `wall-check`
  - `bench-canonical-report`
- `docs/maintainer_guide.md` is the strongest third target because it already owns the authoritative category and threshold policy, but still reads as policy-first support rather than the best first landing center by itself.
- `README.md` is a real support-surface contradiction center, but weaker than the benchmark-local and maintainer-policy surfaces because it already stays compact and intentionally does not own the full governance contract.
- The current regression-sensitive runtime lane remains important, but it is not the best first landing because the strongest immediate problem is role and report comparability clarity, not widening or tightening threshold gates.

### Validation
- Re-read `benchmarks/README.md`, `README.md`, `docs/maintainer_guide.md`, `scripts/bench_canonical_report.sh`, and the relevant `Makefile` sections.
- Rechecked the benchmark-governance/reporting terminology seams with targeted `rg`.
- Reconfirmed that the reviewed baseline and Day 2 truth-surface split remain unchanged by the Day 3 rerank.

### Day 3 Exit State
- Sprint 76 now has one explicit governance contradiction ranking instead of a generic benchmark-reporting backlog.
- The strongest first landing candidate is fixed to the canonical reporting workflow/schema lane.
- Day 4 can now freeze a real first governance boundary from the live benchmark/reporting contract.

## Day 4 - First Governance Boundary

### Goal
Freeze the first Sprint 76 reporting/governance fence so the next design pass starts from one bounded longitudinal-reporting lane rather than from a mixed docs, threshold, and exploratory benchmark backlog.

### Actions
- Re-read the Sprint 76 Day 3 rerank artifact.
- Re-read the Sprint 76 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`.
- Re-ranked the Day 3 contradiction centers against:
  - reporting leverage
  - compatibility risk
  - proof clarity
  - bounded Sprint 76 payoff
- Separated required first-batch surfaces from support-only and explicitly deferred surfaces.
- Fixed the first-batch non-goal fence in writing.

### Findings
- Sprint 76 now has one explicit first governance boundary:
  - required first landing:
    - `scripts/bench_canonical_report.sh`
    - `Makefile`
  - support only if the first landing forces it:
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
    - `README.md`
  - explicitly deferred:
    - canonical benchmark driver sources
    - reviewed proof-owner tests and examples
    - runtime-threshold surfaces:
      - `bench-fast`
      - `wall-check`
      - `bench_reorder`
      - `bench_amd_qg`
- The strongest Day 4 clarification is now fixed:
  - the first Sprint 76 lane is canonical report workflow and schema, not benchmark-driver churn first
  - benchmark-local interpretation remains the strongest support seam, not the first batch center
  - maintainer-policy wording remains authoritative support, not the first landing center
  - threshold-policy work is still real, but it is explicitly second-batch pressure rather than the first move
- The first-batch non-goal fence is explicit now:
  - no widening of the canonical maintained benchmark surface
  - no new timing-threshold gate disguised as longitudinal reporting
  - no broad benchmark-driver rewrite
  - no widened product/platform/backend claim detached from maintained evidence

### Validation
- Re-read the Day 3 rerank and the Sprint 76 project-plan section.
- Reconfirmed that the Day 4 fence stays inside the preserved Sprint 65, Sprint 70, and Sprint 75 truthfulness contract.
- Rechecked branch state before closing the boundary pass.

### Day 4 Exit State
- Sprint 76 now has one exact first reporting/governance landing boundary.
- The next design pass can stay inside the canonical report workflow/schema lane without drifting into threshold, docs, or exploratory benchmark sprawl.

## Day 5 - Reporting Design

### Goal
Define the bounded implementation contract for Sprint 76's first canonical reporting landing before any code or workflow edits begin.

### Actions
- Re-read the Day 4 governance boundary artifact.
- Re-read the current canonical report generator in `scripts/bench_canonical_report.sh`.
- Re-read the current `bench-canonical-report` command wiring in `Makefile`.
- Re-read the compact benchmark-local and maintainer-policy interpretation around the current report surface.
- Fixed the exact ownership split, preserved guarantees, and non-touch set for the first implementation batch.

### Findings
- Sprint 76 now has one explicit first implementation contract:
  - required implementation center:
    - `scripts/bench_canonical_report.sh`
    - `Makefile`
  - support only if the first batch truly forces it:
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
    - `README.md`
- The reporting ownership split is now fixed:
  - `scripts/bench_canonical_report.sh` should own:
    - report-directory layout
    - stable report-bundle metadata
    - manifest and index-style cross-run comparison metadata
    - exact canonical benchmark command capture
  - `Makefile` should own:
    - the public `make bench-canonical-report` entry point
    - the default report output location
    - any bounded override seam for report destination or label input
- The first batch should strengthen the canonical report surface without changing what is canonical:
  - keep the same four canonical maintained benchmark emitters
  - keep one CSV per emitter
  - preserve the threshold-free reading
  - add only cheap, truthful metadata that improves longitudinal comparison
- The safe first-batch metadata lane is now explicit:
  - generated timestamp
  - git commit or branch identity when available
  - exact command mapping
  - stable report-surface identity
  - explicit artifact inventory
  - optional bounded user-supplied comparison label
- The unsafe first-batch lane is explicit too:
  - no new pass/fail timing thresholds
  - no machine-specific performance verdicts
  - no widening into runtime or exploratory benchmark capture
  - no rewriting canonical benchmark row schemas inside the benchmark drivers

### Validation
- Re-read the Day 4 boundary and the current report command/script surfaces.
- Reconfirmed that the design stays inside the preserved Sprint 65, Sprint 70, and Sprint 75 truthfulness fence.
- Rechecked branch state before closing the design pass.

### Day 5 Exit State
- Sprint 76 now has one exact Day 6 implementation contract for canonical reporting.
- The next batch can improve cross-run and cross-branch artifact comparability without drifting into threshold policy or benchmark-driver churn.

## Day 6 - Canonical Reporting Batch

### Goal
Land the first bounded canonical reporting batch on the maintained report workflow without widening the canonical benchmark surface or introducing timing-threshold policy.

### Actions
- Updated `scripts/bench_canonical_report.sh` to emit bounded cross-run metadata in addition to the existing canonical CSV bundle.
- Updated `Makefile` to add the bounded `BENCH_CANONICAL_REPORT_LABEL` workflow override seam while preserving the same public command:
  - `make bench-canonical-report`
- Ran a smoke bundle with:
  - `make bench-canonical-report BENCH_CANONICAL_REPORT_DIR=build/bench-reports/canonical-day6-smoke BENCH_CANONICAL_REPORT_LABEL=day6-smoke`
- Re-read the generated:
  - `manifest.txt`
  - `index.tsv`

### Findings
- The Day 6 result stayed inside the Day 5 fence:
  - the same four canonical maintained benchmark emitters still define the report bundle
  - one CSV per canonical emitter remains the numeric artifact surface
  - the report command remains threshold-free
- `scripts/bench_canonical_report.sh` now owns a stronger but still lightweight bundle contract:
  - generated timestamp
  - report label
  - git commit
  - git branch
  - exact command mapping
  - explicit artifact inventory
  - one structured `index.tsv` row per canonical emitted artifact
- `Makefile` now owns the bounded label override seam:
  - `BENCH_CANONICAL_REPORT_LABEL`
- The first batch did not widen into:
  - benchmark-driver edits
  - runtime or exploratory benchmark capture
  - timing thresholds
  - machine-specific verdict logic
  - support-surface doc churn

### Validation
- Ran:
  - `make bench-canonical-report BENCH_CANONICAL_REPORT_DIR=build/bench-reports/canonical-day6-smoke BENCH_CANONICAL_REPORT_LABEL=day6-smoke`
- Verified the generated smoke bundle includes:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`
- Verified the smoke manifest reports:
  - `report_label=day6-smoke`
  - `git_commit=<current sprint-76 commit at run time>`
  - `git_branch=sprint-76`

### Day 6 Exit State
- Sprint 76 now has one stronger canonical report bundle with bounded longitudinal metadata.
- The first landing improved artifact comparability without reopening threshold policy or benchmark-driver schema ownership.

## Day 7 - Post-Landing Audit

### Goal
Re-audit the benchmark-governance surface after the Day 6 canonical reporting landing so the next batch targets the strongest remaining contradiction instead of reworking the same workflow seam.

### Actions
- Re-read the Day 6 canonical reporting batch artifact.
- Re-read the current benchmark-local interpretation in `benchmarks/README.md`.
- Re-read the current maintainer-policy interpretation in `docs/maintainer_guide.md`.
- Re-read the compact top-level benchmark summary in `README.md`.
- Rechecked the live report-bundle metadata terms (`index.tsv`, `report_label`, `git_commit`, `git_branch`) across the touched and support surfaces.

### Findings
- The Day 6 landing closed the strongest pure reporting-workflow contradiction:
  - `scripts/bench_canonical_report.sh` no longer reads like the strongest remaining Sprint 76 seam
  - `Makefile` no longer reads like the strongest remaining Sprint 76 seam
  - a second workflow-only script/Makefile batch is not the highest-value next move
- The strongest remaining seam has now shifted to support-surface drift around the landed stronger bundle contract:
  - required next batch:
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
  - support only if wording truly forces it:
    - `README.md`
- That rerank is explicit now:
  - `benchmarks/README.md` is the strongest next target because it is the main user-facing benchmark-governance interpretation surface and still describes the older manifest-only bundle
  - `docs/maintainer_guide.md` is the strongest second target because it owns the authoritative policy reading and still describes the older report-bundle shape
  - `README.md` remains support-only because its compact top-level summary is still broadly truthful even without naming the new structured bundle metadata
- The strongest still-deferred lane remains:
  - threshold-policy work around `bench-fast`, `wall-check`, `bench_reorder`, and `bench_amd_qg`
  - that remains real Sprint 76 pressure, but it is no longer the next batch center while the support surfaces still lag the landed Day 6 contract

### Validation
- Re-read `benchmarks/README.md`, `docs/maintainer_guide.md`, and `README.md` against the landed Day 6 workflow.
- Rechecked the current metadata terms across the touched and support surfaces.
- Reconfirmed that the reviewed baseline and Day 2 truth-surface split remain unchanged.

### Day 7 Exit State
- Sprint 76 no longer needs another workflow-only reporting batch.
- The strongest remaining seam is now fixed to support-surface reconciliation around the landed canonical report bundle.
- Day 8 can design a bounded documentation/policy follow-through batch from a current-state rerank instead of from the original backlog.

## Day 8 - Support-Surface Design

### Goal
Define the bounded documentation and policy follow-through contract for the landed Day 6 canonical report bundle before any support-surface edits begin.

### Actions
- Re-read the Day 7 rerank artifact.
- Re-read the current canonical-report wording in:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- Reconfirmed the compact top-level benchmark summary in `README.md`.
- Fixed the exact Day 9 touch set, preserved wording guarantees, and non-touch fence for the support-surface batch.

### Findings
- Sprint 76 now has one exact support-surface reconciliation batch:
  - required Day 9 batch:
    - `benchmarks/README.md`
    - `docs/maintainer_guide.md`
  - support only if wording truly forces it:
    - `README.md`
- The useful Day 8 clarification is now explicit:
  - `benchmarks/README.md` should become the clearer benchmark-local explanation of the stronger canonical bundle:
    - one CSV per canonical maintained benchmark still remains true
    - `manifest.txt` still remains true
    - `index.tsv` and bounded bundle metadata now need to be named directly
  - `docs/maintainer_guide.md` should move with it because it owns the authoritative policy reading of the canonical report surface
  - `README.md` remains support-only because the top-level statement that the report writes one bounded snapshot of the maintained canonical surface is still broadly accurate
- The Day 9 preservation fence is fixed now:
  - preserve:
    - threshold-free interpretation
    - the same four canonical maintained benchmark emitters
    - benchmark binaries as owners of emitted CSV row semantics
    - runtime and exploratory lanes staying outside the canonical report bundle
  - non-touch:
    - `scripts/bench_canonical_report.sh`
    - `Makefile`
    - canonical benchmark driver sources
    - threshold-policy work around `bench-fast`, `wall-check`, `bench_reorder`, and `bench_amd_qg`
    - reviewed proof-owner tests and examples

### Validation
- Re-read the Day 7 rerank plus the current benchmark-local and maintainer-policy wording.
- Reconfirmed that the support-surface batch can stay bounded without reopening workflow, threshold, or benchmark-driver work.
- Rechecked branch state before closing the design pass.

### Day 8 Exit State
- Sprint 76 now has one exact Day 9 support-surface reconciliation contract.
- The next batch can reconcile the benchmark-local and maintainer-policy wording with the landed stronger bundle without widening the sprint into threshold or workflow churn.

## Day 9 - Support-Surface Reconciliation Batch

### Goal
Reconcile the benchmark-local and maintainer-policy wording with the landed Day 6 canonical report bundle without reopening workflow, threshold, or benchmark-driver work.

### Actions
- Updated `benchmarks/README.md` to describe the stronger canonical report bundle directly.
- Updated `docs/maintainer_guide.md` to reflect the same landed bundle shape at the authoritative policy layer.
- Rechecked whether `README.md` needed compact top-level follow-through; it did not.
- Ran the targeted docs-only sanity set:
  - diff review
  - terminology/alignment reread across the touched support surfaces
  - touched-surface `wc -l`
  - branch-state verification

### Findings
- The Day 9 result stayed inside the Day 8 fence:
  - `benchmarks/README.md` now names:
    - `manifest.txt`
    - `index.tsv`
    - explicit artifact inventory
    - generated timestamp
    - bounded report-label support
    - bounded git metadata support
  - `docs/maintainer_guide.md` now names the same landed bundle shape at the policy layer
- The preserved governance split stayed intact:
  - one CSV per canonical maintained benchmark remains the numeric artifact surface
  - benchmark binaries still own emitted CSV row semantics
  - `make bench-canonical-report` still reads as threshold-free artifact reporting
  - runtime and exploratory lanes still stay outside the canonical report bundle
- No README follow-through was needed:
  - the compact top-level statement still accurately describes the report as one bounded snapshot of the maintained canonical surface

### Validation
- Ran the Sprint 76 docs-only sanity set:
  - diff review
  - terminology/alignment reread
  - touched-surface `wc -l`
  - branch-state recheck

### Day 9 Exit State
- The support surfaces now reconcile cleanly with the landed stronger canonical report bundle.
- The strongest remaining Sprint 76 seam is no longer support-surface drift around the canonical reporting batch.
