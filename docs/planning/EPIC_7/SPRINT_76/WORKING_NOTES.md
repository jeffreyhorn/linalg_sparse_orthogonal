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
