# Sprint 79 Working Notes

## Day 1 - Baseline and Scope

### Goal
Establish a precise Sprint 79 baseline for numerical assurance expansion, cross-surface integration, and Epic 7 closeout using the live tree, the Sprint 78 carry-forward queue, and the current permanent proof-owner and support-policy surfaces rather than another broad planning reset.

### Actions
- Re-read the Sprint 79 plan in `docs/planning/EPIC_7/SPRINT_79/PLAN.md` and the Sprint 79 section in `docs/planning/EPIC_7/PROJECT_PLAN.md`.
- Re-read the Sprint 78 closeout handoff in `docs/planning/EPIC_7/SPRINT_78/artifacts/day14-closeout-and-handoff.md`.
- Rechecked the maintained reviewed wrapper surface with `make -n quality-review-full`.
- Re-materialized the reviewed CMake parity tree with `make quality-review-cmake-compile`.
- Reconfirmed the reviewed CMake parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Re-read the strongest current proof-owner, lifecycle, reporting, packaging, and workflow policy surfaces:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
- Rechecked the strongest current public contract seams in:
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- Captured the live raw `wc -l` hotspot map for the strongest likely Sprint 79 touch surfaces across proof-owner tests, support surfaces, workflows, and install/reporting surfaces.
- Rechecked high-signal lifecycle, progress/cancel, repeated-run, writeback, oracle, benchmark, and install-proof wording with targeted `rg`.

### Findings
- Sprint 79 starts from the same strongest local reviewed baseline as Sprint 78:
  - `make quality-review-full`
- Reviewed CMake parity was re-materialized live and remains explicit:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
- The highest-value Sprint 79 pressure is now clearly narrowed to:
  - assurance-gap rerank
  - bounded oracle/property/lifecycle regression expansion
  - final cross-surface integration sweep
  - full validation sweep
  - Epic 7 summary and residual finalization
  - retrospective and handoff
- The strongest likely Sprint 79 proof-owner and assurance surfaces are now explicit from the live tree:
  - `tests/test_chol_csc.c` = `4724`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_iterative.c` = `2841`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_integration.c` = `2543`
  - `tests/test_reorder_nd.c` = `2287`
  - `tests/test_fuzz.c` = `651`
- The strongest likely Sprint 79 support and final-truth surfaces are also explicit from the live tree:
  - `README.md` = `1045`
  - `INSTALL.md` = `265`
  - `docs/maintainer_guide.md` = `685`
  - `benchmarks/README.md` = `393`
  - `docs/tutorial.md` = `473`
  - `examples/README.md` = `166`
  - `include/sparse_iterative.h` = `773`
  - `include/sparse_eigs.h` = `651`
  - `include/sparse_ldlt.h` = `335`
  - `include/sparse_cholesky.h` = `227`
- The strongest likely Sprint 79 workflow, install, and reporting proof surfaces are explicit before deeper audit:
  - `tests/test_install.sh` = `172`
  - `tests/test_cmake_install.sh` = `146`
  - `scripts/bench_canonical_report.sh` = `101`
  - `Makefile` = `899`
  - `CMakeLists.txt` = `413`
  - `.github/workflows/ci.yml` = `223`
  - `.github/workflows/macos-ci.yml` = `117`
  - `.github/workflows/windows-ci.yml` = `63`
- Sprint 78's carry-forward queue still matters directly to Sprint 79:
  - `src/sparse_iterative.c`
  - `tests/test_ldlt_csc.c`
  - `src/sparse_chol_csc.c`
  - `tests/test_qr.c`
  - later mixed backlog only after those higher-value lanes move
- The strongest Day 1 clarification is now explicit:
  - Sprint 79 is not another broad cleanup sprint.
  - It is a bounded final assurance and integration phase that should spend its budget on the highest-value remaining numerical, lifecycle, oracle, and cross-surface truth gaps.
- The maintained Sprint 79 non-goal fence is already clear and must be preserved:
  - no fake algorithm-breadth or platform-breadth claim widening
  - no broad subsystem redesign disguised as final assurance work
  - no threshold-driven benchmark or performance-governance reopening
  - no public-surface rewrite when narrower proof-owner or truth-surface fixes suffice
  - no Epic 7 closeout summary that outruns the maintained evidence

### Validation
- Rechecked `make -n quality-review-full`.
- Rebuilt the reviewed CMake tree with `make quality-review-cmake-compile`.
- Reconfirmed the reviewed parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Captured the live proof-owner, support-surface, workflow, and install/reporting hotspot maps from direct `wc -l` measurement plus targeted terminology scans.

### Day 1 Exit State
- Sprint 79 no longer starts from a generic “final wrap-up” prompt.
- The assurance, integration, summary, and closeout workstreams, strongest likely touch surfaces, and preserved non-goal fence are fixed in writing.
- The branch is clean after the Day 1 baseline commit.

## Day 2 - Validation Baseline

### Goal
Reconfirm the Sprint 79 implementation-day validation contract and the live proof-surface split across reviewed CMake proof owners, representative examples, canonical report command surfaces, and install/export proof owners before any final assurance batch lands.

### Actions
- Re-read the Sprint 79 Day 2 plan expectations in `docs/planning/EPIC_7/SPRINT_79/PLAN.md`.
- Reconfirmed the reviewed CMake parity anchor with `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner tests and representative examples most likely to matter in Sprint 79:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- Rechecked the maintained canonical report command surface with `make -n bench-canonical-report`.
- Reconfirmed the root `build/` canonical benchmark emitters consumed by the canonical report path:
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`
- Re-read the strongest current policy wording across:
  - `docs/maintainer_guide.md`
  - `README.md`
  - `benchmarks/README.md`
  - `INSTALL.md`
  - `docs/tutorial.md`
  - `examples/README.md`

### Findings
- Sprint 79 inherits the same strongest local reviewed baseline:
  - `make quality-review-full`
- Reviewed CMake parity remains the main truthfulness anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- The Sprint 79 authority split is now fixed explicitly:
  - bounded `*.c` / `*.h` landing days:
    - `make format`
    - `make lint`
    - `make test`
  - substantial assurance or integration batches:
    - `make quality-review-full`
  - docs-only audit/design/review days:
    - targeted sanity checks only
- The reviewed CMake tree currently owns the key Sprint 79 proof surfaces most likely to be stressed by final assurance and integration work:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
- The representative example surfaces most likely to matter remain explicit:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- The canonical benchmark/reporting lane remains command- and script-owned rather than reviewed-binary-owned:
  - `make bench-canonical-report`
  - `scripts/bench_canonical_report.sh`
  - root `build/` canonical emitters:
    - `build/bench_refactor_csc`
    - `build/bench_chol_csc`
    - `build/bench_iterative_reuse`
    - `build/bench_eigs_reuse`
- The maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- The strongest current proof and truth-surface ownership split is already stable:
  - `tests/test_chol_csc.c` owns family-local large-`n` Cholesky CSC handoff, publish-back, and backend-contract proof
  - `tests/test_integration.c` owns public one-shot vs explicit repeated-run lifecycle parity plus callback/cancel truth
  - `tests/test_fuzz.c` remains bounded seeded generative follow-through for retained lifecycle/property pressure
  - `benchmarks/README.md` and `docs/maintainer_guide.md` already keep the canonical benchmark/reporting and threshold-free reading explicit
  - `INSTALL.md` plus the install scripts already keep the local install/export proof split explicit
- The highest-signal Sprint 79 rerun set is now fixed around the likely touched final-closeout seams:
  - `./build/quality-review-cmake/test_chol_csc`
  - `./build/quality-review-cmake/test_ldlt_csc`
  - `./build/quality-review-cmake/test_ldlt`
  - `./build/quality-review-cmake/test_iterative`
  - `./build/quality-review-cmake/test_qr`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_eigs`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `make bench-canonical-report`
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

### Validation
- Reconfirmed `ctest -N --test-dir build/quality-review-cmake`.
- Rechecked the strongest reviewed proof-owner test/example binaries most likely to matter in Sprint 79.
- Rechecked `make -n bench-canonical-report` and the root `build/` canonical maintained emitters it consumes.
- Re-read the strongest current benchmark, lifecycle, and install/export ownership wording across the maintained support and policy surfaces.

### Day 2 Exit State
- Sprint 79 now has one explicit implementation-day validation contract.
- The live proof split across reviewed tests, examples, canonical report workflow, and install/export proof owners is fixed in writing.
- The highest-signal rerun set is explicit before the assurance-gap audit begins.

## Day 3 - Assurance Gap Re-audit

### Goal
Re-rank the remaining highest-value oracle, property, lifecycle, and platform-confidence gaps after the main Epic 7 implementation work so Sprint 79 starts from one explicit closeout contradiction map instead of a generic “final assurance” bucket.

### Actions
- Re-read the Sprint 79 Day 3 plan expectations in `docs/planning/EPIC_7/SPRINT_79/PLAN.md`.
- Re-read the strongest current residual-assurance sections in `docs/maintainer_guide.md`:
  - direct-usability queue
  - platform-confidence interpretation
  - benchmark-governance ownership
  - packaging/platform residual dispositions
- Rechecked the strongest current support-surface wording across:
  - `README.md`
  - `INSTALL.md`
  - `benchmarks/README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
- Rechecked high-signal residual, deferred, and truth-boundary wording with targeted `rg` across:
  - proof-owner tests
  - support surfaces
  - workflow surfaces
- Re-read the strongest likely proof owners for residual lifecycle/property/oracle value:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_integration.c`
  - `tests/test_iterative.c`
  - `tests/test_qr.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_fuzz.c`

### Findings
- Sprint 79's broad final-assurance problem is now reduced to one ranked contradiction map instead of one generic closeout bucket:
  - strongest first target:
    - direct-family lifecycle/property assurance
  - strongest second target:
    - platform-confidence-limited property coverage
  - strongest third target:
    - family-local differential/oracle follow-through
  - strongest later target:
    - support-surface overclaim sweep
- The strongest current contradiction center is not benchmark reporting, packaging, or another broad documentation pass.
- It is the bounded residual direct-usability queue already named in policy:
  - no-reorder linked-list Cholesky bit-identical cancellation restoration
  - broader CSC progress-callback parity beyond the landed bounded Cholesky orchestration checkpoints
  - any later LDL^T callback follow-through
- That makes direct-family lifecycle/property assurance the best first Sprint 79 target by closeout rules:
  - highest user-facing correctness value still left in the explicit residual queue
  - bounded enough to land without reopening broad subsystem design
  - strongest proof payoff because it sits at the line between public callback/cancel truth and family-local implementation caveats
  - strongest final-integration leverage because the same seam touches public docs, policy wording, and durable regression ownership
- The strongest likely proof owners for that first lane are already clear:
  - `tests/test_integration.c` as the public lifecycle/callback/cancel oracle owner
  - `tests/test_fuzz.c` as the bounded seeded generative/property follow-through owner
  - family-local support only if the first lane truly forces it:
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `tests/test_ldlt_csc.c`
- Platform-confidence-limited property coverage is the strongest second target because:
  - Windows still excludes `test_fuzz` from the reviewed CMake subset
  - Linux and macOS still exercise that property lane directly
  - the support surfaces already state that boundary truthfully, so this is a real confidence limit but a weaker first landing than the direct lifecycle/property seam itself
- Family-local differential/oracle follow-through is the strongest third target:
  - QR, LDL^T, LDL^T CSC, and ND already carry strong residual- and oracle-heavy proof
  - those lanes still have value for later assurance expansion
  - but they currently read more like coherent large proof owners than the strongest unresolved contradiction center
- Support-surface overclaim is the weakest current assurance lane:
  - the current maintained docs already say the strongest caveats directly
  - callback/path-local limitations, benchmark truth, install/export proof, and Windows fuzz exclusion are all already explicit
  - that means Sprint 79 should not start from another wording-first pass unless the later integration sweep reveals a real contradiction
- Where the current proof package is already coherent:
  - repeated-run direct workflow and benchmark/reporting ownership
  - install/export proof split
  - benchmark-governance and threshold-free reporting
  - width/scalar/platform truth fences
- The strongest Day 3 clarification is now explicit:
  - Sprint 79 is not primarily missing broad residual-norm tests.
  - It is primarily missing the best final bounded expansion of lifecycle/property assurance on the direct-family callback/cancel seam.

### Validation
- Reconfirmed the residual direct-usability and platform-confidence queues from `docs/maintainer_guide.md`.
- Rechecked the strongest current support-surface caveat wording in `README.md`, `INSTALL.md`, `benchmarks/README.md`, `docs/tutorial.md`, and `examples/README.md`.
- Rechecked residual/deferred terminology across the likely proof owners, workflows, and support surfaces with targeted `rg`.

### Day 3 Exit State
- Sprint 79 no longer carries a generic “final assurance” prompt.
- The remaining assurance problem is reduced to a ranked lifecycle/property/oracle/platform-confidence map.
- Day 4 can now freeze a first final-assurance fence from a real current-state ranking.

## Day 4 - Assurance Boundary Freeze

### Goal
Refine the Day 3 assurance ranking into one exact first landing fence so Sprint 79 spends its final implementation budget on the strongest lifecycle/property seam without reopening broader documentation, benchmark, packaging, or platform campaigns.

### Actions
- Re-read the Sprint 79 Day 4 plan expectations in `docs/planning/EPIC_7/SPRINT_79/PLAN.md`.
- Re-read the Day 3 ranked contradiction map in `docs/planning/EPIC_7/SPRINT_79/artifacts/day3-assurance-gap-reaudit.md`.
- Re-read the strongest current residual direct-usability and platform-confidence queues in `docs/maintainer_guide.md`.
- Re-ranked the Day 3 seams against:
  - closeout payoff
  - proof clarity
  - runtime cost
  - bounded landing value
- Separated the likely first-batch landing surfaces, support-only surfaces, and later/deferred closeout lanes.

### Findings
- Sprint 79 now has one explicit first final-assurance fence instead of a generic numerical-assurance backlog:
  - required first landing:
    - `tests/test_integration.c`
    - `tests/test_fuzz.c`
  - support only if the first landing forces it:
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `tests/test_ldlt_csc.c`
    - `docs/maintainer_guide.md`
    - `README.md`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
  - explicitly deferred:
    - `tests/test_qr.c`
    - `tests/test_reorder_nd.c`
    - benchmark/reporting surfaces
    - install/export proof scripts
    - workflow YAML surfaces
- The strongest first Sprint 79 lane is now fixed as:
  - public lifecycle/property assurance
- The useful Day 4 clarification is explicit:
  - the first batch should target the public callback/cancel and repeated-run lifecycle truth seam
  - it should use `tests/test_integration.c` as the public oracle owner and `tests/test_fuzz.c` as the bounded seeded property owner
  - it should move family-local direct-solver tests only if the public/property landing cannot be expressed cleanly without them
- Platform-confidence-limited property coverage remains the strongest second lane, not the first landing:
  - Windows fuzz exclusion is a real confidence boundary
  - but it is not the best first batch center because the current docs already describe it truthfully
- Family-local differential/oracle follow-through remains explicitly later than the first batch:
  - `tests/test_qr.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_reorder_nd.c`
- The strongest non-goal fence is now explicit:
  - no broad proof campaign across all solver families
  - no new benchmark, package, or workflow mechanics
  - no public-surface rewrite unless the first landing forces narrower wording follow-through
  - no summary or retrospective work before the first proof gap is bounded

### Validation
- Re-ranked the Day 3 assurance seams against closeout value, proof clarity, runtime cost, and bounded landing value.
- Reconfirmed the residual direct-usability and platform-confidence queues from `docs/maintainer_guide.md`.
- Reconfirmed the likely first-batch proof-owner roles against the current public, family-local, and property ownership split.

### Day 4 Exit State
- Sprint 79 now has one explicit first final-assurance fence.
- The first landing, support-only surfaces, and deferred lanes are fixed in writing before design begins.
- Day 5 can now define one bounded implementation/proof contract instead of another generic assurance expansion.

## Day 5 - Assurance Design

### Goal
Define the bounded implementation/proof contract for the first Sprint 79 assurance landing so Day 6 can improve one real lifecycle/property seam without widening into broader family-local oracle churn, support-surface edits, or workflow work.

### Actions
- Re-read the Sprint 79 Day 5 plan expectations in `docs/planning/EPIC_7/SPRINT_79/PLAN.md`.
- Re-read the Day 4 boundary in `docs/planning/EPIC_7/SPRINT_79/artifacts/day4-first-assurance-boundary.md`.
- Re-read the strongest current public callback/cancel and lifecycle ownership split in:
  - `tests/test_integration.c`
  - `tests/test_fuzz.c`
  - `docs/maintainer_guide.md`
- Rechecked the strongest likely support proof and interpretation surfaces:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
  - `README.md`
- Fixed the exact ownership split, preserved guarantees, and non-touch set for the first batch.

### Findings
- Sprint 79 now has one explicit first implementation contract:
  - required implementation center:
    - `tests/test_integration.c`
    - `tests/test_fuzz.c`
  - support only if the first batch truly forces it:
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `tests/test_ldlt_csc.c`
    - `docs/maintainer_guide.md`
    - `README.md`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
- The Day 5 ownership split is now fixed:
  - primary public oracle owner:
    - `tests/test_integration.c`
  - bounded seeded property owner:
    - `tests/test_fuzz.c`
  - family-local support proof owners only if a public/property seam truly needs them:
    - `tests/test_chol_csc.c`
    - `tests/test_ldlt.c`
    - `tests/test_ldlt_csc.c`
  - public/support interpretation owners only if wording truly moves:
    - `docs/maintainer_guide.md`
    - `README.md`
    - `include/sparse_cholesky.h`
    - `include/sparse_ldlt.h`
- The strongest Day 6 batch goal is now fixed:
  - strengthen the public callback/cancel and repeated-run lifecycle assurance surface
  - prefer one bounded public-oracle improvement first in `tests/test_integration.c`
  - add bounded seeded generative/property follow-through in `tests/test_fuzz.c` only where it increases assurance without widening the contract
- The preserved guarantees are now explicit:
  - preserve current public callback/cancel behavior
  - preserve the current family/path-local caveat reading
  - preserve current reviewed validation scope, including the Windows fuzz exclusion truth
  - preserve current benchmark/reporting, install/export, and workflow ownership splits
  - preserve current runtime-bounded interpretation of the fuzz/property lane
- The useful Day 5 clarification is now explicit:
  - the first batch should not try to “fix every deferred direct-usability item”
  - it should instead land one proof improvement that narrows the residual queue with the highest public assurance payoff
  - the best shape is a public oracle first, with property follow-through only where it makes that oracle harder to regress
- The exact first-batch non-touch set is now fixed:
  - unrelated solver-family proof owners:
    - `tests/test_qr.c`
    - `tests/test_reorder_nd.c`
  - benchmark/reporting surfaces
  - install/export proof scripts
  - workflow YAML surfaces
  - broader docs churn
  - unrelated implementation or API work

### Validation
- Re-read the Day 4 first-assurance fence against the first-batch files.
- Reconfirmed the strongest public oracle and bounded property owners in `tests/test_integration.c` and `tests/test_fuzz.c`.
- Reconfirmed the family-local, support-surface, and platform-confidence fences against the current maintained policy wording.

### Day 5 Exit State
- Sprint 79 now has one explicit first implementation/proof contract.
- The ownership split, preserved guarantees, and non-touch set are fixed in writing before edits begin.
- Day 6 can now land one bounded lifecycle/property assurance improvement from a precise design.

## Day 6 - Differential / Oracle Batch

### Goal
Land the first bounded Sprint 79 assurance batch by strengthening the public repeated-run lifecycle oracle and the bounded seeded property lane without widening into family-local proof churn, support-surface edits, or implementation/API work.

### Actions
- Added one new public lifecycle parity oracle to `tests/test_integration.c`:
  - `test_public_lifecycle_refactor_same_pattern_matches_one_shot_ldlt`
- Added one bounded seeded large-`n` property lane to `tests/test_fuzz.c`:
  - `test_property_large_n_ldlt_public_lifecycle_same_pattern_csc`
- Added the local KKT-style indefinite matrix helpers needed to express the new property cleanly in `tests/test_fuzz.c`:
  - `build_large_kkt(...)`
  - `perturb_large_kkt_values_in_place(...)`
- Kept the batch inside the Day 5 fence:
  - no family-local direct-solver proof-owner edits
  - no support-surface wording edits
  - no benchmark, install/export, workflow, or API churn
- Ran the full required validation gate because `*.c` changed:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`

### Findings
- Sprint 79 now has one landed first assurance batch in the required implementation center:
  - `tests/test_integration.c`
  - `tests/test_fuzz.c`
- The new public oracle in `tests/test_integration.c` proves that the explicit repeated-run LDL^T lifecycle stays aligned with the one-shot CSC-backed LDL^T lane across same-pattern refactors on a large indefinite KKT family.
- The new bounded seeded property in `tests/test_fuzz.c` extends that same assurance shape into a generative large-`n` lane:
  - seeds:
    - `809u`
    - `1451u`
    - `2029u`
  - matrix shape:
    - `n_top = SPARSE_CSC_THRESHOLD + 12`
    - `n_bot = 8`
  - proof focus:
    - repeated-run public lifecycle parity
    - same-pattern refactor stability
    - one-shot CSC path agreement
- The landed batch preserves the Day 5 guarantees:
  - current public callback/cancel behavior remains unchanged
  - current family/path-local caveat reading remains unchanged
  - current Windows fuzz exclusion truth remains unchanged
  - benchmark/reporting, install/export, and workflow ownership splits remain unchanged
- The support-only surfaces stayed untouched:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
  - `docs/maintainer_guide.md`
  - `README.md`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

### Validation
- `make format` passed.
- `make lint` passed.
- `make test` passed.
- `make quality-review-full` passed.
- Reviewed anchors stayed exact:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
  - Makefile/CMake parity = `53 vs 53`
  - reviewed CMake `ctest` = `53 / 53`
  - `Total Test time (real) = 344.29 sec`
- The new assurance surfaces passed directly in the maintained reviewed path:
  - `test_integration` = `51 / 51`
  - `test_fuzz` = `26 / 26`
  - retained new proofs:
    - `test_public_lifecycle_refactor_same_pattern_matches_one_shot_ldlt`
    - `test_property_large_n_ldlt_public_lifecycle_same_pattern_csc`
  - retained property output:
    - `large-n LDLT CSC lifecycle property: 3/3 passed`

### Day 6 Exit State
- The first Sprint 79 assurance batch is landed.
- The public repeated-run LDL^T lifecycle now has both a bounded oracle test and a bounded seeded large-`n` property test.
- Sprint 79 can now rerank the next final-assurance seam from a stronger public lifecycle baseline.
