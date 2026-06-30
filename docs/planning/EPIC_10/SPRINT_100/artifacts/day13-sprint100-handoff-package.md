# Sprint 100 Day 13 Handoff Package

## Purpose

Day 13 integrates Sprint 100 evidence into the launch package for Epic 10
implementation sprints. The package reconciles baseline artifacts, target
language, residual ownership, claim state, evidence templates, and public-claim
audit results.

This artifact is the first place Sprints 101-109 should look before promoting
any product-maturity, comparison, performance, package, platform, or
state-of-the-art-adjacent claim.

## Artifact Index

| artifact | primary use |
|---|---|
| `day1-scope-baseline.md` | Sprint 100 workstream boundary and artifact organization |
| `day1-authoritative-inputs.txt` | input list for Epic 10 baseline and prior-epic context |
| `day2-reviewed-quality-baseline.md` | live reviewed quality baseline and CMake/Make parity counts |
| `day3-build-package-ci-baseline.md` | build, package, install, CI, and platform support baseline |
| `day4-source-test-maintainability-metrics.md` | source/test hotspot and maintainability baseline |
| `day5-comparison-benchmark-baseline.md` | external comparison, benchmark, coverage, and reporting surface map |
| `day6-state-of-the-art-target.md` | bounded Epic 10 target, comparison set, non-goals, and disallowed claims |
| `day7-residual-claim-map.md` | Epic 9 residual conversion and sprint owner draft |
| `day8-claim-dependency-model.md` | dependency-aware claim map for Sprints 101-109 |
| `day9-solver-comparison-template.md` | solver comparison evidence rules |
| `day9-solver-template-pilot-cholesky-csc.md` | filled Cholesky CSC comparison pilot |
| `day10-benchmark-coverage-performance-template.md` | benchmark, coverage, and performance evidence rules |
| `day10-benchmark-template-pilot-canonical-report.md` | filled canonical benchmark report pilot |
| `day11-platform-packaging-evidence-template.md` | package, platform, consumer, and ABI evidence rules |
| `day12-public-claim-audit.md` | public/support claim classification and wording queue |

Reusable templates:

| template | use when |
|---|---|
| `templates/solver-comparison-evidence-template.md` | adding or widening solver-family comparison evidence |
| `templates/benchmark-interpretation-template.md` | interpreting benchmark or generated report output |
| `templates/coverage-evidence-template.md` | changing coverage thresholds, backends, or reviewed/supplemental status |
| `templates/performance-sentinel-template.md` | adding thresholded local runtime regression gates |
| `templates/package-proof-template.md` | changing install/export/package metadata or downstream consumer proof |
| `templates/platform-tier-template.md` | changing platform scope, CI counts, or reviewed/supplemental lanes |
| `templates/abi-decision-template.md` | deciding whether static-first remains enough or shared-library/ABI proof is added |
| `templates/consumer-validation-checklist.md` | auditing package/platform wording and consumer validation completeness |

## Earned Claim Register

| claim | proof source | carry-forward rule |
|---|---|---|
| post-Epic-9 reviewed baseline is clean | Day 2 quality run: `make quality-review-full`; CMake tests `54`; Make/CMake parity `54` vs `54`; CTest `54 / 54` | rerun required before final Sprint 100 closeout and after later source/test changes |
| static-first package support is current package truth | Day 3 package baseline; Make install `14 / 14`; CMake install `16 / 16`, `0` skips; Day 11 package template | do not widen to shared library, ABI stability, package-manager, or symmetric platform claims without Sprint 108 proof |
| tiered platform support wording is accurate | Day 3 platform draft; Day 11 platform template; Day 12 public audit | keep Linux/macOS/Windows scopes different unless new lanes are implemented and validated |
| threshold-free canonical benchmark report exists | Day 5 benchmark map; Day 10 benchmark pilot | interpret as local report artifact, not a pass/fail timing gate |
| selected external dense-reference solver lanes exist | Day 5 comparison map; Day 9 Cholesky pilot | do not generalize to every direct solver or every solver family |
| real-only double scalar contract remains explicit | Day 8 claim map; Day 12 audit | keep complex/mixed precision as non-claims unless replanned |
| default reviewed index width remains 32-bit | Day 8 claim map; Day 12 audit | keep 64-bit width as bounded compile-time seam, not broad maturity claim |

## Candidate Claim Register

| candidate claim | owner | required Sprint 100 evidence input |
|---|---|---|
| compressed-first workflows are primary product path | Sprint 101 | Day 6 target, Day 8 claim map, Day 12 public audit |
| mutable shell is supported but secondary | Sprint 101 / Sprint 107 | Day 6 target, Day 8 claim map |
| direct solver oracle evidence is deeper | Sprint 102 | Day 5 comparison map, Day 9 solver template, Cholesky pilot |
| iterative/eigensolver/SVD comparison architecture exists | Sprint 103 | Day 5 gap table, Day 9 solver template |
| backend/runtime contract is clearer and observable | Sprint 104 | Day 5 benchmark baseline, Day 10 benchmark/performance templates |
| local performance sentinels are decision-grade | Sprint 104 | Day 10 performance-sentinel template |
| reorder/fill and graph evidence is clearer | Sprint 105 | Day 5 benchmark map, Day 10 benchmark template |
| source and giant-test risk is reduced | Sprint 106 | Day 4 maintainability metrics, Day 8 claim map |
| user-facing solver-selection path is clearer | Sprint 107 | Day 6 target, Day 8 claim map, Day 12 public audit |
| platform support tiers are explicit | Sprint 108 | Day 3 baseline, Day 11 platform template, Day 12 audit |
| shared-library or ABI support is available | Sprint 108 stretch | Day 11 ABI decision template; implementation and runtime-loader proof if promoted |
| final competitive calibration is truthful | Sprint 109 | all Sprint 100 artifacts plus Sprints 101-108 evidence |

## Blocked and Non-Goal Register

These claims remain blocked, stretch, or non-goal unless a later sprint
explicitly replans them and adds proof:

- broad state-of-the-art replacement for SuiteSparse/PETSc/Trilinos-class
  ecosystems;
- SuiteSparse replacement or parity;
- every-solver-family external validation;
- universal vendor backend parity;
- portable timing superiority;
- universal reorder/fill superiority;
- GPU sparse kernels;
- distributed-memory sparse solvers;
- broad complex-number support;
- broad mixed-precision maturity;
- stable dynamic ABI guarantee;
- shared-library-first package contract;
- package-manager ecosystem ownership;
- Windows Makefile parity;
- Windows reviewed install-validation parity;
- symmetric Linux/macOS/Windows reviewed parity;
- full replacement of the mutable linked-list shell.

## Sprint Handoff Instructions

| sprint | must use | before claim is earned |
|---|---|---|
| Sprint 101 | Day 6 target, Day 8 claim map, Day 12 audit | compressed-first implementation, lifecycle/error tests, docs/examples, compatibility-shell wording |
| Sprint 102 | Day 5 comparison baseline, Day 9 solver template | direct-family fixture taxonomy, oracle/reference behavior, tolerance model, focused validation |
| Sprint 103 | Day 9 solver template, Sprint 102 helper patterns | convergence/eigen/rank fixture taxonomy, residual criteria, unsupported cases, focused validation |
| Sprint 104 | Day 10 benchmark/performance templates | backend observability tests, benchmark fields, sentinel baseline/rationale if thresholds are added |
| Sprint 105 | Day 10 benchmark template, Day 7 residual queue | named fixture evidence, fill field contract, local timing caveats, graph/reorder validation |
| Sprint 106 | Day 4 metrics, Day 8 claim map | before/after size metrics, source-list parity, focused tests, full C quality chain for code changes |
| Sprint 107 | Day 12 audit, prior sprint evidence | solver-selection docs and examples aligned with earned technical evidence only |
| Sprint 108 | Day 3 baseline, Day 11 package/platform templates | support tier table, expected counts, exclusions, package proof, ABI decision if touched |
| Sprint 109 | Day 8 claim map, Day 12 audit, all closeouts | final validation, unsupported-claim cleanup, residual queue, competitive calibration |

## Reconciliation Result

No blocking contradictions were found between the Sprint 100 artifacts and the
Epic 10 project plan.

Important reconciliations:

- "State-of-the-art" remains a bounded target, not an earned project claim.
- Static-first package support is earned; shared-library and ABI maturity are
  not.
- Linux, macOS, and Windows support remain tiered, not symmetric.
- Benchmarks are measurement/reporting evidence, not portable performance
  superiority.
- External solver comparison evidence is selected and family-local, not
  universal.
- Source/test maintainability improvements are candidate claims until Sprint
  106 produces before/after evidence.

## Deferred Items

| item | owner | reason |
|---|---|---|
| soften README "benchmarks prove" wording if desired | Sprint 107 | not currently false, but "provide measurement evidence for" may be clearer |
| add current-benchmark caveat to `docs/algorithm.md` | Sprint 105 or Sprint 107 | algorithm guide contains historical measurement prose; benchmark docs own current interpretation |
| decide whether public header "ABI break" wording should become "API/source compatibility" wording | Sprint 108 | avoid implying stable dynamic ABI unless proof exists |
| produce first-class support tier table with expected counts and staged exclusions | Sprint 108 | current docs are accurate, but Sprint 108 owns final support-tier publication |
| fill performance sentinel template for `wall-check` if it becomes decision-grade Epic 10 evidence | Sprint 104 or Sprint 109 | current sentinel exists, but Sprint 100 did not rerun or rebaseline it |

## Day 14 Closeout Checklist

Day 14 should complete these checks before Sprint 100 is closed:

- [ ] confirm every Sprint 100 plan deliverable has a corresponding artifact;
- [ ] confirm all Day 1-13 artifacts are linked by this handoff or the working
      notes;
- [ ] confirm no artifact promotes blocked/non-goal claims as earned;
- [ ] run final documentation hygiene:
      `git diff --check`;
- [ ] run final trailing-whitespace scan over
      `docs/planning/EPIC_10/SPRINT_100`;
- [ ] if any `.c` or `.h` files are touched on Day 14, run
      `make format && make lint && make test`;
- [ ] write Sprint 100 closeout notes and artifact index;
- [ ] record final validation results in `WORKING_NOTES.md`.

## Day 13 Conclusion

Sprint 100 now has an integrated baseline and evidence contract for Epic 10.
Sprints 101-109 can proceed from earned claims, candidate claims, and explicit
non-goals without relying on broad aspirational state-of-the-art language.
