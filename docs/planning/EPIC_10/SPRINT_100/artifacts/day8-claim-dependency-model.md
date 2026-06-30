# Sprint 100 Day 8 Claim Map & Sprint Dependency Model

## Purpose

Day 8 turns the Epic 10 target and residual conversion into a dependency-aware
claim model. Future sprint closeouts should use this artifact to decide
whether a claim is earned, still candidate, blocked, or explicitly outside
scope.

## Claim State Definitions

| state | meaning |
|---|---|
| earned | supported by live implementation, tests/docs/artifacts, and validation commands |
| candidate | planned or partially supported, but still needs implementation and evidence |
| blocked | cannot be claimed until a prerequisite decision, implementation, or validation gap is resolved |
| non-goal | intentionally outside current Epic 10 scope unless replanned |

## Current Epic 10 Claim Map

| claim | current state | sprint owner | prerequisites | minimum evidence to mark earned |
|---|---|---|---|---|
| post-Epic-9 reviewed baseline is clean | earned | Sprint 100 | Day 2 quality run | `make quality-review-full` passed; CMake tests `54`; Make/CMake parity `54` vs `54`; CTest `54 / 54` |
| static-first package support is the current package truth | earned | Sprint 100 / Sprint 108 | Day 3 package baseline | install/export proof surfaces named; inherited Make install `14 / 14`; CMake install `16 / 16`, `0` skips; docs preserve static-first wording |
| compressed-first workflows are the primary product path | candidate | Sprint 101 | Sprint 100 claim map and evidence templates | API/design artifact, implementation batch, lifecycle tests, docs/examples, compatibility-shell non-claim preserved |
| mutable matrix shell remains compatibility-supported | earned but needs clearer secondary wording | Sprint 101 / Sprint 107 | Day 6 non-goal fence | compatibility docs and tests remain passing; public wording describes shell as supported secondary path |
| direct solver oracle evidence is deeper than Epic 9 | candidate | Sprint 102 | Sprint 101 ownership/lifecycle stability; solver template | fixture taxonomy, comparison helper extraction, focused direct-family tests, external/dense-reference artifacts |
| LDLT CSC broad indefinite corpus parity | blocked/stretch | Sprint 102 | proof architecture and runtime budget | named corpus, external oracle, tolerance model, focused LDLT CSC validation; otherwise remains non-claim |
| iterative solver external comparison architecture exists | candidate | Sprint 103 | solver template; direct oracle helper patterns | convergence fixture taxonomy, residual criteria, external or deterministic reference artifact, focused iterative validation |
| eigensolver/LOBPCG external comparison architecture exists | candidate | Sprint 103 | solver template; convergence fixture design | eigenpair fixtures, tolerance/cluster policy, runtime caps, focused eigensolver validation |
| QR/SVD external comparison is stronger | candidate | Sprint 102 / Sprint 103 | direct/SVD fixture taxonomy | reference ownership, rank/conditioning cases, tolerance model, focused QR/SVD validation |
| backend/runtime contract is clearer and observable | candidate | Sprint 104 | Sprint 101-103 solver surfaces stable enough to measure | backend descriptor/observability tests, benchmark fields, fallback docs, no vendor-parity overclaim |
| local performance sentinels are decision-grade | candidate | Sprint 104 | benchmark/performance template | bounded local thresholds or artifacts with machine/context caveats; no portable superiority wording |
| reorder/fill evidence is clearer | candidate | Sprint 105 | benchmark template; runtime contract | fill metric contract, named matrix artifacts, `nnz_L` primary field, local timing caveat |
| graph/large-matrix scalability evidence is stronger | candidate | Sprint 105 | graph/reorder audit | deterministic memory/runtime guardrails where practical; focused graph/reorder validation |
| large source ownership risk is reduced | candidate | Sprint 106 | Day 4 hotspot baseline; family-local seams identified | before/after metrics, extraction artifacts, source-list parity, focused tests, full C quality chain if code changes |
| giant test ownership risk is reduced | candidate | Sprint 106 | Day 4 hotspot baseline; test owner map | fixture/helper extraction, Make/CMake count parity, focused tests, before/after metrics |
| user-facing solver-selection path is clearer | candidate | Sprint 107 | Sprint 101-105 technical evidence | solver-selection guide, compressed-first examples, ownership/error docs, doc/example validation |
| benchmark interpretation is clearer to users | candidate | Sprint 107 | Sprint 104-105 benchmark/reorder artifacts | docs distinguish local artifacts, sentinels, and non-claims; no portable timing wording |
| platform support tiers are explicit | candidate | Sprint 108 | Day 3 platform baseline; Sprint 107 docs ready | Linux/macOS/Windows tier table, reviewed/supplemental/staged lanes, expected counts, exclusion register |
| shared-library/ABI support is available | blocked/stretch | Sprint 108 | explicit ABI support decision | implementation and install/export/runtime-loader proof; otherwise keep static-first non-claim |
| Windows Makefile/install parity exists | blocked/stretch | Sprint 108 | explicit platform decision | Windows workflow lane, expected counts, install proof; otherwise preserve non-claim |
| final Epic 10 competitive calibration is truthful | candidate | Sprint 109 | Sprints 101-108 evidence package | final validation, claim audit, unsupported-claim cleanup, residual queue, Epic retrospective |
| broad state-of-the-art replacement claim | non-goal | Sprint 109 | not applicable | remains disallowed unless all prerequisite dimensions earn evidence far beyond current plan |

## Sprint Dependency Table

| sprint | depends on | feeds |
|---|---|---|
| Sprint 101 | Sprint 100 claim map and evidence templates | Sprint 102 direct solver evidence, Sprint 107 docs/examples, Sprint 109 product calibration |
| Sprint 102 | Sprint 101 compressed ownership stability; solver evidence template | Sprint 103 comparison patterns, Sprint 106 extraction targets, Sprint 107 solver guide, Sprint 109 comparison package |
| Sprint 103 | Sprint 102 oracle helper patterns; solver evidence template | Sprint 104 benchmark/runtime interpretation, Sprint 107 solver guide, Sprint 109 comparison package |
| Sprint 104 | Sprint 101-103 solver surfaces stable enough to measure; benchmark template | Sprint 105 reorder/runtime evidence, Sprint 107 benchmark docs, Sprint 109 performance calibration |
| Sprint 105 | Sprint 104 runtime contract; benchmark template | Sprint 106 graph/reorder extraction targets, Sprint 107 docs, Sprint 109 reorder/fill calibration |
| Sprint 106 | Day 4 hotspot baseline; Sprints 102-105 touched-family seams | Sprint 107 maintainer/user docs, Sprint 109 maintainability metrics |
| Sprint 107 | Sprints 101-106 technical evidence | Sprint 108 package/platform docs, Sprint 109 public claim audit |
| Sprint 108 | Day 3 package/platform baseline; Sprint 107 user-facing docs | Sprint 109 final validation and support-tier claims |
| Sprint 109 | Sprints 101-108 evidence package | Epic 10 retrospective and post-epic residual queue |

## Claim Dependency Graph

1. **Product model foundation**
   - Sprint 101 must make compressed-first workflows clearer before Sprint
     107 can teach them and Sprint 109 can calibrate them.
2. **Solver evidence expansion**
   - Sprint 102 and Sprint 103 must use Day 9 solver templates before any
     broader oracle claim can be earned.
3. **Performance/runtime interpretation**
   - Sprint 104 must define backend/runtime and benchmark guardrails before
     Sprint 105 expands reorder/fill and large-matrix evidence.
4. **Maintainability evidence**
   - Sprint 106 must compare against Day 4 hotspot metrics and preserve
     source-list/CMake parity.
5. **User-facing coherence**
   - Sprint 107 must wait for enough technical evidence from Sprints 101-106
     to avoid writing aspirational docs.
6. **Package/platform truth**
   - Sprint 108 must decide support tiers before Sprint 109 final claims.
7. **Final calibration**
   - Sprint 109 can only mark claims earned if it can point to implementation,
     artifacts, docs, and validation.

## Minimum Evidence Criteria by Claim Family

| claim family | minimum evidence |
|---|---|
| compressed-first product model | public API/design artifact, implementation, lifecycle/error tests, examples/docs, compatibility-shell non-claim |
| direct solver oracle | fixture taxonomy, external/dense-reference helper, focused tests, tolerance/failure-mode docs |
| iterative/eigensolver/SVD comparison | convergence/eigen/rank fixture taxonomy, residual/tolerance criteria, reference artifacts or deterministic fallback checks |
| backend/runtime | descriptor/selection docs, observability tests, builtin fallback proof, benchmark fields |
| reorder/fill/graph | named matrix fixtures, fill field contract, deterministic guardrails where practical, local timing caveats |
| maintainability | before/after file metrics, extraction boundary docs, source-list parity, focused/full validation as appropriate |
| API/docs/examples | solver-selection guide, compressed-first examples, ownership/error/runtime docs, doc/example validation |
| package/platform | install/export proof, support-tier table, workflow expected counts, staged exclusion register |
| final closeout | full validation package, final claim audit, unsupported-claim cleanup, residual queue, retrospective |

## Non-Goal Guardrail Table

| non-goal | guardrail |
|---|---|
| full linked-list shell replacement | Sprint 101 may centralize compressed-first workflows but must preserve compatibility truth |
| broad complex/mixed precision | do not claim without family-wide implementation and proof |
| GPU/distributed solvers | keep out of Epic 10 unless replanned |
| vendor backend parity | Sprint 104 may improve optional backend observability but not claim parity |
| shared-library ABI guarantee | Sprint 108 must explicitly decide and validate, otherwise static-first remains truth |
| symmetric platform parity | Sprint 108 should publish support tiers, not fake parity |
| portable timing superiority | benchmark templates must keep local timing caveats |
| every-solver-family external validation | Sprint 102-103 can deepen selected families; universal validation remains non-goal unless actually earned |

## Day 8 Conclusion

The Epic 10 claim model is now dependency-aware. Most future claims are
candidate claims, not earned claims. This is intentional: Sprints 101-109 must
earn them with implementation, artifacts, validation, and final calibration.

