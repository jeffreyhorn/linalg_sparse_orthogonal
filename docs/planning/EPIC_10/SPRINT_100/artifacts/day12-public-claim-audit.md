# Sprint 100 Day 12 Public Claim Audit

## Purpose

Day 12 audits the live public and support surfaces against the Sprint 100 claim
map and the Day 9-11 evidence templates. The goal is to prevent unsupported
state-of-the-art, package, performance, platform, or comparison claims from
flowing into Sprints 101-109 as if they were already earned.

## Audited Surfaces

| surface | role |
|---|---|
| `README.md` | project front door, capability summary, workflow chooser, compact quality/install story |
| `INSTALL.md` | operational setup, install/export behavior, platform notes, installed-consumer validation |
| `benchmarks/README.md` | benchmark command groups, CSV schemas, measurement interpretation |
| `examples/README.md` | example selection and example-local workflow guidance |
| `docs/tutorial.md` | longer user learning path |
| `docs/algorithm.md` | algorithm details, historical measurement context, tuning caveats |
| `docs/matrix_market.md` | Matrix Market format support |
| `docs/maintainer_guide.md` | support-surface ownership, quality-contract interpretation, package/platform policy |
| `include/*.h` | API-local contracts, warnings, and call-site caveats |

Generated API HTML and historical planning artifacts were not treated as live
public claim sources for this audit. They are either generated from headers or
historical evidence records.

## Claim Classification

| public/support claim | surface | state | evidence or reason |
|---|---|---|---|
| strongest local reviewed baseline is `make quality-review-full` | `README.md`, `docs/maintainer_guide.md` | supported | Day 2 ran `make quality-review-full`; Day 8 marks the reviewed baseline earned |
| CMake parity is a reviewed path with `ctest -N` and full `ctest` | `README.md`, `docs/maintainer_guide.md`, `Makefile` references | supported | Day 2 recorded Make/CMake test-count parity and CTest execution |
| static-first package surface is maintained | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | supported | Day 3 records Make and CMake install proof surfaces; Day 11 templates preserve static-first fields |
| `pkg-config` and `find_package(Sparse)` describe installed static archive consumers | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | supported | `tests/test_install.sh` and `tests/test_cmake_install.sh` own local Unix-side proof |
| shared-library packaging is intentionally deferred | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | supported non-claim | Day 8 marks shared-library/ABI support blocked/stretch; Day 11 ABI template requires an explicit decision |
| Linux is the broadest reviewed source of truth | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | supported | Day 3 platform draft and workflow comments identify Linux reviewed + supplemental scope |
| macOS reviewed support is narrower with supplemental GCC/install confidence | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | supported | `.github/workflows/macos-ci.yml` separates Apple Clang reviewed path from supplemental lanes |
| Windows support is the reviewed CMake-first MSVC subset | `README.md`, `INSTALL.md`, `.github/workflows/windows-ci.yml`, `docs/maintainer_guide.md` | supported | Day 3 records Windows expected CTest count `51`; Day 11 platform template requires exclusions |
| Windows Makefile or Windows install-validation parity | public docs | blocked/non-claim | Current wording avoids this claim; Day 8 marks it blocked/stretch |
| benchmark rows are branch-local/local measurement artifacts, not portable guarantees | `README.md`, `benchmarks/README.md` | supported | Day 10 benchmark template and pilot encode threshold-free report semantics |
| `make bench-canonical-report` is threshold-free and not pass/fail timing | `README.md`, `benchmarks/README.md` | supported | `scripts/bench_canonical_report.sh` manifest explicitly carries this note |
| `bench-fast` is a bounded PR-time runtime signal | `README.md`, `Makefile`, `benchmarks/README.md` | supported | Day 5 benchmark baseline and Day 10 template classify it as runtime lane, not portable performance |
| `wall-check` is a bounded runtime regression sentinel | `Makefile`, `docs/algorithm.md`, `benchmarks/README.md` | supported with caveat | Threshold/baseline owner exists; future claims should fill Day 10 sentinel template |
| broad portable performance superiority | public docs | non-goal | Current README/benchmark docs avoid this claim; Day 8 marks portable timing superiority a guardrail |
| external dense-reference proof exists for selected Cholesky/LDLT lanes | maintainer/planning context, tests | supported but narrow | Day 5 identifies Cholesky CSC and LDLT CSC maintained external dense-reference lanes |
| every solver family has external oracle comparison | public docs | non-goal/currently unsupported | Day 8 marks universal external validation a non-goal unless actually earned |
| direct/iterative/eigensolver/SVD comparison evidence will deepen in Epic 10 | Epic 10 planning | candidate | Day 8 maps Sprint 102-103 owners; Day 9 template defines proof fields |
| compressed-first workflows are primary product path | README workflow chooser | candidate/partially supported | README already names compressed-first one-shot entry, but Day 8 keeps full product-model claim for Sprint 101 |
| mutable linked-list shell remains supported compatibility path | README, headers | supported but needs ongoing wording discipline | Current docs preserve one-shot and mutable shell support; Day 8 calls for clearer secondary wording in Sprint 101/107 |
| broad complex or generic scalar support | README, headers, maintainer guide | non-claim | Current wording explicitly says real-only double remains the shipped contract |
| broad 64-bit index maturity | README, headers, maintainer guide | candidate/non-claim | Current wording limits `SPARSE_IDX_BITS=64` to a compile-time seam and preserves 32-bit reviewed default |
| package-manager ecosystem support | public docs | unsupported/non-claim | Day 3 lists package-manager integration as a non-claim; public docs do not claim recipes |
| state-of-the-art replacement library | public docs | non-goal | No live public surface claims this; Day 8 marks broad replacement claim non-goal |

## Supported Claims

These claims are safe to carry into later sprint planning as already earned,
provided later code/docs changes keep the same proof surfaces green:

- strongest local reviewed baseline: `make quality-review-full`;
- reviewed CMake parity surface;
- static-first install/export package story;
- local Unix-side Make install/`pkg-config` proof ownership;
- local Unix-side CMake install/`find_package(Sparse)` proof ownership;
- tiered Linux/macOS/Windows support wording;
- threshold-free canonical benchmark report interpretation;
- real-only double scalar contract;
- default reviewed 32-bit index lane.

## Candidate Claims

These claims should remain future work until their owning sprint fills the
relevant Day 9-11 template and validates the result:

| candidate | likely owner | required evidence |
|---|---|---|
| compressed-first workflow as the primary product path | Sprint 101 | API/design artifact, implementation follow-through, lifecycle tests, docs/examples |
| broader direct solver external oracle evidence | Sprint 102 | fixture taxonomy, oracle helper, focused tests, tolerance and non-claim fields |
| iterative/eigensolver external comparison architecture | Sprint 103 | convergence/eigenpair fixture taxonomy, residual criteria, validation commands |
| decision-grade local performance sentinels | Sprint 104 | filled performance-sentinel template, baseline, machine-class and threshold rationale |
| reorder/fill and graph evidence clarity | Sprint 105 | named fixtures, fill metric contract, local timing caveats |
| maintainability risk reduction | Sprint 106 | before/after metrics, extraction artifacts, source-list/CMake parity |
| clearer public solver-selection guidance | Sprint 107 | technical evidence from Sprints 101-106 plus doc/example validation |
| explicit platform tier publication | Sprint 108 | filled platform-tier template, expected counts, staged exclusions |
| shared-library or ABI support | Sprint 108 stretch | explicit ABI decision plus install/export/runtime-loader proof |

## Unsupported or Blocked Claims

These claims should not be made in public docs unless future sprints add the
required proof:

- broad state-of-the-art replacement for established sparse linear algebra
  packages;
- universal external oracle validation across every solver family;
- portable performance superiority;
- vendor backend parity;
- GPU or distributed solver support;
- broad complex or mixed-precision support;
- stable dynamic ABI guarantee;
- shared-library package maturity;
- Windows Makefile parity;
- Windows reviewed install-validation parity;
- symmetric Linux/macOS/Windows reviewed support parity;
- package-manager ecosystem integration.

## Candidate Wording-Change Queue

No immediate Day 12 wording fix is required, but these areas should be watched
or tightened during their owning sprints:

| wording area | current reading | recommended future action |
|---|---|---|
| README "benchmarks prove the retained workflow/performance story" | acceptable when read with adjacent benchmark caveats | Sprint 107 can soften to "provide measurement evidence for" if public benchmark docs are rewritten |
| README compressed-first workflow wording | already present but not yet the full primary product claim | Sprint 101 should align wording with implementation and lifecycle evidence |
| algorithm guide historical "production default" measurements | useful historical algorithm context, but dense with old sprint provenance | Sprint 105 or Sprint 107 can add a short front-matter caveat that benchmark docs own current performance interpretation |
| public header `ABI break` warnings | API-local release notes, not a stable ABI promise | Sprint 108 should decide whether to keep this wording, rename it to source/API compatibility, or add explicit ABI proof |
| platform support table | currently tiered and accurate | Sprint 108 should convert this to a first-class support tier artifact with expected counts and exclusion register |

## Immediate Fix Recommendation List

Day 12 does not recommend immediate public-doc edits. The current live surfaces
already avoid the major unsupported claims identified by the Sprint 100 claim
map.

If Day 13 chooses to make a small closeout polish pass, the safest candidates
are documentation-only:

1. Add a one-sentence pointer in `docs/algorithm.md` that current benchmark
   command interpretation lives in `benchmarks/README.md`.
2. Optionally soften "benchmarks prove" wording in `README.md` to "benchmarks
   provide measurement evidence for".
3. Leave package/platform wording unchanged until Sprint 108, because it is
   currently accurate and deliberately tiered.

## Validation Path for Any Immediate Fix

If Day 13 edits only documentation:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_100 README.md INSTALL.md docs benchmarks examples include
```

If Day 13 changes public headers or any `.c` / `.h` source surface, run the
full required C quality chain:

```sh
make format && make lint && make test
```

## Day 12 Conclusion

The public/support surfaces are mostly aligned with the Sprint 100 evidence
contract. The main risk is not an existing false claim; it is future sprint
work accidentally promoting candidate or blocked claims without filling the
Day 9-11 evidence templates. Day 13 should integrate this audit into the Sprint
100 handoff package so Sprints 101-109 inherit explicit claim boundaries.
