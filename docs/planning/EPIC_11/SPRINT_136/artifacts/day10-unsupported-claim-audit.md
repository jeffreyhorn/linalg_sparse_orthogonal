# Sprint 136 Day 10 Unsupported-Claim Audit

## Purpose

Day 10 scanned public, maintainer, benchmark, package, and Sprint 136 planning
surfaces for claim drift before cleanup work begins on Day 11. The audit used
the Day 8 competitive evidence baseline and Day 9 claim recalibration as the
source of truth.

## Audit Scope

| Surface | Files checked | Result |
|---|---|---|
| Front-door and install docs | `README.md`, `INSTALL.md` | No required public-doc cleanup found. High-risk package, platform, benchmark, and shared-library wording is already fenced. |
| Adoption and solver docs | `docs/solver_selection.md`, `docs/cookbook.md`, `examples/README.md` | No unsupported solver superiority or broad parity claim found. Solver-selection wording remains workflow-oriented. |
| Algorithm docs | `docs/algorithm.md`, `docs/algorithm_history.md` | No required cleanup found. Performance-sensitive language is scoped to algorithm behavior, historical captures, fixtures, or local evidence. |
| Benchmark docs | `benchmarks/README.md` | No required cleanup found. Benchmark reports are repeatedly described as local measurement artifacts, not portable guarantees. |
| Maintainer docs | `docs/maintainer_guide.md` | No required cleanup found. Oracle, package, platform, coverage, dead-code, and report boundaries remain explicit. |
| Sprint 136 artifacts | `docs/planning/EPIC_11/SPRINT_136/` | No claim drift found. Day 1-9 artifacts preserve local, supplemental, staged, deferred, and unsupported distinctions. |

## Scan Evidence

The audit used targeted claim-drift scans over the public/support docs and
Sprint 136 artifacts:

- competitive overclaim terms:
  `state.?of.?the.?art|best.?in.?class|outperform|fastest|superior`
- performance overclaim terms:
  `portable performance|performance guarantee|benchmark.*guarantee|scalab|memory|speedup|faster|universal reorder|universal fill`
- package/platform overclaim terms:
  `shared.?librar|dynamic ABI|ABI compat|runtime.?loader|package.?manager|homebrew|apt|rpm|dnf|pacman|vcpkg|conan|BUILD_SHARED_LIBS|install.?parity|platform parity|reviewed .*Windows|reviewed .*macOS`
- coverage/external-parity overclaim terms:
  `complete|comprehensive|all solver|every solver|full .*coverage|coverage completeness|normalized cross-report|release proof|correctness proof|SuiteSparse corpus|optional-data|external oracle|SciPy|PETSc|Trilinos|Eigen|ARPACK`

The positive hits fell into three acceptable categories:

1. explicit non-claims and support-tier fences;
2. dependency/prerequisite instructions, not package-manager support claims;
3. local, fixture-specific, historical, or algorithm-context wording.

## Package And Platform Findings

| Finding | Evidence | Day 10 decision |
|---|---|---|
| Static-first install/export remains the maintained package surface. | `README.md` and `INSTALL.md` describe `pkg-config`, CMake exports, exact package versioning, and the static archive contract. | Keep. This is an earned claim under Sprint 133-134 and Day 5-7 validation. |
| Shared-library packaging and dynamic ABI support remain deferred. | `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` state that `BUILD_SHARED_LIBS=ON` is rejected and shared-library/runtime-loader support is deferred. | Keep. Non-claim is explicit and findable. |
| Package-manager terms appear only in prerequisite commands and non-claims. | `INSTALL.md` mentions `apt`, `dnf`, and Homebrew for toolchain prerequisites; package-manager support is separately listed as a non-claim. | Keep. No package-manager support claim found. |
| Platform support remains tiered. | `README.md` and `INSTALL.md` distinguish Linux reviewed package-contract ownership from macOS/Windows supplemental install/downstream confidence. | Keep. No reviewed macOS/Windows install parity claim found. |
| Windows staged-test boundaries remain explicit. | `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` keep pthread/POSIX-backed Windows staged tests outside the reviewed subset. | Keep. No staged-to-reviewed promotion found. |

## Performance And Report Findings

| Finding | Evidence | Day 10 decision |
|---|---|---|
| Benchmark reports are local measurement artifacts. | `benchmarks/README.md` states that benchmarks do not prove portable performance and that generated indexes are navigation and interpretation aids only. | Keep. No portable performance guarantee found. |
| README benchmark wording is bounded. | `README.md` describes canonical reports and performance sentinels as branch-local or bounded runtime signals. | Keep. No competitive performance claim found. |
| Algorithm speed and memory wording is contextual. | `docs/algorithm.md` uses phrases such as "converge faster" and memory bounds inside algorithm explanation and fixture contexts; `docs/algorithm_history.md` marks historical captures as local evidence. | Keep. No public portability or superiority claim found. |
| Generated report metadata remains non-claim evidence. | Sprint 136 validation metadata says generated rows do not create portable timing, scalability, memory, backend parity, platform parity, broad correctness, or state-of-the-art claims. | Keep. Non-claim is explicit and findable. |

## Competitive And Parity Findings

| Finding | Evidence | Day 10 decision |
|---|---|---|
| State-of-the-art wording is fenced. | Public hits are limited to `docs/solver_selection.md` non-claim wording and maintainer guidance against state-of-the-art proof claims. Sprint artifacts use it as a non-claim register. | Keep. No unqualified state-of-the-art claim found. |
| External library parity remains bounded. | `docs/maintainer_guide.md` keeps QR, SVD, iterative, and eigensolver helper evidence fixture-specific and explicitly rejects broad LAPACK, NumPy, SciPy, PETSc, Trilinos, Eigen, ARPACK, and ecosystem parity claims. | Keep. No broad external parity claim found. |
| Coverage and dead-code boundaries remain intact. | `docs/maintainer_guide.md` describes coverage as supplemental and dead-code as report-completeness evidence, not zero-findings or removal-ready proof. | Keep. No coverage-completeness claim found. |
| Report-index boundaries remain intact. | `benchmarks/README.md`, `docs/algorithm_history.md`, and Sprint 136 validation files keep report indexes as freshness/navigation evidence. | Keep. No normalized cross-report correctness or release proof claim found. |

## Cleanup Priority List

| Priority | Item | Owner surface | Day 11 action |
|---|---|---|---|
| P0 | Required unsupported public-doc cleanup | Public docs | None identified on Day 10. |
| P1 | Verify package/platform support-tier wording still matches Sprint 133-134. | `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Re-scan and record no-op or focused wording edits if new drift appears. |
| P1 | Verify benchmark/report wording remains local and freshness-scoped. | `README.md`, `benchmarks/README.md`, `docs/algorithm_history.md` | Preserve local-measurement fences; avoid portable performance wording. |
| P1 | Verify competitive and external-parity wording remains bounded. | `docs/solver_selection.md`, `docs/maintainer_guide.md`, Sprint 136 closeout artifacts | Preserve explicit non-claims for state-of-the-art, ecosystem parity, and broad solver-family coverage. |
| P2 | Keep algorithm explanatory wording contextual. | `docs/algorithm.md` | No edit required unless Day 11 decides to add an extra cross-link to benchmark interpretation. |

## Completion Criteria

| Criterion | Status | Evidence |
|---|---|---|
| Unsupported or ambiguous wording is located before edits begin. | Complete | Targeted scans and surface review found no P0 public-doc blockers; P1 high-risk surfaces are queued for Day 11 verification. |
| Cleanup scope is bounded to wording, links, and claim fences unless code changes are explicitly required. | Complete | Day 10 found no code changes and no behavior changes required. |
| Non-claims remain explicit and findable. | Complete | Shared-library, dynamic ABI, runtime-loader, package-manager, platform parity, portable performance, external parity, state-of-the-art, coverage-completeness, and report-index non-claims remain present in owner surfaces. |
