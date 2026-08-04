# Sprint 137 Day 11 - Quality Surface Map

## Purpose

Day 11 maps required validation by touched surface for Epic 12 implementation
sprints. Later sprints should select checks from this map based on what they
change, then add focused checks for the specific solver, package, report, or
platform behavior under review.

The map separates required local proof, supplemental confidence, and hosted CI
evidence. Supplemental or hosted-only checks cannot be reported as local proof
unless they actually ran in the required environment.

## Baseline Rule

Every change must run the smallest required quality set for its touched
surfaces. When a change touches multiple surfaces, run the union of required
checks.

If any `.c` or `.h` file changes, the full C quality chain is required:

```bash
make format && make lint && make test
```

Focused tests can run before the full chain, but they do not replace it for
C/header changes.

## Touched-Surface Quality Matrix

| Touched surface | Required checks | Supplemental checks | Notes |
| --- | --- | --- | --- |
| Sprint planning docs under `docs/planning/**` only | `git diff --check`; trailing whitespace scan for touched planning directory; focused Markdown local link/path validation for the affected planning epic. | Broader docs link scan if paths are moved. | Does not require C quality gates unless code or public claim surfaces also change. |
| Public docs: `README.md`, `INSTALL.md`, `docs/*.md`, `benchmarks/README.md` | `git diff --check`; focused Markdown local link/path validation; claim-boundary scan for state-of-the-art, platform, package, ABI, performance, corpus, report, and external parity wording. | Build examples or package proof when docs describe changed commands. | Public wording cannot widen support without matching evidence rows and validation. |
| Examples under `examples/**` | `git diff --check`; compile affected example where feasible; `make examples-build` when examples are broadly touched. | Run example binaries when they are part of an adoption or package claim. | If `.c` examples change, full C quality chain is required by baseline rule. |
| Benchmarks under `benchmarks/**` | `git diff --check`; `make bench-build`; focused benchmark compile/run for changed benchmark. | `make bench-fast`; `make bench-canonical-report`; full `make bench` only when explicitly justified. | Benchmark rows remain local unless cross-platform report proof exists. |
| Tests under `tests/**` | Focused test binary or script; update Make/CMake test registration if needed; full C quality chain if `.c` or `.h` changed. | `make quality-review`; `make quality-review-cmake` if CMake registration changes. | Test additions must preserve Make/CMake ownership where applicable. |
| Library source under `src/**` | Full C quality chain; focused solver tests for touched behavior. | `make quality-review`; `make quality-review-full`; sanitizer/TSan/OpenMP lanes when risk warrants. | Source changes require implementation-specific proof before public claims. |
| Public headers under `include/**` | Full C quality chain; docs/API claim scan; examples or downstream proof when install/API behavior changes. | CMake/package install checks for exported header changes. | Public header changes can affect package and ABI claims even when implementation is untouched. |
| Private headers under `src/**` or internal helpers | Full C quality chain; focused solver tests. | Source-list/CMake parity checks if helper ownership changes. | Keep helper movement tied to selected proof owners. |
| Makefile, source lists, or generated version plumbing | `git diff --check`; `make source-list-check`; `make format-check`; `make lint`; relevant build target; package/version proof if install/version behavior changes. | `make quality-review-compile`; `make quality-review-full` for broad build ownership changes. | If source membership changes, verify Make and CMake parity. |
| CMake files under `CMakeLists.txt` or `cmake/**` | `git diff --check`; `make quality-review-cmake` or focused CMake configure/build/CTest path; package proof if install/export changes. | Hosted Windows/macOS CMake lanes when platform support is claimed. | Local CMake success does not prove hosted platform promotion. |
| pkg-config template `sparse.pc.in` | `git diff --check`; `bash tests/test_install.sh`; exact-version and downstream pkg-config proof. | macOS supplemental install/pkg-config hosted lane if macOS package wording changes. | Normalize path and whitespace behavior before relying on `pkg-config --cflags` or `--libs`. |
| Install scripts `tests/test_install.sh`, `tests/test_cmake_install.sh` | Run the touched install script; run paired install path if shared metadata changes; `git diff --check`. | Linux package-contract CI; macOS supplemental package lanes; Windows CMake install/downstream hosted lane. | Package proof rows prove only their package mode and support tier. |
| Static package deferral script | `bash scripts/static_package_deferral_check.sh`; affected package docs claim scan. | Linux package-contract CI. | Required when shared/ABI non-claim boundaries are touched. |
| Report generators under `scripts/*report*`, `scripts/performance_sentinels.sh`, `scripts/large_matrix_guardrails.sh` | Script syntax check where applicable; run the focused report command when feasible; `git diff --check`; report non-claim scan. | `make bench-canonical-report`; `make performance-sentinels`; `make large-matrix-guardrails`; `make deadcode-report`; `make deadcode-check` depending on generator. | Report changes must preserve row meanings and freshness fields. |
| Python scripts | `python3 -m py_compile <script>` for touched scripts; focused command if script is executable in local environment. | Full report command using script outputs. | Do not rewrite generated artifacts unless the sprint calls for regenerated evidence. |
| Shell scripts | `bash -n <script>` for touched scripts; focused command if safe and feasible. | Package/report CI lanes for shell scripts that drive install or reports. | Preserve macOS/Linux shell portability when scripts are used by both. |
| CI workflows under `.github/workflows/**` | `git diff --check`; YAML structural review; command/support-tier claim scan; local command validation where feasible. | Hosted workflow run evidence from GitHub Actions. | Hosted CI proof is required before platform support promotion. |
| Corpus manifests, oracle rows, or generated report indexes | Schema/field validation if available; `git diff --check`; report/corpus non-claim scan. | Regenerate corpus/oracle/report indexes when implementation exists. | Skips and stale rows cannot be counted as solver passes. |

## Required Command Map

| Change trigger | Required command or action |
| --- | --- |
| Any `.c` or `.h` change | `make format && make lint && make test` |
| C/header change with CMake registration or build-system impact | Full C quality chain plus `make quality-review-cmake` |
| Source-list ownership change | `make source-list-check` plus affected Make/CMake build checks |
| Broad source or test ownership change | `make quality-review` |
| Broad Make plus CMake behavior change | `make quality-review-full` |
| Public docs or planning docs only | `git diff --check` plus focused Markdown local link/path validation |
| Public claim wording change | Claim-boundary scan plus evidence rows and validation commands for the claim |
| `Makefile` package install behavior | `bash tests/test_install.sh`; package claim scan |
| CMake install/export behavior | `bash tests/test_cmake_install.sh`; CMake package claim scan |
| `sparse.pc.in` change | `bash tests/test_install.sh` |
| `cmake/SparseConfig.cmake.in` or CMake package target change | `bash tests/test_cmake_install.sh` |
| Static/shared package boundary change | `bash scripts/static_package_deferral_check.sh` |
| Python script change | `python3 -m py_compile <script>` plus focused script command |
| Shell script change | `bash -n <script>` plus focused script command |
| Dead-code report generator change | `python3 -m py_compile scripts/deadcode_report.py`; `make deadcode-report`; `make deadcode-check` when tool dependencies are available |
| Benchmark report generator change | `bash -n scripts/bench_canonical_report.sh`; `make bench-canonical-report` when benchmark runtime is feasible |
| Performance sentinel script change | `bash -n scripts/performance_sentinels.sh`; `make performance-sentinels` when feasible |
| Large-matrix guardrail script change | `bash -n scripts/large_matrix_guardrails.sh`; `make large-matrix-guardrails` when feasible |
| Workflow change | Local command validation where possible plus hosted CI follow-up for affected lane |

## Supplemental Command Map

| Supplemental check | When to use | Boundary |
| --- | --- | --- |
| `make sanitize` | Source changes with undefined-behavior risk or Apple Clang/Linux sanitizer lane relevance. | Supplemental unless the sprint names sanitizer as required. |
| `make asan` | Memory-risk source changes. | Supplemental local confidence. |
| `make sanitize-all` | High-risk source changes where combined ASan/UBSan is feasible. | Supplemental local confidence. |
| `make sanitize-thread` | Threading/eigensolver/OpenMP-adjacent changes where local TSan runtime works. | Supplemental; hosted Linux TSan may carry separate evidence. |
| `make omp` | OpenMP build/runtime changes. | Supplemental and platform/toolchain-sensitive. |
| `make bench-fast` | Runtime-sensitive changes needing quick performance smoke. | Supplemental local signal, not portable performance proof. |
| `make bench-canonical-report` | Benchmark/report changes needing maintained canonical snapshot. | Threshold-free local report evidence. |
| `make performance-sentinels` | Sentinel/report/runtime governance changes. | Local sentinel evidence with documented thresholds only where explicit. |
| `make large-matrix-guardrails` | Large-matrix/report/guardrail changes. | Structural and bounded report evidence, not broad scale claim. |
| `make coverage` | Coverage/report or final evidence work. | Supplemental tree-mutating coverage signal, not behavioral completeness. |
| `make deadcode-report` and `make deadcode-check` | Dead-code report, source ownership, or closeout work. | Report completeness and triage evidence, not removal-ready proof. |
| `make examples-build` | Adoption, public docs, or example surface changes. | Compile-only example confidence unless example execution is also required. |

## Hosted-CI Dependency Notes

| Hosted lane | Required for | Boundary |
| --- | --- | --- |
| Linux package-contract CI | Reviewed Linux static-first package proof. | Local install scripts are useful before push but hosted pass is the reviewed package lane. |
| Linux CMake CI | Reviewed CMake parity path. | Local CMake proof does not replace hosted failure triage for CI changes. |
| Linux dead-code CI | Enforced dead-code report generation and completeness. | Dead-code rows remain triage evidence. |
| macOS Apple Clang CI | Reviewed macOS source/build quality lane. | Does not imply reviewed macOS install/export parity. |
| macOS install/pkg-config and CMake install/export jobs | Supplemental macOS package confidence. | Cannot be reported as reviewed macOS install/export parity until a platform promotion sprint earns it. |
| Windows CMake build/test CI | Reviewed Windows CMake consumer subset. | Does not prove Makefile, POSIX/pthread, or package-manager parity. |
| Windows CMake install/downstream CI | Selected Sprint 144 platform promotion candidate. | Only hosted pass plus docs/support-tier updates can promote this exact lane. |
| Linux sanitizer/TSan/coverage/benchmark jobs | Supplemental runtime, race, coverage, and benchmark evidence. | Do not widen behavioral completeness or portable performance claims. |

## Stop Conditions

- A `.c` or `.h` change cannot pass `make format && make lint && make test`.
- A build-system change alters Make/CMake source ownership but does not run
  source-list or CMake parity checks.
- A package change cannot pass the relevant install/downstream proof.
- A public claim changes without evidence rows, validation commands,
  support-tier wording, and non-claims.
- A platform promotion relies on local evidence where hosted CI evidence is
  required.
- A report/schema change flattens row meanings or treats stale report evidence
  as a numerical failure.
- A skipped optional-data row is counted as solver pass evidence.
- Coverage percentage is treated as behavioral completeness.
- Dead-code output is treated as removal-ready proof without owner/API review.
- Benchmark or sentinel output is treated as portable performance evidence.
- Shared-library, ABI, loader, or package-manager support is implied by
  static-first package changes.

## Day 11 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Later sprints can select validation from touched surfaces. | Complete | Touched-surface quality matrix and required command map define checks by changed file family and claim type. |
| `.c`/`.h` changes clearly require the full C quality chain. | Complete | Baseline rule and required command map require `make format && make lint && make test` for every C/header change. |
| Supplemental and hosted-CI-only checks are not treated as local proof. | Complete | Supplemental command map and hosted-CI dependency notes define evidence boundaries and platform-promotion requirements. |
