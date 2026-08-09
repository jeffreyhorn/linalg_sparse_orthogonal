# Sprint 147 Day 12 Quality Surface Map

## Purpose

Day 12 converts the Sprint 147 evidence gates into a touched-surface validation
map for Sprints 148-156. The map answers two questions before implementation
work starts:

1. Which checks are required because of the files changed?
2. Which supplemental checks are required because a sprint makes a package,
   platform, corpus, freshness, ABI, or external-comparison claim?

Generated evidence and hosted evidence remain separate. Generated local report
rows can support a freshness or oracle claim only when the selected gate requires
them; hosted CI logs are still required for platform-support claims.

## Touched-Surface Validation Table

| Touched surface | Typical files | Validation owner | Required validation | Supplemental validation |
| --- | --- | --- | --- | --- |
| C implementation | `src/*.c`, `tests/*.c`, `benchmarks/*.c`, `examples/*.c` | Solver or implementation owner | Focused build/test for the changed family, then `make format && make lint && make test`. | CMake configure/build/CTest if CMake registration or Windows support is affected; corpus, package, benchmark, or comparison checks when those claims are touched. |
| Public and internal headers | `include/sparse/*.h`, internal helper headers | API owner with solver-owner review | Public declaration review, compatibility review for exported APIs, focused build/test, then `make format && make lint && make test`. | Install/export and downstream consumer checks when public headers or package metadata are affected. |
| Test and corpus proof owners | `tests/test_*_corpus.c`, focused solver tests | Test owner with numerical-owner review | Focused executable build/run plus full C gate for `.c` or `.h` changes. | Corpus schema validation, oracle generation, report-index checks, and hosted Windows CTest count update when a reviewed Windows test is promoted. |
| Scripts | `scripts/*.py`, `tests/*.py`, `*.sh` | Tooling owner | Script-specific syntax/unit command and every documented consumer command for the script. | Report freshness, corpus oracle, package install, or static-deferral checks if the script owns those gates. |
| Make build surface | `Makefile`, included make fragments, source lists | Build owner | Build-target smoke test and relevant changed target. | Full C gate is required when `.c` or `.h` files changed; strongly required for source-list/test-registration changes before claiming Makefile coverage. Package install scripts are required for install target changes. |
| CMake build surface | `CMakeLists.txt`, `cmake/*.in`, install/export metadata | Build/package owner | `cmake -S . -B build`, `cmake --build build`, and relevant `ctest --test-dir build --output-on-failure` subset. | `tests/test_cmake_install.sh` for install/export changes; hosted Windows CMake proof for Windows-reviewed claims. |
| CI workflows | `.github/workflows/*.yml` | CI/platform owner | Static review of workflow logic, expected test counts, job names, permissions, and claim wording. | Hosted run evidence is required before platform promotion. Local tests cannot substitute for hosted Windows/macOS/Linux claim evidence. |
| Package and install metadata | `sparse.pc.in`, CMake package config templates, install tests, INSTALL docs | Package owner | Local install/export tests matching the touched package path. | `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and `bash scripts/static_package_deferral_check.sh` for static-first package claims. |
| Corpus metadata | `tests/corpus/**`, corpus manifests, schemas, expected rows | Corpus maintainer with solver-owner review | Schema validation and source-controlled row review for fixture meaning, tolerance, owner, provenance, and non-claims. | Solver proof-owner test, generated oracle rows, normalized report checks, and freshness checks when claim-bearing generated rows are selected. |
| Report index and normalizer | `tests/corpus/manifests/report_families.tsv`, `scripts/normalize_report_index.py`, report tests | Report owner | Normalizer unit tests and source-controlled report-index checks. | Strict generated freshness checks only for selected generated families; advisory rows do not count as pass evidence. |
| Documentation and public claims | `README.md`, `INSTALL.md`, maintainer guide, tutorials, cookbook, solver-selection docs, headers | Documentation owner with technical reviewer | `git diff --check`, wording scan for unsupported claims, and evidence link review. | Claim-specific proof from the relevant gate before describing platform, package, performance, ABI, external parity, or state-of-practice support. |
| Benchmarks and sentinels | `benchmarks/**`, benchmark scripts, sentinel thresholds, benchmark docs | Benchmark owner | Relevant benchmark command or dry-run plus documentation review of interpretation limits. | Hard-gate sentinel or guardrail checks only when a selected sprint makes them claim-bearing; otherwise they remain advisory. |
| Generated artifacts | Ignored `build/corpus/**`, `build/corpus-reports/**`, `build/bench-reports/**` | Report/tooling owner | Generated rows are reviewed as run output, not source-controlled pass evidence, unless a sprint explicitly defines retention. | Freshness metadata must include command, artifact path, commit, branch, timestamp, platform, compiler, configuration, support tier, status, claim scope, and non-claims. |
| External comparison | Comparison harness, dependency docs, corpus/report rows | Numerical lead with tooling owner | Dependency/version capture, bounded fixture set, comparison metric/tolerance review, and local comparison command. | Corpus schema validation, oracle/report freshness, full C gate for C/H changes, and narrow wording review before any external-comparison claim. |

## Full C Gate Trigger Rules

The full C quality gate is mandatory whenever a change modifies any `.c` or
`.h` file:

```sh
make format && make lint && make test
```

The gate must finish successfully before committing or responding to review
comments. If the gate fails, stop and fix the failure before proceeding; if the
failure is unclear, stop and ask for direction with the failing command and
first actionable diagnostic.

The same gate is strongly required before claim-bearing changes to build
registration, source lists, test registration, public package metadata, or
platform support, even when the immediate diff is in Makefile, CMake, scripts,
or CI only. That stronger rule prevents non-C file changes from silently
changing the compiled test surface.

Documentation-only changes do not require the full C gate unless the docs
change depends on a new or changed executable claim. Documentation changes still
require whitespace validation and claim-evidence review.

## Supplemental Check Map

| Gate | Trigger | Required checks | Evidence boundary |
| --- | --- | --- | --- |
| Windows reviewed CMake | Windows staged-test promotion, CMake test registration, or Windows support wording | Hosted `windows-2022` CMake configure/build/CTest log, reviewed CTest count update, staged exclusion review | Proves only the reviewed Windows CMake subset unless install/downstream is explicitly promoted. |
| Windows install/downstream | Windows CMake package or downstream consumer wording | Hosted supplemental Windows CMake install/downstream proof | Supplemental until explicitly promoted; does not prove Windows Makefile or `pkg-config` parity. |
| Corpus family | QR or partial-SVD maintained corpus rows, expected rows, schema, or proof-owner tests | `python3 scripts/validate_corpus_schema.py`, focused corpus proof executable, oracle command for selected family, normalized report check | Source-controlled rows prove coverage intent; solver pass requires executable proof. |
| Generated freshness | Required-generated report family selected by a sprint | Normalizer tests, selected `scripts/run_corpus_oracle.py` command, `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` or sprint-specific equivalent | Fresh generated rows support only the named family and claim scope. |
| Static package | Install targets, `pkg-config`, CMake package metadata, static-first docs | `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, `bash scripts/static_package_deferral_check.sh` | Proves static archive install/export only; no shared ABI or package-manager claim. |
| Shared ABI decision | Any attempt to support shared libraries, loader behavior, import/export macros, or ABI compatibility | Product decision record, symbol/export policy proof, shared build/install/export/downstream/loader checks, hosted platform proof | No shared-library support exists until every selected platform has direct proof. |
| External comparison | External-library comparison harness, report rows, docs, or claim wording | Named dependency/version/install method, bounded fixture command, metric/tolerance review, corpus schema validation, report normalization, full C gate if C/H changed | Supports only the named library, version, fixture set, metric, tolerance, and platform. |
| Public claim freeze | README, INSTALL, benchmark docs, support docs, tutorials, public headers | Claim wording scan and evidence reference check | Absence of evidence requires explicit non-claim or residual wording. |
| Benchmark/sentinel | Benchmark command, threshold, report, or performance wording | Relevant benchmark/sentinel command and normalized report if selected | Advisory rows cannot support performance claims; hard gates require selected thresholds and repeatable evidence. |

## Stop-Condition Register

Stop and ask for direction when any of these conditions occurs:

| Stop condition | Why it stops the sprint |
| --- | --- |
| `make format && make lint && make test` fails after `.c` or `.h` changes. | Code-quality policy blocks commit and review replies. |
| A required supplemental check fails. | Claim-bearing evidence is missing or contradicted. |
| Review feedback is ambiguous or conflicts with an existing support boundary. | Guessing can widen unsupported claims or change product posture. |
| Hosted Windows, macOS, or Linux evidence is required but unavailable. | Local proof cannot substitute for hosted platform claims. |
| CTest expected-count drift is not explained by a reviewed test promotion or removal. | Platform CI can silently lose or gain coverage. |
| Generated rows are stale, missing, or advisory but are needed as pass evidence. | Freshness and pass evidence would be overstated. |
| Corpus rows lack owner, provenance, tolerance, comparison kind, or non-claim scope. | Maintained corpus coverage would be unverifiable. |
| Package metadata implies shared libraries, ABI compatibility, selectors, or package-manager support without a product decision. | Static-first contract would be broken. |
| External comparison lacks exact dependency/version, bounded fixture set, or metric/tolerance definition. | External parity or state-of-practice wording would be unsupported. |
| Documentation states platform, package, performance, ABI, external parity, or state-of-the-art support without direct evidence. | Public claims would exceed implemented proof. |
| Generated artifacts are proposed for commit without an explicit retention policy. | Source-controlled evidence boundaries would be unclear. |

## Sprint 156 Validation Package Seed

Sprint 156 should close Epic 13 with a validation package containing:

| Package component | Minimum expectation |
| --- | --- |
| Code-quality result | Final `make format && make lint && make test` result for all Epic 13 C/H changes, or explicit note that the final change set is docs-only. |
| Corpus proof | Schema validation, selected QR and partial-SVD proof-owner tests, oracle commands, normalized report checks, and freshness status for required-generated rows. |
| Platform proof | Hosted Windows, macOS, and Linux evidence records for every promoted support claim, with Windows reviewed and supplemental lanes kept distinct. |
| Package proof | Static install/export, CMake package, `pkg-config`, downstream consumer, uninstall, and static-deferral checks. |
| External comparison proof | Named external dependency, version, install method, bounded fixture set, metric/tolerance, command output, report row, and wording boundary. |
| Public claim audit | README, INSTALL, benchmark docs, maintainer guide, tutorial/cookbook, solver-selection docs, and public headers scanned for unsupported widened claims. |
| Residual register | Explicit unresolved items with support tier, reason, owner, and follow-up recommendation, without using residuals as completed evidence. |

## Day 13 Handoff

Day 13 should use this map to audit public and support surfaces before
implementation sprints begin. The audit should classify each claim as supported,
explicit non-claim, needs fix, or residual, and should apply documentation fixes
only when the missing evidence is clear.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every planned touched surface has a validation owner. | Complete | The touched-surface table assigns owners for code, headers, scripts, build, CI, package, corpus, report, docs, benchmarks, generated artifacts, and external comparison. |
| C/header changes require the full quality gate. | Complete | The full C gate section makes `make format && make lint && make test` mandatory for `.c` and `.h` changes. |
| Generated and hosted evidence requirements stay separate. | Complete | The supplemental map and stop conditions distinguish generated freshness rows from hosted platform CI proof. |
