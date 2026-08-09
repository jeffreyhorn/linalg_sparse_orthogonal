# Day 4 Final Validation Baseline Design

## Scope

Day 4 defines the strongest feasible final local validation package for Sprint
146 before running it. The baseline is derived from the Day 2 numerical
evidence inventory and the Day 3 support evidence inventory. It separates
required local checks, conditional checks, optional generated-evidence refresh
commands, hosted-CI-only evidence, skip rationale, pass/fail capture format,
and stop conditions.

## Validation Principles

1. Every claimed surface must have a validation command, source-controlled
   owner, hosted CI lane, or explicit non-claim/residual entry.
2. Generated local rows under `build/` are reproducibility evidence for the
   command, commit, platform, compiler, and configuration that created them;
   they are not source-controlled pass proof.
3. Hosted CI support-tier claims require hosted logs or PR checks, not local
   report-index rows.
4. C and public-header changes trigger the full local C quality gate:
   `make format && make lint && make test`.
5. Documentation-only changes require Markdown hygiene and claim-boundary
   scans, with heavier checks selected when the documentation cites package,
   report, corpus, platform, runtime, or adoption proof.
6. Failed required checks stop Day 5 until fixed or explicitly reclassified as
   an environment constraint with no claim promotion.

## Required Day 5 Local Baseline

| Order | Command | Evidence Family | Required Because | Pass Capture |
| ---: | --- | --- | --- | --- |
| 1 | `python3 scripts/validate_corpus_schema.py` | corpus, report | Corpus/report schemas and report-family metadata are core Epic 12 evidence contracts. | Record exit code and any diagnostic lines. |
| 2 | `python3 tests/test_normalize_report_index.py` | report | Normalized report index behavior and freshness diagnostics are claimed surfaces. | Record test pass/fail summary. |
| 3 | `python3 scripts/normalize_report_index.py --no-generated --check` | report | Source-controlled report rows must normalize deterministically without local generated artifacts. | Record row count and exit code. |
| 4 | `python3 scripts/normalize_report_index.py --check` | report | Generated-aware default normalization must remain valid without treating missing generated rows as failures. | Record row count, missing-generated diagnostics, and exit code. |
| 5 | `python3 scripts/normalize_report_index.py --check-freshness` | report | Freshness diagnostics must remain coherent for current local generated state. | Record freshness summary and any advisory/stale rows. |
| 6 | `python3 scripts/normalize_report_index.py --family documentation --family package --family ci --family runtime_backend --check` | report, package, platform, adoption, runtime/backend | Day 3 support evidence depends on these source-controlled advisory/lane rows. | Record selected row count and exit code. |
| 7 | `python3 scripts/normalize_report_index.py --family documentation --family package --family ci --family runtime_backend --check-freshness` | report, package, platform, adoption, runtime/backend | Support-tier row freshness must not imply generated proof. | Record selected freshness summary and exit code. |
| 8 | `bash scripts/static_package_deferral_check.sh` | package, ABI | Static-first package posture and shared-library deferral are final closeout constraints. | Record script pass lines. |
| 9 | `bash tests/test_install.sh` | package, adoption | Make install, `pkg-config`, downstream consumer, version, uninstall, and static archive checks support public install docs. | Record pass/fail summary. |
| 10 | `bash tests/test_cmake_install.sh` | package, adoption | CMake install/export, downstream example, exact-version, mismatch-version, and unsupported metadata checks support package docs. | Record pass/fail/skip summary. |
| 11 | `make examples-build` | adoption | Maintained examples are part of the first-use adoption surface. | Record built example count or make summary. |
| 12 | `make build/test_qr_corpus && ./build/test_qr_corpus` | QR, corpus | QR fixture-local claim has a focused compiled proof owner. | Record test count, failure count, assertion count, and residual line if printed. |
| 13 | `make build/test_svd_partial_corpus && ./build/test_svd_partial_corpus` | partial-SVD, corpus | Partial-SVD fixture-local claim has a focused compiled proof owner. | Record test count, failure count, assertion count, and status checks. |
| 14 | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | corpus, QR, partial-SVD, report | Local oracle/report rows refresh reproducibility evidence for both named fixtures. | Record generated oracle/report paths and row counts; do not treat rows as source-controlled pass proof. |
| 15 | `git diff --check` | all touched files | Whitespace and patch hygiene for all Sprint 146 changes. | Record clean output. |
| 16 | `rg -n "[ \t]+$" docs/planning/EPIC_12/SPRINT_146` | docs | Sprint 146 artifacts are Markdown-only so far and require trailing-whitespace hygiene. | Record no matches. |

## Conditional Local Baseline

| Condition | Command | Action Rule |
| --- | --- | --- |
| Any `.c` or `.h` file changed in Sprint 146 | `make format && make lint && make test` | Required before proceeding. If it fails, fix or stop. |
| Report family metadata, report schema, or normalization script changed | Report schema, normalization, freshness, and `tests/test_normalize_report_index.py` commands from the required baseline | Required before proceeding. |
| Corpus manifests, expected rows, corpus schemas, or oracle script changed | `python3 scripts/validate_corpus_schema.py`; affected oracle commands; focused corpus proof tests | Required before proceeding. |
| Package metadata, install scripts, CMake package files, `sparse.pc.in`, or install docs changed | `bash scripts/static_package_deferral_check.sh`; `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh` | Required before proceeding. |
| Examples or adoption examples docs changed | `make examples-build` and relevant install/downstream checks if examples cite installed usage | Required before proceeding. |
| Runtime/backend sentinel scripts or benchmark docs changed | `make performance-sentinels`; sentinel report-index freshness commands | Required if sentinel behavior or claims changed. |
| Workflow files changed | YAML parse/inspection plus hosted CI reconciliation | Local syntax checks are necessary but not sufficient for platform claim promotion. |

## Optional Generated-Evidence Refresh Commands

These commands can strengthen local evidence but should not be required unless
Day 5 or later surfaces claim the associated generated rows.

| Command | Use | Non-Claim Boundary |
| --- | --- | --- |
| `make performance-sentinels` | Refresh local hard/advisory sentinel rows. | No portable performance, benchmark superiority, backend portability, or platform claim. |
| `make bench-canonical-report` | Refresh local canonical benchmark rows. | No release benchmark, algorithmic superiority, or state-of-the-art claim. |
| `make large-matrix-guardrails` | Refresh large-matrix guardrail rows. | No broad scalability or memory-footprint guarantee. |
| `make deadcode-report` | Refresh local dead-code classification rows. | No zero-dead-code or release quality claim. |
| `make coverage` | Refresh local coverage report if coverage tooling is available. | No coverage completeness or hosted coverage parity claim. |
| `python3 scripts/normalize_report_index.py --require-generated <family> --check-freshness` | Promote missing generated rows for a selected family to a hard local review failure. | Only use when Sprint 146 explicitly requires current generated rows for that family. |

## Hosted-CI-Only Evidence List

| Evidence | Hosted Source | Local Substitute | Day 6-7 Reconciliation Rule |
| --- | --- | --- | --- |
| Linux reviewed source-of-truth package/quality/workflow proof | `.github/workflows/ci.yml` run logs | Local Make, package, CMake, report, and docs commands | Hosted status must be inspected before final support-tier promotion. |
| macOS reviewed static-first Make install/`pkg-config` and CMake install/export proof | `.github/workflows/macos-ci.yml` run logs on `macos-latest` | Local install scripts and workflow syntax inspection | Do not cite local Linux/macOS-neutral scripts as hosted macOS proof without successful hosted run logs. |
| Windows reviewed MSVC CMake subset | `.github/workflows/windows-ci.yml` run logs on Windows/MSVC | Local CMake files and source registration checks | Do not claim Windows Makefile, `pkg-config`, install-validation parity, or staged POSIX/pthread closure unless hosted lanes earn it. |
| Windows supplemental CMake install/downstream confidence | `.github/workflows/windows-ci.yml` supplemental PowerShell block | Local `tests/test_cmake_install.sh` is not equivalent | Treat as supplemental unless promoted by explicit reviewed hosted evidence. |
| PR review and CI status for Sprint 146 branch | GitHub PR checks and review comments | Local validation log | Local pass does not replace required hosted status for platform support claims. |

## Environment Constraint Register

| Constraint | Effect | Capture Requirement |
| --- | --- | --- |
| Host OS is not macOS | Cannot execute hosted macOS proof locally. | Record as hosted-only; inspect CI on Days 6-7. |
| Host OS is not Windows/MSVC | Cannot execute reviewed Windows CMake subset locally. | Record as hosted-only; inspect CI on Days 6-7. |
| Optional external corpus data may be absent | Optional-data rows remain skip/defer and cannot become pass evidence. | Record unset or unavailable optional-data state if oracle commands are run. |
| Generated reports may be absent or stale | Default freshness may warn/advisory rather than fail. | Record diagnostics and avoid generated freshness claims unless refreshed. |
| Coverage tooling may be unavailable or slow | `make coverage` is optional unless coverage rows are claimed fresh. | Record skip rationale; do not claim coverage completeness. |
| Benchmark timing depends on local machine | Benchmark/sentinel rows are local context only. | Record hardware/host caveat if refreshed; no portable performance claim. |

## Pass/Fail Capture Template

Use this format for each Day 5 command:

```text
Command:
Surface:
Required: yes/no
Result: pass/fail/skip
Exit code:
Evidence captured:
Generated artifacts:
Claim impact:
Notes:
```

The final local validation log should keep exact commands in execution order.
For generated artifacts, record the path and row count but state whether the
artifact is ignored, source-controlled, or hosted external.

## Stop Conditions

| Failure | Stop Or Fix Rule |
| --- | --- |
| Required command fails | Stop Day 5 after one focused diagnosis unless the fix is obvious and scoped. |
| `make format && make lint && make test` fails after C/header changes | Fix before proceeding; do not defer with source changes pending. |
| Corpus schema validation fails | Fix schema/manifest/expected rows before any corpus, QR, or partial-SVD claim is preserved. |
| Install or CMake install proof fails | Fix before preserving package/adoption install claims. |
| Static package deferral guard fails | Fix before preserving static-first or shared-library non-claim wording. |
| Report freshness emits unexpected hard errors | Fix report metadata or explicitly move generated evidence to residual; do not claim freshness. |
| Hosted CI evidence is required but unavailable | Stop claim promotion and record hosted-only uncertainty for Days 6-7. |
| Documentation wording widens beyond evidence | Fix wording before final claim audit; if unclear, record as residual/non-claim. |

## Day 5 Execution Plan

Day 5 should run the required local baseline in order, capture results using
the template above, and stop on the first unclear required failure. Because
Sprint 146 has only Markdown changes at the end of Day 4, the full C gate is
conditional, but the focused QR and partial-SVD corpus proof commands remain
part of the final evidence refresh plan.
