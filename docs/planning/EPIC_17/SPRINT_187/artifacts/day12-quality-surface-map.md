# Sprint 187 Day 12: Quality Surface Map

## Purpose

Provide one reusable validation map for Epic 17 implementation sprints. Each
future sprint should use this map to decide the minimum required checks,
stronger optional checks, hosted evidence needs, and local skip behavior for
the files it changes.

## Core Policy

Any `.c` or `.h` change requires the full C quality gate:

```sh
make format
make lint
make test
```

Focused tests, script guards, report freshness checks, docs checks, hosted CI,
and manual inspection may add evidence. They do not replace the mandatory C
gate for C or header edits.

Planning-only documentation changes do not require the full C quality gate
unless they also modify `.c` or `.h` files.

## Quality Surface Matrix

| Surface | Typical files | Minimum required validation | Stronger optional validation | Hosted dependency | Skip/unavailable rule |
| --- | --- | --- | --- | --- | --- |
| Planning-only docs | `docs/planning/**` | `git diff --check`; trailing-whitespace scan; Markdown link check for changed planning tree. | Full repo Markdown link check. | None. | Not applicable. |
| User documentation | `README.md`, `INSTALL.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `examples/README.md` | `git diff --check`; Markdown link check; claim-boundary review against active gates. | Matching install, example, package, report, or header guard. | None unless support claim depends on hosted evidence. | Missing optional external tools must be stated as unavailable rather than treated as proof. |
| Maintainer documentation | `docs/maintainer_guide.md`, schema docs, report docs | `git diff --check`; Markdown link check; guard named by the touched owner. | `make docs-check`; selected report or package guards. | None unless it cites hosted promotion. | Historical planning links should be retained only when they explain current support truth. |
| Public headers | `include/*.h` | `make format && make lint && make test`; `make docs-check`; `make api-docs-freshness`. | Header-specific guards such as `bash scripts/check_qr_header_docs_guard.sh` and `bash scripts/check_lu_header_docs_guard.sh`; CMake parity. | None by default. | Generated Doxygen output stays local-only and ignored unless a future publication decision changes that. |
| C implementation | `src/*.c`, `src/*.h` | `make format && make lint && make test`. | Focused owner binary; `make quality-review-full`; sanitizers when memory/thread risk is high. | Hosted workflow evidence only for platform promotion. | Local sanitizer or optional backend absence is not pass evidence. |
| Test implementation | `tests/*.c`, `tests/*.h`, test helper headers | `make format && make lint && make test`; focused proof-owner binary. | `make quality-review-cmake-compile`; CTest label; owner guard. | Hosted evidence when platform-specific support changes. | Skipped optional dependency rows must stay explicit. |
| Test registration/source lists | `Makefile`, `CMakeLists.txt`, `build-metadata/library_sources.txt` | `make source-list-check`; relevant focused test; `make format && make lint && make test` if C/header files changed. | `make quality-review-cmake-compile`; `make quality-review-cmake`. | Hosted CI if platform claims or workflow registration change. | Registration drift cannot be accepted as docs-only. |
| Scripts | `scripts/*.sh`, `scripts/*.py` | Syntax/targeted script run; owner guard or test named by the surface. | `make lint`; script-specific unit test; workflow guard. | Hosted evidence if the script only proves behavior on hosted runner. | Missing local tool should produce explicit unavailable/skip status. |
| Workflows | `.github/workflows/*.yml` | Workflow-specific guard tests; claim-boundary review; `git diff --check`. | `gh workflow run ...`; `gh run watch ...`; failed-log inspection. | Required before support promotion tied to hosted CI. | Local validation cannot replace required hosted proof. |
| Package/install | `packaging/**`, `INSTALL.md`, package metadata, install scripts | `bash scripts/package_manager_deferral_check.sh`; `bash scripts/static_package_deferral_check.sh`; selected install proof. | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; Homebrew proof when selected. | Hosted macOS/Windows evidence for platform package promotion. | `scripts/homebrew_local_formula_proof.sh` exit `2` is blocker evidence, not support proof. |
| Homebrew proof | `packaging/homebrew/**`, `scripts/homebrew_local_formula_proof.sh` | `SPARSE_HOMEBREW_LICENSE=<accurate-id> scripts/homebrew_local_formula_proof.sh`; package guards. | `--keep-temp` rerun for diagnostics; docs/install validation. | Local macOS/Homebrew or hosted macOS proof if selected. | Missing `brew` or license metadata keeps proof unclaimed. |
| Windows validation | `.github/workflows/windows-ci.yml`, PowerShell snippets, Windows docs | Windows workflow guard tests; selected PowerShell validation command when available. | Hosted Windows workflow run and log review. | Required for Windows support promotion. | Missing local `pwsh` must be skip/unavailable, not pass. |
| Selected reports/manifests | `tests/corpus/manifests/*.tsv`, `scripts/normalize_report_index.py`, report docs | `python3 scripts/validate_corpus_schema.py`; `python3 tests/test_selected_report_targets_manifest.py`; selected normalizer check. | Full `python3 tests/test_normalize_report_index.py`; family freshness target. | Hosted evidence for hosted freshness promotion. | Missing generated artifacts are not pass evidence when `--require-generated` is selected. |
| External comparison | `scripts/run_external_comparison.py`, comparison manifest rows, comparison workflow tests | `make report-index-comparison-freshness`; `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness`; workflow guard. | Target-specific dependency/failure tests; hosted Linux/macOS workflow run. | Required before hosted comparison support promotion. | Optional dependency unavailable rows must not become selected pass evidence. |
| Oracle reports | `scripts/run_corpus_oracle.py`, oracle fixtures, oracle manifest rows | `make report-index-oracle-freshness`; `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness`. | Full corpus schema/tests. | Hosted evidence if promoted beyond local selected freshness. | Missing generated rows fail selected freshness. |
| Benchmarks/performance | `benchmarks/**`, `scripts/bench_canonical_report.sh`, `scripts/check_bench_canonical_freshness.py`, benchmark manifest row | `make bench-canonical-report-freshness`; `python3 scripts/check_bench_canonical_freshness.py --mode local`; report-index benchmark checks. | `python3 scripts/check_bench_canonical_freshness.py --mode hosted`; hosted workflow run; `make performance-sentinels`. | Required before hosted performance promotion. | Local hosted-mode emulation validates shape only, not hosted evidence. |
| Large matrix guardrails | Reorder/graph tests, guardrail script, benchmark guardrails | `make large-matrix-guardrails`. | Supplemental guardrails with `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1`. | None unless promoted to hosted evidence. | Supplemental skipped rows are explicit context, not failure. |
| Allocation-failure reliability | `src/sparse_alloc_internal.*`, selected owner tests, focused gates | Focused owner gate; `make format && make lint && make test`. | CTest label; registration guard; sanitizer when memory ownership changes. | Hosted evidence only for platform promotion. | Fail hooks must be reset on every path; skipped failure injection is not proof. |
| Generated API output | `docs/api/html/**`, Doxygen output | `make api-docs-freshness`; `bash scripts/check_api_docs_local_only.sh`. | `make docs-check`. | None unless a future hosted publication decision exists. | Generated HTML stays ignored/local-only and must not be staged. |
| Generated report outputs | `build/**` report artifacts | Family freshness target when generated output is claim-bearing. | Normalized report index regeneration and family-specific tests. | Hosted upload evidence for hosted claim-bearing artifacts. | Generated build artifacts generally stay ignored and uncommitted. |

## Selected Sprint Validation Map

| Sprint | Closure | Required validation anchor |
| --- | --- | --- |
| Sprint 188 | Homebrew proof completion | Day 7 package gates, Homebrew proof, package guards, install/docs checks, full C gate if C/header files change. |
| Sprint 189 | PowerShell validation ownership | Day 8 Windows gates, PowerShell validation command, workflow guard tests, hosted Windows proof for promotion. |
| Sprint 190 | Windows report freshness decision | Day 8 promotion or renewed-deferral path, selected manifest tests, workflow guards, hosted Windows evidence if promoted. |
| Sprint 191 | Bounded external comparison family | Day 9 comparison gates, selected comparison freshness, manifest/schema tests, workflow artifact guards, full C gate if needed. |
| Sprint 192 | Methodology-bound performance lane | Day 9 performance gates, benchmark freshness, hosted-mode checker, exact artifact upload evidence, full C gate if needed. |
| Sprint 193 | Selected large review-surface reduction | Day 10 maintainability gates, focused proof-owner test, source-list/registration guards, full C gate. |
| Sprint 194 | Adoption and API coherence simplification | Day 11 documentation gates, link/API/docs/example/install/header checks selected by changed surfaces. |
| Sprint 195 | Selected reliability and failure-path proof | Day 10 reliability gates, focused reliability gate, allocation-failure registration guard, full C gate. |
| Sprint 196 | Final validation and closeout | All changed-surface gates plus final claim-boundary review. |

## Mandatory Full C Gate

Run the following whenever any `.c` or `.h` file changes:

```sh
make format
make lint
make test
```

Also run surface-specific focused checks. The full C gate is necessary but not
always sufficient for package, workflow, report, benchmark, documentation, or
hosted platform promotion.

## Hosted Evidence Rules

Hosted proof is required before promoting:

- Windows support beyond the already reviewed CMake/static install boundary;
- Windows selected report freshness;
- hosted Linux/macOS selected comparison claims;
- hosted selected performance freshness;
- package/platform support that depends on hosted runner behavior;
- workflow artifact publication or upload-scope claims.

Use `gh workflow run`, `gh run watch`, and `gh run view --log-failed` when a
sprint changes a workflow and intends to cite hosted evidence. If a hosted run
cannot be triggered or inspected, the sprint must retain the affected support
surface as a non-claim or stop for clarification.

## Local Skip and Unavailable Rules

Local absence of a required tool must be explicit:

| Tool or condition | Accepted local interpretation |
| --- | --- |
| `pwsh` missing | PowerShell validation is skipped/unavailable locally; hosted Windows must own proof before promotion. |
| `brew` missing | Homebrew local formula proof is unavailable; package support remains unclaimed. |
| Homebrew license metadata missing | Proof exits unavailable; the missing metadata is blocker evidence. |
| Optional external comparison dependency missing | Dependency status row records unavailable; selected pass evidence is not created. |
| Generated report artifact missing | Selected freshness fails when `--require-generated` is active. |
| Hosted performance metadata unavailable locally | Local check may validate shape only; hosted promotion needs hosted run evidence. |
| Sanitizer/toolchain unavailable | Record as unavailable and use the required non-sanitizer gate unless sanitizer proof is the selected claim. |

Unavailable is not pass. Skip is not pass. A deferral can close only when it is
explicitly guarded, documented, and paired with revisit criteria.

## Stronger Optional Baselines

Use these when risk or blast radius is high:

```sh
make quality-review
make quality-review-cmake
make quality-review-full
make sanitize
make asan
make tsan
make coverage
```

Use stronger baselines for broad source movement, memory ownership changes,
threading/backend changes, CMake registration changes, or final closeout when
local runtime is acceptable.

## Validation

Day 12 changed planning documentation only. No `.c` or `.h` files were
modified, so the full C quality gate is not required.
