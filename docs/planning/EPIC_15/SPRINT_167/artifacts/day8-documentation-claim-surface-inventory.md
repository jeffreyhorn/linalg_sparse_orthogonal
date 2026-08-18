# Sprint 167 Day 8: Documentation And Claim Surface Inventory

## Purpose

Day 8 inventories README, install, maintainer, generated report, API, corpus,
benchmark, and planning documentation surfaces. The goal is to assign public
claim wording to named source files, separate current authoritative docs from
historical planning artifacts, and prepare stale or ambiguous wording
candidates for the Sprint 167 evidence ledger.

## Documentation Surfaces Reviewed

| Surface | Files | Current role |
| --- | --- | --- |
| Front-door user guidance | `README.md` | Primary user-facing claim surface for capabilities, workflows, build/test commands, platform tiers, report interpretation, benchmarks, and install summary. |
| Installation and package details | `INSTALL.md` | Operational source for install, package metadata, downstream consumer workflows, supported platforms, and package non-claims. |
| Benchmark guidance | `benchmarks/README.md` | Benchmark command, generated report, methodology, and local-measurement interpretation owner. |
| API reference | `docs/api_reference.md` | Source-header-first API reference path and generated API HTML policy owner. |
| Tutorial and cookbook | `docs/tutorial.md`, `docs/cookbook.md` | Adoption workflow and practical usage guidance. |
| Solver selection | `docs/solver_selection.md` | Solver-choice guidance, benchmark handoff, and evidence-bound solver claim caveats. |
| Maintainer policy | `docs/maintainer_guide.md` | Maintainer-facing evidence interpretation, workflow ownership, claim hygiene, package/ABI boundaries, generated report handling, and generated API rules. |
| Algorithm docs | `docs/algorithm.md`, `docs/algorithm_history.md` | Algorithm explanation and historical algorithm notes; not the primary current support claim surface. |
| Matrix Market docs | `docs/matrix_market.md` | Matrix Market usage and format guidance. |
| Corpus docs and schemas | `tests/corpus/README.md`, `tests/corpus/schemas/*.md`, `tests/corpus/manifests/*.tsv` | Source-controlled evidence vocabulary, fixture-local claim scopes, support tiers, and non-claim boundaries. |
| Sprint and epic planning artifacts | `docs/planning/EPIC_*/...` | Historical and current planning/retrospective evidence. Current epic/sprint artifacts can feed ledgers; older artifacts are historical unless cited by a current doc. |

## Authoritative Claim Source Map

| Claim area | Authoritative current source files | Evidence owners to map in ledger |
| --- | --- | --- |
| Build and local test commands | `README.md`, `Makefile`, `CMakeLists.txt` | `make quality-review-*`, `make test`, CMake configure/build/CTest paths. |
| Static-first install/package support | `INSTALL.md`, `README.md`, `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, CI workflows | Unix Make/pkg-config proof, CMake install/export proof, Windows CMake downstream proof, static deferral guard. |
| Windows support tier | `README.md`, `INSTALL.md`, `.github/workflows/windows-ci.yml`, `docs/maintainer_guide.md` | Windows CMake configure/build/CTest count `59`, CMake install/downstream validation, explicit Makefile/pkg-config non-claims. |
| macOS support tier | `README.md`, `INSTALL.md`, `.github/workflows/macos-ci.yml`, `docs/maintainer_guide.md` | Apple Clang reviewed path, Homebrew GCC supplemental path, static install/export proof. |
| Linux support tier | `README.md`, `INSTALL.md`, `.github/workflows/ci.yml`, `docs/maintainer_guide.md` | Makefile compile-quality, CMake parity, package contract, selected hosted oracle/comparison freshness, dead-code, coverage, sanitizer, bench-fast. |
| Generated API HTML | `docs/api_reference.md`, `docs/maintainer_guide.md`, `README.md` | Local-only generated HTML policy, source-header-first API authority, docs-check/API coverage commands. |
| Public API and examples | `docs/api_reference.md`, `docs/tutorial.md`, `docs/cookbook.md`, `examples/README.md`, public headers | Header comments, examples, generated docs checks, declaration-preservation evidence where applicable. |
| Solver selection and solver claims | `docs/solver_selection.md`, `README.md`, `docs/maintainer_guide.md`, solver tests | Solver-family tests, corpus fixtures, selected oracle/comparison rows, explicit non-claims. |
| Corpus and oracle rows | `tests/corpus/README.md`, `tests/corpus/manifests/*.tsv`, `tests/corpus/schemas/*.md`, `scripts/run_corpus_oracle.py`, `scripts/normalize_report_index.py` | Source-controlled metadata, generated local rows, selected hosted Linux freshness lane. |
| Selected comparison rows | `tests/corpus/README.md`, `tests/corpus/manifests/report_families.tsv`, `scripts/run_external_comparison.py`, `.github/workflows/ci.yml`, `docs/maintainer_guide.md` | `qr-minnorm`, `qr-compatible-ls`, `partial-svd-diag6-k2` generated rows and hosted selected comparison lane. |
| Benchmarks and performance | `benchmarks/README.md`, `README.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`, Makefile benchmark targets | Local benchmark rows, local sentinels, bench-fast CI smoke, no portable performance superiority. |
| State-of-the-art positioning | `README.md`, Epic retrospectives, Epic 15 review | Explicit non-claim unless comprehensive competitive evidence exists. |

## Current Versus Historical Interpretation

| Documentation class | Examples | How to interpret |
| --- | --- | --- |
| Current user-facing docs | `README.md`, `INSTALL.md`, `benchmarks/README.md`, `docs/api_reference.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md` | Treat as current public claim surfaces that must map to evidence or explicit non-claims. |
| Current maintainer policy | `docs/maintainer_guide.md`, corpus schemas/manifests | Treat as authoritative for support tiers, report row semantics, owner expectations, and claim hygiene. |
| Source-controlled proof owners | tests, scripts, Makefile, CMake, CI workflows | Treat as evidence owners only when commands or hosted lanes are named and current. |
| Generated local output | ignored `build/`, `coverage/`, generated report paths, generated API HTML | Treat as local-only unless promoted by a reviewed hosted lane or publication decision. |
| Hosted external logs | GitHub Actions runs | Treat as hosted evidence only for exact workflow/job/commit/scope. |
| Historical planning artifacts | older `docs/planning/EPIC_*` sprint artifacts and retrospectives | Treat as historical rationale or prior evidence. Do not treat as current support unless current docs/CI/tests still own the claim. |
| Current Epic 15 planning artifacts | `docs/planning/EPIC_15/**` | Treat as planning baseline and future evidence ledger, not implementation proof. |

## Claim Wording Candidates For Ledger Mapping

| Candidate | Source files | Mapping needed |
| --- | --- | --- |
| "Linux is the strongest reviewed source of truth" | `README.md`, `INSTALL.md`, `.github/workflows/ci.yml` | Map to Linux reviewed Makefile compile-quality, CMake parity, dead-code, package, and selected report-freshness jobs. |
| Windows CMake-first support | `README.md`, `INSTALL.md`, `.github/workflows/windows-ci.yml`, `docs/maintainer_guide.md` | Map to CMake configure/build/CTest `59` and install/downstream validation. |
| Static-first package contract | `README.md`, `INSTALL.md`, `sparse.pc.in`, CMake/Make install rules, install tests | Map to Unix Make/pkg-config, CMake install/export, Windows CMake metadata/downstream proof, and static deferral guard. |
| Generated API local-only policy | `docs/api_reference.md`, `docs/maintainer_guide.md`, README links | Map to source-header-first docs, `make docs`, docs checks, and absence of hosted/generated HTML publication. |
| Selected hosted oracle/comparison freshness | `.github/workflows/ci.yml`, `tests/corpus/README.md`, report-family manifest | Map to selected QR/partial-SVD oracle rows and selected comparison families only. |
| Benchmark/report interpretation | `README.md`, `benchmarks/README.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md` | Map to local-only benchmark rows, sentinel rows, bench-fast smoke, methodology fields, and performance non-claims. |
| State-of-the-art non-claim | README, corpus docs, Epic retrospectives, Epic 15 review | Keep unsupported unless broad competitive evidence exists. |
| External-library parity non-claim | README, corpus docs, solver-selection, maintainer guide | Preserve fixture-local comparison boundaries and avoid broad ecosystem wording. |

## Stale Or Ambiguous Wording Notes

| Area | Observation | Risk | Day 9/Day 10 handling |
| --- | --- | --- | --- |
| Sprint prompts naming Epic 12 for Epic 15 sprint sections | The current Sprint 167 prompt referenced `docs/planning/EPIC_12/PROJECT_PLAN.md`, but active Sprint 167 lives in Epic 15. | Future artifacts could cite the wrong source plan path. | Keep source artifact notes in sprint plans and working notes; ledger should cite Epic 15 project plan for Sprint 167. |
| Generated API HTML | Current policy is local-only and ignored, while users may expect hosted API docs. | "Generated API" wording can be mistaken for hosted or release-published docs. | Ledger should classify as local-only unless Sprint 173 changes publication status. |
| Benchmark publication | Benchmark rows are local measurement artifacts, but "publication" wording can sound release-grade. | Could imply portable performance or state-of-the-art speed. | Ledger should map benchmark claims to local-only/support-tier fields and R167-02 hosted decision. |
| Package-manager support | Install docs mention package prerequisites and staging but explicitly do not claim package-manager distribution. | Readers may confuse source install with package-manager availability. | Ledger should distinguish source install, CMake/pkg-config metadata, and package-manager non-claim. |
| Windows `sparse.pc` | Windows installs and inspects `sparse.pc` metadata but does not run `pkg-config`. | Easy to overread as Windows pkg-config execution parity. | Ledger should keep Windows `pkg-config` command execution unsupported. |
| Historical planning docs | Older sprint artifacts contain prior decisions, stale paths, and historical command states. | Current docs could accidentally cite historical artifacts as live proof. | Ledger should use historical artifacts only as rationale unless current evidence owners still validate the claim. |
| Broad comparison wording | Maintainer guide lists many solver fixtures and comparisons. | Large fixture lists can look like broad ecosystem parity. | Ledger should preserve fixture-local and selected-family qualifiers. |

## Day 9 Handoff

Day 9 should use this claim surface inventory to draft the Epic 15 evidence
ledger. Each ledger row should include:

- claim area;
- current support status;
- authoritative source file;
- validation command or hosted CI owner;
- support tier;
- non-claims;
- future sprint owner or retained deferral.

Initial ledger rows should cover build/test, static package, Windows/macOS/Linux
platform tiers, generated API HTML, public API/header cleanup, selected
oracle/comparison freshness, benchmarks/performance, package-manager support,
shared-library/ABI, allocation-failure evidence, and state-of-the-art/external
parity.

## Validation Notes

Day 8 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public claim wording has named source files. | Complete | Authoritative claim source map assigns claim areas to current source files and evidence owners. |
| Historical planning artifacts are separated from current user-facing docs. | Complete | Current versus historical interpretation table distinguishes public docs, maintainer policy, source proof owners, generated local output, hosted logs, and planning history. |
| Stale or ambiguous claim surfaces are ready for cleanup planning. | Complete | Stale or ambiguous wording notes identify path mismatch, generated API, benchmarks, package-manager, Windows `sparse.pc`, historical artifacts, and broad comparison wording. |
