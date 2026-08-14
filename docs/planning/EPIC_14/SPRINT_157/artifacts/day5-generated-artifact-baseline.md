# Day 5 Generated Artifact Baseline

## Scope

Day 5 inventories generated artifact families and freezes the current support
tier interpretation. It separates checked-in source metadata from ignored local
outputs. It does not regenerate reports, publish generated artifacts, or
promote any generated row to source-controlled pass evidence.

## Inventory Commands

| Command | Result used |
| --- | --- |
| `git ls-files docs/api` | No tracked generated API files were listed. |
| `git status --ignored=matching --short docs/api build coverage` | `docs/api/` and `build/` are ignored local-output trees. |
| `git check-ignore -v docs/api/html/index.html build/corpus/oracle/index.tsv build/corpus-reports/index.tsv build/comparison/qr_minnorm/study.tsv build/bench-reports/canonical/index.tsv build/deadcode/report.tsv coverage/coverage-src.info` | `.gitignore` owns `docs/api/`, `build/`, and `coverage/` generated-output exclusion. |
| `find tests/corpus -maxdepth 3 -type f | sort` | Captured the source-controlled corpus metadata, expected rows, manifests, and schemas. |
| `sed -n '1,80p' tests/corpus/manifests/report_families.tsv` | Captured report-family row origins, support tiers, freshness policies, and non-claims. |
| `rg -n "report-index-oracle-freshness|report-index-comparison-freshness|bench-canonical-report|performance-sentinels|large-matrix-guardrails|deadcode-report|coverage" Makefile docs/maintainer_guide.md benchmarks/README.md tests/corpus/README.md` | Captured maintained freshness and generated-report command surfaces. |
| `find build/corpus build/corpus-reports build/comparison build/report-index build/bench-reports build/deadcode coverage -maxdepth 3 -type f` | Found local ignored oracle, corpus-report, comparison, and normalized report-index outputs. |

## Generated API Documentation Baseline

| Surface | Source-controlled input | Generated output | Current tracking state | Day 5 interpretation |
| --- | --- | --- | --- | --- |
| Doxygen API HTML | `Doxyfile`, `docs/api_reference.md`, checked-in public headers under `include/`, and `include/sparse_version.h.in` for installed generated version metadata | `docs/api/html/` | No tracked `docs/api` files; `.gitignore` ignores `docs/api/` | Local convenience view only unless `make docs` is run, warnings are triaged, page coverage is checked, generated output is committed, and the PR states generated output changed. |

The API source of truth remains the checked-in public headers and
`docs/api_reference.md`. Sprint 158 should start from this baseline and decide
whether generated HTML is published or the local-only policy is retained.

## Corpus And Report-Family Baseline

| Source-controlled corpus surface | Count | Day 5 interpretation |
| --- | ---: | --- |
| `tests/corpus/expected/*.tsv` | 10 | Checked-in expected-result rows define fixture-local targets and status conditions, not observed pass evidence. |
| `tests/corpus/manifests/*.tsv` | 4 | Checked-in fixture, generator, optional-data, and report-family metadata own row identity and support-tier semantics. |
| `tests/corpus/schemas/*.md` | 3 | Checked-in schema docs define field meanings and validation expectations. |
| `tests/corpus/README.md`, `tests/corpus/expected/README.md`, `tests/corpus/fixtures/README.md` | 3 | Checked-in interpretation docs own corpus non-claims and generated-output rules. |

The report-family manifest currently separates source-controlled rows from
generated-local rows. Source-controlled corpus metadata is advisory context
until a solver test, oracle output, comparison output, or hosted CI lane supplies
the relevant observed evidence.

## Generated Artifact Classification

| Family | Source of truth | Generated output | Freshness or generation command | Support tier | Claim boundary | Epic 14 candidate |
| --- | --- | --- | --- | --- | --- | --- |
| API HTML | `Doxyfile`, public headers, `docs/api_reference.md` | `docs/api/html/` | `make docs` | local-only unless explicitly published | Convenience reference, not ABI, package, platform, external-parity, performance, or state-of-the-art proof. | Sprint 158 primary candidate. |
| Corpus expected metadata | `tests/corpus/expected/*.tsv`, manifests, schemas | none required | `python3 scripts/validate_corpus_schema.py` | source-controlled advisory | Defines expected rows and row identity, not observed pass evidence. | Baseline retained. |
| Oracle reports | corpus metadata, oracle runner, solver tests | `build/corpus/oracle/*.tsv`, `build/corpus-reports/*` | `make report-index-oracle-freshness` | local-only generated compare inputs | Fixture-local generated evidence only; no hosted proof, broad corpus completeness, external parity, or platform portability. | Sprint 159 selected-hosted promotion candidate. |
| QR external comparison | `scripts/run_external_comparison.py`, QR dense reference helper, report-family row | `build/comparison/qr_minnorm/{baseline_observations.tsv,dependency_status.tsv,manifest.tsv,project_observations.tsv,study.tsv,summary.md}` | `make report-index-comparison-freshness` | local-only generated compare inputs | One selected fixture-level comparison only; no broad QR, NumPy, SciPy, LAPACK, SuiteSparse, Eigen, hosted, performance, or platform claim. | Sprint 160/Sprint 164 comparison-methodology candidate. |
| Canonical benchmark reports | benchmark sources and scripts | `build/bench-reports/canonical/index.tsv`, `manifest.txt` | `make bench-canonical-report` | local-only advisory | Threshold-free local measurement snapshot; no portable performance or superiority claim. | Sprint 163 methodology-bound publication candidate. |
| Runtime sentinels | sentinel scripts, benchmark commands, wall-check policy | `build/bench-reports/sentinels/sentinels.tsv`, `manifest.txt` | `make performance-sentinels` | local-only; existing wall-check lane is the bounded hard gate | Bounded local wall behavior only; no throughput, backend, or state-of-the-art claim. | Sprint 163 governance candidate. |
| Large-matrix guardrails | guardrail scripts and benchmark docs | `build/bench-reports/large-matrix-guardrails/index.tsv`, `manifest.txt` | `make large-matrix-guardrails`; optional `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails` | local-only reviewed/supplemental rows | Explicit pass, fail, or skip rows; no broad scalability, memory, platform, or external-corpus claim. | Candidate only if selected by later evidence contract. |
| Dead-code report | compile database, dead-code scripts, maintainer policy | `build/deadcode/report.tsv` plus report artifacts | `make deadcode-report`; `make deadcode-check` | local-only advisory with Linux reviewed completeness lane | Maintainer classification signal; no zero-dead-code or semantic correctness guarantee. | Quality surface context for Day 10. |
| Coverage report | coverage backend and test suite | `coverage/coverage-src.info`, `coverage/html/` | `make coverage`, `make coverage-lcov`, `make coverage-gcovr` | supplemental local/tree-mutating | Backend-specific coverage signal; no completeness, branch-parity, hosted platform, or quality claim. | Quality surface context for Day 10. |
| Package report rows | install scripts, CMake/pkg-config templates, static-package deferral check | source-controlled scripts/templates plus generated install trees in test temp dirs | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh`; Windows CMake install workflow | reviewed cross-platform where CI owns it | Static-first install/export proof only; no package-manager, shared-library, or dynamic ABI support. | Sprint 162/Sprint 165 package-boundary candidate. |
| CI lane definitions | `.github/workflows/*.yml` | hosted logs outside source control | GitHub Actions | reviewed or supplemental by workflow comments | Lane definitions are checked in; absent logs do not create pass evidence. | Day 10 quality-map input. |
| Normalized report index | `scripts/normalize_report_index.py`, report-family manifest, generated families | `build/report-index/normalized-index.tsv` and focused outputs | `python3 scripts/normalize_report_index.py --check-freshness` and focused family variants | local-only advisory unless required families fail freshness | Preserves row meaning, support tier, freshness context, claim scope, and non-claims; not release proof by itself. | Sprint 159 selected-hosted report candidate. |

## Local Ignored Output Observed

The workspace currently contains ignored local outputs under:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/manifest.txt`
- `build/corpus-reports/skips.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/manifest.tsv`
- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/report-index/day7-corpus-oracle-normalized.tsv`
- `build/report-index/day8-comparison-normalized.tsv`

These files are useful local context only. They remain ignored and should not
be committed unless a later sprint explicitly changes the publication policy.

## Freshness Command List

| Purpose | Command | Output policy |
| --- | --- | --- |
| Validate corpus metadata schemas | `python3 scripts/validate_corpus_schema.py` | Source-controlled metadata check. |
| Regenerate and check selected oracle output | `make report-index-oracle-freshness` | Ignored local output under `build/corpus/`, `build/corpus-reports/`, and `build/report-index/`. |
| Regenerate and check selected comparison output | `make report-index-comparison-freshness` | Ignored local output under `build/comparison/qr_minnorm/` and `build/report-index/`. |
| Normalize selected report families | `python3 scripts/normalize_report_index.py --check-freshness` with focused `--family` selectors | Ignored local output unless a sprint explicitly promotes a stable example. |
| Generate canonical benchmark reports | `make bench-canonical-report` | Ignored local measurement output. |
| Generate performance sentinels | `make performance-sentinels` | Ignored local output; wall-check lane remains the bounded hard gate. |
| Generate large-matrix guardrails | `make large-matrix-guardrails` | Ignored local output, with supplemental mode opt-in. |
| Generate dead-code report | `make deadcode-report` | Ignored local output; CI may upload artifacts outside source control. |
| Run coverage | `make coverage` | Ignored tree-mutating local output. |
| Generate API HTML | `make docs` | Ignored local output unless explicitly published with warnings/page coverage triaged. |

## Source-Controlled Versus Ignored Decision List

| Decision | Paths | Rationale |
| --- | --- | --- |
| Keep as source-controlled source of truth | `tests/corpus/**`, `scripts/*.py`, `scripts/*.sh`, `Doxyfile`, public docs, public headers, `Makefile`, `CMakeLists.txt`, `.github/workflows/*.yml`, package templates | These files define commands, row meaning, claim boundaries, and maintained validation surfaces. |
| Keep ignored unless explicitly promoted | `docs/api/html/`, `build/corpus/`, `build/corpus-reports/`, `build/comparison/`, `build/report-index/`, `build/bench-reports/`, `build/deadcode/`, `coverage/` | These are generated local artifacts whose freshness depends on command, commit, platform, compiler, configuration, and support tier. |
| Do not reinterpret ignored rows as source-controlled pass evidence | All generated output rows | Generated rows need explicit freshness, support-tier, and publication decisions before they support public claims. |
| Preserve selected promotion candidates for Epic 14 | `docs/api/html/`, selected oracle/report-index rows, selected comparison rows, methodology-bound benchmark/sentinel rows | These match Epic 14 complete-gap targets and should be handled by later sprint evidence contracts. |

## Completion Check

- Generated API HTML tracking state is captured.
- Corpus manifests, expected rows, schemas, and report-family rows are
  inventoried.
- Oracle, comparison, benchmark, sentinel, large-matrix, coverage, dead-code,
  package, CI, and normalized report-index families are classified by support
  tier and claim boundary.
- Freshness commands are listed with output policies.
- Ignored generated output cannot be mistaken for source-controlled pass
  evidence.
- Sprint 158 generated API docs work has a concrete baseline.
