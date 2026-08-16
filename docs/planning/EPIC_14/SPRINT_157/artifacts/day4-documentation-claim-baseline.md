# Day 4 Documentation And Claim Baseline

## Scope

Day 4 inventories the public and maintainer documentation surfaces that own
claims, support tiers, non-claims, and generated-evidence interpretation. This
artifact records the current claim baseline only. It does not edit public
wording or promote any deferred claim.

## Documentation Inputs

| Document | Primary role | Day 4 interpretation |
| --- | --- | --- |
| `README.md` | Public front door, capability summary, build/test route, install summary, known limitations. | Owns compact user-facing claims and routes deeper evidence to specialized docs. |
| `INSTALL.md` | Static-first install/export and platform support contract. | Owns package, supported-platform, Windows CMake-first, and static/shared boundary wording. |
| `docs/api_reference.md` | API reference entry point and generated HTML boundary. | Owns source-header-first API guidance and generated Doxygen non-claims. |
| `docs/tutorial.md` | First-use learning path and workflow escalation. | Owns adoption flow and beginner-safe boundaries for advanced evidence. |
| `docs/cookbook.md` | Workflow recipes for data input, solver use, SVD/eigs, and measurement. | Owns practical recipe language without broad performance or state-of-the-art claims. |
| `docs/solver_selection.md` | Solver-family decision tree and evidence boundaries. | Owns solver selection, QR/SVD proof limits, diagnostics, and benchmark handoffs. |
| `docs/matrix_market.md` | Matrix Market input/output support details. | Owns supported file-format behavior and I/O limits. |
| `docs/algorithm.md` | Algorithm description and architecture context. | Owns technical algorithm explanation, not install/support/API claims. |
| `docs/algorithm_history.md` | Historical measurement and design notes. | Owns historical context, not current release or performance guarantees. |
| `docs/maintainer_guide.md` | Authoritative maintainer policy, support tiers, evidence interpretation, and claim ownership. | Owns detailed claim boundaries and future-change rules. |
| `benchmarks/README.md` | Benchmark command and report interpretation. | Owns benchmark-local meaning and portable-performance non-claims. |
| `examples/README.md` | Example catalog, diagnostics, and teaching boundaries. | Owns example usage and avoids broad parity/performance claims. |
| `tests/corpus/README.md` | Maintained corpus layout, row interpretation, stale reports, and residuals. | Owns fixture-local corpus and generated-oracle non-claims. |
| `include/*.h` | Exact public declarations and local API contracts. | Owns call-site contracts, option/result semantics, warnings, and local non-claims where needed. |

## Positive Claim Register Draft

| Claim category | Positive claim currently present | Owner document(s) | Evidence surface |
| --- | --- | --- | --- |
| Library capability | The project is a C sparse matrix library using an orthogonal linked-list representation with direct, iterative, SVD, eigensolver, graph/reorder, I/O, and compressed-format workflows. | `README.md`, `docs/algorithm.md`, public headers | Implementation and tests under `src/`, `include/`, and `tests/`. |
| First-use path | Users should start from build, first solve, data input, solver choice, diagnostics, and install only when downstream consumption is needed. | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `examples/README.md` | Maintained examples and cookbook/solver-selection docs. |
| QR fixture-local evidence | QR has maintained fixture-local rank/nullity/nullspace/minimum-norm evidence and one local generated comparison for `qr_underdetermined_minnorm_2x4`. | `README.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` | `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_qr_corpus.c`, QR corpus rows, oracle freshness, comparison freshness. |
| Partial-SVD fixture-local evidence | Partial-SVD has maintained fixture-local generated diagonal, rank-deficient, sparse-output, and fail-closed recovery evidence. | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` | `tests/test_svd.c`, `tests/test_svd_partial_corpus.c`, corpus expected rows, oracle freshness. |
| Static-first package | Installed static archive, headers, CMake package metadata, and Unix-side `pkg-config` metadata are maintained. | `INSTALL.md`, `README.md`, `docs/maintainer_guide.md` | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`, CI package lanes. |
| Platform support tiers | Linux is strongest reviewed source of truth; macOS has reviewed Apple Clang and static-first package proof; Windows has reviewed MSVC CMake CTest and CMake install/downstream validation. | `INSTALL.md`, `README.md`, `docs/maintainer_guide.md`, `.github/workflows/*.yml` | Linux/macOS/Windows CI lanes and local equivalents. |
| Generated report semantics | Normalized reports preserve row meaning, freshness context, artifact paths, support tiers, claim scopes, and non-claims. | `docs/maintainer_guide.md`, `tests/corpus/README.md`, `benchmarks/README.md` | `scripts/normalize_report_index.py`, corpus report-family metadata, selected freshness commands. |
| Benchmark/report meaning | Benchmark rows and sentinel bundles provide local measurement and bounded guardrail evidence. | `README.md`, `benchmarks/README.md`, `docs/maintainer_guide.md` | `make bench-fast`, `make bench-canonical-report`, `make performance-sentinels`, `make large-matrix-guardrails`. |
| API source of truth | Checked-in public headers own exact declarations and call-site contracts; generated Doxygen HTML is convenience output. | `docs/api_reference.md`, `docs/maintainer_guide.md`, `include/*.h` | `include/`, `Doxyfile`, `make docs`. |

## Explicit Non-Claim Register Draft

| Non-claim | Current owner | Evidence from scan |
| --- | --- | --- |
| No unqualified state-of-the-art sparse linear algebra claim | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `docs/api_reference.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md` | Sensitive wording appears as explicit rejection or boundary language. |
| No broad external-library or ecosystem parity | `README.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md`, `tests/corpus/README.md`, `benchmarks/README.md`, examples docs | QR/SVD/corpus/comparison docs keep evidence fixture-local. |
| No portable performance guarantee or superiority claim | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `docs/solver_selection.md`, `benchmarks/README.md`, `docs/maintainer_guide.md`, algorithm docs | Benchmarks are documented as local measurements and bounded guardrails. |
| No shared-library support | `INSTALL.md`, `README.md`, `docs/maintainer_guide.md`, `docs/api_reference.md` | `BUILD_SHARED_LIBS=ON` remains rejected and shared-library blockers are named. |
| No dynamic ABI compatibility promise | `INSTALL.md`, `README.md`, `docs/api_reference.md`, `docs/maintainer_guide.md` | Static-first packaging does not imply ABI compatibility. |
| No package-manager distribution support | `INSTALL.md`, `README.md`, `docs/api_reference.md`, `docs/maintainer_guide.md` | Package-manager support remains out of scope. |
| No Windows Makefile parity | `INSTALL.md`, `README.md`, `docs/maintainer_guide.md`, Windows CI comments | Windows remains CMake-first. |
| No Windows `pkg-config` execution parity | `INSTALL.md`, `README.md`, `docs/maintainer_guide.md`, Windows CI comments | Installed `sparse.pc` metadata is checked, but Windows execution parity is not claimed. |
| No broad platform parity | `INSTALL.md`, `README.md`, `docs/api_reference.md`, `docs/maintainer_guide.md` | Platform support is tiered by CI lane. |
| No generated local row as source-controlled pass evidence | `docs/maintainer_guide.md`, `tests/corpus/README.md`, `benchmarks/README.md` | Generated rows remain local-only/advisory unless explicitly promoted. |

## Support-Tier Ownership Map

| Claim surface | Public owner | Maintainer owner | Validation owner |
| --- | --- | --- | --- |
| First-use adoption | `README.md`, `docs/tutorial.md`, `docs/cookbook.md`, `examples/README.md` | `docs/maintainer_guide.md` documentation ownership rules | examples and docs checks |
| Solver choice | `docs/solver_selection.md`, README workflow sections | `docs/maintainer_guide.md` capability snapshot | focused solver tests and corpus/oracle checks |
| API declarations | `docs/api_reference.md`, `include/*.h` | maintainer guide API reference/generated Doxygen section | header declaration preservation and `make docs` policy |
| Static install/export | `INSTALL.md`, README installation section | maintainer guide packaging/ABI contract | install scripts and CI package lanes |
| Platform support | `INSTALL.md`, README quality section | maintainer guide reviewed-baseline and support-tier sections | GitHub Actions workflows |
| Corpus and oracle reports | `tests/corpus/README.md`, solver docs summaries | maintainer guide selected oracle/corpus maintenance sections | corpus tests, oracle runner, report normalizer |
| External comparison | solver docs and maintainer guide comparison section | maintainer guide selected comparison freshness gate | comparison runner and report normalizer |
| Performance and benchmarks | README performance section, `benchmarks/README.md` | maintainer guide benchmark governance | benchmark/report/sentinel/guardrail commands |
| Historical algorithm context | `docs/algorithm.md`, `docs/algorithm_history.md` | maintainer guide cross-reference rules | no live claim without current validation owner |

## Claim-Sensitive Scan Result

The Day 4 scan did not find an unsupported broad public claim that needs an
immediate wording fix. The sensitive terms found in the scanned files are
framed as explicit limits, residuals, deferred work, or evidence boundaries.

Examples of protected boundaries:

- `docs/api_reference.md` says the API reference does not imply dynamic ABI,
  shared-library, package-manager, broad platform, external-library,
  portable-performance, or state-of-the-art coverage.
- `INSTALL.md` keeps dynamic ABI compatibility, runtime-loader behavior,
  package-manager distribution, static/shared selectors, Windows Makefile
  parity, and Windows `pkg-config` parity out of scope.
- `README.md` keeps Windows package/platform support CMake-first and rejects
  portable performance and broad state-of-the-art readings.
- `tests/corpus/README.md` states that corpus metadata is fixture-local and
  does not claim broad corpus completeness, SuiteSparse parity, external
  parity, broad QR/SVD correctness, package/platform support, portable
  performance, coverage completeness, or state-of-the-art status.

## Documentation Coherence Risks

| Risk | Current mitigation | Day 4 follow-up |
| --- | --- | --- |
| Claim wording is duplicated across README, INSTALL, maintainer guide, solver docs, corpus docs, and benchmark docs. | Maintainer guide owns detailed interpretation and public docs link to it. | Day 11 claim register must name one owner per positive claim. |
| Generated API HTML is documented but not yet a closed publication decision. | API reference says headers are source of truth; maintainer guide defines freshness rules. | Sprint 158 must close or explicitly retain the generated HTML residual. |
| Generated oracle/comparison outputs are local-only while public docs mention their commands. | Docs state generated outputs are ignored/local-only and not hosted proof. | Sprint 159 must decide selected hosted promotion scope. |
| Windows package language can be confused because Windows installs `sparse.pc` metadata but does not claim `pkg-config` execution parity. | INSTALL, README, maintainer guide, and Windows CI comments keep CMake-first boundaries explicit. | Sprint 162 must either promote or strengthen the retained non-claim. |
| Benchmark reports are easy to overread as performance proof. | Benchmark docs and maintainer guide classify rows as local/advisory or bounded guardrails. | Sprint 163 must preserve methodology-bound wording. |
| Historical algorithm docs can be mistaken for current support claims. | Algorithm/history docs state they are not install/support/API/performance guarantees. | Day 11 claim register should avoid using history rows as live claim proof. |

## Sprint 158 Handoff Notes

The generated API reference work should begin from:

- `docs/api_reference.md` as the user-facing API entry point;
- `docs/maintainer_guide.md` lines around API reference and generated Doxygen
  HTML freshness rules;
- `README.md` build docs for `make docs`;
- `Doxyfile`;
- checked-in headers under `include/`;
- generated installed header template `include/sparse_version.h.in`.

Sprint 158 must keep exact declarations anchored to checked-in public headers
unless generated HTML is refreshed, warning-triaged, coverage-checked, and
published according to its selected product decision.

## Day 4 Handoff

Day 5 should consume this claim baseline and capture generated-artifact
evidence in more detail:

- generated API HTML tracking/freshness state;
- corpus manifests, expected rows, optional-data rows, and report-family rows;
- oracle and comparison generated output paths;
- benchmark, sentinel, large-matrix, coverage, dead-code, and package report
  family classifications;
- source-controlled metadata versus ignored generated artifacts.

## Completion Check

- Public documentation inventory is captured.
- Positive claim and non-claim register drafts are captured.
- Support-tier ownership map is captured.
- No immediate unsupported broad claim defect was found in the Day 4 scan.
- Documentation ownership is clear enough for Sprint 158 handoff work.
