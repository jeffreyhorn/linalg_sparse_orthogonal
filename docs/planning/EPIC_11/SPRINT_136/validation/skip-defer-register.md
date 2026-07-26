# Sprint 136 Validation Skip/Defer Register

## Active After Day 5

| Lane | Status | Reason | Promotion or execution condition |
| --- | --- | --- | --- |
| Full C quality gate | Skipped for Day 5 | No tracked or untracked `.c` or `.h` files changed. | Required immediately if any `.c` or `.h` file changes. |
| Focused public-doc link/path checks | Skipped for Day 5 | Public docs were not changed; only Sprint 136 planning docs changed. | Required if README, INSTALL, docs, examples, or benchmark docs change. |
| Make install/`pkg-config` downstream proof | Deferred | Package/install behavior and metadata were not changed; static deferral proof passed. | Run if package/install/pkg-config surfaces change or final package confidence is selected. |
| CMake configure/build/test proof | Deferred to Day 6 decision | CMake/build surfaces were not changed on Day 5. | Run if CMake/build surfaces change or Day 6 selects optional confidence. |
| CMake install/export proof | Deferred to Day 6 decision | CMake package/export behavior was not changed on Day 5. | Run if CMake package/export surfaces change or Day 6 selects optional confidence. |
| Canonical benchmark report | Deferred to Day 7 | Benchmark/report evidence is runtime-dependent and belongs to supplemental/report validation. | Run if Day 7 needs fresh canonical report evidence and runtime budget permits. |
| Performance sentinels | Deferred to Day 7 | Runtime-dependent local evidence with existing wall-check boundary. | Run if Day 7 needs fresh sentinel evidence and runtime budget permits. |
| Large-matrix guardrails | Deferred to Day 7 | Runtime/data-dependent generated report evidence. | Run if Day 7 needs fresh guardrail evidence and data/runtime budget permits. |
| Dead-code report/check | Deferred | No source or public-surface cleanup needs dead-code context yet. | Run if source/public-surface cleanup requires dead-code report-completeness evidence. |
| Coverage | Deferred | Coverage is supplemental and tree-mutating; no coverage wording changed. | Run only if coverage wording or evidence is explicitly required. |
| Hosted Linux package CI | Deferred to branch/PR CI | Requires GitHub-hosted runner. | Use hosted CI result after push/PR. |
| Hosted macOS package confidence | Deferred to branch/PR CI | Supplemental hosted lane; cannot be proven locally. | Use hosted CI result after push/PR without promoting support tier. |
| Hosted Windows install/downstream confidence | Deferred to branch/PR CI | Supplemental hosted lane; cannot be proven locally. | Use hosted CI result after push/PR without promoting support tier. |
| Windows staged pthread/POSIX tests | Deferred | Staged until source portability or Windows-native replacement exists. | Requires implementation, CTest count update, and hosted MSVC proof. |
| Shared-library, dynamic ABI, runtime-loader, package-manager support proof | Unsupported/deferred | Sprint 133 keeps these as explicit non-claims. | Requires a future product decision and full proof stack. |
| QR residual implementation | Deferred | Sprint 136 publishes the residual queue with promotion criteria, not implementation. | Future-epic promotion criteria must be satisfied first. |

## Active After Day 7

| Lane | Status | Reason | Promotion or execution condition |
| --- | --- | --- | --- |
| Full C quality gate | Skipped for Day 7 | No tracked or untracked `.c` or `.h` files changed. | Required immediately if any `.c` or `.h` file changes. |
| Local CMake configure/build/test | Passed on Day 6 | Local CMake configure/build, CTest registration, and full CTest passed. | Does not replace hosted platform CI or create cross-platform parity. |
| Local CMake install/export proof | Passed on Day 6 | `bash tests/test_cmake_install.sh` passed 21 checks, 0 failures, 0 skips. | Does not promote macOS/Windows supplemental hosted lanes to reviewed support. |
| Local Make install/`pkg-config` proof | Passed on Day 7 | `bash tests/test_install.sh` passed 22 checks, 0 failures. | Static-first local package evidence only. |
| Canonical benchmark report | Passed on Day 7 | `make bench-canonical-report` generated four threshold-free local measurement rows. | Use only as local freshness-scoped measurement evidence. |
| Performance sentinels | Passed on Day 7 | `make performance-sentinels` generated 11 rows; S5 wall-check rows passed and S2 rows are threshold-free. | Use only as local sentinel/report evidence, not portable performance. |
| Large-matrix guardrails | Passed on Day 7 | `make large-matrix-guardrails` generated six rows: four reviewed pass rows and two supplemental skip rows. | Use reviewed rows as bounded structural/report evidence; skipped supplemental rows remain opt-in. |
| Generated report metadata | Captured on Day 7 | Manifests and indexes record branch `sprint-136`, commit `b178de48`, Darwin platform, AppleClang compiler, timestamps, row counts, and non-claim notes. | Day 8 may use freshness context without widening claims. |
| Focused public-doc link/path checks | Still skipped | Public docs were not changed; only Sprint 136 planning docs changed. | Required if README, INSTALL, docs, examples, or benchmark docs change. |
| Dead-code report/check | Deferred | No source or public-surface cleanup needs dead-code context yet. | Run if source/public-surface cleanup requires dead-code report-completeness evidence. |
| Coverage | Deferred | Coverage is supplemental and tree-mutating; no coverage wording changed. | Run only if coverage wording or evidence is explicitly required. |
| Hosted Linux package CI | Deferred to branch/PR CI | Requires GitHub-hosted runner. | Use hosted CI result after push/PR. |
| Hosted macOS package confidence | Deferred to branch/PR CI | Supplemental hosted lane; cannot be proven locally. | Use hosted CI result after push/PR without promoting support tier. |
| Hosted Windows install/downstream confidence | Deferred to branch/PR CI | Supplemental hosted lane; cannot be proven locally. | Use hosted CI result after push/PR without promoting support tier. |
| Windows staged pthread/POSIX tests | Deferred | Staged until source portability or Windows-native replacement exists. | Requires implementation, CTest count update, and hosted MSVC proof. |
| Shared-library, dynamic ABI, runtime-loader, package-manager support proof | Unsupported/deferred | Sprint 133 keeps these as explicit non-claims. | Requires a future product decision and full proof stack. |
| QR residual implementation | Deferred | Sprint 136 publishes the residual queue with promotion criteria, not implementation. | Future-epic promotion criteria must be satisfied first. |

## Active After Day 6

| Lane | Status | Reason | Promotion or execution condition |
| --- | --- | --- | --- |
| Full C quality gate | Skipped for Day 6 | No tracked or untracked `.c` or `.h` files changed. | Required immediately if any `.c` or `.h` file changes. |
| Local CMake configure/build/test | Passed for local confidence | `cmake -S . -B build-sprint136-cmake`, `cmake --build build-sprint136-cmake`, `ctest -N`, and full `ctest` passed locally. | Does not replace hosted platform CI or create cross-platform parity. |
| Local CMake install/export proof | Passed for local confidence | `bash tests/test_cmake_install.sh` passed 21 checks, 0 failures, 0 skips. | Does not promote macOS/Windows supplemental hosted lanes to reviewed support. |
| Focused public-doc link/path checks | Still skipped | Public docs were not changed; only Sprint 136 planning docs changed. | Required if README, INSTALL, docs, examples, or benchmark docs change. |
| Make install/`pkg-config` downstream proof | Still deferred | Package/install behavior and metadata were not changed; static deferral proof passed on Day 5. | Run if package/install/pkg-config surfaces change or final package confidence is selected. |
| Canonical benchmark report | Deferred to Day 7 | Benchmark/report evidence is runtime-dependent and belongs to supplemental/report validation. | Run if Day 7 needs fresh canonical report evidence and runtime budget permits. |
| Performance sentinels | Deferred to Day 7 | Runtime-dependent local evidence with existing wall-check boundary. | Run if Day 7 needs fresh sentinel evidence and runtime budget permits. |
| Large-matrix guardrails | Deferred to Day 7 | Runtime/data-dependent generated report evidence. | Run if Day 7 needs fresh guardrail evidence and data/runtime budget permits. |
| Dead-code report/check | Deferred | No source or public-surface cleanup needs dead-code context yet. | Run if source/public-surface cleanup requires dead-code report-completeness evidence. |
| Coverage | Deferred | Coverage is supplemental and tree-mutating; no coverage wording changed. | Run only if coverage wording or evidence is explicitly required. |
| Hosted Linux package CI | Deferred to branch/PR CI | Requires GitHub-hosted runner. | Use hosted CI result after push/PR. |
| Hosted macOS package confidence | Deferred to branch/PR CI | Supplemental hosted lane; cannot be proven locally. | Use hosted CI result after push/PR without promoting support tier. |
| Hosted Windows install/downstream confidence | Deferred to branch/PR CI | Supplemental hosted lane; cannot be proven locally. | Use hosted CI result after push/PR without promoting support tier. |
| Windows staged pthread/POSIX tests | Deferred | Staged until source portability or Windows-native replacement exists. | Requires implementation, CTest count update, and hosted MSVC proof. |
| Shared-library, dynamic ABI, runtime-loader, package-manager support proof | Unsupported/deferred | Sprint 133 keeps these as explicit non-claims. | Requires a future product decision and full proof stack. |
| QR residual implementation | Deferred | Sprint 136 publishes the residual queue with promotion criteria, not implementation. | Future-epic promotion criteria must be satisfied first. |
