# Day 13 Claim Closure And Sprint 143 Handoff

## Purpose

Day 13 closes the Sprint 142 claim surface before final closeout. It compares
the sprint outcomes against the Day 1 scope, lists only the runtime/backend
claims earned by implemented and validated work, preserves remaining
non-claims, and hands concrete package/ABI prerequisites to Sprint 143.

## Sprint 142 Outcome Against Scope

| Project-plan item | Day owner(s) | Outcome | Evidence |
| --- | --- | --- | --- |
| Runtime Control Audit | Days 1-3 | Complete | Day 2 inventory and Day 3 dispatch audit classify public typed controls, maintainer/env controls, build flags, report controls, defaults, fallback paths, and evidence owners. |
| Precedence Contract | Days 4-5 | Complete | Day 4 contract and Day 5 validation ledger define explicit typed-option precedence, AUTO/default behavior, compatibility env boundaries, invalid typed values, and fallback vocabulary. |
| Typed-Control Batch | Days 6-7 | Complete as explicit non-expansion | Day 6/7 artifacts record that no new public typed API was promoted; existing public typed controls remain the supported surface and other controls are explicitly maintainer-only or deferred. |
| Sentinel Expansion | Days 8-9 | Complete | `S3` LDLT KKT advisory sentinel rows were added to `make performance-sentinels`, normalized synthetic coverage was added, and `S5`/`S2` semantics were preserved. |
| Docs and Examples | Day 10 | Complete | README, cookbook, algorithm docs, maintainer guide, benchmark docs, and report-index schema wording now distinguish public controls, maintainer controls, and local sentinel meaning. |
| Validation | Days 11-12 | Complete | Focused runtime/backend tests, sentinel generation, report-index freshness, schema checks, script checks, `make format && make lint`, and repository hygiene passed. |
| Closeout | Days 13-14 | In progress | This artifact publishes earned claims, non-claims, and Sprint 143 prerequisites; Day 14 owns the final consistency pass. |

## Earned Runtime/Backend Claims

| Earned claim | Evidence | Boundary |
| --- | --- | --- |
| Public runtime/backend selection is documented as the existing typed option surface. | Day 10 docs list `sparse_cholesky_opts_t.backend`, `sparse_ldlt_opts_t.backend`, `sparse_eigs_opts_t.backend`, and `sparse_analysis_opts_t.reorder_opts`; Day 5 focused tests passed for Cholesky, LDLT, eigensolver, and analysis/reorder owners. | This is a public API/control claim only for the named typed options, not for environment variables or package/link settings. |
| Explicit typed options take precedence over AUTO/default or compatibility behavior where the existing API supports explicit selection. | Day 4 precedence contract; Day 5 focused validation for Cholesky, LDLT, eigensolver AUTO/forced routing, and analysis typed-vs-env precedence. | This does not change ABI or add new typed fields. |
| Maintainer/runtime environment controls are intentionally not public typed API unless separately promoted. | Day 6 candidate matrix and Day 7 implementation artifact record explicit deferrals for dense-helper selectors, SVD low-rank env selection, FM/debug/profile variables, OpenMP runtime context, package/link controls, and benchmark/test opt-ins. | These controls remain compatibility, diagnostic, build, or report context. |
| `make performance-sentinels` now emits local advisory LDLT KKT backend context through `S3`. | Day 9 implementation updated `scripts/performance_sentinels.sh`, `Makefile`, normalizer tests, benchmark docs, and maintainer docs; Day 11/12 normalization reported 21 sentinel rows. | `S3` is threshold-free local evidence, not a hard gate or portable performance proof. |
| `S5` remains the only hard local performance sentinel gate while `S2` and `S3` remain advisory. | Day 8 design, Day 9 implementation, Day 10 docs, and Day 12 freshness checks preserve hard-gate/advisory separation. | Hard-gate status applies only to the existing local wall-check sentinel semantics. |
| Runtime/backend report rows can be discovered and freshness-checked through the normalized report index. | Day 11/12 ran sentinel and combined freshness checks successfully; `tests/test_normalize_report_index.py` covers synthetic `S3` row parsing and backend request/selected/fallback preservation. | Freshness diagnostics do not convert local timing rows into release or platform proof. |

## Remaining Non-Claims

| Non-claim | Current status | Owner |
| --- | --- | --- |
| Shared-library ABI support | Not claimed. Current install/docs remain static-first and reject unsupported shared-library expectations. | Sprint 143 package/ABI decision. |
| Dynamic loader behavior or long-term ABI compatibility | Not claimed. No runtime-loader or symbol-versioning proof was added. | Sprint 143 package/ABI decision. |
| Package-manager availability | Not claimed. Existing install proof is local/static package surface, not distro/Homebrew/vcpkg/MSI packaging. | Sprint 143 or later distribution work. |
| Broad backend portability | Not claimed. Backend evidence is scoped to documented typed options, local build flags, and fallback context. | Future backend/platform owners. |
| Platform parity | Not claimed. Sprint 142 did not promote macOS or Windows support tiers. | Sprint 144 platform promotion lane. |
| Portable performance | Not claimed. `S2` and `S3` are local advisory rows; `S5` remains a local wall-check hard gate. | Benchmark/report maintainers. |
| Optional dense-kernel availability | Not claimed. Dense helper env selectors remain maintainer/compatibility controls with fallback context. | Future typed-control or backend owner. |
| State-of-the-art sparse linear algebra status | Not claimed. Sprint 142 improves governance and local evidence only. | Epic-level competitive evidence owner. |

## Sprint 143 Package/ABI Prerequisites

Sprint 143 should start from the following concrete inputs rather than vague
runtime/backend debt:

| Prerequisite | Required Sprint 143 action | Source evidence |
| --- | --- | --- |
| Runtime/backend public-control boundary | Treat existing typed controls as caller-facing behavior and env/build/report controls as non-API unless Sprint 143 explicitly promotes package-relevant controls. | Day 6, Day 7, and Day 10 artifacts. |
| Static-first install baseline | Audit current static archive, headers, `pkg-config`, CMake export, exact-version package config, and shared-library rejection before deciding whether to keep or widen the package contract. | `README.md`, `INSTALL.md`, existing install scripts, Sprint 143 project-plan item 1. |
| Sentinel non-claim boundary | Do not use `S2`/`S3` timing rows as package, ABI, platform, or portable performance proof. | Day 8-12 sentinel artifacts. |
| Build/report controls | Keep `SPARSE_OPENMP`, `OMP_NUM_THREADS`, package/link settings, and benchmark opt-ins as build/report context unless a package decision explicitly owns them. | Day 2 inventory and Day 10 docs. |
| CMake/pkg-config consumer proof | Revalidate downstream consumer workflows as package evidence, separately from runtime/backend sentinel evidence. | Sprint 143 item 4 and existing `tests/test_install.sh` / `tests/test_cmake_install.sh`. |
| Shared-library decision gate | Choose one path: implement shared-library ABI support with symbol/export/loader proof, or strengthen static-first-only deferral wording and guards. | Sprint 143 project-plan items 1-3. |
| Platform tier dependency | Avoid using Sprint 143 package work to imply macOS/Windows reviewed parity; route platform promotion to Sprint 144. | Sprint 143 and Sprint 144 project-plan prerequisites. |

## Residual Risk Register

| Risk | Impact | Stop condition |
| --- | --- | --- |
| Sprint 143 may conflate static install proof with ABI support. | Unsupported shared-library or dynamic-ABI claims could leak into docs or package metadata. | Stop if package wording claims shared ABI before export, loader, versioning, and downstream proof exist. |
| Package proof could reuse runtime sentinel rows as performance evidence. | Local timing rows could be overread as release-quality performance proof. | Stop if `S2` or `S3` appears in package/ABI docs as proof beyond local advisory context. |
| Platform support wording could widen during package CI updates. | macOS/Windows supplemental confidence could become accidental reviewed parity. | Stop if package CI text claims platform parity outside the reviewed lane selected by Sprint 144. |
| Environment controls could be promoted implicitly through package docs. | Maintainer-only variables could become perceived public API. | Stop if env/build/report controls are documented as package-stable public ABI without a typed-control decision. |

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Runtime/backend claims are backed by specific evidence. | Complete | Earned claims above each cite implementation, docs, tests, sentinels, or report-index validation. |
| Residual non-claims are explicit. | Complete | Non-claims table preserves package/ABI, platform, performance, backend portability, and state-of-the-art boundaries. |
| Sprint 143 receives package/ABI prerequisites rather than vague runtime debt. | Complete | Handoff table names the exact audit and decision inputs for Sprint 143. |
