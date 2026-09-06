# Day 4 Public Claim Surface Audit

## Purpose

Day 4 audits public-facing documentation before any claim recalibration edits.
It compares current public docs against the Day 2 outcome ledger and Day 3
evidence boundaries.

## Audited Public Surfaces

| Surface | Current role | Finding |
| --- | --- | --- |
| `README.md` | Top-level adoption route and capability summary. | Broad but calibrated; future evidence promotions should update it by linking to support owners instead of duplicating full caveats. |
| `INSTALL.md` | Public support/readiness owner. | Strong current source of truth for static-first install, package deferral, Windows boundaries, selected evidence, generated API local-only status, and non-claims. |
| `examples/README.md` | First-run example route. | Should remain adoption-focused and not become a support or benchmark claim owner. |
| `docs/cookbook.md` | Data-format and workflow recipes. | Good recipe surface; support claims should route back to INSTALL. |
| `docs/tutorial.md` | Guided API usage. | Keep package, platform, and performance claims out unless linked to support truth. |
| `docs/solver_selection.md` | Problem-shape decision tree and selected evidence notes. | Claim-sensitive; update named target/platform statements only with selected evidence. |
| `benchmarks/README.md` | Benchmark command and methodology owner. | Correctly preserves local, methodology-bound, threshold-free benchmark interpretation. |
| `docs/api_reference.md` | API reference and generated-doc routing. | Must remain local-only until Sprint 204 decides otherwise. |
| `tests/corpus/README.md` | Corpus and report evidence interpretation. | Must change with selected target manifest/report freshness promotions. |
| `tests/corpus/schemas/report_index_fields.md` | Report-index schema and policy owner. | Strong row-level owner for support tier, claim scope, freshness, and non-claim semantics. |

## Overclaim And Stale-Claim Risks

| Area | Current status | Risk | Evidence required before promotion |
| --- | --- | --- | --- |
| Package-manager/Homebrew | Not claimed. | Formula proof material could be read as support. | Approved metadata, passing proof, package guards, install checks, docs. |
| Windows selected Cholesky freshness | Guarded workflow only. | Workflow path could be read as promoted freshness. | Hosted evidence, artifact inspection, manifest metadata, workflow/normalizer tests. |
| Windows QR incompatible comparison | Local-only selected baseline. | Linux/macOS evidence could be inherited onto Windows. | MSVC/CMake proof, hosted Windows artifacts, manifest promotion, QR tests. |
| Benchmarks | Methodology-bound and non-portable. | Additional hosted rows could be read as portable performance. | Platform/row metadata, hosted bundle, freshness checks, retained non-portable docs. |
| Generated API | Local-only. | Doxygen output could be read as hosted publication. | Explicit publication/local-only decision and matching guards. |
| Allocation failure | Selected owner proofs only. | One new proof could be read as broad reliability. | Owner-specific invariant, deterministic tests, focused gate, full C gate when needed. |
| Shared library/ABI | Deferred static-first only. | Install/CMake updates could imply ABI support. | Shared-library design, ABI policy, compatibility tests, docs. |
| State-of-the-art/release | Not claimed. | Closeout language could become release or state-of-the-art posture. | Broad comparative correctness, performance, package, ABI, platform, release, and ecosystem evidence. |

## Routing Plan

| Claim type | Public source of truth | Supporting docs |
| --- | --- | --- |
| Support/readiness | `INSTALL.md#support-readiness-matrix` | README, examples, solver-selection, maintainer guide. |
| Benchmark interpretation | `benchmarks/README.md` | README, INSTALL, solver-selection. |
| API/generated docs | `docs/api_reference.md` plus INSTALL support row | Maintainer guide and Doxygen policy docs. |
| Selected report rows | Corpus README/schema docs and selected target manifest | README, INSTALL, solver-selection. |
| Package proof | INSTALL and packaging Homebrew docs | README and maintainer guide. |
| Maintainer proof interpretation | `docs/maintainer_guide.md` | Public docs link only when proof semantics matter. |

## Edit Plan For Later Days

- Do not edit public claims until outcome evidence exists.
- If package proof closes, update README/INSTALL/Homebrew docs with only the
  exact provider support level earned.
- If selected Windows freshness closes, update INSTALL, README, solver
  selection, corpus docs, and schema docs with exact target/platform metadata.
- If hosted benchmark freshness expands, update benchmark docs and INSTALL
  without portable performance claims.
- If generated API publication policy changes, update API reference, INSTALL,
  README routing, and maintainer guidance together.
- If support/adoption docs are consolidated, preserve `INSTALL.md` as public
  support truth unless Sprint 205 explicitly changes ownership.

## Acceptance Evidence

- Public claim surfaces are classified.
- Overclaim and stale risks are listed with required evidence.
- Duplicate caveat routing is documented.
- Later edit triggers identify the public docs and guardrails that must change
  together.

