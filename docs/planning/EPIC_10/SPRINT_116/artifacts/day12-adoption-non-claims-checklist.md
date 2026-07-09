# Day 12 Adoption Non-Claims Checklist

## Purpose

Day 12 verifies that adoption-facing documentation does not advertise
unreviewed support surfaces. The checklist focuses on Matrix I/O boundaries,
package/platform claims, shared-library and ABI claims, benchmark/performance
claims, and proof-owner/internal-helper promotion into public adoption docs.

## Documents Audited

| Document | Role in audit |
|---|---|
| `README.md` | Front-door claims, support summary, benchmark handoff, install summary. |
| `INSTALL.md` | Static-first install contract, reviewed platform lanes, install-validation non-claims. |
| `docs/tutorial.md` | Adoption workflow examples and Matrix Market ownership. |
| `docs/solver_selection.md` | Solver choice, Matrix Market boundary, benchmark handoff, state-of-the-art non-claims. |
| `docs/matrix_market.md` | Matrix Market public surface and builder/module non-claims. |
| `docs/algorithm.md` | Technical background positioning and performance non-claims. |
| `benchmarks/README.md` | Benchmark local-evidence boundaries and report-lane semantics. |
| `examples/README.md` | Runnable example scope and Matrix Market/builder non-claims. |

## Adoption Non-Claims Checklist

| Non-claim | Status | Evidence | Day 13 action |
|---|---|---|---|
| No separate public Matrix I/O module | Fenced | `docs/matrix_market.md`, `docs/solver_selection.md`, and `examples/README.md` say Matrix Market is public load/save functions, not a separate module. | No edit. |
| No public matrix builder API | Fenced | Matrix Market docs and example docs explicitly reject a public builder API claim. | No edit. |
| No shared-library package support | Fenced | README and `INSTALL.md` say shared-library packaging is deferred and install contract is static archive surface. | No edit. |
| No dynamic ABI guarantee | Fenced | `INSTALL.md` says static-first install/export is not a broad shared-library or dynamic-ABI promise; Day 9 positions `docs/algorithm.md` as not an ABI reference. | No edit. |
| No package-manager support claim | Absent/fenced | Day 5 removed ambiguous README wording; `INSTALL.md` does not advertise Homebrew/vcpkg/distro recipes as supported package-manager channels. | No edit. |
| No expanded Windows install-validation parity | Fenced | README and `INSTALL.md` describe Windows as reviewed CMake subset/CMake-first consumer path, not separate install-validation parity. | No edit. |
| No full macOS install/export parity | Fenced | `INSTALL.md` says reviewed platform claims remain narrower than local scripts and explicitly does not claim full macOS install/export parity. | No edit. |
| No universal benchmark or performance guarantee | Fenced | README, `benchmarks/README.md`, `docs/solver_selection.md`, and `docs/algorithm.md` tie timing to local measured context and reject portable guarantees. | No edit. |
| No maintainer-proof-first adoption path | Fenced | README routes maintainer/quality policy to `docs/maintainer_guide.md`; benchmarks say tests own correctness; algorithm doc is technical background. | No edit. |
| No proof-owner/internal-helper public contract | Fenced | `docs/algorithm.md` is positioned as technical background; examples and solver-selection use public APIs and headers. | No edit. |

## Package And Platform Claim-Fence Table

| Surface | Current adoption wording | Fence quality |
|---|---|---|
| Static install package | README and `INSTALL.md` describe maintained static archive surface with `pkg-config` and `find_package(Sparse)`. | Clear. |
| Shared library | README and `INSTALL.md` say shared-library packaging is deferred. | Clear. |
| Dynamic ABI | `INSTALL.md` says static install/export is not a dynamic-ABI promise. | Clear. |
| Linux | Linux is strongest reviewed source of truth, not a claim that all install paths are reviewed everywhere. | Clear. |
| macOS | Apple Clang reviewed path plus supplemental Homebrew GCC/static-first evidence; no full install/export parity claim. | Clear. |
| Windows | Reviewed CMake subset and CMake-first consumer story; no separate reviewed install-validation lane. | Clear. |
| Package managers | No Homebrew/vcpkg/distro package-manager support claim remains. | Clear. |

## Proof-Owner And Internal-Helper Claim-Fence Table

| Surface | Current adoption wording | Fence quality |
|---|---|---|
| `README.md` | Keeps maintainer policy in `docs/maintainer_guide.md` and uses public workflow language. | Clear. |
| `docs/solver_selection.md` | Directs users to public headers and public workflow routes. | Clear. |
| `examples/README.md` | Describes examples as public-usage references and routes allocation helpers as example-local. | Clear. |
| `benchmarks/README.md` | Says benchmarks are measurement surfaces and tests own regression/oracle/property guarantees. | Clear. |
| `docs/algorithm.md` | Day 9 note says the document is technical background, not first-use adoption guidance, support contract, package/ABI reference, or performance guarantee. | Clear. |

## Day 13 Cleanup List

No required Day 13 wording fixes were found during the Day 12 audit.

Day 13 should still perform a final focused recheck of the touched adoption
surfaces and write the claim-guardrail follow-through artifact. If the recheck
stays clean, Day 13 should be a no-edit follow-through.

## Sprint 117 Residual List

- Optional future split of `docs/algorithm.md` into a concise public algorithm
  reference and historical measurement appendix if scanability remains a
  recurring concern.
- Optional future generated benchmark index work if benchmark report artifacts
  need stronger discoverability.

Neither residual is required to make Sprint 116 adoption non-claims safe.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Adoption non-claims are explicit and auditable | Complete. |
| Sprint 117 has a clear residual list if any claims remain ambiguous | Complete; no ambiguous claims require Sprint 117, only optional scanability/indexing residuals. |
| No implementation or package support claim is introduced | Complete. |

## Validation Notes

- Day 12 changed Sprint 116 planning documentation only.
- Adoption-facing docs were inspected but not edited.
- No `.c` or `.h` files were modified.
