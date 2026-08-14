# Sprint 156 Day 10: Public Claim And Non-Claim Audit

## Purpose

Audit public and support documentation for unsupported state-of-the-art,
external parity, package, platform, performance, ABI, and report wording. The
goal is to confirm that every public claim is either evidence-bound or stated
as an explicit non-claim before final Epic 13 closeout.

## Inputs Reviewed

- `README.md`
- `INSTALL.md`
- `docs/tutorial.md`
- `docs/cookbook.md`
- `docs/solver_selection.md`
- `docs/api_reference.md`
- `examples/README.md`
- `benchmarks/README.md`
- `tests/corpus/README.md`
- `docs/maintainer_guide.md`
- public headers under `include/`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day2-evidence-inventory.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day5-package-validation.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day6-platform-reconciliation.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day7-corpus-report-validation.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day8-comparison-reconciliation.md`
- `docs/planning/EPIC_13/SPRINT_156/artifacts/day9-adoption-api-reconciliation.md`

## Checks Run

| Check | Result | Notes |
| --- | --- | --- |
| Public claim scan for state-of-the-art, parity, performance, ABI, package, shared-library, runtime-loader, package-manager, Windows Makefile, Windows `pkg-config`, and broad platform wording | Passed | Matches in public docs and public headers were evidence-bound statements or explicit non-claims. |
| Public support/report scan for reviewed, supplemental, support tier, local-only, freshness, report-index, and generated-report wording | Passed | Matches mapped to Day 5-9 evidence boundaries and did not widen the claim surface. |
| Maintainer claim-boundary scan | Passed | Maintainer guidance preserved reviewed/supplemental ownership and explicit non-claim language. |
| `INSTALL.md` support and package inspection | Passed | Static-first package wording, support-tier table, and install validation notes were mutually consistent. |

No `.c` or public `.h` edits were made on Day 10.

## Public Claim Inventory

| Claim family | Current public reading | Evidence source | Day 10 result |
| --- | --- | --- | --- |
| Front-door adoption | README routes users to examples, cookbook, solver selection, install, benchmarks, API reference, and maintainer evidence without competing first-use paths. | Day 9 adoption/API reconciliation. | Evidence-bound. |
| Static package surface | Installed package support is static-first through Make, CMake install/export, `pkg-config`, and exact-version metadata. | Day 5 package validation and Day 6 platform reconciliation. | Evidence-bound. |
| Platform support | Linux is the strongest reviewed source of truth; macOS has reviewed Apple Clang plus reviewed static-first install/export; Windows is reviewed CMake-first with CTest and CMake install/downstream validation. | Day 6 platform reconciliation. | Evidence-bound. |
| QR corpus/report evidence | Maintained QR rows support fixture-local rank, nullity, nullspace, minimum-norm, residual, norm, and selected value behavior. | Day 7 corpus/report validation. | Evidence-bound and local-only. |
| Partial-SVD corpus/report evidence | Maintained partial-SVD rows support fixture-local top-k, rank, projector, residual, orthogonality, sparse-output, fail-closed, and recovery behavior. | Day 7 corpus/report validation. | Evidence-bound and local-only. |
| QR comparison study | One narrow `qr_underdetermined_minnorm_2x4` minimum-norm comparison agrees with the selected source-controlled dense reference helper. | Day 8 comparison reconciliation. | Evidence-bound and local-only. |
| API/header surface | Checked-in headers remain declaration source of truth; generated API HTML remains a convenience view with documented refresh residuals. | Day 9 adoption/API reconciliation and Sprint 155 preservation artifacts. | Evidence-bound with residuals explicit. |
| Benchmarks and reports | Benchmark/report rows are local measurements, freshness diagnostics, and artifact indexes. | Day 7 report validation, Day 8 comparison reconciliation, and `benchmarks/README.md`. | Evidence-bound; no portable performance claim. |

## Unsupported Claim Correction List

No public documentation correction was required on Day 10. The focused scans
found broad phrases only where the docs already reject or bound them. Examples
include static/shared and ABI wording in `INSTALL.md`, platform and
performance caveats in `README.md`, generated-report caveats in
`benchmarks/README.md`, and fixture-local corpus caveats in
`tests/corpus/README.md`.

## Final Non-Claim Register

The following remain blocked from public product claims:

- unqualified state-of-the-art sparse linear algebra status;
- broad ecosystem parity or external-library parity against LAPACK, NumPy,
  SciPy, SuiteSparse, Eigen, PETSc, Trilinos, or package-manager ecosystems;
- portable performance superiority, portable timing guarantees, or portable
  iteration-count guarantees;
- shared-library support, dynamic ABI compatibility, runtime-loader behavior,
  Linux SONAME policy, macOS install-name/RPATH policy, Windows DLL/import
  library support, or static/shared selectors;
- package-manager distribution support through Homebrew, apt, dnf, pacman,
  vcpkg, Conan, or similar channels;
- Windows Makefile parity, Windows `pkg-config` execution parity, or broad
  Windows platform parity;
- hosted CI proof for the local-only corpus, oracle, and comparison rows;
- generated API HTML completeness or freshness beyond the documented source
  header and `make docs` refresh boundary;
- broad QR correctness, broad rank-threshold policy, raw Q/R basis identity,
  sign/orientation, pivot-order, reorder, sparse-mode, or global
  minimum-norm/nullspace behavior;
- broad partial-SVD correctness, raw singular-vector identity,
  repeated-spectrum basis behavior, convergence-rate guarantees,
  partial-result guarantees, or sparse-output/drop-tolerance optimality.

## Documentation Patch List

No documentation patches were needed for Day 10. The existing public docs
already keep claim wording inside the Day 1-9 evidence boundary.

## Support-Tier And Package Consistency

The public support-tier and package wording is internally consistent:

- Linux remains the strongest reviewed source of truth and includes reviewed
  static-first package-contract validation.
- macOS carries reviewed Apple Clang coverage and reviewed static-first
  Make install/`pkg-config` plus CMake install/export proof, with Homebrew GCC
  treated as supplemental.
- Windows carries reviewed MSVC CMake-first CTest and CMake install/downstream
  validation for the maintained static package surface.
- Static-first package claims are real and maintained, but they do not imply
  shared-library, dynamic ABI, runtime-loader, package-manager, Windows
  Makefile, Windows `pkg-config`, or broad platform parity support.
- Generated corpus/report/comparison rows remain local-only until a future
  sprint promotes them to reviewed hosted lanes and updates support tiers.

## Residual Queue

| Residual | Owner | Promotion criteria |
| --- | --- | --- |
| Broad state-of-the-art claim | Epic/product owner | Requires competitive comparison breadth, hosted platform evidence, package maturity, ABI decision, performance methodology, and documented support policy. |
| External-library parity claims | Solver and comparison owners | Add bounded fixture families one at a time with selected external baselines, provenance, tolerance policy, hosted evidence, and explicit non-claim wording. |
| Portable performance claims | Benchmark owner | Define workload suite, hardware/compiler matrix, variance policy, thresholds, regression gates, and recurring hosted publication. |
| Shared-library and dynamic ABI claims | Package/ABI owner | Decide product support, export/import policy, symbol visibility, versioning, loader metadata, installed shared consumers, and cross-platform validation. |
| Package-manager support | Package owner | Add selected channels, metadata ownership, install validation, release workflow, and support policy. |
| Hosted local-only report promotion | CI, corpus, comparison, and report owners | Promote selected freshness gates into reviewed hosted lanes, then update support tiers and public wording. |
| Generated API HTML completeness | Documentation/API owner | Run `make docs`, triage warnings, verify header-page coverage, and commit refreshed generated output. |

## Completion Criteria Check

- Every audited public claim maps to evidence or is already framed as a
  non-claim.
- Broad state-of-the-art and ecosystem parity claims remain blocked.
- Support-tier and package wording are internally consistent.
- No public documentation correction was required.
- Day 11 can use this claim audit as the final wording baseline for closeout
  risk reconciliation.
