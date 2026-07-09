# Sprint 117 Day 9 Residual Queue Intake

## Purpose

Day 9 builds the post-Epic residual intake from Sprint 114, Sprint 115, and
Sprint 116 deferred-debt sections. It assigns a disposition to every named
residual before Day 10 publishes the final residual queue and explicit
non-claims.

This artifact is classification only. It does not promote implementation,
package/platform, ABI, install, benchmark, workflow, public API, or support
claims.

## Source Inputs

| Source | Residual focus | Day 9 use |
|---|---|---|
| `SPRINT_114/RETROSPECTIVE.md` | Proof-owner, eigensolver source boundary, direct/iterative oracle, SVD helper, and future touched-surface validation. | Classify proof-owner and source-boundary carry-forward work. |
| `SPRINT_115/RETROSPECTIVE.md` | Linux install CI, macOS install/export, Windows install/thread/fuzz/property, shared-library ABI, package-manager, and Sprint 114 carry-forward. | Classify package/platform and ABI carry-forward work. |
| `SPRINT_116/RETROSPECTIVE.md` | Adoption scanability, generated benchmark indexes, and adoption/support non-claims. | Classify adoption/documentation carry-forward work and non-claims. |
| `SPRINT_117/artifacts/day8-final-comparison-cleanup.md` | Final evidence-bounded claim table and no-edit cleanup record. | Prevent any residual from becoming a public claim without validation. |

## Disposition Definitions

| Disposition | Meaning |
|---|---|
| Post-Epic residual | Valid carry-forward debt that should remain visible after Epic 10 closes. |
| Future-epic candidate | Larger product work that needs a dedicated plan, implementation, validation, and claim cleanup before promotion. |
| Optional scanability work | Documentation/report discoverability improvement that is useful but not a support-safety blocker. |
| Consciously closed | Work completed or intentionally not carried as unresolved Sprint 117 debt. |
| Explicit non-claim | Unsupported support/product claim that must remain unclaimed until future proof exists. |

## Sprint 114 Residual Disposition

| Residual | Disposition | Dependency / proof required before promotion |
|---|---|---|
| Move one eigensolver private owner. | Post-Epic residual. | Exact old/new files, source-list and CMake updates, focused consumer proof, reviewed CTest count evidence where applicable, and rollback instructions. |
| Revisit `s20_select_indices` movement. | Post-Epic residual. | Source-list/build metadata proof covering grow-m, thick-restart, and LOBPCG consumers. |
| Revisit `s20_lift_ritz_vectors` movement. | Post-Epic residual. | Shared proven owner for grow-m and thick-restart partial-publication states. |
| Revisit shift-invert setup/conversion movement. | Post-Epic residual. | LDLT factor lifecycle, `used_csc_path_ldlt`, operator selection, public error propagation, and cleanup ownership proof. |
| Revisit `lanczos_iterate_op` movement. | Post-Epic residual. | Explicit compile-unit proof for all current consumers. |
| Decide whether QR, CG, GMRES, BiCGSTAB, and MINRES generated-RHS setup can share a direct/iterative oracle. | Future-epic candidate. | Common ownership design plus focused direct/iterative tests and validation. |
| Decide whether SVD proof helpers can share an owner. | Future-epic candidate. | Storage, leading-dimension, product-dimension, fixture, and threshold proof for reconstruction, U/Vt orthogonality, Moore-Penrose, low-rank, sparse-vs-dense, and condition-number helpers. |
| Validate package, ABI, Windows, CMake parity, install-header, and adoption surfaces when future work touches them. | Explicit non-claim guardrail. | Run the relevant lane whenever those surfaces change; do not infer proof from Sprint 117 docs. |
| Sprint 114 completed proof designs, focused implementations, validation, metrics, and closeout handoff. | Consciously closed. | Already completed in Sprint 114; not carried as unresolved Sprint 117 debt. |

## Sprint 115 Residual Disposition

| Residual | Disposition | Dependency / proof required before promotion |
|---|---|---|
| Promote Linux install proof to reviewed CI. | Future-epic candidate. | Accepted runtime/dependency ownership, reviewed CI lane, support wording update, and install evidence. |
| Promote macOS CMake install/export parity. | Future-epic candidate. | Reviewed CI proof for `cmake --install`, installed CMake package files, downstream `find_package(Sparse)`, exact-version behavior, and static artifact shape. |
| Add Windows install-validation. | Future-epic candidate. | MSVC `cmake --install`, downstream installed target lookup, compile/link/run proof, reviewed-count clarity, and non-claims. |
| Port or split Windows thread/fuzz/property proof. | Future-epic candidate. | Native Windows thread/temp-file behavior and explicit CTest count updates. |
| Add shared-library/dynamic ABI support. | Future-epic candidate. | Build rules, package metadata, runtime-loader proof, symbol policy, versioning policy, ABI tests, and platform ownership. |
| Add package-manager support. | Future-epic candidate. | Real recipes and install/consumer proof for each claimed manager/platform. |
| Carry Sprint 114 non-package residuals unless explicitly promoted. | Post-Epic residual. | Use the Sprint 114 dependency gates above. |
| Sprint 115 package/platform intake, deferral decisions, and validation handoff. | Consciously closed. | Already completed as Sprint 115 planning/decision work. |

## Sprint 116 Residual Disposition

| Residual | Disposition | Dependency / proof required before promotion |
|---|---|---|
| Split `docs/algorithm.md` into a concise public algorithm reference plus a historical measurement appendix. | Optional scanability work. | Future docs sprint should preserve technical-reference history, adoption routing, and local-performance caveats. |
| Add generated benchmark artifact indexes in public or maintainer docs. | Optional scanability work. | Future benchmark/docs sprint should define generated index ownership without changing benchmark semantics or portable-performance claims. |
| Keep adoption/support non-claims for Linux install CI, macOS install/export, Windows install/thread/fuzz/property/Makefile, macOS coverage, Homebrew GCC reviewed-lane promotion, shared-library, dynamic ABI, package-manager, Matrix I/O module/builder API, proof-owner public contract, source-list/helper-target/CTest membership, and implementation changes. | Explicit non-claim guardrail. | Must remain unclaimed until implementation and validation proof exists. |
| Sprint 116 adoption QA intake, external reference QA, README/benchmark/algorithm/performance follow-through, non-claims checklist, validation, and handoff. | Consciously closed. | Already completed in Sprint 116; not carried as unresolved Sprint 117 debt. |

## Consolidated Residual Intake

| Queue | Included items | Day 10 publication handling |
|---|---|---|
| Proof-owner and source-boundary post-Epic residuals | Eigensolver owner movement, `s20_select_indices`, `s20_lift_ritz_vectors`, shift-invert movement, `lanczos_iterate_op`, Sprint 114 carry-forward gates. | Publish as post-Epic residuals with exact promotion prerequisites. |
| Solver oracle future-epic candidates | Shared direct/iterative generated-RHS oracle and broader SVD helper ownership. | Publish as future-epic candidates, not completed Epic 10 claims. |
| Package/platform and ABI future-epic candidates | Linux install CI, macOS install/export parity, Windows install validation, Windows thread/fuzz/property portability, shared-library/dynamic ABI, package-manager support. | Publish as future-epic candidates and explicit non-claims. |
| Adoption and report scanability | Algorithm-doc split and generated benchmark artifact indexes. | Publish as optional scanability work. |
| Explicit non-claims | Deferred package/platform, ABI, Windows, package-manager, Matrix I/O module/builder, proof-owner public contract, implementation/source-list/helper-target/CTest changes. | Publish in Day 10 non-claim register. |
| Closed prior-sprint work | Sprint 114 proof work, Sprint 115 decisions, Sprint 116 adoption QA and cleanup. | Mark consciously closed; do not duplicate in residual queue. |

## Promotion Guardrails

- No residual is promoted during Sprint 117 Day 9.
- Any future implementation residual needs implementation proof, focused tests,
  reviewed validation, and public-claim cleanup.
- Any future package/platform residual needs install/export or CI proof for the
  exact platform and support wording being claimed.
- Any future benchmark/report residual must preserve local-measurement wording
  unless it introduces a separately validated portable-performance contract.
- Any future public API, install-header, source-list, helper-target, CTest, or
  workflow change must carry matching validation and rollback evidence.

## Day 10 Publication Checklist

- Publish the proof-owner/source-boundary post-Epic residual queue.
- Publish direct/iterative oracle and SVD helper work as future-epic
  candidates.
- Publish package/platform, ABI, Windows, and package-manager work as
  future-epic candidates and explicit non-claims.
- Publish algorithm-doc split and generated benchmark indexes as optional
  scanability work.
- Publish a final explicit non-claim register that matches Day 8 public/support
  cleanup decisions.
- Keep consciously closed Sprint 114-116 work out of the unresolved queue.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| All named residuals from Sprint 114-116 have a disposition. | Complete. |
| No deferred item is silently dropped. | Complete. |
| No residual is promoted without validation and public claim cleanup. | Complete. |
