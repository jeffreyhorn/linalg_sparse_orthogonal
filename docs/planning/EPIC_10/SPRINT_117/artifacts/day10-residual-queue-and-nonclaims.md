# Sprint 117 Day 10 Residual Queue And Non-Claims Publication

## Purpose

Day 10 publishes the final Sprint 117 post-Epic residual queue and explicit
non-claim register. It converts the Day 9 intake into a handoff that future
work can start from without rediscovering Sprint 114-116 deferred decisions or
accidentally promoting unsupported Epic 10 claims.

This is a documentation and planning artifact only. It does not implement,
validate, or claim new source, package, ABI, platform, install, benchmark,
workflow, public API, CTest, helper-target, or support behavior.

## Evidence Links

| Evidence | Role |
|---|---|
| `artifacts/day6-final-validation-package.md` | Current Sprint 117 validation and touched-surface proof. |
| `artifacts/day8-final-comparison-cleanup.md` | Final public/support claim cleanup and evidence-bounded claim table. |
| `artifacts/day9-residual-queue-intake.md` | Residual classification source for Day 10 publication. |
| `../SPRINT_114/RETROSPECTIVE.md` | Proof-owner/source-boundary/direct-iterative/SVD deferred debt source. |
| `../SPRINT_115/RETROSPECTIVE.md` | Package/platform/ABI/Windows/package-manager deferred debt source. |
| `../SPRINT_116/RETROSPECTIVE.md` | Adoption scanability and public non-claim deferred debt source. |

## Post-Epic Residual Queue

| ID | Residual | Owner surface | Promotion prerequisite | Status |
|---|---|---|---|---|
| R1 | Move one eigensolver private owner. | Eigensolver source boundary. | Exact old/new files, source-list and CMake updates, focused consumer proof, reviewed CTest count evidence where applicable, and rollback instructions. | Post-Epic residual. |
| R2 | Revisit `s20_select_indices` movement. | Eigensolver selection helpers. | Source-list/build metadata proof covering grow-m, thick-restart, and LOBPCG consumers. | Post-Epic residual. |
| R3 | Revisit `s20_lift_ritz_vectors` movement. | Ritz vector publication helpers. | Shared proven owner for grow-m and thick-restart partial-publication states. | Post-Epic residual. |
| R4 | Revisit shift-invert setup/conversion movement. | Shift-invert and LDLT composition. | LDLT factor lifecycle, `used_csc_path_ldlt`, operator selection, public error propagation, and cleanup ownership proof. | Post-Epic residual. |
| R5 | Revisit `lanczos_iterate_op` movement. | Lanczos iteration helper. | Explicit compile-unit proof for all current consumers. | Post-Epic residual. |
| R6 | Carry Sprint 114 non-package residuals when future package/platform work touches adjacent surfaces. | Cross-surface validation guardrail. | Source-list, CMake, focused consumer, reviewed CTest, and rollback evidence for any promoted item. | Post-Epic residual. |

## Future-Epic Candidates

| ID | Candidate | Owner surface | Required proof before public claim |
|---|---|---|---|
| F1 | Shared direct/iterative generated-RHS oracle for QR, CG, GMRES, BiCGSTAB, and MINRES. | Direct/iterative test proof owners. | Common ownership design, focused direct/iterative tests, tolerance policy, and reviewed validation. |
| F2 | Shared SVD proof-helper owner. | SVD test proof owners. | Storage, leading-dimension, product-dimension, fixture, and threshold proof for reconstruction, U/Vt orthogonality, Moore-Penrose, low-rank, sparse-vs-dense, and condition-number helpers. |
| F3 | Reviewed Linux install CI lane. | Package/install and CI. | Accepted runtime/dependency ownership, reviewed CI lane, support wording update, and install evidence. |
| F4 | Reviewed macOS CMake install/export parity. | macOS package/platform support. | Reviewed CI proof for `cmake --install`, installed CMake package files, downstream `find_package(Sparse)`, exact-version behavior, and static artifact shape. |
| F5 | Windows install-validation lane. | Windows package/platform support. | MSVC `cmake --install`, downstream installed target lookup, compile/link/run proof, reviewed-count clarity, and public non-claims. |
| F6 | Windows thread/fuzz/property proof split or port. | Windows validation and CTest ownership. | Native Windows thread/temp-file behavior and explicit CTest count updates. |
| F7 | Shared-library and dynamic ABI support. | Packaging, loader, ABI, and platform ownership. | Build rules, package metadata, runtime-loader proof, symbol policy, versioning policy, ABI tests, and platform ownership. |
| F8 | Package-manager support. | Packaging and downstream consumer support. | Real recipes and install/consumer proof for each claimed manager/platform. |

## Optional Scanability Work

| ID | Optional work | Owner surface | Guardrail |
|---|---|---|---|
| O1 | Split `docs/algorithm.md` into a concise public algorithm reference plus a historical measurement appendix. | Public and technical documentation. | Preserve technical-reference history, adoption routing, and local-performance caveats. |
| O2 | Add generated benchmark artifact indexes in public or maintainer docs. | Benchmark docs and generated report metadata. | Define generated index ownership without changing benchmark semantics or portable-performance claims. |

## Explicit Non-Claim Register

These items remain unclaimed at Epic 10 closeout unless future work implements
and validates them with matching public-claim cleanup:

- unqualified state-of-the-art replacement for established sparse linear
  algebra ecosystems;
- SuiteSparse, PETSc, Trilinos, ARPACK, LAPACK, SciPy, NumPy, or vendor
  backend parity;
- every-family external solver validation;
- broad direct/iterative oracle ownership;
- broad SVD proof abstraction ownership;
- portable performance superiority;
- universal reorder/fill superiority;
- cross-platform max-RSS or timing thresholds;
- reviewed Linux install CI lane;
- full reviewed macOS CMake install/export parity;
- Windows install-validation parity;
- Windows thread/fuzz/property parity;
- Windows Makefile parity;
- macOS coverage reviewed-lane parity;
- Homebrew GCC reviewed-lane promotion;
- shared-library package support;
- dynamic ABI compatibility guarantee;
- package-manager support;
- public Matrix I/O module;
- public Matrix Market builder API;
- proof-owner/internal-helper public contract expansion;
- public API or install-header expansion;
- source-list, helper-target, or reviewed CTest membership change from Sprint
  117;
- complete closure of all proof-owner or source-boundary debt.

## Consciously Closed Work

The following Sprint 114-116 work is not carried as unresolved Sprint 117 debt:

- Sprint 114 residual intake, duplicate fencing, eigensolver proof designs and
  implementations, direct/iterative exact-RHS cleanup, bounded SVD proof-owner
  cleanup, validation, metrics, and closeout handoff.
- Sprint 115 package/platform intake, Linux/macOS/Windows deferral decisions,
  Windows thread/fuzz staged-exclusion follow-through, shared-library/dynamic
  ABI product-contract decision, package-manager support decision, and
  validation handoff.
- Sprint 116 adoption QA intake, external-reference inventory and QA,
  README/benchmark/algorithm/performance follow-through, non-claims checklist,
  validation, and handoff.

## Residual Owner And Dependency Notes

- Proof-owner/source-boundary residuals require source-list, build metadata,
  CMake, consumer, CTest, and rollback evidence before they can move from
  residual to implementation.
- Package/platform residuals require platform-specific install/export or CI
  proof before they can affect support wording.
- ABI and package-manager residuals require product-contract work before any
  public install or versioning claim can change.
- Benchmark/report scanability work must not change benchmark semantics or
  local-measurement caveats unless a separate validation contract is added.
- Public docs should continue to route unsupported behavior to the non-claim
  register rather than burying it in sprint history.

## Item 5 Closeout

| Requirement | Status | Evidence |
|---|---|---|
| Publish post-Epic residual queue. | Complete. | Post-Epic residual queue above. |
| Publish explicit non-claims. | Complete. | Explicit non-claim register above. |
| Link residuals to final validation and claim cleanup evidence. | Complete. | Evidence links and Day 8/Day 9 references. |
| Cross-check Sprint 114-116 deferred debt. | Complete. | Source-specific residuals and consciously closed work sections. |
| Mark residuals by disposition. | Complete. | Post-Epic residual, future-epic candidate, optional scanability, consciously closed, and explicit non-claim sections. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 5 is complete. | Complete. |
| Future work can start from an explicit residual queue. | Complete. |
| Epic 10 closes without implying deferred support or implementation claims. | Complete. |
