# Day 13: Sprint 114 Package/Platform Residual Intake

## Purpose

Day 13 re-reads Sprint 114 residual deferred debt and the Epic 10 deferral
decision, then records which residuals Sprint 115 consumed and which residuals
must remain outside Sprint 115. The goal is to prevent package/platform work
from being forgotten while also preventing source-boundary and proof-owner
debt from being pulled into the wrong sprint.

## Inputs

- `docs/planning/EPIC_10/SPRINT_114/RETROSPECTIVE.md`
- `docs/planning/EPIC_10/PROJECT_PLAN.md`
- Sprint 115 artifacts for Days 1-12
- Sprint 115 working notes

## Sprint 114 Residual Routing

| Sprint 114 residual | Sprint 115 handling |
|---|---|
| Validate package, ABI, Windows, CMake parity, install-header, and adoption surfaces when those surfaces change | Consumed as package/platform claim-fence work across Sprint 115 Days 1-12. |
| Move one eigensolver private owner | Deferred out of Sprint 115; belongs to Sprint 117 residual queue or post-Epic work unless explicitly promoted with full build/source-list proof. |
| Revisit `s20_select_indices` movement | Deferred out of Sprint 115; requires grow-m, thick-restart, and LOBPCG consumer proof. |
| Revisit `s20_lift_ritz_vectors` movement | Deferred out of Sprint 115; requires shared ownership for grow-m and thick-restart partial-publication states. |
| Revisit shift-invert setup/conversion movement | Deferred out of Sprint 115; requires LDLT lifecycle, operator selection, public error propagation, and cleanup ownership proof. |
| Revisit `lanczos_iterate_op` movement | Deferred out of Sprint 115; requires compile-unit proof for all current consumers. |
| Decide whether direct/iterative generated-RHS setup can share a common oracle | Deferred out of Sprint 115; not package/platform-facing. |
| Decide whether SVD proof helpers can share a broad owner | Deferred out of Sprint 115; not package/platform-facing. |

## Package/Platform Claim-Fence Checklist

Sprint 115 leaves the following support truth explicit:

- Linux install proof stays local Unix-side evidence; no reviewed Linux install
  CI lane was added.
- macOS CMake install/export parity remains deferred; macOS keeps the reviewed
  Apple Clang lane plus supplemental Homebrew GCC and Make install/`pkg-config`
  confidence.
- Windows install-validation remains deferred; Windows keeps the reviewed MSVC
  CMake-first subset.
- Windows `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain
  staged exclusions.
- macOS backend, coverage, Homebrew GCC, OpenMP, and TSan claims remain
  evidence-bounded and do not become new reviewed lanes.
- Shared-library and dynamic ABI support remain future product contracts.
- Package-manager support remains future work.
- No public install-header, public API, Make/CMake source-list, helper-target,
  reviewed CTest membership, or platform parity claim changed during Day 13.

## Sprint 116 Handoff

Sprint 116 adoption QA should use Sprint 115 decisions as the package/platform
truth source. Adoption-facing documentation should not advertise:

- reviewed Linux install CI proof;
- full reviewed macOS CMake install/export parity;
- Windows install-validation parity;
- Windows thread/fuzz/property parity;
- shared-library or dynamic ABI support;
- package-manager support;
- broad platform parity beyond the reviewed tier model.

## Sprint 117 Handoff

Sprint 117 should carry the Sprint 114 non-package residual queue unless it
explicitly promotes one bounded item. Promotion requires exact old/new owner
files plus source-list, CMake, focused consumer proof, reviewed CTest count
evidence where applicable, and rollback instructions.

The carry-forward source-boundary/proof-owner queue is:

- eigensolver private-owner movement;
- `s20_select_indices` movement;
- `s20_lift_ritz_vectors` movement;
- shift-invert setup/conversion movement;
- `lanczos_iterate_op` movement;
- broad direct/iterative generated-RHS oracle abstraction;
- broad SVD proof-helper abstraction.

## Epic Closeout Handoff

Epic closeout should treat Sprint 115's package/platform decisions as the
current truth unless a later sprint lands reviewed evidence that changes them.
The closeout narrative should present static-first install/export support as
maintained while preserving explicit non-claims for dynamic ABI,
package-manager, Windows install-validation, full macOS install/export, and
broader Windows thread/fuzz/property parity.

## Validation

Day 13 is documentation-only. No README, INSTALL, maintainer-guide, workflow,
CMake, Makefile, source, header, package metadata, or test-registration changes
were made.
