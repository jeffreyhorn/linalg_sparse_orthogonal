# Sprint 60 Day 10: Validation and Platform Contract Freeze

Date: 2026-06-08
Branch: `sprint-60`


## Purpose

Freeze the validation, gate-selection, and platform-truthfulness contract that
later Epic 6 implementation sprints must follow.

This contract is downstream of the Day 9 architecture fence. It defines how
future work proves itself, what counts as the reviewed truth surface, and where
platform claims remain intentionally narrower than the Linux baseline.

## 1. Reviewed Baseline Contract

### 1.1 Strongest local reviewed baseline

The strongest local reviewed baseline remains:

- `make quality-review-full`

Interpretation:

- this remains the top local truth surface for code changes unless a later
  measured contract revision explicitly changes it
- later Epic 6 work should not invent a stronger-but-unreviewed ad hoc local
  gate and treat it as authoritative

### 1.2 Reviewed local sub-paths remain distinct and meaningful

The maintained local reviewed paths remain:

- Makefile reviewed compile-quality path:
  - `make quality-review-compile`
- Makefile reviewed local quality path:
  - `make quality-review`
- reviewed CMake parity compile path:
  - `make quality-review-cmake-compile`
- reviewed CMake parity execution path:
  - `make quality-review-cmake`

Contract rule:

- these remain separate named operator surfaces
- later Epic 6 work may build on them, but should not blur their meanings

### 1.3 Parity-count anchor remains explicit

The current reviewed CMake parity-count anchor remains:

- `ctest -N --test-dir build/quality-review-cmake`

Current Sprint 60 baseline:

- reviewed Linux/normal local parity anchor = `53`

Contract rule:

- when later work changes the registered reviewed test surface, it must update
  parity expectations deliberately
- parity count is part of the contract, not just an incidental diagnostic

## 2. Gate-Selection Policy

### 2.1 Default policy for code-touching work

For code-touching work in later Epic 6 sprints, the default required local gate
is:

- `make format`
- `make lint`
- `make test`

When the work is positioned against the reviewed baseline or changes cross-cut
surfaces, the stronger default becomes:

- `make quality-review-full`

Interpretation:

- later implementation work should not stop at compile-only or single-binary
  proof when it changes shared implementation or contract surfaces

### 2.2 Stronger policy for substantial architecture, performance, or platform work

The following change classes should default to the stronger reviewed path:

- architecture/control-plane work
- backend/AUTO-policy work
- benchmark-governance contract work
- build/package/platform work
- validation/truthfulness contract work

Preferred local gate:

- `make quality-review-full`

with focused follow-ons where relevant, for example:

- direct lifecycle proof binaries/examples
- iterative/eigensolver proof binaries/examples
- workflow-proof benchmark drivers

### 2.3 Docs-only exception

Docs-only work remains exempt from the code-touching gate when it does not
modify `*.c` or `*.h` files and does not change executable scripts or build
logic.

Allowed Day-10-style docs-only validation mode:

- targeted diff review
- `rg` truthfulness checks
- file-shape or surface-map checks

Contract rule:

- docs-only work must still be grounded in the live repo state
- "docs-only" does not authorize stale or aspirational claims

### 2.4 Script/build/workflow edits are not docs-only

Changes to any of the following should be treated as code/contract work even if
no library `*.c` file changes:

- `Makefile`
- `CMakeLists.txt`
- CI workflow files
- validation scripts
- dead-code scripts

These surfaces participate directly in the reviewed truth contract.

## 3. Dead-Code Contract

### 3.1 Dead-code remains a distinct reviewed surface

The dead-code workflow remains:

- `make deadcode`
- `make deadcode-report`
- `make deadcode-check`

Contract rule:

- `make deadcode-check` remains a report-completeness gate
- it is not a zero-findings gate
- it is not automatic deletion authority

### 3.2 Serialized execution remains a live operational limit

The dead-code workflow still shares:

- `build/deadcode-cmake`
- `build/deadcode/`

Contract rule:

- dead-code execution remains intentionally serialized
- later Epic 6 work must not imply this limit is already solved
- topology redesign is a separate change class, not incidental cleanup

## 4. Coverage Contract

Coverage remains a supplemental signal, not the primary reviewed baseline.

Current live policy:

- `make coverage`
- 80% line-coverage threshold on the active `src/` surface
- Linux supplemental CI signal

Contract rule:

- coverage stays useful and enforced where already wired
- coverage is not currently an active reviewed-baseline residual
- later Epic 6 work should not rewrite the coverage story unless a real
  contradiction emerges

## 5. Platform Truthfulness Contract

### 5.1 Linux remains the enforced reviewed source of truth

Linux currently owns the fullest reviewed surface:

- reviewed Makefile compile-quality path
- reviewed CMake parity path
- dead-code report/check path

Supplemental Linux signals remain additive:

- direct runtime path
- `bench-fast`
- TSan
- coverage

Contract rule:

- Linux remains the authoritative reviewed product/quality truth surface until
  explicit measured revision

### 5.2 macOS remains an enforced but narrower reviewed platform

macOS currently enforces:

- Apple Clang `make quality-review-compile`
- Apple Clang `make quality-review-cmake`
- `make wall-check`
- Apple Clang `make sanitize`

macOS remains staged or supplemental for:

- dead-code
- Homebrew GCC direct `make` + `make test`
- install/pkg-config verification

Contract rule:

- macOS claims must stay aligned with this split
- dead-code on macOS remains staged pending fresh measurement
- second-compiler and install/pkg-config coverage stay supplemental rather than
  silently promoted to reviewed parity

### 5.3 Windows remains an enforced reviewed CMake subset

Windows currently enforces the reviewed CMake subset only:

- configure
- build
- `ctest -N`
- full `ctest`

Current Windows reviewed subset anchor:

- expected CTest count = `50`

with explicit staged exclusions still called out in the workflow:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

Windows remains staged for:

- reviewed Makefile wrappers
- dead-code flow

Contract rule:

- later Epic 6 work must not imply full Windows reviewed parity unless those
  staged surfaces actually move into the enforced set

## 6. Platform/Validation Claim Rules

Any Epic 6 work that changes validation, platform, benchmark, or packaging
surfaces must preserve these truthfulness rules:

1. enforced, staged, and supplemental surfaces must stay labeled distinctly
2. a staged surface may not be described as reviewed parity
3. platform-specific exclusions must remain explicit
4. reviewed baseline language must match the live workflows and wrappers
5. local operator commands in docs must stay aligned with the maintained
   Makefile/workflow behavior

## 7. Immediate Implications for Later Epic 6 Sprints

The Day 10 freeze implies:

- control-plane and backend-policy implementation sprints should normally prove
  themselves against `quality-review-full`
- packaging/platform work must preserve the Linux truth surface while staying
  honest about macOS and Windows limits
- benchmark-governance work may refine claim policy, but must not weaken the
  evidence bar
- docs-only closeout or audit days may stay lightweight, but only when their
  claims are checked against the live repo state

## Day 10 Exit State

Sprint 60 now has a frozen validation and platform contract:

- strongest local reviewed baseline is explicit
- gate-selection policy is explicit
- docs-only exceptions are explicit
- dead-code and coverage meanings are explicit
- Linux/macOS/Windows truthfulness boundaries are explicit
- later Epic 6 implementation work now has a stable proof and platform fence
