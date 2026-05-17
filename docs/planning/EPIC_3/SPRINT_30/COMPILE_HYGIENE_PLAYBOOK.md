# Sprint 30 Compile-Hygiene Playbook

**Status:** Finalized on Day 12  
**Date:** 2026-05-16  
**Sprint:** 30  
**Related artifacts:**

- [Day 1 warning baseline](./artifacts/day1-warning-baseline.md)
- [Day 2 warning taxonomy](./artifacts/day2-warning-taxonomy.md)
- [Day 3 core warning audit](./artifacts/day3-core-warning-audit.md)
- [Sprint 30 working notes](./WORKING_NOTES.md)
- [Epic 3 project plan](../PROJECT_PLAN.md)
- [Epic 3 remediation plan](../reviews/todo-codex-2026-05-15.md)

## Purpose

This playbook defines how Sprint 30 and the following Epic 3 sprints should treat compile-hygiene debt that already exists in the repository. Its job is to stop ad hoc debates about whether a warning matters, which build path is authoritative, and what evidence is required to claim that a warning class is resolved.

This document is intentionally scoped to existing code-quality debt. It does not define feature work.

## Current Measured State

Sprint 30 started from the Day 1 Apple Clang CMake full-build baseline:

- full-tree warnings: `123`
- `src`: `11`
- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`

By Day 5, the targeted core-library cleanup reduced that to:

- full-tree warnings: `112`
- `src`: `0`
- `tests`: `98`
- `benchmarks`: `13`
- `examples`: `1`

The remaining warning debt is therefore auxiliary-code debt, not core-library debt.

Day 9 then ran a stricter prototype-focused compile pass and found one additional `src/` warning in `src/sparse_types.c` (`-Wmissing-prototypes` on `sparse_set_errno_`). That issue was fixed the same day, and the strict-tree warning profile returned to the same post-Day-5 counts above. No separate strict-only `src/` backlog remains open after Day 9.

## Authoritative Build Surfaces

For Sprint 30 closeout and the follow-on Epic 3 cleanup, use these rules:

1. The authoritative full-tree warning inventory is the clean Apple Clang CMake build used in the Day 1 baseline.
2. The Makefile `all` path is a secondary cross-check for library compilation only; it is not yet a repository-wide warning inventory because it does not compile the whole tree by default.
3. CI-supported build surfaces are part of the engineering standard even when a given local day only measures one of them:
   - Ubuntu `make` build/test path
   - Ubuntu CMake + `ctest`
   - macOS `make` build/test path under Apple Clang and Homebrew GCC
   - Windows CMake + `ctest` under MSVC
4. A warning that appears on any supported build surface is real engineering debt unless it is demonstrated to be a compiler false positive.

## Warning Disposition Rules

### Must-fix now

These warnings block the code from being considered compile-hygiene clean for the affected scope:

- Any warning in `src/` on a supported build surface
- Any warning introduced by the current change set, even in auxiliary code
- Any warning that indicates a real API drift or portability defect in public-facing tooling
- Any warning that is resolved by a narrow, behavior-preserving fix in touched files

### Backlog-acceptable for Sprint 30

These warnings can remain temporarily only when they are pre-existing, measured, and explicitly documented in Sprint 30 artifacts:

- existing `tests/` warning debt that does not affect the core library and is already in the Day 1 and Day 2 baseline
- existing `benchmarks/` and `examples/` warnings queued for later Sprint 30 or Sprint 31 cleanup
- dormant-test warning debt that has been identified but not yet converted or removed

Backlog-acceptable does not mean “ignore indefinitely.” It means “do not expand this debt, and carry it in the explicit cleanup queue.”

### Not allowed

- turning on global `-Werror` before the baseline debt is actually removed
- suppressing warnings without documenting why the warning is a false positive or toolchain artifact
- merging new warning debt on the basis that the tree already had warnings elsewhere
- using the narrower Makefile `all` result as proof that the whole repository is warning-clean

## Area-Specific Expectations

### Core library: `src/`

- The core library is held to warning-free status on supported build surfaces.
- Behavior-preserving cleanup is preferred over stylistic refactors.
- Small, local fixes are preferred to new abstraction layers when the issue is narrow and mechanical.
- Any residual `src/` warning is Sprint-blocking until explicitly triaged and documented.

### Tests: `tests/`

- Tests may carry documented baseline debt during Sprint 30, but they must not accumulate new warnings.
- Touched test files should leave warning count no worse than they started.
- Dormant or non-executed scaffolding should not remain as “unused function” debt once a later sprint reaches that file.
- Test-only portability shims are acceptable when they keep behavior honest across supported compilers.

### Benchmarks: `benchmarks/`

- Benchmarks are part of the supported developer toolchain and should stay aligned with public enums, public option structs, and supported C/POSIX APIs.
- Warnings that indicate stale CLI surface, missing headers, or feature-test-macro drift are not harmless; they are usability and portability defects.
- Benchmark code may trail the core library for one sprint, but the debt must stay enumerated and scheduled.

### Examples: `examples/`

- Examples are public-facing code and should model the intended API style.
- Warnings caused by positional struct initialization, stale options usage, or portability mistakes should be treated as documentation debt, not merely sample-code debt.
- Examples should compile cleanly on the supported build paths that build them.

### Generated artifacts and templates

- Generated files should not be hand-edited unless the generator is intentionally bypassed and the reason is documented.
- If a generated header or template drives warning behavior, fix the source template or generator path rather than patching downstream copies ad hoc.
- Generated artifacts are not exempt from compile-hygiene expectations if they are compiled as part of supported builds.

## Portable vs Compiler-Specific Warnings

Treat warning classes using this decision order:

1. If the warning appears on more than one supported compiler or build path, it is portable debt and should be prioritized.
2. If the warning appears on only one supported compiler, but the warning points to a real type mismatch, implicit declaration, enum drift, or public-API usage problem, it is still real debt and should be fixed.
3. If the warning appears only on one compiler and looks like a plausible false positive, document:
   - the compiler and version
   - the exact warning text
   - why the code is believed to be correct
   - why suppression or deferral is safer than source change
4. Do not call a warning “toolchain noise” merely because another compiler stays silent.

For Sprint 30 specifically:

- Apple Clang CMake is the baseline inventory path.
- Makefile builds are a useful secondary check for library compilation behavior.
- Ubuntu GCC/Clang CI and Windows MSVC remain part of the acceptance surface even when Day 6 itself does not replay every warning there yet.

## Evidence Required To Close A Warning Class

A warning class is only considered “addressed” for a stated scope when all of the following exist:

1. A before/after warning count tied to a named build path
2. Stable artifact capture of the relevant build logs
3. Identification of the touched files and the exact warning class removed
4. A regression check proportional to the touched code
5. Updated sprint notes explaining what remains and what moved out of scope

Expected evidence by change type:

- core-library source edits:
  - fresh clean build log
  - per-file warning deltas when practical
  - targeted test slice, plus end-of-day standard checks
- benchmark/example portability edits:
  - rebuild of the affected binaries
  - proof that CLI/help/public API usage remains in sync
- test-scaffolding cleanup:
  - proof the code is either removed, executed, or intentionally formalized
  - note on whether coverage or test honesty improved

## Day-End Validation Standard

At the end of each Sprint 30 day, before commit:

1. Run `make format`
2. Run `make lint`
3. Run `make test`
4. Fix any reported issues before committing

This day-end sequence is additive. It does not replace narrower targeted validation such as CMake rebuilds, warning-capture reruns, or focused `ctest` slices used to prove a specific warning cleanup.

## Guidance For Later Epic 3 Work

This playbook supports the remaining Epic 3 sequence already laid out in the project plan and remediation plan:

- Sprint 30:
  - finish baseline, policy, and workflow capture
  - keep the core library warning-free
- follow-on warning cleanup:
  - benchmarks/examples portability drift
  - designated-initializer cleanup for evolving public option structs
  - dormant test scaffolding removal or formalization
- later enforcement:
  - add stricter gates only after the measured baseline has actually been reduced for the relevant scopes

The intended progression is:

1. measure
2. classify
3. remove high-signal debt
4. document evidence
5. gate only what the repository is ready to sustain

## Sprint 30 Rule Summary

- `src/` warnings are not acceptable.
- Pre-existing auxiliary warnings may remain temporarily only if they are measured and explicitly queued.
- New warnings in any area are not acceptable.
- Supported build surfaces define the quality bar, not just the easiest local command.
- Warning closure claims require captured evidence, not informal observation.
