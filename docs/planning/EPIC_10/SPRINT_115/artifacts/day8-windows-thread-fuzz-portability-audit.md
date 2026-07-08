# Day 8: Windows Thread and Fuzz Portability Audit

## Purpose

Day 8 audits the Windows staged exclusions for thread, integration, fuzz, and
property coverage. The goal is to decide whether Sprint 115 should promote one
bounded Windows-native proof owner on Day 9 or preserve the current staged
exclusion contract without overstating Windows parity.

## Surfaces Reviewed

| Surface | Current Windows status | Evidence |
|---|---|---|
| `test_threads` | Excluded from Windows CTest | `CMakeLists.txt` registers it only under `Threads_FOUND AND NOT WIN32`. |
| `test_sprint4_integration` | Excluded from Windows CTest | `CMakeLists.txt` registers it only under `Threads_FOUND AND NOT WIN32`. |
| `test_fuzz` | Excluded from Windows CTest | `CMakeLists.txt` registers it only under `NOT WIN32 AND NOT MSVC`. |
| Windows workflow | Reviewed CMake subset only | `.github/workflows/windows-ci.yml` guards `EXPECTED_WINDOWS_CTEST_COUNT=51`. |
| Maintainer docs | Exclusions documented | `docs/maintainer_guide.md` records the three tests as staged exclusions. |

## Current Windows Reviewed Contract

Windows currently proves:

- MSVC CMake configure on `windows-2022`;
- Release build;
- guarded `ctest -N` registration count;
- full `ctest` execution for the reviewed CMake subset.

Windows currently does not prove:

- Makefile reviewed parity;
- dead-code flow parity;
- separate install-validation parity;
- thread/fuzz/property parity;
- package-manager, dynamic ABI, DLL, or runtime-loader support.

## Portability Findings

### `test_threads`

`test_threads` is a POSIX-thread proof owner. It directly includes
`<pthread.h>` and uses `pthread_create` / `pthread_join` throughout the
suite. The proof surface covers independent LU work, shared factored solves,
concurrent Cholesky solves, stress cases, concurrent norm computations, and
optional `SPARSE_MUTEX` concurrent insert coverage.

Promoting this binary on Windows would require one of:

- a cross-platform thread helper abstraction;
- a Windows-native `CreateThread` or `_beginthreadex` implementation;
- a split Windows-specific bounded thread test.

That is more than a narrow package/platform claim correction and would change
the reviewed Windows CTest count.

### `test_sprint4_integration`

`test_sprint4_integration` mixes serial SuiteSparse/CSR/Cholesky integration
checks with a pthread-backed concurrent Cholesky proof over `nos4.mtx`. The
serial checks are more portable than the concurrent proof, but the file-level
owner still directly includes `<pthread.h>` and is registered behind the same
`Threads_FOUND AND NOT WIN32` gate as `test_threads`.

Promoting this binary would require splitting the serial integration checks
from the thread proof or adding native Windows thread support. Either route is
a real test-ownership change, not a Day 8 audit-only action.

### `test_fuzz`

`test_fuzz` combines Matrix Market parser fuzz cases with seeded solver
property checks. It directly includes `<unistd.h>` and uses POSIX temp-file
helpers:

- `mkstemps`;
- `close`;
- `unlink`;
- Unix-style `/tmp` fallback.

The same binary also owns the large-n CSC lifecycle property lanes. Those
lanes are useful Linux/macOS evidence, but they are explicitly not reviewed
Windows evidence today.

Promoting this binary would require a Windows-safe temporary-file helper and a
careful decision about whether Windows should own only parser fuzz coverage or
also the broader solver property lanes.

## Bounded Proof-Owner Decision

Day 8 selects no bounded Windows-native proof owner.

This is the least misleading decision because all three candidates would
either require nontrivial source/test restructuring or broaden the reviewed
Windows CTest surface. A partial promotion would also create a support-wording
risk: readers could infer broad Windows thread/fuzz/property parity from one
small Windows-native proof.

## Staged-Exclusion Contract

The current staged exclusions remain:

- `test_threads`;
- `test_sprint4_integration`;
- `test_fuzz`, including the bounded lifecycle property lane.

The current reviewed Windows count remains `51`. Any future promotion must
update all of the following together:

- `CMakeLists.txt` test registration gates;
- `.github/workflows/windows-ci.yml` `EXPECTED_WINDOWS_CTEST_COUNT`;
- workflow output describing staged exclusions;
- maintainer-guide Windows support wording;
- focused validation evidence for the promoted binary or split owner.

## Day 9 Follow-Through

Day 9 should publish the staged-exclusion follow-through unless a narrow
documentation or workflow-comment update is discovered. It should not add a
Windows thread or fuzz binary without also owning the reviewed-count change and
the support wording that follows from that change.

## Non-Claims Preserved

This audit does not claim:

- Windows thread-safety parity;
- Windows pthread compatibility;
- Windows fuzz/property parity;
- reviewed Windows lifecycle property coverage;
- Windows Makefile parity;
- Windows install-validation parity;
- broader Windows platform parity beyond the reviewed CMake subset.

## Day 8 Validation

Day 8 is documentation-only. No source, CMake, workflow, or test-registration
changes were made.
