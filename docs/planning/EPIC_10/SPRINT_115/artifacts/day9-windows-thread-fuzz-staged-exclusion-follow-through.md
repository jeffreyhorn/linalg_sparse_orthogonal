# Day 9: Windows Thread and Fuzz Staged-Exclusion Follow-Through

## Purpose

Day 9 applies the Day 8 Windows portability audit. The work either promotes
one bounded Windows-native proof owner or publishes the staged-exclusion
contract for thread, fuzz, and property coverage.

## Decision

Sprint 115 does not add a bounded Windows-native thread, fuzz, or property
proof owner.

The current staged exclusions remain in force:

- `test_threads`;
- `test_sprint4_integration`;
- `test_fuzz`, including the bounded large-n lifecycle property lanes.

## Why No Proof Was Added

The Day 8 audit found no low-risk bounded candidate:

- `test_threads` is a pthread-based proof owner with many concurrent cases and
  optional `SPARSE_MUTEX` coverage.
- `test_sprint4_integration` includes serial integration checks, but the file
  also owns a pthread-backed concurrent Cholesky proof.
- `test_fuzz` uses POSIX temporary-file helpers and combines parser fuzz cases
  with broad seeded solver property lanes.

Adding any of these binaries to Windows CTest would require real test
ownership work and would change the reviewed Windows surface. A partial proof
would risk implying broader Windows parity than Sprint 115 can honestly claim.

## Registration Contract

The current CMake registration remains unchanged:

| Test | Current registration gate | Day 9 status |
|---|---|---|
| `test_threads` | `Threads_FOUND AND NOT WIN32` | staged |
| `test_sprint4_integration` | `Threads_FOUND AND NOT WIN32` | staged |
| `test_fuzz` | `NOT WIN32 AND NOT MSVC` | staged |

The Windows reviewed CTest count remains `51`.

## Workflow Contract

`.github/workflows/windows-ci.yml` already owns the required clarity:

- `EXPECTED_WINDOWS_CTEST_COUNT` is `51`;
- the `ctest -N` inspection step fails if the count changes unexpectedly;
- job output explicitly says
  `test_threads`, `test_sprint4_integration`, and `test_fuzz` remain staged
  exclusions;
- workflow comments describe the lane as reviewed CMake-first consumer proof
  only.

No workflow change is needed for Day 9.

## Documentation Contract

The existing documentation already matches the Day 9 decision:

- `README.md` describes Windows as the reviewed CMake subset and CMake-first
  consumer story.
- `INSTALL.md` describes Windows as reviewed CMake subset only and does not
  claim a separate install-validation lane.
- `docs/maintainer_guide.md` records the 51-test Windows count and the staged
  exclusions.
- `docs/maintainer_guide.md` also states that `test_fuzz` property lanes are
  not reviewed Windows evidence.

No public documentation change is needed for Day 9.

## Future Promotion Requirements

Any future Windows promotion must update these surfaces together:

- source or test helper code for native Windows thread/temp-file behavior;
- `CMakeLists.txt` registration gates;
- `.github/workflows/windows-ci.yml` reviewed count and exclusion output;
- maintainer-guide platform-confidence wording;
- focused Windows CMake configure/build/`ctest -N`/`ctest` evidence.

If only a subset of `test_fuzz` is promoted, the promoted owner must separate
parser fuzz evidence from solver property evidence so the bounded lifecycle
property lane is not accidentally claimed as reviewed Windows proof.

## Non-Claims Preserved

Day 9 does not claim:

- Windows thread-safety parity;
- Windows pthread compatibility;
- Windows fuzz/property parity;
- reviewed Windows lifecycle property coverage;
- Windows Makefile parity;
- Windows install-validation parity;
- Windows package-manager support;
- Windows shared-library, DLL, dynamic ABI, or runtime-loader support;
- broader Windows platform parity beyond the reviewed CMake subset.

## Validation

Day 9 is documentation-only. No `.c`, `.h`, CMake, workflow, or test
registration changes were made.
