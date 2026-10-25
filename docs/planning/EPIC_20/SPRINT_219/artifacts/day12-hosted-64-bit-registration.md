# Day 12: Hosted 64-Bit Registration

Date: 2026-10-24

## Registered Workflows

| Platform | Job or step | Width and scope | Status |
| --- | --- | --- | --- |
| Linux | `CI / selected-index-width-64` | Fresh CMake `SPARSE_IDX_BITS=64` tree; archive/header/compile-command identity and five selected functional cases. | Registered; hosted execution unavailable on this unpublished branch. |
| macOS | `macOS CI / selected-index-width-64` | Same fresh 64-bit selected functional lane. | Registered; hosted execution unavailable on this unpublished branch. |
| Linux | `CI / package-contract` | Existing default-width proofs plus CMake and pkg-config 32/64-bit installed-consumer matrices and archive-width pre-use checks. | Registered; hosted execution unavailable on this unpublished branch. |
| macOS | `macOS CI / install-and-pkgconfig` and `cmake-install-export` | Existing default-width proofs plus 32/64-bit pkg-config and CMake matrices, respectively; CMake job also runs pre-use checks. | Registered; hosted execution unavailable on this unpublished branch. |
| Windows | `Windows CI / cmake-install` | Default 32-bit generated header, imported-target metadata, package metadata, and installed CMake consumers checked before matrix use. Expected CTest count updated to 62. | Registered; hosted execution unavailable on this unpublished branch. No Windows pkg-config or 64-bit functional claim. |

The new selected job builds in a temporary clean tree and checks the explicit
64-bit cache selector, generated header, library and test compile flags,
runtime archive query, and selected CTest results. Existing reviewed jobs
remain default-width. These checks are not an all-algorithm 64-bit claim.

## Execution Evidence

| Environment | Check | Outcome | Provenance / limit |
| --- | --- | --- | --- |
| Local macOS x86_64 | `bash tests/test_cmake_idx_width_functional.sh` | PASS: `typedef=64 header=64 archive=64`; five named cases; focused CTest 1/1. | Local Day 12 run, not hosted evidence. |
| Local macOS x86_64 | `python3 tests/test_validate_windows_powershell.py` | PASS: structural mutation tests, including the 62-test count guard. | Cannot execute Windows PowerShell. |
| Local macOS x86_64 | Ruby YAML parsing of all three workflow files | PASS: Linux 10 jobs, macOS 6, Windows 5; selected job present in Linux and macOS. | Syntax/structure only. |
| Local macOS x86_64 | Fresh default CMake configure and `ctest -N` | PASS: 62 tests registered. | Enumeration only; no CTest execution in this tree. |
| Local macOS x86_64 | `python3 scripts/validate_windows_powershell.py` | UNAVAILABLE for executable checks: `pwsh` absent; structural checks passed. | Not a Windows validation pass. |
| Local Linux container | Docker daemon probe | UNAVAILABLE: daemon stopped. | No local Linux execution. |
| Hosted `sprint-219` | `gh run list --branch sprint-219`; PR lookup | UNAVAILABLE: no runs and no PR; branch has no upstream. | Not queued, skipped, failed, or passed. |

The latest successful `master` runs are prior-baseline provenance only:
[Linux CI](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/40141237868),
[macOS CI](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/40141237965), and
[Windows CI](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/40141237935)
at SHA `145b5a5a21dbb5191d5d1e1e2550b80c98819861`. They do not contain
this Day 12 workflow change and cannot close the 64-bit hosted gate.

## Platform Scope And Open Gate

The local selected fixture passed without an observed width failure. Hosted
compiler/linker diagnostics remain unknown. Windows checks use the reviewed
CMake install route and default 32-bit package; `sparse.pc` is inspected as
metadata there, not exercised through pkg-config. The selected Linux/macOS
64-bit jobs do not imply full-suite or all-solver 64-bit validation.

**Gate OPEN:** Item 219.5 and Day 12 completion require actual green hosted
results for the selected 64-bit jobs and the relevant install/consumer jobs.
After branch publication/PR creation, record their run URLs, commit SHA,
conclusions, and any platform warnings here. Local success must not be
substituted for hosted results.
