# Day 14: Quality And Width-Contract Closeout

Date: 2026-10-24
Branch: `sprint-219` (local, no upstream or PR)

## Final Local Quality

| Check | Result | Scope |
| --- | --- | --- |
| `make BUILDDIR=build/s219-day14-default format`; `make format-check`; `make source-list-check`; `git diff --check` | PASS; 49 library sources in the source-list check. | Final C formatting, source inventory, and whitespace. |
| `make -s BUILDDIR=build/s219-day14-default -j4 lint` | PASS after the Day 14 test-warning fixes. | Default 32-bit strict compile, clang-tidy, and cppcheck. |
| `make -s BUILDDIR=build/s219-day14-default -j4 test` | PASS after the fixes; all Make test binaries passed. | Full local default-width suite. Nonfatal duplicate `-lm` linker warnings remain. |
| `make quality-review-cmake` | PASS after the fixes; clean rebuild, 62 CTests registered, 62/62 passed. | Default 32-bit CMake/CTest parity. Apple Clang emitted nonfatal `-Wdouble-promotion` warnings for existing `NAN`/`INFINITY` macro uses. |
| `python3 tests/test_validate_windows_powershell.py` | PASS. | Workflow structure and PowerShell snippet validation on macOS; not a hosted Windows execution. |

Both default trees generated `SPARSE_BUILT_IDX_BITS 32`; the reviewed CMake
cache selected `SPARSE_IDX_BITS=32`. The focused sanitizer tree selected 64
and generated `SPARSE_BUILT_IDX_BITS 64`.

## Installed And 64-Bit Contract Matrix

| Check | Result | Boundary |
| --- | --- | --- |
| `bash tests/test_make_idx_width.sh` | PASS, 26/26. | Default and explicit Make 32/64 selectors, build stamps, and stale/mismatched tree negatives. |
| `bash tests/test_cmake_idx_width_install.sh` | PASS, 18/18. | 32/64 CMake installed consumers and wrong-width/package negatives. |
| `bash tests/test_pkgconfig_idx_width_install.sh` | PASS, 40/40. | Make/CMake 32/64 pkg-config consumers and mixed-prefix negatives. |
| `bash tests/test_installed_width_check.sh` | PASS, 12/12. | Both-width pre-use archive checks; swapped archives rejected before solver use. |
| `python3 tests/test_static_install_contract.py` | PASS, 26 tests. | Static package metadata and archive validation mutations. |
| `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh` | PASS, 26/26 and 38/38. | Default-width installed static consumers. |
| `bash tests/test_cmake_idx_width_functional.sh` | PASS, focused CTest 1/1 and five selected cases. | Fresh 64-bit tree: typedef, header, and archive agree; insert/LU, Matrix Market, CSR/CSC, and index text boundaries exercised. |
| 64-bit UBSan focused CMake build with `-Werror=format -Werror=absolute-value -Werror=shorten-64-to-32`; CTest selector `^(test_idx_width_functional|test_suitesparse|test_qr)$` | PASS, 3/3. | Selected 64-bit functional and solver cases only. Width-sensitive `idx_t` print and `abs()` narrowing in two tests were corrected before this final run. |

The broader `SPARSE_IDX_BITS=64 make sanitize` attempt did **not** finish:
after earlier test binaries passed without a reported sanitizer error,
`test_reorder_nd` remained CPU-active for over ten minutes under UBSan and
was interrupted. Its overall target exited nonzero. This is an execution
limit, not a full-suite sanitizer pass or a diagnosed sanitizer defect. The
bounded 3/3 UBSan result above is the only passing 64-bit sanitizer claim.

## Diff And Hosted Audit

The final tracked diff has no generated build files, and `git diff --check`
is clean. Generated headers and CMake caches in separate ignored build trees
show default 32 and selected 64 as intended. The Make width stamp and CMake
cache guards reject reuse of the opposite-width tree; both installed routes
were checked against their real static archives. The two Day 14 C edits are
limited to width-safe test diagnostics and a non-narrowing QR nnz comparison.

`gh run list --branch sprint-219 --limit 10` and
`gh pr list --head sprint-219 --state all` returned no results. The branch
has no upstream. Therefore there are no run URLs, hosted commit identities,
failure logs, or rerun results for the new Linux/macOS selected 64-bit lanes,
package matrices, or Windows default-width route. Earlier `master` runs are
baseline only. **The hosted gate, Day 12 completion, item 219.5, and the
Sprint 219 acceptance gate remain OPEN.** Item 219.6 has local quality and
contract evidence, but its final allocation and the 140-hour sprint total
must not be marked closed before required hosted jobs pass.

The package remains static-only. The maintained consumers check
`sparse_idx_bits()` against the installed header and type size before any
index-bearing call, but arbitrary manually linked consumers do not receive
an automatic runtime guard. No mixed-width, cross-version, shared-library,
or universal 64-bit algorithm ABI claim is supported.
