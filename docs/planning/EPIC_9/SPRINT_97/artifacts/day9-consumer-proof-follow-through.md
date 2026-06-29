# Sprint 97 Day 9: Consumer Proof Follow-Through

## Purpose

Day 9 aligns the installed consumer proof with the Day 8 static-first package
decision. The goal is not to add a new consumer workflow; it is to make the
existing proof scripts explicitly guard the package shape that README, INSTALL,
CMake, and CI now describe.

## Reviewed Surfaces

Consumer-facing and proof surfaces reviewed:

- `README.md`
- `INSTALL.md`
- `examples/README.md`
- `examples/cmake_example/`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `CMakeLists.txt`
- `cmake/SparseConfig.cmake.in`
- `sparse.pc.in`
- Linux, macOS, and Windows workflows

## Proof Update

Day 9 updates the two local install/export proof scripts:

- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

Both scripts already proved the supported downstream consumer paths:

- Make install plus `pkg-config`
- CMake install/export plus `find_package(Sparse)`
- exact installed version acceptance
- mismatched installed version rejection in the CMake package path
- maintained example source builds and runs from installed package metadata

Day 9 adds one static-shape assertion to each script: after installation, the
temporary install prefix must not contain shared-library artifacts matching
`.so`, `.so.*`, `.dylib`, or `.dll`.

This keeps the proof aligned with the Day 8 decision:

- static archive installed: yes
- shared-library artifact installed: no
- downstream consumer metadata still resolves and links successfully

## Public Guidance Check

Public guidance remains aligned:

- `README.md` points install users to `INSTALL.md`, describes
  `pkg-config`/`find_package(Sparse)` as the maintained static package surface,
  and states shared-library packaging is deferred.
- `INSTALL.md` owns operational install detail, static-first package shape,
  local install/export validation, and reviewed-platform interpretation.
- `examples/README.md` remains build-tree and example focused; it does not
  imply a shared-library package path.
- `examples/cmake_example/` remains the maintained downstream CMake consumer
  fixture used by `tests/test_cmake_install.sh`.

## Preserved Limitations

Day 9 preserves these non-claims:

- no shared-library build output
- no dynamic ABI promise
- no Windows Makefile install path
- no separate reviewed Windows install-validation lane
- no full reviewed macOS install/export parity claim
- no package-manager integration claim

## Validation

Focused validation:

```sh
bash tests/test_install.sh
bash tests/test_cmake_install.sh
python3 scripts/check_library_sources.py
git diff --check
rg -n "[ \t]+$" tests/test_install.sh tests/test_cmake_install.sh docs/planning/EPIC_9/SPRINT_97
```

Observed proof totals after Day 9:

- `tests/test_install.sh`: 14 passed, 0 failed
- `tests/test_cmake_install.sh`: 16 passed, 0 failed, 0 skipped
- `scripts/check_library_sources.py`: `source-list-check: PASS (42 library sources)`
- `git diff --check`: passed
- trailing-whitespace scan: passed with no matches

No `.c` or `.h` files are modified by this consumer-proof update, so the full
`make format && make lint && make test` chain is not required.

## Day 9 Result

The maintained installed consumer proof now asserts both positive and negative
package shape: the static archive is installed, downstream consumers work, and
no shared-library artifact appears in the installed package prefix.
