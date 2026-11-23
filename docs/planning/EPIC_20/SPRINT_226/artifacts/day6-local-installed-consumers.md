# Sprint 226 Day 6: Exploratory Installed-Consumer and Width Proof

## Decision and scope

Local static installed consumers **passed** for the [Day 3 exploratory source archive](day3-exploratory-source-archive.md) on macOS 15.7.9 x86_64 with Apple Clang 17.0.0, CMake 4.4.4, and pkg-config 2.5.1. Both 32-bit and 64-bit installed packages compiled, linked, and ran the maintained example through CMake and pkg-config. Package, pre-use, stale-tree, and wrong-width checks also passed. The detailed [Day 5 source-to-install map](day5-archive-install-map-and-platform-matrix.md) covers the retained 32-bit prefix.

**No release-candidate or hosted platform proof is claimed.** The candidate freeze remains blocked by the missing approved changelog/version decision, uncommitted Sprint 226 inputs, and unassigned release/rollback reviewers. The tested source is base commit `c42d6715fad7e895bae4941a006df3563f1afc38`, not a later selected candidate. These macOS local results do not establish Linux or Windows behavior.

## Retained clean-prefix builds and consumers

The retained 32-bit CMake package is the fresh [Day 5](day5-archive-install-map-and-platform-matrix.md) install at `build/sprint226/day5/prefix`. Day 6 configured and built a separate 64-bit tree from the same extracted source, installed it at `build/sprint226/day6/prefix64`, and ran `check_static_install_contract.py --mode cmake --platform unix --version 2.2.0`: **PASS**. The 64-bit build used `-DSPARSE_IDX_BITS=64`, its generated `sparse_version.h` reports width 64, and its installed static archive has SHA-256 `f5bfd32cb6c8a8077d324308709d1be0fd99b31f71722c09270986bee7b2176c`. The 32-bit installed archive SHA-256 is `f9ee20a7fdc169ad4559009234284780a2bc5e690a613736cd4bcbb7b8b9db40`.

The explicit 64-bit commands were:

```sh
cmake -S build/sprint226/day3/sparse-lu-ortho-2.2.0 \
  -B build/sprint226/day6/cmake64 \
  -DCMAKE_INSTALL_PREFIX=/Users/jeff/experiments/linalg_sparse_orthogonal/build/sprint226/day6/prefix64 \
  -DSPARSE_IDX_BITS=64 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/sprint226/day6/cmake64 --target sparse_lu_ortho --parallel 2
cmake --install build/sprint226/day6/cmake64
python3 scripts/check_static_install_contract.py build/sprint226/day6/prefix64 \
  --mode cmake --platform unix --version 2.2.0
```

CMake then configured `examples/cmake_example` twice, each with an explicit `Sparse_DIR` pointing to its respective installed prefix, built `example`, and ran it. Both runs printed the selected width, `Sparse library version 2.2.0 (int 20200)`, solution `[1.000000, 1.000000, 1.000000]`, and `OK`. The CMake cache confirmed each consumer was bound to the intended prefix. Binary SHA-256 values: 32-bit `2d433747a2e7c74201027cf371fafb338a73e0152af2779b23acbdcd568810ff`; 64-bit `7608e42a3bbfa9ddf44fc8f8a0f17051755a1eea9fa59ab51688f1e2d104dc14`.

For pkg-config, `PKG_CONFIG_LIBDIR` was set to the respective prefix's `lib/pkgconfig` and `PKG_CONFIG_PATH` was empty. `pkg-config --cflags sparse` returned that prefix's include path and `-DSPARSE_PACKAGE_IDX_BITS=32` or `64`; `--libs` returned its `-L` path, `-lsparse_lu_ortho`, and `-lm`; `--modversion` returned `2.2.0`. The maintained `examples/cmake_example/main.c` was compiled with `cc -std=c11 -Wall -Wextra -Werror`, those exact pkg-config flags, and linked to each installed package. Both binaries solved and printed `OK`. The full argv, environment path, stdout, exit status, and binary/archive/header hashes are retained in ignored `build/sprint226/day6/pkg-consumer-receipts.json` (SHA-256 `0eb9c8276529e34f7f4b8b7974b336e7afd75b52e6828c270a299d1b44c15fc2`). Pkg-consumer binary SHA-256 values: 32-bit `6becd4a5bdcf59a6eab9fa40cfb5be54ca0d3ea76a6b6bd906890b82220da2e1`; 64-bit `faa8383b5626b480a5b916b9cccb4c3505068249dc8b5e4cf5ff1607c11772f4`.

## Extracted-source regression suites

Each command below invoked the script **inside the extracted exploratory source tree** at `build/sprint226/day3/sparse-lu-ortho-2.2.0/`. All exited 0. The scripts use separate temporary build/install prefixes and clean them afterward. Their full stdout/stderr summaries were redirected to the listed ignored Day 6 log files; internal temporary build logs are not retained. These script passes complement, but do not replace, the retained prefixes and consumer receipts above.

| Extracted-source command | Result | Retained log SHA-256 |
| --- | --- | --- |
| `bash tests/test_install.sh` | Make install, consumer, missing-archive and uninstall: **26 passed, 0 failed**. | `make-install.log`: `647ac536687c817c8e6e97b09626fe669b1098304c7cccf9ec1914ef080078df` |
| `bash tests/test_cmake_install.sh` | CMake install, consumer, negative and uninstall: **38 passed, 0 failed, 0 skipped**. | `cmake-install.log`: `7b3d49879fdbce68ed601af523e33150ffd53262778f7eb9efa5bbf3326ba6d6` |
| `bash tests/test_pkgconfig_idx_width_install.sh` | Make and CMake 32/64 pkg-config installs, consumers, and negative cases: **40 passed, 0 failed**. | `pkgconfig-width.log`: `77c18f393ecd3066337b326c712ef4852d9180012d8dba3600c29e926937b44a` |
| `bash tests/test_cmake_idx_width_install.sh` | CMake 32/64 relocated consumers, metadata and mismatch cases: **20 passed, 0 failed**. | `cmake-width.log`: `73e3ef17a063a2deac13c80f84868b68fb59a5a9b1ea3d3411e656331c88fd37` |
| `bash tests/test_installed_width_check.sh` | Pre-use archive-width and swapped-package rejection: **12 passed, 0 failed**. | `preuse-width.log`: `2d84473abb27abb6653e1fdaeb381a884aafe6dfddb3503f30f85187a3ef8d80` |
| `bash tests/test_make_idx_width.sh` | Make width selection, stale/unstamped tree rejection: **28 passed, 0 failed**. | `make-width.log`: `81f639f4599f85e5b61c120ebbd2f3b70f5faa8be918cf5413ef6349d5e9bac7` |
| `bash tests/test_cmake_idx_width_functional.sh` | Fresh 64-bit CMake functional CTest: **1/1 passed** (matrix/CSR/CSC and text-index checks inside the test). | `functional-64.log`: `0305e3956724dea5903dfb00662ae0a1c34002c64174c0c4ba59b0a8ea7777a8` |
| `python3 tests/test_static_install_contract.py` | Static package positive/mutation suite: **26 tests, OK**. | `static-contract-mutations.log`: `4922cff1fc6eec0db9788a986bc60c273f98e5c43d8fb83791483f0819097f7d` |
| `bash scripts/static_package_deferral_check.sh` | Static-only and unsupported-claim guard: **passed**. | `static-deferral.log`: `9738701a9cfb33ca373dae346a59fc2e37016d014aff68aecead9ab431d1718b` |

Two additional direct negative compiles were retained under `build/sprint226/day6/wrong-width-32-to-64.log` and `wrong-width-64-to-32.log`. Each compiled the maintained example with an explicit consumer width opposite to the installed package width. Both returned exit 1 with the public-header diagnostic `SPARSE_IDX_BITS must match SPARSE_BUILT_IDX_BITS`, before any solver execution. There were no unexpected failures in the regression suites. The local negative-fixture diagnostics in those suites were reviewed through their pass results; only these two direct wrong-width stderr streams were retained after test cleanup.

## Gate and handoff

| Gate | State |
| --- | --- |
| Local exploratory Make/CMake static installed consumers, 32/64 width metadata, pre-use checks | **Passed** on the named macOS host for the base source. |
| Exact candidate commit, archive, clean-prefix proof | **Blocked**: no frozen reviewed SHA or approved changelog/owners. All Day 6 results must be rerun for that SHA. |
| Linux/Windows hosted consumers and reviewed platform quality | **Not run**; macOS local results cannot stand in for them. |
| Sprint 225 canonical timing | **Blocked** by missing accepted baseline and exact-candidate timing result. |

Day 7 can inventory and exercise available quality gates, but any hosted candidate claim requires the exact frozen SHA and retained run provenance. These Day 6 results establish that the static consumer procedure works locally, not that the release checklist is complete.
