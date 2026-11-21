# Sprint 226 Day 9: Clean-Prefix First Use

## Scope and source identity

This is a **local exploratory first-use proof**, not a frozen release-candidate result. The source archive is the Day 3 base-commit archive `sparse-lu-ortho-2.2.0-src-EXPLORATORY.tar.gz` (SHA-256 `cdfceff5ef4fdf405f8bf3b15e6a7fe608d5f4566345ea3fd2f8d6ec0e27be8f`) from `c42d6715fad7e895bae4941a006df3563f1afc38`. It was extracted afresh into ignored `build/sprint226/day9/sparse-lu-ortho-2.2.0/`. Builds and prefixes under `build/sprint226/day9/` did not reuse Days 3–8 binaries or install trees. The commands below follow the edited working-tree README/INSTALL text, while the extracted archive still contains the pre-edit documentation. Therefore this archive and earlier hashes **cannot** be promoted as proof of a package containing the Day 9 documentation corrections. A future selected candidate needs a new committed SHA, archive, install and hosted proofs.

Host: macOS 15.7.9 x86_64, Apple Clang 17.0.0, CMake 4.4.4. All three installs used separate absolute, initially empty local prefixes. Each consumer build used a separate build tree and selected its matching prefix.

## Command transcript and results

`source_dir` below is the fresh extracted root, and `root` is the repository root. Logs are retained at `build/sprint226/day9/` and ignored by Git.

```sh
root="$PWD"
source_dir="$root/build/sprint226/day9/sparse-lu-ortho-2.2.0"
mkdir -p build/sprint226/day9
tar -xzf build/sprint226/day3/sparse-lu-ortho-2.2.0-src-EXPLORATORY.tar.gz -C build/sprint226/day9

# README -> examples/README.md first local solve
make -C "$source_dir"
make -C "$source_dir" examples
"$source_dir/build/example_basic_solve"

# INSTALL.md Unix Make/pkg-config tutorial (default 32-bit width)
make -C "$source_dir" install PREFIX="$root/build/sprint226/day9/make-prefix"
export PKG_CONFIG_PATH="$root/build/sprint226/day9/make-prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
pkg-config --exists sparse
pkg-config --cflags sparse
pkg-config --libs sparse
# In build/sprint226/day9/pkg-consumer, with main.c copied exactly from INSTALL.md:
cc -std=c11 $(pkg-config --cflags sparse) main.c $(pkg-config --libs sparse) -o sparse_pkgconfig_smoke
./sparse_pkgconfig_smoke
```

The basic example returned solution `[1,1,1,1,1]` and residual `0`. `pkg-config --variable=prefix sparse` resolved the exact Day 9 Make prefix. Its `--cflags` contained that prefix's `include` directory and `-DSPARSE_PACKAGE_IDX_BITS=32`; `--libs` contained that prefix's `lib`, `-lsparse_lu_ortho` and `-lm`. The tutorial's copied program compiled and reported `index width: 32`, version `2.2.0`, `nnz: 1`, `OK`. The Make package passed `check_static_install_contract.py --mode make --platform unix --version 2.2.0`.

The CMake route followed the edited [CMake Build](../../../../../INSTALL.md#cmake-build) staged-prefix commands with an external `build/sprint226/day9/cmake-build` tree and `cmake-prefix` instead of the guide's generic `build/install` and `_install` names. A second 64-bit CMake build used a different tree and prefix:

```sh
cmake -S "$source_dir" -B build/sprint226/day9/cmake-build \
  -DCMAKE_INSTALL_PREFIX="$root/build/sprint226/day9/cmake-prefix" \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_C_STANDARD=11
cmake --build build/sprint226/day9/cmake-build --parallel 2
ctest --test-dir build/sprint226/day9/cmake-build --output-on-failure
cmake --install build/sprint226/day9/cmake-build
cmake -S "$source_dir/examples/cmake_example" \
  -B build/sprint226/day9/cmake-consumer \
  -DCMAKE_PREFIX_PATH="$root/build/sprint226/day9/cmake-prefix"
cmake --build build/sprint226/day9/cmake-consumer
build/sprint226/day9/cmake-consumer/example

cmake -S "$source_dir" -B build/sprint226/day9/cmake64-build \
  -DCMAKE_INSTALL_PREFIX="$root/build/sprint226/day9/cmake64-prefix" \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_C_STANDARD=11 -DSPARSE_IDX_BITS=64
cmake --build build/sprint226/day9/cmake64-build --target sparse_lu_ortho --parallel 2
cmake --install build/sprint226/day9/cmake64-build
cmake -S "$source_dir/examples/cmake_example" \
  -B build/sprint226/day9/cmake64-consumer \
  -DCMAKE_PREFIX_PATH="$root/build/sprint226/day9/cmake64-prefix"
cmake --build build/sprint226/day9/cmake64-consumer
build/sprint226/day9/cmake64-consumer/example
```

The 32-bit CMake build completed **63/63 CTests** and installed into `cmake-prefix`. Its maintained consumer cache bound `Sparse_DIR` to that prefix and printed width `32`, version `2.2.0`, solution `[1,1,1]`, `OK`. The CMake tutorial's exact `CMakeLists.txt` and the same copied `main.c` were also configured as a separate consumer against that prefix; they compiled and printed width `32`, `nnz: 1`, `OK`. The 64-bit installed consumer's CMake cache records the correct `Sparse_DIR` under `cmake64-prefix/lib/cmake/Sparse`; its build cache records `SPARSE_IDX_BITS=64`. It ran the maintained solve and printed width `64`, version `2.2.0`, solution `[1,1,1]`, `OK`. Both CMake static packages passed `check_static_install_contract.py --mode cmake --platform unix --version 2.2.0`.

## First-use documentation corrections

| File | Correction | Reason and check |
| --- | --- | --- |
| [README](../../../../../README.md) | Replaced the duplicate `/usr/local` install command block with links to the staged Make/CMake guides and installed-consumer tutorial. Updated the external-baseline paragraph to acknowledge Sprint 222's two-sided local execution while retaining the unaccepted/no-publication status. | The former sketch could require system-wide write access; Sprint 215/221-only prose had become stale. Checked the links and the [Sprint 222 retrospective](../../SPRINT_222/RETROSPECTIVE.md). |
| [INSTALL](../../../../../INSTALL.md) | Made the Make and CMake quick starts use writable `_install` prefixes; removed duplicate CMake install commands from the consumer subsection; corrected the consumer prefix example to match the preceding 32-bit install; updated the support matrix's external-baseline row. | The old `cmake-64` prefix did not match the default-width install. The exact Make tutorial program and maintained CMake consumer were compiled and run from fresh prefixes. No accepted external comparison is claimed. |

The local Markdown audit checked README, INSTALL, examples README and API reference: 92, 19, 27 and 12 relative file links respectively, and 45, 17, 12 and 1 anchor links. No missing local target or unmatched Markdown anchor was found in this scoped check. External URLs and generated HTML were not part of it. The installed Make, 32-bit CMake and 64-bit CMake prefixes each contain all 18 checked-in public headers plus generated `sparse_version.h`; the [API reference](../../../../../docs/api_reference.md#source-of-truth) lists the 18 and describes the generated header. There is no undocumented installed header in this comparison.

## Final verification and remaining gate

| Evidence | Result | SHA-256 of retained log or installed archive |
| --- | --- | --- |
| Make installed tutorial run, `pkg-run.log` | width 32, version 2.2.0, `nnz: 1`, `OK` | `8a5cb2671d34deda99dab2be309e4d0d315f1da986cb2f0ca77e54d4443efd66` |
| Make installed static archive | contract validator passed | `5f6e64a8b0ad632fc5ff22b5aea166e3d71c7bb3379898a7f1a5dd3ac7b6dc19` |
| Default CMake CTest, `cmake-ctest.log` | **63/63 passed** | `ad2a97738d44f9eb728d776e469c5f54e1e78c4c9b4f789983a05c7013cc9a03` |
| Default CMake maintained solve, `cmake-consumer-run.log` | width 32, version 2.2.0, `[1,1,1]`, `OK` | `5c55dce7178720a9b736330cbc4516830b4f1ffda4031825cca051d95268e66e` |
| Default CMake exact tutorial, `cmake-tutorial-run.log` | width 32, `nnz: 1`, `OK` | `8a5cb2671d34deda99dab2be309e4d0d315f1da986cb2f0ca77e54d4443efd66` |
| Default CMake installed static archive | contract validator passed | `e95a35a94dd25f07c9d7a007b8ae5fc80c63c609694d35db32c9c6298bc20dc8` |
| 64-bit CMake maintained solve, `cmake64-consumer-run.log` | width 64, version 2.2.0, `[1,1,1]`, `OK` | `858a3641502b85dce55b6b6a762266f042b93c44d95d61240d4dc322df1fe6e1` |
| 64-bit CMake installed static archive | contract validator passed | `c49058f6951172d846f22243714a2b1d50f29f38c6de95036cad22ed8171fe2a` |

`make support-docs-guard`, `bash scripts/static_package_deferral_check.sh`, and `python3 tests/test_api_docs_routing.py` passed after the text edits. `git diff --check` and the scoped local link/anchor audit passed. Day 9 changed Markdown only; C/header format, lint and test gates were not triggered by source edits. The clean-prefix commands and focused documentation/contract checks are separate from the missing frozen-candidate and hosted-platform proof. The release candidate remains blocked by missing approved changelog/version decision, uncommitted sprint/documentation changes, unassigned release/rollback owners, exact-candidate archive/install and hosted reruns, incomplete historical Linux parity, and the Sprint 225 timing baseline or explicit scope decision.
