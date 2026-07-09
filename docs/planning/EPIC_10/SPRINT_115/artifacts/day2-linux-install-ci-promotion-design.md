# Day 2 Linux Install Proof CI Promotion Design

## Purpose

Day 2 designs the evidence contract for deciding whether the local Unix-side
install proofs should become reviewed Linux CI lanes. The goal is to make Day
3 a bounded promotion/no-promotion decision without changing CI or package
claims during the design step.

## Current Linux CI Contract

`.github/workflows/ci.yml` currently says Linux is the enforced reviewed
source-of-truth baseline for:

- reviewed Makefile compile-quality path;
- reviewed CMake parity path;
- dead-code report/check path.

It also carries supplemental Linux signals:

- direct `make test`;
- UBSan;
- ASan;
- TSan;
- benchmark compile plus `bench-fast`;
- coverage.

The workflow explicitly documents that focused install/package regression
scripts remain developer-side proof surfaces rather than a separate reviewed CI
lane. Sprint 115 Day 3 must either preserve that contract or update it with a
narrow reviewed install lane and matching support wording.

## Local Proof Inventory

### Make Install and `pkg-config`

`tests/test_install.sh` proves the local Unix-side Make install surface:

| Proof | Evidence in script |
|---|---|
| clean staged install | runs `make clean` and `make install PREFIX=<tmp>` |
| static library package shape | checks `lib/libsparse_lu_ortho.a` |
| no shared-library artifacts | rejects `.so`, `.so.*`, `.dylib`, and `.dll` under installed lib/bin paths |
| installed headers | counts public headers plus generated `sparse_version.h` |
| `pkg-config` metadata | checks installed `lib/pkgconfig/sparse.pc` |
| compiler flags | checks `pkg-config --cflags sparse` includes an include path |
| linker flags | checks `pkg-config --libs sparse` includes `-lsparse_lu_ortho` |
| version metadata | checks `pkg-config --modversion sparse` against `VERSION` |
| downstream compile/link/run | compiles and runs a generated installed consumer |
| maintained example source | compiles and runs `examples/cmake_example/main.c` through `pkg-config` |
| uninstall | runs `make uninstall` and verifies library, headers, and `sparse.pc` are removed |

### CMake Install, Export, and `find_package`

`tests/test_cmake_install.sh` proves the local Unix-side CMake install/export
surface:

| Proof | Evidence in script |
|---|---|
| CMake configure/build/install | configures, builds, and installs into a temp prefix |
| static library package shape | checks installed `lib/libsparse_lu_ortho.a` |
| no shared-library artifacts | rejects `.so`, `.so.*`, `.dylib`, and `.dll` |
| installed headers | verifies installed public headers |
| CMake package files | checks `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and `SparseTargets.cmake` |
| `pkg-config` metadata | checks installed `sparse.pc` |
| installed CMake consumer | configures/builds/runs `examples/cmake_example` with `find_package(Sparse)` |
| exact version contract | checks `find_package(Sparse <VERSION> EXACT REQUIRED)` |
| mismatch rejection | rejects lower same-major version when one exists |
| pkg-config version | checks `pkg-config --modversion sparse` against `VERSION` |

## Reviewed-Lane Promotion Criteria

Day 3 may promote the local proof into Linux CI only if all of the following
are true:

1. The lane can run both `bash tests/test_install.sh` and
   `bash tests/test_cmake_install.sh` from a clean Ubuntu checkout.
2. Required tools are explicit and stable on `ubuntu-latest`:
   `make`, C compiler, CMake, and `pkg-config`.
3. The lane has a clear reviewed claim:
   Linux reviewed static install/export proof, not broader package-manager,
   shared-library, dynamic ABI, Windows, or macOS parity.
4. The lane does not duplicate an existing reviewed path without adding
   support-surface value.
5. Runtime and cache impact are acceptable relative to the existing reviewed
   Linux CMake and compile-quality gates.
6. The workflow comments, README, INSTALL guide, and maintainer guide are
   updated if the support contract changes.
7. Failure output remains actionable enough for maintainers to distinguish
   package metadata failures from compiler or environment failures.

## Local-Only No-Promotion Criteria

Day 3 should publish a no-promotion contract instead if any of the following
hold:

1. The scripts remain valuable as local release/package proof but do not need
   to run on every PR.
2. The existing reviewed CMake parity and Makefile compile-quality lanes are
   already the stronger Linux source-of-truth for ordinary PR review.
3. Adding the scripts would materially increase CI runtime or flakiness without
   changing a public support claim.
4. CI dependency installation would become more complex than the reviewed
   claim warrants.
5. Existing docs remain more accurate if install scripts stay described as
   developer-side/local proof.

## CI Surface Risks

| Risk | Impact | Mitigation if promoted |
|---|---|---|
| runtime expansion | PR latency increases because both scripts build/install downstream consumers | put both scripts in one focused install job and keep it static-first |
| environment dependency | missing or changed `pkg-config`, CMake, compiler, or install path behavior can fail the job | install dependencies explicitly and log versions |
| duplicate build cost | scripts rebuild pieces already covered by reviewed compile/CMake paths | document that the lane proves installed consumers, not compile parity |
| support overclaim | reviewed Linux install proof could be misread as package-manager, shared-library, or dynamic ABI support | update docs/comments with explicit non-claims |
| workflow drift | reviewed lane may require new expected checks in future package edits | assign ownership in Sprint 115 artifacts and maintainer docs if promoted |

## Day 3 Decision Checklist

Before changing CI on Day 3, answer:

1. Does a reviewed Linux install lane materially improve public package truth?
2. Is the lane narrow enough to prove only static install/export and installed
   consumers?
3. Can the lane run without new unstable dependencies?
4. Are docs and workflow comments updated to match the promoted or local-only
   decision?
5. Is the validation command set clear for any touched files?

## Non-Claims

- No CI workflow was changed on Day 2.
- No package, ABI, shared-library, package-manager, Windows, or macOS support
  claim changed on Day 2.
- Local install scripts remain developer-side proof until Day 3 explicitly
  promotes them or publishes a no-promotion contract.
- Static-first package support remains the only maintained package surface.

## Day 2 Validation

Day 2 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_115
```

Full C quality gates are not required for Day 2 because no `.c` or `.h` files
changed.
