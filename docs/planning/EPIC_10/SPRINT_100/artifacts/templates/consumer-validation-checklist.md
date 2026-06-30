# Consumer Validation Checklist

Use this checklist when a sprint changes install/export behavior, package
metadata, public headers, version metadata, examples, or platform support
wording.

## Package Metadata

- [ ] repo `VERSION` is the single version source
- [ ] generated `sparse_version.h` is installed
- [ ] `pkg-config --modversion sparse` matches `VERSION`
- [ ] `SparseConfigVersion.cmake` behavior is recorded
- [ ] exact-version CMake behavior is tested when CMake package metadata changes
- [ ] mismatched-version behavior is tested when a lower same-major version can
      be formed

## Make Install and pkg-config

- [ ] `make install PREFIX=<temp>` succeeds
- [ ] static archive is installed
- [ ] public headers are installed
- [ ] generated version header is installed
- [ ] no unexpected shared-library artifacts are installed when static-first is
      still the contract
- [ ] `sparse.pc` is installed
- [ ] `pkg-config --cflags sparse` returns an include path
- [ ] `pkg-config --libs sparse` returns the library flag and required extras
- [ ] a basic external C consumer compiles, links, and runs
- [ ] the maintained example source compiles, links, and runs through
      `pkg-config`
- [ ] `make uninstall PREFIX=<temp>` removes the installed library, headers,
      and `sparse.pc`

## CMake Install and find_package

- [ ] CMake configure succeeds with a temp install prefix
- [ ] CMake build succeeds
- [ ] `cmake --install` succeeds
- [ ] static archive is installed
- [ ] no unexpected shared-library artifacts are installed when static-first is
      still the contract
- [ ] public headers and generated version header are installed
- [ ] `SparseConfig.cmake` is installed
- [ ] `SparseConfigVersion.cmake` is installed
- [ ] `SparseTargets.cmake` is installed
- [ ] `sparse.pc` is installed
- [ ] `examples/cmake_example` configures with `find_package(Sparse)`
- [ ] the CMake example builds and runs
- [ ] exact installed package version is accepted
- [ ] mismatched package version is rejected when applicable

## Platform Interpretation

- [ ] Linux reviewed and supplemental lanes are named separately
- [ ] macOS Apple Clang reviewed path remains distinct from supplemental GCC
      and install/pkg-config confidence lanes
- [ ] Windows reviewed scope is stated as CMake-first consumer proof unless a
      new Windows install or Makefile lane is actually added
- [ ] expected CTest counts are recorded with the workflow that enforces them
- [ ] staged exclusions are named explicitly
- [ ] platform wording does not imply symmetric parity unless each platform has
      matching proof

## ABI and Shared-Library Guardrails

- [ ] static-first wording remains if no shared-library proof was added
- [ ] `BUILD_SHARED_LIBS=ON` behavior is documented if it is relevant to the
      changed surface
- [ ] no dynamic ABI guarantee is implied by exact-version CMake package
      metadata
- [ ] shared-library support is not claimed without runtime-loader and
      downstream consumer proof
- [ ] platform-specific shared-library behavior is not claimed without per-tier
      validation

## Completion Rule

Package or platform claims are not earned unless the artifact records:

- the command;
- the platform;
- installed artifacts;
- downstream consumer behavior;
- version metadata behavior;
- reviewed or supplemental status;
- exclusions and non-claims.
