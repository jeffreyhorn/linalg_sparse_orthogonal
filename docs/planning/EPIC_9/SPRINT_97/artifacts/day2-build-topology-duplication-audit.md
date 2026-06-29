# Sprint 97 Day 2: Build-Topology Duplication Audit

## Purpose

Day 2 ranks the build and workflow duplication candidates from Day 1 by
maintenance cost, proof risk, centralization risk, and fit with Sprint 97. The
goal is to select bounded fix-now candidates without mistaking independent
proof assertions for accidental duplication.

## Ranking Method

Candidates were weighted by:

- number of repeated entries
- frequency of touch during source, test, benchmark, or example additions
- risk of Make/CMake drift
- coupling to CI and install/export proof
- platform-specific exclusions
- ability to centralize without weakening proof strength
- fit with the Sprint 97 sequence

The rerank favors reductions that lower ongoing review cost while keeping
reviewed Make, reviewed CMake, install/export, and platform proof explicit.

## Mechanical Parity Snapshot

Day 2 compared the current Make and CMake registrations mechanically.

| Surface | Make count | CMake count | Current parity | Day 2 reading |
|---|---:|---:|---|---|
| Library sources | 42 | 42 | yes | high-cost duplicated topology |
| Test executables | 54 | 54 | yes | high-cost duplicated topology plus platform gates |
| Benchmark executables | 16 | 16 | yes | medium-cost duplicated topology |
| Example executables | 12 via wildcard | 12 explicit | yes | lower-cost mixed registration model |

No current Make-only or CMake-only entries were found in the library source,
test, benchmark, or example sets. The audit therefore ranks future drift risk
and maintenance cost, not an existing parity failure.

## Ranked Duplication Candidates

### 1. Library Source List

Surfaces:

- `Makefile` `LIB_SRCS`
- `CMakeLists.txt` `add_library(sparse_lu_ortho STATIC ...)`

Current signal:

- 42 library source files are listed in both surfaces.
- Sprint 96 added focused source owners such as `src/sparse_iterative_block.c`
  and `src/sparse_ldlt_dense.c`, both of which required Make and CMake
  registration updates.
- Library source registration is correctness-critical because missing a source
  can break one build system while the other still passes.

Maintenance cost: very high.

Drift risk: high.

Centralization risk: medium. A shared generated include, manifest, or check can
reduce duplication, but both build systems must keep readable, reviewed source
membership.

Proof role: build topology, not independent behavioral proof.

Sprint 97 role: strongest fix-now candidate for Day 3 architecture design.

### 2. Test Registration List

Surfaces:

- `Makefile` `TEST_SRCS`
- `CMakeLists.txt` `add_sparse_test(...)`
- `make quality-review-cmake-compile` Make/CMake test-count parity check
- `.github/workflows/windows-ci.yml` expected Windows CTest count

Current signal:

- 54 Make tests and 54 CMake tests are in parity locally.
- CMake also owns platform exclusions, including pthread tests and fuzz gating.
- Windows CI expects 51 CTest registrations and explicitly excludes
  `test_threads`, `test_sprint4_integration`, and `test_fuzz`.
- The local Make/CMake count parity check is already a valuable proof
  assertion.

Maintenance cost: very high.

Drift risk: high.

Centralization risk: medium-high. Test registration is not just a list; it also
encodes platform eligibility and reviewed-scope messaging.

Proof role: both build topology and proof-surface assertion.

Sprint 97 role: second fix-now candidate, but only after Day 3 defines how to
preserve platform exclusions and test-count assertions.

### 3. Benchmark Registration List

Surfaces:

- `Makefile` `BENCH_SRCS`
- CMake benchmark `add_executable(...)` blocks
- `bench-build`, `bench-fast`, and benchmark reporting targets

Current signal:

- 16 benchmark programs are represented in both Make and CMake.
- Some CMake benchmark registrations are platform-gated because several
  benches use POSIX-only APIs.
- Benchmark compile coverage matters, but benchmark registration drift is less
  likely to break correctness proof than library or test drift.

Maintenance cost: medium.

Drift risk: medium.

Centralization risk: medium. Any reduction must preserve specialized benchmark
subsets such as `bench-fast` and canonical report targets.

Proof role: runtime and reporting support, not primary correctness proof.

Sprint 97 role: residual unless Day 3 finds a low-risk manifest structure that
also helps library or test registration.

### 4. Example Registration List

Surfaces:

- `Makefile` `EX_SRCS = $(wildcard $(EXDIR)/*.c)`
- explicit CMake `add_executable(example_...)` blocks
- README, examples README, and install consumer guidance

Current signal:

- 12 examples are present and currently in parity.
- Make already discovers examples by wildcard.
- CMake explicitly registers examples, which keeps target names and link
  behavior visible but creates an update point when adding examples.

Maintenance cost: low-medium.

Drift risk: medium.

Centralization risk: low-medium.

Proof role: public usage and compile coverage support.

Sprint 97 role: residual unless folded into a broader manifest after higher
value list surfaces are solved.

### 5. Install/Export Package Proof

Surfaces:

- `CMakeLists.txt` install/export rules
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- `INSTALL.md`
- README package guidance
- `sparse.pc.in`
- `cmake/SparseConfig.cmake.in`

Current signal:

- The package story is currently static-first.
- Make install plus pkg-config proof and CMake install plus `find_package`
  proof are intentionally separate.
- Several surfaces repeat the same static-first contract so consumer-facing
  claims do not drift.

Maintenance cost: medium-high.

Drift risk: medium.

Centralization risk: high if treated as pure duplication. These surfaces are
different forms of consumer proof and public contract.

Proof role: consumer/install/export proof and product claim consistency.

Sprint 97 role: preserve for Day 7-9 package decision work, not Day 3
source-list centralization.

### 6. Workflow Reviewed-Scope Messages And Expected Counts

Surfaces:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- README cross-platform CI contract
- Make reviewed quality target names and messages

Current signal:

- Linux workflow names the strongest reviewed source-of-truth baseline.
- macOS workflow separates enforced Apple Clang proof from supplemental GCC
  and install/pkg-config confidence.
- Windows workflow names the reviewed CMake-first consumer subset, expected
  CTest count, and staged exclusions.
- These messages are repetitive, but they are intentionally visible proof
  boundaries.

Maintenance cost: medium.

Drift risk: medium-high when test counts or package claims change.

Centralization risk: high. Moving the language too far away from workflows can
make CI logs less self-explanatory.

Proof role: reviewed-scope assertion and platform truth calibration.

Sprint 97 role: preserve as explicit proof in Day 3; revisit during Day 10-12
workflow and platform follow-through.

## Fix-Now Queue

| Rank | Candidate | Recommended Sprint 97 handling |
|---:|---|---|
| 1 | Library source list | Design a bounded convergence path on Day 3. Prefer a manifest, generated fragment, or automated parity assertion that reduces two-list drift without hiding reviewed source membership. |
| 2 | Test registration list | Design only if platform exclusions and test-count assertions remain explicit. This may become a proof-preserving check rather than full centralization. |
| 3 | Benchmark registration list | Keep residual unless it falls out naturally from a shared manifest approach. |
| 4 | Example registration list | Keep residual; Make wildcard already lowers one side of the cost. |
| 5 | Package/install/export proof | Defer to Day 7-9 package decision and consumer proof follow-through. |
| 6 | Workflow messages and expected counts | Defer to Day 10-12 workflow/platform calibration unless Day 3 needs a small assertion helper. |

## Preserve-Independent-Proof Split

The following repeated surfaces should not be removed merely because they are
duplicated:

- local Make/CMake test-count parity assertion in
  `make quality-review-cmake-compile`
- Windows expected CTest count and staged-exclusion messages
- macOS reviewed versus supplemental lane descriptions
- Linux reviewed versus supplemental lane descriptions
- Make install/pkg-config proof separate from CMake install/find_package proof
- static-first package wording in both public docs and build configuration

These surfaces repeat claims on purpose. Day 3 should preserve the claim at the
point where users or maintainers encounter the proof.

## Platform-Specific Duplication Notes

### Linux

Linux duplicates several local Make targets inside CI because it is the
strongest reviewed baseline. This is mostly intentional. The main audit risk
is stale naming between workflow jobs and Make target messages.

### macOS

macOS keeps Apple Clang reviewed proof separate from supplemental Homebrew GCC
and Make install/pkg-config confidence. That split should remain explicit.
Reducing text duplication is lower priority than preserving lane meaning.

### Windows

Windows has the most fragile count assertion because the reviewed CMake subset
is smaller than the full local Make/CMake test surface. Any test-registration
convergence must keep Windows exclusions and the expected CTest count visible.

## Day 2 Result

The highest-value convergence candidate is the library source list. The test
registration list is the second candidate but has more proof semantics because
it encodes platform eligibility and reviewed CTest counts. Day 3 should design
a bounded approach around source-list convergence first, then decide whether
test registration receives centralization, an automated parity check, or only
clearer proof-preserving assertions.
