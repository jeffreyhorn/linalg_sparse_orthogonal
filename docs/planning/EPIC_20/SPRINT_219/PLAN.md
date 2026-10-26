# Sprint 219 Plan: Enforced Installed Index-Width Contract

**Sprint Duration:** 14 days
**Goal:** Make 32/64-bit `idx_t` selection explicit and safe for installed
static consumers instead of relying on a header warning.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds
12 hours; the Epic 20 sprint capacity is 168 hours. CI elapsed time is not
included in the estimates.

**Prerequisites:** Sprint 218's source and consumer baseline, Sprint 214's
static install/export tests, the public `SPARSE_IDX_BITS` typedef, and the
existing `sparse_idx_bits()` library query. Preserve the static-first package
boundary and the default 32-bit build.

**Primary scope:** `include/sparse_types.h`, generated package metadata,
`CMakeLists.txt`, `Makefile`, `sparse.pc.in`, installed CMake target usage
requirements, `tests/test_install.sh`, clean-prefix CMake consumers, package
validation scripts, and selected 64-bit solver/IO/conversion tests. Use
separate build and install trees for each width; never relabel a 32-bit
archive as 64-bit by changing consumer flags alone.

**Evidence rule:** Record exact source revision, compiler, width selector,
build tree, install prefix, archive identity, consumer command, exit status,
and expected failure phase. Negative mismatch tests must fail before any
`idx_t` array or index-bearing struct crosses the library boundary. A failed,
skipped, or unavailable required gate blocks completion; a local run does not
stand in for a required hosted lane.

**Non-goals:** Shared-library packaging, cross-version ABI compatibility,
mixed-width interoperability, a claim that all algorithms support matrices
larger than `INT32_MAX`, or a broad 64-bit correctness claim beyond the tested
lane.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 219.1 ABI Width Inventory | 20 h | Days 1-2: 10 + 10 |
| 219.2 Build Configuration | 30 h | Days 3-5: 10 + 10 + 10 |
| 219.3 Consumer Propagation | 26 h | Days 6-8: 10 + 10 + 6 |
| 219.4 Positive/Negative Consumers | 24 h | Days 8-10: 4 + 10 + 10 |
| 219.5 Selected 64-Bit Lane | 20 h | Days 11-12: 10 + 10 |
| 219.6 Contract Validation | 20 h | Days 13-14: 10 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Public ABI Width Inventory

**Title:** Public ABI Width Inventory
**Theme:** Locate every caller/library boundary that depends on `idx_t` size.
**Time estimate:** 10 hours (219.1: 10 h)

### Tasks

1. Trace `SPARSE_IDX_BITS`, `idx_t`, `IDX_MAX`, `sparse_idx_bits()`, and the
   generated version header through public headers and library objects.
2. Catalog public structures with index fields or arrays, callbacks taking
   `idx_t`, pointer-returning APIs, and size-sensitive IO/conversion and solver
   entry points. Distinguish opaque handles from caller-allocated layouts.
3. Inventory default Make/CMake compile flags, installed headers/archive,
   `Sparse::sparse_lu_ortho`, `sparse.pc`, package validators, and downstream
   consumer tests. Note where a consumer can currently override the width.
4. Start `WORKING_NOTES.md` with an owner map, risk ledger, baseline commands,
   and a table of supported versus untested width claims.

### Deliverables

- Public index-bearing ABI inventory and current package-propagation map.
- Baseline risk and evidence log without changing the ABI yet.

### Completion Criteria

- Every public width-sensitive boundary and both installed-consumer routes
  have a named owner and test candidate.
- The default 32-bit behavior and the present silent-mismatch path are
  reproducible with exact commands.

---

## Day 2: Width Contract And Failure Design

**Title:** Width Contract And Failure Design
**Theme:** Define how supported consumers receive or reject a width.
**Time estimate:** 10 hours (219.1: 10 h)

### Tasks

1. Specify accepted selectors (`32`, `64`), the default, unsupported-value
   diagnostics, and what a clean build means for Make versus CMake.
2. Define one installed width identity, how the exported target and
   `pkg-config` convey it, and how an explicit conflicting consumer setting
   is rejected. Address duplicate `-D` flags and direct-header use rather
   than assuming the last macro wins safely.
3. Design a small executable package-width check using `sizeof(idx_t)` and
   the archive's `sparse_idx_bits()` result; specify the check point before
   calling any index-bearing API. Choose a compile/link guard where needed
   so supported consumer routes cannot silently bypass the contract.
4. Freeze a positive/negative matrix covering 32/64 packages, both consumer
   routes, wrong selectors, stale archives, and mixed prefixes. Define
   expected failure phase and diagnostic for each negative case.

### Deliverables

- Width contract and mismatch-detection design with explicit limitations.
- Test matrix and acceptance criteria for items 219.2-219.5.

### Completion Criteria

- The design explains why a 64-bit header cannot silently call a 32-bit
  archive through either maintained installed-consumer route.
- Runtime checks occur before unsafe use and do not masquerade as broad ABI
  compatibility.
- Item 219.1's 20-hour inventory/design allocation is closed.

---

## Day 3: CMake Width Selection

**Title:** CMake Width Selection
**Theme:** Build the library and tests from one validated CMake choice.
**Time estimate:** 10 hours (219.2: 10 h)

### Tasks

1. Add a named CMake cache selector with default `32` and reject values
   other than `32` or `64` at configure time with a clear error.
2. Apply the selected definition to the static library, test binaries,
   examples, and benchmarks without relying on ambient `CFLAGS`.
3. Confirm the library's `sparse_idx_bits()` reports the configured value
   from distinct clean 32- and 64-bit build trees.
4. Add configure-level negative checks for `0`, `16`, nonnumeric, and
   conflicting selector attempts; retain diagnostics in working notes.

### Deliverables

- Validated CMake width selector and separately built 32/64 archives.
- Configure rejection and library-width smoke evidence.

### Completion Criteria

- Both supported CMake configurations compile against their own width.
- Unsupported values fail at configure, not after a consumer uses an archive.

---

## Day 4: Make Width Selection

**Title:** Make Width Selection
**Theme:** Give Make builds the same explicit width contract.
**Time estimate:** 10 hours (219.2: 10 h)

### Tasks

1. Add a validated Make `SPARSE_IDX_BITS` setting, defaulting to `32`, and
   propagate it to library, tests, examples, benchmarks, and lint compile
   commands.
2. Prevent width changes from silently reusing an archive or object files
   built for another width. Document or enforce separate `BUILDDIR` values
   and check for an incompatible cached build identity.
3. Build and run a focused width smoke test in clean 32- and 64-bit Make
   trees; compare `sizeof(idx_t)` with `sparse_idx_bits()`.
4. Exercise unsupported and conflicting Make selections and retain their
   expected early failures.

### Deliverables

- Validated Make width selector with clean-tree isolation.
- Focused Make 32/64 and invalid-selector evidence.

### Completion Criteria

- No Make target silently links a stale opposite-width archive.
- Both supported widths compile and report their own library identity.

---

## Day 5: Build-System Parity And Width Identity

**Title:** Build-System Parity And Width Identity
**Theme:** Reconcile selected widths across builds and installed artifacts.
**Time estimate:** 10 hours (219.2: 10 h)

### Tasks

1. Compare CMake and Make defaults, option spellings, diagnostics, generated
   headers, and archive-reported widths in clean 32/64 trees.
2. Add focused width-identity regression coverage that checks the configured
   library and public typedef agree, without requiring huge allocations.
3. Inspect CMake/Make install paths and decide how package metadata records
   width without embedding build-tree paths or mutating the source header.
4. Audit warnings and any 64-bit format/cast assumptions exposed by the
   small smoke case; fix build-contract defects within scope and log
   algorithmic defects for the selected lane.

### Deliverables

- Build-system parity table and reusable width-identity test.
- Stable inputs for package metadata work on Days 6-8.

### Completion Criteria

- Clean Make and CMake builds agree on default and selected width behavior.
- Item 219.2's 30-hour build-configuration allocation is closed.

---

## Day 6: CMake Installed Target Propagation

**Title:** CMake Installed Target Propagation
**Theme:** Make `find_package(Sparse)` carry the archive's width.
**Time estimate:** 10 hours (219.3: 10 h)

### Tasks

1. Export the chosen width through the `Sparse::sparse_lu_ortho` usage
   requirements and/or package config, preserving the static-only target.
2. Install any generated width metadata alongside headers and verify it is
   relocatable and tied to the archive in that prefix.
3. Build clean-prefix downstream CMake consumers for both widths without
   manually supplying `-DSPARSE_IDX_BITS`; assert typedef and library query.
4. Test an explicit conflicting consumer definition and identify whether it
   fails at configure, compile, or before the first index-bearing call.

### Deliverables

- Width-aware installed CMake target and clean-prefix consumer smoke tests.
- Exported metadata and mismatch diagnostic evidence.

### Completion Criteria

- The imported target automatically supplies the installed archive's width.
- A deliberate conflicting width cannot yield a seemingly working consumer.

---

## Day 7: pkg-config Propagation

**Title:** pkg-config Propagation
**Theme:** Carry the same width through Make and CMake package metadata.
**Time estimate:** 10 hours (219.3: 10 h)

### Tasks

1. Update `sparse.pc.in` and both generation paths so `pkg-config --cflags
   sparse` exposes the selected width, without changing the static link
   contract or leaking a build-tree path.
2. Extend the static-install contract checker to validate width metadata,
   permitted cflags, and agreement among `sparse.pc`, installed headers,
   and the archive's reported width.
3. Compile and run clean-prefix `pkg-config` consumers at 32 and 64 bits,
   using the emitted flags without an extra manual width definition.
4. Check mutation cases for missing, duplicated, unsupported, and mismatched
   width flags in package metadata.

### Deliverables

- Width-bearing pkg-config metadata for both install routes.
- Package-validator mutations and positive downstream compile/run evidence.

### Completion Criteria

- Make and CMake packages advertise the same width as their archives.
- Invalid metadata fails validation instead of reaching a consumer.

---

## Day 8: Executable Width Check And First Consumer

**Title:** Executable Width Check And First Consumer
**Theme:** Finish package propagation and start installed-consumer proof.
**Time estimate:** 10 hours (219.3: 6 h; 219.4: 4 h)

### Tasks

1. Implement and document the designed executable package-width check,
   comparing the consumer's `sizeof(idx_t)` with `sparse_idx_bits()` before
   any index array or public index-bearing struct is passed to the library.
2. Integrate the check into the maintained CMake and pkg-config consumer
   fixtures; ensure it cannot be optimized away or skipped by a zero-test
   path. Close item 219.3's remaining 6 hours.
3. Run one clean-prefix 32-bit and one 64-bit positive consumer through
   configure/compile/link/check/use, recording exact archive and header paths.
4. Confirm a deliberate mismatch is rejected at the documented phase and
   does not execute the unsafe API call.

### Deliverables

- Reusable executable width check and first positive/negative consumers.
- Recorded package paths and early-failure evidence.

### Completion Criteria

- The check distinguishes both supported widths and rejects a mismatch
  before the consumer uses index-bearing data.
- Item 219.3's 26-hour propagation allocation is closed.

---

## Day 9: CMake Consumer Matrix

**Title:** CMake Consumer Matrix
**Theme:** Prove automatic width selection and deliberate rejection.
**Time estimate:** 10 hours (219.4: 10 h)

### Tasks

1. Install independent 32- and 64-bit CMake packages into clean prefixes;
   build downstream consumers using only each prefix's `find_package` target.
2. Exercise wrong explicit width, wrong header path, wrong archive path,
   and a stale CMake cache. Verify failure before an index-bearing call.
3. Check static archive identity and exact package version without
   claiming cross-version compatibility or shared-library support.
4. Add these cases to the existing CMake install/downstream suite with
   isolated build/prefix paths and stable failure assertions.

### Deliverables

- CMake installed-consumer positive/negative matrix for both widths.
- Reproducible archive-identity and stale-cache rejection results.

### Completion Criteria

- Both clean-prefix CMake consumers run with their own width automatically.
- All mismatched combinations fail at the expected safe phase.

---

## Day 10: pkg-config And Mixed-Prefix Matrix

**Title:** pkg-config And Mixed-Prefix Matrix
**Theme:** Close both installed-consumer routes end to end.
**Time estimate:** 10 hours (219.4: 10 h)

### Tasks

1. Run `tests/test_install.sh`-style positive consumers for Make-installed
   and CMake-installed 32/64 packages in isolated prefixes.
2. Mutate `PKG_CONFIG_PATH`, include order, and archive lookup to create
   intentional mixed-width combinations; assert the checker or compiler
   rejects them before any unsafe call.
3. Verify the tests cannot pass against an unrelated host installation and
   that negative-link expectations are portable across supported linkers.
4. Record a compact four-route pass/fail matrix: Make/CMake install times
   32/64 width, exercised through both applicable consumer mechanisms.

### Deliverables

- Clean-prefix pkg-config and mixed-prefix regression cases.
- Cross-route archive/header/metadata identity table.

### Completion Criteria

- Both widths pass the maintained installed-consumer paths, and wrong-width
  or mixed-prefix cases fail before index-bearing use.
- Item 219.4's 24-hour positive/negative consumer allocation is closed.

---

## Day 11: Selected 64-Bit Functional Lane

**Title:** Selected 64-Bit Functional Lane
**Theme:** Exercise real library behavior beyond typedef-size checks.
**Time estimate:** 10 hours (219.5: 10 h)

### Tasks

1. Select small deterministic 64-bit cases covering matrix creation and
   insertion, a representative direct or iterative solver, Matrix Market
   IO, and CSR/CSC conversion. Record exact test selectors and expected
   values before execution.
2. Run the selected cases against a genuinely 64-bit library and test
   binaries; use `sparse_idx_bits()` plus build metadata to confirm identity.
3. Add a boundary-focused case for index conversion/formatting that does
   not require allocating a matrix with billions of entries.
4. Fix in-scope width assumptions revealed by the selected lane and rerun
   the affected cases; retain any failures and their resolution.

### Deliverables

- Selected solver, IO, and conversion 64-bit regression lane.
- Exact commands, expected results, and width-identity evidence.

### Completion Criteria

- The lane executes real 64-bit library objects, not only a 64-bit consumer
  typedef linked to a default archive.
- Selected results meet their numerical and structural expectations.

---

## Day 12: Hosted 64-Bit Registration

**Title:** Hosted 64-Bit Registration
**Theme:** Keep the selected lane recurring and scoped.
**Time estimate:** 10 hours (219.5: 10 h)

### Tasks

1. Register the selected 64-bit build and tests in the relevant hosted
   workflow, with a clean build tree and explicit width selector. Preserve
   default-width reviewed lanes.
2. Run the selected lane locally on available platforms and inspect hosted
   result links; distinguish queued, skipped, unavailable, failed, and passed.
3. Confirm hosted install/consumer jobs use the same package-width checks;
   keep Windows claims limited to the supported CMake install route if
   pkg-config execution is not reviewed there.
4. Record any platform-specific width warnings or format assumptions and
   either resolve them within scope or leave the sprint gate explicitly open.

### Deliverables

- Recurring hosted selected-64-bit lane and provenance/exit-status log.
- Platform-scoped result table without an all-algorithm 64-bit claim.

### Completion Criteria

- Required hosted 64-bit evidence is actually green; local success alone
  is not called hosted validation.
- Item 219.5's 20-hour selected-lane allocation is closed.

---

## Day 13: Public Contract And Install Documentation

**Title:** Public Contract And Install Documentation
**Theme:** Explain what consumers must select, check, and not assume.
**Time estimate:** 10 hours (219.6: 10 h)

### Tasks

1. Update `include/sparse_types.h` and relevant public API text to state
   the configured width, default, build-time choice, and mismatch behavior.
2. Update `INSTALL.md` and package guidance with separate Make and CMake
   32/64 examples, clean-prefix `pkg-config`/`find_package` use, and the
   executable width check before index-bearing calls.
3. State the exclusions explicitly: static-only package, no mixed-width or
   cross-version ABI, no shared-library claim, and selected rather than
   universal 64-bit functional coverage.
4. Reconcile validator expectations and documentation examples with the
   actual generated files from both installation routes.

### Deliverables

- Public and install documentation matching observed package behavior.
- Claims table tied to the positive/negative and selected-lane evidence.

### Completion Criteria

- A downstream user can build and verify either supported width without
  guessing flags or linking against an opposite-width archive.
- Every documented command uses the tested package interface.

---

## Day 14: Full Quality And Contract Closure

**Title:** Full Quality And Contract Closure
**Theme:** Validate the complete width contract and retain closeout evidence.
**Time estimate:** 10 hours (219.6: 10 h)

### Tasks

1. Run `make format && make lint && make test` in a clean normal/default
   build; run source-list and formatting checks after any generated changes.
2. Run both-width Make and CMake builds, installed-consumer matrices,
   package-contract validator, CMake/CTest parity, and the selected 64-bit
   solver/IO/conversion tests. Run sanitizer gates where compatible with
   the selected width and record their limits.
3. Review hosted status for the required install and 64-bit lanes; retain
   exact URLs, commands, archive identities, failures, and rerun results.
4. Audit the final diff for generated-file drift, accidental default-width
   changes, stale objects, and overbroad ABI claims; summarize open limits
   in `WORKING_NOTES.md` and close only evidenced items.

### Deliverables

- Full C and package-quality record for 32-bit default and selected 64-bit
  paths, including negative mismatch evidence.
- Sprint closeout notes with exact supported scope and unresolved risks.

### Completion Criteria

- `make format`, `make lint`, `make test`, install validation, CMake/CTest
  parity, both-width consumers, and required hosted lanes all pass.
- Consumers obtain the archive's width or fail before unsafe use; no
  cross-version/shared ABI claim is made.
- Item 219.6's 20-hour contract-validation allocation and the 140-hour
  sprint plan are closed only with retained evidence.
