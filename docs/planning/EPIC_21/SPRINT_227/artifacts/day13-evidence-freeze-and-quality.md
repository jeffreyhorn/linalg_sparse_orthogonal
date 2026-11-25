# Day 13 Evidence Freeze and Quality Closure

## Outcome

The Sprint 227 contract, ownership, complexity, and selected local timing statements are now
reconciled in the [claim ledger](day13-claim-ledger.md). Public and internal documentation agree
with the implementation. All Day 13 format, lint, focused correctness, CMake parity, and focused
sanitizer checks pass. Day 14's complete repository gates remain pending.

## Evidence chain

The selected timing statement is traceable through the following immutable Day 12 layers:

1. The [profile](day12-add-profile.md) names the exact source, build, host, workload, timing,
   memory, and claim boundaries.
2. The [sample table](day12-samples.csv) contains all 60 validated measured processes.
3. The [raw bundle](day12-raw/) retains stdout, stderr, timing, status, and invocation metadata
   for every process.
4. The [raw manifest](day12-raw-sha256.txt) verifies all 300 raw files.
5. The historical [collection-time source manifest](day12-source-sha256.txt) records the exact
   source text used for measurement.

The Day 12 profile SHA-256 is
`e82579a7caa2e7f0ce996c0ebec7389e802706ab6a396ac309108b3502b3c1a4`; the sample-table hash is
`dd36939fbdb2b7cfdaf9868819711c990564e67789a90eb0acd8ed1e3d8d6afb`; and the raw-manifest hash
is `0cf43a479b00e187f94d816845fdc7b7e6879c7ff95c64dfb6d3c5f8881b4cb1`.

Day 13 added explanatory comments to `include/sparse_matrix.h` and `src/sparse_matrix.c` and
updated `README.md`. The Day 12 source manifest is therefore intentionally historical and must
not be expected to verify the later text. The current relevant source hashes are frozen in
[the Day 13 source manifest](day13-source-sha256.txt).

## Post-collection binary equivalence

The candidate was rebuilt in a clean tree with the exact Day 12 compiler, generated 32-bit index
header, SDK, and library/driver flags. Although Apple `ar` member timestamps changed the archive
container hash, the affected object and linked benchmark executable did not change:

| Item | Day 12 | Day 13 rebuild | Result |
| --- | --- | --- | --- |
| `sparse_matrix.o` SHA-256 | `6a96c2d82f204c294e6e4aaa11b9f8187e7707b8c67048620cd6b6f451cd3db2` | same | Byte-identical |
| `bench_add` candidate executable SHA-256 | `516700b283d24997070eb2c2633b789fa8b37607ff9fd92d3edea0108ae090db` | same | Byte-identical; `cmp` exit 0 |
| Static archive SHA-256 | `3e42e54229bb5cdba7aaddcade7e8b06f625877dbe32d4ca45ce7be34eb328d3` | `86f30323363eee9ecf75c60c6ed963945cb33de5e800aeca525b93c9b39bca8c` | Container differs only after member timestamp refresh; not used as equivalence proof |

This establishes that the documentation-only Day 13 source edits do not alter the measured
candidate executable. It does not broaden the measurement.

## Day 13 quality results

| Check | Result |
| --- | --- |
| `make format` | Passed. |
| `make lint` | Passed across 112 cppcheck translation units; no new finding. |
| Focused Make suites | `test_sparse_arith`: 49/49 tests, 79,666 assertions; `test_matmul`: 41/41 tests, 372,263 assertions; `test_sprint4_integration`: 5/5 tests, 49 assertions. Total: 95 tests and 451,978 assertions. |
| Reviewed CMake compile/parity path | Passed clean configure/build and `ctest -N`; 63 CMake tests equal 62 Make tests plus one focused CTest-only selector. |
| Fresh ASan/UBSan focused path | 3/3 CTests passed for arithmetic, matmul, and Sprint 4 integration; no sanitizer finding. |
| Documentation | `make docs-check` passed for all 18 checked-in public headers. |
| Evidence integrity | The strict summarizer revalidated all 60 samples from 300 raw files; all raw and current-source SHA-256 checks passed. |
| Patch hygiene | `git diff --check` passed. |

Existing nonfatal compiler warnings outside the changed arithmetic path remain visible in the
full CMake build and are not recast as Day 13 findings.

## Remaining limits and Day 14 handoff

- Independent timing replication, thermal/frequency isolation, and allocator instrumentation
  were not added.
- Day 13 does not rerun or replace any Day 12 sample.
- The selected timing claim remains local and workload-specific; the memory result remains
  inconclusive.
- Day 14 must run the complete required quality chain and close or explicitly block the sprint.
