# Day 14 Sprint 227 Closeout Evidence Index

## Decision

Sprint 227 is **complete** on the final working tree. Every sprint completion criterion and every
Day 14 local gate passed. The selected timing observation remains bounded to its recorded local
workloads and is not a release, portable-performance, memory-improvement, superiority, or
state-of-the-art claim.

## Epic item traceability

| Item | Closure evidence | Result |
| --- | --- | --- |
| 227.1 Contract Freeze | [Day 1 baseline](day1-arithmetic-contract-baseline.md), [Day 1 oracle fixtures](day1-dense-oracle-fixtures.tsv), [Day 2 frozen contract](day2-arithmetic-contract.md), [Day 2 fixture matrix](day2-contract-fixture-matrix.tsv) | Physical/original-state requirement, pointer/shape/state precedence, threshold, ownership, alias, and failure behavior are explicit. |
| 227.2 Permutation Guard | [Day 3 fixtures](day3-permutation-fixtures.md), [Day 3 corpus](day3-permutation-regression-corpus.tsv), [Day 4 add enforcement](day4-add-permutation-enforcement.md), [Day 5 matmul enforcement](day5-matmul-enforcement-and-builder-design.md) | All three APIs use one shared original-state guard; unsupported states cannot return misleading success. |
| 227.3 Sorted Output Builder | [Day 6 implementation](day6-sorted-add-builder.md), [Day 7 invariants](day7-builder-invariants-and-edge-corpus.md), [Day 8 allocation sweep](day8-add-allocation-failure-sweep.md) | Out-of-place add performs a sorted merge and direct row/column-tail append; retained nodes are published once without repeated head scans. |
| 227.4 Failure And Alias Matrix | [Day 9 out-of-place closure](day9-add-failure-alias-and-randomized-closure.md), [Day 10 in-place closure](day10-inplace-atomicity-and-alias-closure.md), [Day 11 integrated matrix](day11-integrated-failure-matrix-and-scaling-protocol.md) | Dense-oracle, structural, rectangular, cancellation, alias, stale-output, partial-failure, rollback/retry, and allocation-boundary cases pass. |
| 227.5 Scaling Evidence | [Day 11 protocol](day11-integrated-failure-matrix-and-scaling-protocol.md), [Day 12 profile](day12-add-profile.md), [samples](day12-samples.csv), [raw hashes](day12-raw-sha256.txt), [collection source hashes](day12-source-sha256.txt) | Exactly 60 validated processes and 300 hashed raw files support only the selected same-host kernel-time statement; overlapping memory ranges support no memory claim. |
| 227.6 Quality Closure | [Day 13 claim ledger](day13-claim-ledger.md), [Day 13 quality packet](day13-evidence-freeze-and-quality.md), [current source hashes](day13-source-sha256.txt), [Day 14 final packet](day14-final-quality-and-handoff.md) | Format, lint, full Make/CMake tests, sanitizer, package, width, consumer, documentation, and evidence-integrity gates pass. |

## Final gate matrix

| Gate | Final result |
| --- | --- |
| Format | `make format` followed by `make format-check`: pass; no formatter delta remains. |
| Lint | `make lint`: strict `-Werror`, clang-tidy, and cppcheck pass across 49 library sources and 112 cppcheck translation units. |
| Full Make tests | `make test`: all 62 registered test binaries pass; only intentional framework/conditional skips are reported. |
| Full CMake parity/tests | `quality-review-cmake`: clean rebuild, 63 registrations, exact 62 Make + 1 focused-selector parity, and 63/63 CTests pass. |
| Focused final regression | Arithmetic 49/49 and 79,666 assertions; matmul 41/41 and 372,263 assertions; Sprint 4 integration 5/5 and 49 assertions. Total 95/95 and 451,978 assertions. |
| Sanitizers | Fresh AppleClang ASan/UBSan build; 3/3 affected CTests pass with no finding. |
| Make width contract | 28/28 checks pass, including both widths, stamp revalidation, and stale/opposite-width rejection. |
| Make install/pkg-config | 26/26 checks pass, including maintained consumers and missing-archive rejection. |
| CMake install/find-package | 38/38 checks pass, including package relocation, versioning, consumer execution, and missing-archive rejection. |
| CMake width functional/install | 64-bit functional CTest passes; 20/20 installed-width cases pass across both widths. |
| pkg-config width/install | 40/40 positive and negative link/width cases pass across Make/CMake and both widths. |
| Installed width swap checks | 12/12 checks pass across CMake/pkg-config consumers and swapped archives. |
| Static package mutations | 26/26 Python contract tests pass. |
| Documentation/source list | `make docs-check` covers all 18 public headers; `make source-list-check` confirms all 49 library sources. |
| Sprint documentation links | All relative Markdown targets under `SPRINT_227` resolve. |
| Evidence integrity | Strict parser validates 60 samples from 300 raw files; raw and current-source SHA-256 manifests pass; collector/summarizer compile. |

## Evidence identity

- Day 12 raw-manifest SHA-256:
  `0cf43a479b00e187f94d816845fdc7b7e6879c7ff95c64dfb6d3c5f8881b4cb1`.
- Day 12 sample-table SHA-256:
  `dd36939fbdb2b7cfdaf9868819711c990564e67789a90eb0acd8ed1e3d8d6afb`.
- Day 13 current-source manifest verifies the final C/header/test/benchmark/scripts tree. Day 14
  adds closeout Markdown only and does not rewrite the historical Day 12 source receipt.

## Residual risks retained

- Arithmetic on nonidentity or factored matrices is rejected, not evaluated as a logical view.
- In-place addition remains intentionally non-atomic after validation when a new-node allocation
  fails; the valid scaled/prefix state is documented and tested.
- Selected timing comes from one short local macOS collection without independent replication,
  frequency/thermal isolation, allocator instrumentation, or confidence intervals.
- Whole-process memory measurements overlap and authorize no reduced-memory statement.
- Hosted Linux, Windows, and macOS CI will provide independent platform confirmation after the
  branch is committed and proposed; local completion does not predict those results.
- Existing nonfatal double-promotion warnings outside the changed arithmetic path remain visible
  in ordinary CMake builds and are not Sprint 227 regressions.
