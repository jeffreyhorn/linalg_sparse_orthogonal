# Sprint 202 Day 2: Platform And Row Candidate Ranking

## Summary

Day 2 ranked candidate hosted platform and benchmark row pairs for Sprint 202.
The preferred shortlist is macOS hosted selected benchmark freshness for the
existing selected `SRT-BENCH-REFACTOR-CSC-NOS4` row. The actual selection is
left for Day 3, as planned.

## Ranking Criteria

Each candidate was scored qualitatively against:

- evidence value for adding one non-Linux hosted freshness lane;
- hosted runtime and build-system fit;
- artifact path and upload diagnosability;
- reuse of Sprint 192 threshold-free methodology;
- public claim-safety cost.

## Platform Ranking

| Rank | Platform | Result |
| ---: | --- | --- |
| 1 | macOS hosted runner | Preferred candidate because the repo already has macOS hosted CI, selected comparison freshness precedent, POSIX-style paths, and Make-based quality/tooling lanes. |
| 2 | Windows hosted runner | Valuable but riskier because benchmark generation would need Windows-specific executable paths, shell handling, CMake/MSVC config handling, and public non-claim promotion. |
| 3 | Linux second row | Deferred because it would not add a hosted platform. |
| 4 | Local sentinel promotion | Deferred because it would move toward timing evidence rather than hosted selected freshness. |

## Benchmark Row Ranking

| Rank | Row | Result |
| ---: | --- | --- |
| 1 | `SRT-BENCH-REFACTOR-CSC-NOS4` / `bench_refactor_csc` / `nos4.mtx --repeat 1` | Preferred because it already has selected manifest identity, Linux hosted precedent, checker coverage, docs wording, and bounded runtime. |
| 2 | `bench_chol_csc` / `nos4.mtx --repeat 1` | Deferred because selecting it would broaden the canonical publication surface. |
| 3 | `bench_iterative_reuse` | Deferred because it is currently local-only canonical context and implies broader iterative performance coverage. |
| 4 | `bench_eigs_reuse` | Deferred because it is currently local-only canonical context and has higher interpretation risk. |
| 5 | `bench_refactor_csc --indefinite-kkt` | Deferred because it introduces a separate workload and selected-row contract. |

## Preferred Shortlist

| Rank | Pair | Disposition |
| ---: | --- | --- |
| 1 | macOS hosted runner + `SRT-BENCH-REFACTOR-CSC-NOS4` | Carry forward to Day 3 as the primary selection candidate. |
| 2 | Windows hosted runner + `SRT-BENCH-REFACTOR-CSC-NOS4` | Keep as backup only if macOS is unsuitable. |

## Deferred Candidates

| Candidate | Reason |
| --- | --- |
| Windows selected benchmark freshness as first lane | Higher shell, path, CMake config, executable-location, and public-claim coordination risk. |
| New selected benchmark row | Unnecessary to close the additional-platform freshness gap and likely to imply broader benchmark publication. |
| Broad canonical benchmark upload | Would weaken the selected-artifact boundary. |
| Timing thresholds or regression baselines | Out of scope for the threshold-free freshness sprint. |
| Portable Linux-vs-macOS performance comparison | Out of scope because hosted freshness does not prove comparable timing. |

## Day 3 Handoff

Day 3 should select exactly one pair, freeze the workflow job and artifact
names, and record the final in-scope/out-of-scope map before implementation.
The expected Day 3 primary candidate is:

- platform: macOS hosted runner;
- selected row: `SRT-BENCH-REFACTOR-CSC-NOS4`;
- artifact: `build/bench-reports/canonical/bench_refactor_csc.csv`;
- checker: `scripts/check_bench_canonical_freshness.py`;
- claim boundary: hosted selected, threshold-free, no portable performance.

## Non-Claims

Day 2 does not claim macOS selected benchmark freshness, Windows selected
benchmark freshness, platform parity, portable performance, timing thresholds,
release benchmark evidence, benchmark superiority, package-manager support, ABI
support, OpenMP speedup, backend superiority, or state-of-the-art status.
