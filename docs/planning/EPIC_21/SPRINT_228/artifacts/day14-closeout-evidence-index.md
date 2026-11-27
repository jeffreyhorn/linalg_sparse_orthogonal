# Sprint 228 Day 14: Closeout Evidence Index

## Source identity and disposition

- Branch: `sprint-228`.
- Base commit: `363600ea27e854a4334cff5fa06ace6529045ccd`.
- Final Day 14 validation was run against the complete uncommitted sprint diff
  rooted at that commit. The eventual review commit must record the resulting
  commit SHA; the base SHA alone does not identify the dirty development tree.
- Outcome: the default sparse low-rank reconstruction route is accepted for
  its narrow reconstruction-memory and behavior contract.
- Blocked claim: the Day 11 packet has `timing_pass=false` and
  `protocol_accepted=false`; it supports no speed or promotion-level timing
  claim.

## Deliverable map

| Sprint result | Primary evidence | Final gate |
| --- | --- | --- |
| Frozen route, output, status, memory, and timing contract | `day2-reconstruction-contract-and-measurement-protocol.md`, `day2-reconstruction-fixture-matrix.tsv` | Contract remains unchanged through closeout |
| Independent square/tall/wide and degenerate numerical evidence | `day3-core-differential-harness.md`, `day4-degenerate-and-numerical-differential-corpus.md`, `day5-differential-acceptance-and-promotion-design.md` | Complete `test_svd` execution passes |
| Hardened per-cell reconstruction and default promotion | `day6-outer-route-hardening.md`, `day7-default-promotion-and-selector.md` | Unset and exact `on` select per-cell reconstruction; exact `off` is compatibility-only |
| Allocation ownership and failure recovery | `day8-default-route-and-initial-allocation-matrix.md`, `day9-allocation-and-numeric-failure-evidence.md`, `day10-lifecycle-and-contract-closure-matrix.md` | Exhaustive 26/37/23/27 allocation sweeps and mixed-call recovery pass |
| Component-level memory evidence | `day11-reconstruction-memory-packet.md`, `day11-memory-component-accounting.tsv` | All 290 retained raw-file hashes verify |
| Documentation and claim limits | `day12-documentation-and-timing-limitations.md` | Public header, README, benchmark guide, maintainer guide, and generated docs pass |
| Width, sanitizer, leak, and registration validation | `day13-focused-cross-build-and-sanitizer-validation.md` | Both widths and maintained sanitizer/leak lanes pass; 63 CTests reconcile to 62 Make tests plus one selector |
| Complete final-tree closure | `day14-final-quality-and-handoff.md` | `make format`, `make lint`, `make test`, and full reviewed CMake parity pass |

## Final command record

| Command | Result |
| --- | --- |
| `make format` | Pass |
| `make lint` | Pass: strict compile, clang-tidy, and cppcheck over 112 files |
| `make test` | Pass: all 62 Make test binaries |
| `make QUALITY_REVIEW_CMAKE_DIR=build/s228-day14-cmake quality-review-cmake` | Pass: clean build, 63/62+1 registration parity, and 63/63 CTests |
| `make format-check` | Pass |
| `make source-list-check` | Pass: 49 library sources |
| `make docs-check` | Pass: all 18 checked-in public headers covered |
| `python3 -m py_compile scripts/run_sprint228_memory_packet.py` | Pass |
| `shasum -a 256 -c manifest.sha256` in `build/sprint228-day11-memory` | Pass: all 290 entries |
| `git diff --check` | Pass |

## Completion-criterion mapping

1. **No additional default `m*n` reconstruction accumulator:** source dispatch,
   exact counters, both-width probe runs, and all 40 measured default samples
   agree on zero reconstruction-scratch bytes and allocations.
2. **Accepted output/error behavior:** independent oracles, threshold-boundary
   fixtures, selector precedence, numerical failures, exhaustive allocation
   sweeps, and mixed-call retries pass in the complete suites.
3. **Remaining dense storage is explicit:** economy `sigma`, `U`, and `Vt`,
   bidiagonalization/decomposition work, shell/maps, and retained node slabs
   remain documented and separately counted.
4. **Build and test coherence:** Make and CMake registration reconcile exactly;
   32-bit and 64-bit focused lanes and maintained sanitizers passed on the
   final C/header state.
5. **No broader claim:** no whole-call peak-memory, universally sparse output,
   speed, portable-performance, external-ranking, or release claim is
   accepted.
