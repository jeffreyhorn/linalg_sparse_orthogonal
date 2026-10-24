# Day 13: Measurement Freeze And Instrumented Checks

**State:** Item 218.5's selected-kernel measurement is frozen. The focused
normal and instrumented matmul gates passed on 2026-10-23. This record does
not replace Day 14's repository-wide Make and CMake validation.

## Frozen Evidence And Claim Boundary

The [Day 11 protocol](day11-failure-closure-measurement-protocol.md) pins the
baseline commit, identical benchmark driver, fixture definitions, compiler,
flags, warmups, and alternating sample order. The [Day 12 profile](day12-matmul-profile.md)
contains the derived timing and whole-process memory tables. Its
[180 raw stdout/stderr/status files](day12-raw/), [raw-file hash manifest](day12-raw-sha256.txt),
and [60-row derived CSV](day12-samples.csv) are the measurement bundle. All
60 measured processes exited zero and validated every output value, shape,
and nonzero count. `shasum -a 256 -c` revalidated all 180 raw files on Day
13. The parser command and original sampling loop are in those two earlier
artifacts; neither samples nor summaries were edited on Day 13.

| Frozen object | SHA-256 |
| --- | --- |
| `day12-raw-sha256.txt` | `16f1e98114d32dc9c56d23893364ca1050f46b726d973e8f8f1f8a7def279108` |
| `day12-samples.csv` | `c3b30ae97bf8bc8babd847a239c07a53abefd319228e31cd23dffb81f31024fe` |
| `day12-matmul-profile.md` | `c82b584d8e7bb0305f70e9b774154d5190645da6ba44944bbf6afe4d1aa7e8b9` |
| `benchmarks/bench_matmul.c` at `806b31ed` (same driver for both) | `0995357c0f272bfb94ac6150d48f434307e9ec4f77b6368ad1794a8c3b93b3cb` |
| Candidate `src/sparse_matrix.c` | `6da7faf6298260d6708401ce17af351e22824fdd4774c7660fe2ad9530f2e1dc` |
| Baseline benchmark binary | `fb8df0a351faa52991ebbae908d7f8176ad893385e9f084f460ce19aa6a4ad33` |
| Candidate benchmark binary | `d65c68c52d7482ae3c85a0d06be784a47900a6fd91640ae34e40f8287c4a024c` |

The 2026-10-24 POSIX feature-macro addition changed the current benchmark
source hash to
`61d5bf2cd0896dc4b9f6b5507195781f5a72be20f4891370e3d683dc3806aa61`.
The table describes the frozen binaries and samples, not a build from the
latest source; see the [Day 11 provenance note](day11-failure-closure-measurement-protocol.md).

The baseline detached worktree remains clean at
`271bd3b704326296332cb397385ed458436ca0b7`. Both comparison builds
use `/usr/bin/clang`, `-O2`, C11, no sanitizer, and OpenMP off. The frozen
claim is **only** that, on this host and these ten paired samples per shape,
the candidate's median `sparse_matmul` call was faster on `wide-row` and
`rectangular`, with nonoverlapping observed ranges. The `banded` ranges
overlapped and three candidate pairs were slower, so no robust narrow-row
improvement is claimed. Whole-process RSS/footprint cannot establish a
matmul-only memory change; no memory reduction is claimed. The result is
not a cross-platform, all-shapes, or asymptotic wall-clock claim.

## Focused Quality Evidence

Host: macOS 15.7.9, Darwin 24.6.0, x86_64. Compiler: `/usr/bin/clang`,
Apple Clang 17.0.0 (`clang-1700.6.4.2`). The normal Make build used the
repository's C11 warning flags and `-O2`. The CMake sanitizer builds used
`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` and the
repository `SANITIZE=all` mode. Both were new build directories. No default
`make clean` was run, because it would remove the Day 11 baseline worktree
and comparison binaries under `build/`.

| Exact command | Result | Scope |
| --- | --- | --- |
| `make BUILDDIR=build/s218-day13-normal matmul-allocation-failure-gate` | PASS, exit 0; registration passed, 41 tests, 0 failed/skipped, 372262 assertions | Fresh normal build; default 4096-node slabs; includes focused matmul and registered fault gate. |
| `cmake -S . -B build/s218-day13-sanitize -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_C_FLAGS='-O1 -g -DSPARSE_NODES_PER_SLAB=1' -DSANITIZE=all -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` | PASS, exit 0 | One-node-slab ASan/UBSan configure. |
| `cmake --build build/s218-day13-sanitize --target test_matmul --parallel 4` | PASS, exit 0 | Instrumented library and focused test build. |
| `env MallocNanoZone=0 ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --test-dir build/s218-day13-sanitize -R '^test_matmul$' --output-on-failure` | PASS, exit 0; 1/1 CTest passed | One-node-slab fault sweep; direct run reported 41 tests and 30800 assertions. |
| `set -o pipefail; env MallocNanoZone=0 ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 build/s218-day13-sanitize/test_matmul \| tail -n 15` | PASS, exit 0; 41 tests, 0 failed/skipped, 30800 assertions | Confirms later-node positions 1 through 8 under instrumentation. |
| `cmake -S . -B build/s218-day13-sanitize-default -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_C_FLAGS='-O1 -g' -DSANITIZE=all -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` | PASS, exit 0 | Default-slab ASan/UBSan configure. |
| `cmake --build build/s218-day13-sanitize-default --target test_matmul --parallel 4` | PASS, exit 0 | Instrumented default-slab build. |
| `env MallocNanoZone=0 ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --test-dir build/s218-day13-sanitize-default -R '^test_matmul$' --output-on-failure` | PASS, exit 0; 1/1 CTest passed | Default-slab fault sweep, including later slabs. |
| `leaks --atExit -- build/s218-day13-normal/test_matmul` | PASS, exit 0; 41 tests passed; 0 leaks, 0 leaked bytes | Native macOS leak check on default-slab focused test. |
| `make format-check`; `make source-list-check`; `git diff --check` | PASS, exits 0; 49 library sources | Formatting, source manifest, whitespace. |
| `shasum -a 256 -c docs/planning/EPIC_20/SPRINT_218/artifacts/day12-raw-sha256.txt` | PASS, exit 0; 180/180 files | Raw measurement integrity. |

The one-node-slab direct test used the same sanitizer environment and
reported failures injected at nodes 1 through 8 followed by successful
retries. macOS ASan leak detection was disabled (`detect_leaks=0`); the
separate native `leaks` command is the leak observation, not a claim that
ASan checked leaks. The focused sanitizer tests do not cover the full
repository test matrix. Existing `-Wdouble-promotion` warnings in other
sources appeared during the CMake builds but did not fail them. No sanitizer
was unavailable or counted as passing without execution.

## Scope Review And Remaining Risks

`git diff --name-only` showed changes to `src/sparse_matrix.c`,
`tests/test_matmul.c`, its registration checker, `Makefile`, and
`CMakeLists.txt`; the new benchmark and measurement script/artifacts are
untracked until sprint closeout. No public header changed, no library source
was added or removed, and `make source-list-check` still counted 49.
The matrix implementation diff changes matmul output append and its
private shell/slab fault-injection plumbing; the public `sparse_create`,
`pool_alloc`, and `sparse_matrix_make_node` signatures and default allocation
sequence remain unchanged. No unrelated storage-format conversion was
introduced.

Remaining limits: the fixture set is selected, not representative of all
sparsity patterns; the banded timing difference is within observed overlap;
process memory peaks do not isolate kernel allocation; injected failures
cover explicit matmul allocation sites, not allocations internal to system
libraries; and full Make/CMake suites remain Day 14 work. Promotion beyond
these claims stays with the Day 14 quality gate owner.
