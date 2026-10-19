# Sprint 215 Day 9: Opt-In Selected Pilot Harness

## Outcome And Boundary

Implemented an **opt-in, local, correctness-only attempt harness** for the
single Day 3 selection. `scripts/cholmod_pilot.c` reads the exact
`symmetric_4.mtx` matrix with the project loader, checks every expanded
coefficient and the fixed RHS, then exposes two separately invoked paths:
forced linked-list/no-reorder project Cholesky and CHOLMOD natural-order,
postorder-off, simplicial `CHOLMOD_A` solve on sorted lower CSC. It emits raw
phase/status fields and four full-precision solution components per side.
`scripts/run_cholmod_pilot.py` pins fixture/package hashes, compiles the
helper only when explicitly invoked, retains every command's stdout/stderr,
captures build/host/thread/package inputs, strictly parses each side, and
applies the Day 4 residual and oracle limits independently.

**No selected pilot execution was performed on Day 9.** A successful attempt
is labeled `two_sided_correctness_candidate` in `attempt.json`, never
`correctness_observed` or `pass` evidence. The v1 schema/semantic validator,
complete cell-ID serialization and runtime provenance review remain required
before an observed record can be accepted. The runner writes no
`evidence.json`, selected report-index row, comparison-helper row, benchmark
row or timing result. There is no optional dependency in `make test` or the
default library build.

## Invocation And Raw Bundle

This command is **opt-in** and would run the actual external solver. Day 10
must use it only after the Day 8 gates; it was not run on Day 9:

```sh
RUN_ID=local-correctness-001
python3 scripts/run_cholmod_pilot.py --fixture tests/data/symmetric_4.mtx --output "build/external-baseline/s215-cholmod-v7_12_2-symmetric4-spdsolve-v1/$RUN_ID" --correctness-only --execute
```

The output directory must be new and below the selected
`build/external-baseline/` path. `attempt.json` records the selection ID,
provisional status/reason, platform, explicit thread and loader variables,
matrix hash, candidate library hash, project Git commit/worktree status,
project archive hash, compiler version, pkg-config flags, helper binary hash,
each exact command/cwd/exit status and per-side metrics. Every command gets
`<label>.stdout.txt` and `<label>.stderr.txt` with SHA-256 references;
the raw solver outputs are retained before interpretation. The helper also
prints the loaded CHOLMOD library path via `dladdr`; the runner checks its
bytes against the pinned candidate digest. Captured `otool -L` output is
a discovery aid, **not** a complete transitive loaded-library or build-option
attestation. The Day 10 record must still resolve and hash those dependencies,
verify effective thread controls and freeze the complete cell ID.

The machine-output parser accepts only known one-line `key=value` fields,
rejects duplicates/unknowns, requires exact side, load and mode/status
fields, and requires a four-value finite solution. The Python numerical
check uses the untouched fixed matrix/RHS, `T=32*4*DBL_EPSILON`,
`K=17/7`, Day 4 residual/backward and oracle thresholds. One side failing
does not make the pair a pass. A missing package or changed matrix/library
hash produces `blocked`; malformed raw output, a failing solver status or a
failed numerical gate produces `failed`. All are **attempt states**, not v1
evidence statuses. Timing is always `not_run`.

## Focused Synthetic Fixtures

`python3 tests/test_run_cholmod_pilot.py` checks:

| Fixture | Expected result |
| --- | --- |
| Two complete synthetic side outputs with `x=[1,1,1,1]` | Parser accepts both; independent Day 4 limits pass. |
| CHOLMOD `factor_minor=3` | `solver_status_failure`, not a pass. |
| Duplicate or malformed output line | `malformed_output`. |
| NaN or wrong-length solution | Rejected before numerical classification. |
| Wrong side or changed expanded-nnz key | `identity_or_load_failure`; selected cell not reused. |
| Finite but wrong solution vector | Numerical gate fails. |
| Missing optional package probe | Provisional `blocked`, no solver side or timing result. |
| Changed fixture hash | Provisional `blocked` before any package probe. |

These are synthetic parser/status fixtures, not CHOLMOD observations. The
helper itself was syntax-checked and linked against the installed local
package, but it was **not executed**. The local build command was:

```sh
cc -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror -O2 -Iinclude -Ibuild/include -I/usr/local/Cellar/suite-sparse/7.12.2/include/suitesparse scripts/cholmod_pilot.c -Lbuild -lsparse_lu_ortho -L/usr/local/Cellar/suite-sparse/7.12.2/lib -lcholmod -lm -o build/cholmod_pilot_day9
```

## Day 10 Gates And Executable Fallback

Before promoting any provisional candidate, implement the Day 7 v1
schema/validator with positive blueprint, blocked and observed fixtures plus
the negative mutation matrix. Freeze the exact canonical cell-ID JSON and
test vector. Confirm loaded CHOLMOD and transitive dependency hashes,
build/module options, project binary/flags, ABI, effective ordering/backend,
thread behavior, matrix conversion and Day 4 diagnostics. The Day 8
publication gate remains separate. If full provenance or validation cannot
be supplied, preserve `attempt.json` and raw logs as a reproducible
`blocked`/unaccepted attempt; do not rename it `evidence.json` or present it
as a numerical comparison. The runnable harness remains available for a
later reviewed attempt without making CHOLMOD mandatory.

Day 9 changed only the two opt-in scripts, their focused synthetic test,
this artifact and working notes. No C library/header, selected manifest,
normalizer, workflow, fixture or public-claim surface changed.

## Validation Performed

On the local macOS x86-64 host, `python3 tests/test_run_cholmod_pilot.py`
passed all eight synthetic tests. Strict helper syntax checking and the
compile/link command above passed; `clang-format --dry-run --Werror
scripts/cholmod_pilot.c` passed. Because the helper is a `.c` file,
`make format`, `make lint`, and `make test` were run and all passed.
`make docs-check` passed with 18 checked-in public headers and matching
generated pages. None of these checks executed the CHOLMOD pilot helper.
