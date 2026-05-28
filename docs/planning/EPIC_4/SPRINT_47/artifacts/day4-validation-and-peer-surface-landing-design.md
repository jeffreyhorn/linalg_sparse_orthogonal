# Sprint 47 Day 4 Artifact: Validation and Peer-Surface Landing Design

## Purpose

Define the validation contract and mid-sprint landing order for Sprint 47
before the first benchmark/example/tooling code edits begin.

## Core Day 4 Conclusion

Sprint 47 should use the same layered validation model established in Sprint 40
and then specialize it for auxiliary-surface work:

1. mandatory full C/C-header gate for any `*.c` / `*.h` edits
2. stronger reviewed wrapper baseline for broader auxiliary batches
3. compile-only benchmark/example coverage through the maintained tooling
   targets
4. targeted direct benchmark/example CLI sanity checks only when the touched
   surface justifies them

The goal is not to run every benchmark or example on every change. The goal is
to preserve the strongest local reviewed baseline honestly while adding small,
surface-specific checks where Sprint 47's auxiliary claims depend on them.

## Validation Shape for Sprint 47 Implementation Days

### 1. Mandatory full code gate for all `*.c` / `*.h` changes

For any Day 5+ code batch that touches C or header files, the mandatory floor
remains:

```bash
make format
make lint
make test
```

Interpretation:

- this is the non-negotiable floor for Sprint 47 implementation work
- benchmark/example/tooling cleanup does not relax the core gate simply because
  it is "auxiliary"

### 2. Stronger reviewed baseline for broader auxiliary batches

For substantial multi-surface Sprint 47 batches, the stronger local reviewed
baseline should usually be:

```bash
make quality-review-full
```

This is especially appropriate for:

- shared parser-helper landing
- `bench_main` modernization
- reorder-mode parity cleanup if it touches multiple benchmark/build/docs
  surfaces
- broader script-side support-code cleanup

Interpretation:

- Sprint 47 should not claim auxiliary safety or usability improvements from
  only a narrow local compile if the touched surface spans maintained reviewed
  paths

### 3. Maintained compile-only tooling coverage

Sprint 47 inherits a useful compile-only auxiliary surface through:

```bash
make tooling-build
```

which expands to benchmark/example compile coverage via:

- `bench-build`
- `examples-build`

Interpretation:

- this is the strongest maintained compile-only auxiliary gate
- it is the right default follow-on whenever Sprint 47 touches benchmark/example
  code or build wiring

### 4. Direct targeted CLI and binary sanity checks

Sprint 47 should add direct reruns only when the touched surface depends on
them.

Expected targeted checks include:

- `./build/bench_main --help`
- `./build/bench_main --spmv-iters ...`
- `./build/bench_main --reorder ...`
- `./build/bench_eigs --help`
- direct example binary reruns if touched examples change behavior

Interpretation:

- these are evidence checks for Sprint 47's CLI/help/parity claims
- they are not universal mandatory gates for unrelated docs-only or script-only
  work

## First Peer Surfaces After `bench_main`

### 1. `bench_eigs.c`

Status:

- stronger parser/reference surface
- already uses checked `strtol` / `strtod`
- likely later alignment-only consumer if Sprint 47 finds a small shared-helper
  benefit there

Day 4 decision:

- not a first-wave rewrite target
- eligible only for bounded helper-alignment or docs/parity follow-on work

### 2. Repeated-run benchmark drivers

Targets:

- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`

Status:

- small, purpose-specific drivers
- not primary parser-drift hotspots

Day 4 decision:

- keep them out of the first implementation wave
- only touch them if a shared-helper or reporting cleanup falls out trivially

### 3. Small example argument/helper seams

Targets:

- `examples/example_eigs.c`
- `examples/example_iterative.c`
- `examples/example_matrix_free.c`

Status:

- bounded follow-on surfaces
- more safety/helper cleanup than major CLI redesign

Day 4 decision:

- audit and land only after the benchmark helper seam and `bench_main` shape are
  stable

## Explicit Out-of-Scope Boundaries

Sprint 47 should keep the following out of scope:

### 1. Broad benchmark framework redesign

Do not turn Sprint 47 into:

- a new benchmark framework abstraction layer
- a shared benchmark reporting engine
- a broad unification of every benchmark binary

### 2. Public CLI abstractions in the core library

Do not export parser helpers through:

- `include/`
- public library headers
- supported public API contracts

### 3. Large tutorial / README restructuring

Sprint 47 should update only the touched benchmark/example docs it needs for
truthfulness. It should not expand into:

- broad tutorial reorganization
- large README architecture rewriting
- unrelated public-doc cleanup

### 4. Dead-code workflow redesign

Sprint 47 may align touched support-code safety patterns, but it should not
redefine:

- `deadcode-report`
- `deadcode-check`
- the current serialized dead-code workflow contract

## Mid-Sprint Landing Order

The correct Sprint 47 order after Day 4 is now explicit:

1. Day 5:
   - shared parsing-helper implementation
2. Day 6:
   - `bench_main` parser modernization
3. Day 7:
   - post-`bench_main` audit
4. Day 8:
   - reorder-mode / emitted-label parity cleanup
5. Day 9:
   - example safety audit
6. Day 10:
   - bounded example cleanup
7. Day 11:
   - auxiliary tooling cleanup
8. Day 12:
   - benchmark/example docs refresh
9. Day 13:
   - full validation sweep
10. Day 14:
   - closeout and handoff

This order preserves the Day 2 and Day 3 conclusions:

- benchmark helper work first
- `bench_main` before peer surfaces
- examples/scripts after the parser and parity shape is stable

## Main Day 4 Conclusions

### 1. `quality-review-full` is the right stronger default for substantial Sprint 47 code batches

It preserves the strongest local reviewed baseline while keeping Sprint 47
honest about its broader auxiliary-surface claims.

### 2. `tooling-build` is the maintained compile-only auxiliary gate Sprint 47 should lean on

It gives benchmark/example compile coverage without overclaiming runtime
behavior.

### 3. Direct CLI/binary reruns should stay targeted

Sprint 47 should run them when the touched surface makes a specific usability or
parity claim, not as an always-on universal expansion of the code gate.

### 4. The sprint scope remains intentionally bounded

Sprint 47 is about modernizing and aligning the auxiliary surface, not about
rebuilding the benchmark framework or exporting a public CLI support library.
