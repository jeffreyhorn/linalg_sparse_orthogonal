# Sprint 49 Day 4 Artifact: Landing and Validation Design

## Purpose

Turn the Day 3 public lifecycle/workspace design into a concrete
implementation-order and validation contract before public header/source
changes begin.

## Core Day 4 Decision

Sprint 49 should treat public lifecycle exposure as a normal
implementation-heavy Epic 4 landing:

- code/header changes use the full required gate
- substantial public-API batches also use the reviewed local baseline
- examples, benchmarks, and specialized regression binaries are targeted
  follow-ons only when the touched surface justifies them

This preserves the Sprint 40 validation model while keeping Sprint 49 from
degenerating into "run everything always."

## Validation Contract

### Mandatory for any `*.c` / `*.h` landing

For any Day 5/6 Sprint 49 code or header change, the mandatory baseline is:

```bash
make format
make lint
make test
```

Interpretation:

- this remains the non-negotiable floor
- public API work does not get a weaker gate than internal refactors

### Stronger default for substantial public-API batches

When the change spans:

- public headers
- wrapper routing
- caller-visible lifecycle semantics
- multiple affected solver/eigensolver surfaces

Sprint 49 should also default to:

```bash
make quality-review-full
```

Interpretation:

- this is the right proof point for the main public lifecycle landing
- it preserves the strongest maintained local reviewed baseline while the
  public contract is changing

### Targeted follow-ons when the touched surface justifies them

#### Example-focused follow-ons

- `./build/example_iterative`
- `./build/example_matrix_free`
- `./build/example_eigs`

Use when the public lifecycle/workspace landing changes example truth or when
the migration-path docs depend on those examples staying accurate.

#### Repeated-run benchmark follow-ons

- `./build/bench_iterative_reuse`
- `./build/bench_eigs_reuse`

Use when the public lifecycle landing changes the repeated-run story or when
the migration-path guidance needs to stay grounded in the benchmarked repeated
reuse surfaces.

#### Compile-only tooling follow-on

- `make tooling-build`

Use when public API changes plausibly affect example/benchmark compilation even
before any direct runtime reruns are warranted.

#### Solver/eigensolver regression follow-ons

- `./build/test_iterative`
- `./build/test_block_solvers`
- `./build/test_minres`
- `./build/test_bicgstab`
- `./build/test_stagnation`
- `./build/test_eigs`
- `./build/test_eigs_thick_restart`
- `./build/test_eigs_lobpcg`

Use when the touched public lifecycle or wrapper path directly affects the
corresponding solver/eigensolver family.

## Landing Order

The implementation order should remain:

1. public header / API surface
2. implementation / wrapper integration
3. migration-path documentation
4. cross-surface compatibility sweep
5. final residual review
6. full validation and closeout

### Why this order is mandatory

The repo currently has:

- public reusable-lifecycle precedent in `sparse_analysis.h`
- one-shot public iterative/eigensolver usage in headers, README, and examples
- internal repeated-run helpers behind private seams

That means examples, README, and benchmark drivers are not the first design
surface. They are later agreement surfaces that should describe the landed API,
not invent it first.

## Out-of-Scope Boundary

Sprint 49 should not widen into:

- post-Epic-4 feature expansion
- large new benchmark framework work
- new solver families
- broad tutorial rewrite unrelated to the final lifecycle shape
- exposing raw internal workspace layout as public API
- replacing or deprecating the existing one-shot public entries in Sprint 49

These are deliberate scope fences, not signs of incomplete Sprint 49 planning.

## Highest-Value Day 4 Conclusions

### 1. The Sprint 49 implementation days now have a fixed validation floor

Day 5/6 code/header work must use:

- `make format`
- `make lint`
- `make test`

and should usually add:

- `make quality-review-full`

for the main public lifecycle landing.

### 2. Example/benchmark/test follow-ons are now explicitly scoped

The high-value later checks are named already:

- examples
- repeated-run benchmarks
- iterative/eigensolver family regressions
- `make tooling-build`

That gives Sprint 49 a clean targeted-follow-on set without pretending every
day needs every binary.

### 3. The implementation fence is explicit before code movement starts

Sprint 49 now has:

- a public lifecycle design
- a landing order
- a validation contract
- an out-of-scope boundary

That is the right precondition for bounded Day 5/6 public API landing work.
