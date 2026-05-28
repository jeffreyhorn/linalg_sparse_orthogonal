# Sprint 47 Day 2 Artifact: CLI and Auxiliary Surface Inventory

## Purpose

Refresh the live benchmark, example, and auxiliary tooling seam inventory so
Sprint 47's implementation order is grounded in the current code rather than
only in the project-plan labels.

## Inventory Summary

The Sprint 47 auxiliary surface does not reduce to one flat "CLI cleanup"
problem. The live repo now breaks into five distinct seam classes:

1. legacy benchmark CLI parsing and error-reporting drift
2. reorder-mode / emitted-label parity drift
3. modern benchmark CLI comparison/reference surfaces
4. bounded example safety/helper follow-ons
5. script-side support-code and workflow alignment

The important narrowing is that only the first two classes are strong Day 3 /
Day 5 / Day 6 direct implementation targets. The others are real Sprint 47
work, but they should follow after the shared parser/helper seam and the main
`bench_main` landing.

## Live Surface Classification

### 1. Primary modernization hotspot: `benchmarks/bench_main.c`

`bench_main.c` remains the clearest Sprint 47 direct code target because it
still owns the older parsing and reporting style:

- `benchmarks/bench_main.c` = `774` lines
- still uses:
  - `atoi(...)` for `--spmv-iters`
  - `atoi(...)` for `--size`
  - `atoi(...)` for `--repeat`
- still keeps inline mode parsing in the main argument loop for:
  - pivot mode
  - reorder mode
- still owns the largest live usage/help drift risk because the benchmark's
  top-of-file usage text and supported modes are tightly coupled to its ad hoc
  parser logic

This makes `bench_main.c` the strongest shared-helper adoption target for:

- positive integer parsing
- bounded integer parsing
- enum-like mode parsing
- clearer malformed-input reporting

### 2. Real secondary benchmark seam: reorder-mode / label parity

`bench_main.c` is not only a numeric parser problem. It also carries a genuine
mode-parity seam:

- live benchmark parser/usage currently names:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- broader live library reorder surface includes:
  - `SPARSE_REORDER_COLAMD`
- emitted labels and benchmark-mode coverage are still directly tied to the
  same inline parser logic

That means Sprint 47 needs a dedicated reorder-mode parity pass after parser
stabilization instead of trying to fold all parity cleanup into the initial
shared-helper batch.

### 3. Strong comparison/reference surface: `benchmarks/bench_eigs.c`

`bench_eigs.c` is large, but it is not the first modernization target:

- `benchmarks/bench_eigs.c` = `958` lines
- already uses:
  - checked `strtol(...)`
  - checked `strtod(...)`
  - explicit helper functions
  - explicit usage/help text
  - explicit enum-like parse paths for backend / preconditioner / mode

This makes `bench_eigs.c` more valuable as:

- a reference for parser shape and error handling
- a peer surface for smaller parity or helper-alignment follow-ons

rather than as the main Day 5 modernization landing.

### 4. Lower-priority repeated-run benchmark surfaces

The repeated-run benchmark binaries are already comparatively compact and
purpose-specific:

- `benchmarks/bench_iterative_reuse.c` = `251`
- `benchmarks/bench_eigs_reuse.c` = `201`

These are real auxiliary surfaces, but Day 2 evidence does not justify
treating them as primary parser-redesign hotspots. They fit better into:

- later shared-helper adoption if a touched seam becomes obvious
- Day 11 support-code alignment if the cleanup remains bounded

### 5. Bounded example follow-on surfaces

The live example targets are smaller and currently show less direct CLI drift:

- `examples/example_eigs.c` = `284`
- `examples/example_iterative.c` = `144`
- `examples/example_matrix_free.c` = `122`

The examples are therefore better treated as:

- safety/helper follow-on targets
- bounded cleanup surfaces after the benchmark helper/parser shape is stable

not as the first place to invent the new shared parsing conventions.

### 6. Script-side tooling seam

The main script-side auxiliary surfaces are explicit:

- `scripts/deadcode_report.py` = `523`
- `scripts/deadcode_workflow.sh` = `189`

These are large enough to matter, but the Day 2 evidence suggests a bounded
support-code alignment problem rather than a framework replacement project:

- `deadcode_report.py` already uses `argparse`, so it is not the same kind of
  legacy parser target as `bench_main.c`
- `deadcode_workflow.sh` is more about support-code and safety alignment than
  shared numeric parser extraction

That means the script-side surface belongs in the later auxiliary tooling batch
rather than in the first parser-helper implementation.

## Issue Buckets

### Bucket A: Numeric parsing and malformed-input reporting drift

Strongest target:

- `benchmarks/bench_main.c`

Characteristics:

- unchecked `atoi(...)`
- narrow inline argument validation
- benchmark-mode-specific usage/error coupling

### Bucket B: Reorder-mode / emitted-label parity drift

Strongest target:

- `benchmarks/bench_main.c`

Characteristics:

- supported-mode naming drift
- benchmark/report label alignment risk
- benchmark/library reorder-surface mismatch

### Bucket C: Shared-helper reference / peer adoption surfaces

Strongest current comparison point:

- `benchmarks/bench_eigs.c`

Characteristics:

- already has the stronger checked parsing style
- useful for design calibration
- smaller later alignment target if bounded work remains

### Bucket D: Example safety/helper follow-ons

Strongest likely later targets:

- `examples/example_eigs.c`
- `examples/example_iterative.c`
- `examples/example_matrix_free.c`

Characteristics:

- smaller surface
- more safety/helper alignment than full CLI modernization
- better sequenced after the benchmark helper seam exists

### Bucket E: Script-side support-code alignment

Targets:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

Characteristics:

- support-code consistency and safety alignment
- not the same parser problem as `bench_main.c`
- better sequenced after the benchmark/parser batches

## Shared vs Local Helper Targets

### Strongest shared-helper targets

The live code supports a small shared helper seam for:

- positive integer parsing
- bounded integer parsing
- finite double parsing
- enum-like mode parsing

These are strongest because they map directly onto the current `bench_main`
weaknesses and align with patterns already visible in `bench_eigs.c`.

### Better kept local or deferred

The following should stay local or later-batch only:

- benchmark-specific usage text composition
- `bench_eigs.c`'s backend/preconditioner-specific semantics
- script-side `argparse` / shell workflow structure
- example print/report style

That keeps Sprint 47 from turning a small parser-helper improvement into a
general-purpose auxiliary framework.

## First Implementation Order

The correct order after Day 2 is now explicit:

1. shared parsing-helper design
2. shared parsing-helper implementation
3. `bench_main` parser modernization
4. reorder-mode / emitted-label parity cleanup
5. bounded example safety audit and follow-on cleanup
6. script-side support-code alignment
7. benchmark/example docs refresh

## Main Day 2 Conclusions

### 1. `bench_main` is the real first landing zone

Sprint 47 should not start by spreading small edits across every benchmark or
example. The live code says the first real target is `bench_main.c`.

### 2. `bench_eigs.c` is more valuable as a parser-shape reference than as an early rewrite target

Its checked parsing and explicit usage paths already model much of the style
Sprint 47 should move toward.

### 3. The script-side surfaces are real, but not first-wave parser work

They belong to later bounded tooling alignment after the benchmark helper seam
is stable.

### 4. The example batch should stay subordinate to the benchmark helper seam

Examples are follow-on cleanup surfaces, not the place to define the new parser
contract first.
