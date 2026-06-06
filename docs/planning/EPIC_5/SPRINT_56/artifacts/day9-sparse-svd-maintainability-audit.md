# Sprint 56 Day 9 - `sparse_svd.c` maintainability audit

Date: 2026-06-05
Branch: `sprint-56`

## Scope

Reduce `src/sparse_svd.c` to a bounded Sprint 56 maintainability target
instead of leaving the SVD queue as a generic "clean up the large file"
bucket.

## Live hotspot state

Current line counts:

- `src/sparse_svd.c` = `1728`
- `src/sparse_svd_internal.h` = `21`
- `tests/test_svd.c` = `3746`
- `benchmarks/bench_svd.c` = `180`
- `examples/example_svd_lowrank.c` = `120`

Interpretation:

- `src/sparse_svd.c` is still a real production hotspot even after Sprint 55
  reduced the larger iterative/eigensolver files
- the proof surface is broad, but it is also segmented enough to support a
  bounded extraction if the ownership seam is chosen carefully

## Ownership bands in the live file

The current file reduces cleanly to five ownership bands.

### 1. Low-rank sparse reconstruction toggle and outer-product path

Main functions:

- `parse_svd_lowrank_outer(...)`
- `sparse_svd_lowrank_outer_product(...)`

Role:

- environment-gated sparse low-rank reconstruction path
- memory-shape choice for `sparse_svd_lowrank_sparse(...)`

Assessment:

- cohesive, but smaller than the main residual ownership seams
- better treated as an application-side cluster than the first extraction
  target

### 2. Bidiagonal reflector extraction and implicit QR core

Main functions:

- `hh_apply(...)`
- `sparse_svd_extract_uv(...)`
- `bidiag_svd_step(...)`
- `bidiag_svd_iterate(...)`

Role:

- full SVD bidiagonal back half
- shared QR-on-bidiagonal machinery also called by the benchmark harness

Assessment:

- cohesive algorithmically
- but still tightly central to the main full-SVD path
- extracting this first would force the broadest internal-header expansion
  and the most coupling to the existing `sparse_svd_internal.h` test surface

### 3. Full-SVD orchestration and full-mode basis padding

Main functions:

- `pad_orthonormal_basis(...)`
- `sparse_svd_compute(...)`

Role:

- public full-SVD front door
- economy/full-mode policy
- bidiagonalization plus QR orchestration
- full-mode basis completion and output packaging

Assessment:

- should stay in the main file this sprint
- it is the highest public-orchestration coupling band, not the cleanest first
  owned slice

### 4. Partial-SVD Lanczos bidiagonalization backend

Main functions:

- `sparse_svd_partial(...)`

Contained sub-ownership inside that function:

- Lanczos-subspace sizing policy
- `A^T` build and reuse
- `P/Q/alpha/beta` Lanczos storage lifecycle
- bidiagonalization loop
- small bidiagonal QR solve
- singular-value sorting and vector recovery

Role:

- distinct backend family from the full-SVD bidiagonal path
- separate approximation strategy, memory shape, and caller-facing proof set

Assessment:

- strongest maintainability target in the file
- large enough to matter
- cohesive enough to move as one owned slice
- supported by its own benchmark and a dense, clearly named partial-SVD test
  cluster

### 5. Application wrappers and reporting utilities

Main functions:

- `sparse_svd_rank(...)`
- `sparse_pinv(...)`
- `sparse_svd_lowrank(...)`
- `sparse_svd_lowrank_sparse(...)`
- `sparse_cond(...)`

Role:

- public application-layer wrappers built on top of `sparse_svd_compute(...)`

Assessment:

- should remain near the main full-SVD orchestration path for now
- splitting them first would be mechanical rather than ownership-sharpening

## Ranked maintainability targets

From strongest to weakest:

1. partial-SVD Lanczos backend extraction
2. bounded regrouping of the SVD application-wrapper cluster
3. bidiagonal QR helper extraction
4. low-rank sparse outer-product cluster extraction
5. file-local comment/order cleanup only

Interpretation:

- the file does not need a vague cleanup pass first
- it has one clearly superior first owned slice: the partial-SVD backend

## Chosen Sprint 56 Day 10 direction

Sprint 56 Day 10 should emphasize:

- helper extraction

Specifically:

- extract the partial-SVD Lanczos backend into its own owned source file

Proposed new file:

- `src/sparse_svd_partial.c`

Keep in `src/sparse_svd.c`:

- full-SVD/public orchestration
- reflector extraction and bidiagonal QR machinery
- full-mode basis padding
- application wrappers:
  - `sparse_svd_rank(...)`
  - `sparse_pinv(...)`
  - `sparse_svd_lowrank(...)`
  - `sparse_svd_lowrank_sparse(...)`
  - `sparse_cond(...)`

Move into the owned partial-SVD file:

- `sparse_svd_partial(...)`
- any small private helpers introduced only to support that Lanczos backend

## Why the partial-SVD backend is the right first target

### Ownership clarity

It is already a separate backend family:

- full SVD:
  - bidiagonalization + implicit QR on the physical bidiagonal
- partial SVD:
  - Lanczos bidiagonalization + small bidiagonal solve + approximate vector
    recovery

Those are materially different algorithms and lifecycle shapes, not just two
wrappers around one helper core.

### Proof cost is bounded and already explicit

Primary proof surfaces already cluster around partial SVD:

- `tests/test_svd.c`
  - partial sigma-only coverage
  - partial vector recovery coverage
  - timing/parity coverage
- `benchmarks/bench_svd.c`
  - explicit partial-vs-full reporting

That makes the extraction easier to prove than a deeper QR-core split.

### Public-contract risk is lower than a bidiagonal-core split

Moving `sparse_svd_partial(...)` leaves these higher-coupling surfaces in the
main file:

- `sparse_svd_compute(...)`
- `sparse_svd_extract_uv(...)`
- `bidiag_svd_iterate(...)`
- the application wrappers

So Sprint 56 can reduce ownership without simultaneously reopening the most
shared full-SVD core.

## Explicit non-goals for Day 10

The Day 10 SVD batch should not:

- redesign the public SVD API
- change benchmark or example meaning
- move the full bidiagonal QR core first
- reopen `economy=0` design questions
- broaden into a new private-header taxonomy unless it is strictly required
- mix in unrelated low-rank or condition-number redesign

## Expected Day 10 touch set

Primary expected touched set:

- `src/sparse_svd.c`
- `src/sparse_svd_partial.c` (new)
- `src/sparse_svd_internal.h`
- `Makefile`
- `CMakeLists.txt`

Possible secondary touch if strictly needed:

- `benchmarks/bench_svd.c`

Avoid by default:

- `include/sparse_svd.h`
- `tests/test_svd.c`
- `examples/example_svd_lowrank.c`

## Conclusion

Sprint 56 now has a bounded SVD maintainability target rather than an
open-ended cleanup bucket:

- `src/sparse_svd.c` reduces cleanly to five ownership bands
- the strongest first target is the partial-SVD Lanczos backend
- Day 10 should land helper extraction, not broad in-place cleanup
- the full-SVD/public orchestration path should remain in the main file for
  this sprint

That gives Sprint 56 an explicit, defensible SVD landing direction for the
next code day.
