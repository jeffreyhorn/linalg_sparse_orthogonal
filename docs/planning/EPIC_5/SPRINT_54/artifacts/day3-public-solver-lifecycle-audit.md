# Sprint 54 Day 3 - public solver lifecycle audit

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Audit the live public repeated-run iterative and eigensolver surfaces before
Sprint 54 makes support-boundary decisions or code changes.

## Supported public repeated-run surface

The live public repeated-run support is explicit and still intentionally
narrow:

- iterative public handles exist for:
  - CG
  - GMRES
- eigensolver public handles exist for:
  - symmetric eigensolves through the shared
    `sparse_eigs_sym_with_handle(...)` path
- one-shot APIs remain first-class supported entry points on both sides

Interpretation:

- Sprint 54 is not solving “no public repeated-run support”
- it is deciding whether the current bounded support set is the steady-state
  answer or whether a few remaining families should join it

## Iterative-side asymmetry classes

### MINRES

MINRES is the strongest remaining inclusion candidate:

- there is already an internal reusable-workspace helper:
  - `sparse_iter_workspace_prepare_minres(...)`
- MINRES is therefore closer to the CG/GMRES reusable-workspace model than
  the public surface suggests
- however, there are still no public handle prepare/run entry points for
  MINRES

Interpretation:

- the MINRES gap is mainly public-surface exposure and proof/alignment work,
  not missing reusable infrastructure

### BiCGSTAB

BiCGSTAB is a different class of gap:

- it already has broad public one-shot coverage:
  - scalar
  - block
  - matrix-free
- but its implementation still uses a dedicated `bicgstab_workspace_t` path
  rather than the existing public iterative handle owner

Interpretation:

- BiCGSTAB is a real public repeated-run asymmetry
- but it is a more expensive inclusion target than MINRES because the
  reusable seam is less aligned with the existing handle model

### Block workflows

Block iterative workflows are not strong first public-handle targets:

- block CG uses its own internal block workspace view
- block GMRES / MINRES / BiCGSTAB stay on independent or per-column wrapper
  paths
- current docs and examples do not present block workflows as the main public
  repeated-run story

Interpretation:

- block workflows are better treated as compatibility surfaces unless Day 4
  surfaces a clear lifecycle case for them

## Eigensolver-side asymmetry class

The eigensolver repeated-run surface is structurally closer to complete:

- one generic public handle surface already exists
- that handle already fronts the main `sparse_eigs_sym(...)` entry
- the current public repeated-run benchmark proof covers:
  - grow-m Lanczos
  - thick-restart Lanczos
- the strongest remaining drift is around:
  - LOBPCG caller-surface underrepresentation
  - benchmark/example/docs agreement

Interpretation:

- Sprint 54 likely needs eigensolver lifecycle tightening and proof alignment
  more than a broad new eigensolver API shape

## Caller-surface drift

The caller-facing repeated-run story is still underrepresented outside README
bullets and dedicated reuse benchmarks:

- `examples/README.md` explicitly says shipped examples still lean on the
  one-shot public APIs
- `example_iterative.c` is still a one-shot GMRES + ILU demo
- `example_eigs.c` is still a one-shot eigensolver demo, including explicit
  LOBPCG usage without the public handle path
- `example_ic_minres.c` is still a one-shot MINRES / block-MINRES teaching
  surface
- `bench_iterative_reuse.c` only proves the public handle path for:
  - CG
  - GMRES
- `bench_eigs_reuse.c` only proves the public handle path for:
  - grow-m Lanczos
  - thick-restart Lanczos

Interpretation:

- the repo already proves repeated-run support exists
- but it does not yet make a strong public case that MINRES, BiCGSTAB, or
  LOBPCG belong on the same steady-state repeated-run support tier

## Reduced seam classes

The remaining Sprint 54 problem now reduces to five seam classes:

1. iterative public-handle asymmetry:
   - CG/GMRES supported
   - MINRES internal seam exists but public handle is absent
   - BiCGSTAB remains more isolated
2. block-workflow support-boundary ambiguity
3. eigensolver proof/example drift, especially around LOBPCG
4. repeated-run benchmark support-set drift
5. example/README support-boundary drift

## Ranked Day 4 target list

Highest-value Day 4 decisions:

1. MINRES: strongest inclusion candidate
2. BiCGSTAB: explicit inclusion vs explicit bounded exclusion
3. block iterative workflows: likely bounded exclusion unless a strong public
   lifecycle story emerges
4. eigensolver tightening: likely keep one public handle surface and align
   proof/docs rather than inventing new APIs
5. benchmarks/examples/docs: align after the support boundary is fixed

## Conclusion

Day 3 closes with a concrete audit:

- the supported public repeated-run surface is explicit
- the remaining asymmetries are now separated by class
- MINRES and BiCGSTAB are not the same type of gap
- eigensolver work is more likely tightening than expansion
- Day 4 can decide the support boundary from a ranked seam list instead of a
  generic “finish the remaining families” backlog
