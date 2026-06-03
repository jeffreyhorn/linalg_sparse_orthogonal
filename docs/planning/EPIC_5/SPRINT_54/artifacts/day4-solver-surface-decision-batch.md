# Sprint 54 Day 4 - solver surface decision batch

Date: 2026-06-03
Branch: `sprint-54`

## Purpose

Decide the steady-state public repeated-run solver support boundary before
Sprint 54 broadens any handle surface.

## Final decision

Sprint 54 will treat the following as the steady-state public repeated-run
support boundary for this sprint.

### Included

- iterative public repeated-run handles:
  - CG
  - GMRES
  - MINRES
- eigensolver public repeated-run handles:
  - the existing `sparse_eigs_handle_t` surface for symmetric eigensolves
  - with proof/docs/benchmark tightening for LOBPCG where needed

### Explicitly excluded

- BiCGSTAB public repeated-run handle exposure
- selected block iterative workflow public-handle exposure
- backend-specific eigensolver public API families beyond the existing generic
  symmetric-handle surface

## Decision rationale

### Why MINRES is included

MINRES is the strongest inclusion candidate because:

- it already has an internal reusable-workspace seam:
  - `sparse_iter_workspace_prepare_minres(...)`
- it is a first-class documented solver family already
- it has substantial existing regression and example coverage
- its caller shape is close to the current CG/GMRES handle story

This makes MINRES the highest-value repeated-run public asymmetry to close.

### Why BiCGSTAB is excluded

BiCGSTAB remains out of scope for Sprint 54 public-handle exposure because:

- its reusable seam is still isolated around `bicgstab_workspace_t`
- it already has a broad one-shot public footprint:
  - scalar
  - block
  - matrix-free
- adding a public handle would be a larger API and proof commitment than the
  MINRES case

This is an intentional compatibility boundary, not an accidental omission.

### Why block workflows are excluded

Block iterative workflows remain supported, but they are excluded from public
handle exposure because:

- their current public story is compatibility-first
- they span multiple algorithm shapes and do not reduce cleanly to one
  bounded repeated-run public-handle design
- including them would broaden Sprint 54 into a larger API-design sprint

### Why eigensolver work is tightening, not expansion

The eigensolver side already has one coherent public repeated-run handle
surface. The remaining Sprint 54 gap is:

- LOBPCG proof/docs/example underrepresentation
- benchmark alignment
- support-boundary clarity

It is not the absence of a handle model.

## Implementation order

Sprint 54 should now proceed in this order:

1. MINRES public-handle exposure
2. regression proof for the final supported iterative-handle set
3. eigensolver lifecycle/proof/docs tightening, especially for LOBPCG
4. repeated-run benchmark alignment
5. example/README adoption and explicit exclusion wording
6. final validation and closeout

## Resulting sprint boundary

This decision makes Sprint 54 materially smaller and clearer:

- one new iterative repeated-run public family is in scope:
  - MINRES
- one major iterative family is intentionally excluded:
  - BiCGSTAB
- block workflows stay supported but outside the public handle boundary
- eigensolver work stays on the existing generic handle surface

## Conclusion

Day 4 closes with the needed decision:

- the public steady-state repeated-run support set is fixed before new code
  expansion
- exclusions are explicit and justified
- implementation can now proceed from a bounded, credible support boundary
