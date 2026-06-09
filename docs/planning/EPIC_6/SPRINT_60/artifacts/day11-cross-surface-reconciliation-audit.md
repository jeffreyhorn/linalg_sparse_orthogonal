# Sprint 60 Day 11: Cross-Surface Reconciliation Audit

## Purpose

Re-read the Sprint 60 target definition, architecture seam audit,
configuration/performance audits, architecture contract, and
validation/platform contract as one package, then collapse any remaining
wording or interpretation drift before the sprint moves into closeout work.

## Reconciled Cross-Surface Rules

### 1. The Day 5 target bands are outcome bands, not implementation order

The strongest apparent drift across the Sprint 60 package is only superficial:

- Day 5 defines five Epic 6 outcome bands
- Day 7-8 rank the future implementation queue
- Day 9 turns those findings into architecture rules
- Day 10 turns them into proof/validation rules

This now reconciles cleanly as:

- Day 5 = what Epic 6 must achieve
- Day 7-8 = what should likely happen first
- Day 9 = what later implementation work must preserve
- Day 10 = how later implementation work must prove itself

Interpretation:

- benchmark/platform/packaging maturity remains a real Epic 6 target band
- but it does not outrank typed configuration convergence or backend/AUTO
  policy rationalization in likely implementation order

### 2. The preserved workflow fence is stable across every Sprint 60 surface

No blocker-level workflow contradiction remains across Day 5-10.

The package now reads consistently as:

- one-shot workflows remain first-class product entry points
- repeated direct solves remain the explicit analysis/factors lifecycle
- iterative repeated-run support remains bounded to:
  - `CG`
  - `GMRES`
  - `MINRES`
- eigensolver repeated-run support remains bounded to:
  - grow-m Lanczos
  - thick-restart Lanczos
  - explicit `LOBPCG`
- `BiCGSTAB` and block iterative workflows remain one-shot compatibility
  surfaces

Interpretation:

- Epic 6 may improve usability and control placement around those seams
- it does not reopen the public repeated-run support boundary inherited from
  Epic 5

### 3. Public typed control, internal policy, build switches, and legacy overrides now form one coherent control-placement rule

The Day 7-8 audit and Day 9 contract now reconcile without any missing middle
layer:

- public typed options remain the preferred lane for stable caller-visible
  per-call or per-object behavior
- internal typed policy remains the preferred lane for AUTO heuristics and
  implementation strategy
- compile-time build switches remain the correct lane for build shape
- env vars survive only as bounded compatibility or maintainer-facing override
  surfaces

Interpretation:

- the Sprint 60 package no longer describes env-var cleanup as a vague general
  desire
- it now describes one specific later migration target:
  ND/FM ordering and refinement control

### 4. Benchmark proof and benchmark governance are now explicitly separate layers

The Day 6 seam audit, Day 8 performance audit, and Day 9 contract now agree on
one rule:

- existing benchmark binaries stay the evidence/proof layer
- governance sits above them

That means:

- `bench_refactor` and `bench_refactor_csc` stay the direct lifecycle proof
  surfaces
- `bench_iterative_reuse` stays the iterative handle proof surface
- `bench_eigs_reuse` stays the eigensolver handle proof surface
- later Epic 6 performance work should define canonical baselines,
  machine-readable conventions, and regression-sensitive tiers above those
  binaries instead of casually replacing them

### 5. Packaging/platform maturity is a real target band, but subordinate to the truthfulness contract

Day 5 and Day 8 both identify packaging/platform maturity as real Epic 6 work.
Day 10 constrains how that work can be described.

This now reconciles to:

- packaging/platform maturity is a valid Epic 6 goal
- it is not allowed to outrun reviewed evidence
- Linux remains the authoritative reviewed source of truth
- macOS remains reviewed but narrower
- Windows remains a reviewed CMake subset with explicit staged exclusions and
  staged non-CMake surfaces

Interpretation:

- Sprint 60 does not promise platform parity by implication
- later packaging/platform work must land under measured Linux/macOS/Windows
  truthfulness rules

### 6. Coverage is no longer an active ranked residual, and dead-code still is

The Day 3 residual audit and Day 10 validation contract now read consistently:

- coverage is enforced and useful
- coverage is not the main active reviewed-baseline residual
- dead-code remains a real reviewed surface
- serialized dead-code topology remains an intentional operational limit

Interpretation:

- Epic 6 should not spend early sprint bandwidth pretending coverage policy is
  unsettled
- dead-code topology and platform follow-through remain honest deferred work

## Residual Open Questions To Carry Into Sprint 61+

The reconciliation pass leaves a smaller and more concrete carry-forward queue.

### 1. Direct usability convergence start point

The package consistently says direct usability convergence is the highest-value
product outcome, but Sprint 60 intentionally does not yet freeze which
front-door surface should move first:

- one-shot direct entry wording and defaults
- analysis/factors lifecycle discoverability
- matrix-state safety/usability around refactor flows

This is now a Sprint 61 implementation-planning question, not a Sprint 60
contract gap.

### 2. ND/FM typed-control ownership boundary

The package consistently identifies ND/FM tuning as the strongest env-var
productization gap, but later work still needs to decide:

- which controls become public typed options
- which controls remain internal typed policy
- which compatibility overrides survive temporarily

The control-placement rule is now stable; the exact migration cut is still a
future implementation decision.

### 3. AUTO/backend policy rationalization cut line

The package consistently ranks backend/AUTO cleanup second, but later work
still needs to choose the exact first landing seam:

- direct CSC crossover policy
- eigensolver AUTO crossover policy
- shared threshold-ownership cleanup pattern

### 4. Benchmark-governance canonical surface shape

The package consistently separates benchmark proof from benchmark governance,
but later work still needs to choose the concrete governance vehicle:

- metadata file(s)
- wrapper-level contract
- README plus machine-readable output conventions
- some bounded combination of the above

### 5. Packaging/platform maturity depth

The package consistently says packaging/platform work is real but not first.
Later work still needs to choose how far Epic 6 should go within the reviewed
truthfulness fence:

- install/export and pkg-config tightening only
- shared-library/distribution follow-through
- broader Windows/macOS follow-through where evidence supports it

## Day 11 Exit State

Sprint 60’s target definition, non-goal fence, architecture rules, and
validation/platform contract now read as one coherent package:

- no blocker-level workflow contradiction remains
- no target-order versus implementation-order ambiguity remains
- no benchmark-proof versus governance ambiguity remains
- no packaging/platform ambition now floats above the measured truthfulness
  contract

That gives Days 12-14 a stable closeout base instead of another interpretation
pass.
