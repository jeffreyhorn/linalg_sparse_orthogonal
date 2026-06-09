# Sprint 60 Day 9: Epic 6 Architecture Contract Draft

## Purpose

Convert the Sprint 60 Day 5-8 target definition and audit work into an
explicit architecture fence for the rest of Epic 6.

This contract is not an implementation plan. It is the rule set later sprints
must preserve while they improve productization, configuration, backend
policy, benchmark governance, packaging, and assurance.

## Contract Scope

This contract governs:

- public workflow boundaries
- configuration/control placement
- backend and AUTO-policy ownership
- benchmark proof versus performance governance
- validation/platform/packaging truthfulness
- bounded widening and non-goal rules

It does not itself reopen solver-family scope or authorize broad product-surface
expansion.

## 1. Product Workflow Boundary Contract

### 1.1 One-shot workflows remain first-class

The one-shot public workflows remain valid product entry points:

- one-shot LU
- one-shot Cholesky
- one-shot LDL^T
- one-shot QR
- one-shot iterative solves
- one-shot eigensolver calls
- one-shot SVD calls

Epic 6 may improve their usability, defaults, wording, and surrounding docs,
but must not demote them into legacy or second-class compatibility wrappers.

### 1.2 Repeated-run direct solves remain the explicit analysis/factors lifecycle

The repeated-run direct ownership model stays centered on:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

Epic 6 may:

- improve discoverability
- improve safety and usability around matrix-state expectations
- add typed configuration around this path

Epic 6 must not:

- replace this model with a generic universal direct handle
- create a second competing repeated-run ownership model for the same scope

### 1.3 Iterative repeated-run support remains bounded

Iterative repeated-run support remains explicitly bounded to public handles for:

- `CG`
- `GMRES`
- `MINRES`

Epic 6 may improve:

- usability around these handles
- typed control around them
- proof and benchmark governance around them

Epic 6 must not widen public repeated-run handle support implicitly to:

- `BiCGSTAB`
- block iterative workflows

### 1.4 Eigensolver repeated-run support remains bounded

The public eigensolver repeated-run handle remains bounded to:

- grow-m Lanczos
- thick-restart Lanczos
- explicit `LOBPCG`

Epic 6 may improve:

- control placement
- AUTO-policy ownership
- benchmark and assurance layers

Epic 6 must not widen this into a broader eigensolver-family expansion sprint
without explicit re-charter.

## 2. Configuration and Control-Placement Contract

Every control surface introduced or revised in Epic 6 must belong to exactly
one of four placement classes:

1. public typed option
2. internal typed policy
3. compile-time build switch
4. legacy compatibility override

No new control should be added without choosing one of these classes
explicitly.

### 2.1 Public typed option rules

A control belongs in a public typed option when all of the following are true:

- it changes supported caller-visible behavior
- it is meaningful on a per-call or per-object basis
- it is stable enough to document as a supported product surface
- different callers in one process may reasonably want different values

Strong current/future candidates:

- high-value ND/FM strategy and pass-budget controls if they remain supported
- direct-lifecycle usability controls tied to analysis/factor workflows

### 2.2 Internal typed policy rules

A control belongs in internal typed policy when:

- it shapes implementation strategy
- it may need structured ownership and testing
- it is not yet stable enough to promise as a public caller-facing control

Strong current/future candidates:

- backend AUTO heuristics
- internal performance-routing policy
- advisory strategy knobs that survive as implementation policy but not public
  API

### 2.3 Compile-time build switch rules

A control should stay build-time only when it changes build shape rather than
per-call solver behavior.

Current examples:

- `SPARSE_OPENMP`
- `SPARSE_MUTEX`
- `SANITIZE`

Epic 6 should avoid moving these into runtime API just for symmetry with other
controls.

### 2.4 Legacy compatibility override rules

An env-var or similar override may persist only when all of the following are
true:

- it preserves backward compatibility during a control migration
- it is clearly subordinate to the typed ownership model
- its precedence is documented explicitly
- it is not the only supported way to reach a mainstream control path

This is the only acceptable long-term lane for most of the current
process-global advanced tuning surface.

## 3. Backend and AUTO-Policy Contract

### 3.1 Explicit backend forcing remains caller-visible

Where public backend selectors already exist, Epic 6 should preserve them.

Caller-visible explicit forcing is part of the product surface for:

- direct backend selection where already exposed
- eigensolver backend selection

### 3.2 AUTO policy is an owned seam, not incidental glue

AUTO backend selection must be treated as a deliberate internal policy layer.

Epic 6 may:

- rationalize AUTO heuristics
- move threshold ownership away from public compile-time leakage where
  appropriate
- improve telemetry and docs coherence

Epic 6 must preserve:

- explicit backend forcing behavior
- caller visibility into which backend actually ran where telemetry already
  exists

### 3.3 Backend modernization stays bounded

Epic 6 backend work is bounded to modernizing selected hot-path architecture
and policy ownership.

It is not authorization for:

- broad vendor-backend parity work
- distributed accelerator scope
- large algorithm-family expansion as a proxy for architecture improvement

## 4. Benchmark and Performance-Governance Contract

### 4.1 Workflow-proof binaries remain the evidence layer

The existing benchmark binaries remain the benchmark proof surface:

- `bench_refactor`
- `bench_refactor_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`
- the remaining one-shot comparison and backend-sweep binaries

Epic 6 should build governance *above* them, not replace them casually.

### 4.2 Governance is a separate layer above the binaries

Performance-governance work must define:

- canonical benchmark surfaces
- stable output conventions where needed
- regression-sensitive tiers
- claim-bearing versus smoke-only runs

This is distinct from implementing new benchmark binaries.

### 4.3 Performance claims must stay evidence-backed

README-, docs-, or packaging-level performance claims should only rest on:

- maintained benchmark surfaces
- measured reviewed runs or explicitly-labeled local measurements
- clear truthfulness language when results are workload-dependent

Epic 6 must not create a more polished performance story by weakening the
evidence bar.

## 5. Assurance Contract

Epic 6 assurance expansion should focus on the hardest and most leverage-heavy
surfaces:

- repeated-run direct lifecycle
- CSC direct workflows
- iterative handle workflows
- eigensolver handle workflows
- configuration/control migration safety

Expected assurance forms later in Epic 6:

- stronger differential checks
- stronger property/oracle checks
- bounded new regression proof where control or policy ownership changes

Epic 6 should not chase generic test growth without clear leverage against the
product and architecture goals above.

## 6. Validation, Platform, and Packaging Truthfulness Contract

### 6.1 The current reviewed truth surface remains authoritative

Epic 6 continues to treat:

- `make quality-review-full`
- reviewed CMake parity
- the existing Linux/macOS/Windows dispositions

as the authoritative validation/truthfulness baseline unless fresh measured
evidence justifies change.

### 6.2 Platform ambition must remain subordinate to reviewed truthfulness

Epic 6 may improve packaging and platform maturity, but must not claim parity
beyond what reviewed measurement actually supports.

This especially applies to:

- Windows wrapper/build parity
- macOS staged surfaces
- dead-code/platform closure

### 6.3 Packaging work must distinguish product promise from build convenience

Later build/package work must keep a clean separation between:

- internal build convenience
- optional build shape
- public distribution promise

This matters especially because the current package story is credible but still
bounded around a `STATIC` primary target and explicit reviewed-platform limits.

## 7. Bounded Widening Rules

Epic 6 may widen:

- typed control ownership where the control is already effectively supported
- internal architecture seams on selected hot paths
- benchmark governance policy
- packaging/platform maturity within measured truthfulness
- assurance depth on the hardest existing workflows

Epic 6 must keep bounded:

- repeated-run direct ownership model
- iterative handle support-family scope
- eigensolver handle support-family scope
- single-node product scope
- public algorithm-family footprint unless explicitly re-chartered

## 8. Non-Goal Fence

Epic 6 does not authorize:

- distributed-memory / MPI sparse linear algebra scope
- immediate vendor-backend parity as the headline goal
- broad new solver-family expansion as the main story
- fake cross-platform closure without reviewed evidence
- generic maintainability work that does not materially support product,
  configuration, backend, benchmark, packaging, or assurance outcomes

## 9. Contract Implications for Later Sprints

If a later sprint proposes work that:

- adds a new control surface
- widens solver-family support
- changes benchmark meaning
- broadens packaging/platform claims
- weakens reviewed truthfulness

then it should first be checked against this contract, not judged only on local
technical convenience.

The strongest immediate implications are:

1. ND/FM control convergence is the first high-value configuration move
2. backend AUTO policy cleanup is the second high-value move
3. benchmark governance should build on existing proof binaries
4. packaging/platform work should follow, not lead, control coherence

## Day 9 Exit State

Sprint 60 now has a draft Epic 6 architecture contract:

- workflow boundaries are explicit
- control placement rules are explicit
- benchmark proof versus governance separation is explicit
- validation/platform/packaging truthfulness rules are explicit
- bounded widening and non-goal fences are explicit

That is enough for Day 10 to freeze the validation and platform contract
against a stable architecture fence rather than against raw audit notes.
