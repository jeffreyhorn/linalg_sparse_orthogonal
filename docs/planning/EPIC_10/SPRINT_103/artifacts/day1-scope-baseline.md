# Sprint 103 Day 1 Scope and Comparison Baseline

## Purpose

Day 1 converts the Sprint 103 project-plan section and Sprint 100/102 handoffs
into a bounded iterative, eigensolver, and SVD comparison package. It defines
workstreams, daily ownership, validation expectations, and claim boundaries
before any new fixtures, helpers, tests, or public documentation are changed.

## Sprint Goal

Sprint 103 raises evidence quality for iterative solvers, eigensolvers, and SVD
without claiming broad parity with mature external packages prematurely.

The sprint is successful only if new claims are tied to:

- named solver family and exact API path;
- fixture identity and fixture class;
- oracle, deterministic reference, or internal-reference behavior;
- tolerance, residual, orthogonality, convergence, or reconstruction criteria;
- validation command;
- unsupported, skipped, or expected-failure cases;
- explicit non-claims.

## Project-Plan Item Ownership

| item | project-plan estimate | day ownership | Day 1 interpretation |
|---|---:|---|---|
| Solver Family Audit | 18 hours | Days 2 and 7 | rank CG, MINRES, BiCGSTAB, eigen, thick-restart, LOBPCG, and SVD evidence before implementation |
| Convergence Fixture Design | 26 hours | Days 3-5 | define convergence, stagnation, tolerance, restart, preconditioning, residual, and rank fixtures before adding oracle lanes |
| Iterative Oracle Batch | 34 hours | Days 5-7 | add the highest-value iterative comparisons only after fixture and tolerance rules are frozen |
| Eigensolver Oracle Batch | 30 hours | Days 8-10 | add focused eigen, thick-restart, and LOBPCG comparison evidence with bounded residual and orthogonality expectations |
| SVD Comparison Follow-Through | 18 hours | Days 10-11 | extend SVD comparison evidence where it shares fixture or reporting infrastructure with spectral work |
| Reporting and Docs | 20 hours | Day 12 | document convergence-profile interpretation, residual criteria, and explicit non-claims |
| Validation and Closeout | 22 hours | Days 13-14 | reconcile earned, deferred, and non-claim states and hand Sprint 104 a stable evidence base |

## Workstream Inventory

| workstream | primary outputs | first day | closeout day |
|---|---|---:|---:|
| solver-family comparison audit | evidence inventory, weakness ranking, user-impact ranking, expansion queue | 2 | 2 |
| convergence fixture taxonomy | fixture classes, expected outcomes, residual criteria, skip/failure rules | 3 | 3 |
| helper and reporting boundary | helper reuse decision, convergence reporting contract, validation plan | 4 | 4 |
| iterative oracle batch | iterative design, tests/helpers, focused validation, rerank | 5 | 7 |
| eigensolver oracle batch | spectral design, tests/helpers, focused validation, SVD overlap scope | 8 | 10 |
| SVD follow-through | selected SVD comparison updates, validation, fixture/reporting reuse notes | 10 | 11 |
| reporting and documentation | residual interpretation, convergence-profile guidance, non-claim wording | 12 | 12 |
| validation and closeout | full gate, evidence reconciliation, artifact index, Sprint 104 handoff | 13 | 14 |

## Evidence Rules Inherited From Sprint 100

Sprint 100's solver comparison template requires every comparison artifact to
separate:

- correctness evidence;
- convergence evidence;
- timing evidence and local-only timing caveats;
- unsupported or skipped cases;
- non-claims.

For Sprint 103, an iterative, eigensolver, or SVD comparison claim is not earned
unless the artifact records:

| required field | Sprint 103 use |
|---|---|
| solver family and exact API path | prevent family-wide interpretation of one lane |
| fixture set | bind the claim to named matrices or generated cases |
| fixture class | distinguish convergence, stagnation, tolerance, restart, preconditioning, residual, and rank behavior |
| oracle or reference behavior | distinguish external helper, dense reference, deterministic property, and internal consistency |
| tolerance model | define residual, solution/eigenpair/vector, orthogonality, reconstruction, or rank acceptance before implementation |
| validation command | make the claim reproducible |
| unsupported cases | prevent skips, platform exclusions, or expected failures from reading like passes |
| non-claims | preserve boundaries after evidence passes |

## Sprint 102 Handoff Constraints

Sprint 102 gives Sprint 103:

- reusable external-reference status and reason-string conventions in
  `tests/test_solver_helpers.h`;
- a shared vector parser for helpers that emit the `OK n` / `ERROR` contract;
- fixture taxonomy habits for separating correctness, expected failure, and
  unsupported cases;
- direct-solver examples of bounded external dense-reference lanes;
- maintainer-guide trust-boundary language tying claims to evidence owners and
  fixture names;
- explicit warnings against promoting bounded fixture evidence into broad
  external-oracle or state-of-the-art claims.

Sprint 103 must not assume:

- iterative, eigensolver, or SVD paths already have external helper parity;
- helper reuse is justified before status, output, and fixture contracts are
  checked;
- one residual or convergence fixture proves broad package equivalence;
- local timing output proves portable performance superiority;
- direct compressed solver APIs or universal compressed solver parity exist.

## Validation Expectations

| change type | required validation |
|---|---|
| planning docs only | `git diff --check`; `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_103` |
| public docs only | `git diff --check`; trailing-whitespace scan on touched docs |
| helper script | focused helper command if runnable; docs hygiene |
| iterative/eigen/SVD test `.c` file | focused test binary; `make format`; `make lint`; `make test` |
| library `.c` or public `.h` file | focused affected tests; `make format`; `make lint`; `make test` |
| CMake/Make registration | focused build or CTest registration check plus any code-touch gate |

If any `.c` or `.h` file is modified, Sprint 103 must run:

```sh
make format && make lint && make test
```

## Initial Risk Register

| risk | mitigation |
|---|---|
| bounded residual fixtures interpreted as broad external parity | require non-claims in every comparison artifact |
| iteration counts treated as portable performance claims | record convergence separately from timing and avoid portable superiority wording |
| fixture behavior unclear across solver families | define fixture taxonomy before implementation |
| helper reuse hides family-specific output contracts | freeze helper/status behavior before code changes |
| spectral orthogonality and residual thresholds chosen after implementation | define acceptance criteria in Day 8 before spectral changes |
| SVD work expands beyond shared infrastructure | freeze SVD scope after spectral closeout |
| new test executable changes CTest count unexpectedly | prefer existing binaries unless a new executable is justified and registered consistently |

## Day 1 Deliverable Checklist

- [x] Sprint 103 project-plan section re-read.
- [x] Sprint 100 solver evidence rules re-read.
- [x] Sprint 102 closeout and oracle-helper handoff re-read.
- [x] Workstream inventory recorded.
- [x] Project-plan items mapped to day ownership.
- [x] Validation expectations recorded.
- [x] Sprint 100/102 non-claims preserved.

## Day 1 Conclusion

Sprint 103 is bounded as an iterative, eigensolver, and SVD comparison sprint.
It starts from Sprint 100 comparison discipline and Sprint 102 helper patterns,
but must earn every new trust claim through named fixtures, explicit
reference behavior, acceptance criteria, validation commands, unsupported-case
handling, and non-claim wording.
