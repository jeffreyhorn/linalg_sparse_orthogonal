# Sprint 102 Day 1 Scope and Evidence Baseline

## Purpose

Day 1 converts the Sprint 102 project-plan section and Sprint 100/101 handoffs
into a bounded direct-solver evidence package. It defines workstreams, daily
ownership, validation expectations, and claim boundaries before any direct
solver tests, helpers, or documentation are changed.

## Sprint Goal

Sprint 102 deepens correctness evidence for direct solvers with named
external-oracle or dense-reference comparisons and cleaner family-local proof
ownership.

The sprint is successful only if new claims are tied to:

- named solver family and exact API path;
- fixture identity and fixture class;
- oracle or deterministic reference behavior;
- tolerance or acceptance criteria;
- validation command;
- unsupported, skipped, or expected-failure cases;
- explicit non-claims.

## Project-Plan Item Ownership

| item | project-plan estimate | day ownership | Day 1 interpretation |
|---|---:|---|---|
| Direct Solver Gap Audit | 18 hours | Days 2 and 6 | inventory and rerank Cholesky, LDLT, LU, QR, SVD, and dispatch evidence before implementation |
| Fixture Taxonomy | 22 hours | Day 3 plus later boundary days | define fixture classes and expected failures before adding oracle lanes |
| Oracle Helper Extraction | 28 hours | Days 4-6 | extract only helpers that reduce duplication or proof-owner concentration |
| LDLT/Cholesky Expansion | 28 hours | Days 7-9 | add the highest-value CSC direct-family oracle expansion within a frozen boundary |
| LU/QR/SVD Expansion | 34 hours | Days 9-11 | add the highest-value general direct-solver oracle or failure-mode expansion |
| Solver Selection Docs | 18 hours | Day 12 | update guidance only after evidence exists |
| Validation and Closeout | 20 hours | Days 13-14 | reconcile earned/deferred/non-claim states and hand Sprint 103 a stable evidence base |

## Workstream Inventory

| workstream | primary outputs | first day | closeout day |
|---|---|---:|---:|
| direct-solver evidence audit | current evidence inventory, proof-depth classification, ranked queue | 2 | 2 |
| fixture taxonomy | fixture classes, expected failures, storage/naming rules | 3 | 3 |
| helper extraction | extraction boundary, helper implementation, focused validation, rerank | 4 | 6 |
| CSC direct-family expansion | LDLT/Cholesky boundary, tests/helpers, validation, residuals | 7 | 9 |
| LU/QR/SVD expansion | selected family boundary, tests/helpers, validation, residuals | 9 | 11 |
| solver guidance | trust-boundary documentation and supported/non-supported wording | 12 | 12 |
| validation and closeout | full gate, claim reconciliation, artifact index, Sprint 103 handoff | 13 | 14 |

## Evidence Rules Inherited From Sprint 100

Sprint 100's solver comparison template requires every comparison artifact to
separate:

- correctness evidence;
- convergence evidence where relevant;
- timing evidence and local-only timing caveats;
- unsupported or skipped cases;
- non-claims.

For Sprint 102, a direct-solver comparison claim is not earned unless the
artifact records:

| required field | Sprint 102 use |
|---|---|
| solver family and exact API path | prevent family-wide interpretation of one lane |
| fixture set | bind the claim to named matrices |
| fixture class | distinguish SPD, indefinite, singular, rectangular, scaled, ordered, and malformed cases |
| oracle or reference behavior | distinguish external/dense reference from internal smoke tests |
| tolerance model | define residual and solution-difference acceptance before implementation |
| validation command | make the claim reproducible |
| unsupported cases | prevent skips or xfails from reading like passes |
| non-claims | preserve boundaries after evidence passes |

## Sprint 101 Handoff Constraints

Sprint 101 gives Sprint 102:

- validated CSR/CSC constructor front-door behavior;
- copy-ownership and diagnostic compressed-input rules;
- focused CSR/CSC regression tests in `tests/test_csr.c`;
- representative CSR-to-LU and CSC-to-Cholesky solver-entry smoke tests;
- an executable compressed-input example;
- explicit non-claims around direct CSR/CSC solver APIs and broad solver
  parity.

Sprint 102 must not assume:

- direct CSR/CSC solver entry objects exist;
- compressed constructors bypass `SparseMatrix` ownership;
- all solver families have compressed-input parity proof;
- external oracle coverage is universal;
- one new direct-solver lane proves state-of-the-art status.

## Validation Expectations

| change type | required validation |
|---|---|
| planning docs only | `git diff --check`; `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_102` |
| public docs only | `git diff --check`; trailing-whitespace scan on touched docs |
| helper script | focused helper command if runnable; docs hygiene |
| direct-solver test `.c` file | focused test binary; `make format`; `make lint`; `make test` |
| library `.c` or public `.h` file | focused affected tests; `make format`; `make lint`; `make test` |
| CMake/Make registration | focused build or CTest registration check plus any code-touch gate |

If any `.c` or `.h` file is modified, Sprint 102 must run:

```sh
make format && make lint && make test
```

## Initial Risk Register

| risk | mitigation |
|---|---|
| expanding one fixture lane into broad direct-solver claims | require non-claims in every comparison artifact |
| expected failures interpreted as regressions or passes | define fixture taxonomy before implementation |
| helper extraction becoming a refactor sprint | limit helper work to proof-owner and duplication reductions |
| CTest count drift from new executables | prefer adding tests to existing binaries unless a new executable is justified and registered consistently |
| timing output interpreted as performance superiority | keep timing local-only unless a later performance-sentinel artifact defines thresholds |
| compressed-input work overclaimed as direct CSR/CSC solver APIs | preserve Sprint 101 non-claim language in Day 13 and Day 14 reconciliation |

## Day 1 Deliverable Checklist

- [x] Sprint 102 project-plan section re-read.
- [x] Sprint 100 solver evidence rules re-read.
- [x] Sprint 101 closeout and claim-boundary handoff re-read.
- [x] Workstream inventory recorded.
- [x] Project-plan items mapped to day ownership.
- [x] Validation expectations recorded.
- [x] Sprint 101 non-claims preserved.

## Day 1 Conclusion

Sprint 102 is bounded as a direct-solver evidence sprint. It starts from a
stable compressed-input front door but must earn any new solver trust claim
through named fixtures, explicit oracle behavior, tolerances, validation
commands, unsupported-case handling, and non-claim wording.
