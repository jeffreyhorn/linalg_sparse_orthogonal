# Day 10: Claim Recalibration

## Purpose

Day 10 applies the Day 9 claim-surface inventory without widening Sprint 176
evidence. The only public claim promoted today is the selected
allocation-failure proof for iterative repeated-run handle prepare/growth
cleanup.

## Public Documentation Updates

| Surface | Update | Claim boundary |
| --- | --- | --- |
| `README.md` quality summary | Added a selected allocation-failure proof bullet. | Family-local to CG, GMRES, and MINRES repeated-run handle prepare/growth cleanup. |
| `README.md` command map | Added `make iterative-allocation-failure-gate`. | Focused local proof command, not a hosted CI or release gate by itself. |
| `docs/maintainer_guide.md` proof owner | Added Day 9/Day 10 claim-recalibration interpretation under the existing Sprint 176 owner. | Earned local focused proof only; not package, report-index, performance, release, or state-of-the-art evidence. |
| `docs/planning/EPIC_15/SPRINT_176/WORKING_NOTES.md` | Added the Day 10 evidence ledger, non-claim review, validation, and daily log entry. | Planning closeout now treats selected allocation-failure proof as earned while retaining broad non-claims. |

## Evidence Ledger Update

| Evidence | Status | Owner | Public wording allowed |
| --- | --- | --- | --- |
| Selected iterative allocation-failure proof | Earned, local focused gate | `tests/test_iterative.c`, `tests/test_iterative_handle_helpers.h`, `make iterative-allocation-failure-gate`, `ctest -L allocation_failure` | Family-local proof for CG/GMRES/MINRES repeated-run handle prepare/growth cleanup. |
| Public iterative cleanup invariant | Earned, header-documented | `include/sparse_iterative.h`, README repeated-run lifecycle section, maintainer guide | Handles are safe to free when NULL, zeroed, or already freed; invalid prepare calls do not publish internal state. |
| Broad allocation-failure coverage | Unsupported | Retained non-claim | Do not claim coverage across direct solvers, eigensolvers, matrix construction, package/install flows, generated-report tooling, or unrelated allocation paths. |
| Selected report freshness | Previously earned only for selected hosted lanes | Linux/macOS workflows and report-index gates | Do not cite as allocation-failure evidence. |
| Static-first package contract | Previously earned, separate surface | Install scripts, CMake/Make package lanes, static deferral guard | Do not cite as allocation-failure evidence. |
| Performance/report rows | Local or selected hosted depending on row | Benchmark/report targets and workflow lanes | Do not cite as allocation-failure evidence or portable speed proof. |

## Retained Non-Claims

The Day 10 wording preserves these non-claims:

- broad allocation-failure cleanup coverage across all solvers and allocation
  paths;
- state-of-the-art sparse linear algebra status;
- portable performance superiority;
- broad external-library parity;
- shared-library support, dynamic ABI compatibility, and runtime-loader
  behavior;
- package-manager provider availability;
- broad platform parity;
- Windows Makefile parity or Windows `pkg-config` command execution parity;
- broad report freshness or Windows report freshness;
- hosted generated API HTML publication;
- release evidence.

## Guard Decision

No new mechanical guard was added on Day 10. Reason:

- the existing focused proof is already mechanically reachable through
  `make iterative-allocation-failure-gate`;
- CMake label selection is already available through `ctest -L
  allocation_failure`;
- Day 10 changed only documentation and did not alter package, ABI,
  package-manager, report-index, generated-output, performance, workflow, or C
  surfaces.

If future wording promotes this local focused proof into hosted CI, that change
should add a workflow guard at the same time.

## Validation

Day 10 changed documentation only. No `.c` or `.h` files were modified for
this day, so the full C quality gate is not required.

Validation commands:

```sh
make iterative-allocation-failure-gate
git diff --check
```

Result: passed.
