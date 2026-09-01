# Sprint 191 Day 4: Reference Dependency Policy

## Purpose

Define the exact reference implementation, dependency-status semantics,
unsupported-environment handling, and remediation wording for the selected
`qr-incompatible-ls` external comparison family.

## Dependency Decision

`qr-incompatible-ls` uses the same source-controlled QR dense helper dependency
model as the existing QR selected comparison targets.

| Dependency | Required | Status policy | Pass evidence? | Rationale |
| --- | --- | --- | --- | --- |
| Current Python executable | Yes | Must be available because the runner is itself Python and invokes the dense helper through `sys.executable`. | Yes, only as execution-environment evidence. | The selected reference path is a source-controlled Python helper, not an external package. |
| `tests/qr_external_dense_reference.py` | Yes | Must exist and return the expected `OK <count>` record for `qr_overdetermined_incompatible_4x2`. | Yes, as the selected dense-reference baseline. | This is the bounded reference implementation for the fixture. |
| NumPy | No | Always `defer` with `optional_package_baseline_not_selected`. | No. | NumPy is not the selected baseline and must not create external ecosystem parity evidence. |
| SciPy | No | Always `defer` with `optional_package_baseline_not_selected`. | No. | SciPy is not the selected baseline and must not create external ecosystem parity evidence. |

No external package is required for Sprint 191. Optional package absence must
remain visible as deferred non-proof evidence and must never be interpreted as
selected comparison pass evidence.

## Reference Command

The selected reference command is:

```sh
python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2
```

The runner implementation should continue invoking the helper with
`sys.executable` so the dependency-status row reflects the actual interpreter
used by `scripts/run_external_comparison.py`.

Expected helper contract:

```text
OK 3
<solution_x0>
<solution_x1>
<residual_norm>
```

For the selected fixture, the values are expected to agree with:

```text
solution = [2.0, -1.0]
solution_norm = 2.2360679774997898
residual_norm = 1.7320508075688772
```

## Dependency-Status Row Contract

The new target should emit these dependency rows in `dependency_status.tsv`:

| dependency | status | status_reason | required | caveat |
| --- | --- | --- | --- | --- |
| `python3` | `pass` | `selected_interpreter_available` | `yes` | `current Python executable only; no package-manager inference` |
| `tests/qr_external_dense_reference.py` | `pass` or `error` | `baseline_helper_available` or `baseline_helper_missing` | `yes` | `source-controlled dense reference helper; not an external package` |
| `numpy` | `defer` | `optional_package_baseline_not_selected` | `no` | `deferred rows are not pass evidence` |
| `scipy` | `defer` | `optional_package_baseline_not_selected` | `no` | `deferred rows are not pass evidence` |

The existing `dependency_status_rows()` helper already emits this contract for
QR targets, so Sprint 191 should reuse it unless implementation uncovers a
target-specific issue.

## Failure Semantics

| Condition | Expected failure class or status | Required handling |
| --- | --- | --- |
| Dense helper file is absent | `missing_baseline_helper` and dependency row `baseline_helper_missing` | Fail the selected comparison generation; absence is not a skip/pass. |
| Helper command exits nonzero | `baseline_command_failed` | Fail with the captured command output and exact command context. |
| Helper emits no output | `baseline_malformed_output` | Fail before writing pass evidence. |
| Helper first line is not `OK 3` | `baseline_malformed_output` | Fail with the observed first line. |
| Helper value count is not integer or is not `3` | `baseline_malformed_output` | Fail with the expected count and fixture key. |
| Helper emits fewer or more numeric value lines than promised | `baseline_malformed_output` | Fail before metric rows are accepted. |
| Helper emits non-numeric values | `baseline_malformed_output` | Fail with the malformed values. |
| Project probe fails | `project_probe_failed` | Fail project status and selected comparison generation. |
| Project and baseline residuals disagree | `metric_tolerance_miss` through study-row validation | Fail the residual comparison row. |

## Unsupported Environment Policy

The new target should be required on the same local/Linux/macOS selected
comparison surfaces as the existing QR targets after workflow integration.
Windows selected comparison metadata should remain limited to the Sprint 190
Cholesky target unless a later Sprint 191 day adds hosted Windows proof for
`qr-incompatible-ls`.

Unsupported or unavailable environments must be represented as explicit
non-claims, not hidden pass evidence. In particular:

- missing NumPy or SciPy remains deferred optional package state;
- missing source-controlled helper is an error, not a deferral;
- Windows CMake/MSVC build support does not imply Windows selected report
  freshness for this new target;
- Linux/macOS hosted upload scope must name exact generated files if the
  target is added to those jobs.

## Remediation Wording

Preferred target-specific command after implementation:

```sh
python3 scripts/run_external_comparison.py --target qr-incompatible-ls
```

Preferred selected freshness command if target-specific freshness is needed:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

Preferred full selected comparison freshness command after the Makefile target
is updated:

```sh
make report-index-comparison-freshness
```

Diagnostics should avoid recommending NumPy/SciPy installation because those
packages are not the selected reference implementation for this target.

## Retained Non-Claims

The dependency policy retains these non-claims:

- no NumPy, SciPy, LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, or broad
  external-library ecosystem parity;
- no broad QR correctness or broad least-squares parity;
- no global rank-threshold policy;
- no Windows selected report freshness expansion beyond the Sprint 190
  Cholesky lane;
- no package-manager proof, shared-library ABI proof, performance superiority,
  release proof, or state-of-the-art claim.

## Day 4 Validation

Read-only/source checks:

```sh
git status --short --branch --ahead-behind
sed -n '130,170p' docs/planning/EPIC_17/SPRINT_191/PLAN.md
sed -n '1,260p' docs/planning/EPIC_17/SPRINT_191/artifacts/day3-fixture-metric-contract.md
sed -n '1500,1565p' scripts/run_external_comparison.py
sed -n '180,225p' tests/test_run_external_comparison.py
rg -n "dependency_status|numpy|scipy|missing_baseline_helper|baseline_command_failed|baseline_malformed_output|dependency_status_rows|optional" scripts/run_external_comparison.py tests/test_run_external_comparison.py tests/corpus/README.md docs/maintainer_guide.md
git diff --check
```

No `.c` or `.h` files were changed on Day 4, so `make format && make lint &&
make test` is not required.
