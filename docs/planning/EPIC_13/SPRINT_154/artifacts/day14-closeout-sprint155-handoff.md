# Sprint 154 Day 14 Closeout And Sprint 155 Handoff

## Purpose

Day 14 closes Sprint 154 by consolidating the external comparison target,
dependency policy, output schema, harness, report-index integration, published
study, validation status, residual comparative gaps, and Sprint 155
tutorial/API handoff.

## Sprint 154 Closure Summary

Sprint 154 built and published the first normalized local external-comparison
lane:

- target: `qr-minnorm`;
- fixture: `qr_underdetermined_minnorm_2x4`;
- baseline: source-controlled external-process dense reference helper
  `tests/qr_external_dense_reference.py`;
- project path: temporary C probe calling `sparse_qr_solve_minnorm`;
- generated artifact root: `build/comparison/qr_minnorm/`;
- normalized report family: `comparison/qr_minnorm`;
- maintained freshness command: `make report-index-comparison-freshness`;
- source-controlled study snapshot:
  `first-narrow-qr-minnorm-comparison-study.md`.

All six selected generated comparison rows pass under the selected local
freshness gate:

| Metric | Delta | Tolerance | Status |
| --- | --- | --- | --- |
| `project_status` | | status-only | `pass` |
| `baseline_status` | | status-only | `pass` |
| `residual_norm` | `1.5700924586837752e-16` | `1e-10` | `pass` |
| `solution_norm` | `1.1102230246251565e-16` | `1e-10` | `pass` |
| `solution_values` | `1.1102230246251565e-16` | `1e-10` | `pass` |
| `project_vs_baseline_max_abs_delta` | `1.1102230246251565e-16` | `1e-10` | `pass` |

The sprint did not claim broad QR parity or external-library ecosystem parity.

## Artifact Index

| Day | Artifact | Evidence Role |
| --- | --- | --- |
| Day 1 | `day1-comparison-boundary.md` | Scope, stop conditions, and external-comparison non-claim boundary. |
| Day 2 | `day2-target-candidate-audit.md` | QR and partial-SVD candidate audit plus baseline option review. |
| Day 3 | `day3-comparison-target-selection.md` | Selected `qr_underdetermined_minnorm_2x4` target and metric contract. |
| Day 4 | `day4-dependency-pinning-policy.md` | Baseline command, dependency policy, provenance, and skip/defer rules. |
| Day 5 | `day5-comparison-output-schema-design.md` | Study row schema, selected metrics, status semantics, and freshness model. |
| Day 6 | `day6-harness-architecture-design.md` | Harness architecture, artifact paths, row ids, and failure classes. |
| Day 7 | `day7-harness-project-runner-scaffold.md` | Project-side probe implementation record. |
| Day 8 | `day8-baseline-runner-implementation.md` | Baseline discovery, execution, parsing, and dependency diagnostics. |
| Day 9 | `day9-comparison-logic-implementation.md` | Project-vs-baseline comparison rows and self-check implementation. |
| Day 10 | `day10-report-integration-design.md` | Report-index product decision and integration policy. |
| Day 11 | `day11-report-integration-implementation.md` | `comparison/qr_minnorm` report-family implementation and Make target. |
| Day 12 | `day12-documentation-alignment.md` | README, maintainer, report, and solver-selection documentation alignment. |
| Day 13 | `first-narrow-qr-minnorm-comparison-study.md` | Source-controlled study publication snapshot. |
| Day 13 | `day13-integrated-validation-and-study-publication.md` | Integrated validation and residual comparative gap register. |
| Day 14 | `day14-closeout-sprint155-handoff.md` | Final closeout and Sprint 155 handoff. |

## Final Validation Results

Day 14 reran the final focused comparison, report-index, schema, documentation,
and whitespace validation set.

| Validation | Result | Evidence |
| --- | --- | --- |
| Comparison freshness | Pass | `make report-index-comparison-freshness` passed and reported local-only generated comparison freshness ok. |
| Harness self-check | Pass | `python3 scripts/run_external_comparison.py --self-check` passed. |
| Corpus/report schema | Pass | `python3 scripts/validate_corpus_schema.py` passed. |
| Combined report-index structure | Pass | `python3 scripts/normalize_report_index.py --family corpus --family oracle --family comparison --check` reported `85` rows ok. |
| Required comparison freshness | Pass | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` reported freshness ok for `7` rows. |
| Focused stale wording scan | Pass | Active public/maintainer hits are non-claims or scoped boundaries. |
| Whitespace | Pass | `git diff --check` passed. |

No `.c` or public `.h` files were modified during Day 14. The final full
`make format && make lint && make test` gate is not required for the Day 14
documentation-only closeout. Earlier Sprint 154 implementation changed Python,
Makefile, report metadata, and docs; the focused comparison/report/schema gates
cover those surfaces.

## Report And Claim Boundary

The maintained comparison command is:

```sh
make report-index-comparison-freshness
```

The normalized report index family is:

- `report_family=comparison`;
- `subfamily=qr_minnorm`;
- `row_meaning=external_process_dense_reference_comparison`;
- `support_tier=local_only`;
- `freshness_policy=generated_compare_inputs`.

The required freshness gate expects:

- one source-controlled contract row;
- six generated selected rows for `qr_underdetermined_minnorm_2x4`;
- every selected generated row present exactly once;
- every selected generated row `pass`;
- no `skip`, `defer`, `fail`, or `error` row counted as proof.

Optional NumPy and SciPy baselines remain `defer`, not pass evidence.

## Non-Claims

Sprint 154 does not claim:

- broad QR parity;
- NumPy parity;
- SciPy parity;
- LAPACK parity;
- SuiteSparse parity;
- Eigen parity;
- external-library ecosystem parity;
- hosted CI proof;
- release proof;
- platform portability proof;
- package-manager proof;
- shared-library or ABI proof;
- performance superiority;
- state-of-the-art status.

## Residuals Carried Forward

Still deferred after Sprint 154:

- QR comparison beyond `qr_underdetermined_minnorm_2x4`;
- optional NumPy and SciPy package baselines;
- LAPACK, SuiteSparse, Eigen, PETSc, Trilinos, and other ecosystem baselines;
- QR raw Q/R basis, sign/orientation/order, pivot-order, and rank-threshold
  comparison;
- broad rank-deficient, nullspace, economy-mode, sparse-mode, and reorder
  comparison;
- partial-SVD publication under the normalized `comparison` family;
- portable runtime or performance comparison;
- hosted CI comparison publication;
- package-manager, shared-library, loader, and ABI comparison lanes.

## Sprint 155 Handoff

Sprint 155 is `Tutorial, Header Cleanup & API Reference Coherence`. It should
consume Sprint 154 output as documentation and API-reference context, not as a
reason to widen public API claims.

Sprint 155 should:

1. Audit `docs/tutorial.md` for references to QR, minimum-norm solves, report
   indexes, and comparison evidence.
2. Mention `make report-index-comparison-freshness` only in maintainer or
   advanced-report contexts, not in first-use tutorial flow unless the section
   is explicitly about evidence validation.
3. Keep public tutorial language tied to workflows and diagnostics, not broad
   external parity.
4. Preserve the QR comparison boundary if public headers or Doxygen comments
   mention `sparse_qr_solve_minnorm`.
5. Avoid adding header comments that imply NumPy, SciPy, LAPACK, SuiteSparse,
   Eigen, performance, hosted CI, package, ABI, or platform proof.
6. If API reference docs mention report-index evidence, link it to
   `comparison/qr_minnorm` and the selected fixture-local scope.
7. Run declaration-preservation checks for header cleanup and full quality
   gates if public headers change.

## Closeout Status

Sprint 154 is ready for retrospective preparation. The sprint closed one
complete narrow external-comparison lane and left broader comparison ambitions
explicitly deferred.
