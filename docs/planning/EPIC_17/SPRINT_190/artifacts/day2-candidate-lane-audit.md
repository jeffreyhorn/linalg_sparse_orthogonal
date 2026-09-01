# Sprint 190 Day 2: Candidate Lane Audit

## Purpose

Audit the selected report freshness candidates and choose the best bounded
Windows promotion candidate, while preserving an explicit fallback path if the
Windows-safe generator probe is not feasible.

## Audited Inputs

| Input | Finding |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | Selected rows currently cover one oracle lane, five selected comparison lanes, and one benchmark lane. None list `windows`. |
| `scripts/run_external_comparison.py` | All selected comparison lanes share one generated C project-probe path that defaults to `cc`, calls `make` for a missing static library, links `build/libsparse_lu_ortho.a`, passes `-lm`, and executes an extensionless temporary probe. |
| `scripts/validate_corpus_schema.py` | Manifest schema validates hosted metadata completeness and artifact/platform cardinality, but does not yet encode a positive one-row Windows promotion allowlist. |
| `tests/test_selected_report_targets_manifest.py` | Current test helper asserts no selected row lists `windows` while the Sprint 182 deferral record is active. |
| `tests/test_selected_comparison_workflow.py` | Current workflow guard requires Windows report freshness to remain formally deferred and rejects selected report freshness commands/artifact names in Windows CI. |
| `scripts/validate_windows_powershell.py` | Current Windows workflow guard forbids selected report generation and upload names while the deferral remains active. |

## Shared Comparison Generator Blockers

All selected comparison candidates inherit the same Windows blockers:

- default compiler is `cc`;
- missing library build fallback runs `make`;
- default library is the Unix archive `build/libsparse_lu_ortho.a`;
- generated probe compile command uses `-std=c99`, `-I`, `-lm`, and `-o`;
- temporary executable path has no `.exe` suffix;
- project probe execution assumes direct execution of that extensionless path.

The comparison candidates remain the best promotion family, but none can be
promoted without a Windows-safe CMake/MSVC probe path.

## Candidate Matrix

| Candidate | Expected rows | Runtime risk | Artifact risk | Claim risk | Result |
| --- | ---: | --- | --- | --- | --- |
| Cholesky SPD tridiagonal comparison | 6 | Medium | Low-medium | Low | Select as primary. |
| LU nonsymmetric square comparison | 6 | Medium | Low-medium | Medium-low | Keep as first fallback. |
| QR minimum-norm comparison | 6 | Medium | Low-medium | Medium | Keep as second fallback. |
| QR compatible least-squares comparison | 6 | Medium | Low-medium | Medium | Keep as third fallback. |
| Partial-SVD diag6 k2 comparison | 10 | Medium-high | Low-medium | Medium-high | Keep as late fallback. |
| Canonical benchmark freshness | 1 | High | Medium | High | Reject for first promotion. |
| QR/partial-SVD oracle freshness | 52 | High | Medium-high | Medium-high | Reject for first promotion. |

## Primary Candidate

`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` is the Day 2 selected candidate.

It is the best fit because it has:

- one exact generator command:
  `python3 scripts/run_external_comparison.py --target cholesky-spd-tridiag-5`;
- six expected rows;
- one report directory: `build/comparison/cholesky_spd_tridiag_5/`;
- the standard six-file comparison bundle;
- a narrow claim scope: fixture-local Cholesky SPD tridiagonal solve
  comparison only;
- explicit non-claims that already reject broad correctness, fill superiority,
  package/ABI support, performance superiority, and state-of-the-art status.

## Rejected First-Promotion Candidates

The benchmark lane has only one expected row, but it is not the safest first
promotion because it is Bash/Makefile-oriented and performance-adjacent.

The oracle lane is rejected for first promotion because it is broad, includes
52 rows, depends on a Makefile freshness wrapper, and would expand the Windows
decision beyond one narrow lane.

Partial-SVD is kept behind the smaller comparison candidates because its ten
rows and vector/subspace diagnostics create more claim-boundary surface.

## Promotion Path

Day 3 should attempt to prove a Windows-safe path for the Cholesky candidate:

1. Avoid `make` in the Windows generator path.
2. Build the generated probe through CMake/MSVC or a reviewed Windows compiler
   command surface.
3. Link against the reviewed static `.lib` shape or another reviewed Windows
   build artifact.
4. Add `.exe`-aware temporary executable handling.
5. Define exact Windows workflow job, artifact name, required files, expected
   rows, and row IDs.
6. Convert current no-Windows guards into an exact Cholesky allowlist while
   rejecting all other selected Windows report freshness lanes.

## Fallback Deferral Path

If the Cholesky candidate cannot satisfy the Day 3 feasibility gates, Sprint
190 should renew the formal deferral with stronger evidence instead of
partially promoting report freshness.

The renewed deferral should name these blockers:

- selected comparison generation still has Unix compile/link assumptions;
- no reviewed Windows CMake/MSVC generated-probe mode exists;
- no `.exe`-aware generated comparison probe path has been implemented;
- no exact Windows selected comparison artifact upload has been proven;
- existing guards are intentionally no-promotion guards and need allowlist work
  before any Windows row can be added to the manifest.

## Day 2 Decision

Proceed to Day 3 with `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` as the only promotion
candidate under active investigation. Keep LU as the first fallback if the
Cholesky-specific solve path adds unexpected risk. Do not pursue benchmark or
oracle promotion unless comparison promotion is conclusively blocked and a
replacement candidate can still satisfy the one-lane Sprint 190 acceptance
gate.

## Validation

Day 2 performed source inspection and documentation updates only. No `.c` or
`.h` files were modified, so `make format && make lint && make test` is not
required.

Run `git diff --check` after this artifact is added.
