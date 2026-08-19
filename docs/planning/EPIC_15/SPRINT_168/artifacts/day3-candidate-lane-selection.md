# Sprint 168 Day 3: Candidate Lane Selection

## Purpose

Day 3 selects the exact hosted performance publication candidate for Sprint
168. The selection is based on the canonical maintained benchmark surface
inventoried on Day 2 and the Sprint 167 claim gates for `G167-01`.

This artifact selects a lane for Day 4 local runtime suitability and Day 5
metadata design. It does not yet claim hosted performance evidence.

## Selection Criteria

| Criterion | Meaning |
| --- | --- |
| Runtime suitability | The command is likely small enough for hosted CI after Day 4 dry-run confirmation. |
| Output stability | The report emits stable row identity and metadata fields suitable for freshness checks. |
| User value | The lane represents an adoption-relevant workflow users may care about. |
| Methodology clarity | The command, fixture, repeat count, backend context, and interpretation can be described without ambiguity. |
| Claim-risk containment | The lane can be documented without implying portable superiority, broad backend superiority, external parity, broad platform parity, or state-of-the-art performance. |

## Candidate Scoring

Scores use `High`, `Medium`, and `Low`, with notes focused on Sprint 168
hosted publication suitability rather than absolute benchmark importance.

| Candidate | Command shape today | Runtime suitability | Output stability | User value | Methodology clarity | Claim-risk containment | Selection result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bench_refactor_csc` | `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` through `make bench-canonical-report` | Medium pending Day 4 dry run | High | High | High | High | Selected primary lane |
| `bench_chol_csc` | `build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1` through `make bench-canonical-report` | Medium pending Day 4 dry run | High | High | Medium | Medium | Alternate only |
| `bench_iterative_reuse` | Default workload through `make bench-canonical-report` | High likely | Medium | Medium | Medium | Medium | Defer as alternate |
| `bench_eigs_reuse` | Default workload through `make bench-canonical-report` | Medium pending dry run | Medium | Medium | Medium | Medium | Defer as alternate |
| `make performance-sentinels` | Local sentinel bundle with wall-check and advisory rows | Medium | Medium | Medium | Medium | Low for Sprint 168 | Not selected |
| `make bench-fast` | Existing CI smoke subset | High | Low for publication | Medium | Low | Low | Not selected |

## Selection Decision

Sprint 168 selects the **direct repeated-run CSC factorization performance
publication lane** centered on:

- benchmark binary: `build/bench_refactor_csc`;
- canonical command owner: `make bench-canonical-report`;
- selected report artifact: `build/bench-reports/canonical/bench_refactor_csc.csv`;
- selected metadata artifacts:
  - `build/bench-reports/canonical/index.tsv`;
  - `build/bench-reports/canonical/manifest.txt`;
- fixture: `tests/data/suitesparse/nos4.mtx`;
- repeat semantics: `--repeat 1`;
- scenario: SPD / Cholesky repeated-run `analyze once, refactor many`;
- initial hosted platform lane: Linux hosted CI;
- initial support tier target: hosted selected performance publication, not
  broad performance support.

The chosen lane follows the Sprint 167 recommendation because it already has a
fixed fixture, fixed repeat count, canonical-report ownership, adoption value
for repeated direct factorization workflows, and a clear documentation
boundary.

## Exact Scope For Day 4

| Field | Selected Day 3 scope |
| --- | --- |
| Benchmark family | Direct repeated-run CSC factorization |
| Benchmark binary | `build/bench_refactor_csc` |
| Local generator | `make bench-canonical-report` |
| Focused command equivalent | `build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1` |
| Fixture | `tests/data/suitesparse/nos4.mtx` |
| Matrix scope | Single SuiteSparse `nos4.mtx` fixture |
| Repeat semantics | `configured_repeat_1` / `--repeat 1` |
| Thread setting | Record `OMP_NUM_THREADS`; leave unset unless Day 4/Day 5 requires an explicit CI value. |
| Build mode | Record detected or explicit `SPARSE_CANONICAL_BUILD_MODE`; likely `serial` for first hosted lane unless CI sets otherwise. |
| Platform/toolchain | Linux hosted CI runner and default C compiler for the workflow selected on Day 9. |
| Report directory | `build/bench-reports/canonical/` |
| Selected CSV | `bench_refactor_csc.csv` |
| Metadata files | `index.tsv`, `manifest.txt` |
| Runtime budget | To be measured on Day 4 before CI wiring. |
| Freshness owner | To be designed on Day 7 after metadata design. |

## Why Alternatives Are Deferred

| Alternate | Deferral reason |
| --- | --- |
| `bench_chol_csc` | Valuable direct backend comparison, but documentation can more easily drift into backend-superiority wording. Keep as an alternate if `bench_refactor_csc` runtime is unsuitable. |
| `bench_iterative_reuse` | Likely CI-friendly, but default workload and convergence interpretation need more solver-specific caveats before first hosted publication. |
| `bench_eigs_reuse` | Useful eigensolver workflow evidence, but backend-selection caveats make it a less clean first hosted performance lane. |
| `make performance-sentinels` | Mixes the existing wall-check hard gate with advisory threshold-free rows. Preserve it as local sentinel evidence for Sprint 169 methodology work. |
| `make bench-fast` | Already hosted as smoke coverage in Linux CI; it lacks the report metadata and methodology framing required for publication. |

## Explicit Out-Of-Scope Claims

The selected `bench_refactor_csc` hosted lane will not support claims of:

- portable performance superiority;
- broad backend superiority;
- broad Cholesky, LDLT, or direct-solver performance;
- broad SuiteSparse corpus performance;
- external-library performance parity or superiority;
- release benchmark proof;
- cross-platform performance parity;
- OpenMP speedup;
- state-of-the-art sparse linear algebra performance;
- solver correctness beyond the selected benchmark residual checks.

## Claim-Safe Wording Draft

Future docs may describe the selected lane only after implementation and
validation with wording like:

> The hosted performance publication lane records a methodology-bound
> `bench_refactor_csc` measurement for `tests/data/suitesparse/nos4.mtx` with
> `--repeat 1` on the named Linux CI workflow. It is report freshness and
> interpretation evidence for that selected workflow, not a portable
> performance guarantee or backend superiority claim.

## Day 4 Handoff

Day 4 should build and run:

```sh
make bench-canonical-report
```

and, if needed for focused timing diagnostics:

```sh
build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1
```

Day 4 should record:

- wall-clock runtime for the full canonical report and selected focused
  command when practical;
- output size for `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`;
- whether generated row fields are stable enough for freshness checks;
- whether Day 5 must add explicit CI metadata such as runner, compiler, build
  flags, thread count, or build-mode overrides;
- whether the selected lane remains suitable or should fall back to
  `bench_chol_csc` or another narrower target.

## Validation Notes

Day 3 changed only Sprint 168 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| One hosted performance publication candidate is selected. | Complete | `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1` is selected. |
| Selected evidence boundary is narrow and reviewable. | Complete | Scope table defines command, fixture, repeat semantics, platform assumption, report path, and metadata files. |
| Portable superiority and broad backend claims remain non-claims. | Complete | Out-of-scope claims and claim-safe wording preserve the boundary. |
