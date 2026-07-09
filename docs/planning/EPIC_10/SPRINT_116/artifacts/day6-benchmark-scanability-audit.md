# Day 6 Benchmark Scanability Audit

## Purpose

Day 6 reviews `benchmarks/README.md` for adoption-facing scanability, live
lane names, report mechanics, interpretation entry points, and performance
claim boundaries. Day 7 should apply only bounded documentation cleanup if it
improves scanability or claim accuracy.

## Inputs Reviewed

| Input | Reviewed for |
|---|---|
| `benchmarks/README.md` | Result interpretation, workflow groups, lane names, report bundle mechanics, CSV schema, CLI sections, performance-claim boundaries. |
| `Makefile` | Live benchmark/report target names and whether README target references are current. |
| `README.md` | Top-level benchmark wording and handoff into `benchmarks/README.md`. |
| `docs/maintainer_guide.md` | Reviewed/staged benchmark-governance interpretation and non-claim boundaries. |

## Live Target Check

| Target referenced by benchmark docs | Makefile status | Notes |
|---|---|---|
| `make tooling-build` | Present | Compile-only benchmark/example gate. |
| `make bench-build` | Present | Focused benchmark compile-only subset. |
| `make bench-fast` | Present | Bounded runtime lane. |
| `make bench-reorder-sprint86` | Present | Historical target name for current bounded two-fixture reorder slice. |
| `make bench-canonical-report` | Present | Threshold-free canonical report bundle. |
| `make performance-sentinels` | Present | Local sentinel bundle; hard timing gate remains `wall-check`. |
| `make large-matrix-guardrails` | Present | Guardrail bundle with reviewed and supplemental lanes. |
| `make bench-suitesparse` | Present | Smoke target for `bench_main`. |
| `make bench-eigs` | Present | Smoke target for `bench_eigs`. |
| `make wall-check` | Present | Narrow thresholded regression gate. |

No stale benchmark target names were found in the Day 6 audit.

## Report Mechanics Notes

| Report surface | Current documentation state | Day 7 decision |
|---|---|---|
| `bench-canonical-report` | Documents output directory, `manifest.txt`, `index.tsv`, label env var, threshold-free semantics, and non-portable timing boundary. | No claim edit required. |
| `performance-sentinels` | Documents output directory, sentinel TSV, manifest, wall-check raw output, Cholesky CSC context, skip semantics, and local-evidence boundary. | No claim edit required. |
| `large-matrix-guardrails` | Documents output directory, reviewed lanes `G1`-`G4`, supplemental lanes `S1`/`S2`, manifest, structural-test ownership, and non-portable max-RSS boundary. | No claim edit required. |
| `bench_refactor` / `bench_refactor_csc` | Documents public repeated-run direct measurement role and test-owned error/property boundaries. | No claim edit required. |
| `bench_chol_csc` | Documents backend-aware measurement role and test-owned callback/oracle boundaries. | No claim edit required. |
| `bench_iterative_reuse` / `bench_eigs_reuse` | Documents narrow public-handle measurement roles and explicit exclusions. | No claim edit required. |
| `bench_main` / `bench_eigs` CLI sections | Live enough for command discovery and CSV interpretation. | No claim edit required. |

## Performance-Claim Audit

| Wording class | Finding | Disposition |
|---|---|---|
| Universal performance claims | The guide explicitly says benchmarks do not prove portable performance across machines, compilers, OSes, BLAS/dense backends, OpenMP runtimes, thread counts, corpora, or build options. | Keep. |
| Timing gates | The guide keeps `bench-canonical-report` threshold-free and `wall-check` as the narrow thresholded gate. | Keep. |
| Report bundles | The guide treats reports as artifact-friendly local evidence, not broad performance proof. | Keep. |
| Speedup columns | `speedup` fields are described as row values inside specific benchmark schemas, not broad claims. | Keep. |
| "Strongest" wording | The guide says refactor benchmarks are the strongest benchmark-side adoption surfaces for repeated-run direct lifecycle, not a universal performance claim. | Keep. |

No universal performance claim or unsupported benchmark-support claim was
found.

## Scanability Findings

| Area | Finding | Day 7 decision |
|---|---|---|
| Top-level interpretation | Strong. The guide opens with result-reading rules and adoption handoff table. | Keep. |
| Navigation | Weak. The file is long and has useful sections, but no compact quick-navigation table near the top. | Add a small quick-navigation table on Day 7. |
| Workflow grouping | Strong. The guide separates one-shot, direct repeated-run, iterative reuse, and eigensolver reuse groups. | Keep. |
| Maintained category split | Strong. Canonical, runtime, and exploratory lanes are explicitly separated. | Keep. |
| Report bundle mechanics | Detailed and accurate, but users must scroll to find the relevant bundle section. | Quick navigation should link to report bundle sections. |
| CLI sections | Acceptable. `bench_main` and `bench_eigs` have dedicated CLI sections. | Keep. |

## Day 7 Edit Checklist

| Item | Edit decision | Rationale |
|---|---|---|
| Add compact quick-navigation table near the top of `benchmarks/README.md` | Edit | Improves scanability for adoption-facing readers without changing benchmark commands, report semantics, or performance claims. |
| Leave benchmark target names unchanged | No edit | Referenced Makefile targets are live. |
| Leave performance caveats unchanged | No edit | They already tie results to local measured evidence and reject portable guarantees. |
| Leave report bundle semantics unchanged | No edit | Existing wording correctly separates threshold-free reports, hard gates, skips, reviewed lanes, and supplemental lanes. |
| Leave benchmark workflow groups unchanged | No edit | Current grouping matches the documented maintained category split. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Benchmark docs have a clear edit or no-edit path | Complete. |
| Performance language remains tied to measured local evidence | Complete. |
| No benchmark workflow changes are implied | Complete. |

## Validation Notes

- Day 6 changed Sprint 116 planning documentation only.
- `benchmarks/README.md`, `Makefile`, `README.md`, and
  `docs/maintainer_guide.md` were inspected but not edited.
- No `.c` or `.h` files were modified.
