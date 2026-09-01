# Sprint 192 Day 3: Methodology Contract

## Contract Summary

Day 3 defines the methodology contract for the selected Sprint 192 performance
lane:

```text
SRT-BENCH-REFACTOR-CSC-NOS4
bench_refactor_csc
tests/data/suitesparse/nos4.mtx --repeat 1
```

The lane remains methodology-bound and threshold-free unless the Day 9
regression policy explicitly changes that decision. It is selected hosted
freshness evidence for one benchmark row, not a pass/fail timing gate or
portable performance claim.

## Selected Row Identity

| Field | Required value |
| --- | --- |
| `surface` | `canonical` |
| `category` | `measurement` |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `report_family` | `benchmark` |
| `status` | `measurement` |
| `fixture_or_workload` | `nos4.mtx` |
| `matrix_size` | `n=100` |
| `repeat_semantics` | `configured_repeat_1` |

These values identify the one selected benchmark lane. If any value changes,
the selected target manifest, freshness checker, workflow summary, and docs
must be reviewed together.

## Required Methodology Metadata

The selected row must include every field below in `index.tsv` and the
corresponding manifest value where listed.

| Field | Source | Required behavior | Manifest agreement |
| --- | --- | --- | --- |
| `report_label` | `BENCH_CANONICAL_REPORT_LABEL` or default | Nonempty; hosted mode must not be `unlabeled`. | Yes |
| `generated_at_utc` | report generator timestamp | Nonempty ISO timestamp in `YYYY-MM-DDTHH:MM:SSZ` form. | No |
| `git_commit` | `git rev-parse --short HEAD` | Nonempty; `unknown` only if Git metadata is unavailable. | Yes |
| `git_branch` | `git rev-parse --abbrev-ref HEAD` | Nonempty; `detached` allowed for detached HEAD. | Yes |
| `platform` | `uname -a` or hosted equivalent | Nonempty platform context. | Yes |
| `compiler` | `${CC:-cc} --version` first line | Nonempty compiler context. | Yes |
| `runner_context` | `SPARSE_CANONICAL_RUNNER_CONTEXT` or default | Nonempty; hosted mode must not be `local`. | Yes |
| `build_flags` | `SPARSE_CANONICAL_BUILD_FLAGS` or default | Nonempty; hosted mode must not be `not_recorded`. | Yes |
| `cpu_model` | `SPARSE_CANONICAL_CPU_MODEL` or default | Nonempty; `unknown` remains acceptable context. | Yes |
| `build_mode` | override or OpenMP runtime detection | Nonempty build-mode context. | Yes |
| `omp_num_threads` | `OMP_NUM_THREADS` or default | Nonempty; `unset` is explicit context. | Yes |
| `support_tier` | generator environment / selected mode | Hosted selected row must be `hosted_selected`; local mode may be local shape evidence. | Yes |
| `claim_boundary` | generator environment / selected mode | Hosted selected row must be `hosted_selected_threshold_free`. | Yes |
| `baseline` | generator constant | Required `n/a` until a reviewed threshold policy exists. | Yes |
| `threshold` | generator constant | Required `n/a` until a reviewed threshold policy exists. | Yes |
| `warmup` | generator constant | Required `none_configured`; must be read as a limitation. | Yes |
| `variance` | generator constant | Required `not_computed_single_sample`; must be read as a limitation. | Yes |
| `matrix_size` | selected workload contract | Required `n=100` for the selected row. | Yes, as `selected_matrix_size` |
| `methodology_notes` | generator environment / default | Must contain `not_portable_performance_claim`. | Yes |

## Advisory Context Fields

These fields are required to be present and nonempty, but they are context, not
proof of portability or stable machine class:

| Field | Interpretation |
| --- | --- |
| `platform` | Identifies the machine/runtime context that produced the row; it does not create broad platform parity. |
| `compiler` | Identifies the compiler string for the run; it does not prove compiler-wide performance behavior. |
| `cpu_model` | Records CPU context when available. GitHub-hosted CPU assignment can vary, and `unknown` remains acceptable. |
| `build_mode` | Distinguishes serial/OpenMP context when detectable or overridden; it is not an OpenMP speedup claim. |
| `omp_num_threads` | Records the thread setting used for the run; `unset` is explicit context. |
| `report_label` | Names a generated report for review; it is not evidence by itself. |

## Sample, Warmup, Variance, Baseline, and Threshold Policy

Day 3 keeps the selected lane threshold-free:

| Field | Required value | Meaning |
| --- | --- | --- |
| `repeat_semantics` | `configured_repeat_1` | The selected benchmark command uses one configured repeat. |
| `warmup` | `none_configured` | No warmup policy is configured. Do not describe the row as warmup-controlled. |
| `variance` | `not_computed_single_sample` | No distribution or variance estimate is computed. Do not describe the row as statistically summarized. |
| `baseline` | `n/a` | No timing baseline is selected for this canonical freshness lane. |
| `threshold` | `n/a` | No timing threshold is selected for this canonical freshness lane. |
| `status` | `measurement` | The row reports measurement context and freshness, not pass/fail timing status. |

Day 9 may revisit whether a separate conservative sentinel is justified. Until
then, any selected hosted performance claim must remain threshold-free.

## Local and Hosted Mode Semantics

| Mode | Required selected-row support tier | Required selected-row claim boundary | Evidence meaning |
| --- | --- | --- | --- |
| Local checker mode | `local_only` or `hosted_selected` | `local_threshold_free` or `hosted_selected_threshold_free` | Shape, artifact, and metadata validation. Local results are not hosted evidence. |
| Hosted checker mode | `hosted_selected` | `hosted_selected_threshold_free` | Hosted selected freshness evidence when produced by the reviewed Linux CI lane. |

Unselected canonical rows must remain `support_tier=local_only` and
`claim_boundary=local_threshold_free` in both modes.

## Artifact Contract

The selected manifest requires:

- `bench_refactor_csc.csv`;
- `index.tsv`;
- `manifest.txt`.

The current hosted workflow uploads the full canonical bundle:

- `bench_refactor_csc.csv`;
- `bench_chol_csc.csv`;
- `bench_iterative_reuse.csv`;
- `bench_eigs_reuse.csv`;
- `index.tsv`;
- `manifest.txt`.

Day 3 treats the additional canonical CSV uploads as contextual artifacts, not
selected hosted performance rows. Day 7/Day 8 must decide whether to keep this
contextual bundle with clearer guards or narrow uploads to manifest-required
files only.

## Validation Behavior

The dedicated selected benchmark freshness checker must fail when:

- required artifacts are missing;
- `index.tsv` is empty, malformed, has duplicate headers, has missing columns,
  or has row-width mismatches;
- the selected row is missing or duplicated;
- required selected metadata is empty;
- selected identity fields do not match the manifest/checker contract;
- hosted mode uses `runner_context=local`, `build_flags=not_recorded`,
  `report_label=unlabeled`, wrong support tier, or wrong claim boundary;
- unselected rows are promoted to hosted support tier or hosted claim boundary;
- manifest fields disagree with selected row fields;
- `methodology_notes` omits `not_portable_performance_claim`.

`scripts/normalize_report_index.py --family benchmark --check-freshness`
currently reports benchmark rows as advisory local measurement freshness. Day 6
must decide whether to harden normalized benchmark freshness by reusing the
selected benchmark contract or to keep the dedicated checker as the authority.

## Allowed Claim

Allowed hosted claim shape:

> The selected `bench_refactor_csc` canonical benchmark row for
> `tests/data/suitesparse/nos4.mtx --repeat 1` has fresh methodology-bound
> hosted Linux evidence with recorded runner, compiler, build, CPU, thread,
> fixture, repeat, warmup, variance, baseline, threshold, and non-portable
> performance context.

Required paired non-claim shape:

> This is threshold-free selected-row freshness evidence, not a timing
> threshold, portable performance guarantee, architecture-independent speedup,
> algorithmic superiority claim, broad benchmark publication, package proof,
> ABI proof, external-library parity claim, release benchmark result, or
> state-of-the-art performance claim.

## Day 4 Implementation Checklist

Day 4 should inspect whether current generator and fixture ownership fully
matches this contract, especially:

1. whether `bench_refactor_csc.csv` exposes enough row-level workload context
   for the selected methodology;
2. whether `matrix_size=n=100` should remain a selected-row constant or be
   parsed from the benchmark CSV;
3. whether full canonical bundle uploads are acceptable contextual evidence;
4. whether control-character validation covers every environment-provided
   metadata field;
5. whether normalized benchmark freshness should call or mirror the dedicated
   selected benchmark checker.
