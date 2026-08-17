# Sprint 163 Day 4 Methodology Contract

## Purpose

This contract defines the fields, row states, threshold semantics, variance
rules, and public caveats required before Sprint 163 changes any report schema,
script output, documentation, or generated report interpretation. It applies to
the Day 3 selected surface:

- `make bench-canonical-report`
- `make performance-sentinels`

## Row Classes

| Class | Meaning | Selected Rows | Publication Status |
| --- | --- | --- | --- |
| Published threshold-free local measurement | Timing or metric row that may be surfaced with full methodology context but no pass/fail or superiority meaning. | Canonical report rows for `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, `bench_eigs_reuse`; sentinel S2 and S3 rows. | Publishable once required methodology fields are present. |
| Published thresholded local gate | Timing row with a local baseline and threshold that may fail the selected command. | Sentinel S5 wall-check rows. | Publishable only with baseline, threshold, support tier, and local gate caveats. |
| Supplemental | Optional local row that can help maintainers but is not selected as Sprint 163 publication evidence. | Current examples: `bench-fast`, supplemental large-matrix guardrail rows. | Do not publish as Sprint 163 evidence unless later reselected with this contract. |
| Advisory | Navigation, freshness, or contextual row that helps locate evidence but is not proof. | Normalized report-index rows, freshness diagnostics. | May link selected evidence but must not be described as release proof. |
| Local-only raw artifact | Generated CSV, TSV, manifest, or raw text under ignored `build/` paths. | Raw canonical CSVs, raw sentinel CSVs, `wall_check.txt`. | Regenerate from maintained commands; do not hand-edit or treat as source-controlled timing proof. |
| Deferred | Row or command excluded from Sprint 163 publication. | Full `make bench`, `make bench-eigs`, `make large-matrix-guardrails`, package rows, correctness rows. | Keep out of Sprint 163 performance claims. |

## Required Fields By Class

### Published Threshold-Free Local Measurement

Every published threshold-free row must provide or inherit:

- `report_family` or equivalent family identifier;
- row or artifact identity;
- category or support tier;
- claim boundary;
- generation command;
- generated artifact path;
- generated timestamp in UTC;
- git commit;
- git branch;
- platform or operating system string;
- compiler string including version when available;
- build mode;
- `OMP_NUM_THREADS` or explicit `unset`;
- benchmark binary or script owner;
- fixture, matrix, workload, or default workload identity;
- repeat count or an explicit "single configured repeat" caveat;
- metric name;
- metric value;
- threshold state set to `n/a`;
- baseline state set to `n/a`;
- backend/runtime request where relevant;
- selected backend where relevant;
- fallback state where relevant;
- notes preserving threshold-free and local-only interpretation.

### Published Thresholded Local Gate

Every published thresholded local gate row must provide:

- all applicable threshold-free local measurement fields;
- `status` with pass, fail, or skip semantics;
- support tier identifying the row as thresholded;
- claim boundary identifying the row as a local wall gate;
- baseline value;
- threshold multiplier or threshold value;
- raw gate artifact;
- baseline provenance;
- fixture identity;
- metric name and measured value;
- failure handling that preserves nonzero command exit behavior.

### Supplemental Rows

Supplemental rows must provide:

- command;
- generated artifact;
- platform/compiler/build/thread context when available;
- explicit supplemental status;
- explanation of why the row is not selected publication evidence.

### Advisory Rows

Advisory rows must provide:

- source report family;
- source command or remediation command;
- row identity fields used for navigation;
- freshness or indexing status;
- explicit statement that advisory rows do not convert local generated outputs
  into release proof.

### Local-Only Raw Artifacts

Local-only raw artifacts must provide:

- generating command;
- output directory;
- artifact filename;
- parent manifest or index when available;
- instruction to regenerate instead of editing by hand.

## Required Methodology Dimensions

Sprint 163 report and documentation updates must preserve these dimensions:

| Dimension | Required Treatment |
| --- | --- |
| Platform / OS | Record the local platform string or CI runner context. Do not generalize to unsupported platforms. |
| Compiler / version | Record compiler string when available. Treat compiler differences as local methodology context. |
| Build flags / mode | Record serial/OpenMP or explicit override. Do not claim OpenMP speedup portability. |
| Thread count | Record `OMP_NUM_THREADS` or `unset`. Treat unset as a meaningful context value. |
| Backend/runtime settings | Record dense-backend requests, selected backend, fallback, dense-kernel, and panel-solver fields when rows expose them. |
| Fixture / workload | Name matrix, fixture, scenario, or default workload. Unknown fixture identity blocks publication. |
| Matrix size | Include when emitted by the source row; otherwise document the fixture identity and record the missing-size gap for Day 5. |
| Repeat count | Name the configured repeat count. Current selected script commands use `--repeat 1` for selected matrix-backed rows unless a benchmark has its own default. |
| Warmup | Record as `not recorded` until a script adds a warmup field. Do not imply warmup exists. |
| Variance | Record as `not recorded` unless multiple repeats or variance fields are explicitly added. |
| Threshold | Required only for S5 gate rows; threshold-free rows must use `n/a`. |
| Date / provenance | Record UTC generation time, git commit, git branch, command, artifact, and report label where available. |

## Row-State Semantics

| State | Meaning | Publication Handling |
| --- | --- | --- |
| `present` / `report` | Row was generated successfully as threshold-free context. | Publish only with threshold-free caveat and full methodology context. |
| `pass` | Thresholded gate row passed its configured local threshold. | Publish only for S5-style local gate rows with baseline and threshold visible. |
| `fail` | Thresholded gate row exceeded its configured local threshold or selected command failed. | Preserve failure; do not downgrade to advisory evidence. |
| `skip` | Row could not run because a binary, fixture, baseline, or opt-in was absent. | Scope information only; not passing evidence. |
| `missing` | Required artifact or row is absent. | Blocks publication for that row. |
| `stale` | Generated artifact does not match the current required source or freshness check. | Blocks publication until regenerated or explicitly deferred. |
| `malformed` | Row exists but lacks required columns, parseable values, or valid TSV/CSV structure. | Blocks publication and requires script/schema repair. |
| `deferred` | Row is intentionally outside Sprint 163 selection. | May be named in deferred register only. |
| `local-only` | Raw generated artifact under ignored build output. | May support selected rows but should not be hand-edited or cited alone. |

## Threshold Rules

- S5 wall-check rows are the only selected hard timing gates.
- S5 rows must keep measured value, baseline, threshold, status, fixture,
  metric, command, and artifact visible together.
- Current S5 thresholds are local wall-check thresholds, not portable
  performance expectations.
- Canonical rows are threshold-free and must not gain pass/fail wording.
- S2 and S3 sentinel rows are threshold-free context and must not be described
  as passing, failing, or proving backend superiority.
- Skipped rows are scope signals, not passing evidence.

## Variance And Repeat Rules

- Current selected matrix-backed commands use `--repeat 1` where the script
  sets that flag.
- Rows without emitted variance must be described as single-run or configured
  local measurements.
- A future script change may add repeat count, sample count, min/max, mean,
  standard deviation, or confidence interval fields, but no publication may
  imply variance evidence before those fields exist.
- Do not compare rows across machines, compilers, OpenMP settings, backend
  settings, or fixtures unless the comparison text explicitly constrains those
  dimensions.

## Public Caveat Wording

Use this wording, or equivalent wording with the same constraints, in public
documentation and report interpretation:

> These rows are methodology-bound local measurement artifacts. They record the
> command, fixture, artifact, commit, branch, platform, compiler, build mode,
> thread setting, and backend context available at generation time. They are
> not portable performance guarantees, state-of-the-art claims, broad platform
> parity claims, package evidence, package-manager claims, shared-library or ABI
> guarantees, runtime-loader claims, external-library parity claims, OpenMP
> speedup claims, or backend superiority claims.

For S5 thresholded rows, add:

> S5 is the existing local wall-check regression gate. Its status is meaningful
> only with the recorded baseline, threshold, fixture, command, and machine
> context. It is not a portable timing promise.

For S2/S3 threshold-free sentinel rows, add:

> S2 and S3 rows are threshold-free local backend-context rows. They preserve
> backend request, selected backend, fallback, dense-kernel, and panel-solver
> context where emitted, but they do not pass or fail and do not prove backend
> superiority.

## Unsupported Claim Blockers

Any report, script, or documentation change must be revised if it claims or
implies:

- portable performance superiority;
- state-of-the-art performance;
- broad platform performance parity;
- package proof as performance proof;
- package-manager support;
- shared-library or dynamic ABI support;
- runtime-loader behavior;
- external-library parity;
- OpenMP speedup portability;
- backend superiority;
- generated report freshness as release proof;
- local skipped rows as passing evidence.

## Day 5 Gap-Analysis Checklist

Day 5 should compare the current selected scripts and docs against this
contract and identify any missing fields, stale wording, or malformed row
semantics before implementation.

Minimum checks:

- canonical `index.tsv` fields versus required published threshold-free fields;
- canonical `manifest.txt` caveat wording versus public caveat wording;
- sentinel `sentinels.tsv` fields versus S5/S2/S3 requirements;
- sentinel `manifest.txt` caveat wording versus S5 and S2/S3 caveats;
- README and `benchmarks/README.md` wording versus unsupported claim blockers;
- normalized report-index preservation of support tier and claim boundary.

## Completion Check

- Exact methodology fields are defined before report edits.
- Gate rows and threshold-free publication rows are distinguished.
- Missing, stale, malformed, skipped, deferred, failed, advisory, and local-only
  semantics are explicit.
- Unsupported performance claims are blocked before implementation work begins.
