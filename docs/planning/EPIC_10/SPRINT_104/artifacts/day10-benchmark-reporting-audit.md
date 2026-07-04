# Sprint 104 Day 10 Benchmark Reporting Audit

## Purpose

Day 10 audits benchmark-facing reporting surfaces before changing public
documentation language. The audit reconciles benchmark docs, maintainer docs,
scripts, Makefile targets, and README summaries with the Day 3 runtime
contract and Day 8 sentinel design.

The target outcome is a Day 11 documentation update plan that keeps local
timing, optional acceleration, OpenMP runtime context, and CI-reviewed
sentinels separated from unsupported portable performance claims.

## Authoritative Contracts

| source | relevant rule |
|---|---|
| Day 3 runtime contract | builtin dense kernels are the portable baseline; optional dense backend requests are best-effort and fall back to builtin when unavailable or unsuitable |
| Day 3 OpenMP contract | serial behavior remains the reference behavior; interpreted OpenMP timing must disclose runtime thread settings |
| Day 3 observability split | public API telemetry, benchmark context, test ownership, and maintainer diagnostics must not be collapsed into one broad product claim |
| Day 8 sentinel design | sentinels detect local regressions; only already-justified lanes should hard-fail; residual and agreement columns remain context unless tests own the correctness claim |
| Day 9 sentinel batch | `make performance-sentinels` currently combines S5 hard wall-check rows with S2 threshold-free Cholesky CSC report rows |

## Reporting Surface Inventory

| surface | current role | audit result | Day 11 action |
|---|---|---|---|
| `README.md` performance section | compact top-level performance summary and route to benchmark docs | mostly aligned; names local benchmark rows as non-portable, but does not mention the new `performance-sentinels` target | add one bounded sentence for the sentinel bundle and retain the route to `benchmarks/README.md` |
| `README.md` feature list | first-contact feature and OpenMP summary | mostly aligned; OpenMP and eigensolver speedup examples are useful but should avoid sounding like broad runtime scalability or portable timing guarantees | keep concise feature bullets, but route interpreted timing and benchmark claims to benchmark docs |
| `benchmarks/README.md` category split | benchmark command groups, CSV fields, and interpretation | aligned for canonical/report lanes; missing Day 9 sentinel bundle documentation | add a local performance-sentinel subsection that names S5 as hard gate and S2 as threshold-free report context |
| `benchmarks/README.md` Cholesky CSC section | backend-aware Cholesky benchmark schema and caveats | aligned with descriptor truth; already names builtin dense-kernel descriptor and avoids portability claims | preserve wording; add cross-reference from sentinel section when S2 is documented |
| `benchmarks/README.md` canonical report section | threshold-free maintained-surface report description | aligned conceptually; wording is good, but generated `index.tsv` category currently says `proof`, which is stronger than the docs' threshold-free local-report contract | either change the script category label later or explicitly document that `proof` means proof-owner surface identity, not timing proof |
| `docs/maintainer_guide.md` benchmark governance | authoritative owner split for canonical, runtime, and exploratory benchmark lanes | aligned; missing `performance-sentinels` because Day 9 added a new target | add governance row for the sentinel bundle and keep hard-fail scope limited to wall-check |
| `docs/maintainer_guide.md` backend-aware surface section | Cholesky/LDLT backend semantics and proof ownership | aligned; fallback and builtin truth are present | no immediate change beyond possible cross-reference to sentinel/benchmark docs |
| `docs/algorithm.md` performance regression gates | historical algorithm/performance notes and wall-check explanation | aligned for `wall-check`; missing the newer Day 9 sentinel bundle | add a short note that `performance-sentinels` wraps wall-check plus threshold-free Cholesky CSC context, without widening the algorithm claims |
| `Makefile` benchmark targets | executable target contracts | aligned; comments for `performance-sentinels` accurately limit hard-fail behavior to wall-check | no Day 11 code change needed unless docs wording exposes a mismatch |
| `scripts/bench_canonical_report.sh` | threshold-free canonical report generator | script notes are aligned; `index.tsv` and manifest use `category=proof`, which may be misread | prefer a future script/doc cleanup to use `category=measurement` or `surface=canonical-maintained`; if left unchanged, document the narrow meaning |
| `scripts/performance_sentinels.sh` | bounded local sentinel wrapper | aligned; records build mode, `OMP_NUM_THREADS`, dense backend env values, command, metric, threshold, and notes | document output fields and non-claims in benchmark docs |
| `scripts/wall_check.sh` | existing thresholded reorder regression gate | aligned; clear thresholds and rationale | preserve as the only current hard timing gate |

## Stale Wording and Claim Risks

| risk | why it matters | replacement rule |
|---|---|---|
| `bench_canonical_report.sh` emits `category=proof` | readers can mistake a threshold-free timing snapshot for a timing proof | use `canonical maintained measurement surface` in prose; reserve `proof` for ownership of schema or workflow visibility, not speed |
| new `performance-sentinels` target is undocumented outside Makefile/script | users may miss the intended local regression workflow or overread generated TSV rows | document S5 as the only hard gate and S2 as threshold-free local context |
| OpenMP bullets can be read as universal speedup claims | OpenMP build and runtime behavior depend on machine, workload, and nested runtime settings | say OpenMP may parallelize selected paths when compiled with `SPARSE_OPENMP`; interpreted timing must record thread settings |
| speedup columns in benchmark schemas can look like portable evidence | benchmark speedup is branch-local and fixture-local unless a reviewed comparison artifact says otherwise | describe speedup columns as local measurement columns tied to command, fixture, backend state, and thread context |
| optional backend descriptor fields can look like broad vendor support | Cholesky/LDLT dense backend seams are bounded and fallback to builtin | always pair requested/selected/fallback language with builtin-portable baseline language |
| residual/agreement fields in benchmarks can look like correctness or oracle ownership | correctness ownership belongs to tests and external oracle artifacts | call residual/agreement fields diagnostic context unless a test owns the assertion |
| historical algorithm timing notes can look current and portable | older sprint measurements are useful but not current portable product claims | mark them as historical local evidence and keep live benchmark commands in benchmark docs |

## Backend Disclosure Wording Rules

Use this language pattern for benchmark and documentation updates:

- builtin kernels are the portable baseline;
- optional dense backend requests are best-effort;
- unavailable, unsupported, unsuitable, or unprobed optional providers fall
  back to builtin under the current product behavior;
- `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND` are optional
  dense-kernel request controls, not broad vendor-backend product guarantees;
- Cholesky, LDLT, and eigensolver public backend enums choose library
  algorithms or paths, not universal vendor providers;
- benchmark rows that include backend fields must explain requested, selected,
  and fallback context when timing is interpreted.

Avoid:

- "uses Accelerate/BLAS/LAPACK" without saying "when available and selected";
- "backend proof" when the surface only reports a local timing row;
- "portable speedup" unless a reviewed artifact explicitly owns that claim.

## Timing and Sentinel Wording Rules

| evidence type | allowed wording | disallowed wording |
|---|---|---|
| local benchmark row | branch-local measurement artifact for a named command and fixture | portable performance guarantee |
| canonical report | threshold-free maintained-surface snapshot for local before/after comparison | pass/fail quality gate |
| `wall-check` | narrow thresholded regression gate for named reorder rows and stored baselines | broad reorder benchmark proof |
| `performance-sentinels` S5 rows | existing hard wall-check gate reported in structured form | new benchmark superiority claim |
| `performance-sentinels` S2 rows | threshold-free Cholesky CSC backend-aware local context | hard Cholesky performance gate |
| OpenMP timing | timing under a disclosed OpenMP build and runtime thread context | general OpenMP scalability proof |
| residual/agreement fields | diagnostic context emitted by benchmark rows | substitute for tests or oracle correctness |

## Documentation Update Plan

Day 11 should make only documentation/script-text changes needed to align the
reporting surface with this audit:

1. Add `make performance-sentinels` to the README benchmark command list as a
   local sentinel bundle with bounded hard-fail behavior.
2. Add a `performance-sentinels` section to `benchmarks/README.md` that
   documents output files, S5 hard-fail scope, S2 threshold-free scope, runtime
   context fields, skip behavior, and non-claims.
3. Add the sentinel bundle to `docs/maintainer_guide.md` benchmark governance,
   preserving `wall-check` as the only current hard timing gate.
4. Add a short `docs/algorithm.md` note under performance regression gates so
   historical wall-check documentation remains coherent with the newer bundle.
5. Decide whether Day 11 should change `bench_canonical_report.sh`
   `category=proof` wording. If changed, validate with
   `make bench-canonical-report`; if not changed, document the narrow meaning
   in benchmark governance.

## Completion Check

| criterion | status |
|---|---|
| benchmark reporting inventory completed | complete |
| Day 3 runtime contract compared against benchmark surfaces | complete |
| Day 8 and Day 9 sentinel scope compared against docs/scripts | complete |
| stale wording and claim risks listed | complete |
| backend disclosure wording rules defined | complete |
| local timing, optional acceleration, and reviewed sentinel evidence separated | complete |
| Day 11 documentation update plan written | complete |
