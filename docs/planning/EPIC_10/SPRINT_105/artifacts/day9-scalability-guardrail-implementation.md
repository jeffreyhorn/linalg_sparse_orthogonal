# Sprint 105 Day 9 Scalability Guardrail Implementation

## Purpose

Day 9 implements the large-matrix guardrail batch selected on Day 8. The
implementation adds one focused reviewed target that runs deterministic
structural guardrails by default and keeps broader supplemental benchmark
reports opt-in.

## Implemented Surface

New script:

```sh
scripts/large_matrix_guardrails.sh
```

New Makefile target:

```sh
make large-matrix-guardrails
```

Default report directory:

```text
build/bench-reports/large-matrix-guardrails
```

Default behavior:

- runs reviewed structural lanes only;
- writes a manifest and lane index;
- captures focused test output artifacts;
- captures the bounded `bench_reorder --sprint86-slice --skip-factor` CSV;
- records supplemental lanes as explicit skips.

Supplemental opt-in:

```sh
SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 make large-matrix-guardrails
```

Supplemental mode adds threshold-free local report artifacts for:

- `build/bench_reorder --skip-factor`;
- `build/bench_amd_qg --skip-bitset`.

## Reviewed Guardrail Lanes

| lane id | command | category | pass/fail basis |
|---|---|---|---|
| `G1` | `build/test_reorder_amd_qg` | reviewed | qg-AMD wrapper behavior and `banded-n10000-bw5` structural guardrail |
| `G2` | `build/test_reorder_nd` | reviewed | ND generated-family, named-matrix, residual, and policy structural tests |
| `G3` | `build/test_graph` | reviewed | graph partition, separator, generated-family, and determinism structural tests |
| `G4` | `build/bench_reorder --sprint86-slice --skip-factor` | reviewed | exact bounded CSV shape and structural fill rows |

The script validates the `G4` CSV shape:

- exact header;
- 10 data rows;
- fixtures `bcsstk14` and `Pres_Poisson`;
- orderings `none`, `rcm`, `amd`, `colamd`, and `nd`;
- `factor_ms=skip`;
- `reorder_path=direct`;
- `fixture_slice=sprint86`;
- `nd_base_threshold=160`.

## Supplemental Lanes

| lane id | command | default status | interpretation |
|---|---|---|---|
| `S1` | `build/bench_reorder --skip-factor` | `skip` | threshold-free full named-matrix reorder/fill report |
| `S2` | `build/bench_amd_qg --skip-bitset` | `skip` | threshold-free qg-AMD and generated-banded report; max-RSS is platform-local |

These lanes run only when `SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1`.

## Generated Report Shape

Default command:

```sh
make large-matrix-guardrails
```

Generated artifacts:

```text
build/bench-reports/large-matrix-guardrails/index.tsv
build/bench-reports/large-matrix-guardrails/manifest.txt
build/bench-reports/large-matrix-guardrails/test_graph.txt
build/bench-reports/large-matrix-guardrails/test_reorder_nd.txt
build/bench-reports/large-matrix-guardrails/test_reorder_amd_qg.txt
build/bench-reports/large-matrix-guardrails/bench_reorder_sprint86.csv
```

Captured lane index:

```tsv
lane_id	status	category	command	artifact	notes
G3	pass	reviewed	build/test_graph	test_graph.txt	graph partition, separator, generated-family structural tests
G2	pass	reviewed	build/test_reorder_nd	test_reorder_nd.txt	ND generated-family and named-matrix structural tests; explicit skips remain in artifact
G1	pass	reviewed	build/test_reorder_amd_qg	test_reorder_amd_qg.txt	qg-AMD wrapper and banded-n10000-bw5 structural guardrail
G4	pass	reviewed	build/bench_reorder --sprint86-slice --skip-factor	bench_reorder_sprint86.csv	bounded bench_reorder CSV shape and structural fill rows
S1	skip	supplemental	build/bench_reorder --skip-factor	n/a	set SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 to run
S2	skip	supplemental	build/bench_amd_qg --skip-bitset	n/a	set SPARSE_LARGE_GUARDRAILS_SUPPLEMENTAL=1 to run
```

Captured bounded CSV:

```csv
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
bcsstk14,1806,none,190791,0.0,skip,direct,sprint86,160
bcsstk14,1806,rcm,178311,13.9,skip,direct,sprint86,160
bcsstk14,1806,amd,116071,135.3,skip,direct,sprint86,160
bcsstk14,1806,colamd,146037,202.7,skip,direct,sprint86,160
bcsstk14,1806,nd,132634,571.1,skip,direct,sprint86,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,sprint86,160
Pres_Poisson,14822,rcm,3187081,160.4,skip,direct,sprint86,160
Pres_Poisson,14822,amd,2668793,7846.9,skip,direct,sprint86,160
Pres_Poisson,14822,colamd,3415793,17242.7,skip,direct,sprint86,160
Pres_Poisson,14822,nd,2474435,6807.9,skip,direct,sprint86,160
```

Runtime fields are local context only.

## Validation

Validation commands run:

```sh
bash -n scripts/large_matrix_guardrails.sh
make large-matrix-guardrails
```

Results:

```text
large-matrix-guardrails: wrote build/bench-reports/large-matrix-guardrails
  - index.tsv
  - manifest.txt
  - test_graph.txt
  - test_reorder_nd.txt
  - test_reorder_amd_qg.txt
  - bench_reorder_sprint86.csv
```

The generated `index.tsv` recorded reviewed lanes `G1` through `G4` as
`pass` and supplemental lanes `S1` and `S2` as explicit opt-in `skip` rows.

## Residual Queue

| item | status | reason |
|---|---|---|
| supplemental mode full validation | deferred | useful but slower; default reviewed path is the maintained Day 9 guardrail |
| hard timing thresholds for `S1` or `S2` | deferred | requires baseline and machine-class design |
| max-RSS pass/fail bounds | deferred | platform-sensitive; report-only for now |
| `bench_fillin` arrow guardrail | deferred | current owner is human-readable LU context |
| large ND threshold sweeps | local-only | policy exploration and runtime cliffs |

## Non-Claims

This implementation does not claim:

- new portable timing thresholds;
- max-RSS comparability across platforms;
- supplemental lanes are required quality gates;
- local generated-family coverage replaces named-matrix evidence;
- numeric factor timing is covered by the default reviewed guardrail.

## Completion Check

| criterion | status |
|---|---|
| reviewed guardrail target implemented | complete |
| supplemental commands separated from reviewed lanes | complete |
| bounded report artifacts generated | complete |
| focused validation passed | complete |
| residual large-matrix queue recorded | complete |
