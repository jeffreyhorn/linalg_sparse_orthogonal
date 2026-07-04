# Sprint 105 Day 5 Reorder/Fill Reporting Batch 1

## Purpose

Day 5 validates the first Sprint 105 reporting batch against the Day 3 field
contract and Day 4 evidence boundary. The selected first lane is the existing
bounded `bench_reorder` slice because it already emits stable CSV rows with
named fixtures, ordering labels, fill counts, local runtime context, path
metadata, fixture-slice metadata, and ND policy metadata.

## Reporting Path Decision

No source schema change is required for Day 5. The current `bench_reorder`
reporting path already emits the selected first-lane fields:

```text
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
```

The Day 5 update is therefore a reporting-batch proof and sample refresh,
not a benchmark migration. This preserves existing consumers while documenting
how the current fields map to the Sprint 105 contract.

## Contract Mapping

| current field | Sprint 105 contract role | interpretation |
|---|---|---|
| `matrix` | `fixture` | canonical named-matrix identifier |
| `n` | `nrows` and `ncols` for square fixtures | size context |
| `reorder` | `ordering` | `none`, `rcm`, `amd`, `colamd`, or `nd` |
| `nnz_L` | `fill_metric=nnz_L`, `fill_value` | symbolic Cholesky fill context |
| `reorder_ms` | `runtime_metric=reorder_ms`, `runtime_ms` | local timing context only |
| `factor_ms` | factor runtime context | `skip` is intentional for this reviewed first slice |
| `reorder_path` | `ordering_path` | `direct` for the selected Day 5 lane |
| `fixture_slice` | `fixture_slice` | `sprint86`, the historical label for the bounded two-fixture slice |
| `nd_base_threshold` | `policy` | ND base-threshold setting for the run; most relevant to `reorder=nd` |

## Preserved Consumer Behavior

- The header is unchanged.
- The `make bench-reorder-sprint86` target is unchanged.
- `--sprint86-slice` remains the live compatibility flag.
- `fixture_slice=sprint86` remains the emitted compatibility value.
- `factor_ms=skip` remains explicit when `--skip-factor` is used.
- Rows remain threshold-free report rows; no pass/fail interpretation is added.

## Focused Parser Expectations

The focused smoke proof checks:

- exact CSV header;
- exactly 10 data rows for two fixtures across five orderings;
- fixture set is exactly `bcsstk14` and `Pres_Poisson`;
- ordering set for each fixture is exactly `none`, `rcm`, `amd`, `colamd`,
  and `nd`;
- `factor_ms=skip` for every data row;
- `reorder_path=direct` for every data row;
- `fixture_slice=sprint86` for every data row;
- `nd_base_threshold=160` for every data row.

Focused parser command:

```sh
build/bench_reorder --sprint86-slice --skip-factor | awk -F, '
NR == 1 {
    expected = "matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold"
    if ($0 != expected) {
        printf("unexpected header: %s\n", $0) > "/dev/stderr"
        exit 1
    }
    next
}
{
    row_count++
    seen[$1 "," $3] = 1
    fixtures[$1] = 1
    if ($6 != "skip" || $7 != "direct" || $8 != "sprint86" || $9 != "160") {
        printf("unexpected row metadata: %s\n", $0) > "/dev/stderr"
        exit 1
    }
}
END {
    split("bcsstk14 Pres_Poisson", fixture_names, " ")
    split("none rcm amd colamd nd", ordering_names, " ")
    if (row_count != 10) {
        printf("unexpected row count: %d\n", row_count) > "/dev/stderr"
        exit 1
    }
    for (i in fixture_names) {
        if (!(fixture_names[i] in fixtures)) {
            printf("missing fixture: %s\n", fixture_names[i]) > "/dev/stderr"
            exit 1
        }
        for (j in ordering_names) {
            key = fixture_names[i] "," ordering_names[j]
            if (!(key in seen)) {
                printf("missing row: %s\n", key) > "/dev/stderr"
                exit 1
            }
        }
    }
}
'
```

## Regenerated Bounded Sample

Command:

```sh
make bench-reorder-sprint86
```

Stderr context:

```text
# nd_base_threshold=160, factor=no, via_analyze=no, slice=sprint86
```

Stdout:

```csv
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
bcsstk14,1806,none,190791,0.0,skip,direct,sprint86,160
bcsstk14,1806,rcm,178311,11.6,skip,direct,sprint86,160
bcsstk14,1806,amd,116071,93.3,skip,direct,sprint86,160
bcsstk14,1806,colamd,146037,148.5,skip,direct,sprint86,160
bcsstk14,1806,nd,132634,442.4,skip,direct,sprint86,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,sprint86,160
Pres_Poisson,14822,rcm,3187081,136.6,skip,direct,sprint86,160
Pres_Poisson,14822,amd,2668793,6771.8,skip,direct,sprint86,160
Pres_Poisson,14822,colamd,3415793,14006.4,skip,direct,sprint86,160
Pres_Poisson,14822,nd,2474435,6217.6,skip,direct,sprint86,160
```

## Interpretation

- `nnz_L` is the primary fill evidence.
- `reorder_ms` is local timing context and is not portable performance
  evidence.
- `factor_ms=skip` is expected because the Day 5 reviewed first slice uses
  `--skip-factor`.
- `fixture_slice=sprint86` is a historical compatibility label for the current
  bounded two-fixture slice.
- The Day 5 lane remains report-only and threshold-free.

## Completion Check

| criterion | status |
|---|---|
| selected reporting path checked against Day 3 contract | complete |
| existing consumers preserved | complete |
| bounded sample regenerated | complete |
| focused parsing expectations defined | complete |
| public performance non-claims retained | complete |
