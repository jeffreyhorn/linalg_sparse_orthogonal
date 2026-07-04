# Sprint 105 Day 6 Named-Matrix Evidence Expansion

## Purpose

Day 6 expands Sprint 105 named-matrix evidence from the Day 5 reviewed
two-fixture proof to the full committed `bench_reorder --skip-factor` fixture
set. The artifact records fill counts, fill ratios, local runtime context,
skipped-lane reasons, and claim boundaries using the Day 3 metric contract and
Day 4 evidence boundary.

## Evidence Command

Command:

```sh
build/bench_reorder --skip-factor
```

Stderr context:

```text
# nd_base_threshold=160, factor=no, via_analyze=no, slice=all
```

Interpretation:

- `fixture_slice=all` means all fixtures owned by `bench_reorder`, not all
  possible SuiteSparse fixtures in the repository.
- `factor_ms=skip` is expected because the command intentionally skips numeric
  factorization.
- `reorder_path=direct` means the direct reordering path was measured, not the
  analyze-time reorder path.
- `reorder_ms` is local timing context only.
- `nnz_L` is the primary structural fill metric.

## Raw Named-Matrix Rows

```csv
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
nos4,100,none,805,0.0,skip,direct,all,160
nos4,100,rcm,888,0.1,skip,direct,all,160
nos4,100,amd,637,0.4,skip,direct,all,160
nos4,100,colamd,778,0.4,skip,direct,all,160
nos4,100,nd,637,0.4,skip,direct,all,160
bcsstk04,132,none,3763,0.0,skip,direct,all,160
bcsstk04,132,rcm,3633,0.9,skip,direct,all,160
bcsstk04,132,amd,3143,1.8,skip,direct,all,160
bcsstk04,132,colamd,3622,2.4,skip,direct,all,160
bcsstk04,132,nd,3143,3.3,skip,direct,all,160
Kuu,7102,none,2993061,0.0,skip,direct,all,160
Kuu,7102,rcm,1024794,55.8,skip,direct,all,160
Kuu,7102,amd,406264,532.8,skip,direct,all,160
Kuu,7102,colamd,830665,4262.2,skip,direct,all,160
Kuu,7102,nd,753755,3904.6,skip,direct,all,160
bcsstk14,1806,none,190791,0.0,skip,direct,all,160
bcsstk14,1806,rcm,178311,12.7,skip,direct,all,160
bcsstk14,1806,amd,116071,117.5,skip,direct,all,160
bcsstk14,1806,colamd,146037,165.2,skip,direct,all,160
bcsstk14,1806,nd,132634,499.9,skip,direct,all,160
s3rmt3m3,5357,none,2208972,0.0,skip,direct,all,160
s3rmt3m3,5357,rcm,636993,31.7,skip,direct,all,160
s3rmt3m3,5357,amd,474609,593.2,skip,direct,all,160
s3rmt3m3,5357,colamd,607647,1142.2,skip,direct,all,160
s3rmt3m3,5357,nd,484890,4368.6,skip,direct,all,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,all,160
Pres_Poisson,14822,rcm,3187081,166.7,skip,direct,all,160
Pres_Poisson,14822,amd,2668793,8452.1,skip,direct,all,160
Pres_Poisson,14822,colamd,3415793,17570.0,skip,direct,all,160
Pres_Poisson,14822,nd,2474435,6800.6,skip,direct,all,160
```

## Fill Ratios

Ratios are derived from the raw `nnz_L` values. `ratio_to_none` uses the
fixture's `none` row as denominator. `ratio_to_amd` uses the fixture's `amd`
row as denominator.

```csv
fixture,n,ordering,nnz_L,ratio_to_none,ratio_to_amd
nos4,100,none,805,1.000,1.264
nos4,100,rcm,888,1.103,1.394
nos4,100,amd,637,0.791,1.000
nos4,100,colamd,778,0.966,1.221
nos4,100,nd,637,0.791,1.000
bcsstk04,132,none,3763,1.000,1.197
bcsstk04,132,rcm,3633,0.965,1.156
bcsstk04,132,amd,3143,0.835,1.000
bcsstk04,132,colamd,3622,0.963,1.152
bcsstk04,132,nd,3143,0.835,1.000
Kuu,7102,none,2993061,1.000,7.367
Kuu,7102,rcm,1024794,0.342,2.522
Kuu,7102,amd,406264,0.136,1.000
Kuu,7102,colamd,830665,0.278,2.045
Kuu,7102,nd,753755,0.252,1.855
bcsstk14,1806,none,190791,1.000,1.644
bcsstk14,1806,rcm,178311,0.935,1.536
bcsstk14,1806,amd,116071,0.608,1.000
bcsstk14,1806,colamd,146037,0.765,1.258
bcsstk14,1806,nd,132634,0.695,1.143
s3rmt3m3,5357,none,2208972,1.000,4.654
s3rmt3m3,5357,rcm,636993,0.288,1.342
s3rmt3m3,5357,amd,474609,0.215,1.000
s3rmt3m3,5357,colamd,607647,0.275,1.280
s3rmt3m3,5357,nd,484890,0.220,1.022
Pres_Poisson,14822,none,5061932,1.000,1.897
Pres_Poisson,14822,rcm,3187081,0.630,1.194
Pres_Poisson,14822,amd,2668793,0.527,1.000
Pres_Poisson,14822,colamd,3415793,0.675,1.280
Pres_Poisson,14822,nd,2474435,0.489,0.927
```

## Fixture-Level Summary

| fixture | size tier | strongest `nnz_L` row | noteworthy comparison |
|---|---|---|---|
| `nos4` | smoke | `amd` and `nd` tie at `637` | RCM increases fill versus `none` on this small fixture |
| `bcsstk04` | smoke/supplemental | `amd` and `nd` tie at `3143` | RCM and COLAMD are close to `none`, not close to AMD/ND |
| `Kuu` | supplemental | `amd` at `406264` | ND remains `1.855x` AMD fill; useful bimodal-degree stress fixture |
| `bcsstk14` | reviewed/supplemental | `amd` at `116071` | ND is `1.143x` AMD and `0.695x` none |
| `s3rmt3m3` | supplemental/local | `amd` at `474609` | ND is close to AMD at `1.022x` |
| `Pres_Poisson` | reviewed/local | `nd` at `2474435` | ND beats AMD at `0.927x` AMD fill |

## Skipped and Unavailable Lanes

| lane | status | reason |
|---|---|---|
| numeric factor timing | skipped | `--skip-factor` intentionally keeps the named-matrix evidence bounded |
| analyze-time reorder path | not run | Day 6 focuses on direct-path named-matrix fill evidence |
| QR/COLAMD `nnz_R` evidence | deferred | owned by `bench_colamd`, not promoted in Day 6 |
| LU `nnz_LU` evidence | deferred | owned by `bench_fillin`, not promoted in Day 6 |
| external matrices outside committed fixtures | deferred | Day 6 uses repo-local deterministic inputs only |

## Focused Validation

Focused validation for this lane checks:

- exact `bench_reorder` CSV header;
- exactly 30 data rows;
- six expected fixtures;
- five expected ordering labels per fixture;
- `factor_ms=skip`, `reorder_path=direct`, `fixture_slice=all`, and
  `nd_base_threshold=160` for every row.

Validation command:

```sh
build/bench_reorder --skip-factor | awk -F, '
NR == 1 {
    expected = "matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold"
    if ($0 != expected) exit 1
    next
}
{
    row_count++
    seen[$1 "," $3] = 1
    fixtures[$1] = 1
    if ($6 != "skip" || $7 != "direct" || $8 != "all" || $9 != "160") exit 1
}
END {
    split("nos4 bcsstk04 Kuu bcsstk14 s3rmt3m3 Pres_Poisson", fixture_names, " ")
    split("none rcm amd colamd nd", ordering_names, " ")
    if (row_count != 30) exit 1
    for (i in fixture_names) {
        if (!(fixture_names[i] in fixtures)) exit 1
        for (j in ordering_names) {
            if (!((fixture_names[i] "," ordering_names[j]) in seen)) exit 1
        }
    }
}
'
```

## Non-Claims

This artifact does not claim:

- local `reorder_ms` values are portable across machines;
- `fixture_slice=all` covers every useful named matrix;
- direct-path results replace analyze-time reorder evidence;
- `colamd` rows are a QR-specific `nnz_R` comparison;
- skipped factor timing is a performance result;
- any ordering is globally superior across sparse matrix classes.

## Completion Check

| criterion | status |
|---|---|
| named-matrix evidence uses canonical fixture and fill fields | complete |
| fill counts and ratios recorded | complete |
| runtime context remains bounded and non-portable | complete |
| skipped and unavailable lanes are explicit | complete |
| focused validation command defined | complete |
