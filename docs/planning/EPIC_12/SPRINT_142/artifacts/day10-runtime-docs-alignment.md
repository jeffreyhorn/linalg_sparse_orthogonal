# Day 10 Runtime Docs Alignment

## Purpose

Day 10 aligns user and maintainer documentation with the runtime/backend
contract implemented through Day 9. The work documents the public typed-control
surface, maintainer-only controls, sentinel interpretation, normalized
report-index commands, and non-claim boundaries without changing public API or
ABI.

## Documentation Changes

| File | Alignment |
| --- | --- |
| `README.md` | Added a runtime/backend control boundary, updated `make performance-sentinels` summaries, and documented `S5` hard-gate versus `S2`/`S3` advisory-row interpretation. |
| `docs/cookbook.md` | Added sentinel row interpretation for `S5`, `S2`, and `S3` in the measurement handoff path. |
| `docs/algorithm.md` | Updated the performance regression gate note so `performance-sentinels` names Cholesky CSC and LDLT KKT threshold-free context. |
| `docs/maintainer_guide.md` | Added the authoritative maintainer boundary between public typed controls and maintainer/build/report controls. |
| `tests/corpus/schemas/report_index_fields.md` | Clarified that unresolved runtime/backend policy can remain deferred, while selected Sprint 142 sentinel rows live under the `sentinel` family with local-only boundaries. |

## Public Typed Controls

The docs now identify the current public typed controls as:

- `sparse_cholesky_opts_t.backend`;
- `sparse_ldlt_opts_t.backend`;
- `sparse_eigs_opts_t.backend`;
- `sparse_analysis_opts_t.reorder_opts`.

These are caller-facing because they select solver or analysis behavior at the
API boundary. Zero-initialized options retain default/AUTO behavior, and
explicit typed values win over legacy compatibility environment variables where
both exist.

## Maintainer-Only And Deferred Controls

The docs keep these out of public typed API and ABI wording:

- `SPARSE_CHOL_DENSE_BACKEND`;
- `SPARSE_LDLT_DENSE_BACKEND`;
- `SPARSE_SVD_LOWRANK_OUTER`;
- FM strategy/debug/profile variables;
- OpenMP runtime context such as `SPARSE_OPENMP` and `OMP_NUM_THREADS`;
- package/link settings;
- test/benchmark opt-ins.

They remain useful for diagnostics, build configuration, local report context,
or future productization, but they do not create package, ABI, platform,
portable performance, optional-backend availability, or state-of-the-art
claims.

## Sentinel Interpretation

The docs now consistently describe:

- `S5` as the existing local `wall-check` hard gate;
- `S2` as threshold-free Cholesky CSC backend/path context;
- `S3` as threshold-free LDLT KKT backend context;
- generated sentinel outputs as ignored local artifacts under
  `build/bench-reports/sentinels/`;
- normalized report-index output as discovery and freshness diagnostics, not
  broad proof.

Maintainers should use:

```sh
make performance-sentinels
python3 scripts/normalize_report_index.py --family sentinel \
  --output build/report-index/normalized-index.tsv
python3 scripts/normalize_report_index.py --family sentinel --check-freshness
```

## Non-Claims Preserved

Day 10 did not add or broaden claims for:

- shared-library ABI support;
- package-manager availability;
- broad platform support or parity;
- portable performance;
- optional dense-backend availability;
- external-library parity;
- state-of-the-art behavior.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Docs match implemented runtime/backend behavior. | Complete | README, cookbook, algorithm, maintainer guide, and report-index schema wording now match Day 9 S3 sentinel behavior. |
| Users can distinguish public typed controls from maintainer-only controls. | Complete | README and maintainer guide list public typed controls separately from env/build/report controls. |
| Sentinel rows are framed as local regression evidence only. | Complete | Current-facing docs identify `S5` as the hard local gate and `S2`/`S3` as threshold-free local context. |
