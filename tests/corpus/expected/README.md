# Expected Results

Expected-result files in this directory are small committed baselines for
maintained corpus fixtures. Each expected-result file must be tied to a row in
`../manifests/fixtures.tsv` and must preserve the fixture-local claim boundary.

Expected results are not observed oracle output. Observed rows, comparison
logs, generated indexes, and local run manifests belong under ignored
`build/corpus/` or `build/corpus-reports/`.

Raw QR basis vectors should not be the primary expected artifact for
rank-deficient QR fixtures. Prefer rank, nullity, residual, and
projector/subspace comparisons with fixture-local tolerances.
