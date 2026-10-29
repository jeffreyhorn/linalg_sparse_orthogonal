# Day 12: Semantic Mutation Proof

The CLI mutation suite starts from a valid planning, synthetic observed or
synthetic measured record. It stages generated artifacts only in a temporary
bundle, validates the positive, writes a syntax-valid mutation, asserts the
exact diagnostic code and JSON pointer, restores the original record and
artifact bytes, then validates the positive again. These are test fixtures,
not CHOLMOD measurements or publication evidence.

| Mutation | Expected code | Pointer |
| --- | --- | --- |
| Selected matrix SHA-256 changed | `selection_mismatch` | `/cell/matrix/sha256` |
| Hashed solver artifact bytes changed | `artifact_path_or_hash` | `/artifacts/0/path` |
| Solver artifact SHA-256 changed | `artifact_path_or_hash` | `/artifacts/0/path` |
| Artifact path changed to `./project.json` | `artifact_path_or_hash` | `/artifacts/0/path` |
| Canonical cell ID changed | `cell_id_mismatch` | `/cell_id` |
| Baseline factor ordering diagnostic changed | `status_contradiction` | `/correctness/baseline/factor_diagnostics/ordering` |
| Loaded dependency SHA-256 changed | `cell_id_mismatch` | `/cell_id` |
| Planned record given a numerical metric | `status_contradiction` | `/correctness/project` |
| Execution-blocked baseline marked pass | `missing_provenance` | `/correctness/baseline/raw_artifact` |
| Observed record loses one solver side | `status_contradiction` | `/correctness` |
| Measured record loses raw-pair reference | `measured_requires_two_passes_and_raw_samples` | `/timing/raw_samples_artifact` |
| Timing unit changed from ns to ms | `unit_or_sample_contract` | `/timing/unit` |
| One timed raw pair removed and new artifact SHA-256 recorded | `unit_or_sample_contract` | `/timing/windows/0` |
| Reported median changed without raw samples changing | `unit_or_sample_contract` | `/timing/windows/0/project/median_ns` |
| Unsupported speedup claim added | `unearned_claim` | `/non_claims` |
| Publication marked approved without review artifact | `missing_provenance` | `/publication_review_artifact` |

The matrix digest and timing unit are frozen schema selections, so they fail
at the schema gate. The dependency hash is part of canonical identity; its
change makes the stored cell ID stale. To reach the solver-control gate, the
test restages the modified raw solver output with a matching artifact hash;
otherwise the earlier raw/report binding would correctly reject it. The
sample-count test similarly rehashes the shortened raw pair file, proving
that a matching digest alone does not confer measurement validity.

No mutation edits the selected matrix, methodology files, source corpus,
or canonical benchmark output. The existing strict-JSON parser mutations
remain in the Day 11 suite; every Day 12 record is separately parsed as
valid JSON before its semantic rejection is asserted.
