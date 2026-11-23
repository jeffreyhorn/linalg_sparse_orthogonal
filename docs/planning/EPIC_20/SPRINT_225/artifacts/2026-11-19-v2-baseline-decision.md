# Sprint 225 `s225-v2` hosted baseline intake: rejected

**Decision:** The first predeclared three-job collection under the
owner-approved `s225-v2` protocol is rejected. No accepted baseline manifest
or `active.json` is written. The macOS timing gate remains blocked and no
candidate timing or performance claim is authorized.

The [hosted collection run](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44623787150)
used exact source commit `3969e5b060faa722dc44d4c769b18d236f8d74d9`.
Each attempt checked out that commit, built the serial 32-bit benchmark,
ran `nos4.mtx --repeat 100` for five warmups and 20 measurements, and
uploaded a raw bundle. Attempts 1 and 2 were downloaded before rerunning;
attempt 3 was downloaded after completion. The original ZIP bytes are
retained below. The `s225-v2` validator verified each ZIP's member set,
hash list, run metadata, raw trial streams, trial order, and CSV metrics.
The three resolved fingerprints and executable hashes match. Library
archive hashes differ, as in the earlier rejected intake; these records
do not support a byte-identical archive claim.

| Attempt | Hosted job | Artifact ID | Retained ZIP SHA-256 | Median ms | MAD/median | p90/median |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [1](2026-11-19-rejected-v2-baseline/attempt-1.zip) | `135912111115` | `14579430344` | `7af9ed018b56102dc5acaec685e422d37599615b75cdad7b94ae37c48c97a06c` | 0.064 | 0.0078125 | 1.046875 |
| [2](2026-11-19-rejected-v2-baseline/attempt-2.zip) | `135912838382` | `14578667174` | `03e548b24d13930ef9877979b1b053fc8d4be05c53ad33691681b5e82bfa7cee` | 0.084 | 0.2619047619 | 1.3690476190 |
| [3](2026-11-19-rejected-v2-baseline/attempt-3.zip) | `135913661276` | `14578692362` | `a820f18b8c6a995f02e2ad026c7cae0569821d0a0c8875aa6b92c0490f4d7081` | 0.071 | 0.0774647887 | 1.0704225352 |

Attempt 2 exceeds the approved per-job MAD/median ceiling of `0.10`.
The between-job spread is `(0.084 - 0.064) / 0.071 =
0.2816901408450704225352112676`, exceeding the `0.10` ceiling.
Both failures independently prohibit acceptance. Selecting attempts 1
and 3 or repeatedly sampling until a favorable group appears would change
the predeclared decision rule and is not permitted.

The next method needs an explicit review before collection. One option is
a dedicated machine with controlled contention; another is a versioned
same-machine paired comparison protocol that measures the baseline and
candidate in each hosted job. Merely increasing `--repeat` again cannot be
assumed to repair the observed across-job variation. Any new protocol must
state its runner, workload, sampling, dispersion, and acceptance rules in
advance, retain every original bundle, and receive independent methodology
and evidence review. The existing `s225-v2` gate continues to fail closed.
