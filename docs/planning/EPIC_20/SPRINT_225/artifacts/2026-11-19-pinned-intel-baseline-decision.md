# Sprint 225 pinned Intel baseline intake: rejected

**Decision:** Do not accept this `s225-v1` collection. The required
`benchmarks/baselines/refactor-csc-nos4/active.json` remains absent, and the
macOS timing check must continue to report `S225_BASELINE_MISSING`.

The collection used source commit `b621cd3fde885cd3cbc89f5edd6415c73de9532a`
in [hosted run 44621067803](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44621067803).
The three consecutive attempts were downloaded and retained before each rerun.
All three raw ZIPs contain 54 distinct files, pass the `s225-v1` raw bundle,
run, trial, and hash checks, and record five warmups plus 20 measured trials.
They agree on resolved `macos15` image version `20260824.0482.1`, macOS
`15.7.9`, x86-64, `Intel(R) Core(TM) i7-8700B CPU @ 3.20GHz`, Apple Clang
`17.0.0`, build settings, and executable SHA-256
`f24a4b68303d575cc8f986bb81261aef7ec8f7fb6bcbc8e8b228402509157c63`.
The archive hashes differ; no byte-identical library claim follows from these
records.

| Attempt | Hosted job | Artifact ID | Retained ZIP SHA-256 | Median ms | MAD/median | p90/median |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [1](2026-11-19-rejected-intel-baseline/attempt-1.zip) | `135902545908` | `14575668702` | `e067d6351d6dd92fdfd9f26288bbbcbf56b56c65510583badd12ace1b3400d37` | 0.0695 | 0.1079136691 | 1.4100719424 |
| [2](2026-11-19-rejected-intel-baseline/attempt-2.zip) | `135903810918` | `14577361223` | `af8355b18ffec6f18aba0acf08bcc75c82b286b1b610773595742ff8f187cae1` | 0.0745 | 0.1208053691 | 1.1812080537 |
| [3](2026-11-19-rejected-intel-baseline/attempt-3.zip) | `135904579589` | `14577696103` | `09059fd287218fab9e64e89afa1561796a8775169d8508b5cce11a777bddaa49` | 0.1105 | 0.0542986425 | 1.0588235294 |

Attempts 1 and 2 exceed the locked `0.10` MAD/median limit. The three-job
spread is `(0.1105 - 0.0695) / 0.0745 = 0.5503355705`, exceeding the
locked `0.10` limit. A matching fingerprint does not remedy either failure.
No favorable subset of these attempts is an accepted baseline. The ZIP
copies are evidence of a rejected group, not candidate timing results.

## Next review decision

The `--repeat 1` target yields only about 0.07–0.11 ms per process on this
runner class, exposing the metric to substantial short-interval noise. A
proposed `s225-v2` method would keep the same matrix, SPD scenario, metric,
runner identity checks, 5+20 process schedule, 30-second process cap,
20-minute job cap, three-job requirement, dispersion limits, 25% candidate
ceiling, raw-byte retention, and 30-day baseline lifetime, while changing
the benchmark command to `--repeat 100` and averaging 100 CSC refactors per
process. This is a **proposal, not an approved policy or baseline**. It needs
independent methodology review before any `s225-v2` evidence is accepted.
After approval, collect a fresh predeclared three-job group under the new
version, inspect every original ZIP and statistic, record the independent
acceptance decision, then source-control the accepted manifest and pointer.
Only then can the exact-candidate timing job run. Until that happens, there
is no timing pass, release approval, or performance claim.
