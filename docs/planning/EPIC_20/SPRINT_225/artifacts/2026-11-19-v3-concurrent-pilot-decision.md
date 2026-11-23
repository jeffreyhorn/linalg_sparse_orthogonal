# Sprint 225 `s225-v3` concurrent pilot: conflicting decisions

**Disposition:** Do not accept either pilot as the reviewed timing result.
The same exact source commit, `70da48b2cc0fa881ff03710e6ab61dc12e1ae73a`,
triggered a feature-branch push run and a pull-request run concurrently.
Both used the proposed `s225-v3` paired method. The push group passed the
locked numerical rules, while the pull-request group failed between-job
dispersion. The protocol has been changed to run once per candidate on the
pull-request path, with a separate `master` push trigger after merge. No
numerical threshold was changed, and the conflicting evidence is retained.

| Trigger | Run | Job median candidate/baseline ratios | Between-job spread | Decision |
| --- | --- | --- | ---: | --- |
| Push | [44646070212](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44646070212) | `0.9707594235`, `1.0294691224`, `1.0000000000` | `0.0587096989` | `S225_PAIR_PASS` (provisional numerical result) |
| Pull request | [44646080235](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44646080235) | `1.0208333333`, `1.0438707634`, `0.8982238692` | `0.1426745086` | `S225_PAIR_DISPERSION` (fails `0.10` limit) |

Each run has three original paired ZIPs and one decision ZIP. The original
ZIP SHA-256 values match the artifact digests returned by GitHub. Each pair
ZIP contains 106 distinct members; its hash manifest, raw streams, trial
order, commit binding, and recomputed statistics pass independent local
validation. The digests below identify the exact evidence for review.

| Run | Pair 1 ZIP | Pair 2 ZIP | Pair 3 ZIP | Decision ZIP |
| --- | --- | --- | --- | --- |
| Push `44646070212` | `16cf0555a53e6170e685072fd41b8fde035f0b141e872b360b4c9fd50c8e743c` | `a8a54b2dafd991b18b3811c026b4e44d41887db9e2a48a7afb534c2e393db165` | `c6fd4d32fc426ce564790e617e939dfbbe49727b5176e80e1cb069b2073b4ede` | `e6f9d16b6e884c94404fa7cf2323b7e081d2e6f496fcbc5bed7d09408d715b84` |
| Pull request `44646080235` | `014de6241b3184bdb6707e8188319a7fb3c9fa23b8835d2dca41c69e5eecd88e` | `95001a2a599be7c5378d1639183f9c173908507a34b870e7a66c9775d877a590` | `d1fcb4e853559d70f825bcd1219aade6c19e46dba94475baa7c983cd6602edd6` | `7ad74b83e1399618bee586f527692d3c77c43c3d68d6bb79e89a1a725277c1e9` |

The discrepancy is not explained by a wrong source or corrupted upload.
Concurrent hosted collection is a plausible source of interference, not a
proven cause. The next exact-head pull-request run must be inspected as a
complete group; neither pilot can be substituted for it. Independent review
must consider the conflicting pilot before accepting any timing claim.
