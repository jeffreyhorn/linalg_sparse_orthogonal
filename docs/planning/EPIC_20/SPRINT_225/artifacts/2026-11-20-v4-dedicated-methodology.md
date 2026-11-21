# Proposed s225-v4 dedicated same-machine timing protocol

**Status: proposed; no v4 collection or independent approval yet.** The
reviewed v3 thresholds remain in force. Exact-candidate hosted run
[`44839416899`](https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/44839416899)
failed `S225_PAIR_DISPERSION`: replica 1 had MAD/median 0.14985,
above 0.10. Replicas 2 and 3 were within limit. The three-job result is
rejected in full. This method does not reinterpret those samples or change
their decision.

## Fixed cell and isolation

Use baseline source commit `c42d6715fad7e895bae4941a006df3563f1afc38`
and the exact candidate commit, each in its own clean worktree. Both sides
must contain the locked `nos4.mtx` fixture and byte-identical
`bench_refactor_csc.c` source. Build both serially with the same Make
recipe, `SPARSE_IDX_BITS=32`, default `cc`, `OMP_NUM_THREADS=1`, and no
backend or compiler overrides. The sole metric is `refactor_csc_ms` from
the SPD `chol_spd` row of
`build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 10000`.
The longer repeat is a new workload version, selected before collection to
increase useful work per process. It cannot be pooled with v3's
`--repeat 1000` measurements.

Use one exclusive, otherwise idle Intel macOS machine for the entire
collection. Run offline, outside public pull-request workflows. Record a
stable host ID, OS version, CPU model, compiler path/version, SDK/build
settings, operator identity, and an explicit exclusive-window attestation.
This attestation is a reviewable declaration, not proof of isolation by
itself. An independent reviewer must corroborate the machine and the window.
Do not attach a persistent self-hosted runner to this public repository's
pull-request workflow.

## Collection and decision

Run three **sequential** replicas on that machine, with no competing
benchmark work. Each replica performs five warmup and 20 measured pairs;
odd pairs run baseline then candidate, even pairs run candidate then
baseline. Each side is a separate process. Allow at most 30 seconds per
side and 20 minutes per replica including builds. Preserve raw stdout,
stderr, exit status, elapsed time, timestamps, trial order, executable
digests, build receipts, source/fixture digests, and a SHA-256 manifest in
each raw bundle. Preserve ZIP digests separately. No trial or replica may
be dropped or replaced.

For each measured pair, calculate `candidate_ms / baseline_ms` from the
raw CSV using decimal arithmetic. Per replica require MAD/median
`<= 0.10` and nearest-rank p90/median `<= 1.50`. Across the three
replica medians require `(max-min)/median <= 0.10`. Every replica median
must be `<= 1.25`. These are the v3 limits, unchanged. Reject missing
or malformed evidence, mismatched host/build/binary/source identity,
timeout, changed schedule, or excessive dispersion. A numerical pass is
**provisional** until independent review of the original archives, hashes,
method, host isolation, and recomputed statistics. It is a one-cell
decision only.

Collector (after an independently reviewed exclusive window is arranged):

```sh
python3 scripts/collect_refactor_csc_dedicated.py \
  --output-root build/sprint225-v4-raw \
  --host-id '<stable-machine-id>' \
  --operator '<operator-name>' \
  --exclusive-window-attested
```

Independent recomputation from the three extracted directories or original
ZIPs:

```sh
python3 scripts/check_refactor_csc_dedicated.py \
  --bundle-root build/sprint225-v4-raw \
  --candidate-commit '<full-candidate-sha>' \
  --collection-id '<collector-id>' \
  --output build/sprint225-v4-recomputed.json
```

The existing v3 hosted gate remains separate and fail-closed. A v4
provisional pass alone does not change the release decision, establish
portable performance, or authorize a cross-library ranking.
