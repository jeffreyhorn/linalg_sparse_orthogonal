# Multi-Strategy FM Ensemble — Design (Sprint 28 Day 3)

## Background

Sprint 27 PLAN.md "parking lot" item (line 681): "run baseline + FIFO
+ annealing in parallel per finest-level FM call, pick the strategy
that produces the best (lowest-cut) result per partition.  Doubles
wall but explores 2× the FM landscape.  Documented here for
traceability; no fixed budget — fires only if items 4-5 succeed AND
closing-day budget remains."

Sprint 27 items 4 (annealing FM) and 5 (root-spectral) both regressed
Pres_Poisson individually (items 4: +2.2-3.1pp; item 5: +2.3pp), so
the parking-lot ensemble was not fired in Sprint 27 (RETROSPECTIVE.md
"Items deferred" #3).  Sprint 28 Item 2 picks this up.

## Contract

The ensemble runner runs K FM strategies in parallel at each
finest-level FM call (level == 0 in `graph_uncoarsen`), scores each
strategy's resulting partition by cut weight, and picks the strategy
with the lowest cut as the partition for the next pass.

Doubles-to-triples wall-time worst-case (one FM walk per strategy per
pass × `passes` passes; default 3 passes × 3 strategies = 9 FM walks
where Sprint 27 master runs 3).  Sprint 27 default Pres_Poisson ND
wall ~10 s → ensemble Pres_Poisson ND wall budget ≤ 30 s; well under
the `make wall-check` 70.5 s ceiling.

Deterministic given the env state: each strategy carries its own
per-call (n, k)-seeded RNG for annealing's temperature schedule + the
gain_noise_formal sigma decay (Sprint 28 Day 2); the partition input
to each strategy is the same starting state (cloned).  Same env state
→ same output bit-for-bit across runs.

## Architecture

### Strategy selector list

New env var `SPARSE_FM_ENSEMBLE_STRATEGIES` is a comma-separated list
of strategy values from the existing `SPARSE_FM_FINEST_STRATEGY` enum
(Sprint 26 Day 7 + Sprint 27 Day 5-6 + Day 11 + Sprint 28 Day 2).
Default: `baseline,fifo,annealing`.

Recognized values: `baseline` (Sprint 22 default FM), `fifo` (Sprint 26
Day 7 tail-pop), `annealing` (Sprint 27 Day 5-6 accept-worsening-with-
exp(-Δgain/T)), `thick_restart` (Sprint 27 Day 10-11 global-best-
tracking with per-pass perturbation — joins the ensemble's strategy
list iff explicitly requested; default selector list excludes it
because thick_restart's anchor-restoration semantics duplicate the
ensemble's pick-best contract).

Parser rejects unrecognized strategy names (silently skipped; ensemble
runs the recognized subset).  If the resulting subset is empty,
ensemble degenerates to baseline (matching Sprint 27 default).

### Ensemble dispatch site

`graph_uncoarsen` lines ~2526-2540 (the existing
`if (level == 0 && finest_strategy == FINEST_FM_<STRATEGY>) { ... }`
chain) gets a new branch:

```c
if (level == 0 && finest_strategy == FINEST_FM_ENSEMBLE) {
    // Parse the strategy selector list (cached at function entry).
    // Set per-strategy thread-locals at strategy-dispatch time.
}
```

The ensemble runner runs INSIDE the per-pass loop (lines ~2565-2619),
not outside it.  For each pass:

1. Clone the current partition state (`next[]` from the prior pass
   or from the coarsening projection at p == 0).
2. For each strategy in the selector list:
   - Clear all per-strategy thread-locals to their defaults
     (`fm_pop_use_tail = 0`, `fm_use_annealing = 0`,
     `fm_use_thick_restart = 0`, `fm_anneal_pass_idx = p`,
     `fm_anneal_total_passes = passes`).
   - Set the strategy's specific thread-locals (e.g. for
     `annealing`, set `fm_use_annealing = 1` + `fm_anneal_schedule`).
   - Restore the clone into a working buffer.
   - Call `graph_refine_fm(dst_graph, working)`.
   - Compute `cur_cut = compute_cut_weight(dst_graph, working)`.
   - Track the (cut, partition) pair under the current best.
3. After all strategies finish, copy the best partition back into
   `next[]`.  Ties broken by first-by-listed-order (the selector
   list's index).

The pass loop continues; pass p+1 starts from the ensemble-best of
pass p (which is what `next[]` now holds), guaranteeing monotone
improvement across passes when at least one strategy can lower the
cut.

### Per-strategy result struct

Internally, the ensemble runner tracks per-strategy `(cut, wall_ns,
strategy_index)` triples for the per-pass winner selection.  Under
`SPARSE_FM_ENSEMBLE_DEBUG=1`, each per-strategy entry is emitted to
stderr (per-pass row: `strategy=annealing cut=2462 wall_ms=12.3 won=0`),
useful for the Day 5 sweep + decision-doc evidence.

### Back-compat

Existing strategy values (`baseline`, `fifo`, `annealing`,
`thick_restart`) keep their semantics; the new `ensemble` value joins
the `finest_fm_strategy_t` enum.  Sprint 27 captures + replays under
the existing strategies produce bit-identical output (the ensemble
branch is opt-in via the new `SPARSE_FM_FINEST_STRATEGY=ensemble`).

## Env-var summary

| Env var | Default | Recognized values |
|---|---|---|
| `SPARSE_FM_FINEST_STRATEGY` | `baseline` | `baseline`, `fifo`, `annealing`, `thick_restart`, **`ensemble`** (new) |
| `SPARSE_FM_ENSEMBLE_STRATEGIES` | `baseline,fifo,annealing` | comma-separated subset of `{baseline, fifo, annealing, thick_restart}` |
| `SPARSE_FM_ENSEMBLE_DEBUG` | unset | `1` (per-pass per-strategy stats to stderr) |

`SPARSE_FM_ANNEALING_SCHEDULE` (Sprint 27 Day 5-6) and
`SPARSE_FM_GAIN_NOISE_SCHEDULE` (Sprint 28 Day 2) continue to control
their respective sub-axes when those strategies fire inside the
ensemble.  `SPARSE_FM_THICK_RESTART_PERTURB` likewise controls the
thick_restart strategy when included in the selector list.

## Determinism contract

Each strategy's internal RNG is keyed by `(n, k)` (Sprint 27 Day 6
recipe) where `n` is the dst_graph size and `k` is the pass index.
The strategies share the pass index `k` but operate on different
internal state (annealing-T, gain-noise sigma, thick-restart anchor),
so per-strategy outputs are reproducible across runs.

The ensemble's pick-best step is purely deterministic (compare-and-
select by cut weight; tie → strategy_index).  No additional RNG state.

Two runs with the same env state produce bit-identical ensemble
output.  Pinned by `test_finest_fm_ensemble_deterministic` (Sprint 28
Day 4 task 5).

## Pick-correctness contract

For a synthetic fixture where one strategy provably dominates (e.g.
baseline FM finds the optimal cut, FIFO/annealing land worse cuts),
the ensemble runner picks the dominant strategy's result.  Pinned by
`test_finest_fm_ensemble_picks_best_strategy` (Sprint 28 Day 3 stub;
Day 4 implementation lights it up).

## Validation plan (Sprint 28 Day 5)

Day 5 task 1 — full corpus sweep under the ensemble × {selector list
variants}:

| Selector list | Rationale |
|---|---|
| `baseline,fifo,annealing` | Default; runs all three Sprint-27-shipped strategies |
| `baseline,annealing` | Drop FIFO if FIFO never wins on the corpus |
| `fifo,annealing` | Drop baseline if baseline never wins on the corpus |
| `baseline,fifo,annealing,thick_restart` | 4-way ensemble joins Sprint 27's thick_restart |

Day 5 task 2 — flip-or-stay decision per `ensemble_fm_decision.md`:

- **Flip** to ensemble default iff:
  - Pres_Poisson improves ≥ 1pp vs Sprint 27 default 0.923× of AMD,
    AND
  - No fixture regresses past 5pp, AND
  - `make wall-check` Pres_Poisson ≤ 2× Sprint 27 default (~20 s)
- **Stay**: ship as advisory; document per-selector evidence

Most likely outcome per Sprint 27 evidence pattern: advisory only.

## Risk + budget

LOC estimate for the ensemble runner: ~200-300 LOC (new branch in
`graph_uncoarsen`'s per-pass loop, new parser
`parse_fm_ensemble_strategies`, new enum value, optional debug emit).
Fits Sprint 28 Day 4's 12-hour implementation budget comfortably.

Wall risk: ensemble runs K=3 strategies per pass per finest-level
call.  Sprint 27 default Pres_Poisson ND wall ~10 s; ensemble worst
case ~30 s.  Well under the 70.5 s `make wall-check` ceiling.

Deterministic-seed risk: low.  Each strategy's RNG is independently
seeded by `(n, k)`; running them in series in the ensemble loop
matches Sprint 27's serial pass model.

Corpus-safety risk: low.  The ensemble's pick-best contract ensures
the resulting partition is no worse than the best individual
strategy's output; ensemble can't regress below the best individual
strategy's per-fixture outcome.  Per-fixture worst case is the
ensemble matches the best individual strategy's Sprint 27 result.

## Implementation plan (Sprint 28 Day 4)

1. Add `FINEST_FM_ENSEMBLE = 5` enum value to `finest_fm_strategy_t`
   in `src/sparse_graph.c`.
2. Add `parse_fm_finest_strategy` branch for `ensemble`.
3. Add `parse_fm_ensemble_strategies()` that parses the comma-
   separated env var into a bitset / sorted list of strategy
   indices; default `{baseline, fifo, annealing}`.
4. Modify `graph_uncoarsen`'s level-0 thread-local-setup block + the
   per-pass loop:
   - Cache the parsed strategy list at function entry.
   - When `finest_strategy == FINEST_FM_ENSEMBLE`, allocate a
     working buffer + per-pass ensemble-best buffer.
   - Replace the single-strategy per-pass FM call with the
     per-strategy loop described above.
   - Track `(cut, strategy_index)` per strategy; pick winner.
5. Add the optional `SPARSE_FM_ENSEMBLE_DEBUG` emit.
6. Light up `test_finest_fm_ensemble_picks_best_strategy` (Day 3
   stub) + add `test_finest_fm_ensemble_deterministic` (new in Day 4).
7. Run `make format && make lint && make test && make wall-check`.

## Files generated Day 3

- `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_design.md` (this doc)
- `tests/test_graph.c::test_finest_fm_ensemble_picks_best_strategy`
  stub (RUN_TEST commented out; Day 4 enables)

## References

- `docs/planning/EPIC_2/SPRINT_28/PLAN.md` Day 3 + Day 4 + Day 5
- `docs/planning/EPIC_2/SPRINT_27/PLAN.md` parking-lot section (line 681)
- `docs/planning/EPIC_2/SPRINT_27/RETROSPECTIVE.md` "Items deferred" #3
- `docs/planning/EPIC_2/SPRINT_26/RETROSPECTIVE.md` "Sprint 27 inputs" #3
  — the original parking-lot motivation
- `src/sparse_graph.c::graph_uncoarsen` — finest-level dispatch site
- `src/sparse_graph.c::graph_refine_fm` — per-strategy FM call
