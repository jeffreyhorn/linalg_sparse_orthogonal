# Sprint 47 Day 8 Artifact: Reorder-Mode Parity Batch

## Purpose

Land the bounded reorder-mode parity cleanup identified on Day 7 by removing
the remaining internal drift in `bench_main.c` between the supported
`--reorder` surface, the printed reorder labels, and the user guidance around
specialized COLAMD benchmark ownership.

## Main Day 8 Conclusion

Sprint 47 now has a cleaner, more truthful main benchmark reorder surface.

The Day 8 batch stayed intentionally narrow:

- touched only `bench_main.c`
- preserved the intended supported main-benchmark reorder set:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- clarified the specialized handoff for COLAMD work:
  - `bench_reorder`
  - `bench_colamd`

The important boundary held:

- `bench_main` did **not** expand into a second general reorder-comparison tool
- `bench_reorder` still owns the broader `none|rcm|amd|colamd|nd` sweep
- `bench_colamd` still owns the QR/COLAMD-focused comparison surface

## Landed Parity Cleanup

### Help and usage text now match the intended reorder surface explicitly

The `bench_main` help/usage surface now states:

- `--reorder none|rcm|amd|nd`
- COLAMD comparisons should use:
  - `bench_reorder`
  - `bench_colamd`

Interpretation:

- the main benchmark CLI is now explicit about what it supports
- users are pointed toward the specialized tools instead of left with an
  implicit mismatch

### Runtime reorder labeling now matches the supported main surface

`bench_main.c` now uses a main-benchmark-specific reorder label path that
reports:

- `none`
- `rcm`
- `amd`
- `nd`

and treats `SPARSE_REORDER_COLAMD` as outside the supported `bench_main`
surface.

Interpretation:

- Day 8 removed the internal source-level drift where the label helper knew
  about `colamd` while the accepted/advertised CLI surface did not

### Unsupported `colamd` input now fails with a clearer ownership handoff

`./build/bench_main --reorder colamd` now fails with a message that explicitly
points the user to:

- `bench_reorder`
- `bench_colamd`

Interpretation:

- the batch improved user guidance without widening the main benchmark’s
  capability scope

## Direct CLI Proof

The touched reorder-parity surface was exercised directly:

- `./build/bench_main --help`
- `./build/bench_main --reorder nd --size 8 --repeat 1`
- `./build/bench_main --reorder colamd`

Observed direct behavior:

- help text now documents the supported main-benchmark reorder set and the
  specialized COLAMD handoff
- valid `--reorder nd` input completed successfully and reported:
  - `Reorder: nd`
- unsupported `--reorder colamd` input failed cleanly with an explicit handoff
  message naming:
  - `bench_reorder`
  - `bench_colamd`

## Validation

Because `*.c` changed, the required gate was:

```bash
make format
make lint
make test
```

Those passed.

Because Day 8 still touches the main benchmark CLI surface, the stronger
reviewed baseline also ran:

```bash
make quality-review-full
```

That passed too.

## Sprint 47 Position After Day 8

The front-half benchmark sequence is now in a clean state:

1. Day 5:
   - shared internal parser helper seam
2. Day 6:
   - `bench_main` parser modernization
3. Day 7:
   - residual audit
4. Day 8:
   - reorder-mode / emitted-label parity cleanup

That means the remaining Sprint 47 queue can now move on to:

- example safety audit / bounded cleanup
- auxiliary tooling cleanup
- touched-doc refresh
- validation closeout

## Bottom Line

Day 8 delivered:

- a truthful supported reorder surface in `bench_main`
- clearer user guidance around COLAMD ownership
- preserved benchmark-surface boundaries
- a fully green validation baseline for the touched parity batch

That is the right bounded reorder-parity landing for Sprint 47.
