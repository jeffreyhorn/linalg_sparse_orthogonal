# Sprint 47 Day 7 Artifact: Post-`bench_main` Audit

## Purpose

Audit the post-Day-6 auxiliary state so Sprint 47 can separate the real
remaining reorder-mode parity work from lower-priority peer benchmark, example,
and script surfaces before the next implementation batch lands.

## Main Day 7 Conclusion

The remaining benchmark/example/tooling queue is now much narrower than the
generic Sprint 47 starting shape.

After Day 5 and Day 6, the strongest residual direct benchmark seam is:

- reorder-mode / emitted-label parity in `bench_main.c`

The other auxiliary surfaces are now mostly:

- peer reference or ownership surfaces
- later bounded helper-alignment candidates
- explicitly lower-priority example/script follow-ons

Interpretation:

- Day 8 should be a narrow reorder-mode parity batch
- Sprint 47 should not reopen main parser modernization or drift into broad peer
  benchmark churn

## Residual Benchmark Queue by Class

### 1. Direct next implementation target: reorder-mode parity in `bench_main.c`

The live `bench_main.c` state now has:

- help text and modern malformed-input handling
- shared-helper parsing for `--reorder`
- `reorder_name()` support for:
  - `none`
  - `rcm`
  - `amd`
  - `colamd`
  - `nd`

But the advertised/accepted `--reorder` surface is still:

- `none`
- `rcm`
- `amd`
- `nd`

Interpretation:

- the strongest remaining direct seam is not parser mechanics anymore
- it is supported-mode / emitted-label parity for the main benchmark CLI

### 2. Specialized ownership surface: `bench_reorder`

`bench_reorder.c` already owns the broader reorder-comparison surface:

- `none`
- `rcm`
- `amd`
- `colamd`
- `nd`

Interpretation:

- Day 8 must preserve the specialized ownership boundary
- parity cleanup should make `bench_main` honest and internally consistent
  without turning it into a second general reorder-comparison tool

### 3. Specialized QR/COLAMD comparison surface: `bench_colamd`

`bench_colamd.c` remains the QR/COLAMD-focused comparison tool.

Interpretation:

- Sprint 47 should not blur that surface into the main benchmark harness
- any Day 8 `colamd` exposure decision must respect this ownership split

### 4. Modern reference surface: `bench_eigs.c`

`bench_eigs.c` already has:

- explicit `--help`
- explicit unknown-option handling
- checked parse helpers

Interpretation:

- it remains a reference surface for CLI behavior shape
- it is not a required Day 8 code target

### 5. Lower-priority helper-alignment surfaces

These remain bounded later follow-ons rather than main Sprint 47 drivers:

- `bench_iterative_reuse.c`
- `bench_eigs_reuse.c`
- `example_eigs.c`
- `example_iterative.c`
- `example_matrix_free.c`
- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

Interpretation:

- the front half of Sprint 47 has done its job by narrowing the queue
- Day 8 should not pull these surfaces forward

## Day 8 Target Set

The correct Day 8 scope is now explicit:

- align supported reorder modes and emitted reporting in the touched
  `bench_main` surface
- remove the current internal drift between:
  - `reorder_name()`
  - help/usage text
  - accepted `--reorder` values

The correct Day 8 non-goals are also explicit:

- no broad peer benchmark rewrite
- no helper-layer redesign
- no benchmark framework expansion
- no example or script cleanup early

## Sprint 47 Position After Day 7

Sprint 47 now has a clean front-half sequence:

1. Day 5:
   - shared internal parser helper seam
2. Day 6:
   - main `bench_main` parser modernization
3. Day 7:
   - residual audit
4. Day 8:
   - narrow reorder-mode / emitted-label parity cleanup

That is a much stronger shape than the original generic auxiliary backlog.

## Bottom Line

Day 7 confirms:

- the remaining direct benchmark queue is concrete rather than generic
- reorder-mode parity is the next real implementation seam
- peer benchmark, example, and script surfaces remain bounded later follow-ons

That is the right audit result before the next implementation batch.
