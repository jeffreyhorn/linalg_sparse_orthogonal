# Sprint 47 Working Notes

## Day 1

**Objective:** Turn the Sprint 47 project-plan scope plus the Sprint 40/41/45/46
execution rules into a concrete benchmark/example/tooling starting point by
confirming the preserved reviewed contracts, naming the Sprint 47 workstreams
explicitly, and defining the authoritative benchmark, example, and auxiliary
script inputs before CLI/helper modernization begins.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
2. Re-read the Sprint 47 project-plan source and the new sprint plan:
   - `sed -n '257,288p' docs/planning/EPIC_4/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
3. Reconfirm the inherited reviewed CMake baseline:
   - `ctest -N --test-dir build/quality-review-cmake`
4. Reconfirm the current maintained reviewed/dead-code command surfaces:
   - `make -n quality-review-full deadcode-report deadcode-check`
5. Measure the live benchmark/example/tooling hotspot sizes:
   - `wc -l scripts/deadcode_report.py scripts/deadcode_workflow.sh benchmarks/bench_main.c benchmarks/bench_eigs.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_eigs.c examples/example_iterative.c examples/example_matrix_free.c`
6. Refresh the live benchmark/example/tooling seam markers:
   - `rg --files | rg 'deadcode|workflow|bench_main|bench_eigs|example_(eigs|iterative|matrix_free)'`
   - `sed -n '1,220p' benchmarks/bench_main.c`
   - `sed -n '220,420p' benchmarks/bench_main.c`
   - `sed -n '1,220p' benchmarks/bench_eigs.c`
   - `rg -n "atoi|atof|strtol|strtod|reorder|Usage:|--" benchmarks/bench_main.c benchmarks/bench_eigs.c examples/example_eigs.c examples/example_iterative.c examples/example_matrix_free.c scripts/deadcode_report.py scripts/deadcode_workflow.sh`
7. Re-read one recent Day 1 artifact/notes pattern for format calibration:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_46/artifacts/day1-scope-and-eigensolver-baseline.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_46/WORKING_NOTES.md`

### Day 1 Findings

#### 1. Sprint 47 starts from a preserved Sprint 40/41/45/46 baseline, not from baseline repair work

The inherited starting contract remains explicit and stable:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code/reporting paths already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized
- Sprint 41 already left behind the shared internal arithmetic/allocation seam:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- Sprint 45 and Sprint 46 already left behind repeated-run benchmark examples:
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`

Interpretation:

- Sprint 47 is not a solver-quality-baseline sprint
- Sprint 47 is a benchmark/example/tooling modernization sprint on top of an
  already-validated Epic 4 baseline

#### 2. `bench_main.c` is the main modernization hotspot

The live benchmark surface is not flat:

- `benchmarks/bench_main.c` = `774`
- `benchmarks/bench_eigs.c` = `958`
- `benchmarks/bench_iterative_reuse.c` = `251`
- `benchmarks/bench_eigs_reuse.c` = `201`

But the important difference is not only size. It is parser maturity:

- `bench_main.c` still carries:
  - `atoi(...)` for `--spmv-iters`
  - `atoi(...)` for `--size`
  - `atoi(...)` for `--repeat`
  - inline reorder-mode parsing
- `bench_eigs.c` already carries:
  - checked `strtol` / `strtod`
  - explicit parser helpers
  - explicit usage/help text
  - clearer enum-like mode handling

Interpretation:

- Sprint 47 should treat `bench_eigs.c` as a comparison point for better CLI
  shape
- Sprint 47 should treat `bench_main.c` as the main direct parser landing zone

#### 3. Reorder-mode parity is a real Sprint 47 seam

The live `bench_main.c` usage and parser currently name:

- `none`
- `rcm`
- `amd`
- `nd`

while the broader library reorder surface already includes:

- `SPARSE_REORDER_COLAMD`

Interpretation:

- Sprint 47 has a real supported-mode / emitted-label parity seam
- this is not only an input-validation cleanup sprint
- Day 8 should be driven by live mode-parity evidence, not by generic wording
  cleanup

#### 4. The example surface is smaller and should stay bounded

The main touched example surface is comparatively compact:

- `examples/example_eigs.c` = `284`
- `examples/example_iterative.c` = `144`
- `examples/example_matrix_free.c` = `122`

Interpretation:

- examples are real Sprint 47 follow-ons
- they are not the main complexity driver
- example cleanup should follow the shared parser/helper seam rather than
  expanding into a broad pedagogical rewrite

#### 5. The main auxiliary tooling seam is script-side safety alignment, not framework replacement

The inherited script-side hotspot surface is explicit:

- `scripts/deadcode_report.py` = `523`
- `scripts/deadcode_workflow.sh` = `189`

Interpretation:

- Sprint 47 should treat auxiliary tooling as a bounded safety and helper
  alignment target
- the existing dead-code workflow contract should be preserved, not redesigned

#### 6. The Sprint 47 workstreams are explicit and already bounded by the plan

Day 1 confirms the sprint's eight bounded workstreams directly from the plan:

- CLI/helper seam inventory
- shared parsing-helper design
- `bench_main` modernization
- reorder-mode parity cleanup
- example safety audit
- auxiliary tooling cleanup
- benchmark/example docs refresh
- validation closeout

Interpretation:

- the front half of the sprint should stay benchmark/helper-first
- the back half should then pivot into bounded example/tooling/doc cleanup and
  validation

#### 7. Sprint 47 inherits a clear preserve-not-reopen boundary

Sprint 47 should not reopen:

- core solver architecture work
- public library API expansion for CLI helpers
- broad benchmark framework redesign
- dead-code topology changes
- cross-platform CI contract changes
- broad tutorial/README restructuring beyond the touched benchmark/example docs

Interpretation:

- the correct Sprint 47 shape is:
  - land shared auxiliary parsing helpers
  - modernize the main benchmark CLI
  - reconcile reorder-mode parity
  - align examples and scripts to the current safety conventions
  - refresh only the touched docs

#### 8. The Day 1 landing order is fixed before implementation starts

The correct early sprint order is:

1. baseline and seam inventory
2. shared parsing-helper design
3. `bench_main` modernization
4. reorder-mode parity cleanup
5. example audit and bounded cleanup
6. auxiliary tooling cleanup
7. docs refresh and validation closeout

Interpretation:

- Sprint 47 should preserve Sprint 40's core rule: usability and safety cleanup
  should be guided by measured seams and an explicit validation anchor before
  broader surface work lands

## Day 2

**Objective:** Refresh the benchmark, example, and auxiliary tooling seam
inventory so Sprint 47's shared parsing-helper work, `bench_main`
modernization, reorder-mode parity cleanup, and later example/tooling batches
are sequenced from the live post-Sprint-46 repo state rather than only from the
project-plan labels.

### Commands Run

1. Re-read the Sprint 47 Day 2 plan section:
   - `sed -n '55,94p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the Day 1 baseline artifact:
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_47/artifacts/day1-scope-and-cli-auxiliary-baseline.md`
3. Re-read the main peer benchmark and script-side surfaces:
   - `sed -n '1,220p' benchmarks/bench_eigs_reuse.c`
   - `sed -n '1,220p' scripts/deadcode_report.py`
   - `sed -n '1,220p' scripts/deadcode_workflow.sh`
4. Re-read the main benchmark modernization target and its deeper parser body:
   - `sed -n '1,220p' benchmarks/bench_main.c`
   - `sed -n '220,420p' benchmarks/bench_main.c`
5. Refresh the live parser/helper seam markers across the main auxiliary
   surfaces:
   - `rg -n "atoi|atof|strtol|strtod|reorder|Usage:|--" benchmarks/bench_main.c benchmarks/bench_eigs.c examples/example_eigs.c examples/example_iterative.c examples/example_matrix_free.c scripts/deadcode_report.py scripts/deadcode_workflow.sh`
6. Reconfirm the current hotspot sizes for the bounded Sprint 47 targets:
   - `wc -l scripts/deadcode_report.py scripts/deadcode_workflow.sh benchmarks/bench_main.c benchmarks/bench_eigs.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_eigs.c examples/example_iterative.c examples/example_matrix_free.c`

### Day 2 Findings

#### 1. The Sprint 47 auxiliary surface reduces to five seam classes, not one generic cleanup bucket

The live repo now breaks into:

- legacy benchmark CLI parsing and malformed-input reporting drift
- reorder-mode / emitted-label parity drift
- modern benchmark CLI comparison/reference surfaces
- bounded example safety/helper follow-ons
- script-side support-code alignment

Interpretation:

- only the first two are strong direct Day 3 / Day 5 / Day 6 implementation
  targets
- the others are real Sprint 47 work, but they should follow after the shared
  parser/helper seam and `bench_main` modernization

#### 2. `bench_main.c` is still the strongest direct shared-helper adoption target

The live `bench_main.c` evidence is now concrete:

- `benchmarks/bench_main.c` = `774`
- still uses:
  - `atoi(...)` for `--spmv-iters`
  - `atoi(...)` for `--size`
  - `atoi(...)` for `--repeat`
- still keeps mode parsing inline for:
  - pivot
  - reorder

Interpretation:

- Sprint 47 should define the shared positive/bounded integer and enum-like
  parser contract around `bench_main` first
- this is the clearest place where checked parsing and clearer error handling
  buy immediate consistency

#### 3. Reorder-mode parity remains a separate real seam after parser cleanup

The live `bench_main.c` usage/parser still advertises:

- `none`
- `rcm`
- `amd`
- `nd`

while the broader reorder surface already includes:

- `SPARSE_REORDER_COLAMD`

Interpretation:

- reorder parity is not merely a parser spelling issue
- Sprint 47 needs a distinct post-parser reconciliation batch for supported
  mode names and emitted labels

#### 4. `bench_eigs.c` is a comparison point, not the first rewrite target

The live `bench_eigs.c` surface already carries:

- checked `strtol(...)`
- checked `strtod(...)`
- explicit parser helpers
- explicit usage/help text

Interpretation:

- `bench_eigs.c` should guide Sprint 47's parser-helper design shape
- it is more useful as a reference and later bounded alignment surface than as
  the first direct modernization target

#### 5. The repeated-run benchmark drivers are lower-priority follow-ons

The repeated-run benchmark binaries remain comparatively compact:

- `benchmarks/bench_iterative_reuse.c` = `251`
- `benchmarks/bench_eigs_reuse.c` = `201`

Interpretation:

- they are real Sprint 47 auxiliary surfaces
- they do not justify driving the first parser/helper design
- they fit better into later helper-alignment only if a touched seam remains
  small and obvious

#### 6. The example surface is a bounded safety/helper follow-on class

The live example surface remains comparatively small:

- `examples/example_eigs.c` = `284`
- `examples/example_iterative.c` = `144`
- `examples/example_matrix_free.c` = `122`

Interpretation:

- examples should follow after the benchmark helper seam is stable
- Sprint 47 should keep the example batch focused on touched safety/helper
  patterns rather than broad presentation churn

#### 7. The script-side surfaces are support-code alignment work, not first-wave parser work

The inherited script-side surfaces are:

- `scripts/deadcode_report.py` = `523`
- `scripts/deadcode_workflow.sh` = `189`

And the live parser evidence is different from `bench_main.c`:

- `deadcode_report.py` already uses `argparse`
- `deadcode_workflow.sh` is mainly workflow/safety support code

Interpretation:

- these should land in the later auxiliary tooling batch
- they are not the main shared numeric-parser extraction problem

#### 8. The first implementation order is now fixed from the live code state

The correct order after Day 2 is:

1. shared parsing-helper design
2. shared parsing-helper implementation
3. `bench_main` parser modernization
4. reorder-mode / emitted-label parity cleanup
5. bounded example safety audit and cleanup
6. script-side support-code alignment
7. docs refresh and validation closeout

Interpretation:

- Sprint 47 should stay benchmark-helper-first through the front half
- examples and scripts should remain bounded follow-on surfaces rather than
  competing first-wave targets

## Day 3

**Objective:** Define the shared parsing-helper seam for benchmark/example
auxiliary CLIs so Sprint 47 can modernize `bench_main` first, keep
benchmark-specific semantics local, and avoid creating a broad public-facing or
framework-level CLI abstraction.

### Commands Run

1. Re-read the Sprint 47 Day 3 plan section:
   - `sed -n '94,127p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the Day 2 inventory artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day2-cli-and-auxiliary-surface-inventory.md`
3. Re-read the main `bench_main` argument loop and parse-related seam:
   - `sed -n '620,760p' benchmarks/bench_main.c`
4. Re-read the stronger parser reference surface in `bench_eigs.c`:
   - `sed -n '740,960p' benchmarks/bench_eigs.c`
5. Refresh the exact parser/helper markers across both benchmark binaries:
   - `rg -n "parse_|strtol|strtod|atoi|Usage:|--help|unknown option|reorder" benchmarks/bench_main.c benchmarks/bench_eigs.c`

### Day 3 Findings

#### 1. The correct shared helper seam is small and mechanical, not semantic

The live contrast is explicit:

- `bench_main.c` needs checked parse mechanics and consistent diagnostics
- `bench_eigs.c` already has stronger helpers, but still keeps command-specific
  mode semantics local

Interpretation:

- Sprint 47 should share:
  - positive/bounded integer parsing
  - finite double parsing
  - bounded mode-string matching patterns
- Sprint 47 should not try to centralize:
  - benchmark-specific usage text
  - backend/preconditioner semantics
  - command-specific aliases and policy

#### 2. Parse-plus-range-check should be one helper contract, not two caller steps

The Day 3 design should not split parsing into:

- string-to-number conversion
- then separate caller-side range enforcement

Interpretation:

- helpers should own both checked parse and the common lower-bound check
- this is the main way to prevent `bench_main.c` from recreating the same drift
  after helper adoption

#### 3. The helper layer should be internal-only

The live Sprint 47 targets are benchmark/example/tooling surfaces, not public
library APIs.

Interpretation:

- the shared parser helper seam should not live under `include/`
- it should be reusable by benchmark/example binaries without becoming a
  supported public API surface

#### 4. Reorder-mode parsing must stay caller-configured

Day 2 already showed that reorder-mode parity is a separate Sprint 47 seam.

Interpretation:

- the shared helper layer should not hard-code today's `bench_main` reorder
  set
- callers should supply the accepted strings and own final mode policy
- this preserves space for the Day 6 parity cleanup

#### 5. `bench_main` remains the first intended consumer, while `bench_eigs.c` remains partly local

The live parser evidence supports a bounded adoption model:

- `bench_main.c` should adopt the new shared helper seam directly
- `bench_eigs.c` should keep command-specific backend/preconditioner/mode logic
  local even if a small shared numeric parser seam later helps it

Interpretation:

- Day 5 can stay narrow and high-value
- Sprint 47 does not need to force both benchmark binaries into the same
  full parser structure

#### 6. The Day 3 design boundary is now fixed

Sprint 47 should explicitly avoid designing:

- a public CLI parsing API
- a benchmark framework abstraction layer
- a shared help/usage text renderer
- a generalized shell/workflow parsing system

Interpretation:

- the right Sprint 47 helper seam is intentionally small
- this preserves momentum toward Day 5 implementation without opening another
  architecture sprint

## Day 4

**Objective:** Fix the validation contract and peer-surface landing order for
Sprint 47 before code changes begin, so the shared parser-helper batch,
`bench_main` modernization, reorder-mode cleanup, example work, and auxiliary
tooling cleanup all inherit explicit validation rules and scope boundaries.

### Commands Run

1. Re-read the Sprint 47 Day 4 plan section:
   - `sed -n '127,161p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the Day 3 parser-helper design:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day3-shared-cli-parsing-helper-design.md`
3. Re-read the Sprint 40 validation anchor:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
4. Inspect the maintained benchmark/example build and quality wrapper surfaces:
   - `sed -n '130,220p' Makefile`
   - `rg -n "bench_main|bench-eigs|example|deadcode|quality-review-full|tooling-build" Makefile CMakeLists.txt README.md docs -g '!docs/planning/EPIC_4/SPRINT_47/**'`

### Day 4 Findings

#### 1. Sprint 47 should keep the same layered validation model as the earlier refactor sprints

The Sprint 40 validation anchor still applies cleanly:

- mandatory full gate for all `*.c` / `*.h` changes:
  - `make format`
  - `make lint`
  - `make test`
- stronger reviewed wrapper baseline for substantial batches:
  - `make quality-review-full`

Interpretation:

- Sprint 47 auxiliary cleanup does not weaken the core code gate
- substantial parser/helper/benchmark batches should still prove themselves
  against the stronger reviewed baseline when justified

#### 2. `tooling-build` is the right maintained compile-only auxiliary gate

The live `Makefile` already provides:

- `bench-build`
- `examples-build`
- `tooling-build`

Interpretation:

- Sprint 47 should use `make tooling-build` as the default compile-only
  benchmark/example follow-on for touched auxiliary code
- this gives honest compile coverage without pretending it proves full runtime
  behavior

#### 3. Direct CLI and binary reruns should stay targeted

The live surfaces suggest specific targeted checks, such as:

- `./build/bench_main --help`
- direct `bench_main` option sanity checks
- `./build/bench_eigs --help`
- touched example binary reruns when example behavior changes

Interpretation:

- Sprint 47 should add these only when the touched surface makes a usability or
  parity claim
- they are not a universal always-run expansion of the mandatory code gate

#### 4. `bench_eigs.c` remains a later bounded alignment surface, not a first-wave rewrite target

Day 4 reconfirms the Day 2 / Day 3 shape:

- `bench_eigs.c` already has the stronger checked parser style
- it is a good reference surface and possible later helper-alignment consumer
- it is not the main modernization hotspot

Interpretation:

- Sprint 47 should not dilute the early batches by trying to rewrite both main
  benchmark CLIs at once

#### 5. The repeated-run benchmark drivers and small examples remain later follow-on surfaces

The live maintained auxiliary build/docs surfaces still support the narrower
sequence:

- parser helper first
- `bench_main` second
- parity cleanup third
- examples and repeated-run binaries only after that shape is stable

Interpretation:

- Day 10 and Day 11 should remain bounded cleanup phases rather than broad
  peer-surface churn

#### 6. The out-of-scope boundary is now explicit before code edits begin

Sprint 47 should not expand into:

- benchmark framework redesign
- public CLI support APIs in the core library
- large README/tutorial restructuring
- dead-code workflow redesign

Interpretation:

- the sprint remains an auxiliary-surface modernization pass
- the implementation days can now proceed without architectural ambiguity

#### 7. The mid-sprint landing order is fixed

The correct post-Day-4 order is:

1. shared parsing-helper implementation
2. `bench_main` parser modernization
3. post-`bench_main` audit
4. reorder-mode / emitted-label parity cleanup
5. example safety audit and bounded cleanup
6. auxiliary tooling cleanup
7. docs refresh
8. full validation and closeout

Interpretation:

- the front half remains benchmark-helper-first
- examples/scripts/docs stay explicitly subordinate to the stabilized parser and
  parity shape

## Day 5

**Objective:** Land the bounded shared CLI parsing helper seam designed on Day 3
and Day 4, prove it in `bench_main.c` on the highest-signal parser-drift
arguments, and validate the batch from the full required gate through direct
`bench_main` sanity checks without widening yet into reorder-mode parity or
peer benchmark/example cleanup.

### Commands Run

1. Re-read the Sprint 47 Day 5 plan section:
   - `sed -n '102,133p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the Day 3 helper design and Day 4 landing contract:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day3-shared-cli-parsing-helper-design.md`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day4-validation-and-peer-surface-landing-design.md`
3. Inspect the maintained benchmark build wiring and current `bench_main`
   parser body:
   - `sed -n '130,210p' Makefile`
   - `sed -n '1,220p' CMakeLists.txt`
   - `sed -n '1,220p' benchmarks/bench_main.c`
   - `sed -n '220,420p' benchmarks/bench_main.c`
   - `sed -n '420,760p' benchmarks/bench_main.c`
4. Inspect the stronger parser/reference shape in `bench_eigs.c`:
   - `sed -n '1,220p' benchmarks/bench_eigs.c`
5. Land the Day 5 code batch:
   - `apply_patch` on:
     - `benchmarks/bench_cli_parse_internal.h`
     - `benchmarks/bench_main.c`
6. Run the mandatory gate plus the stronger reviewed wrapper baseline:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
7. Run direct touched-surface sanity checks:
   - `./build/bench_main --spmv --size 8 --spmv-iters 5 --repeat 2 --pivot partial`
   - `./build/bench_main --size 8 --repeat 1 --pivot partial`
   - `./build/bench_main --spmv-iters nope`

### Day 5 Findings

#### 1. A header-only internal helper seam was the right first landing

The maintained benchmark build surfaces already compile benchmark binaries
directly from their `.c` files.

Interpretation:

- a small internal header-only seam let Day 5 land the helper contract without
  broad build-wiring churn
- this kept the batch focused on parser behavior rather than infrastructure

#### 2. `bench_main.c` now proves the shared helper seam in the main parser-drift hotspot

The new helper seam in `benchmarks/bench_cli_parse_internal.h` now covers:

- checked bounded integer parsing
- checked positive integer parsing
- finite double parsing support for later batches
- caller-configured enum-like choice parsing

`bench_main.c` now uses that seam for:

- `--spmv-iters`
- `--size`
- `--repeat`
- `--pivot`

Interpretation:

- Day 5 landed real parser modernization, not only scaffolding
- the first proof is in exactly the file Day 1 and Day 2 identified as the main
  modernization hotspot

#### 3. Parse-plus-range-check is now one helper contract in the touched paths

The migrated argument paths no longer rely on:

- raw `atoi(...)`
- split parse-first / validate-later logic
- local ad hoc choice-branching for pivot mode

Interpretation:

- the Day 3 helper contract is now real in code
- Day 6 can widen `bench_main` from a stable checked-parse base instead of
  re-arguing helper shape

#### 4. The Day 5 boundary held: reorder-mode cleanup is still deferred

The batch intentionally left `--reorder` handling in its existing local shape.

Interpretation:

- Day 5 stayed within the planned shared-helper boundary
- Day 8 can still handle reorder-mode / emitted-label parity as its own bounded
  implementation step instead of piggybacking on the first helper landing

#### 5. Direct CLI proof already shows the right success and failure behavior

The touched parse paths were exercised directly:

- valid generated-matrix SpMV input completed successfully
- valid solve-path `--pivot partial` input completed successfully
- malformed `--spmv-iters nope` input failed with a flag-aware checked parse
  error

Interpretation:

- Day 5 proved both the success path and the malformed-input rejection path
- the helper seam is already improving observable auxiliary-surface behavior,
  not only internal code shape

#### 6. Validation stayed fully green for the touched helper batch

Because `*.c` and `*.h` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

Those all passed.

The stronger reviewed wrapper baseline also passed:

- `make quality-review-full`

Interpretation:

- the first helper landing did not regress the maintained reviewed contract
- Sprint 47 can continue widening the benchmark modernization work from a green
  Day 5 baseline

#### 7. Day 6 is now correctly constrained

After Day 5, the next correct step is:

- widen `bench_main` modernization from the shared helper seam

and not:

- broad `bench_eigs.c` rewrite
- reorder-mode parity cleanup early
- example or script-side churn

Interpretation:

- Sprint 47 remains helper-first and benchmark-main-first, exactly as intended

## Day 6

**Objective:** Widen the Day 5 helper landing into `bench_main.c` by replacing
the remaining ad hoc parser branches, tightening malformed-input and
unsupported-option handling, and validating the modernized main benchmark CLI
without widening yet into the later reorder-mode parity batch.

### Commands Run

1. Re-read the Sprint 47 Day 6 plan section:
   - `sed -n '193,221p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the Day 5 helper landing artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day5-shared-cli-parsing-helper-batch.md`
3. Re-read the live `bench_main.c` parser and dispatch body:
   - `sed -n '1,260p' benchmarks/bench_main.c`
   - `sed -n '260,520p' benchmarks/bench_main.c`
   - `sed -n '520,820p' benchmarks/bench_main.c`
4. Refresh the current parser drift markers and the peer help/unknown-option
   shape in `bench_eigs.c`:
   - `rg -n "unknown|help|Usage:|argv\\[i\\]\\[0\\] != '-'|--reorder|--dir|--cholesky|--iterative|--spmv" benchmarks/bench_main.c`
   - `rg -n "parse_choice|unknown option|--help|Usage:" benchmarks/bench_eigs.c`
5. Land the Day 6 code batch:
   - `apply_patch` on:
     - `benchmarks/bench_main.c`
6. Run the required gate plus the stronger reviewed baseline:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
7. Run direct touched-surface CLI sanity checks:
   - `./build/bench_main --help`
   - `./build/bench_main --reorder amd --size 8 --repeat 1`
   - `./build/bench_main --reorder`
   - `./build/bench_main --bogus`
   - `./build/bench_main --spmv --iterative`

### Day 6 Findings

#### 1. The remaining `bench_main` drift was real parser behavior, not just style

Before Day 6, `bench_main.c` still had:

- an ad hoc `--reorder` parse branch
- no explicit `--help` / `-h` handling
- weak missing-value behavior for string-valued options
- no consistent unknown-option rejection

Interpretation:

- Day 6 needed to be a real main-CLI behavior cleanup, not only a cosmetic
  helper-follow-on

#### 2. `bench_main` is now a full shared-helper consumer on the intended parser set

After the Day 6 batch, `bench_main.c` now routes:

- `--spmv-iters`
- `--size`
- `--repeat`
- `--pivot`
- `--reorder`

through the shared parsing helper seam or the same explicit value-requirement
shape.

Interpretation:

- the main benchmark CLI no longer mixes a modern helper path with a legacy
  mode-string parser on the highest-signal argument set

#### 3. Help and malformed-input behavior are now explicit

The Day 6 batch added:

- `--help`
- `-h`

and explicit missing-value failures for:

- `--spmv-iters`
- `--size`
- `--repeat`
- `--dir`
- `--pivot`
- `--reorder`

Interpretation:

- `bench_main` now behaves more like a maintained CLI surface and less like a
  thin benchmark harness with incidental parsing

#### 4. Unsupported and conflicting input reporting is now clearer

The modernized CLI now rejects:

- unknown options
- multiple positional matrix inputs
- mixed matrix path plus `--dir`
- mixed `--spmv` plus `--iterative`

Interpretation:

- Day 6 improved user-facing failure clarity without changing benchmark feature
  scope
- the batch stayed within parser/error-reporting modernization rather than
  becoming a capability redesign

#### 5. The direct CLI proof already shows the intended Day 6 contract

The touched surface was exercised directly:

- `--help` now prints usage and exits cleanly
- valid `--reorder amd` input completes successfully and reports `Reorder: amd`
- missing `--reorder` value fails with a clear message
- unknown option fails with a clear `try --help` message
- conflicting `--spmv` and `--iterative` flags fail with a clear exclusivity
  message

Interpretation:

- Day 6 proved both the positive path and the clearer malformed/unsupported
  input paths

#### 6. Validation stayed fully green for the broader main-CLI batch

Because `*.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

Those all passed.

The stronger reviewed baseline also passed:

- `make quality-review-full`

including the reviewed CMake parity path:

- `53 / 53` tests passed

Interpretation:

- the broader `bench_main` modernization did not regress the maintained local
  reviewed contract
- Sprint 47 can now move into the Day 7 audit and Day 8 parity batch from a
  fully green main-CLI baseline

#### 7. Day 7 is now correctly constrained

After Day 6, the next correct step is:

- audit the residual benchmark drift honestly

and not:

- reopen main parser modernization
- jump straight into broad peer benchmark/example churn

Interpretation:

- the sprint remains sequenced correctly: helper seam first, main CLI second,
  residual audit next, parity cleanup after that

## Day 7

**Objective:** Audit the post-Day-6 auxiliary state so Sprint 47 can separate
the real remaining reorder-mode parity work from lower-priority peer benchmark,
example, and script surfaces before the next implementation batch lands.

### Commands Run

1. Re-read the Sprint 47 Day 7 plan section:
   - `sed -n '221,260p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the Day 6 modernization artifact:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day6-bench-main-parser-modernization-batch.md`
3. Refresh the live auxiliary seam markers across the main benchmark/example/
   tooling surfaces:
   - `rg -n "colamd|reorder|Reorder:|--reorder|unknown option|--help|Usage:" benchmarks/bench_main.c benchmarks/bench_eigs.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_eigs.c examples/example_iterative.c examples/example_matrix_free.c scripts/deadcode_report.py scripts/deadcode_workflow.sh README.md docs -g '!docs/planning/EPIC_4/SPRINT_47/**'`
   - `wc -l benchmarks/bench_main.c benchmarks/bench_eigs.c benchmarks/bench_iterative_reuse.c benchmarks/bench_eigs_reuse.c examples/example_eigs.c examples/example_iterative.c examples/example_matrix_free.c scripts/deadcode_report.py scripts/deadcode_workflow.sh`
4. Re-read prior benchmark ownership and review notes relevant to reorder-surface
   scope:
   - `sed -n '24,36p' docs/planning/EPIC_3/reviews/todo-codex-2026-05-15.md`
   - `sed -n '640,670p' docs/planning/EPIC_3/SPRINT_37/WORKING_NOTES.md`
5. Re-read the specialized peer benchmark ownership surfaces:
   - `sed -n '1,220p' benchmarks/bench_reorder.c`
   - `sed -n '1,220p' benchmarks/bench_colamd.c`

### Day 7 Findings

#### 1. The remaining direct benchmark seam is now reorder-mode parity, not parser mechanics

After Day 6, `bench_main.c` already has:

- explicit help/usage handling
- explicit unknown-option handling
- shared-helper parsing for:
  - `--spmv-iters`
  - `--size`
  - `--repeat`
  - `--pivot`
  - `--reorder`

But the live main residual drift is still:

- `reorder_name()` supports `colamd`
- help text and accepted `--reorder` values still stop at:
  - `none`
  - `rcm`
  - `amd`
  - `nd`

Interpretation:

- Day 8 should be about supported-mode / emitted-label parity, not more parser
  refactoring

#### 2. `bench_reorder` still owns the broader reorder-comparison surface

The specialized reorder benchmark still explicitly owns:

- `none`
- `rcm`
- `amd`
- `colamd`
- `nd`

Interpretation:

- Sprint 47 must preserve that ownership boundary
- Day 8 should make `bench_main` honest and internally consistent, not turn it
  into a second general reorder-comparison harness

#### 3. `bench_colamd` remains a specialized QR/COLAMD comparison surface

The dedicated COLAMD benchmark still exists as its own bounded tool.

Interpretation:

- Day 8 should not blur the QR/COLAMD comparison ownership into `bench_main`
- any `bench_main` parity decision must respect that specialized surface

#### 4. `bench_eigs.c` remains a reference surface, not a required next target

The current `bench_eigs.c` already has:

- explicit usage/help text
- explicit unknown-option handling
- checked parse helpers

Interpretation:

- it remains useful as a CLI-shape reference
- it does not need to join the Day 8 batch unless something unexpectedly tiny
  falls out

#### 5. The rest of the auxiliary queue is now clearly later follow-on work

The remaining bounded auxiliary surfaces are:

- `bench_iterative_reuse.c`
- `bench_eigs_reuse.c`
- `example_eigs.c`
- `example_iterative.c`
- `example_matrix_free.c`
- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

Interpretation:

- the front half of Sprint 47 has done the narrowing work it needed to do
- Day 8 should not pull these surfaces forward

#### 6. Day 8 is now tightly bounded

The correct Day 8 target set is:

- align supported reorder modes and emitted reporting in the touched
  `bench_main` surface
- remove the current internal drift between:
  - `reorder_name()`
  - help/usage text
  - accepted `--reorder` values

The correct Day 8 non-goals are:

- no broad peer benchmark rewrite
- no helper-layer redesign
- no example or script cleanup yet

Interpretation:

- Sprint 47’s next implementation batch is now concrete rather than generic

## Day 8

**Objective:** Land the bounded reorder-mode / emitted-label parity cleanup in
`bench_main.c` by making the supported `--reorder` surface, printed labels, and
user guidance line up cleanly with the intended benchmark ownership split,
without broadening into peer benchmark rewrites or example/script work.

### Commands Run

1. Re-read the Sprint 47 Day 8 plan section:
   - `sed -n '237,268p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the Day 7 audit and the current benchmark ownership notes:
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day7-post-bench-main-audit.md`
   - `sed -n '1,90p' benchmarks/README.md`
   - `sed -n '650,666p' docs/planning/EPIC_3/SPRINT_37/WORKING_NOTES.md`
3. Re-read the live `bench_main` reorder surface and the older review note:
   - `sed -n '1,140p' benchmarks/bench_main.c`
   - `sed -n '632,760p' benchmarks/bench_main.c`
   - `sed -n '188,206p' docs/planning/EPIC_4/reviews/review-codex-2026-05-21.md`
4. Refresh the current parity markers and peer benchmark ownership surfaces:
   - `rg -n "bench_main --reorder|Reorder:|colamd|bench_reorder|bench_colamd" README.md docs benchmarks -g '!docs/planning/EPIC_4/SPRINT_47/**'`
   - `./build/bench_main --reorder colamd`
5. Land the Day 8 code batch:
   - `apply_patch` on:
     - `benchmarks/bench_main.c`
6. Run the required gate plus the stronger reviewed baseline:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
7. Run direct touched-surface CLI sanity checks:
   - `./build/bench_main --help`
   - `./build/bench_main --reorder nd --size 8 --repeat 1`
   - `./build/bench_main --reorder colamd`

### Day 8 Findings

#### 1. The right Day 8 fix was to clarify the main benchmark surface, not widen it

The current benchmark contract still intentionally splits reorder ownership:

- `bench_main`:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- `bench_reorder`:
  - broader `none|rcm|amd|colamd|nd` comparison surface
- `bench_colamd`:
  - QR/COLAMD-focused comparison surface

Interpretation:

- the correct Day 8 move was not “accept every enum in `bench_main`”
- the correct move was to make `bench_main`’s supported surface, labels, and
  guidance internally consistent

#### 2. `bench_main` no longer carries the internal `colamd` drift

Before Day 8:

- `reorder_name()` knew about `colamd`
- the parser and help text did not

After Day 8:

- `bench_main` uses a main-benchmark-specific reorder label path
- help/usage text and runtime labels now align with the supported main surface
- unsupported `colamd` input now points users to:
  - `bench_reorder`
  - `bench_colamd`

Interpretation:

- the remaining internal drift identified on Day 7 is now gone

#### 3. The direct CLI proof shows the intended parity behavior

The touched surface was exercised directly:

- `--help` now documents the supported main-benchmark reorder set and the
  specialized COLAMD handoff
- valid `--reorder nd` input succeeds and reports `Reorder: nd`
- unsupported `--reorder colamd` input fails with an explicit handoff message

Interpretation:

- Day 8 improved the user-facing benchmark contract without changing the
  benchmark capability boundary

#### 4. The peer benchmark ownership boundary held

No Day 8 changes were needed in:

- `bench_reorder.c`
- `bench_colamd.c`
- `bench_eigs.c`

Interpretation:

- the batch stayed narrow and respected the ownership split confirmed on Day 7
- Sprint 47 did not drift back into broad peer benchmark churn

#### 5. Validation stayed fully green for the touched parity batch

Because `*.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

Those passed.

The stronger reviewed baseline also passed:

- `make quality-review-full`

Interpretation:

- the parity cleanup did not regress the maintained local reviewed contract
- Sprint 47 can now move into the later example/tooling/docs queue from a clean
  benchmark front-half baseline

## Day 9

**Objective:** Audit the example surface for unchecked arithmetic, weak helper
patterns, and stale auxiliary conventions after the Day 8 benchmark-front-half
landing, then choose a bounded Day 10 cleanup batch instead of carrying a
generic "example cleanup" backlog.

### Commands Run

1. Re-read the Sprint 47 Day 9-10 plan section:
   - `sed -n '269,340p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the touched small-example surfaces and the current example allocation
   helper seam:
   - `sed -n '1,260p' examples/example_eigs.c`
   - `sed -n '1,240p' examples/example_iterative.c`
   - `sed -n '1,220p' examples/example_matrix_free.c`
   - `sed -n '1,220p' examples/example_alloc_helpers.h`
3. Refresh the broader example queue and raw-allocation markers:
   - `rg --files examples`
   - `rg -n "malloc\\(|calloc\\(|strtol|strtod|atoi|example_[mc]alloc_array|argc|argv|SPARSE_ERR_ALLOC|sqrt\\(\\(double\\)n\\)" examples`
4. Re-read the strongest remaining raw-allocation candidates and the already
   aligned COLAMD example:
   - `sed -n '1,220p' examples/example_analysis.c`
   - `sed -n '1,220p' examples/example_condition.c`
   - `sed -n '1,280p' examples/example_ic_minres.c`
   - `sed -n '220,360p' examples/example_eigs.c`
   - `sed -n '1,220p' examples/example_colamd.c`
5. Re-read the current example README contract:
   - `sed -n '1,220p' examples/README.md`

### Day 9 Findings

#### 1. The example queue is narrower than a generic cleanup backlog

The examples now fall into three practical classes:

- already aligned to the current helper/safety direction
- clear direct shared-helper adoption targets
- larger raw-allocation examples that would turn Day 10 into a broad rewrite

Interpretation:

- the right Day 10 move is a narrow helper-adoption batch
- the wrong move is to treat every example with a raw `malloc` or `calloc` as
  an equal-priority Sprint 47 target

#### 2. Three current examples are already aligned enough to leave alone

These examples already use `examples/example_alloc_helpers.h` where dynamic
scratch is part of the public example story:

- `example_iterative.c`
- `example_matrix_free.c`
- `example_colamd.c`

Interpretation:

- they should remain intentionally untouched on Day 10 unless a very small
  follow-on falls out for free

#### 3. `example_eigs.c` is the strongest direct Day 10 target

`example_eigs.c` still repeats raw allocation shapes such as:

- `calloc((size_t)n * (size_t)k, sizeof(double))`
- `malloc((size_t)n * sizeof(double))`

across multiple sub-demos.

Interpretation:

- it is the cleanest direct shared-helper adoption candidate
- the cleanup is mostly helper adoption rather than algorithm churn

#### 4. The remaining raw-allocation examples are real, but not the right first batch

The strongest later raw-allocation surfaces are:

- `example_ic_minres.c`
- `example_analysis.c`
- `example_condition.c`

Interpretation:

- they do contain real safety/helper cleanup opportunities
- they are not the right first Sprint 47 batch because they either widen the
  scope too much or offer too little payoff for the churn

#### 5. Day 10 is now bounded

Primary target:

- `examples/example_eigs.c`

Allowed tiny follow-on only if it stays obviously narrow:

- one helper-seam adoption in `example_condition.c`

Explicit non-targets:

- `example_ic_minres.c`
- `example_analysis.c`
- `example_iterative.c`
- `example_matrix_free.c`
- `example_colamd.c`
- broad example README churn

Interpretation:

- Sprint 47 now has a concrete example cleanup day rather than another generic
  auxiliary bucket

## Day 10

**Objective:** Land the bounded Day 9 example cleanup by aligning
`example_eigs.c` to the current example allocation-helper seam, without
broadening into larger multi-demo examples or unrelated benchmark/script churn.

### Commands Run

1. Re-read the primary target and current helper seam:
   - `sed -n '1,340p' examples/example_eigs.c`
   - `sed -n '1,220p' examples/example_alloc_helpers.h`
   - `sed -n '341,420p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Land the bounded Day 10 code batch:
   - `apply_patch` on:
     - `examples/example_eigs.c`
3. Run the required code-quality gate and the auxiliary build/runtime checks:
   - `make format`
   - `make lint`
   - `make test`
   - `make tooling-build`
   - `./build/example_eigs`

### Day 10 Findings

#### 1. The right Day 10 move was helper adoption, not example redesign

`example_eigs.c` had a clear repeated allocation pattern across its three
sub-demos:

- eigenvector bundles
- per-demo `A*v` scratch vectors
- repeated raw `malloc` / `calloc` usage

Interpretation:

- the correct Sprint 47 cleanup was to route those allocations through the
  existing example helper seam
- the batch did not need algorithm or output redesign

#### 2. `example_eigs.c` now follows the shared example allocation seam

The file now includes `example_alloc_helpers.h`, and the touched dynamic
buffers now allocate through:

- `example_calloc_array(...)`
- `example_malloc_array(...)`

Covered buffers:

- `vecs`
- `Av`
- `kvecs`
- `KAv`
- `bvecs`
- `BAv`

Interpretation:

- the strongest direct shared-helper adoption target from Day 9 is now complete

#### 3. The multi-vector bundles avoid pre-multiplied count drift

The cleanup kept the row count as the helper count argument and encoded the
fixed sub-vector width into the element size where appropriate:

- `example_calloc_array(n, sizeof(double[5]), ...)`
- `example_calloc_array(nk, sizeof(double[3]), ...)`
- `example_calloc_array(nb, sizeof(double[3]), ...)`

Interpretation:

- the touched example is safer than the original raw allocation form rather
  than merely cosmetically different

#### 4. Validation and direct runtime proof both stayed green

Because `*.c` changed, the required gate was:

- `make format`
- `make lint`
- `make test`

Those passed.

Auxiliary build/runtime checks also passed:

- `make tooling-build`
- `./build/example_eigs`

Interpretation:

- the helper adoption did not regress the maintained code-quality baseline
- the public example runtime behavior remained intact

#### 5. The Day 9 defer/keep boundary held

No Day 10 changes were needed in:

- `example_iterative.c`
- `example_matrix_free.c`
- `example_colamd.c`
- `example_analysis.c`
- `example_condition.c`
- `example_ic_minres.c`

Interpretation:

- the example batch stayed bounded
- Sprint 47 did not turn into a broad example rewrite

## Day 11

**Objective:** Land the bounded auxiliary tooling cleanup by tightening the
dead-code workflow support path around malformed coverage metadata and malformed
compile-database entries, without redesigning the broader dead-code workflow
contract.

### Commands Run

1. Re-read the Sprint 47 Day 11 plan section:
   - `sed -n '309,390p' docs/planning/EPIC_4/SPRINT_47/PLAN.md`
2. Re-read the primary tooling targets and the Sprint 47 inventory/design notes:
   - `sed -n '1,260p' scripts/deadcode_report.py`
   - `sed -n '1,260p' scripts/deadcode_workflow.sh`
   - `sed -n '1,260p' docs/planning/EPIC_4/SPRINT_47/artifacts/day2-cli-and-auxiliary-surface-inventory.md`
   - `sed -n '1,220p' docs/planning/EPIC_4/SPRINT_47/artifacts/day4-validation-and-peer-surface-landing-design.md`
3. Re-read one existing bounded benchmark support seam for scope calibration:
   - `sed -n '1,220p' benchmarks/bench_backend_compare_helpers.h`
4. Refresh the live auxiliary weak-pattern markers:
   - `rg -n "atoi|strtol|strtod|int\\(|assert |compile_commands_json|missing_benchmarks|missing_examples" scripts benchmarks examples -g '!docs/**'`
5. Land the bounded Day 11 tooling batch:
   - `apply_patch` on:
     - `scripts/deadcode_report.py`
     - `scripts/deadcode_workflow.sh`
6. Run targeted touched-tool validation:
   - `python3 -m py_compile scripts/deadcode_report.py`
   - `bash -n scripts/deadcode_workflow.sh`
   - synthetic valid artifact round-trip through:
     - `python3 scripts/deadcode_report.py <tmpdir>`
     - `python3 scripts/deadcode_report.py --check <tmpdir>`
   - synthetic malformed coverage-note rejection via:
     - `parse_coverage_notes(...)` on a bad temp file

### Day 11 Findings

#### 1. The right Day 11 target was the dead-code metadata path

The strongest live script-side safety seam was the dead-code support path:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

Interpretation:

- Sprint 47 did not need a broad workflow rewrite
- it needed stricter validation around malformed support metadata

#### 2. `deadcode_report.py` now rejects malformed coverage-note input explicitly

The Day 11 batch tightened `parse_coverage_notes(...)` so it now:

- parses non-negative counts through an explicit helper
- rejects malformed section entries
- rejects unrecognized coverage-note lines
- requires a `compile_commands_json` line

Interpretation:

- malformed coverage metadata now fails with a clearer contract instead of
  depending on weaker implicit parsing assumptions

#### 3. `deadcode_workflow.sh` now validates compile-database shape more clearly

The embedded Python coverage-note generator now rejects:

- invalid JSON
- non-array compile databases
- non-object entries
- entries missing `file`
- relative entries missing a usable `directory`

Interpretation:

- the workflow now fails earlier and more clearly when the compile database is
  malformed

#### 4. Script-level validation covered both success and failure paths

The touched-tool validation covered:

- Python syntax compilation
- shell syntax validation
- a synthetic valid artifact round-trip
- a synthetic malformed coverage-note rejection path

Interpretation:

- Day 11 is grounded in direct support-code checks, not only code inspection

#### 5. The batch stayed bounded

No Day 11 changes were needed in:

- peer benchmark drivers
- examples
- broader dead-code semantics or workflow topology

Interpretation:

- Sprint 47 remained in a narrow tooling-safety lane rather than drifting into
  framework redesign
