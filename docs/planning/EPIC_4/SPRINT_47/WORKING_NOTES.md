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
