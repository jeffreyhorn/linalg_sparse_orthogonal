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
