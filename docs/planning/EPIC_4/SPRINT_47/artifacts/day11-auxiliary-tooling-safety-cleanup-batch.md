## Sprint 47 Day 11: Auxiliary Tooling Safety Cleanup Batch

### Objective

Land the bounded auxiliary tooling cleanup by tightening the dead-code workflow
support path around malformed coverage metadata and malformed compile-database
entries, without redesigning the broader dead-code workflow contract.

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

### Findings

#### 1. The right Day 11 target was the dead-code metadata path, not broader script redesign

The strongest live script-side safety seam was the dead-code support path:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

Those files already had the right broad structure:

- `argparse` on the Python side
- explicit shell workflow staging on the bash side

The real cleanup need was narrower:

- malformed coverage-note handling
- malformed compile-database structure handling
- clearer failure behavior when support metadata is wrong

Interpretation:

- Sprint 47 did not need a broad workflow rewrite
- it needed stricter input validation in the touched support-code seam

#### 2. `deadcode_report.py` now rejects malformed coverage-note metadata explicitly

The Day 11 batch tightened `parse_coverage_notes(...)` so it now:

- parses non-negative count fields through an explicit helper
- rejects malformed section entries instead of silently falling through
- rejects unknown/unrecognized coverage-note lines
- requires a `compile_commands_json` line to be present

This replaces a weaker path that previously relied more on:

- implicit `int(...)` conversion
- section-state assumptions
- generic `assert`-style expectations elsewhere in the file

Interpretation:

- malformed dead-code support metadata now fails with a clearer contract
- the report path is less likely to accept or obscure bad staging inputs

#### 3. `deadcode_workflow.sh` now validates compile-database shape more defensibly

The embedded Python coverage-note generator now:

- rejects invalid JSON in `compile_commands.json`
- requires the top-level JSON value to be an array
- requires each entry to be an object
- requires each entry to provide a `file`
- requires relative-path entries to provide a usable `directory`

Interpretation:

- the workflow now fails earlier and more clearly when the compile database is
  malformed
- Day 11 improved support-code safety without changing the workflow topology

#### 4. The direct touched-tool checks proved both success and failure paths

The script-level validation covered:

- Python syntax compilation for `deadcode_report.py`
- shell syntax validation for `deadcode_workflow.sh`
- a synthetic valid artifact round-trip:
  - report generation
  - report self-check
- a synthetic malformed coverage-note failure path:
  - invalid count rejected as expected

Interpretation:

- the Day 11 claims are grounded in direct support-code checks rather than only
  static reading

#### 5. The batch stayed narrow and left the broader auxiliary queue intact

No Day 11 changes were needed in:

- `bench_eigs.c`
- `bench_iterative_reuse.c`
- `bench_eigs_reuse.c`
- `examples/`
- broader dead-code target semantics

Interpretation:

- Sprint 47 stayed in a bounded tooling-safety lane
- the batch did not drift into benchmark framework or workflow redesign

### Bottom Line

Sprint 47 Day 11 tightened the dead-code auxiliary support seam in the right
place:

- touched tooling targets:
  - `scripts/deadcode_report.py`
  - `scripts/deadcode_workflow.sh`
- improved behaviors:
  - stricter coverage-note validation
  - stricter compile-database structure validation
  - clearer malformed-input failure paths

The batch stayed bounded, and the targeted touched-tool validation proved both
the success path and the intended malformed-input rejection path.
