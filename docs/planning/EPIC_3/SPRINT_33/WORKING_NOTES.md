# Sprint 33 Working Notes

## Day 1

**Objective:** Convert the Sprint 32 handoff into a precise Sprint 33 dead-code baseline by confirming the inherited clean-state invariants, auditing the current Makefile/CMake/static-analysis entry points, generating a dedicated `compile_commands.json` path for later `xunused` work, and naming the first low-risk cleanup surfaces before any code removal begins.

### Commands Run

1. Read Sprint 33 scope and Sprint 32 handoff inputs:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_33/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/HANDOFF.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/RETROSPECTIVE.md`
2. Inspect current build and analysis entry points:
   - `rg -n "deadcode|cppcheck|clang-tidy|compile_commands|xunused" Makefile CMakeLists.txt scripts tests src include docs/planning/EPIC_3 -g '!docs/planning/EPIC_3/SPRINT_33/**'`
   - `sed -n '1,260p' Makefile`
   - `sed -n '400,470p' Makefile`
   - `sed -n '1,280p' CMakeLists.txt`
   - `find build -maxdepth 3 \( -name compile_commands.json -o -name '*.json' \) | sort`
3. Reconfirm the current validated test-registry baseline and local tool availability:
   - `ctest -N --test-dir build/sprint32-day1-cmake`
   - `command -v cppcheck || true; command -v xunused || true; command -v clang-tidy || true`
4. Generate a dedicated Sprint 33 compilation database path:
   - `cmake -S . -B build/sprint33-day1-cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
   - `wc -l build/sprint33-day1-cmake/compile_commands.json`
   - `grep -c '"file":' build/sprint33-day1-cmake/compile_commands.json`
   - `sed -n '1,20p' build/sprint33-day1-cmake/compile_commands.json`
5. Derive baseline scope counts and coverage gaps:
   - `find src -maxdepth 1 -name '*.c' | wc -l`
   - `find tests -maxdepth 1 -name '*.c' | wc -l`
   - `find benchmarks -maxdepth 1 -name '*.c' | wc -l`
   - `find examples -maxdepth 1 -name '*.c' | wc -l`
   - `grep '"file":' build/sprint33-day1-cmake/compile_commands.json | awk '...'`
   - `printf '%s\n' benchmarks/*.c | sed 's#benchmarks/##; s#\.c$##' | sort > /tmp/s33_bench_all.txt`
   - `rg -o "add_executable\\((bench_[^ )]+)" -r '$1' CMakeLists.txt | sort > /tmp/s33_bench_cmake.txt`
   - `comm -23 /tmp/s33_bench_all.txt /tmp/s33_bench_cmake.txt`
   - `printf '%s\n' examples/*.c | sed 's#examples/##; s#\.c$##' | sort > /tmp/s33_ex_all.txt`
   - `rg -o "add_executable\\((example_[^ )]+)" -r '$1' CMakeLists.txt | sort > /tmp/s33_ex_cmake.txt`
   - `comm -23 /tmp/s33_ex_all.txt /tmp/s33_ex_cmake.txt`
   - `ls src/*internal*.h src/*errno*.h 2>/dev/null | sort`
   - `rg -n "static [^;]*\\(" tests benchmarks examples src -g '*.c' -g '*.h' | head -n 120`

### Baseline Findings

- Sprint 33 starts from the validated Sprint 32 clean-state contract, not from residual warning debt:
  - full-tree warnings: `0`
  - dormant-scaffold debt: `0`
  - active `ctest` registry: `53`
  - opt-in truthfulness policy already in force through `RUN_TEST_SLOW(...)`, `RUN_TEST_EXPERIMENTAL(...)`, `SPARSE_TEST_SLOW=1`, and `SPARSE_TEST_EXPERIMENTAL=1`
- Current branch head at Day 1 baseline capture: `ff3cfe6`
- There is no existing dead-code Makefile support yet:
  - no `deadcode`
  - no `deadcode-report`
  - no `deadcode-check`
- Existing static-analysis flow today is still `lint`-centric:
  - `tooling-build`
  - strict `src/*.c` compile with `-Werror`
  - `clang-tidy` over `src/*.c`
  - `cppcheck` over `src/` and `tests/`
- Before Day 1, there was no `compile_commands.json` in any existing build tree under `build/`.
- A dedicated Day 1 CMake configure path now exists:
  - `build/sprint33-day1-cmake/compile_commands.json`
  - total translation units recorded: `97`
- Local tool availability at Day 1:
  - `cppcheck`: present at `/usr/local/bin/cppcheck`
  - `clang-tidy`: present at `/usr/local/opt/llvm/bin/clang-tidy`
  - `xunused`: not installed / not found in `PATH`

### Scope Counts

Repository `.c` file counts at Day 1:

- `src`: `25`
- `tests`: `54`
- `benchmarks`: `14`
- `examples`: `12`

Day 1 `compile_commands.json` coverage by area:

- `src`: `25`
- `tests`: `53`
- `benchmarks`: `13`
- `examples`: `6`

Interpretation:

- the compilation database fully covers the library sources
- it covers the current active CTest registry exactly (`53` test translation units)
- it does **not** cover the full Makefile-only tooling surface yet

### Coverage Gaps Relevant To Sprint 33

Compared with the current Makefile source lists and directories:

- benchmark source present in the repo but absent from CMake `compile_commands.json`:
  - `bench_svd`
- example sources present in the repo but absent from CMake `compile_commands.json`:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Day 1 implication:

- a naive `xunused build/compile_commands.json` pass would reason about a narrower bench/example surface than the Makefile currently builds
- Sprint 33 should document or resolve that coverage mismatch before treating `xunused` output as a complete first-pass dead-code signal

### Likely First-Pass Candidate Areas

These are **candidate audit surfaces**, not Day 1 dead-code findings:

1. `tests/`
   - largest low-risk internal area by file count (`54` `.c` files)
   - already governed by Sprint 32 truthfulness constraints
   - must preserve active and opt-in test coverage, including `tests/test_framework_optin.c`
2. `benchmarks/` and `examples/`
   - likely place for private one-off helpers, legacy fixture code, and stale experiment scaffolding
   - need careful handling because current `compile_commands.json` under-covers them relative to the Makefile
3. private helper layer in `src/`
   - internal-header surfaces currently include:
     - `src/sparse_analysis_internal.h`
     - `src/sparse_bicgstab_internal.h`
     - `src/sparse_chol_csc_internal.h`
     - `src/sparse_colamd_internal.h`
     - `src/sparse_eigs_internal.h`
     - `src/sparse_errno_internal.h`
     - `src/sparse_graph_internal.h`
     - `src/sparse_ldlt_csc_internal.h`
     - `src/sparse_matrix_internal.h`
     - `src/sparse_reorder_amd_qg_internal.h`
     - `src/sparse_reorder_nd_internal.h`
     - `src/sparse_svd_internal.h`
   - these are appropriate later cleanup candidates only after Sprint 33 establishes the tool/report policy
4. candidate public API or documented surface
   - must remain a separate audit queue
   - not appropriate for the first “definitely-unused internal code” removal pass

### Day 1 Interpretation

- Day 1 did **not** discover inherited cleanup debt from Sprint 32; it confirmed that Sprint 33 starts from a clean quality baseline and a live truthfulness policy.
- The first real Sprint 33 infrastructure gap is not warning debt but tooling completeness:
  - no dead-code targets exist yet
  - `xunused` is currently a local prerequisite gap
  - `compile_commands.json` must be generated deliberately rather than assumed
- The second key Day 1 finding is scope coverage mismatch:
  - CMake gives Sprint 33 a usable `compile_commands.json`
  - but that database does not yet cover every benchmark/example source the Makefile builds
- That means Day 2 and Day 3 should define the policy and evidence standard before any deletion work begins, and Day 4/Day 5 must treat compilation-database coverage as part of the infrastructure task, not as an afterthought.

### Day 1 Outputs

- `artifacts/day1-dead-code-baseline.md`
- `artifacts/day1-scope-counts.txt`
- `artifacts/day1-tooling-inventory.txt`

## Day 2

**Objective:** Audit the repository’s dead-code policy boundaries precisely enough to separate live active/opt-in coverage from historical evidence, public/documented surface from internal-only surface, and documentation debt from actual removable code before Sprint 33 starts implementing dead-code targets.

### Commands Run

1. Re-read the Sprint 33 baseline and Sprint 32 truthfulness policy:
   - `git status --short --branch`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_33/WORKING_NOTES.md`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_32/artifacts/day3-test-truthfulness-policy.md`
   - `sed -n '560,630p' README.md`
2. Inspect the live opt-in test framework surface:
   - `sed -n '1,260p' tests/test_framework_optin.c`
   - `sed -n '1,260p' tests/test_framework.h`
   - `rg -n "RUN_TEST\\(|RUN_TEST_SLOW\\(|RUN_TEST_EXPERIMENTAL\\(|SKIP_TEST\\(|commented-out|historical|retired|advisory|experimental|slow" tests docs/planning README.md`
   - `rg -n "//.*RUN_TEST|/\\*.*RUN_TEST|RUN_TEST\\(.*\\).*//|^\\s*//\\s*RUN_TEST" tests -g '*.c'`
3. Inspect public-surface and documentation-bearing files:
   - `find include -maxdepth 1 -name '*.h' | sort`
   - `find examples -maxdepth 1 -name '*.c' | sort`
   - `sed -n '1,220p' examples/README.md`
   - `sed -n '1,160p' benchmarks/README.md`
   - `rg -n "example_basic_solve|example_condition|example_iterative|example_least_squares|example_matrix_free|example_svd_lowrank|example_ldlt|example_ic_minres|example_analysis|example_minnorm|example_colamd|example_eigs|bench_svd" README.md benchmarks/README.md CMakeLists.txt Makefile docs include examples`
4. Sample ambiguous “stub” / “future” narrative sites in active tests:
   - `sed -n '600,720p' tests/test_graph.c`
   - `sed -n '940,990p' tests/test_eigs.c`
   - `sed -n '780,820p' tests/test_reorder_nd.c`
   - `sed -n '288,316p' tests/test_sprint20_integration.c`
   - `rg -n "TODO|FIXME|historical|advisory only|retired|dead code|unused|stub|not yet wired|future" src tests benchmarks examples include README.md docs/planning/EPIC_3 -g '!docs/planning/EPIC_3/SPRINT_33/**' | head -n 240`
5. Refresh concrete counts for the audit note:
   - `printf 'run_test ... run_test_slow ... run_test_experimental ... skip_test ...'`
   - `printf 'commented_run_test ...'`
   - `printf 'public_headers ... examples ... bench_docs_mentions ... example_docs_mentions ...'`
   - `rg -n "^static inline|^static [^;]*\\(" src/*internal*.h src/*.c tests/*.c benchmarks/*.c examples/*.c -g '!build/**' | head -n 200`

### Day 2 Audit Findings

#### 1. Sprint 32 truthfulness rules remain the first policy boundary

- The repo still has `0` commented-out `RUN_TEST(...)` registrations in `tests/`.
- The current test surface includes:
  - `RUN_TEST(...)`: `1725` call sites
  - `RUN_TEST_SLOW(...)`: `2` call sites
  - `RUN_TEST_EXPERIMENTAL(...)`: `2` call sites
  - `SKIP_TEST(...)`: `1` call site
- The opt-in wrappers are currently exercised by `tests/test_framework_optin.c`, which is live registered coverage under both the Makefile and CTest paths.

Interpretation:

- Sprint 33 must not classify opt-in test support or its self-check coverage as dormant scaffold.
- A dead-code report that only sees local reachability or unusual env-var gating could mislabel this area unless the policy explicitly says active registration beats “rarely enabled” semantics.

#### 2. Comment-level “stub” language is not enough to call code dead

Sampled active tests still carry historical sprint-language comments such as:

- `tests/test_graph.c::test_graph_subgraph_is_stub`
- `tests/test_sprint20_integration.c` eigs validation comments discussing the earlier stub phase
- `tests/test_eigs.c` comments describing a prior “failing-as-expected stub” state
- `tests/test_reorder_nd.c` comments that explicitly say the historical dormant scaffold was already retired into docs

Interpretation:

- these are documentation/history traces inside actively registered tests
- they are **not** the same as Sprint 32’s deleted dormant registrations
- Sprint 33 should treat active registration plus current assertions as authoritative, even when function names or surrounding comments still contain “stub”, “future”, or historical sprint language

#### 3. Public and documented surface must be audited separately from internal cleanup

Current exported/documented surface at Day 2:

- installed public headers: `18`
- example source files: `12`
- benchmark table rows documented in `benchmarks/README.md`: `12`
- explicit example sections documented in `examples/README.md`: `5`

Important Day 1 / Day 2 crossover finding:

- `bench_svd` is documented in `benchmarks/README.md` and built by the Makefile, even though it is absent from the current CMake `compile_commands.json`
- `example_basic_solve`, `example_condition`, `example_iterative`, `example_least_squares`, `example_matrix_free`, and `example_svd_lowrank` are documented public examples even though they are absent from the current CMake `compile_commands.json`

Interpretation:

- absence from `compile_commands.json` is **not** evidence of dead code
- absence from current CMake coverage and absence from local callers are separate facts
- Sprint 33 must treat installed headers, documented examples, and documented benchmark entry points as manual-review surfaces, not first-pass deletion targets

#### 4. Likely “definitely-unused” candidates remain internal-only by policy, not yet by proof

The low-risk candidate classes remain:

- static functions and helpers in `tests/`, `benchmarks/`, and `examples/`
- private helpers in `src/*.c`
- helper declarations in `src/*internal*.h`

But Day 2 does **not** yet promote any named helper to “definitely-unused” because:

- the dead-code tooling has not been implemented
- `xunused` is still unavailable locally
- the compilation-database coverage is incomplete for part of the tooling surface
- local grep evidence alone is not enough for deletion when build-path coverage is known to be partial

#### 5. Repository dead-code policy needs an explicit evidence hierarchy

Day 2 makes the needed ordering clear:

1. Active test registration and documented public surface override superficial “unused-looking” signals.
2. Documentation/history comments do not by themselves create dead code.
3. Definitely-unused internal code needs stronger evidence than “not in local call sites”.
4. Candidate public API or documented tooling surface must be routed to a separate manual-review queue.

### Day 2 Classification Rules To Carry Into Day 3

#### Keep-active / keep-live

- any function or file reached through active `RUN_TEST(...)`, `RUN_TEST_SLOW(...)`, or `RUN_TEST_EXPERIMENTAL(...)`
- `tests/test_framework_optin.c` and the wrapper macros in `tests/test_framework.h`
- any installed public header in `include/`
- any documented example or benchmark entry point, even if absent from the current `compile_commands.json`

#### Candidate internal cleanup only after tool evidence

- unexported `static` helpers in `src/*.c`, `tests/*.c`, `benchmarks/*.c`, and `examples/*.c`
- private declarations in `src/*internal*.h`
- internal bench/example helpers whose code is not part of a documented user-facing contract

#### Docs/comment cleanup, not dead-code cleanup

- active tests that still use “stub”, “future”, or sprint-history wording in comments
- narrative references to retired or advisory behavior when the executable test remains active and truthful

#### Separate manual-review queue

- exported/public headers
- documented examples
- documented benchmark entry points
- any CLI or API surface named in README or other maintainer/user docs

### Day 2 Interpretation

- Sprint 33’s first policy job is narrower than “find all unused things”: it must prevent the new tooling from collapsing four distinct cases into one bucket:
  - live active coverage
  - live opt-in coverage
  - historical/doc evidence
  - candidate internal dead code
- The most important false-positive class is now clear:
  - public or documented code that is under-covered by `compile_commands.json`
- The second important false-positive class is also clear:
  - active tests whose surrounding comments still mention earlier stub states
- Day 3 should therefore formalize an evidence hierarchy and limitations note before Day 4 / Day 5 implement any `deadcode` target.

### Day 2 Outputs

- `artifacts/day2-dead-code-policy-audit.md`

## Day 3

**Objective:** Convert the Day 2 audit into a repository-level dead-code policy that Sprint 33 can apply consistently: define the code classes, capture confirmed and expected tooling limitations, and set a conservative deletion threshold for “definitely-unused internal code.”

### Commands Run

1. Re-read the Sprint 33 Day 2 outputs and Sprint 33 project-plan scope:
   - `git status --short --branch`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_33/PLAN.md`
   - `sed -n '145,360p' docs/planning/EPIC_3/SPRINT_33/WORKING_NOTES.md`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_33/artifacts/day2-dead-code-policy-audit.md`
   - `sed -n '120,170p' docs/planning/EPIC_3/PROJECT_PLAN.md`
2. Tighten the tooling-limitations section with direct local evidence:
   - `cppcheck --help | sed -n '1,220p'`
   - `grep '"file":' build/sprint33-day1-cmake/compile_commands.json | awk '...'`
3. Reconfirm the active truthfulness policy language that Sprint 33 must preserve:
   - `sed -n '560,630p' README.md`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_32/artifacts/day3-test-truthfulness-policy.md`

### Day 3 Policy Decisions

#### 1. Sprint 33 dead-code policy is evidence-based, not grep-based

The repository will not treat “no obvious local caller” as enough evidence for deletion.

Reason:

- the codebase already has multiple entry paths:
  - Makefile
  - CMake / CTest
  - documented examples
  - documented benchmark binaries
  - env-gated opt-in test registration
- the current compilation database does not cover the full bench/example surface

Day 3 consequence:

- “unused-looking” code is only a candidate until the policy checks, documentation checks, and validation checks all agree

#### 2. The code classes for Sprint 33 are now explicit

Sprint 33 uses these categories:

1. **Active code**
   - current default behavior
   - active tests registered with `RUN_TEST(...)`
   - code required by `make test`, `ctest`, normal examples, or documented benchmark flows
2. **Live opt-in code**
   - current supported behavior behind `RUN_TEST_SLOW(...)`, `RUN_TEST_EXPERIMENTAL(...)`, or other documented non-default execution knobs
   - includes the policy self-check file `tests/test_framework_optin.c`
3. **Historical evidence**
   - sprint artifacts, planning docs, retired-target evidence, and explanatory notes
   - belongs in `docs/planning/`, not as dormant compiled registrations
4. **Documented public surface**
   - installed headers in `include/`
   - documented examples
   - documented benchmark entry points and named CLI flows
5. **Candidate unused public API**
   - exported or documented surface with no obvious in-repo callers
   - requires manual review and is not part of the first deletion pass
6. **Definitely-unused internal code**
   - internal-only code with no active registration, no public/documented contract, and sufficient corroborating evidence to delete safely
7. **Comment/documentation debt**
   - active code with outdated “stub”, “future”, or sprint-history wording
   - not dead code by itself

#### 3. Confirmed Day 3 tooling limitations

The locally confirmed limitations are:

- `cppcheck --enable=all` includes `unusedFunction`, and its own help recommends that check only when the whole program is scanned.
- Sprint 33’s current Day 1 compilation database covers:
  - `src`: `25`
  - `tests`: `53`
  - `benchmarks`: `13`
  - `examples`: `6`
- That coverage is narrower than the repository source surface:
  - `bench_svd` is missing
  - `example_basic_solve`, `example_condition`, `example_iterative`, `example_least_squares`, `example_matrix_free`, and `example_svd_lowrank` are missing
- `xunused` is not installed locally yet, so its behavior is still a planned-input limitation rather than a Day 3 exercised fact.

#### 4. Expected Sprint 33 false-positive classes

Based on the Day 1/Day 2 audit and the planned tooling shape, Sprint 33 should expect at least these false-positive classes:

1. **Under-covered bench/example surface**
   - code absent from the current `compile_commands.json` may look unused to `xunused` even when it is built by the Makefile and documented
2. **Live opt-in coverage**
   - env-gated test wrappers and their self-check code may look unusual to shallow reachability or scanner logic
3. **Public/documented API surface**
   - exported declarations or documented entry points may have no in-repo callers and still be real supported surface
4. **Comment-level historical wording**
   - active tests with names or comments mentioning “stub” or earlier sprint states are not automatically dead
5. **Scanner-only `unusedFunction` noise**
   - `cppcheck` over a limited path can report “unused” without seeing all real entry paths

#### 5. Deletion threshold for Sprint 33

Sprint 33 will treat code as deletable “definitely-unused internal code” only when **all** of the following are true:

1. The code is internal-only:
   - not in `include/`
   - not part of a documented example, benchmark, or CLI contract
   - not active or opt-in test registration
2. There is at least one tooling or compile-surface signal that the code is unused.
3. Manual review finds no live registration, no documented contract, and no intended current non-default role.
4. The code does not fall into a known coverage-gap or false-positive bucket.
5. After removal, the normal validation and any relevant dead-code target reruns still pass.

Corollary:

- Sprint 33 will **not** delete code solely because a single scanner flags it
- Sprint 33 will **not** delete code solely because local grep finds no call site

#### 6. Manual-review threshold for candidate public API

Any finding is automatically routed out of the first cleanup pass when it touches:

- installed headers under `include/`
- named/documented examples
- named/documented benchmark binaries
- README-described API or CLI behavior
- active or opt-in test framework surface

Those findings may still be reportable, but they are “candidate public API / documented surface” items, not first-pass removals.

### Day 3 Implementation Guidance For Later Sprint Days

Implications for Day 4 / Day 5 tooling work:

- `deadcode` should start as an evidence-gathering target, not a blunt fail-on-any-finding gate
- report output must preserve category boundaries instead of merging all findings into one list
- coverage gaps in `compile_commands.json` must be visible in the workflow because they affect what `xunused` can prove

Implications for Day 9 / Day 10 / Day 11 cleanup work:

- first-pass deletion candidates should come from internal-only code in tests, benchmarks, examples, and private helpers
- public/documented findings should remain in a separate review queue even if they look unused locally
- comment cleanup and naming cleanup should not be mixed into dead-code counts unless compiled unused code is actually removed

### Day 3 Interpretation

- Day 3 turns Sprint 33 from a vague “find dead code” effort into a conservative classification workflow.
- The key policy decision is that Sprint 33 is proving “safe to delete internal code,” not proving “no one could ever call this.”
- That keeps the first cleanup pass aligned with the Epic 3 goal: reduce real dead internal code without regressing the truthful active surface or casually deleting documented/public entry points.

### Day 3 Outputs

- `artifacts/day3-dead-code-policy-and-limitations.md`

## Day 4

**Objective:** Design the concrete Makefile and workflow shape for Sprint 33 dead-code tooling so Day 5 can implement it without inventing target behavior or artifact paths mid-edit.

### Commands Run

1. Re-read the Sprint 33 policy and implementation scope:
   - `git status --short --branch`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_33/artifacts/day3-dead-code-policy-and-limitations.md`
   - `sed -n '120,170p' docs/planning/EPIC_3/PROJECT_PLAN.md`
2. Inspect current Makefile quality targets and helper-script precedent:
   - `sed -n '380,470p' Makefile`
   - `find scripts -maxdepth 1 -type f | sort`
   - `sed -n '1,220p' scripts/epic3_warning_workflow.sh`
3. Reconfirm the current CMake coverage and build-surface mismatch:
   - `sed -n '1,320p' CMakeLists.txt`
   - `grep '"file":' build/sprint33-day1-cmake/compile_commands.json | awk '...'`
   - `ls build | sort`

### Day 4 Design Decisions

#### 1. Use Makefile entry points with a helper-script workflow pattern

Chosen shape:

- Makefile provides operator-facing targets
- a helper script should own the nontrivial command flow and prerequisite checks when implementation complexity grows past one or two shell lines

Reason:

- the repo already uses this pattern successfully in `warning-workflow`
- the dead-code flow needs:
  - prerequisite checks
  - deterministic artifact paths
  - multiple tool invocations
  - coverage-gap notes
  - later report normalization

Day 4 consequence:

- Day 5 should keep the Makefile target readable and thin
- if command logic starts expanding, move it into a dedicated `scripts/` helper rather than embedding brittle shell in the Makefile

#### 2. Separate the compilation-database build path from normal local build paths

Chosen path:

- dead-code workflow owns a dedicated CMake build directory:
  - `build/deadcode-cmake`

Reason:

- Sprint 33 needs a reliable `compile_commands.json` producer
- reusing ad hoc previous sprint build trees would make the target stateful and fragile
- a dedicated path lets the target:
  - configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
  - refresh predictably
  - avoid colliding with other sprint artifact trees

Day 4 consequence:

- Day 5 should not depend on `build/sprint33-day1-cmake`
- it should generate or refresh `build/deadcode-cmake/compile_commands.json` directly

#### 3. Keep raw evidence artifacts in `build/`, not `docs/`

Chosen local artifact root:

- `build/deadcode/`

Planned raw outputs:

- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`
- `build/deadcode/coverage-notes.txt`

Reason:

- these are operator-generated local workflow artifacts, not sprint-history records
- `docs/planning/` should hold curated sprint evidence, not routine generated outputs
- local reruns should be cheap and overwritable

Day 4 consequence:

- later sprint-day artifacts can copy or summarize representative findings into `docs/planning/EPIC_3/SPRINT_33/artifacts/`
- but the workflow itself should write to `build/`

#### 4. `deadcode` should be evidence-gathering only in Sprint 33

Chosen behavior for the first target:

- `make deadcode`
  - ensures `compile_commands.json` exists
  - runs raw analysis commands
  - writes raw outputs
  - exits non-zero only for infrastructure/tool invocation failure, not for ordinary findings

Reason:

- Day 3 policy explicitly rejects treating raw scanner findings as deletion proof
- the first implementation needs to gather evidence and make limitations visible before any enforcement semantics are introduced
- forcing failure on any finding too early would collapse false positives, coverage gaps, and true positives into one unusable gate

Day 4 consequence:

- Day 6 / Day 7 can design and implement a stricter `deadcode-check` layer later
- but Day 5 should ship `deadcode` as a reproducible evidence-gathering command

#### 5. Coverage-gap visibility is a first-class workflow requirement

Chosen reporting rule:

- the raw workflow should emit an explicit note that current CMake compile-db coverage is narrower than the Makefile bench/example surface

Known gap to surface:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

Reason:

- this gap affects what `xunused` can prove
- hiding it inside Sprint 33 notes would make local operator output misleading

Day 4 consequence:

- Day 5 should write the gap into a stable artifact such as `build/deadcode/coverage-notes.txt`
- Day 6 / Day 7 should preserve that note in the report layer

#### 6. Day 5 implementation contract for the raw commands

Chosen initial command shape:

1. CMake configure step for `compile_commands.json`
   - `cmake -S . -B build/deadcode-cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
2. raw `cppcheck`
   - `cppcheck --enable=all --quiet src/`
3. raw `xunused`
   - `xunused build/deadcode-cmake/compile_commands.json`

Important Day 4 qualification:

- these are the required raw analysis inputs from `PROJECT_PLAN.md`
- they are not yet the final report format
- additional normalization, categorization, and readable output belong in the later reporting target

#### 7. Day 5 should not wire `deadcode` into `lint` yet

Chosen boundary:

- keep `deadcode` separate from `lint` in Sprint 33

Reason:

- `lint` is already a stable quality path
- Sprint 33’s dead-code flow still has known coverage gaps and a missing local `xunused` prerequisite
- folding it into `lint` before the reporting and category model exist would create noisy, hard-to-interpret failures

Day 4 consequence:

- Day 5 target should be explicitly opt-in
- Day 7 or later can revisit whether a future enforcement target belongs in broader quality flows

### Proposed Target Topology

Day 4 recommended target structure:

- `deadcode-compile-db`
  - configure `build/deadcode-cmake` with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`
- `deadcode`
  - depends on `deadcode-compile-db`
  - runs raw `cppcheck` and `xunused`
  - writes raw artifacts to `build/deadcode/`
- `deadcode-report`
  - later target
  - consumes raw artifacts and emits a readable categorized summary
- `deadcode-check`
  - later target
  - wraps report semantics or narrower enforcement semantics once the false-positive model is understood

### Day 4 Interpretation

- The key design decision is not the command lines themselves; it is keeping raw analysis, readable reporting, and future enforcement as separate layers.
- That separation is what lets Sprint 33 preserve the Day 3 conservative policy while still shipping useful tooling quickly.
- Day 5 can now implement the raw workflow without needing to invent target semantics or artifact placement on the fly.

### Day 4 Outputs

- `artifacts/day4-tooling-integration-design.md`

## Day 5

**Objective:** Implement the first executable Sprint 33 dead-code workflow so maintainers can refresh a dedicated compilation database, run the agreed raw `cppcheck` and `xunused` passes from one entry point, preserve the known bench/example coverage gap in a stable artifact, and capture the initial raw evidence for later reporting/classification work.

### Commands Run

1. Re-read the Day 4 contract before editing:
   - `sed -n '110,145p' docs/planning/EPIC_3/SPRINT_33/PLAN.md`
   - `sed -n '560,680p' docs/planning/EPIC_3/SPRINT_33/WORKING_NOTES.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_33/artifacts/day4-tooling-integration-design.md`
2. Validate the newly-installed `xunused` behavior locally and derive the macOS invocation shape:
   - `command -v xunused`
   - `xunused --help`
   - `xunused build/sprint33-day1-cmake/compile_commands.json`
   - `SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"; xunused --extra-arg-before=-isysroot --extra-arg-before="$SDKROOT" build/sprint33-day1-cmake/compile_commands.json`
   - `LLVM_PREFIX="$(brew --prefix llvm@18)"; RESOURCE_DIR="$($LLVM_PREFIX/bin/clang -print-resource-dir)"; xunused --extra-arg-before=-isysroot --extra-arg-before="$SDKROOT" --extra-arg-before=-resource-dir="$RESOURCE_DIR" build/sprint33-day1-cmake/compile_commands.json`
3. Implement the Makefile entry points and helper script:
   - edited `Makefile`
   - added `scripts/deadcode_workflow.sh`
   - `bash -n scripts/deadcode_workflow.sh`
4. Validate the workflow end to end:
   - `git status --short --branch`
   - `make deadcode-compile-db`
   - `make deadcode`
   - `wc -l build/deadcode/cppcheck.txt build/deadcode/xunused.txt build/deadcode/coverage-notes.txt`
   - `cat build/deadcode/coverage-notes.txt`
   - `rg '^/.+: warning:' build/deadcode/xunused.txt`
   - `rg -o '\[[^]]+\]$' build/deadcode/cppcheck.txt | sort | uniq -c | sort -nr | sed -n '1,20p'`

### Implementation Notes

- Added `deadcode-compile-db` to refresh `build/deadcode-cmake/compile_commands.json` on every run rather than only when the file is missing.
- Added `deadcode`, which stays Makefile-thin and delegates the nontrivial workflow to `scripts/deadcode_workflow.sh`.
- The helper script now owns:
  - tool prerequisite checks
  - compile-database presence validation
  - stable raw artifact paths under `build/deadcode/`
  - compile-db coverage-gap notes
  - raw `cppcheck` capture
  - raw `xunused` capture
- Actual Day 5 command contract is slightly stricter than the Day 4 sketch:
  - `cppcheck` needs `-Iinclude -Ibuild/include -Isrc --std=c11 --suppress=missingIncludeSystem` to avoid turning the raw artifact into mostly header-resolution noise
  - on macOS, `xunused` needs `xcrun`-provided `-isysroot` plus an LLVM `-resource-dir` so it can parse system and builtin headers successfully

### Validation Results

- `make deadcode` now runs end to end from a clean working tree on `sprint-33`.
- The target refreshes its own dedicated CMake build tree:
  - `build/deadcode-cmake/compile_commands.json`
- The raw artifacts now land in the expected local workflow directory:
  - `build/deadcode/cppcheck.txt`
  - `build/deadcode/xunused.txt`
  - `build/deadcode/coverage-notes.txt`
- Final Day 5 artifact sizes after the successful rerun:
  - `cppcheck.txt`: `920` lines
  - `xunused.txt`: `107` lines
  - `coverage-notes.txt`: `17` lines
- The compile-db gap is now surfaced exactly as intended:
  - missing benchmark: `bench_svd`
  - missing examples:
    - `example_basic_solve`
    - `example_condition`
    - `example_iterative`
    - `example_least_squares`
    - `example_matrix_free`
    - `example_svd_lowrank`

### First Raw Findings Snapshot

- `xunused` is narrow and high-signal on the current compilation database:
  - `5` unused-function warnings total
  - reported names:
    - `chol_csc_dump_supernodes`
    - `givens_apply_right`
    - `sparse_print_dense`
    - `sparse_print_entries`
    - `sparse_print_info`
- Those `xunused` results already demonstrate why the Day 3 policy boundary matters:
  - one candidate is internal-only (`chol_csc_dump_supernodes`)
  - several others are declared through installed public headers, so they belong in a public-surface/manual-review bucket rather than an auto-delete bucket
- `cppcheck` is much broader and not yet a dead-code-only signal:
  - top recurring IDs in the raw output are:
    - `constVariablePointer`: `106`
    - `staticFunction`: `90`
    - `unusedFunction`: `80`
    - `normalCheckLevelMaxBranches`: `23`
  - Day 6 therefore needs a report layer that separates likely dead-code evidence from general style/static-analysis noise

### Day 5 Conclusion

- Sprint 33 now has a reproducible, opt-in `make deadcode` entry point that matches the Day 4 layered-tooling design.
- The workflow is useful as raw evidence gathering, not as an enforcement gate yet.
- The next job is not more Makefile plumbing; it is report design and classification so Day 6 / Day 7 can turn these raw artifacts into an auditable cleanup queue without violating the Day 3 conservative policy.

## Day 6

**Objective:** Turn the Day 5 raw scanner outputs into a conservative reporting model by defining stable categories, a readable report layout, and a `deadcode-check` invariant that enforces report completeness rather than pretending the tools prove reachability perfectly.

### Commands Run

1. Re-read the Day 6 / Day 7 scope and Day 5 implementation context:
   - `sed -n '140,190p' docs/planning/EPIC_3/SPRINT_33/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_33/artifacts/day5-deadcode-target-implementation.md`
   - `tail -n 180 docs/planning/EPIC_3/SPRINT_33/WORKING_NOTES.md`
2. Reinspect the current raw Day 5 artifacts:
   - `sed -n '1,240p' build/deadcode/coverage-notes.txt`
   - `sed -n '1,180p' build/deadcode/xunused.txt`
   - `rg -o '\[[^]]+\]$' build/deadcode/cppcheck.txt | sort | uniq -c | sort -nr | sed -n '1,40p'`
3. Derive category-driving counts from the raw outputs:
   - `python3 - <<'PY' ... parse build/deadcode/cppcheck.txt by checker id and file ... PY`
   - `python3 - <<'PY' ... parse build/deadcode/xunused.txt warning/note pairs ... PY`
   - `rg -n "chol_csc_dump_supernodes|givens_apply_right|sparse_print_dense|sparse_print_entries|sparse_print_info" src include`
4. Re-read the later-sprint sequencing so the Day 6 report/check design leaves room for Day 8 public-surface review and Day 10 / Day 11 cleanup batches:
   - `sed -n '190,310p' docs/planning/EPIC_3/SPRINT_33/PLAN.md`

### Raw-Finding Shape Confirmed On Day 6

- `coverage-notes.txt` confirms the known compile-db gap remains unchanged:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- `xunused` remains the narrow, high-confidence signal:
  - `5` warnings total
  - `1` internal/private-helper candidate:
    - `chol_csc_dump_supernodes`
  - `4` exported-header/public-surface review items:
    - `givens_apply_right`
    - `sparse_print_dense`
    - `sparse_print_entries`
    - `sparse_print_info`
- `cppcheck` remains broader and mixed-purpose:
  - top ids:
    - `constVariablePointer`: `106`
    - `staticFunction`: `90`
    - `unusedFunction`: `80`
    - `normalCheckLevelMaxBranches`: `23`
  - top files by total raw findings:
    - `src/sparse_matrix.c`: `31`
    - `src/sparse_qr.c`: `25`
    - `src/sparse_chol_csc.c`: `24`
    - `src/sparse_lu.c`: `22`
    - `src/sparse_svd.c`: `22`

### Chosen Classification Scheme

Day 6 category model for the later report:

1. `coverage-gap`
   - meaning:
     - code that is outside the current `compile_commands.json` surface, so scanner silence is not evidence
   - source:
     - `build/deadcode/coverage-notes.txt`
   - Day 6 handling:
     - always shown near the top of the report
     - never mixed into the cleanup queue
2. `definitely-unused-internal-candidate`
   - meaning:
     - scanner finding that currently points to internal-only implementation/declaration surface with no installed-header evidence
   - current Day 6 example:
     - `chol_csc_dump_supernodes`
   - Day 6 handling:
     - eligible for Day 9 batching after Day 8 confirms no public-surface conflict
3. `public-surface-review`
   - meaning:
     - finding touches `include/`, documented examples, documented bench entry points, or other outward-facing surface
   - current Day 6 examples:
     - `givens_apply_right`
     - `sparse_print_dense`
     - `sparse_print_entries`
     - `sparse_print_info`
   - Day 6 handling:
     - explicitly deferred to Day 8 review
     - never auto-promoted into cleanup batches
4. `secondary-candidate-signal`
   - meaning:
     - tool findings that may help prioritize inspection but are too noisy to treat as cleanup-ready on their own
   - current Day 6 source:
     - `cppcheck` `unusedFunction`
     - `cppcheck` `staticFunction`
   - Day 6 handling:
     - summarized by file and checker id
     - not emitted as a line-by-line deletion queue in the primary report
5. `non-deadcode-static-analysis-noise`
   - meaning:
     - findings that are real static-analysis observations but not dead-code evidence
   - current Day 6 source:
     - `constVariablePointer`
     - `variableScope`
     - `normalCheckLevelMaxBranches`
     - other style-only ids
   - Day 6 handling:
     - counted in an appendix/summary only
     - omitted from the cleanup queue entirely

### Chosen `deadcode-report` Design

Day 7 should generate two stable report artifacts under `build/deadcode/`:

1. human-readable summary
   - proposed path:
     - `build/deadcode/report.md`
   - purpose:
     - show coverage gaps
     - show categorized `xunused` findings
     - show aggregated `cppcheck` evidence
     - name the current cleanup-ready queue explicitly
2. machine-stable findings table
   - proposed path:
     - `build/deadcode/report.tsv`
   - purpose:
     - one normalized finding per line
     - stable sorting for later sprint comparisons
     - easier parsing if CI wiring is added later

Proposed summary section order:

1. run metadata
2. compile-db coverage gaps
3. definitely-unused internal candidates
4. public-surface review items
5. secondary `cppcheck` candidate signals by file/id
6. deferred noise summary
7. next-action queue for the current sprint

Proposed normalized TSV columns:

- `bucket`
- `tool`
- `symbol`
- `path`
- `line`
- `detail`
- `disposition`

Why this split:

- Markdown is better for maintainer review
- TSV is better for reproducibility and later diffability
- both can be generated from the same parser logic on Day 7

### Chosen `deadcode-check` Behavior

Day 6 decision:

- `deadcode-check` should **not** fail merely because findings exist in Sprint 33
- `deadcode-check` should enforce report completeness and category hygiene instead

Proposed Day 7 `deadcode-check` invariant:

1. run `deadcode-report`
2. fail if report generation fails
3. fail if any `xunused` warning is left uncategorized
4. fail if the coverage-gap section is missing
5. pass even when categorized candidates remain

Reason:

- Sprint 33 still expects real candidate findings before cleanup
- failing on any finding would block the planned Day 8 public-surface audit and Day 10 / Day 11 cleanup batches
- requiring complete categorization still gives the workflow a meaningful contract

Future-tightening note:

- a later sprint can choose to fail on non-empty `definitely-unused-internal-candidate`
  buckets after the first cleanup passes have landed
- Sprint 33 should not do that yet

### Day 6 Interpretation

- `xunused` should drive the primary actionable queue because it is the smallest high-confidence signal currently available.
- `cppcheck` should remain visible, but as secondary evidence summarized by file/checker rather than as a direct deletion list.
- `deadcode-check` should validate that the report tells the truth about the current evidence, not that the repository is already empty of all candidate findings.

### Day 6 Outputs

- `artifacts/day6-reporting-classification-design.md`

## Day 7

**Objective:** Implement the Day 6 reporting layer by wiring `deadcode-report` and `deadcode-check`, generating the first classified dead-code report from the raw Day 5 artifacts, and turning the scanner output into a concrete Sprint 33 action queue.

### Commands Run

1. Re-read the Day 7 scope and Day 6 design inputs:
   - `sed -n '140,220p' docs/planning/EPIC_3/SPRINT_33/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_33/artifacts/day6-reporting-classification-design.md`
   - `nl -ba Makefile | sed -n '455,520p'`
   - `nl -ba scripts/deadcode_workflow.sh | sed -n '1,220p'`
2. Implement the report layer:
   - edited `Makefile`
   - added `scripts/deadcode_report.py`
   - `python3 -m py_compile scripts/deadcode_report.py`
3. Generate and validate the first report:
   - `make deadcode-report`
   - `make deadcode-check`
4. Inspect the generated report artifacts:
   - `sed -n '1,260p' build/deadcode/report.md`
   - `wc -l build/deadcode/report.tsv`
   - `sed -n '1,40p' build/deadcode/report.tsv`
   - `python3 - <<'PY' ... summarize report.tsv bucket counts and queues ... PY`

### What Shipped

- Added `deadcode-report` to the Makefile.
  - behavior:
    - runs `deadcode`
    - then normalizes the raw artifacts into:
      - `build/deadcode/report.md`
      - `build/deadcode/report.tsv`
- Added `deadcode-check` to the Makefile.
  - behavior:
    - depends on `deadcode-report`
    - then runs the report script in `--check` mode
    - fails on report-generation/completeness problems rather than on the mere presence of candidates
- Added `scripts/deadcode_report.py`.
  - responsibilities:
    - parse `coverage-notes.txt`
    - parse `xunused.txt`
    - parse and aggregate `cppcheck.txt`
    - classify findings into the Day 6 buckets
    - emit Markdown and TSV reports
    - enforce the Day 6 `deadcode-check` invariants

### Validation Results

- `python3 -m py_compile scripts/deadcode_report.py`: passed
- `make deadcode-report`: passed
- `make deadcode-check`: passed

Day 7 report/check invariant now enforced:

1. `deadcode-report` must regenerate the report artifacts successfully
2. every `xunused` finding must land in a named category
3. the report must preserve the compile-db coverage-gap section

### First Classified Report

Generated report artifacts:

- `build/deadcode/report.md`
- `build/deadcode/report.tsv`

Current normalized bucket counts:

- `coverage-gap`: `7`
- `definitely-unused-internal-candidate`: `1`
- `public-surface-review`: `4`
- `secondary-candidate-signal`: `35`
- `non-deadcode-static-analysis-noise`: `6`

Current cleanup-relevant queue from the report:

- definitely-unused internal candidate:
  - `chol_csc_dump_supernodes`
- public-surface review items:
  - `givens_apply_right`
  - `sparse_print_dense`
  - `sparse_print_entries`
  - `sparse_print_info`

Top secondary `cppcheck` signal files in the human report:

- `src/sparse_chol_csc.c`: `20`
- `src/sparse_ldlt_csc.c`: `17`
- `src/sparse_matrix.c`: `16`
- `src/sparse_qr.c`: `14`
- `src/sparse_graph.c`: `14`

### Day 7 Interpretation

- The actionable Sprint 33 queue is now narrow and truthful.
- Day 8 has a clearly bounded public-surface audit set of four symbols.
- Day 9 starts with a single high-confidence internal cleanup candidate:
  - `chol_csc_dump_supernodes`
- The broader `cppcheck` rows remain supporting evidence only and are now intentionally separated from the cleanup-ready queue instead of being mixed into it.

### Day 7 Outputs

- `artifacts/day7-report-wiring-and-first-report.md`
