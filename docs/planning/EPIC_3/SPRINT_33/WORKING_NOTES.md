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
