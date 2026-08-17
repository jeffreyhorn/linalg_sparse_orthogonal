# Day 4 Warning Triage Policy

## Scope

Day 4 normalizes the Day 2 Doxygen warning log into stable categories and
assigns each warning an explicit disposition before any generated API HTML
publication decision.

This artifact does not edit public headers or regenerate documentation. It
defines the warning closure policy and quality gates for the later warning-fix
batch.

## Triage Inputs

| Input | Day 4 use |
| --- | --- |
| `docs/planning/EPIC_14/SPRINT_158/artifacts/day2-doxygen-baseline.md` | Source warning log and generated-output inventory. |
| `docs/planning/EPIC_14/SPRINT_158/artifacts/day3-public-header-coverage-map.md` | Confirms page coverage succeeds for the 18 checked-in public headers while warnings remain unresolved. |
| `include/sparse_lu_csr.h` | Owner of five unknown-command warnings. |
| `include/sparse_types.h` | Owner of four undocumented typedef/macro warnings. |
| `include/sparse_iterative.h` | Owner of one undocumented struct-member warning. |
| Sprint 157 Day 10 quality map | Requires full C/header gate for public-header edits, including comment-only edits. |

## Normalized Warning Categories

| Category ID | Warning family | Count | Locations | Cause | Disposition |
| --- | --- | ---: | --- | --- | --- |
| W158-01 | Unknown Doxygen command `\U` | 5 | `include/sparse_lu_csr.h`: 105, 152, 268, 286, 288 | Public-header prose uses `L\U`; Doxygen interprets `\U` as an unknown command. | Selected for Sprint 158 closure. |
| W158-02 | Undocumented public index typedef and macros | 4 | `include/sparse_types.h`: 44, 45, 46, 47 | `idx_t`, `IDX_MAX`, `SPARSE_PRIDX`, and `SPARSE_SCNIDX` lack Doxygen-member comments at the conditional definition sites. | Selected for Sprint 158 closure. |
| W158-03 | Undocumented GMRES progress context member | 1 | `include/sparse_iterative.h`: 128 | `sparse_gmres_opts_t::progress_user` has no member comment even though adjacent progress callback semantics are documented. | Selected for Sprint 158 closure. |

## Disposition Table

| Category ID | Fix/defer/exclude/blocker | Owner | Required action | Publication effect |
| --- | --- | --- | --- | --- |
| W158-01 | Fix selected | Documentation/API owner with LU CSR header owner | Replace or escape `L\U` prose so Doxygen no longer parses `\U` as a command. | Blocks generated API HTML publication until fixed and `make docs` reruns cleanly or the warning is reclassified. |
| W158-02 | Fix selected | Documentation/API owner with core type header owner | Add Doxygen comments for `idx_t` and public index-limit/format macros without changing preprocessor behavior. | Blocks generated API HTML publication until fixed and `make docs` reruns cleanly or the warning is reclassified. |
| W158-03 | Fix selected | Documentation/API owner with iterative solver header owner | Add member documentation for `progress_user` aligned with `progress_cb` semantics. | Blocks generated API HTML publication until fixed and `make docs` reruns cleanly or the warning is reclassified. |

No Day 4 warning is accepted as an exclusion. No Day 4 warning is deferred out
of Sprint 158. The current policy is that the generated API HTML should not be
published or described as fresh while these warnings remain unclosed.

## Header Edit Classification

| Category ID | Expected edit type | Declaration change selected? | Code behavior change selected? | Required validation if edited |
| --- | --- | --- | --- | --- |
| W158-01 | Public-header documentation/prose cleanup | No | No | `make format && make lint && make test`; `make docs`; docs hygiene. |
| W158-02 | Public-header Doxygen comments on typedef/macro definitions | No | No | `make format && make lint && make test`; `make docs`; docs hygiene. |
| W158-03 | Public-header Doxygen member comment | No | No | `make format && make lint && make test`; `make docs`; docs hygiene. |

If any later fix changes declarations, macros, typedefs, struct layout,
function signatures, or implementation behavior, that change exceeds this Day
4 policy and must be recorded as an explicit API/code change before proceeding.

## Generated And Unsupported-Claim Check

| Risk | Day 4 policy |
| --- | --- |
| Generated docs are cited despite warnings | Block publication/freshness claims until warning closure is validated. |
| Warning fixes introduce broad solver claims | Keep fixes local to documented syntax/member semantics; do not add broad LU, iterative, package, ABI, platform, parity, performance, or state-of-the-art claims. |
| Warning fixes change public declarations accidentally | Require full C/header gate and declaration review if anything beyond comments changes. |
| Warning fixes affect generated installed version-header policy | No Day 4 warning touches `sparse_version.h`; generated version-header treatment remains the Day 3 separate policy row. |

## Day 5 Handoff

Day 5 publication-option analysis should assume:

1. committed generated HTML is not viable as fresh evidence until W158-01
   through W158-03 are fixed or explicitly reclassified;
2. CI-published generated HTML still needs the same warning-closure policy;
3. local-only generated HTML can remain available as local context, but public
   docs must not describe it as fresh, complete, or warning-free until warning
   closure is validated;
4. any warning-fix implementation that edits public headers requires the full
   C/header quality gate.

## Completion Check

- All 10 Day 2 warnings have an explicit category and disposition.
- No warning is silently excluded or deferred.
- Selected fixes are bounded to public-header documentation cleanup unless a
  later artifact explicitly changes scope.
- Generated API HTML publication remains blocked until warnings are closed or
  deliberately reclassified with owner and support-tier impact.
