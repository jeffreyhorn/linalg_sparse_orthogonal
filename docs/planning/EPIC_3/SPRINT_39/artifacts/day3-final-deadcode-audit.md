# Sprint 39 Day 3 Artifact: Final Dead-Code Audit

## Purpose

Map the post-Sprint-38 residual dead-code state into final Epic 3
closeout-ready categories: resolved, justified keep, supporting-only, appendix
noise, and workflow-topology limitation.

## Day 3 Bottom Line

Sprint 39 no longer inherits a dead-code discovery queue. It inherits a final
disposition queue.

Already closed before Day 3:

- `coverage-gap = 0`
- `definitely-unused-internal-candidate = 0`

Current residual buckets:

- `public-surface-review = 4`
- `secondary-candidate-signal = 35`
- `non-deadcode-static-analysis-noise = 6`

## Current Residual-Bucket Classification

### 1. Public-surface reviewed keeps

Current rows:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

Current status:

- exported through installed headers
- already audited as `keep`
- useful as contextual evidence, not as active cleanup targets

Sprint 39 interpretation:

- this is effectively a justified-keep list
- the residual work here is wording/closeout clarity, not code removal

### 2. Secondary `cppcheck` candidate signals

Current status:

- supporting evidence only
- not direct removal instructions
- not pass/fail criteria

Highest-density files in the current summary:

- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_matrix.c`
- `src/sparse_qr.c`
- `src/sparse_graph.c`

Sprint 39 interpretation:

- Day 3 found no evidence that these rows should be promoted into a new
  cleanup-ready deletion queue
- they should remain summarized, justified as supporting-only, and preserved
  for later focused analysis if future work chooses to revisit them

### 3. Non-deadcode static-analysis noise

Current summary:

- `constVariablePointer = 106`
- `normalCheckLevelMaxBranches = 23`
- `variableScope = 4`
- `constParameterPointer = 1`
- `constVariable = 1`
- `unreadVariable = 1`

Current status:

- appendix-only
- not cleanup candidates

Sprint 39 interpretation:

- this is documentation/appendix residue, not a live engineering-removal queue

## Workflow-Topology Limitation

Separate from all content-level findings:

- dead-code execution remains authoritative only when serialized
- shared paths still exist:
  - `build/deadcode-cmake`
  - `build/deadcode/`

Sprint 39 interpretation:

- the workflow-topology limit remains real
- it must stay explicit in final Epic 3 closeout
- it should not be collapsed into the content-level bucket discussion

## Day 6 Likely Implementation Shape

Unless a stronger rerun surfaces a new real candidate, the expected dead-code
closeout batch is narrow:

1. preserve the audited-keep meaning of the public bucket
2. preserve the supporting-only meaning of the `cppcheck` secondary bucket
3. preserve the appendix-only meaning of the static-analysis-noise bucket
4. keep the serialized-execution limit explicit as workflow topology, not as a
   hidden caveat

## Immediate Guidance For Later Sprint 39 Work

- Do not reopen compile-db breadth work; it is already closed for the named
  benchmark/example list.
- Do not manufacture a new removal batch from summarized `cppcheck` density
  alone.
- Treat the remaining dead-code work as final justification and closeout
  language unless a stronger rerun proves otherwise.
