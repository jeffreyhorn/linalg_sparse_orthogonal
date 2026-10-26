# Sprint 220 Day 3: Edge Oracle And Reader Contract

**Scope:** Close item 220.1's fixture freeze and select the boundary for item
220.2. This is a design and fixture artifact; the production guard still uses
the old normalizer. The local-only policy is unchanged.

## Frozen Edge Cases

[day3-edge-fixture-matrix.tsv](day3-edge-fixture-matrix.tsv) is the machine-
readable oracle for the ten complete workflows under
`tests/fixtures/api_docs_yaml_edges/`. To exercise a case, copy its text to
`.github/workflows/api-docs.yml` in the temporary Git worktree created by
`tests/test_api_docs_local_only_guard.py:write_fixture()` and invoke
`run_guard()`. `REJECT` means a nonzero guard result; `failure_stage` also
specifies the required *kind* of rejection. `ALLOW` means exit zero. The
stage is important: a parse error must not pass simply because a separate
path substring happened to reject the same file.

| Family | Required behavior | Legacy observation |
| --- | --- | --- |
| Duplicate `with.path` and duplicate step `run` | Reject the duplicate key before constructing a step. | Both rejected only through generated-path text detection. |
| Malformed flow and unknown alias | Reject with a YAML parse error. | Malformed flow rejected through path text; unknown alias allowed. |
| Merge key and custom tag | Reject explicitly as unsupported syntax. | Merge case rejected through path text; custom tag allowed. |
| Dynamic upload path and dynamic action identity | Reject unresolved publication fields. | Upload path rejected; dynamic action identity allowed. |
| Static anchored and static unanchored `build/output` uploads | Allow. | Both allowed. |

These ten cases extend the Day 2 matrix of 190 registered functions; they do
not replace it. Existing anchor/alias, flow, quoted, multiline, shell, and
PowerShell cases remain frozen at their Day 2 verdicts. The three legacy
`ALLOW` outcomes above are **known gaps**, not accepted behavior. The other
legacy `REJECT` outcomes are not proof of correct parsing or diagnostic stage.

## Parser Decision

Use pinned PyYAML 6.0.3 for the structured workflow reader, with
`yaml.compose_all(..., Loader=yaml.SafeLoader)` and explicit node validation.
Do **not** use `safe_load` as the sole boundary: construction turns a mapping
into a dictionary and can discard duplicate-key evidence. Composition keeps
ordered key/value node pairs, scalar style, and source marks, and handles
the anchored, aliased, flow, block, and quoted forms already in the corpus.
Reject more than one YAML document. Inspect every mapping **key node and
value node** before turning it into a dictionary, including mappings outside
publishing steps, because an ambiguous workflow can hide a later publishing
field. In particular, PyYAML tags a `<<` merge on the key node; checking
values alone would miss it.

The alternative is another hand-written fail-closed YAML subset. It avoids a
dependency, but the maintained 190-function corpus already exercises enough
YAML forms that a narrow grammar would either reject valid existing workflow
fixtures or recreate a general parser in the guard. A maintained parser is
the lower-risk ownership boundary. At the Day 3 freeze, PyYAML was **not
installed or declared** in this repository's default local environment; Day
4 must add the dependency pin and explicit Linux/macOS CI provisioning before
changing the production guard. Missing dependency must be a clear nonzero
diagnostic,
not a silent fallback to the legacy scanner or a network install at runtime.
The pin and provisioning route should be recorded in the maintainer guide
on Day 12. Windows CI is not currently an owner of the Make docs gate; a
future Windows invocation must provision the same parser first.

**Day 4 update:** The pin and Linux/macOS reader-test provisioning were added;
the production guard has not switched to the new reader.

PyYAML's YAML 1.1 implicit typing can tag plain `on` as boolean. For
workflow *keys*, use the scalar's original lexeme (`node.value`) rather than
constructing it as a Python boolean; accept quoted and plain `on` as the
same workflow key. For values, preserve the YAML-decoded scalar text and
style, then validate the field's expected type. Do not run arbitrary YAML
constructors. Direct aliases to supported scalar/mapping/sequence nodes are
allowed; cyclic aliases, non-scalar mapping keys, unknown aliases, merge
keys (`<<`), custom tags, and unsupported node shapes fail closed with a
location-bearing diagnostic. This deliberately does **not** claim support
for merge-key expansion.

The parser choice is grounded in [PyYAML's node and SafeLoader
documentation](https://pyyaml.org/wiki/PyYAMLDocumentation) and GitHub's
[documented workflow anchors and aliases](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations).
This is a Sprint 220 implementation choice, not a claim that all YAML
features or GitHub workflow extensions are accepted.

## Structured Extraction Contract

The reader returns one `WorkflowRecord` per `.github/workflows/*.yml` or
`*.yaml`, with source path, document location, workflow defaults, and an
ordered collection of `JobRecord`s. Each job has a unique job key, source
mark, runner specification, job defaults, and ordered steps. For each step,
emit a record with at least:

| Field | Required representation |
| --- | --- |
| `workflow_path`, `job_key`, `step_index` | Stable provenance and ordering; never splice steps from different jobs. |
| `source_location` | File and 1-based line/column for the step and each publication-relevant field; retain anchor/alias origin when relevant. |
| `runs_on`, `effective_shell` | Runner expression and shell identity after precedence: step `shell`, then job `defaults.run.shell`, then workflow `defaults.run.shell`, then known runner default. Unknown/dynamic shell is explicit, not guessed. |
| `action_identity` | Decoded `uses` scalar, plus source mark; unresolved expression remains marked dynamic. |
| `with_inputs` | Exact input names and decoded scalar values with source marks, especially path-like publishing inputs; do not flatten a mapping into unrelated text. |
| `run_value`, `run_source` | YAML-decoded command scalar for policy and original source span/style for diagnostics; keep block folding and line breaks as YAML defines them. |

Preserve step-local `env` and relevant job/workflow `env` as structured
inputs for later policy decisions, but do not eagerly substitute expressions
or shell variables. A step with both `uses` and `run`, a non-mapping step, a
non-sequence `steps`, or a non-mapping `jobs` value is malformed for this
reader. `uses`/`run` aliases must resolve to the appropriate scalar; a job
alias resolves to a complete job record without merging it with its caller.

The reader emits records only after a whole workflow passes syntax and
structural validation. The policy engine then classifies actions, decoded
`run` commands, paths, and explicit staging relationships by job/step. It
must not concatenate a whole workflow, unrelated steps, or commands separated
by YAML scalar boundaries into one fictitious command. A dynamic `uses`,
publishing `with` path, or shell/path needed to establish safe staging is
rejected by policy, including when no literal `docs/api` text appears.

## Error Boundary And Follow-Up

- **Parse error:** malformed YAML, unknown alias, duplicate key, multiple
  documents, or cyclic alias. Include workflow path and source location.
- **Unsupported syntax:** merge key, custom tag, non-scalar key, or node
  shape that the reader cannot safely interpret. Include the unsupported
  field and location; never drop it and continue.
- **Policy rejection:** successfully parsed, structurally valid workflow
  whose publication or staging fields are forbidden or unresolved.

Day 4 should wire this corpus into executable tests that assert both verdict
and failure stage. Day 5 should run every real workflow and the frozen Day 2
corpus through the reader before the legacy parser is removed. Day 9 owns
the full old/new differential, including the three intentional legacy gaps.
