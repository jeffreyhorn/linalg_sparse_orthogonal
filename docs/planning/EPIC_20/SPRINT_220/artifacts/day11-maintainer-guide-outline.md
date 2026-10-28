# Sprint 220 Day 11: Structured Guard Maintainer Outline

Day 12 should integrate this outline into the existing
`docs/maintainer_guide.md` section **API reference and generated Doxygen
HTML**, not create a competing policy page. Day 13 should complete the
escalation wording after the legacy workflow scans are retired.

## 1. Product Boundary And Dependency

- Keep `docs/api/html/` ignored, untracked, unstaged, and local-only.
  Passing a guard does not authorize a hosted page, retained artifact,
  release asset, or committed generated HTML.
- The structured reader uses the exact `PyYAML==6.0.3` pin in
  `requirements-api-docs.txt`. Describe a local virtual environment and
  the Linux/macOS CI provisioning step; missing PyYAML must fail closed.
- Keep Doxygen as the `make docs-check` prerequisite for the integrated
  `make api-docs-local-only` route. Focused reader/policy tests do not
  replace that gate.

## 2. Supported Workflow Forms

- Reader scope: all `.yml` and `.yaml` files in `.github/workflows/`, with
  workflow/job/step records, source line and column, per-step `uses`, `with`,
  `run`, and effective shell. Explain step shell, job default, workflow
  default, then runner fallback precedence.
- Supported YAML extraction includes block and flow collections, static
  anchors and aliases, quoted, literal, and folded scalars, and YAML
  comments. Parsed scalar values and source marks, not whole-file text
  concatenation, feed the policy.
- Structured policy checks generated API references, broad docs/root
  publication paths, explicit action inputs, supported POSIX and PowerShell
  staging/publisher commands, and same-job producer-to-upload paths.
  Explain that unrelated uploads and separate jobs do not inherit taint.

## 3. Unsupported And Ambiguous Forms

- Duplicate keys, malformed YAML, unknown aliases, cycles, merge keys,
  custom tags, missing step `run`/`uses`, and unsupported record shapes are
  reader errors, not accepted approximations.
- Dynamic action identities and publication paths fail closed; an
  unsupported action form or typed publication path must be made explicit
  before approval. Do not suggest suppressing the guard or weakening the
  fixture oracle to permit it.
- Shell staging recognition is currently demonstrated for POSIX and
  PowerShell. `cmd` and unknown-shell staging are not a broad coverage
  claim. The six maintained `cmd` run steps were inspected on Day 10 and
  contain no shell publication; their upload actions are parsed separately.
  A new command family or shell needs a targeted fixture and review.

## 4. Reproduction And Triage Commands

The Day 12 guide should give an executable sequence, using a venv Python
path appropriate to the maintainer's host:

```bash
python3 -m venv /tmp/api-docs-reader-venv
/tmp/api-docs-reader-venv/bin/python -m pip install -r requirements-api-docs.txt
make api-docs-local-only API_DOCS_PYTHON=/tmp/api-docs-reader-venv/bin/python
/tmp/api-docs-reader-venv/bin/python tests/test_api_docs_workflow_reader.py
/tmp/api-docs-reader-venv/bin/python tests/test_api_docs_workflow_policy.py
/tmp/api-docs-reader-venv/bin/python tests/test_api_docs_workflow_mutations.py
```

Distinguish a dependency/setup error from a source-located YAML parse error
and a job/step-scoped publication rejection. Direct reruns can use
`scripts/api_docs_workflow_reader.py` and
`scripts/api_docs_workflow_policy.py`; inspect `git status --short --ignored
docs/api` for generated-output state. Explain that diagnostic text may name
the staged producer step rather than the later upload step.

## 5. Repair And Escalation

- For a rejected workflow, inspect the indicated file/line and the
  corresponding job's action inputs or `run` commands. Repair path ownership
  or make the action identity and source path static; add a benign or
  rejecting mutation regression before changing parser/policy code.
- Preserve source-controlled API routes through `docs/api_reference.md` and
  checked-in public headers. Regenerate local HTML with Doxygen; do not
  commit the ignored output as a workaround.
- If publication is genuinely required, reopen the explicit product
  decision and define hosting/retention, freshness, routing, rollback,
  security, and claim-boundary evidence. A guard bypass is not the
  escalation path.
