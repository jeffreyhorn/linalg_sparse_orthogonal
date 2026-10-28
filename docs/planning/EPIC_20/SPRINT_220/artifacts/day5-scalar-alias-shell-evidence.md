# Sprint 220 Day 5: Scalars, Aliases, And Shell Identity

**Scope:** Complete the reader's rendered-command and shell-identity record.
The production local-only guard still uses the legacy policy path.

## Extraction Decisions

The reader continues to use PyYAML's decoded scalar value as the command
passed to policy. It also retains the original YAML source span and style.
Literal, folded, single-quoted, and double-quoted `run` forms are tested for
line folding, YAML escapes, comments, and chomping. Shell comments are **not**
stripped by the reader: the policy layer must interpret the decoded command
with the effective shell. In particular, `run: >` folds a following
`docs/api/html` line onto a preceding `#` comment, while `run: |` leaves it
on a new line. Stripping comments before scalar rendering would invert that
distinction.

Aliases are resolved through the structured node graph. A reader-specific
`SafeLoader` gives an alias use its own source span and location while
retaining the anchor definition line/column in `SourceLocation`. Scalar
`source` is the alias token (for example, `*stage`) and `value` is the
resolved command or path. Alias mappings used by `with` and jobs have their
use and definition locations; their children retain their definition marks.
Flow mappings and flow-sequence steps are extracted as bounded step records.
Merge keys remain unsupported and fail closed; cyclic aliases and duplicate
flow keys also fail closed. No override precedence is guessed.

Each step now carries `effective_shell`, `shell_source` (`step`, `job`,
`workflow`, `runner`, or `unknown`), and `shell_family` (`powershell`, `posix`,
`cmd`, or `unknown`). Explicit step shell wins over job and workflow defaults;
known runner defaults apply only if no shell is set. Dynamic or unrecognized
shells stay `unknown`. The reader does not normalize command words: Windows
`\` separators and PowerShell backtick continuations remain in the decoded
`run` value even when a comparable Bash step is parsed. Policy will use the
shell family to interpret them in later sprint days.

## Regression Evidence

`tests/test_api_docs_workflow_reader.py` has 20 passing tests after Day 5.
The new cases cover:

- Folded comments versus literal newlines, indentation-preserved folded
  lines, and keep/strip chomping.
- Multiline single/double quotes, YAML hex escapes, and trailing YAML comments.
- Scalar aliases for `run` and publishing `with.path`, mapping aliases in
  flow steps, whole publishing-step aliases, and job alias provenance.
- Step, job, workflow, and runner shell precedence; PowerShell, POSIX, `cmd`,
  unknown, and custom shell templates.
- Backtick continuations and literal/double-quoted Windows paths from the
  frozen PowerShell corpus, without lossy backslash normalization.
- Explicit merge-key rejection and duplicate-key rejection inside flow maps.

All three maintained workflows still parse, including the nine artifact
upload steps. This establishes extraction behavior only; the new reader does
not yet replace the existing shell guard or make publication decisions.
No C/H files changed, so the C format/lint/test sequence is not required for
this Python-only reader slice.
