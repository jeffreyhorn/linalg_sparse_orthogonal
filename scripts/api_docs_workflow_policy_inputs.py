#!/usr/bin/env python3
"""Step-scoped inputs for the generated API documentation publication policy."""

from __future__ import annotations

import argparse
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .api_docs_workflow_reader import (
        ScalarValue, SourceLocation, WorkflowReadError, read_workflows,
    )
else:
    from api_docs_workflow_reader import (
        ScalarValue, SourceLocation, WorkflowReadError, read_workflows,
    )


PATH_INPUT_NAMES = frozenset(
    ("path", "publish_dir", "publish-dir", "directory", "folder", "files", "asset_path")
)
_DYNAMIC_PATH = re.compile(r"\$\{|\$\(|\$[A-Za-z_]|%[A-Za-z_][A-Za-z_0-9]*%|^~")


@dataclass(frozen=True)
class PathEntry:
    text: str
    canonical: str | None
    dynamic: bool
    glob: bool


@dataclass(frozen=True)
class PathInput:
    name: str
    raw: ScalarValue
    entries: tuple[PathEntry, ...]

    @property
    def canonical(self) -> str | None:
        return self.entries[0].canonical if len(self.entries) == 1 else None

    @property
    def dynamic(self) -> bool:
        return any(entry.dynamic for entry in self.entries)

    @property
    def glob(self) -> bool:
        return any(entry.glob for entry in self.entries)


@dataclass(frozen=True)
class StagedOutputRelation:
    """Explicit producer/consumer link; the policy must prove this before using it."""

    source: PathInput
    destination: PathInput
    producer_step_index: int
    consumer_step_index: int


@dataclass(frozen=True)
class PolicyInput:
    workflow_path: Path
    job_key: str
    step_index: int | None
    location: SourceLocation
    action_identity: ScalarValue | None
    action_dynamic: bool
    with_inputs: tuple[tuple[str, ScalarValue], ...]
    env_values: tuple[tuple[str, ScalarValue], ...]
    path_inputs: tuple[PathInput, ...]
    rendered_command: ScalarValue | None
    effective_shell: str
    shell_family: str
    staging_relations: tuple[StagedOutputRelation, ...]


def normalize_path_input(name: str, value: ScalarValue) -> PathInput:
    """Normalize only a path field, never shell command text or arguments."""
    entries: list[PathEntry] = []
    for line in value.value.splitlines():
        raw = line.strip()
        if not raw:
            continue
        dynamic = bool(_DYNAMIC_PATH.search(raw))
        canonical = None if dynamic else posixpath.normpath(raw.replace("\\", "/")).casefold()
        entries.append(PathEntry(raw, canonical, dynamic, any(char in raw for char in "*?[]")))
    return PathInput(name, value, tuple(entries))


def _path_inputs(items: tuple[tuple[str, ScalarValue], ...]) -> tuple[PathInput, ...]:
    return tuple(
        normalize_path_input(name, value)
        for name, value in items
        if name.casefold() in PATH_INPUT_NAMES
    )


def read_policy_inputs(root: Path) -> tuple[PolicyInput, ...]:
    """Validate all workflows, then expose job/step-local policy inputs."""
    records = read_workflows(root)
    inputs: list[PolicyInput] = []
    for workflow in records:
        for job in workflow.jobs:
            if job.uses is not None:
                inputs.append(
                    PolicyInput(
                        workflow.path, job.key, None, job.location, job.uses,
                        bool(_DYNAMIC_PATH.search(job.uses.value)),
                        job.with_inputs, workflow.env + job.env,
                        _path_inputs(job.with_inputs), None,
                        "unknown", "unknown", (),
                    )
                )
            for step in job.steps:
                action = step.action_identity
                inputs.append(
                    PolicyInput(
                        step.workflow_path, step.job_key, step.step_index, step.location,
                        action, action is not None and bool(_DYNAMIC_PATH.search(action.value)),
                        step.with_inputs, workflow.env + job.env + step.env,
                        _path_inputs(step.with_inputs), step.run_value,
                        step.effective_shell, step.shell_family, (),
                    )
                )
    return tuple(inputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        inputs = read_policy_inputs(args.root)
    except WorkflowReadError as exc:
        parser.exit(1, f"api-docs-workflow-policy-inputs: {exc}\n")
    print(f"api-docs-workflow-policy-inputs: validated {len(inputs)} job/step records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
