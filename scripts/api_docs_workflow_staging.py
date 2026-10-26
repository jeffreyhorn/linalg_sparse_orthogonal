"""Recognize shell staging commands without merging unrelated workflow steps."""

from __future__ import annotations

import fnmatch
import posixpath
import re
from dataclasses import dataclass, replace

if __package__:
    from .api_docs_workflow_policy_inputs import PolicyInput
    from .api_docs_workflow_yaml_common import strip_shell_comment_stateful
else:
    from api_docs_workflow_policy_inputs import PolicyInput
    from api_docs_workflow_yaml_common import strip_shell_comment_stateful


_DYNAMIC = re.compile(r"\$\{|\$\(|\$[A-Za-z_]|%[A-Za-z_][A-Za-z_0-9]*%|^~")
_SEPARATORS = frozenset(";|&()\n")
_COPY = frozenset(("cp", "mv", "rsync", "copy", "move", "xcopy", "robocopy"))
_ARCHIVE = frozenset(("tar", "zip", "7z", "7za", "7zr"))
_PS_COPY = frozenset(("copy-item", "move-item"))


@dataclass(frozen=True)
class StagedCommand:
    source: str
    destination: str
    kind: str
    command_index: int = 0


def _without_comments(value: str, powershell: bool) -> str:
    state = (False, False, False)
    lines = []
    for line in value.splitlines():
        active, state = strip_shell_comment_stateful(line, state, powershell=powershell)
        lines.append(active)
    return "\n".join(lines)


def _posix_commands(value: str) -> tuple[tuple[str, ...], ...]:
    active = _without_comments(value, False).replace("\\\n", "")
    commands: list[tuple[str, ...]] = []
    current: list[str] = []
    word: list[str] = []
    started = False
    quote = ""
    index = 0

    def end_word() -> None:
        nonlocal started
        if started:
            current.append("".join(word))
            word.clear()
            started = False

    def end_command() -> None:
        end_word()
        if current:
            commands.append(tuple(current))
            current.clear()

    while index < len(active):
        char = active[index]
        if quote == "'":
            if char == "'":
                quote = ""
            else:
                word.append(char)
        elif quote == '"':
            if char == '"':
                quote = ""
            elif char == "\\" and index + 1 < len(active) and active[index + 1] in '$`"\\\n':
                index += 1
                if active[index] != "\n":
                    word.append(active[index])
            else:
                word.append(char)
        elif char in "'\"":
            quote = char
            started = True
        elif char == "\\":
            if index + 1 >= len(active):
                return ()
            index += 1
            if active[index] != "\n":
                word.append(active[index])
                started = True
        elif char in _SEPARATORS:
            end_command()
        elif char.isspace():
            end_word()
        else:
            word.append(char)
            started = True
        index += 1
    if quote:
        return ()
    end_command()
    return tuple(commands)


def _powershell_commands(value: str) -> tuple[tuple[str, ...], ...]:
    active = _without_comments(value, True)
    commands: list[tuple[str, ...]] = []
    current: list[str] = []
    word: list[str] = []
    quote = ""
    index = 0

    def end_word() -> None:
        if word:
            current.append("".join(word))
            word.clear()

    def end_command() -> None:
        end_word()
        if current:
            commands.append(tuple(current))
            current.clear()

    while index < len(active):
        char = active[index]
        if char == "`" and quote != "'" and index + 1 < len(active):
            next_char = active[index + 1]
            if next_char == "\n":
                index += 2
                continue
            word.append(next_char)
            index += 2
            continue
        if char in "\"'" and (not quote or quote == char):
            quote = "" if quote else char
        elif not quote and char == ",":
            end_word()
            current.append(",")
        elif not quote and char in ";|\n":
            end_command()
        elif not quote and char.isspace():
            end_word()
        else:
            word.append(char)
        index += 1
    if quote:
        return ()
    end_command()
    return tuple(commands)


def _path(value: str) -> str:
    return posixpath.normpath(value.replace("\\", "/")).casefold()


def _resolve_path(item: PolicyInput, value: str) -> str:
    directory = item.working_directory
    if directory is None:
        return value
    path = value.replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
        return path
    return posixpath.normpath(posixpath.join(directory.value.replace("\\", "/"), path))


def _source_is_docs(value: str) -> bool:
    path = _path(value)
    return "docs" in path.split("/") or path in (".", "*")


def _posix_stage(command: tuple[str, ...]) -> tuple[StagedCommand, ...]:
    if len(command) < 3:
        return ()
    executable = posixpath.basename(command[0]).casefold()
    if executable in _COPY:
        args = [arg for arg in command[1:] if not arg.startswith("-")]
        if len(args) >= 2:
            return tuple(StagedCommand(source, args[-1], "stages") for source in args[:-1])
    if executable == "tar":
        args = command[1:]
        for index, arg in enumerate(args):
            if arg in ("-f", "--file") and index + 2 < len(args):
                return tuple(StagedCommand(source, args[index + 1], "archives") for source in args[index + 2:])
            if arg.startswith("--file="):
                return tuple(StagedCommand(source, arg.split("=", 1)[1], "archives") for source in args[index + 1:])
            if re.fullmatch(r"-?[A-Za-z]*f[A-Za-z]*", arg) and index + 2 < len(args):
                return tuple(StagedCommand(source, args[index + 1], "archives") for source in args[index + 2:])
    if executable in _ARCHIVE - {"tar"}:
        args = [arg for arg in command[1:] if not arg.startswith("-")]
        if executable.startswith("7z") and args and args[0].casefold() == "a":
            args = args[1:]
        if len(args) >= 2:
            return tuple(StagedCommand(source, args[0], "archives") for source in args[1:])
    return ()


def _powershell_stage(command: tuple[str, ...]) -> tuple[StagedCommand, ...]:
    if len(command) < 3:
        return ()
    executable = command[0].casefold()
    if executable not in _PS_COPY | {"compress-archive"}:
        return ()
    named: dict[str, tuple[str, ...]] = {}
    positional: list[str] = []
    index = 1
    while index < len(command):
        arg = command[index]
        if arg.startswith("-"):
            name = arg.casefold()
            if name in ("-path", "-literalpath", "-destination", "-destinationpath"):
                if index + 1 < len(command):
                    values = [command[index + 1]]
                    index += 2
                    while index + 1 < len(command) and command[index] == ",":
                        values.append(command[index + 1])
                        index += 2
                    named[name] = tuple(values)
                    continue
        elif arg != ",":
            positional.append(arg)
        index += 1
    sources = named.get("-path", named.get("-literalpath"))
    destinations = named.get("-destination", named.get("-destinationpath"))
    if destinations is None and len(positional) >= 2:
        destinations = (positional[-1],)
    if sources is None:
        sources = tuple(positional[:-1])
    if not sources or not destinations:
        return ()
    destination = destinations[0] if len(destinations) == 1 else "${unsupported_destination_array}"
    return tuple(
        StagedCommand(source, destination, "archives" if executable == "compress-archive" else "stages")
        for source in sources
    )


def _commands(item: PolicyInput) -> tuple[tuple[str, ...], ...]:
    run = item.rendered_command
    if run is None:
        return ()
    if item.shell_family == "powershell":
        return _powershell_commands(run.value)
    if item.shell_family == "posix":
        return _posix_commands(run.value)
    return ()


def staged_commands(item: PolicyInput) -> tuple[StagedCommand, ...]:
    parser = _powershell_stage if item.shell_family == "powershell" else _posix_stage
    return tuple(
        replace(
            stage, source=_resolve_path(item, stage.source),
            destination=_resolve_path(item, stage.destination), command_index=index,
        )
        for index, command in enumerate(_commands(item))
        for stage in parser(command)
    )


def _operands(words: tuple[str, ...], value_options: frozenset[str] = frozenset()) -> list[str]:
    result: list[str] = []
    index = 0
    after_options = False
    while index < len(words):
        word = words[index]
        if word == "--":
            after_options = True
        elif not after_options and word.startswith("-"):
            if word.casefold() in value_options:
                index += 1
        else:
            result.append(word)
        index += 1
    return result


def published_paths(item: PolicyInput) -> tuple[tuple[int, str], ...]:
    """Return explicit local inputs to shell publication commands."""
    paths: list[tuple[int, str]] = []
    for index, command in enumerate(_commands(item)):
        words = tuple(word.casefold() for word in command)
        sources: list[str] = []
        if len(words) >= 5 and words[:3] == ("aws", "s3", "cp"):
            sources = [command[3]]
        elif len(words) >= 5 and words[:3] == ("aws", "s3", "sync"):
            sources = [command[3]]
        elif len(words) >= 4 and words[:2] in (("rclone", "copy"), ("rclone", "sync"), ("rclone", "move")):
            sources = [command[2]]
        elif len(words) >= 3 and words[0] == "gsutil":
            operands = _operands(command[1:], frozenset(("-h", "-o", "-u")))
            if len(operands) >= 3 and operands[0].casefold() == "cp":
                sources = operands[1:-1]
            elif len(operands) >= 3 and operands[0].casefold() == "rsync":
                sources = [operands[-2]]
        elif len(words) >= 5 and words[:3] == ("gh", "release", "upload"):
            sources = _operands(command[4:], frozenset(("--repo", "-r")))
        elif len(words) >= 4 and words[:3] == ("wrangler", "pages", "deploy"):
            sources = [command[3]]
        elif len(words) >= 3 and words[:2] == ("netlify", "deploy"):
            for offset, word in enumerate(words[2:], 2):
                if word.startswith("--dir="):
                    sources = [command[offset].split("=", 1)[1]]
                elif word == "--dir" and offset + 1 < len(command):
                    sources = [command[offset + 1]]
        elif len(words) >= 3 and words[:2] == ("vercel", "deploy"):
            sources = [command[2]]
        elif len(words) >= 4 and words[:4] == ("az", "storage", "blob", "upload"):
            for offset, word in enumerate(words[4:], 4):
                if word == "--file" and offset + 1 < len(command):
                    sources = [command[offset + 1]]
        elif len(words) >= 3 and words[0] in ("scp", "rsync"):
            value_options = (frozenset(("-p", "-i", "-f", "-o", "-j", "-s", "-c", "-l"))
                             if words[0] == "scp" else
                             frozenset(("-e", "--rsh", "--exclude", "--include", "--files-from")))
            operands = _operands(command[1:], value_options)
            if len(operands) >= 2 and ":" in operands[-1]:
                sources = [source for source in operands[:-1] if ":" not in source]
        elif len(words) >= 3 and words[0] == "surge":
            sources = [command[1]]
        paths.extend((index, _resolve_path(item, source)) for source in sources)
    return tuple(paths)


def stage_reaches_upload(stage: StagedCommand, upload_path: str) -> bool:
    return path_reaches_upload(stage.destination, upload_path)


def path_reaches_upload(destination_path: str, upload_path: str) -> bool:
    destination = _path(destination_path)
    uploaded = _path(upload_path)
    if _DYNAMIC.search(destination_path) or _DYNAMIC.search(upload_path):
        return True
    if (destination == uploaded or uploaded.startswith(destination.rstrip("/") + "/")
            or destination.startswith(uploaded.rstrip("/") + "/")):
        return True
    if any(char in uploaded for char in "*?["):
        return fnmatch.fnmatchcase(destination, uploaded) or fnmatch.fnmatchcase(destination + "/index.html", uploaded)
    return False


def stage_has_docs_source(stage: StagedCommand) -> bool:
    return _source_is_docs(stage.source) or bool(_DYNAMIC.search(stage.source))


def stage_is_dynamic(stage: StagedCommand) -> bool:
    return bool(_DYNAMIC.search(stage.source) or _DYNAMIC.search(stage.destination))
