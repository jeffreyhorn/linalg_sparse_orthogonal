"""Recognize shell staging commands without merging unrelated workflow steps."""

from __future__ import annotations

import fnmatch
import posixpath
import re
from dataclasses import dataclass, replace
from typing import Iterator

if __package__:
    from .api_docs_workflow_policy_inputs import PolicyInput
    from .api_docs_workflow_yaml_common import strip_shell_comment_stateful
else:
    from api_docs_workflow_policy_inputs import PolicyInput
    from api_docs_workflow_yaml_common import strip_shell_comment_stateful


_DYNAMIC = re.compile(r"\$\{|\$\(|\$[A-Za-z_0-9@*?#$!-]|%[A-Za-z_][A-Za-z_0-9]*%|^~")
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


def _substitution_end(value: str, start: int) -> int:
    depth = 1
    quote = ""
    index = start
    while index < len(value):
        char = value[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = ""
            elif quote == '"' and char == "$" and value[index + 1:index + 2] == "(":
                depth += 1
                index += 2
                continue
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return len(value) - 1


def _backtick_end(value: str, start: int) -> int:
    index = start
    while index < len(value):
        if value[index] == "\\":
            index += 2
            continue
        if value[index] == "`":
            return index
        index += 1
    return len(value) - 1


def _posix_commands(value: str) -> tuple[tuple[str, ...], ...]:
    active = _without_comments(value, False).replace("\\\n", "")
    commands: list[tuple[str, ...]] = []
    current: list[str] = []
    assignments: list[bool] = []
    word: list[str] = []
    started = False
    quote = ""
    assignment_name_valid = True
    assignment_seen_equal = False
    index = 0

    def end_word() -> None:
        nonlocal started, assignment_name_valid, assignment_seen_equal
        if started:
            current.append("".join(word))
            assignments.append(assignment_name_valid and assignment_seen_equal)
            word.clear()
            started = False
            assignment_name_valid = True
            assignment_seen_equal = False

    def end_command() -> None:
        end_word()
        if current:
            prefix = 0
            while prefix < len(current) and assignments[prefix]:
                prefix += 1
            if prefix < len(current):
                commands.append(tuple(current[prefix:]))
            current.clear()
            assignments.clear()

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
            elif char == "`":
                word.append("${unresolved_substitution}")
                index = _backtick_end(active, index + 1)
            elif char == "\\" and index + 1 < len(active) and active[index + 1] in '$`"\\\n':
                index += 1
                if active[index] != "\n":
                    word.append(active[index])
            else:
                word.append(char)
        elif char in "'\"":
            quote = char
            started = True
            if not assignment_seen_equal:
                assignment_name_valid = False
        elif char == "\\":
            if index + 1 >= len(active):
                return ()
            index += 1
            if active[index] != "\n":
                word.append(active[index])
                started = True
                if not assignment_seen_equal:
                    assignment_name_valid = False
        elif char == "$" and active[index + 1:index + 2] == "(":
            word.append("${unresolved_substitution}")
            started = True
            if not assignment_seen_equal:
                assignment_name_valid = False
            index = _substitution_end(active, index + 2)
        elif char == "`":
            word.append("${unresolved_substitution}")
            started = True
            if not assignment_seen_equal:
                assignment_name_valid = False
            index = _backtick_end(active, index + 1)
        elif char in "<>":
            if started and "".join(word).isdigit():
                word.clear()
                started = False
                assignment_name_valid = True
                assignment_seen_equal = False
            else:
                end_word()
            if active[index + 1:index + 2] in ("<", ">", "&", "|"):
                index += 1
            index += 1
            while index < len(active) and active[index] in " \t":
                index += 1
            target_quote = ""
            while index < len(active):
                target_char = active[index]
                if target_char == "\\" and target_quote != "'":
                    index += 2
                    continue
                if target_quote:
                    if target_char == target_quote:
                        target_quote = ""
                elif target_char in "'\"":
                    target_quote = target_char
                elif target_char.isspace() or target_char in _SEPARATORS:
                    break
                index += 1
            continue
        elif char in "()":
            end_command()
            commands.append((char,))
        elif char in _SEPARATORS:
            end_command()
        elif char.isspace():
            end_word()
        else:
            if char == "=" and not assignment_seen_equal:
                assignment_seen_equal = bool(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", "".join(word)))
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


def _resolve_path(directory: str | None, value: str) -> str:
    if directory is None or value.startswith("~"):
        return value
    path = value.replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
        return path
    if _DYNAMIC.search(directory):
        return directory.rstrip("/") + "/" + path
    return posixpath.normpath(posixpath.join(directory.replace("\\", "/"), path))


def _recognizable_docs_source(value: str) -> bool:
    path = _path(value)
    parts = path.split("/")
    return "docs" in parts or path in (".", "*") or all(part == ".." for part in parts)


def _unbounded_source(value: str) -> bool:
    path = _path(value)
    return path.startswith("/") or bool(re.match(r"^[a-z]:/", path))


def _source_is_docs(value: str) -> bool:
    if _recognizable_docs_source(value):
        return True
    if _unbounded_source(value):
        return True
    path = _path(value)
    if any(char in path for char in "*?["):
        first = path.split("/", 1)[0]
        return first == ".." or any(char in first for char in "*?[")
    return False


def glob_matches_path(pattern: str, path: str) -> bool:
    """Match path segments, allowing ** to consume zero or more directories."""
    pattern_parts = pattern.split("/")
    path_parts = path.split("/")

    def matches(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        if pattern_parts[pattern_index] == "**":
            return matches(pattern_index + 1, path_index) or (
                path_index < len(path_parts) and matches(pattern_index, path_index + 1)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], pattern_parts[pattern_index])
            and matches(pattern_index + 1, path_index + 1)
        )

    return matches(0, 0)


def _tar_stage(command: tuple[str, ...]) -> tuple[StagedCommand, ...]:
    directory = "."
    archive: str | None = None
    sources: list[str] = []
    args = command[1:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-C", "--directory"):
            if index + 1 >= len(args):
                return ()
            directory = args[index + 1]
            index += 2
            continue
        if arg.startswith("--directory="):
            directory = arg.split("=", 1)[1]
        elif arg.startswith("-C") and len(arg) > 2:
            directory = arg[2:]
        elif arg.startswith("--file="):
            archive = arg.split("=", 1)[1]
        elif arg in ("-f", "--file") or (
            (index == 0 or arg.startswith("-"))
            and re.fullmatch(r"-?[A-Za-z]*f[A-Za-z]*", arg)
        ):
            if index + 1 >= len(args):
                return ()
            archive = args[index + 1]
            index += 2
            continue
        elif not arg.startswith("-") and archive is not None:
            sources.append(posixpath.normpath(posixpath.join(directory, arg)))
        index += 1
    if archive is None:
        return ()
    if archive == "-":
        archive = "${stdout_archive}"
    return tuple(StagedCommand(source, archive, "archives") for source in sources)


def _zip_stage(command: tuple[str, ...]) -> tuple[StagedCommand, ...]:
    operands: list[str] = []
    args = command[1:]
    index = 0
    after_options = False
    while index < len(args):
        arg = args[index]
        if arg == "--":
            after_options = True
        elif not after_options and arg == "-b":
            if index + 1 >= len(args):
                return ()
            index += 1
        elif not after_options and arg.startswith("-") and arg != "-":
            if not re.fullmatch(r"-[rjq0-9]+", arg):
                return (StagedCommand("${unsupported_zip_options}", "${unsupported_zip_options}", "archives"),)
        else:
            operands.append(arg)
        index += 1
    if len(operands) < 2:
        return ()
    destination = "${stdout_archive}" if operands[0] == "-" else operands[0]
    return tuple(StagedCommand(source, destination, "archives") for source in operands[1:])


def _posix_stage(command: tuple[str, ...]) -> tuple[StagedCommand, ...]:
    if len(command) < 3:
        return ()
    executable = posixpath.basename(command[0]).casefold()
    if executable in _COPY:
        args: list[str] = []
        target = None
        index = 1
        after_options = False
        while index < len(command):
            arg = command[index]
            if arg == "--":
                after_options = True
            elif not after_options and executable in ("cp", "mv") and arg in ("-t", "--target-directory"):
                if index + 1 >= len(command):
                    return ()
                index += 1
                target = command[index]
            elif not after_options and executable in ("cp", "mv") and arg.startswith("--target-directory="):
                target = arg.split("=", 1)[1]
            elif not after_options and executable in ("cp", "mv") and arg.startswith("-t") and len(arg) > 2:
                target = arg[2:]
            elif after_options or not arg.startswith("-"):
                args.append(arg)
            index += 1
        if target is not None:
            return tuple(StagedCommand(source, target, "stages") for source in args)
        if len(args) >= 2:
            return tuple(StagedCommand(source, args[-1], "stages") for source in args[:-1])
    if executable == "tar":
        return _tar_stage(command)
    if executable == "zip":
        return _zip_stage(command)
    if executable in _ARCHIVE - {"tar", "zip"}:
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
    value_options = frozenset((
        "-erroraction", "-warningaction", "-informationaction", "-progressaction",
        "-errorvariable", "-warningvariable", "-informationvariable", "-outvariable",
        "-pipelinevariable", "-compressionlevel", "-exclude", "-include", "-filter",
        "-credential", "-tosession", "-fromsession",
    ))
    switch_options = frozenset((
        "-recurse", "-force", "-passthru", "-container", "-whatif", "-confirm", "-update",
    ))
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
            if name in value_options:
                if index + 1 >= len(command):
                    return ()
                index += 2
                continue
            if name in switch_options or (":" in name and name.split(":", 1)[0] in value_options):
                index += 1
                continue
            return (StagedCommand("${unsupported_powershell_option}", "${unsupported_powershell_option}",
                                  "archives" if executable == "compress-archive" else "stages"),)
        elif arg != ",":
            positional.append(arg)
        index += 1
    sources = named.get("-path", named.get("-literalpath"))
    destinations = named.get("-destination", named.get("-destinationpath"))
    if destinations is None and positional:
        destinations = (positional[-1],)
        positional.pop()
    if sources is None:
        sources = tuple(positional)
    else:
        sources += tuple(positional)
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


def _commands_with_directories(item: PolicyInput) -> Iterator[tuple[int, tuple[str, ...], str | None]]:
    directory = item.working_directory.value if item.working_directory is not None else None
    subshell_directories: list[str | None] = []
    powershell_directories: list[str | None] = []
    for index, command in enumerate(_commands(item)):
        if item.shell_family == "posix" and command == ("(",):
            subshell_directories.append(directory)
            continue
        if item.shell_family == "posix" and command == (")",):
            directory = (subshell_directories.pop() if subshell_directories
                         else "${unresolved_working_directory}")
            continue
        yield index, command, directory
        if item.shell_family == "posix" and command[0] == "cd":
            args = command[1:]
            if args[:1] == ("--",):
                args = args[1:]
            if (len(args) != 1 or args[0].startswith(("-", "~", "/"))
                    or _DYNAMIC.search(args[0])):
                directory = "${unresolved_working_directory}"
            else:
                directory = _resolve_path(directory or ".", args[0])
        elif item.shell_family == "powershell":
            executable = command[0].casefold()
            if executable in ("cd", "sl", "set-location", "pushd", "push-location"):
                if executable in ("pushd", "push-location"):
                    powershell_directories.append(directory)
                args = command[1:]
                if args[:1] and args[0].casefold() in ("-path", "-literalpath"):
                    args = args[1:]
                if (len(args) != 1 or args[0].startswith(("-", "~", "/"))
                        or _DYNAMIC.search(args[0]) or re.match(r"^[A-Za-z]:[/\\]", args[0])):
                    directory = "${unresolved_working_directory}"
                else:
                    directory = _resolve_path(directory or ".", args[0])
            elif executable in ("popd", "pop-location"):
                directory = (powershell_directories.pop() if powershell_directories
                             else "${unresolved_working_directory}")


def staged_commands(item: PolicyInput) -> tuple[StagedCommand, ...]:
    parser = _powershell_stage if item.shell_family == "powershell" else _posix_stage
    return tuple(
        replace(
            stage, source=_resolve_path(directory, stage.source),
            destination=_resolve_path(directory, stage.destination), command_index=index,
        )
        for index, command, directory in _commands_with_directories(item)
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
            if word in value_options:
                index += 1
        else:
            result.append(word)
        index += 1
    return result


def _publication_operands(
    words: tuple[str, ...], value_options: frozenset[str], switch_options: frozenset[str]
) -> list[str] | None:
    operands: list[str] = []
    index = 0
    after_options = False
    while index < len(words):
        word = words[index]
        if word == "--":
            after_options = True
        elif not after_options and word in value_options:
            if index + 1 >= len(words):
                return None
            index += 1
        elif not after_options and any(word.startswith(option + "=") for option in value_options):
            pass
        elif not after_options and word in switch_options:
            pass
        elif not after_options and word.startswith("-"):
            return None
        else:
            operands.append(word)
        index += 1
    return operands


def published_paths(item: PolicyInput) -> tuple[tuple[int, str], ...]:
    """Return explicit local inputs to shell publication commands."""
    paths: list[tuple[int, str]] = []
    for index, command, directory in _commands_with_directories(item):
        words = tuple(word.casefold() for word in command)
        sources: list[str] = []
        if words[:3] in (("aws", "s3", "cp"), ("aws", "s3", "sync")):
            operands = _publication_operands(
                command[3:],
                frozenset(("--profile", "--region", "--endpoint-url", "--exclude", "--include", "--acl")),
                frozenset(("--recursive", "--delete", "--dryrun", "--quiet", "--no-progress",
                           "--only-show-errors", "--follow-symlinks", "--no-follow-symlinks")),
            )
            if operands and len(operands) >= 2:
                if not operands[0].casefold().startswith("s3://"):
                    sources = [operands[0]]
            else:
                sources = ["${unresolved_publication_source}"]
        elif words[:2] in (("rclone", "copy"), ("rclone", "sync"), ("rclone", "move")):
            operands = _publication_operands(
                command[2:],
                frozenset(("--config", "--exclude", "--include", "--transfers")),
                frozenset(("--progress", "-P", "-v", "-vv", "--dry-run", "--fast-list")),
            )
            sources = [operands[0]] if operands and len(operands) >= 2 else ["${unresolved_publication_source}"]
        elif len(words) >= 3 and words[0] == "gsutil":
            operands = _operands(command[1:], frozenset(("-h", "-o", "-u")))
            if len(operands) >= 3 and operands[0].casefold() == "cp":
                sources = [source for source in operands[1:-1] if not source.casefold().startswith("gs://")]
            elif len(operands) >= 3 and operands[0].casefold() == "rsync":
                sources = [operands[-2]] if not operands[-2].casefold().startswith("gs://") else []
        elif len(words) >= 5 and words[:3] == ("gh", "release", "upload"):
            sources = [source.split("#", 1)[0] for source in _operands(command[4:], frozenset(("--repo", "-r")))]
        elif words[:3] == ("wrangler", "pages", "deploy"):
            operands = _publication_operands(
                command[3:], frozenset(("--project-name", "--branch", "--commit-hash")),
                frozenset(("--commit-dirty",)),
            )
            sources = [operands[0]] if operands else ["${unresolved_publication_source}"]
        elif len(words) >= 3 and words[:2] == ("netlify", "deploy"):
            for offset, word in enumerate(words[2:], 2):
                if word.startswith("--dir="):
                    sources = [command[offset].split("=", 1)[1]]
                elif word == "--dir" and offset + 1 < len(command):
                    sources = [command[offset + 1]]
        elif words[:2] == ("vercel", "deploy"):
            operands = _publication_operands(
                command[2:], frozenset(("--scope", "--token", "--name")),
                frozenset(("--prod", "--yes", "--force", "--prebuilt")),
            )
            sources = [operands[0]] if operands else ["${unresolved_publication_source}"]
        elif len(words) >= 4 and words[:4] == ("az", "storage", "blob", "upload"):
            for offset, word in enumerate(words[4:], 4):
                if word == "--file" and offset + 1 < len(command):
                    sources = [command[offset + 1]]
        elif len(words) >= 3 and words[0] in ("scp", "rsync"):
            value_options = (frozenset(("-P", "-i", "-F", "-o", "-J", "-S", "-c", "-l"))
                             if words[0] == "scp" else
                             frozenset(("-e", "--rsh", "--exclude", "--include", "--files-from")))
            operands = _operands(command[1:], value_options)
            if len(operands) >= 2 and ":" in operands[-1]:
                sources = [source for source in operands[:-1] if ":" not in source]
        elif len(words) >= 3 and words[0] == "surge":
            sources = [command[1]]
        paths.extend((index, _resolve_path(directory, source)) for source in sources)
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
        return glob_matches_path(uploaded, destination) or glob_matches_path(uploaded, destination + "/index.html")
    return False


def stage_has_docs_source(stage: StagedCommand) -> bool:
    return _source_is_docs(stage.source) or bool(_DYNAMIC.search(stage.source))


def stage_is_unbounded(stage: StagedCommand) -> bool:
    return _unbounded_source(stage.source)


def stage_is_dynamic(stage: StagedCommand) -> bool:
    return bool(
        _DYNAMIC.search(stage.source) or _DYNAMIC.search(stage.destination)
        or (any(char in stage.source for char in "*?[")
            and _source_is_docs(stage.source) and not _recognizable_docs_source(stage.source))
    )
