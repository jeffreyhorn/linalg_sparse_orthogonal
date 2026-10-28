#!/usr/bin/env python3
"""Shell-comment handling shared by the API docs workflow policies."""


def strip_shell_comment_stateful(line, state, powershell=False):
    in_single, in_double, escaped = state
    comment_at = None
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == ("`" if powershell else "\\") and not in_single:
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if (
            char == "#"
            and not in_single
            and not in_double
            and (index == 0 or line[index - 1].isspace() or line[index - 1] in ";&|()")
        ):
            comment_at = index
            break
    if comment_at is not None:
        line = line[:comment_at].rstrip()
        in_single = False
        in_double = False
        escaped = False
    return line, (in_single, in_double, escaped)
