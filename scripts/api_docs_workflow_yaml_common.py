#!/usr/bin/env python3
"""Shared YAML and shell parsing helpers for API docs workflow guards."""

import re


def strip_yaml_comment(line):
    in_single = False
    in_double = False
    escaped = False
    skip_next = False
    value_start = 0
    comment_at = None
    for index, char in enumerate(line):
        if skip_next:
            skip_next = False
            continue
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char in ":,{[" and not in_single and not in_double:
            value_start = index + 1
            continue
        if char == "'" and not in_double:
            if in_single:
                if index + 1 < len(line) and line[index + 1] == "'":
                    skip_next = True
                    continue
                in_single = False
            elif line[value_start:index].strip() == "":
                in_single = True
            continue
        if char == '"' and not in_single:
            if in_double:
                in_double = False
            elif line[value_start:index].strip() == "":
                in_double = True
            continue
        if (
            char == "#"
            and not in_single
            and not in_double
            and (index == 0 or line[index - 1].isspace())
        ):
            comment_at = index
            break
    if comment_at is not None:
        line = line[:comment_at].rstrip()
    return line


def strip_yaml_comment_continuation(line, quote):
    in_single = quote == "'"
    in_double = quote == '"'
    escaped = False
    skip_next = False
    comment_at = None
    for index, char in enumerate(line):
        if skip_next:
            skip_next = False
            continue
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double:
            escaped = True
            continue
        if char == "'" and not in_double:
            if in_single:
                if index + 1 < len(line) and line[index + 1] == "'":
                    skip_next = True
                    continue
                in_single = False
            continue
        if char == '"' and not in_single:
            if in_double:
                in_double = False
            continue
        if (
            char == "#"
            and not in_single
            and not in_double
            and (index == 0 or line[index - 1].isspace())
        ):
            comment_at = index
            break
    if comment_at is not None:
        line = line[:comment_at].rstrip()
    return line


def strip_shell_comment_stateful(line, state):
    in_single, in_double, escaped = state
    comment_at = None
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and not in_single:
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


def yaml_double_unescape(value):
    def replace_match(match):
        escape = match.group(1)
        if escape == "0":
            return "\0"
        if escape == "a":
            return "\a"
        if escape == "b":
            return "\b"
        if escape == "t" or escape == "\t":
            return "\t"
        if escape == "n":
            return "\n"
        if escape == "v":
            return "\v"
        if escape == "f":
            return "\f"
        if escape == "r":
            return "\r"
        if escape == "e":
            return "\033"
        if escape in {'"', "/", "\\", "_", " "}:
            return escape
        if escape.startswith("x") and len(escape) == 3:
            return chr(int(escape[1:], 16))
        if escape.startswith("u") and len(escape) == 5:
            return chr(int(escape[1:], 16))
        if escape.startswith("U") and len(escape) == 9:
            return chr(int(escape[1:], 16))
        return f"\\{escape}"

    return re.sub(r"\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)", replace_match, value)


def quoted_scalar_continues(value):
    value = value.strip()
    if not value or value[0] not in "'\"":
        return None
    quote = value[0]
    escaped = False
    index = 1
    while index < len(value):
        char = value[index]
        if quote == '"' and escaped:
            escaped = False
        elif quote == '"' and char == "\\":
            escaped = True
        elif quote == "'" and char == "'" and index + 1 < len(value) and value[index + 1] == "'":
            index += 1
        elif char == quote:
            return None
        index += 1
    return quote


def quoted_value_tail(value, quote):
    escaped = False
    index = 0
    while index < len(value):
        char = value[index]
        if quote == '"' and escaped:
            escaped = False
        elif quote == '"' and char == "\\":
            escaped = True
        elif quote == "'" and char == "'" and index + 1 < len(value) and value[index + 1] == "'":
            index += 1
        elif char == quote:
            return value[:index], value[index + 1:]
        index += 1
    return None


def decode_value(value, quote):
    if quote == "'":
        return value.replace("''", "'")
    return yaml_double_unescape(value)


def decode_plain_value(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return decode_value(value[1:-1], value[0])
    return value


def unquoted_yaml_text(line):
    chars = list(line)
    in_single = False
    in_double = False
    escaped = False
    index = 0
    while index < len(chars):
        char = chars[index]
        if in_double:
            chars[index] = " "
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_double = False
        elif in_single:
            chars[index] = " "
            if char == "'" and index + 1 < len(chars) and chars[index + 1] == "'":
                chars[index + 1] = " "
                index += 1
            elif char == "'":
                in_single = False
        elif char == '"':
            chars[index] = " "
            in_double = True
        elif char == "'":
            chars[index] = " "
            in_single = True
        index += 1
    return "".join(chars)
