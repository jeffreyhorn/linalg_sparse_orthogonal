#!/usr/bin/env python3
"""Validate the installed static package without inferring a shared ABI."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


PC_FIELDS = {
    "Name": "sparse",
    "Description": "Static archive package metadata for sparse linear algebra",
    "Libs": "-L${libdir} -lsparse_lu_ortho -lm",
}
PC_VARIABLES = {"prefix", "exec_prefix", "libdir", "includedir"}
FORBIDDEN_CMAKE = re.compile(
    r"\b(?:SHARED IMPORTED|MODULE IMPORTED|IMPORTED_IMPLIB(?:_[A-Z_]+)?|"
    r"IMPORTED_SONAME(?:_[A-Z_]+)?|IMPORTED_NO_SONAME(?:_[A-Z_]+)?|"
    r"SOVERSION|INSTALL_NAME_DIR|MACOSX_RPATH|CMAKE_INSTALL_RPATH|"
    r"BUILD_SHARED_LIBS|SPARSE_(?:ABI|EXPORT|IMPORT|LINKAGE|LINK_MODE|"
    r"LIBRARY_TYPE|LIBRARY_KIND)(?:_[A-Z0-9_]*)?|"
    r"SPARSE_[A-Z0-9_]*(?:SHARED|STATIC)[A-Z0-9_]*)\b|"
    r"\bCOMPONENTS?\s+(?:STATIC|SHARED)\b|Sparse::[^\s\"']*shared",
    re.IGNORECASE,
)
IMPORTED_DYNAMIC_LOCATION = re.compile(
    r"\bIMPORTED_LOCATION[A-Z_]*[^\n]*(?:\.so(?:\.[0-9]+)*|\.dylib|\.dll)",
    re.IGNORECASE,
)
IMPORTED_TARGET = re.compile(
    r'(?m)^\s*add_library\s*\(\s*(?:"([^"]+)"|([^\s()"]+))\s+([A-Z]+)\s+IMPORTED'
    r"(?:\s+GLOBAL)?\s*\)",
    re.IGNORECASE,
)
TARGETS_INCLUDE = re.compile(
    r'(?m)^[ \t]*(?i:include)\s*\(\s*"?\$\{CMAKE_CURRENT_LIST_DIR\}'
    r'/SparseTargets\.cmake"?\s*\)'
)
PACKAGE_VERSION_ASSIGNMENT = re.compile(
    r'(?m)^[ \t]*(?i:set)\s*\(\s*PACKAGE_VERSION\s+'
    r'(?:"([^"]+)"|([^\s)]+))\s*\)'
)


def read_text(path: Path, errors: list[str]) -> str:
    if not path.is_file() or path.is_symlink():
        errors.append(f"missing regular file: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def check_pc(path: Path, prefix: Path, version: str, errors: list[str]) -> int | None:
    fields: dict[str, str] = {}
    variables: dict[str, str] = {}
    for line in read_text(path, errors).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        assignment = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        field = re.fullmatch(r"([A-Za-z][A-Za-z0-9_.]*):\s*(.*)", line)
        if assignment:
            key = assignment.group(1)
            if key not in PC_VARIABLES or key in variables:
                errors.append(f"unsupported or duplicate pkg-config variable: {key}")
            variables[key] = assignment.group(2)
        elif field:
            key = field.group(1)
            if key in fields:
                errors.append(f"duplicate pkg-config field: {key}")
            fields[key] = " ".join(field.group(2).split())
        else:
            errors.append(f"unrecognized pkg-config line: {line}")

    if set(variables) != PC_VARIABLES:
        errors.append(f"pkg-config variables differ: {sorted(variables)}")
    if "prefix" in variables:
        stored_prefix = Path(variables["prefix"])
        if not stored_prefix.is_absolute():
            errors.append(f"pkg-config prefix must be absolute: {variables['prefix']}")
        elif stored_prefix.resolve() != prefix.resolve():
            errors.append(f"pkg-config prefix differs from install prefix: {variables['prefix']}")
    for key, expected_value in {
        "exec_prefix": "${prefix}",
        "libdir": "${exec_prefix}/lib",
        "includedir": "${prefix}/include",
    }.items():
        if key in variables and variables[key] != expected_value:
            errors.append(f"pkg-config {key} differs from {expected_value}")
    cflags = fields.get("Cflags", "")
    width_match = re.fullmatch(
        r"-I\$\{includedir\} -DSPARSE_PACKAGE_IDX_BITS=(32|64)", cflags
    )
    if not width_match:
        errors.append(
            f"pkg-config Cflags must contain exactly one supported package width: {cflags}"
        )
    expected = {**PC_FIELDS, "Version": version, "Cflags": cflags}
    if fields != expected:
        errors.append(f"pkg-config fields differ from static contract: {fields}")
    return int(width_match.group(1)) if width_match else None


def check_cmake(package_dir: Path, archive_name: str, version: str,
                errors: list[str]) -> int | None:
    required = {
        "SparseConfig.cmake",
        "SparseConfigVersion.cmake",
        "SparseTargets.cmake",
    }
    files = sorted(package_dir.glob("*.cmake"))
    names = {path.name for path in files}
    contents: dict[str, str] = {}
    missing = required - names
    if missing:
        errors.append(f"missing CMake package files: {sorted(missing)}")
    if not any(re.fullmatch(r"SparseTargets-[A-Za-z0-9_]+\.cmake", name)
               for name in names):
        errors.append("missing CMake imported archive configuration")
    for path in files:
        if path.name not in required and not re.fullmatch(
            r"SparseTargets-[A-Za-z0-9_]+\.cmake", path.name
        ):
            errors.append(f"unexpected CMake package file: {path.name}")
        content = read_text(path, errors)
        contents[path.name] = content
        match = FORBIDDEN_CMAKE.search(content)
        if match:
            errors.append(f"unsupported CMake loader/selector metadata in {path.name}: "
                          f"{match.group(0)}")
        dynamic_location = IMPORTED_DYNAMIC_LOCATION.search(content)
        if dynamic_location:
            errors.append(f"dynamic imported location in {path.name}: "
                          f"{dynamic_location.group(0)}")

    declarations = [(quoted or unquoted, library_type.upper())
                    for content in contents.values()
                    for quoted, unquoted, library_type in IMPORTED_TARGET.findall(content)]
    if declarations != [("Sparse::sparse_lu_ortho", "STATIC")]:
        errors.append(f"CMake imported targets differ from static contract: "
                      f"{declarations}")
    package_contents = "\n".join(contents.values())
    expected_location = "${_IMPORT_PREFIX}/lib/" + archive_name
    locations = [quoted or unquoted for quoted, unquoted in re.findall(
        r'(?<!\w)"?IMPORTED_LOCATION(?:_[A-Z0-9_]+)?"?\s+'
        r'(?:"([^"]+)"|([^\s)]+))',
        package_contents,
    )]
    if not locations or any(location != expected_location for location in locations):
        errors.append(
            f"CMake imported locations differ from {expected_location}: {locations}"
        )
    if not TARGETS_INCLUDE.search(contents.get("SparseConfig.cmake", "")):
        errors.append("CMake config does not load the static target")
    versions = [quoted or unquoted for quoted, unquoted in
                PACKAGE_VERSION_ASSIGNMENT.findall(
                    contents.get("SparseConfigVersion.cmake", ""))]
    generated_arch_mismatch_version = '${PACKAGE_VERSION} (${installedBits}bit)'
    if versions not in ([version], [version, generated_arch_mismatch_version]):
        errors.append(f"CMake package version does not match {version}")
    target_values = re.findall(
        r'(?m)^[ \t]*INTERFACE_COMPILE_DEFINITIONS\s+"([^"]*)"',
        contents.get("SparseTargets.cmake", ""),
    )
    definitions = [definition for value in target_values
                   for definition in value.split(";") if definition]
    other_widths = [name for name, content in contents.items()
                    if name != "SparseTargets.cmake" and "SPARSE_PACKAGE_IDX_BITS" in content]
    if definitions not in (["SPARSE_PACKAGE_IDX_BITS=32"],
                           ["SPARSE_PACKAGE_IDX_BITS=64"]) or other_widths:
        errors.append("CMake imported target must declare one supported package width: "
                      f"{definitions}; other files: {other_widths}")
        return None
    return int(definitions[0].split("=", 1)[1])


def check_version_header(path: Path, errors: list[str]) -> int | None:
    definitions = re.findall(r"(?m)^\s*#define\s+SPARSE_BUILT_IDX_BITS\s+(\S+)",
                             read_text(path, errors))
    if len(definitions) != 1 or definitions[0] not in ("32", "64"):
        errors.append(f"installed version header must declare one supported width: "
                      f"{definitions}")
        return None
    return int(definitions[0])


def query_archive_bits(archive: Path, platform: str, errors: list[str]) -> int | None:
    source = (
        "#include <stddef.h>\n#include <stdio.h>\n"
        "size_t sparse_idx_bits(void);\n"
        'int main(void) { printf("%zu\\n", sparse_idx_bits()); return 0; }\n'
    )
    try:
        with tempfile.TemporaryDirectory(prefix="sparse-width-probe-") as temp_dir:
            temp = Path(temp_dir)
            probe_source = temp / "width_probe.c"
            probe_source.write_text(source, encoding="utf-8")
            if platform == "windows":
                (temp / "CMakeLists.txt").write_text(
                    "cmake_minimum_required(VERSION 3.16)\n"
                    "project(sparse_width_probe C)\n"
                    "add_executable(width_probe width_probe.c)\n"
                    'target_link_libraries(width_probe PRIVATE "${SPARSE_ARCHIVE}")\n',
                    encoding="utf-8",
                )
                build = temp / "build"
                commands = [
                    ["cmake", "-S", str(temp), "-B", str(build),
                     f"-DSPARSE_ARCHIVE={archive.resolve().as_posix()}"],
                    ["cmake", "--build", str(build), "--config", "Release"],
                ]
            else:
                executable = temp / "width_probe"
                commands = [
                    [*shlex.split(os.environ.get("CC", "cc")), "-std=c11",
                     str(probe_source), str(archive.resolve()), "-lm", "-o",
                     str(executable)]
                ]
            for command in commands:
                subprocess.run(command, check=True, capture_output=True, text=True,
                               timeout=120)
            if platform == "windows":
                executables = sorted(build.rglob("width_probe.exe"))
                if len(executables) != 1:
                    raise FileNotFoundError("CMake width probe executable was not produced")
                executable = executables[0]
            result = subprocess.run([str(executable)], check=True, capture_output=True,
                                    text=True, timeout=30)
            bits = result.stdout.strip()
            if bits not in ("32", "64"):
                errors.append(f"archive reports unsupported index width: {bits!r}")
                return None
            return int(bits)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        errors.append(f"cannot query installed archive index width: {detail}")
        return None


def validate(prefix: Path, mode: str, platform: str, version: str) -> list[str]:
    errors: list[str] = []
    archive_name = (
        "sparse_lu_ortho.lib" if platform == "windows" else "libsparse_lu_ortho.a"
    )
    library_dir = prefix / "lib"
    package_dir = library_dir / "cmake" / "Sparse"
    allowed = {Path("lib") / archive_name, Path("lib/pkgconfig/sparse.pc")}
    if mode == "cmake":
        allowed.update(path.relative_to(prefix) for path in package_dir.glob("*.cmake"))

    if not (library_dir / archive_name).is_file() or (library_dir / archive_name).is_symlink():
        errors.append(f"missing regular static archive: {library_dir / archive_name}")
    if library_dir.exists():
        for path in library_dir.rglob("*"):
            if (path.is_file() or path.is_symlink()) and path.relative_to(prefix) not in allowed:
                errors.append(f"unexpected installed library/metadata artifact: {path}")
    bin_dir = prefix / "bin"
    if bin_dir.exists() and any(bin_dir.rglob("*")):
        errors.append(f"unexpected installed runtime artifact under {bin_dir}")
    for path in prefix.rglob("*"):
        name = path.name.lower()
        if (path.is_file() or path.is_symlink() or path.is_dir()) and (
            name.endswith((".so", ".dylib", ".dll", ".dll.a", ".tbd", ".exp",
                           ".framework", ".xcframework"))
            or ".so." in name
        ):
            errors.append(f"unexpected shared/import artifact: {path}")

    pc_bits = check_pc(library_dir / "pkgconfig" / "sparse.pc", prefix, version, errors)
    header_bits = check_version_header(prefix / "include" / "sparse" /
                                       "sparse_version.h", errors)
    target_bits = None
    if mode == "cmake":
        target_bits = check_cmake(package_dir, archive_name, version, errors)
    elif package_dir.exists():
        errors.append("Make install unexpectedly contains CMake package metadata")
    widths = {"pkg-config": pc_bits, "header": header_bits}
    if mode == "cmake":
        widths["CMake target"] = target_bits
    if not errors:
        widths["archive"] = query_archive_bits(library_dir / archive_name,
                                                platform, errors)
    if not errors and len(set(widths.values())) != 1:
        errors.append(f"installed index widths disagree: {widths}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--mode", choices=("make", "cmake"), required=True)
    parser.add_argument("--platform", choices=("unix", "windows"), required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    errors = validate(args.prefix, args.mode, args.platform, args.version)
    if errors:
        for error in errors:
            print(f"static-install-contract: FAIL: {error}")
        return 1
    print("static-install-contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
