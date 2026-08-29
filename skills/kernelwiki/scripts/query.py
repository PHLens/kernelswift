from __future__ import annotations

import argparse
from collections.abc import Sequence
import ctypes
import os
from pathlib import Path
import stat

from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli
from role_context import load_authority_snapshot, load_role_context
from role_search import ROLE_GROUPS, parse_role_query_request, role_result_payload, role_search
from search import FILTER_FIELDS, parse_query_request, query_payload
from validate import validate_skill_root


class StableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise KernelWikiError("cli-input-invalid", message)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _group_limit(value: str) -> tuple[str, int]:
    if not isinstance(value, str) or "=" not in value:
        raise argparse.ArgumentTypeError("expected GROUP=LIMIT")
    name, raw_limit = value.split("=", 1)
    if name not in ROLE_GROUPS:
        raise argparse.ArgumentTypeError(f"unknown result group {name!r}")
    return name, _positive_int(raw_limit)


def _parser() -> StableArgumentParser:
    parser = StableArgumentParser(description="Search the local KernelWiki corpus")
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", choices=("cards", "sources", "both"), default="both")
    parser.add_argument("--limit", type=_positive_int, default=20)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--profile-snapshot", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--group-limit", type=_group_limit, action="append", default=[])
    parser.add_argument("--show-excluded", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--type", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--repository", "--repo", dest="repository", action="append")
    parser.add_argument("--language", action="append")
    parser.add_argument("--target", action="append")
    parser.add_argument("--target-match", action="append")
    parser.add_argument("--symptom", action="append")
    parser.add_argument("--kernel-type", action="append")
    parser.add_argument("--evidence-level", action="append")
    parser.add_argument("--reproduction", action="append")
    parser.add_argument("--audience", action="append")
    parser.add_argument("--has-code", choices=("true", "false"), action="append")
    return parser


def _filters(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    values = {}
    for field in FILTER_FIELDS:
        attribute = field.replace("-", "_")
        selected = getattr(args, attribute)
        if selected:
            values[field] = tuple(sorted(set(selected)))
    return values


def _markdown(payload) -> str:
    lines = [f"# KernelWiki search: {payload['query'] or '(filters only)'}", ""]
    if not payload["results"]:
        return "\n".join([*lines, "- _No matches._", ""])
    for result in payload["results"]:
        lines.append(
            f"- [{result['record_id']}]({result['path']}) — {result['title']} — {result['excerpt']}"
        )
    return "\n".join([*lines, ""])


def _role_markdown(payload) -> str:
    lines = ["# KernelWiki role-aware search", ""]
    for group in ROLE_GROUPS:
        lines.extend([f"## {group}", ""])
        records = payload["groups"][group]
        if not records:
            lines.extend(["- _None._", ""])
            continue
        for record in records:
            reasons = ", ".join(record["admission"]["reasons"]) or "none"
            lines.append(
                f"- [{record['id']}]({record['path']}) — {record['title']} — "
                f"{record['admission']['status']} — reasons: {reasons}"
            )
        lines.append("")
    return "\n".join(lines)


def _role_authority(context):
    if context.role != "coder" or context.implementation_profile_status == "missing":
        return None
    return load_authority_snapshot(context)


def _open_output_parent(path: Path) -> tuple[int, str, Path]:
    destination = Path(path)
    raw = os.fspath(destination)
    if not raw or "\x00" in raw:
        raise KernelWikiError("role-output-invalid", "output path must be nonempty normalized text", destination)
    absolute = Path(os.path.abspath(raw))
    parts = absolute.parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts[1:]):
        raise KernelWikiError("role-output-invalid", "output path must be normalized", destination)
    name = parts[-1]
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parts[0], flags)
    try:
        for component in parts[1:-1]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        parent = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        return descriptor, name, parent
    except BaseException:
        os.close(descriptor)
        raise


def _current_directory_path(descriptor: int, output: Path) -> Path:
    try:
        raw = os.readlink(f"/proc/self/fd/{descriptor}")
        if not raw.startswith("/") or raw.endswith(" (deleted)"):
            raise OSError("output directory no longer has a stable absolute path")
        return Path(raw).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("role-output-invalid", f"cannot revalidate output directory: {error}", output) from error


def _inside_project(parent: Path, context) -> bool:
    if context.project_root is None:
        return False
    project = Path(context.project_root).resolve()
    return parent == project or project in parent.parents


def _stable_outside_parent(parent: Path, context) -> bool:
    if context.project_root is not None:
        project = Path(context.project_root).resolve()
        if parent in project.parents:
            return True
    trusted_system_parents = {Path("/tmp").resolve(), Path("/var/tmp").resolve()}
    if parent not in trusted_system_parents:
        return False
    parent_stat = parent.stat()
    root_stat = Path("/").stat()
    return (
        stat.S_ISDIR(parent_stat.st_mode)
        and parent_stat.st_uid == 0
        and bool(parent_stat.st_mode & stat.S_ISVTX)
        and root_stat.st_uid == 0
        and not os.access(Path("/"), os.W_OK)
    )


def _link_open_file_no_clobber(source_fd: int, destination_fd: int, name: str, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    if linkat(source_fd, b"", destination_fd, os.fsencode(name), 0x1000) != 0:  # AT_EMPTY_PATH
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), output)


def _require_owned_output(
    descriptor: int,
    name: str,
    identity: tuple[int, int],
    output: Path,
) -> None:
    try:
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except OSError as error:
        raise KernelWikiError("role-output-invalid", f"cannot revalidate published output: {error}", output) from error
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != identity:
        raise KernelWikiError("role-output-invalid", "published output identity changed", output)


def _write_explicit_output(path: Path, text: str, context) -> None:
    descriptor = -1
    output_fd = -1
    final_name: str | None = None
    final_identity: tuple[int, int] | None = None
    try:
        descriptor, name, parent = _open_output_parent(path)
        if _inside_project(parent, context):
            raise KernelWikiError(
                "role-output-active-project",
                "role query receipts cannot be written inside active project state",
                Path(path),
            )
        if not _stable_outside_parent(parent, context):
            raise KernelWikiError(
                "role-output-unstable-parent",
                "output parent could be relocated into active project state; use an ancestor of the project or exact /tmp or /var/tmp",
                Path(path),
            )
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise KernelWikiError("role-output-exists", "output path already exists", Path(path))
        anonymous_flag = getattr(os, "O_TMPFILE", 0)
        if not anonymous_flag:
            raise KernelWikiError("role-output-invalid", "anonymous temporary files are unsupported", Path(path))
        flags = os.O_WRONLY | anonymous_flag
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        output_fd = os.open(".", flags, 0o600, dir_fd=descriptor)
        data = text.encode("utf-8")
        offset = 0
        while offset < len(data):
            written = os.write(output_fd, data[offset:])
            if written <= 0:
                raise OSError("output write made no progress")
            offset += written
        os.fsync(output_fd)
        temporary_stat = os.fstat(output_fd)
        final_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        if _inside_project(_current_directory_path(descriptor, Path(path)), context):
            raise KernelWikiError(
                "role-output-active-project",
                "output directory moved into active project state before publication",
                Path(path),
            )
        _link_open_file_no_clobber(output_fd, descriptor, name, Path(path))
        final_name = name
        _require_owned_output(descriptor, name, final_identity, Path(path))
        os.fsync(descriptor)
        if _inside_project(_current_directory_path(descriptor, Path(path)), context):
            _require_owned_output(descriptor, name, final_identity, Path(path))
            os.unlink(name, dir_fd=descriptor)
            final_name = None
            os.fsync(descriptor)
            raise KernelWikiError(
                "role-output-active-project",
                "output directory moved into active project state during publication",
                Path(path),
            )
        _require_owned_output(descriptor, name, final_identity, Path(path))
        final_name = None
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise KernelWikiError("role-output-invalid", f"cannot write role query output: {error}", Path(path)) from error
    finally:
        if descriptor >= 0:
            if final_name is not None and final_identity is not None:
                try:
                    current = os.stat(final_name, dir_fd=descriptor, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == final_identity:
                        os.unlink(final_name, dir_fd=descriptor)
                except OSError:
                    pass
            if output_fd >= 0:
                os.close(output_fd)
            os.close(descriptor)


def _main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(list(argv))
    if args.profile_snapshot is not None:
        raise KernelWikiError(
            "phase-c-required",
            "--profile-snapshot is not a role context; use --context",
            args.profile_snapshot,
        )
    corpus = validate_skill_root(args.root)
    if args.context is None:
        if args.group_limit or args.show_excluded:
            raise KernelWikiError("cli-input-invalid", "--group-limit/--show-excluded require --context")
        if args.output is not None:
            raise KernelWikiError("cli-input-invalid", "--output requires --context")
        request = parse_query_request(args.text, _filters(args), args.scope, args.limit)
        payload = query_payload(corpus, request)
        text = _markdown(payload) if args.format == "markdown" else canonical_json_bytes(payload).decode("utf-8")
        print(text, end="")
        return 0

    context = load_role_context(args.context)
    limits = {
        "admitted": args.limit,
        "conditional": args.limit,
        "analogy_only": args.limit,
        "excluded": args.limit,
    }
    for name, value in args.group_limit:
        limits[name] = value
    request = parse_role_query_request(
        args.text,
        _filters(args),
        args.scope,
        limits,
        args.show_excluded,
    )
    result = role_search(corpus, request, context, _role_authority(context))
    payload = role_result_payload(result)
    text = _role_markdown(payload) if args.format == "markdown" else canonical_json_bytes(payload).decode("utf-8")
    if args.output is not None:
        _write_explicit_output(args.output, text, context)
    else:
        print(text, end="")
    return 0


def main(argv: Sequence[str]) -> int:
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
