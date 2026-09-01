"""Build and validate the disposable CoreX clock device helper."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Sequence

TARGET = "ivcore11"
TARGET_TRIPLE = "bi-iluvatar-ilurt"
BITCODE_NAME = "corex-clock.bc"
IR_NAME = "corex-clock.ll"
METADATA_NAME = "clock-helper.json"

Runner = Callable[..., Any]


class ClockHelperBuildError(RuntimeError):
    """Raised when the CoreX clock helper cannot be built or validated."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock_compile_command(
    corex_root: Path,
    source: Path,
    output: Path,
) -> list[str]:
    """Return the CoreX Clang command for LLVM bitcode or textual IR."""
    root = Path(corex_root)
    output = Path(output)
    command = [
        str(root / "bin" / "clang++"),
        "-x",
        "ivcore",
        f"--cuda-path={root}",
        f"--cuda-gpu-arch={TARGET}",
        "--cuda-device-only",
        f"-I{root / 'include'}",
        "-Wno-unused-command-line-argument",
    ]
    if output.suffix == ".bc":
        command.extend(["-emit-llvm", "-c"])
    elif output.suffix == ".ll":
        command.extend(["-S", "-emit-llvm"])
    else:
        raise ClockHelperBuildError(
            f"unsupported clock helper output suffix: {output.suffix or '<none>'}"
        )
    command.extend([str(Path(source)), "-o", str(output)])
    return command


_LLVM_VALUE_RE = re.compile(r"%[-a-zA-Z$._0-9]+")
_LLVM_I64_POINTER = r"(?:ptr(?:\s+addrspace\(\d+\))?|i64(?:\s+addrspace\(\d+\))?\*)"
_CLOCK64_INTRINSIC = "@llvm.nvvm.read.ptx.sreg.clock64"


def _matching_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ClockHelperBuildError(f"unterminated LLVM IR delimiter: {opening}")


def _function_definition(ir_text: str, name: str) -> tuple[str, str, str] | None:
    match = re.search(rf"(?m)^\s*define\b[^@\n]*@{re.escape(name)}\s*\(", ir_text)
    if match is None:
        return None
    parameter_start = ir_text.find("(", match.start())
    parameter_end = _matching_delimiter(ir_text, parameter_start, "(", ")")
    body_start = ir_text.find("{", parameter_end)
    if body_start < 0:
        raise ClockHelperBuildError(f"LLVM IR definition has no body: {name}")
    body_end = _matching_delimiter(ir_text, body_start, "{", "}")
    signature = ir_text[match.start():body_start]
    parameters = ir_text[parameter_start + 1:parameter_end]
    body = ir_text[body_start + 1:body_end]
    return signature, parameters, body


def _attribute_groups(ir_text: str) -> dict[str, str]:
    return {
        number: body
        for number, body in re.findall(
            r"(?ms)^\s*attributes\s+#(\d+)\s*=\s*\{(.*?)\}", ir_text
        )
    }


def _definition_has_attribute(
    signature: str,
    attribute_groups: dict[str, str],
    attribute: str,
) -> bool:
    if re.search(rf"\b{re.escape(attribute)}\b", signature):
        return True
    return any(
        re.search(rf"\b{re.escape(attribute)}\b", attribute_groups.get(group, ""))
        for group in re.findall(r"#(\d+)", signature)
    )


def _body_calls_clock64(body: str) -> bool:
    return any(
        _CLOCK64_INTRINSIC in line and re.search(r"\bcall\b", line)
        for line in body.splitlines()
    )


_LLVM_LABEL = r"[-a-zA-Z$._0-9]+"


def _basic_blocks(body: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {"entry": []}
    current = "entry"
    for raw_line in body.splitlines():
        label = re.match(rf"^\s*({_LLVM_LABEL}):(?:\s*;.*)?$", raw_line)
        if label:
            current = label.group(1)
            blocks.setdefault(current, [])
            continue
        blocks[current].append(raw_line)
    return blocks


def _end_control_checks(parameters: str, body: str) -> dict[str, bool]:
    """Validate token-controlled end clocks and the encoded true arm."""
    token_match = re.search(r"\bi64\b[^,%]*?(%[-a-zA-Z$._0-9]+)", parameters)
    if token_match is None:
        return {
            "token_branch": False,
            "clock_after_branch": False,
            "clocks_in_both_arms": False,
            "encoded_true_arm": False,
            "no_inline_asm": "asm" not in body,
        }

    token_values = {token_match.group(1)}
    token_memory: set[str] = set()
    branch: tuple[int, str, str] | None = None
    lines = body.splitlines()

    for index, raw_line in enumerate(lines):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        store_match = re.match(
            r"store\s+i64\s+(%[-a-zA-Z$._0-9]+)\s*,\s*"
            + _LLVM_I64_POINTER
            + r"\s+(%[-a-zA-Z$._0-9]+)",
            line,
        )
        if store_match:
            value, pointer = store_match.groups()
            if value in token_values:
                token_memory.add(pointer)

        assignment = re.match(r"(%[-a-zA-Z$._0-9]+)\s*=\s*(.*)", line)
        if assignment is not None:
            result, expression = assignment.groups()
            operands = set(_LLVM_VALUE_RE.findall(expression))
            if operands & token_values:
                token_values.add(result)
            load_match = re.search(
                r"\bload\s+i64\s*,?\s*"
                + _LLVM_I64_POINTER
                + r"\s+(%[-a-zA-Z$._0-9]+)",
                expression,
            )
            if load_match and load_match.group(1) in token_memory:
                token_values.add(result)

        branch_match = re.match(
            rf"br\s+i1\s+(%[-a-zA-Z$._0-9]+)\s*,\s*"
            rf"label\s+%({_LLVM_LABEL})\s*,\s*label\s+%({_LLVM_LABEL})",
            line,
        )
        if branch_match and branch_match.group(1) in token_values:
            branch = (index, branch_match.group(2), branch_match.group(3))
            break

    if branch is None:
        return {
            "token_branch": False,
            "clock_after_branch": False,
            "clocks_in_both_arms": False,
            "encoded_true_arm": False,
            "no_inline_asm": "asm" not in body,
        }

    branch_index, true_label, false_label = branch
    clock_after_branch = any(
        _CLOCK64_INTRINSIC in line and re.search(r"\bcall\b", line)
        for line in lines[branch_index + 1 :]
    )
    blocks = _basic_blocks(body)
    true_body = "\n".join(blocks.get(true_label, []))
    false_body = "\n".join(blocks.get(false_label, []))
    true_clock_values = {
        match.group(1)
        for match in re.finditer(
            rf"(?m)^\s*(%[-a-zA-Z$._0-9]+)\s*=\s*[^\n]*\bcall\b[^\n]*"
            rf"{re.escape(_CLOCK64_INTRINSIC)}\s*\(",
            true_body,
        )
    }
    false_has_clock = bool(
        re.search(
            rf"(?m)\bcall\b[^\n]*{re.escape(_CLOCK64_INTRINSIC)}\s*\(",
            false_body,
        )
    )
    encoded_true_arm = any(
        value in set(_LLVM_VALUE_RE.findall(line))
        and re.search(r"\badd\b[^\n]*\bi64\b[^\n]*(?:,\s*1\b|\b1\s*,)", line)
        for value in true_clock_values
        for line in true_body.splitlines()
    )
    return {
        "token_branch": True,
        "clock_after_branch": clock_after_branch,
        "clocks_in_both_arms": bool(true_clock_values) and false_has_clock,
        "encoded_true_arm": encoded_true_arm,
        "no_inline_asm": "asm" not in body,
    }


def validate_clock_helper_ir(ir_text: str) -> dict[str, bool]:
    """Validate each clock helper's body, data dependency, and attributes."""
    start = _function_definition(ir_text, "corex_clock64_start")
    dependent_end = _function_definition(ir_text, "corex_clock64_after_u64")
    groups = _attribute_groups(ir_text)
    control = (
        _end_control_checks(dependent_end[1], dependent_end[2])
        if dependent_end
        else {}
    )

    checks = {
        "start_symbol": start is not None,
        "dependent_end_symbol": dependent_end is not None,
        "start_clock64_intrinsic": bool(start and _body_calls_clock64(start[2])),
        "end_clock64_intrinsic": bool(
            dependent_end and _body_calls_clock64(dependent_end[2])
        ),
        "end_token_conditional_branch": bool(control.get("token_branch")),
        "end_clock_after_branch": bool(control.get("clock_after_branch")),
        "end_clocks_in_both_arms": bool(control.get("clocks_in_both_arms")),
        "end_encoded_true_arm": bool(control.get("encoded_true_arm")),
        "end_no_inline_asm": bool(control.get("no_inline_asm")),
        "start_alwaysinline": bool(
            start and _definition_has_attribute(start[0], groups, "alwaysinline")
        ),
        "end_alwaysinline": bool(
            dependent_end
            and _definition_has_attribute(dependent_end[0], groups, "alwaysinline")
        ),
        "target_triple": bool(
            re.search(
                rf'(?m)^\s*target\s+triple\s*=\s*"{re.escape(TARGET_TRIPLE)}"\s*$',
                ir_text,
            )
        ),
    }
    missing = [name for name, available in checks.items() if not available]
    if missing:
        raise ClockHelperBuildError(
            "clock helper IR validation failed: " + ", ".join(missing)
        )
    return checks


def _run_command(runner: Runner, command: Sequence[str]) -> Any:
    completed = runner(
        list(command),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed").strip()
        raise ClockHelperBuildError(
            f"CoreX command failed ({completed.returncode}): {detail}"
        )
    return completed


def build_corex_clock(
    corex_root: Path,
    source: Path,
    output_dir: Path,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Build validated helper bitcode/text IR and publish one metadata file."""
    root = Path(corex_root).resolve()
    source_path = Path(source).resolve()
    destination = Path(output_dir).resolve()
    if not source_path.is_file():
        raise ClockHelperBuildError(f"clock helper source missing: {source_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    compiler_path = root / "bin" / "clang++"
    version_result = _run_command(runner, [str(compiler_path), "--version"])
    compiler_version = (version_result.stdout or version_result.stderr or "").strip()

    with tempfile.TemporaryDirectory(
        prefix=".corex-clock-build-", dir=destination.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        temporary_bitcode = temporary / BITCODE_NAME
        temporary_ir = temporary / IR_NAME
        bitcode_command = clock_compile_command(root, source_path, temporary_bitcode)
        ir_command = clock_compile_command(root, source_path, temporary_ir)

        _run_command(runner, bitcode_command)
        _run_command(runner, ir_command)

        if not temporary_bitcode.is_file() or temporary_bitcode.stat().st_size == 0:
            raise ClockHelperBuildError("CoreX compiler produced empty clock bitcode")
        if not temporary_ir.is_file() or temporary_ir.stat().st_size == 0:
            raise ClockHelperBuildError("CoreX compiler produced empty clock text IR")

        ir_text = temporary_ir.read_text(encoding="utf-8")
        ir_checks = validate_clock_helper_ir(ir_text)
        metadata: dict[str, Any] = {
            "document_type": "corex-clock-helper",
            "status": "valid",
            "corex_root": str(root),
            "target": TARGET,
            "target_triple": TARGET_TRIPLE,
            "compiler": {
                "path": str(compiler_path),
                "version": compiler_version,
            },
            "commands": {
                "bitcode": bitcode_command,
                "text_ir": ir_command,
            },
            "source_sha256": sha256_file(source_path),
            "bitcode_sha256": sha256_file(temporary_bitcode),
            "bitcode_path": BITCODE_NAME,
            "bitcode_absolute_path": str(destination / BITCODE_NAME),
            "ir_path": IR_NAME,
            "ir_checks": ir_checks,
        }

        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary_bitcode, destination / BITCODE_NAME)
        shutil.copyfile(temporary_ir, destination / IR_NAME)
        metadata_path = destination / METADATA_NAME
        temporary_metadata = destination / f".{METADATA_NAME}.tmp"
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
        return metadata
