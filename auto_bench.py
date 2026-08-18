import argparse
import ast
import importlib.util
import statistics
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

import torch


class KsCompareError(Exception):
    pass


@dataclass
class CaseResult:
    name: str
    passed: bool
    v0_ms: float | None = None
    v1_ms: float | None = None
    speedup: float | None = None
    message: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare KS competition v0/v1 Python files. The v0 file must define "
            "Model/get_init_inputs/get_inputs, and the v1 file must define "
            "ModelNew/get_init_inputs/get_inputs. All tensors and models must be on the same device! "
            "For example: python benchmarks/ks/auto_bench.py --v0_file dlblas/kernels/ks_competition/torch/layer_norm.py "
            "--v1_file dlblas/kernels/ks_competition/triton/layer_norm.py "
        )
    )
    parser.add_argument("--v0_file", type=Path, help="Path to the v0 .py file.")
    parser.add_argument("--v1_file", type=Path, help="Path to the v1 .py file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Export a profiler trace after the accuracy and timing comparison.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Optional profiler trace path; also enables --profile.",
    )
    parser.add_argument(
        "--profile-reference-file",
        type=Path,
        help="Optional v1 file to profile beside --v1_file in the same trace.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=("kernel", "forward"),
        default="kernel",
        help=(
            "Profile preallocated ModelNew.run_out calls or complete forward calls. "
            "Kernel mode follows the grouped-topk v1 interface."
        ),
    )
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=100)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed case.",
    )
    parser.add_argument(
        "--full-traceback",
        action="store_true",
        help="Print full Python traceback for load/run failures.",
    )
    return parser.parse_args()


def _is_safe_literal(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_safe_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _is_safe_literal(key)) and _is_safe_literal(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_literal(node.operand)
    return False


def _filter_module_ast(tree):
    kept_nodes = []
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            kept_nodes.append(node)
        elif isinstance(node, ast.Assign) and _is_safe_literal(node.value):
            kept_nodes.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_safe_literal(node.value)
        ):
            kept_nodes.append(node)
    tree.body = kept_nodes
    ast.fix_missing_locations(tree)
    return tree


def _auto_accel_name() -> str | None:
    """Name of the first available accelerator (cuda/npu/mlu), or None."""
    for name, _ in _iter_accelerators():
        return name
    return None


class _RewriteDeviceStr(ast.NodeTransformer):
    """Rewrite device string literals in ks source so a file written for one
    backend runs on whatever accelerator is available here.

    Bare string constants equal to the source device name (e.g. 'npu') are
    rewritten to the detected target. In ks files these only appear as
    `device='npu'` / `.to('npu')`, so this is a safe, targeted swap.
    """

    def __init__(self, src: str, dst: str):
        self.src = src
        self.dst = dst

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value == self.src:
            return ast.copy_location(ast.Constant(value=self.dst), node)
        return node


def _rewrite_device_for_backend(tree: ast.AST) -> None:
    """In-place: remap device string literals to the available accelerator.

    ks competition files are written against a specific backend ('npu' or the
    device-neutral 'cuda' placeholder); on another backend the literal is
    rejected by torch at runtime, so rewrite it before exec. The 'cuda'
    placeholder is the shared cross-backend reference convention and must be
    remapped to whatever accelerator is actually present; 'npu' is remapped on
    non-Ascend hosts. No-op when no accelerator is present.
    """
    target = _auto_accel_name()
    if target is None:
        return
    if target != "npu":
        _RewriteDeviceStr("npu", target).visit(tree)
    if target != "cuda":
        _RewriteDeviceStr("cuda", target).visit(tree)
    ast.fix_missing_locations(tree)


def load_ks_module(path: Path) -> types.ModuleType:
    if not path.exists():
        raise KsCompareError(f"file does not exist: {path}")
    if path.suffix != ".py":
        raise KsCompareError(f"expected a .py file, got: {path}")

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text()
    except OSError as exc:
        raise KsCompareError(f"failed to read {path}: {exc}") from exc

    try:
        tree = ast.parse(source, filename=str(path))
        _rewrite_device_for_backend(tree)
    except SyntaxError as exc:
        raise KsCompareError(f"syntax error in {path}:{exc.lineno}: {exc.msg}") from exc

    module_name = f"_ks_compare_{path.stem}_{abs(hash(path.resolve()))}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
    sys.modules[module_name] = module
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        code = compile(_filter_module_ast(tree), filename=str(path), mode="exec")
        exec(code, module.__dict__)
    except Exception as exc:
        raise KsCompareError(f"failed to load definitions from {path}: {exc}") from exc
    finally:
        sys.path[:] = old_sys_path
        sys.modules.pop(module_name, None)
    return module


def require_attr(module, attr_name, path: Path):
    if not hasattr(module, attr_name):
        raise KsCompareError(f"{path} must define `{attr_name}`.")
    return getattr(module, attr_name)


def call_with_context(func, description):
    try:
        return func()
    except Exception as exc:
        raise KsCompareError(f"{description} failed: {exc}") from exc


def as_args(value, description):
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise KsCompareError(
        f"{description} must return a list or tuple, got {type(value).__name__}."
    )


def _iter_accelerators():
    """Yield (name, module) for each available accelerator backend.

    Covers cuda / npu (Ascend) / mlu (Cambricon) / gcu (Enflame).
    Add more backends here asneeded;
    set_seed / sync_devices / device detection all derive from this.
    """
    for name in ("gcu", "cuda", "npu", "mlu"):
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            if mod.is_available():
                yield name, mod
        except Exception:
            continue


def set_seed(seed):
    torch.manual_seed(seed)
    for _name, mod in _iter_accelerators():
        try:
            mod.manual_seed_all(seed)
        except Exception:
            pass


def sync_devices():
    for _name, mod in _iter_accelerators():
        try:
            mod.synchronize()
        except Exception:
            pass


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def describe_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )
    if isinstance(value, (list, tuple)):
        inner = ", ".join(describe_value(item) for item in value)
        return f"{type(value).__name__}({inner})"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{key}: {describe_value(item)}" for key, item in value.items()
        )
        return f"dict({inner})"
    return repr(value)


def compare_values(v0, v1, path, atol, rtol):
    if isinstance(v0, torch.Tensor) or isinstance(v1, torch.Tensor):
        if not isinstance(v0, torch.Tensor) or not isinstance(v1, torch.Tensor):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if v0.shape != v1.shape:
            raise KsCompareError(
                f"{path}: tensor shape mismatch: {v0.shape} vs {v1.shape}"
            )
        if (
            v0.dtype.is_floating_point
            or v1.dtype.is_floating_point
            or v0.is_complex()
            or v1.is_complex()
        ):
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True):
                if lhs.is_complex() or rhs.is_complex():
                    diff = (lhs - rhs).abs()
                else:
                    diff = (lhs.float() - rhs.float()).abs()
                if diff.numel() == 0:
                    diff_summary = "empty tensor"
                else:
                    diff_summary = (
                        f"max_abs_diff={diff.max().item():.6e}, "
                        f"mean_abs_diff={diff.mean().item():.6e}"
                    )
                raise KsCompareError(
                    f"{path}: tensor values differ; {diff_summary}, atol={atol}, rtol={rtol}, "
                    f"v0={describe_value(v0)}, v1={describe_value(v1)}"
                )
        else:
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            if not torch.equal(lhs, rhs):
                mismatch = (lhs != rhs).sum().item()
                raise KsCompareError(
                    f"{path}: tensor values differ; mismatched_elements={mismatch}, "
                    f"v0={describe_value(v0)}, v1={describe_value(v1)}"
                )
        return

    if isinstance(v0, tuple) or isinstance(v1, tuple):
        if not isinstance(v0, tuple) or not isinstance(v1, tuple):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if len(v0) != len(v1):
            raise KsCompareError(
                f"{path}: tuple length mismatch: {len(v0)} vs {len(v1)}"
            )
        for i, (item0, item1) in enumerate(zip(v0, v1)):
            compare_values(item0, item1, f"{path}[{i}]", atol, rtol)
        return

    if isinstance(v0, list) or isinstance(v1, list):
        if not isinstance(v0, list) or not isinstance(v1, list):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if len(v0) != len(v1):
            raise KsCompareError(
                f"{path}: list length mismatch: {len(v0)} vs {len(v1)}"
            )
        for i, (item0, item1) in enumerate(zip(v0, v1)):
            compare_values(item0, item1, f"{path}[{i}]", atol, rtol)
        return

    if isinstance(v0, dict) or isinstance(v1, dict):
        if not isinstance(v0, dict) or not isinstance(v1, dict):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if set(v0) != set(v1):
            raise KsCompareError(
                f"{path}: dict keys mismatch: {sorted(v0)} vs {sorted(v1)}"
            )
        for key in sorted(v0):
            compare_values(v0[key], v1[key], f"{path}[{key!r}]", atol, rtol)
        return

    if v0 != v1:
        raise KsCompareError(f"{path}: values differ: {v0!r} vs {v1!r}")


def build_case(v0_path: Path, v1_path: Path, seed: int):
    v0_module = load_ks_module(v0_path)
    v1_module = load_ks_module(v1_path)

    model_cls = require_attr(v0_module, "Model", v0_path)
    model_new_cls = require_attr(v1_module, "ModelNew", v1_path)
    v0_get_init_inputs = require_attr(v0_module, "get_init_inputs", v0_path)
    v1_get_init_inputs = require_attr(v1_module, "get_init_inputs", v1_path)
    v0_get_inputs = require_attr(v0_module, "get_inputs", v0_path)
    v1_get_inputs = require_attr(v1_module, "get_inputs", v1_path)

    for func, name, path in (
        (v0_get_init_inputs, "get_init_inputs", v0_path),
        (v1_get_init_inputs, "get_init_inputs", v1_path),
        (v0_get_inputs, "get_inputs", v0_path),
        (v1_get_inputs, "get_inputs", v1_path),
    ):
        if not callable(func):
            raise KsCompareError(f"{path}: `{name}` must be callable.")

    set_seed(seed)
    v0_init_args = as_args(
        call_with_context(v0_get_init_inputs, f"{v0_path}: get_init_inputs()"),
        f"{v0_path}: get_init_inputs()",
    )
    set_seed(seed)
    v1_init_args = as_args(
        call_with_context(v1_get_init_inputs, f"{v1_path}: get_init_inputs()"),
        f"{v1_path}: get_init_inputs()",
    )

    model = call_with_context(
        lambda: model_cls(*v0_init_args), f"{v0_path}: Model(...)"
    )
    model_new = call_with_context(
        lambda: model_new_cls(*v1_init_args), f"{v1_path}: ModelNew(...)"
    )
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model_new, "eval"):
        model_new.eval()

    set_seed(seed)
    v0_inputs = as_args(
        call_with_context(v0_get_inputs, f"{v0_path}: get_inputs()"),
        f"{v0_path}: get_inputs()",
    )
    set_seed(seed)
    v1_inputs = as_args(
        call_with_context(v1_get_inputs, f"{v1_path}: get_inputs()"),
        f"{v1_path}: get_inputs()",
    )

    if len(v0_inputs) != len(v1_inputs):
        raise KsCompareError(
            f"get_inputs argument count mismatch: {v0_path} returns {len(v0_inputs)} "
            f"args, {v1_path} returns {len(v1_inputs)} args."
        )
    return model, model_new, v0_inputs, v1_inputs


def run_forward(model, inputs, seed, description):
    set_seed(seed)
    cloned_inputs = clone_value(inputs)
    try:
        with torch.no_grad():
            return model.forward(*cloned_inputs)
    except Exception as exc:
        raise KsCompareError(f"{description} forward failed: {exc}") from exc


def time_forward(model, inputs, seed, warmup, repeat):
    def one_call():
        with torch.no_grad():
            model.forward(*inputs)

    for _ in range(warmup):
        one_call()
    sync_devices()

    samples = []
    for _ in range(repeat):
        set_seed(seed)
        start = time.perf_counter()
        one_call()
        sync_devices()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def build_profile_reference(
    path,
    source_model,
    source_inputs,
    source_output,
    target_device,
    seed,
    atol,
    rtol,
):
    module = load_ks_module(path)
    model_cls = require_attr(module, "ModelNew", path)
    get_init_inputs = require_attr(module, "get_init_inputs", path)
    require_attr(module, "get_inputs", path)

    set_seed(seed)
    init_args = as_args(
        call_with_context(get_init_inputs, f"{path}: get_init_inputs()"),
        f"{path}: get_init_inputs()",
    )
    model = call_with_context(
        lambda: model_cls(*init_args), f"{path}: ModelNew(...)"
    )
    if hasattr(model, "eval"):
        model.eval()
    try:
        model.load_state_dict(source_model.state_dict())
    except Exception:
        pass
    if hasattr(model, "to"):
        model = model.to(target_device)

    inputs = clone_value(source_inputs)
    output = run_forward(model, inputs, seed, f"profile reference {path.stem}")
    compare_values(source_output, output, f"profile reference {path.stem}", atol, rtol)
    return model, inputs, output


def make_profile_call(model, inputs, output, mode, description):
    if mode == "forward":
        return lambda: model.forward(*inputs)

    run_out = getattr(model, "run_out", None)
    if not callable(run_out):
        raise KsCompareError(
            f"{description}: kernel profiling requires a callable ModelNew.run_out"
        )
    if not inputs or not isinstance(inputs[-1], torch.Tensor):
        raise KsCompareError(
            f"{description}: kernel profiling requires the final input to be a tensor"
        )
    output_args = output if isinstance(output, (tuple, list)) else (output,)
    if not output_args or not all(isinstance(value, torch.Tensor) for value in output_args):
        raise KsCompareError(
            f"{description}: kernel profiling requires tensor outputs"
        )
    run_kwargs = dict(getattr(model, "run_kwargs", {}))
    gating_output = inputs[-1]
    return lambda: run_out(gating_output, *output_args, **run_kwargs)


def profiler_activities():
    activities = [torch.profiler.ProfilerActivity.CPU]
    accelerator_names = {name for name, _ in _iter_accelerators()}
    if "cuda" in accelerator_names and hasattr(torch.profiler.ProfilerActivity, "CUDA"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif accelerator_names and hasattr(torch.profiler.ProfilerActivity, "PrivateUse1"):
        activities.append(torch.profiler.ProfilerActivity.PrivateUse1)
    return activities


def _is_npu_backend() -> bool:
    """True when the available accelerator is Ascend NPU."""
    return any(name == "npu" for name, _ in _iter_accelerators())


def _export_profile_npu(targets, args):
    """Profile on Ascend NPU with torch_npu.profiler + CANN msprof.

    The stock torch.profiler path exposes only host-side cpu_op events on
    Ascend; NPU AI Core kernel durations live in the CANN msprof sqlite output
    (device_0/sqlite/ai_core_op_summary.db). torch_npu.profiler triggers that
    capture, and its output directory is controlled by the ASCEND_WORK_PATH
    environment variable.

    The CANN sqlite accumulates tasks without any reference/candidate scope
    field, and its start_time clock is unrelated to the chrome trace ts clock,
    so a single combined capture cannot attribute kernels per scope. We
    therefore profile each target (reference, candidate) in its own capture,
    each writing to a distinct ASCEND_WORK_PATH subdirectory, so every
    ai_core_op_summary.db belongs to exactly one scope.
    """
    import os

    try:
        import torch_npu  # noqa: F401
        from torch_npu.profiler import (
            ProfilerActivity as NpuActivity,
            experimental_config as npu_ec,
            profile as npu_profile,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        raise KsCompareError(f"npu backend selected but torch_npu unavailable: {exc}") from exc

    profile_parent = args.profile_output.parent
    profile_parent.mkdir(parents=True, exist_ok=True)
    previous_work_path = os.environ.get("ASCEND_WORK_PATH")

    cann_dirs = {}
    for label, model, inputs, output in targets:
        call = make_profile_call(model, inputs, output, args.profile_mode, label)
        # One CANN capture per scope.
        scope_work = str((profile_parent / "profiling_data" / label).resolve())
        os.environ["ASCEND_WORK_PATH"] = scope_work
        prof = npu_profile(
            activities=[NpuActivity.CPU, NpuActivity.NPU],
            experimental_config=npu_ec._ExperimentalConfig(profiler_level="Level0"),
        )
        try:
            with torch.no_grad():
                for _ in range(args.profile_warmup):
                    call()
                sync_devices()
                with prof:
                    with torch.profiler.record_function(label):
                        for _ in range(args.profile_iterations):
                            call()
                    sync_devices()
        finally:
            if previous_work_path is None:
                os.environ.pop("ASCEND_WORK_PATH", None)
            else:
                os.environ["ASCEND_WORK_PATH"] = previous_work_path
        prof.export_chrome_trace(str(args.profile_output))
        cann_dirs[label] = Path(scope_work) / "profiling_data"

    for label, cann_dir in cann_dirs.items():
        print(f"cann_profiling_data[{label}]={cann_dir}")
    print(f"profile={args.profile_output}")


def export_profile(targets, args):
    if _is_npu_backend():
        _export_profile_npu(targets, args)
        return

    calls = [
        (
            label,
            make_profile_call(model, inputs, output, args.profile_mode, label),
        )
        for label, model, inputs, output in targets
    ]

    with torch.no_grad():
        for _, call in calls:
            for _ in range(args.profile_warmup):
                call()
        sync_devices()

        with torch.profiler.profile(activities=profiler_activities()) as prof:
            for label, call in calls:
                with torch.profiler.record_function(label):
                    for _ in range(args.profile_iterations):
                        call()
            sync_devices()

    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.profile_output))
    print(f"profile={args.profile_output}")


def _get_model_device(model):
    """Return the device of *model*'s first parameter or buffer, or None."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        pass
    return None


def _first_input_device(inputs):
    """Return the device of the first tensor found in nested *inputs*, or None."""
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            d = _first_input_device(item)
            if d is not None:
                return d
    if isinstance(inputs, dict):
        for item in inputs.values():
            d = _first_input_device(item)
            if d is not None:
                return d
    return None


def _detect_target_device(model, model_new, v0_inputs, v1_inputs):
    """Pick a non-CPU device from models/inputs, or auto-detect one.

    Priority: model device > input device > auto-detect (cuda → npu).
    Raises KsCompareError if no accelerator is available.
    """
    for m in (model, model_new):
        d = _get_model_device(m)
        if d is not None and d.type != "cpu":
            return d
    for inputs in (v0_inputs, v1_inputs):
        d = _first_input_device(inputs)
        if d is not None and d.type != "cpu":
            return d
    for name, _ in _iter_accelerators():
        return torch.device(name)
    raise KsCompareError(
        "no accelerator device available (cuda/npu/mlu); "
        "cannot run accuracy or performance comparison on CPU."
    )


def _move_to_device(value, device):
    """Recursively copy every tensor in *value* to *device*."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def compare_case(name, v0_path, v1_path, args):
    model, model_new, v0_inputs, v1_inputs = build_case(v0_path, v1_path, args.seed)

    target_device = _detect_target_device(model, model_new, v0_inputs, v1_inputs)

    try:
        model_new.load_state_dict(model.state_dict())
    except Exception:
        pass

    if hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model_new, "to"):
        model_new = model_new.to(target_device)

    v0_inputs = _move_to_device(v0_inputs, target_device)
    v1_inputs = _move_to_device(v1_inputs, target_device)
    v1_inputs = clone_value(v0_inputs)

    v0_output = run_forward(model, v0_inputs, args.seed, f"{name}: v0")
    v1_output = run_forward(model_new, v1_inputs, args.seed, f"{name}: v1")
    compare_values(v0_output, v1_output, "output", args.atol, args.rtol)

    v0_ms = time_forward(model, v0_inputs, args.seed, args.warmup, args.repeat)
    v1_ms = time_forward(model_new, v1_inputs, args.seed, args.warmup, args.repeat)
    speedup = v0_ms / v1_ms if v1_ms > 0 else float("inf")

    if args.profile:
        candidate = (f"candidate_{v1_path.stem}", model_new, v1_inputs, v1_output)
        if args.profile_reference_file is not None:
            reference_model, reference_inputs, reference_output = build_profile_reference(
                args.profile_reference_file,
                model,
                v0_inputs,
                v0_output,
                target_device,
                args.seed,
                args.atol,
                args.rtol,
            )
            targets = [
                (
                    f"reference_{args.profile_reference_file.stem}",
                    reference_model,
                    reference_inputs,
                    reference_output,
                ),
                candidate,
            ]
        elif args.profile_mode == "forward":
            targets = [
                (f"baseline_{v0_path.stem}", model, v0_inputs, v0_output),
                candidate,
            ]
        else:
            targets = [candidate]
        export_profile(targets, args)

    return CaseResult(name=name, passed=True, v0_ms=v0_ms, v1_ms=v1_ms, speedup=speedup)


def main():
    args = parse_args()
    v0_path = args.v0_file.resolve()
    v1_path = args.v1_file.resolve()
    if not v0_path.is_file():
        raise SystemExit(f"v0_file is not a file: {v0_path}")
    if not v1_path.is_file():
        raise SystemExit(f"v1_file is not a file: {v1_path}")
    if v0_path.suffix != ".py":
        raise SystemExit(f"v0_file must be a .py file: {v0_path}")
    if v1_path.suffix != ".py":
        raise SystemExit(f"v1_file must be a .py file: {v1_path}")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0.")
    if args.profile_warmup < 0 or args.profile_iterations <= 0:
        raise SystemExit(
            "--profile-warmup must be >= 0 and --profile-iterations must be > 0."
        )
    if args.profile_output is not None:
        args.profile = True
    if args.profile_reference_file is not None and not args.profile:
        raise SystemExit("--profile-reference-file requires --profile.")
    if args.profile_reference_file is not None:
        args.profile_reference_file = args.profile_reference_file.resolve()
        if not args.profile_reference_file.is_file():
            raise SystemExit(
                f"profile reference file is not a file: {args.profile_reference_file}"
            )
        if args.profile_reference_file.suffix != ".py":
            raise SystemExit(
                f"profile reference file must be a .py file: {args.profile_reference_file}"
            )
    if args.profile and args.profile_output is None:
        args.profile_output = Path(
            "log"
        ) / f"{v1_path.stem}_{args.profile_mode}_{args.profile_iterations}iter.pt.trace.json"

    name = str(v0_path)

    try:
        result = compare_case(name, v0_path, v1_path, args)
        print(
            f"PASS accuracy; v0={result.v0_ms:.6f} ms, "
            f"v1={result.v1_ms:.6f} ms, speedup={result.speedup:.3f}x"
        )
        passed = 1
        failed = 0
    except Exception as exc:
        if args.full_traceback:
            traceback.print_exc()
        message = str(exc)
        result = CaseResult(name=name, passed=False, message=message)
        print(f"FAIL {message}")
        passed = 0
        failed = 1

    print(f"\nSummary: {passed} passed, {failed} failed, 1 total.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
