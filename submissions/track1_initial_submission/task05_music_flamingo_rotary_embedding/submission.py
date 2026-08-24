import importlib.util
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from base import get_init_inputs as _base_get_init_inputs
from base import get_inputs as _base_get_inputs


_BACKEND_TO_FILE = {'s60': 'triton_rotary_002.py', 'maca': 'triton_rotary_001.py', 'bi150': 'triton_music_flamingo_rotary_embedding_001.py', 'ascend': 'triton_rotary_001.py'}


def _backend_available(name: str) -> bool:
    mod = getattr(torch, name, None)
    if mod is None:
        return False
    try:
        return bool(mod.is_available())
    except Exception:
        return False


def _detect_backend() -> str:
    override = os.environ.get('KS_BACKEND_OVERRIDE')
    if override:
        return override
    if _backend_available('mlu'):
        return 'mlu'
    if _backend_available('gcu'):
        return 's60'
    if _backend_available('npu'):
        return 'ascend'
    if _backend_available('cuda'):
        name = ''
        try:
            name = torch.cuda.get_device_name(0).lower()
        except Exception:
            name = ''
        if os.environ.get('COREX_VERSION') or any(token in name for token in ('iluvatar', 'bi-v150', 'corex', 'ixmma')):
            return 'bi150'
        return 'maca'
    raise RuntimeError('No supported accelerator backend detected for this submission entry.')


def _load_impl_module(backend: str):
    try:
        filename = _BACKEND_TO_FILE[backend]
    except KeyError as exc:
        supported = ', '.join(sorted(_BACKEND_TO_FILE))
        raise RuntimeError(f'Backend {backend!r} is not implemented for this task. Supported backends: {supported}') from exc
    path = Path(__file__).resolve().parent / 'impls' / filename
    spec = importlib.util.spec_from_file_location(f'_submission_impl_{backend}_{path.stem}', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load backend implementation from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelNew(nn.Module):
    def __init__(self, dim: int = 64, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        self.backend = _detect_backend()
        self._impl_module = _load_impl_module(self.backend)
        self.impl = self._impl_module.ModelNew(dim, max_seq_len, base)

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        return self.impl.forward(timestamps, seq_len)

    def load_state_dict(self, state_dict, strict: bool = True):
        if hasattr(self.impl, 'load_state_dict'):
            try:
                return self.impl.load_state_dict(state_dict, strict=False)
            except TypeError:
                try:
                    return self.impl.load_state_dict(state_dict)
                except Exception:
                    return None
            except Exception:
                return None
        return None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            impl = super().__getattr__('impl')
            return getattr(impl, name)


def get_init_inputs():
    return _base_get_init_inputs()


def get_inputs():
    return _base_get_inputs()
