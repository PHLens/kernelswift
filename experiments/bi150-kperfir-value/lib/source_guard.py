"""Verify immutable BI150 experiment inputs before device work."""

from __future__ import annotations

import hashlib
from pathlib import Path

ACCEPTED_SOURCE_HASHES = {
    "kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_001.py": "4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2",
    "kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py": "cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078",
    "kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py": "d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81",
    "kernels/track1-triton/mm_encoder_attention/base.py": "86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2",
    "auto_bench.py": "71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29",
}


class SourceGuardError(RuntimeError):
    """Raised when an accepted source is missing or changed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_accepted_sources(repo_root: Path) -> dict[str, str]:
    """Return observed hashes after verifying every accepted input."""
    root = Path(repo_root).resolve()
    observed: dict[str, str] = {}
    for relative_path, expected_hash in ACCEPTED_SOURCE_HASHES.items():
        path = root / relative_path
        if not path.is_file():
            raise SourceGuardError(f"accepted source missing: {relative_path}")
        actual_hash = sha256_file(path)
        observed[relative_path] = actual_hash
        if actual_hash != expected_hash:
            raise SourceGuardError(
                f"accepted source hash mismatch: {relative_path}: "
                f"expected {expected_hash}, observed {actual_hash}"
            )
    return observed
