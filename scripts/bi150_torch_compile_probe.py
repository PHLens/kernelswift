import json
import sys

import torch


def add_one(value):
    return value + 1


def main() -> int:
    result = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "has_torch_compile": hasattr(torch, "compile"),
    }
    if not result["has_torch_compile"]:
        result["ok"] = False
        print(json.dumps(result, sort_keys=True))
        return 1

    compiled = torch.compile(add_one)
    value = torch.zeros(16, device="cuda", dtype=torch.float32)
    output = compiled(value)
    torch.cuda.synchronize()
    result["max_error"] = float((output - 1).abs().max().item())
    result["ok"] = result["max_error"] == 0.0
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
