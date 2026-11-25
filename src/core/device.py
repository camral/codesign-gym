"""Device selection for environments sharing Torch and MuJoCo Warp buffers."""
from __future__ import annotations


def resolve_mujoco_device(device: str | None = None) -> str:
    """Prefer a CUDA device usable by both runtimes, otherwise use CPU.

    An explicit request is strict: an unavailable GPU is an error, not an
    instruction to silently change devices. Only omitted/None selects a default.
    """
    import torch
    import warp as wp

    if device is not None:
        device = str(device)
        if device == "cpu":
            return "cpu"
        if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError(f"Unsupported MuJoCo device {device!r}; use 'cpu', 'cuda', or 'cuda:N'")

    def available(index: int) -> bool:
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            return False
        try:
            return wp.get_device(f"cuda:{index}").is_cuda
        except (RuntimeError, ValueError):
            return False

    if device is None:
        if torch.cuda.is_available():
            preferred = torch.cuda.current_device()
            for index in dict.fromkeys([preferred, *range(torch.cuda.device_count())]):
                if available(index):
                    return f"cuda:{index}"
        return "cpu"

    if device == "cuda":
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    else:
        index = int(device[5:])
    if not available(index):
        raise RuntimeError(
            f"Requested MuJoCo device {device!r} is unavailable in Torch or Warp; "
            "pass device='cpu' or omit device for automatic selection"
        )
    return f"cuda:{index}"
