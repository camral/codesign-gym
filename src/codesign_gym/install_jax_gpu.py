"""
Authors: Aviraj Newatia

Install the JAX CUDA backend supported by the visible NVIDIA GPUs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import shlex
import shutil
import subprocess
import sys


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    driver_major: int
    compute_capability: tuple[int, int]


def _query_gpus() -> list[GPU]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi was not found; an NVIDIA driver is required")
    command = [executable, "--query-gpu=index,uuid,driver_version,compute_cap", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip() or result.stdout.strip()}")
    devices = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
        try:
            capability = tuple(int(part) for part in parts[3].split("."))
            if len(capability) != 2:
                raise ValueError("compute capability needs major.minor")
            devices.append(GPU(int(parts[0]), parts[1], int(parts[2].split(".")[0]), capability))
        except ValueError as error:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}") from error
    if not devices:
        raise RuntimeError("nvidia-smi reported no NVIDIA GPUs")
    return devices


def _visible_gpus(devices: list[GPU], gpu_index: int | None = None) -> list[GPU]:
    if gpu_index is not None:
        selected = [device for device in devices if device.index == gpu_index]
        if not selected:
            raise RuntimeError(f"nvidia-smi did not report GPU index {gpu_index}")
        return selected
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return devices
    selected = []
    for token in visible.split(","):
        token = token.strip()
        matches = [device for device in devices if token in (str(device.index), device.uuid)]
        if not matches:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES contains unknown GPU {token!r}")
        selected.extend(matches)
    return selected


def _cuda_variant(devices: list[GPU]) -> str:
    if not devices:
        raise RuntimeError("no visible NVIDIA GPUs")
    if all(device.driver_major >= 580 and device.compute_capability >= (7, 5) for device in devices):
        return "cuda13"
    if all(device.driver_major >= 525 and device.compute_capability >= (5, 2) for device in devices):
        return "cuda12"
    raise RuntimeError("visible GPUs do not meet JAX's CUDA 12 minimums (driver >= 525, compute capability >= 5.2)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, help="Physical nvidia-smi GPU index; otherwise use CUDA_VISIBLE_DEVICES or all host GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected install command without running pip")
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("JAX CUDA pip wheels are available only on Linux")
    try:
        devices = _visible_gpus(_query_gpus(), args.gpu_index)
        variant = _cuda_variant(devices)
    except RuntimeError as error:
        parser.error(str(error))
    command = [sys.executable, "-m", "pip", "install", f"jax[{variant}]"]
    print(f"Selected {variant} for NVIDIA GPU(s) {', '.join(str(device.index) for device in devices)}")
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
