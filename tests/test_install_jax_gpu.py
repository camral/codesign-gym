"""Selection tests for the optional JAX GPU installer; no packages are installed."""

from types import SimpleNamespace

import pytest

from codesign_gym import install_jax_gpu as inst


def gpu(index, driver, capability):
    return inst.GPU(index, f"GPU-{index}", driver, capability)


def test_cuda_variant_selects_cuda13_when_every_gpu_supports_it():
    assert inst._cuda_variant([gpu(0, 580, (12, 0)), gpu(1, 590, (8, 0))]) == "cuda13"


def test_cuda_variant_uses_cuda12_for_older_driver_or_mixed_devices():
    assert inst._cuda_variant([gpu(0, 550, (8, 0))]) == "cuda12"
    assert inst._cuda_variant([gpu(0, 580, (8, 0)), gpu(1, 580, (6, 1))]) == "cuda12"


def test_cuda_variant_rejects_unsupported_host():
    with pytest.raises(RuntimeError, match="minimums"):
        inst._cuda_variant([gpu(0, 500, (8, 0))])


def test_visible_gpus_respects_cuda_visible_devices_and_index(monkeypatch):
    devices = [gpu(0, 580, (12, 0)), gpu(1, 580, (8, 0))]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert inst._visible_gpus(devices) == devices[1:]
    assert inst._visible_gpus(devices, 0) == devices[:1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0")
    assert inst._visible_gpus(devices) == devices[:1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="unknown GPU"):
        inst._visible_gpus(devices)


def test_query_gpus_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(inst.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(inst.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="0, GPU-a, 580.105.08, 12.0\n1, GPU-b, 550.22, 8.0\n", stderr=""))
    assert inst._query_gpus() == [inst.GPU(0, "GPU-a", 580, (12, 0)), inst.GPU(1, "GPU-b", 550, (8, 0))]


def test_dry_run_prints_command_without_installing(monkeypatch, capsys):
    monkeypatch.setattr(inst, "_query_gpus", lambda: [gpu(0, 580, (12, 0))])
    monkeypatch.setattr(inst.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not install"))
    monkeypatch.setattr(inst.sys, "platform", "linux")
    inst.main(["--gpu-index", "0", "--dry-run"])
    output = capsys.readouterr().out
    assert "Selected cuda13" in output and "jax[cuda13]" in output
