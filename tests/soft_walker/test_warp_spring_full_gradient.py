"""Full resolved spring gradients against an independent dense CPU solve."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def test_full_spring_gradients_against_dense_reference() -> None:
    if os.environ.get("WARP_TEST_DEVICE", "cpu") == "cpu":
        pytest.skip("The full native episode requires its CUDA graph")
    if os.environ.get("WARP_TEST_DTYPE", "float64") != "float64":
        pytest.skip("The frozen 1400-step regression is a float64 problem")
    import jax
    import warp as wp

    from envs.soft_walker.engine.execution.cupti import DriverAudit
    from envs.soft_walker.engine.physics.spring_config import SpringConfig
    from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
    from envs.soft_walker.engine.warp_native.solid_model import (
        INTEGER_FIELDS,
        STATE_FIELDS,
        SolidBundle,
        SolidDeviceModel,
    )

    root = Path(__file__).resolve().parents[2] / "src/envs/soft_walker"
    source = root / "baselines/warp_followup/spring_biped"
    spec = importlib.util.spec_from_file_location(
        "full_spring_reference", root / "diagnostics/warp_followup_spring.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    bundle = SolidBundle.load(source / "model.npz")
    commands = np.load(source / "commands.npy", allow_pickle=False)
    settings = json.loads((source / "benchmark.json").read_text())
    data = json.loads((source / "jax_spring_config.json").read_text())
    data["grid_shape"], data["size"] = tuple(data["grid_shape"]), tuple(data["size"])
    cfg = replace(SpringConfig(**data), linear_solver="dense", gmres_method="incremental")
    parameters = dict(
        commands=commands,
        model={k: v for k, v in bundle.fields.items() if k not in INTEGER_FIELDS + ("free",)},
        initial={k: v for k, v in bundle.initial.items() if k != "status"},
    )
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        parameters = jax.device_put(parameters, jax.devices("cpu")[0])
        objective = helper.jax_objective(bundle, settings, cfg)
        (expected_loss, expected_state), expected_grad = jax.device_get(
            jax.jit(jax.value_and_grad(objective, has_aux=True))(parameters)
        )
    owner = SolidDeviceModel(bundle, os.environ["WARP_TEST_DEVICE"])
    episode = SolidEpisode(
        owner,
        commands,
        substeps=settings["substeps"],
        settling_steps=settings["settling_steps"],
        checkpoint_stride=settings["checkpoint_stride"],
        memory_mib=settings["memory_mib"],
        length_ref=settings["length_ref"],
        direction=settings["direction"],
        work_weight=settings["work_weight"],
        velocity_weight=settings["velocity_weight"],
        reuse_replay_roots=True,
    )
    episode.run(True)
    stream = wp.get_stream(owner.device)
    wp.synchronize_stream(stream)
    with DriverAudit() as observer:
        wp.capture_launch(episode.graphs["gradient"], stream=stream)
        wp.synchronize_stream(stream)
    audit = observer.report()
    assert audit["passed"], audit
    assert int(episode.final.status.numpy()[0]) == int(expected_state["status"][0]) == 0
    assert int(episode.adjoint_status.numpy()[0]) == 0
    assert np.all(episode.diags.numpy()[:, 0] == 0)
    np.testing.assert_allclose(episode.loss.numpy(), expected_loss, rtol=2e-8, atol=2e-11)
    actual_state = {k: getattr(episode.final, k).numpy() for k in STATE_FIELDS}
    actual_state["travel"] = episode.travel.numpy()
    for name, expected in expected_state.items():
        np.testing.assert_allclose(
            actual_state[name], expected, rtol=2e-8, atol=2e-11, err_msg=name
        )
    actual = episode.gradients()
    np.testing.assert_allclose(actual["commands"], expected_grad["commands"], rtol=1e-5, atol=2e-11)
    for group in ("model", "initial"):
        for name, expected in expected_grad[group].items():
            assert actual[group][name].shape == expected.shape
            assert actual[group][name].dtype == expected.dtype
            np.testing.assert_allclose(
                actual[group][name], expected, rtol=1e-5, atol=2e-11, err_msg=f"{group}/{name}"
            )
