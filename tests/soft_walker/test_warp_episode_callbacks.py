"""Native callback AD, held actions, recurrent carry, replay, and CUDA residency."""

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from envs.soft_walker.engine.warp_native.episode import Episode
from envs.soft_walker.engine.warp_native.episode_callbacks import NativeObjective, NativePolicy
from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel


def _kernels(owner: "DeviceModel | SolidDeviceModel") -> SimpleNamespace:
    """Nonzero polynomial coefficients exercise each callback derivative route."""
    real, Model, State = owner.k.real, owner.k.Model, owner.k.State
    module = f"test_native_callbacks_{getattr(owner.config, 'physics', 'mpm')}_{owner.config.dim}_{owner.precision}"

    @wp.kernel(module=module)
    def policy(
        model: Model,
        state: State,
        parameters: wp.array(dtype=real),
        carry: wp.array(dtype=real),
        index: wp.array(dtype=int),
        command: wp.array(dtype=real),
        next_carry: wp.array(dtype=real),
    ):
        value = wp.tanh(parameters[0] + parameters[1] * state.v[0][0] + parameters[2] * carry[0])
        for channel in range(command.shape[0]):
            command[channel] = value
        next_carry[0] = real(0.8) * carry[0] + real(0.1) * state.v[0][0] + real(0.05) * value

    @wp.kernel(module=module)
    def running(
        model: Model,
        initial: State,
        previous: State,
        next_state: State,
        command: wp.array(dtype=real),
        parameters: wp.array(dtype=real),
        clock: wp.array(dtype=int),
        cost: wp.array(dtype=real),
    ):
        cost[0] = (
            parameters[0] * wp.dot(next_state.v[0], next_state.v[0]) * real(0.01)
            + real(0.2) * wp.dot(previous.v[0], previous.v[0])
            + parameters[2] * command[0] * command[0]
            + real(0.3) * initial.ledger[0]
        )

    @wp.kernel(module=module)
    def terminal(
        model: Model,
        initial: State,
        final: State,
        parameters: wp.array(dtype=real),
        accumulated: wp.array(dtype=real),
        loss: wp.array(dtype=real),
    ):
        loss[0] = (
            parameters[0] * wp.dot(final.v[0], final.v[0])
            + parameters[1] * accumulated[0]
            + parameters[2] * model.young[0] * real(0.0001)
        )

    return SimpleNamespace(policy=policy, running=running, terminal=terminal)


def _episode(kind: str, dtype: str, *, substeps: int = 2) -> "Episode | SolidEpisode":
    """Construct a tiny native problem with a held recurrent controller."""
    device = os.environ.get("WARP_TEST_DEVICE", "cpu")
    if kind.startswith("mpm"):
        bundle = uniform_body(int(kind[-1]), precision=dtype)
        bundle.config = replace(bundle.config, contact_enabled=False)
        owner = DeviceModel(bundle, device)
        constructor = Episode
    else:
        physics = "spring" if kind == "spring_kvaerno" else kind
        bundle = regular_solid(
            physics, precision=dtype, shape=(2, 1) if physics == "spring" else (2, 1, 1)
        )
        bundle.config = replace(
            bundle.config,
            contact_enabled=False,
            integrator="kvaerno3_native" if kind == "spring_kvaerno" else "backward_euler",
        )
        owner = SolidDeviceModel(bundle, device)
        constructor = SolidEpisode
    kernels = _kernels(owner)
    return constructor(
        owner,
        np.zeros((3, owner.channels), dtype),
        substeps=substeps,
        settling_steps=1,
        checkpoint_stride=3,
        policy=NativePolicy(
            kernels.policy, np.array([0.2, -0.3, 0.1], dtype), np.array([0.07], dtype)
        ),
        objective=NativeObjective(
            kernels.terminal, np.array([0.9, 0.4, 0.2], dtype), kernels.running
        ),
    )


@pytest.mark.parametrize("kind", ("spring", "beam", "fem", "mpm2", "mpm3", "spring_kvaerno"))
def test_resident_callbacks_and_gradients(kind: str, dtype: np.dtype) -> None:
    """Check policy, memory, cost, physical state and parameter derivatives by FD."""
    episode = _episode(kind, dtype.name)
    result = episode.run(gradient=True)
    assert result["status"] == result["adjoint_status"] == 0
    grad = episode.gradients()
    callbacks = episode.callbacks
    assert callbacks is not None
    assert np.ptp(episode.commands.numpy()[:, 0]) > 1e-6
    epsilon = 0.01 if dtype.name == "float32" else 1e-4
    tests = [
        (callbacks.policy_parameters, grad["policy_parameters"], (1,)),
        (callbacks.initial_memory, grad["policy_initial_carry"], (0,)),
        (callbacks.objective_parameters, grad["objective_parameters"], (1,)),
        (episode.owner.initial.v, grad["initial"]["v"], (0, 0)),
        (episode.owner.initial.ledger, grad["initial"]["ledger"], (0,)),
    ]
    for array, expected, index in tests:
        original = array.numpy().copy()
        values = []
        for sign in (1, -1):
            value = original.copy()
            value[index] += sign * epsilon
            array.assign(value)
            report = episode.run()
            assert report["status"] == 0
            values.append(report["loss"])
        array.assign(original)
        observed = (values[0] - values[1]) / (2 * epsilon)
        np.testing.assert_allclose(
            observed,
            expected[index],
            rtol=0.025 if dtype.name == "float32" else 0.002,
            atol=3e-4 if dtype.name == "float32" else 2e-7,
            err_msg=f"{kind}: {array.shape}/{index}",
        )
    # Saved actions reproduce the physical trajectory under the ordinary table API.
    episode.run()
    table = episode.commands.numpy().copy()
    reference = type(episode)(
        episode.owner, table, substeps=episode.substeps, settling_steps=episode.settling
    )
    assert reference.run()["status"] == 0
    np.testing.assert_allclose(
        reference.final.v.numpy(), episode.final.v.numpy(), rtol=3e-5, atol=2e-7
    )
    if episode.device.is_cuda:
        from envs.soft_walker.engine.execution.cupti import DriverAudit

        episode.compile(gradient=True)
        wp.synchronize_device(episode.device)
        stream = wp.get_stream(episode.device)
        with DriverAudit() as observer:
            wp.capture_launch(episode.graphs["gradient"], stream=stream)
            wp.synchronize_stream(stream)
        assert observer.report()["passed"], observer.report()


def test_callback_updates_are_shape_checked(dtype: np.dtype) -> None:
    """Input updates cannot invalidate captured buffer ownership or parameter shape."""
    episode = _episode("spring", dtype.name)
    with pytest.raises(ValueError, match="shape"):
        episode.set_callback_parameters(policy=np.zeros(2, dtype))
    episode.set_callback_parameters(
        policy=np.array([0.1, 0.2, 0.3], dtype), initial_carry=np.array([0.0], dtype)
    )
    assert episode.run()["status"] == 0


def _independent_kernels(owner: "SolidDeviceModel") -> SimpleNamespace:
    """Separate callback branches include direct mass/state seeds and invalid outputs."""
    real, Model, State = owner.k.real, owner.k.Model, owner.k.State
    module = f"test_independent_native_callbacks_{owner.precision}"
    not_finite = wp.constant(real(float("nan")))

    @wp.kernel(module=module)
    def stateless(
        model: Model,
        state: State,
        parameters: wp.array(dtype=real),
        carry: wp.array(dtype=real),
        index: wp.array(dtype=int),
        command: wp.array(dtype=real),
        next_carry: wp.array(dtype=real),
    ):
        for channel in range(command.shape[0]):
            command[channel] = wp.tanh(parameters[0] + real(0.2) * state.v[0][0])

    @wp.kernel(module=module)
    def running(
        model: Model,
        initial: State,
        previous: State,
        next_state: State,
        command: wp.array(dtype=real),
        parameters: wp.array(dtype=real),
        clock: wp.array(dtype=int),
        cost: wp.array(dtype=real),
    ):
        cost[0] = (
            real(0.7) * model.mass[0]
            + real(0.3) * previous.v[0][0]
            + real(0.2) * initial.v[0][0]
            + real(0.1) * next_state.v[0][0]
            + parameters[0] * command[0] * command[0]
        )

    @wp.kernel(module=module)
    def terminal(
        model: Model,
        initial: State,
        final: State,
        parameters: wp.array(dtype=real),
        accumulated: wp.array(dtype=real),
        loss: wp.array(dtype=real),
    ):
        loss[0] = accumulated[0] + real(0.2) * final.v[0][0]

    @wp.kernel(module=module)
    def invalid_policy(
        model: Model,
        state: State,
        parameters: wp.array(dtype=real),
        carry: wp.array(dtype=real),
        index: wp.array(dtype=int),
        command: wp.array(dtype=real),
        next_carry: wp.array(dtype=real),
    ):
        for channel in range(command.shape[0]):
            command[channel] = parameters[0]
        if next_carry.shape[0] > 0:
            next_carry[0] = not_finite

    @wp.kernel(module=module)
    def invalid_loss(
        model: Model,
        initial: State,
        final: State,
        parameters: wp.array(dtype=real),
        accumulated: wp.array(dtype=real),
        loss: wp.array(dtype=real),
    ):
        loss[0] = not_finite

    return SimpleNamespace(
        stateless=stateless,
        running=running,
        terminal=terminal,
        invalid_policy=invalid_policy,
        invalid_loss=invalid_loss,
    )


def _independent_owner(dtype: np.dtype, integrator: str = "backward_euler") -> SolidDeviceModel:
    """Use a small moving spring so callback input-state derivatives are nonzero."""
    bundle = regular_solid("spring", precision=dtype.name, shape=(1, 1))
    bundle.config = replace(bundle.config, contact_enabled=False, integrator=integrator)
    owner = SolidDeviceModel(bundle, os.environ.get("WARP_TEST_DEVICE", "cpu"))
    velocity = np.zeros((owner.n, 2), dtype)
    velocity[0, 0] = 0.02
    owner.initial.v.assign(velocity)
    return owner


@pytest.mark.parametrize("integrator", ("backward_euler", "kvaerno3_native"))
@pytest.mark.parametrize("mode", ("policy", "objective", "terminal"))
def test_callbacks_independently_without_carry(mode: str, integrator: str, dtype: np.dtype) -> None:
    """Empty recurrent state and independently enabled callbacks retain direct VJPs."""
    owner = _independent_owner(dtype, integrator)
    kernels = _independent_kernels(owner)
    policy = NativePolicy(kernels.stateless, np.array([0.15], dtype)) if mode == "policy" else None
    objective = (
        None
        if mode == "policy"
        else NativeObjective(
            kernels.terminal,
            np.array([0.3], dtype),
            kernels.running if mode == "objective" else None,
        )
    )
    episode = SolidEpisode(
        owner,
        np.full((2, owner.channels), 0.15, dtype),
        substeps=2,
        settling_steps=1,
        checkpoint_stride=2,
        policy=policy,
        objective=objective,
    )
    report = episode.run(True)
    assert report["status"] == report["adjoint_status"] == 0
    gradients = episode.gradients()
    targets = [
        (owner.model.mass, gradients["model"]["mass"], (0,)),
        (owner.initial.v, gradients["initial"]["v"], (0, 0)),
    ]
    if policy is not None:
        assert gradients["policy_initial_carry"].shape == (0,)
        targets.append((episode.callbacks.policy_parameters, gradients["policy_parameters"], (0,)))
    for array, expected, index in targets:
        original = array.numpy().copy()
        epsilon = 2e-3 if dtype.name == "float32" else 1e-5
        if array is owner.model.mass:
            # This node is only 0.00625 kg: an absolute 0.002 perturbation
            # probes substantial nonlinear curvature rather than its derivative.
            epsilon = min(epsilon, 0.01 * float(original[index]))
        values = []
        for sign in (1, -1):
            value = original.copy()
            value[index] += sign * epsilon
            array.assign(value)
            assert (result := episode.run())["status"] == 0
            values.append(result["loss"])
        array.assign(original)
        np.testing.assert_allclose(
            (values[0] - values[1]) / (2 * epsilon),
            expected[index],
            rtol=0.01 if dtype.name == "float32" else 5e-4,
            atol=1e-4 if dtype.name == "float32" else 1e-8,
        )


@pytest.mark.parametrize("failure", ("command", "carry", "loss", "physics"))
def test_invalid_callback_or_physics_cannot_report_finite_success(
    failure: str, dtype: np.dtype
) -> None:
    """Invalid outputs and failed physical input poison loss and reject gradient access."""
    owner = _independent_owner(dtype)
    kernels = _independent_kernels(owner)
    policy = None
    if failure in ("command", "carry"):
        policy = NativePolicy(
            kernels.invalid_policy,
            np.array([1.5 if failure == "command" else 0.1], dtype),
            np.array([0.0], dtype) if failure == "carry" else (),
        )
    if failure == "physics":
        owner.initial.status.assign(np.asarray([3], np.int32))
    objective = NativeObjective(kernels.invalid_loss if failure == "loss" else kernels.terminal)
    episode = SolidEpisode(
        owner, np.zeros((1, owner.channels), dtype), policy=policy, objective=objective
    )
    result = episode.run(True)
    assert result["status"] != 0
    assert not np.isfinite(result["loss"])
    with pytest.raises(RuntimeError):
        episode.gradients()


def test_callback_definition_rejections_are_explicit(dtype: np.dtype) -> None:
    """Python callbacks and malformed dimensions fail before graph construction."""
    owner = _independent_owner(dtype)
    kernels = _independent_kernels(owner)
    commands = np.zeros((1, owner.channels), dtype)
    for dimension in (0, -1, (), (2, 0), "unknown", True):
        with pytest.raises(ValueError, match="dimension"):
            SolidEpisode(
                owner, commands, policy=NativePolicy(kernels.stateless, [0.1], launch_dim=dimension)
            )
    with pytest.raises(TypeError, match="Warp kernel"):
        SolidEpisode(owner, commands, policy=NativePolicy(lambda: None, [0.1]))
    with pytest.raises(TypeError, match="Warp kernel"):
        SolidEpisode(owner, commands, objective=NativeObjective(lambda: None))
