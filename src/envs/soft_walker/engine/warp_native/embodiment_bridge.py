"""Differentiate embodiment assembly around a native Warp physics episode.

JAX evaluates the decoder/assembly and its vector-Jacobian product. Warp owns all
physics steps, settling, checkpoint replay, and implicit adjoints. Transfers occur
at the complete-episode input/output boundary, never inside the time integrator.
This explicit host optimizer interface is not a JAX-traceable simulation callback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from ..device import resolve_dtype
from ..drivers import environment as drivers
from ..physics import spring
from ..types import Embedding, HostEmbedding, PhysicsModel, PhysicsState

if TYPE_CHECKING:
    from ..drivers.spring_environment import Spring2DEnv
    from ..embodiments.decoders import Fields
    from ..physics import beam, fem, mpm
    from .episode import Episode
    from .host_types import EpisodeOptions, EpisodeResult
    from .model import Bundle, DeviceModel
    from .solid_episode import SolidEpisode
    from .solid_model import SolidBundle, SolidDeviceModel


def _ledger(state: PhysicsState) -> jax.Array:
    """Collect energy (J) and time (s) without detaching assembly derivatives."""
    return jnp.stack((state.work_signed, state.work_positive, state.dissipated, state.time))


def _resolved_arrays(
    kind: str, model: PhysicsModel, state: PhysicsState, template: Mapping[str, ArrayLike]
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Map existing JAX model/state leaves to the native resolved-array schema.

    Template entries supply only physics-inapplicable constants. Every coefficient
    consumed by the selected native force law is replaced by its differentiable
    assembled value. Integer topology and masks are checked separately at setup.
    """
    fields = {key: jnp.asarray(value) for key, value in template.items()}
    if kind.startswith("mpm"):
        model, state = cast("mpm.Model", model), cast("mpm.State", state)
        for name in (
            "alpha",
            "young",
            "poisson",
            "density",
            "strength",
            "direction",
            "kind",
            "channel",
            "contact_area",
        ):
            fields[name] = getattr(model, name)
        for name in ("reference", "volume", "face_offset"):
            fields[name] = getattr(model.cloud, name)
        initial = {name: getattr(state, name) for name in ("x", "v", "F", "C", "actuator")}
    else:
        fields["mass"] = model.mass
        if kind == "spring":
            model, state = cast("spring.Model", model), cast("spring.State", state)
            fields.update(
                reference=model.reference,
                entities=model.edges,
                cells=model.cells,
                cell_alpha=model.occupancy,
                stiffness=model.stiffness,
                damping=model.damping,
                length=model.length,
                routing=model.logstrain_map,
                contact_area=model.contact_k[:, None],
            )
            displacement = state.x - model.reference
        elif kind == "beam":
            model, state = cast("beam.Model", model), cast("beam.State", state)
            fields.update(
                reference=model.topology.reference,
                entities=model.topology.edges,
                inertia=model.inertia,
                extension=model.extension,
                bending=model.bend,
                routing=model.strain_map,
                contact_area=model.contact_area,
                axis=model.topology.axis,
                offsets=model.topology.offsets,
            )
            # Metadata follows the export schema; forces consume the assembled
            # extension/bending tensors above, including their Poisson chain rule.
            i, k = model.topology.edges.T
            shear = model.young / (2 * (1 + model.poisson))
            fields["young"] = 2 / (1 / model.young[i] + 1 / model.young[k])
            edge_shear = 2 / (1 / shear[i] + 1 / shear[k])
            fields["poisson"] = fields["young"] / (2 * edge_shear) - 1
            displacement = state.u
        else:
            model, state = cast("fem.Model", model), cast("fem.State", state)
            fields.update(
                reference=model.mesh.reference,
                entities=model.mesh.cells,
                gradients=model.mesh.grad,
                weights=model.mesh.weights,
                contact_area=model.contact_area[:, None],
            )
            for name in ("alpha", "young", "poisson", "strength", "direction", "kind", "channel"):
                fields[name] = getattr(model, name)
            displacement = state.u
        dim = fields["reference"].shape[1]
        fields["free"] = jnp.broadcast_to(model.active[:, None], fields["reference"].shape).astype(
            fields["reference"].dtype
        )
        if kind == "fem":
            fields["free"] = fields["free"] * ~cast("fem.Model", model).fixed
        rotation = (
            cast("beam.State", state).rotation
            if kind == "beam"
            else jnp.broadcast_to(
                jnp.eye(dim, dtype=fields["reference"].dtype), (len(model.mass), dim, dim)
            )
        )
        initial = dict(
            u=displacement,
            v=state.v,
            rotation=rotation,
            omega=cast("beam.State", state).omega
            if kind == "beam"
            else jnp.zeros_like(displacement),
            actuator=state.actuator,
        )
    initial["ledger"] = _ledger(state)
    initial["status"] = jnp.reshape(state.status, (1,)).astype(jnp.int32)
    return fields, initial


class EmbodimentEpisode:
    """Fixed-topology decoder-to-Warp value/gradient interface for host optimizers.

    Embedding dictionaries retain their decoder schema; returned cotangents cover
    floating-point leaves only. Integer labels/topology are fixed. The supplied JAX
    environment describes assembly and must outlive this object; it is never stepped.
    CUDA episodes still execute as one resident native graph per complete evaluation.
    """

    def __init__(
        self,
        env: Spring2DEnv | drivers.BodyEnv,
        embodiment: HostEmbedding,
        commands: ArrayLike,
        *,
        device: str = "cpu",
        **episode_options: Unpack[EpisodeOptions],
    ) -> None:
        """Validate a body and allocate native storage; reject unsupported exports.

        ``episode_options`` are the selected native Episode constructor's options.
        Native settling defaults to the environment's original settling duration,
        measured in physical steps, including when action holding was overridden.
        """
        resolve_dtype(env.dtype)
        self.env = env
        self.kind = cast(str, getattr(env, "kind", "spring"))
        values, model, errors = env._prepare(embodiment)
        if errors or values is None or model is None:
            raise ValueError("Embodiment rejected: " + ", ".join(errors))
        self.base = {name: np.asarray(value).copy() for name, value in values.items()}
        self.continuous_names = tuple(
            name for name, value in self.base.items() if value.dtype.kind == "f"
        )
        self._fixed_semantics = self._discrete_semantics(values)
        state = (
            drivers.module(self.kind).initial_state(model)
            if self.kind != "spring"
            else spring.initial_state(cast("spring.Model", model))
        )
        if self.kind.startswith("mpm"):
            from .episode import Episode
            from .model import DeviceModel
            from .mpm_conversion import from_jax as export_mpm

            body_env = cast(drivers.BodyEnv, env)
            mpm_bundle = export_mpm(
                cast("mpm.Model", model),
                cast("mpm.State", state),
                cast("mpm.Grid", body_env._grid),
                body_env.config.dynamics,
                dt=env.config.dt,
            )
            bundle: Bundle | SolidBundle = mpm_bundle
            owner: DeviceModel | SolidDeviceModel = DeviceModel(mpm_bundle, device)
            episode_class: Callable[..., Episode | SolidEpisode] = Episode
        else:
            from .solid_conversion import from_jax as export_solid
            from .solid_episode import SolidEpisode
            from .solid_model import SolidDeviceModel

            cfg = (
                cast("Spring2DEnv", env).config
                if self.kind == "spring"
                else cast(drivers.BodyEnv, env).config.dynamics
            )
            solid_bundle = export_solid(
                self.kind,
                cast("spring.Model | beam.Model | fem.Model", model),
                cast("spring.State | beam.State | fem.State", state),
                cfg,
                dt=env.config.dt,
            )
            bundle = solid_bundle
            owner = SolidDeviceModel(solid_bundle, device)
            episode_class = SolidEpisode
        self.bundle: Bundle | SolidBundle = bundle
        self.owner: DeviceModel | SolidDeviceModel = owner
        initial_cfg = env._initialization_config
        episode_options.setdefault("substeps", env.config.substeps)
        episode_options.setdefault(
            "settling_steps", initial_cfg.settle_steps * initial_cfg.substeps
        )
        self.episode: Episode | SolidEpisode = episode_class(owner, commands, **episode_options)
        self._fixed_fields = {
            name: value.copy()
            for name, value in bundle.fields.items()
            if value.dtype.kind != "f" or name == "free"
        }
        self._model_keys = tuple(name for name in bundle.fields if name not in self._fixed_fields)
        assert bundle.initial is not None
        self._initial_keys = tuple(name for name in bundle.initial if name != "status")
        self._assemble = jax.jit(self._continuous_assembly)

    def _decoded_fields(self, embodiment: Embedding) -> Fields:
        """Sample the environment's fields before assembly can discard discrete labels."""
        if self.kind != "spring":
            body_env = cast(drivers.BodyEnv, self.env)
            return body_env.decoder.decode(
                {key: value for key, value in embodiment.items() if key != "poisson"},
                body_env._domain.points,
            )
        env = cast("Spring2DEnv", self.env)
        return env.decoder.decode(embodiment, env._topology.points)

    def _discrete_semantics(self, embodiment: Embedding) -> dict[str, np.ndarray]:
        """Snapshot active sampled cells and actuator labels at the host validation boundary.

        Spring/beam assembly folds labels into floating routing matrices, and MPM
        has no solid ``free`` mask. Checking only native integer buffers therefore
        cannot establish the fixed discrete structure required by this bridge.
        """
        fields = self._decoded_fields(embodiment)
        return dict(
            active=np.asarray(fields.occupancy) > 0,
            kind=np.asarray(fields.actuator_type).copy(),
            channel=np.asarray(fields.channel).copy(),
        )

    def _model(self, embodiment: Embedding) -> PhysicsModel:
        """Decode and instantiate the same model as the environment, without dynamics."""
        fields = self._decoded_fields(embodiment)
        if self.kind != "spring":
            body_env = cast(drivers.BodyEnv, self.env)
            return drivers.build_model(
                self.kind,
                body_env._domain,
                fields,
                body_env.channels,
                embodiment.get("poisson"),
                body_env._terrain,
                body_env.config,
            )
        env = cast("Spring2DEnv", self.env)
        model = spring.instantiate(fields, env._topology, env.config, env.channels)
        if env._terrain is not None:
            shift = drivers.spawn_shift(env._terrain, env.config.size) - env.config.ground_height
            model = model._replace(
                reference=model.reference.at[:, 1].add(shift), terrain=env._terrain
            )
        return model

    def _continuous_assembly(
        self, parameters: dict[str, jax.Array]
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """Return only floating model/state leaves consumed by native adjoints."""
        embodiment = {name: jnp.asarray(value) for name, value in self.base.items()}
        embodiment.update(parameters)
        model = self._model(embodiment)
        state = (
            drivers.module(self.kind).initial_state(model)
            if self.kind != "spring"
            else spring.initial_state(cast("spring.Model", model))
        )
        fields, initial = _resolved_arrays(self.kind, model, state, self.bundle.fields)
        return (
            {name: fields[name] for name in self._model_keys},
            {name: initial[name] for name in self._initial_keys},
        )

    def _validate(self, embodiment: HostEmbedding) -> dict[str, jax.Array]:
        """Reject schema, topology, or discrete-route changes before updating graph inputs."""
        resolve_dtype(self.env.dtype)
        if set(embodiment) != set(self.base):
            raise ValueError("Embodiment keys must match the bound decoder")
        for name, base in self.base.items():
            value = np.asarray(embodiment[name])
            if value.shape != base.shape or value.dtype != base.dtype:
                raise ValueError(f"Embodiment {name} shape/dtype changed")
            if name not in self.continuous_names and not np.array_equal(value, base):
                raise ValueError(f"Discrete embodiment field {name} is fixed")
        values, model, errors = self.env._prepare(embodiment)
        if errors or values is None or model is None:
            raise ValueError("Embodiment rejected: " + ", ".join(errors))
        semantics = self._discrete_semantics(values)
        if any(
            not np.array_equal(semantics[name], value)
            for name, value in self._fixed_semantics.items()
        ):
            raise ValueError("Sampled active masks and discrete actuator routing are fixed")
        state = (
            drivers.module(self.kind).initial_state(model)
            if self.kind != "spring"
            else spring.initial_state(cast("spring.Model", model))
        )
        fields, _ = _resolved_arrays(self.kind, model, state, self.bundle.fields)
        if any(
            not np.array_equal(np.asarray(fields[name]), value)
            for name, value in self._fixed_fields.items()
        ):
            raise ValueError(
                "Resolved topology, active masks, and discrete actuator routing are fixed"
            )
        return {name: jnp.asarray(values[name]) for name in self.continuous_names}

    def evaluate(
        self, embodiment: HostEmbedding, commands: ArrayLike, *, gradient: bool = False
    ) -> tuple[EpisodeResult, dict[str, np.ndarray] | None, np.ndarray | None]:
        """Evaluate one body/controller candidate and optionally its exact chain rule.

        Returns ``(native_result, embodiment_gradient, command_gradient)``. The
        gradients are absent for forward-only calls. Invalid primal/adjoint solves
        raise instead of returning usable derivatives. Storage and graph identities
        persist across updates, and the caller's input arrays are never mutated.
        """
        parameters = self._validate(embodiment)
        command_array = np.asarray(commands, dtype=self.owner.precision)
        if (
            command_array.shape != self.episode.commands.shape
            or not np.all(np.isfinite(command_array))
            or np.any(np.abs(command_array) > 1)
        ):
            raise ValueError("Commands must preserve their shape and remain finite in [-1, 1]")
        if gradient:
            (fields, initial), pullback = jax.vjp(self._assemble, parameters)
        else:
            fields, initial = self._assemble(parameters)
        arrays = [*fields.values(), *initial.values()]
        if any(not np.all(np.isfinite(np.asarray(value))) for value in arrays):
            raise ValueError("Assembly produced nonfinite physical arrays")
        for name, value in fields.items():
            getattr(self.owner.model, name).assign(np.asarray(value))
        for name, value in initial.items():
            getattr(self.owner.initial, name).assign(np.asarray(value))
        self.episode.set_commands(command_array)
        result = self.episode.run(gradient=gradient)
        if not gradient:
            return result, None, None
        derivatives = self.episode.gradients()
        cotangent = (
            {
                name: jax.device_put(derivatives["model"][name], self.env.device)
                for name in self._model_keys
            },
            {
                name: jax.device_put(derivatives["initial"][name], self.env.device)
                for name in self._initial_keys
            },
        )
        (body_gradient,) = pullback(cotangent)
        host_gradient = {name: np.asarray(value) for name, value in body_gradient.items()}
        if any(not np.all(np.isfinite(value)) for value in host_gradient.values()):
            raise RuntimeError("Embodiment assembly pullback produced nonfinite derivatives")
        return result, host_gradient, derivatives["commands"]
