"""Host-only capability checks for the existing integration implementations.

Selection happens before environment construction or graph compilation. This
catalog does not execute physics, change solver tolerances, import optional
backends, or claim that every listed configuration has numerical validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IntegratorName = Literal["backward_euler", "kvaerno3_native", "kvaerno3"]


@dataclass(frozen=True)
class IntegratorSpec:
    """An implementation's supported construction interfaces and physical families.

    ``name`` retains the existing configuration spelling: ``kvaerno3`` is the
    optional JAX/Diffrax implementation, while ``kvaerno3_native`` selects the
    existing matrix-free JAX or native Warp stages. Availability is structural;
    optional dependency installation and specific solver settings are checked
    by the selected environment/configuration.
    """

    name: IntegratorName
    physics: tuple[str, ...]
    backends: tuple[str, ...]
    interfaces: tuple[str, ...]
    description: str


_PHYSICS = ("spring", "beam", "fem", "mpm2", "mpm3")
_BACKENDS = ("jax", "warp")
_INTERFACES = ("engine", "registered")

INTEGRATORS = (
    IntegratorSpec(
        "backward_euler",
        _PHYSICS,
        _BACKENDS,
        _INTERFACES,
        "Existing implicit backward-Euler step for each physical family.",
    ),
    IntegratorSpec(
        "kvaerno3_native",
        ("spring",),
        _BACKENDS,
        _INTERFACES,
        "Existing fixed-step spring stages; JAX requires GMRES, Warp uses native stages.",
    ),
    IntegratorSpec(
        "kvaerno3",
        ("spring",),
        ("jax",),
        ("engine",),
        "Optional Diffrax/Optimistix spring implementation with dense linear solves.",
    ),
)


def available_integrators(
    *, backend: str, physics: str, interface: str = "engine"
) -> tuple[IntegratorName, ...]:
    """List implemented choices without importing JAX, Warp, or Diffrax.

    ``physics`` is spring/beam/fem/mpm2/mpm3, not an environment ID. ``interface``
    distinguishes general engine factories from the registered per-cell Gym
    wrapper. Unknown selectors raise ValueError rather than return an empty
    capability set. Specific solver settings and installed dependencies are not
    inspected; the result is not a convergence or device-certification promise.
    """
    for label, value, choices in (
        ("backend", backend, _BACKENDS),
        ("physics", physics, _PHYSICS),
        ("interface", interface, _INTERFACES),
    ):
        if value not in choices:
            raise ValueError(f"Unknown {label} {value!r}; choose from {choices}")
    return tuple(
        spec.name
        for spec in INTEGRATORS
        if backend in spec.backends and physics in spec.physics and interface in spec.interfaces
    )


def resolve_integrator(
    selection: str | None,
    *,
    backend: str,
    physics: str,
    configured: str = "backward_euler",
    interface: str = "engine",
) -> IntegratorName:
    """Resolve an optional override while preserving the configured default.

    This function is side-effect free and validates the complete interface/
    backend/family combination before callers allocate device resources. It
    never substitutes another method for an unsupported request. Callers own
    immutable configuration replacement and method-specific solver validation.
    """
    choices = available_integrators(backend=backend, physics=physics, interface=interface)
    requested = configured if selection is None else selection
    for choice in choices:
        if requested == choice:
            return choice
    raise ValueError(
        f"Integrator {requested!r} is unsupported for {interface} {backend}/{physics}; "
        f"choose from {choices}"
    )
