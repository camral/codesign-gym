"""Static host contracts shared by benchmark and numerical evidence tools."""

from typing import Any, Protocol, TypedDict


class JittedCallable(Protocol):
    """JAX callable with a lowering boundary; executable metadata is backend-specific."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate with the original pytree arguments."""
        ...

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        """Produce JAX's version-dependent lowered executable interface."""
        ...


class NumericalTolerance(TypedDict):
    """Keyword tolerances for assert_allclose; no solver settings are changed."""

    rtol: float
    atol: float
