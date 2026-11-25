"""Targeted eager compilation at the graph-construction boundary only."""

from types import SimpleNamespace

import warp as wp


def preload_kernels(
    device: "str | wp.Device", *namespaces: "SimpleNamespace | wp.Kernel | None"
) -> "tuple[str, ...]":
    """Load modules directly owned by the supplied kernel namespaces.

    Do not recursively walk models/arrays or Warp's global module registry.
    Remaining library kernels can load lazily during capture on the CUDA 12.3+
    runtime already required by conditional graphs.
    """
    modules = {}
    for namespace in namespaces:
        if namespace is None:
            continue
        values = (
            vars(namespace).values() if isinstance(namespace, SimpleNamespace) else (namespace,)
        )
        for value in values:
            if isinstance(value, wp.Kernel):
                modules[value.module.name] = value.module
    for name in sorted(modules):
        wp.load_module(modules[name], device=device)
    return tuple(sorted(modules))
