"""Pure assembly shared by spring rollout entry points.

Callers own captured terrain/configuration, action validation and initialization
order. Keeping those choices outside this helper preserves each rollout contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..physics import spring

if TYPE_CHECKING:
    from ..embodiments.decoders import Decoder, Fields
    from ..physics.spring_config import SpringConfig
    from ..physics.terrain import Terrain
    from ..types import Embedding


def assemble_spring(
    decoder: Decoder,
    embedding: Embedding,
    topology: spring.Topology,
    config: SpringConfig,
    channels: int,
    terrain: Terrain | None,
) -> tuple[spring.Model, Fields]:
    """Decode and assemble a spring model with the caller's fixed spawn transform.

    Fields stay as device arrays and the returned model owns no mutable host
    state. Terrain/reference positions use metres. This function neither settles
    nor validates a candidate; shape/configuration errors propagate from decoding
    or assembly. It can run inside the caller's existing JAX transformation.
    """
    fields = decoder.decode(embedding, topology.points)
    model = spring.instantiate(fields, topology, config, channels)
    if terrain is not None:
        from ..drivers.environment import spawn_shift

        model = model._replace(
            reference=model.reference.at[:, 1].add(
                spawn_shift(terrain, config.size) - config.ground_height
            ),
            terrain=terrain,
        )
    return model, fields
