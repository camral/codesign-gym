"""Embodiment summaries must be useful without materializing mesh arrays."""
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

import codesign_gym as cg
from codesign_gym import SingleEnv
from envs.nerograsp.spaces import Mesh, MeshSpace


def test_mesh_space_printing_is_informative_and_preserves_sampling(capsys):
    mesh = Mesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]))
    space = MeshSpace({"tip": mesh}, {"tip": 3})
    space.seed(42)
    expected_sample = space.sample()
    space.seed(42)
    vector_env = SimpleNamespace(embodiment_space=space)
    for env in (vector_env, SingleEnv(vector_env)):
        print(env.embodiment_space)
        assert str(env.embodiment_space) == repr(space)
    output = capsys.readouterr().out
    assert "MeshSpace" in output and "'tip'" in output
    assert "base_vertices=4" in output and "base_faces=4" in output
    assert "mounting_vertices=3" in output and "sample_perturbation_m=" in output
    assert "object at" not in output and "array(" not in output
    assert space.contains({"tip": mesh})
    np.testing.assert_array_equal(space.base_meshes["tip"].vertices, mesh.vertices)
    np.testing.assert_array_equal(space.sample()["tip"].vertices, expected_sample["tip"].vertices)


@pytest.mark.parametrize("name", ["ant", "half_cheetah", "hopper", "humanoid",
                                  "humanoid_standup", "swimmer", "walker2d"])
def test_offset_space_printing(name):
    cls = import_module(f"envs.{name}.embodiment").EmbodimentSpace
    space = cls(np.full(3, -1, dtype=np.float32), np.ones(3, dtype=np.float32), 2)
    for value in (space, spaces.Dict({"offsets": space})):
        assert "Box(-1.0, 1.0, (3,), float32)" in str(value)
    assert space.contains(np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("env_id", [name for name in cg.list_envs()
                                   if name.startswith("SoftWalker-")])
def test_soft_walker_factory_summaries(env_id):
    from envs.soft_walker.embodiment import FIELDS, bounds
    representations = []
    for make in (lambda: cg.make(env_id), lambda: cg.make_vec(env_id, 2)):
        env, base = make()
        try:
            space = env.embodiment_space
            representations.append(str(space))
            assert len(str(space)) < 500
            assert "grid=" in str(space) and f"shape={space.shape}" in str(space)
            assert all(field in str(space) for field in FIELDS)
            assert isinstance(space, spaces.Box)
            assert space.contains(base)
            lo, hi = bounds(env.dim, int(np.prod(env.grid)), env.channels)
            np.testing.assert_array_equal(space.low, lo)
            np.testing.assert_array_equal(space.high, hi)
            # A richer repr must not change the Box's seeded sampling semantics.
            box = spaces.Box(space.low, space.high, dtype=space.dtype)
            box.seed(42)
            space.seed(42)
            np.testing.assert_array_equal(space.sample(), box.sample())
        finally:
            env.close()
    assert representations[0] == representations[1]


@pytest.mark.slow
def test_configured_pokemon_factory_summaries():
    pytest.importorskip("pokemon_codesign_env")
    from pokemon_codesign_env.configs import FormatConfig, TeamSearchConfig
    from pokemon_codesign_env.envs.random_player_env import RandomPlayerEnv

    format_config = FormatConfig.from_showdown(RandomPlayerEnv.BATTLE_FORMAT)
    config = TeamSearchConfig(format=format_config, evs="optimized")
    kwargs = dict(shards=1, embodiment_config=config)
    summaries = []
    for make in (lambda: cg.make("Pokenv-Random", **kwargs),
                 lambda: cg.make_vec("Pokenv-Random", 2, **kwargs)):
        env, _ = make()
        try:
            space = env.embodiment_space
            summaries.append(repr(space))
            assert str(space) == repr(space)
            assert len(space.pokemon_keys) == config.team_size
            assert len(space.flat_space.nvec) > 0
            assert f"{config.team_size} Pokemon" in str(space)
            assert len(str(space)) < 200
        finally:
            env.close()
    assert summaries[0] == summaries[1]


@pytest.mark.parametrize("env_id", ["RotateInHand-Cube", "HandPickUp-Cube", "HandPickUp-Hammer", "HandPickUp-SmallHammer"])
def test_hand_factory_summaries(env_id):
    summaries = []
    for make in (lambda: cg.make(env_id), lambda: cg.make_vec(env_id, 2)):
        env, base = make()
        try:
            space = env.embodiment_space
            summaries.append(str(space))
            assert str(space) == repr(space)
            assert len(str(space)) < 500
            assert f"shape={space.shape}" in str(space)
            assert space.contains(base)
        finally:
            env.close()
    assert summaries[0] == summaries[1]
