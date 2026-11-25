"""max_action_space_size / max_observation_space_size were removed from the API: each preset's spaces are fixed at
the largest size its embodiment space allows. Passing either is an error, not a keyword forwarded to an env."""

import pytest

import codesign_gym as cg


@pytest.mark.parametrize("name", ["max_action_space_size", "max_observation_space_size"])
@pytest.mark.parametrize("factory", [lambda **kw: cg.make_vec("LQStructure-LQR", 2, **kw), lambda **kw: cg.make("LQStructure-LQR", **kw)],
                         ids=["make_vec", "make"])
def test_removed_space_size_kwargs_raise(factory, name):
    with pytest.raises(TypeError, match=f"{name} was removed"):
        factory(**{name: 8})
