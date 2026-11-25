"""Dict-design batches (LQ, Microgrid, Network): every accepted form gives the same per-lane designs, and the vector
validate_embodiment returns one bool per lane without raising."""
import warnings

import numpy as np
import pytest

from codesign_gym import make, make_vec
from core.embodiment_batch import split_dict_designs

IDS = ["LQStructure-LQG", "Microgrid-Random", "Network-Urban"]


def _stack(designs):
    return {k: np.stack([np.asarray(d[k]) for d in designs]) for k in designs[0]}


def _variant(env_id, base, i):
    """A valid design that differs from the base, per lane."""
    d = {k: np.array(v, copy=True) for k, v in base.items()}
    if env_id.startswith("LQ"):
        d["masses"] = np.full(6, 0.6 + 0.2 * i)
    elif env_id.startswith("Microgrid"):
        d["pv_capacity_kw"] = np.asarray(float(np.asarray(base["pv_capacity_kw"])) * (0.5 + 0.1 * i))
    else:
        d["num_bs"] = np.asarray(int(base["num_bs"]) + (i % 2), dtype=np.asarray(base["num_bs"]).dtype)
    return d


@pytest.mark.parametrize("env_id", IDS)
def test_stacked_dict_matches_list(env_id):
    envs, base = make_vec(env_id, 3)
    base = base[0] if isinstance(base, list) else base
    designs = [_variant(env_id, base, i) for i in range(3)]
    stacked = _stack(designs)
    assert envs.validate_embodiment(designs).tolist() == envs.validate_embodiment(stacked).tolist() == [True] * 3
    _, rej_list, info_list = envs.reset(seed=0, embodiment=designs)
    obs_list = envs.reset(seed=0, embodiment=designs)[0]
    obs_stacked, rej_stacked, _ = envs.reset(seed=0, embodiment=stacked)
    assert not rej_list.any() and not rej_stacked.any()
    np.testing.assert_allclose(np.asarray(obs_list, dtype=float), np.asarray(obs_stacked, dtype=float))
    envs.close()


@pytest.mark.parametrize("env_id", IDS)
def test_vector_validate_never_raises(env_id):
    envs, _ = make_vec(env_id, 2)
    assert envs.validate_embodiment({"not_a_key": 1.0}).tolist() == [False, False]
    assert envs.validate_embodiment([{"not_a_key": 1.0}, None]).tolist() == [False, True]
    assert envs.validate_embodiment([None, None, None]).tolist() == [False, False]      # wrong length
    _, base = make_vec(env_id, 1)
    base = base[0] if isinstance(base, list) else base
    wrong_size = {k: np.stack([np.asarray(v)] * 3) for k, v in base.items()}         # stacked for 3 lanes, env has 2
    assert envs.validate_embodiment(wrong_size).tolist() == [False, False]
    envs.close()


def test_broadcast_lanes_do_not_share_arrays():
    design = {"a": np.zeros(3)}
    lanes = split_dict_designs(design, 2, {"a": 1})
    lanes[0]["a"][0] = 5.0
    assert lanes[1]["a"][0] == 0.0 and design["a"][0] == 0.0


def test_leading_axis_of_one_broadcasts():
    lanes = split_dict_designs({"a": np.arange(3)[None]}, 2, {"a": 1})
    assert [lane["a"].tolist() for lane in lanes] == [[0, 1, 2], [0, 1, 2]]


def test_microgrid_validate_plain_floats_does_not_warn():
    env, base = make("Microgrid-Random")
    design = {k: float(np.asarray(v)) for k, v in base.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert env.validate_embodiment(design)
    env.close()


def test_network_reset_rejects_what_validate_rejects():
    envs, base = make_vec("Network-Urban", 2)
    base = base[0] if isinstance(base, list) else base
    fractional = {**base, "num_bs": 3.7}
    assert envs.validate_embodiment([fractional, None]).tolist() == [False, True]
    _, rejected, _ = envs.reset(seed=0, embodiment=[fractional, None])
    assert rejected.tolist() == [True, False]
    envs.close()
