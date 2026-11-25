"""Grounding follows final geometry, not the preceding physics contacts."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grounding_tracks_kinematics_without_refreshing_contacts():
    import mujoco_warp as mjw
    import warp as wp
    from codesign_gym import make_vec

    env, _ = make_vec("BallCatcher-Pitch", 2, device="cuda", use_graph=True)
    try:
        env.reset()
        # Settle onto all four feet (the reset spawns clear of the floor), then
        # lift one world without a collision refresh.
        zero = torch.zeros((2, env.act_dim()), device="cuda")
        for _ in range(200):
            if env._all_feet_grounded().all():
                break
            env.step(zero)
        with env._wp_scope():
            mjw.forward(env.gpu_model, env.gpu_data)
        assert env._all_feet_grounded().all()
        contacts = wp.to_torch(env.gpu_data.nacon).clone()
        env.qpos_torch()[0, 2] += 3
        with env._wp_scope():
            mjw.kinematics(env.gpu_model, env.gpu_data)
        assert torch.equal(wp.to_torch(env.gpu_data.nacon), contacts)
        assert env._all_feet_grounded().tolist() == [False, True]

        # Conversely, stale empty contacts must not hide a newly grounded pose.
        env.qpos_torch()[1, 2] += 3
        with env._wp_scope():
            mjw.forward(env.gpu_model, env.gpu_data)
        contacts = wp.to_torch(env.gpu_data.nacon).clone()
        env.qpos_torch()[0, 2] -= 3
        with env._wp_scope():
            mjw.kinematics(env.gpu_model, env.gpu_data)
        assert torch.equal(wp.to_torch(env.gpu_data.nacon), contacts)
        grounded = env._all_feet_grounded()
        assert grounded.tolist() == [True, False]
        assert grounded.is_cuda and grounded.dtype == torch.bool
    finally:
        env.close()
