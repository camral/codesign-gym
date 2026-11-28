"""Exercise the declared Newton/MuJoCo dependency pair without external assets."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_newton_mujoco_free_body(device):
    from newton.solvers import SolverMuJoCo

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable to Warp")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        body = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_cpu=device == "cpu")
        state, next_state = model.state(), model.state()
        control = model.control()
        for _ in range(10):
            state.clear_forces()
            solver.step(state, next_state, control, None, 0.001)
            state, next_state = next_state, state
        pose = state.body_q.numpy()
        assert np.isfinite(pose).all()
        np.testing.assert_allclose(pose[0, :3], [0, 0, 1 - 9.81 * 0.001**2 * 55], atol=1e-5)
