"""Warp kernels for the batched Swimmer step program.

Same shape as HalfCheetah's, with one v4 difference: Swimmer resets BOTH qpos
and qvel with uniform noise (HalfCheetah uses gaussian qvel noise).

Three launches, in order: tick (bump RNG), reset (masked pose + limb deform),
ctrl (map action -> actuator ctrl). All launch-only and mask-branched, so they
record cleanly into the CUDA-graph step program.
"""

from __future__ import annotations

import warp as wp

from .offsettable_logic import SegArrays, DeformArrays, apply_offsets


@wp.struct
class SimArrays:
    """Zero-copy views into the batched ``gpu_data`` the kernels write."""
    qpos: wp.array2d(dtype=wp.float32)
    qvel: wp.array2d(dtype=wp.float32)
    qacc: wp.array2d(dtype=wp.float32)
    qacc_warmstart: wp.array2d(dtype=wp.float32)
    ctrl: wp.array2d(dtype=wp.float32)


@wp.struct
class EnvArrays:
    """Per-world control/design buffers + the reset mask and RNG tick."""
    mask: wp.array(dtype=wp.bool)
    tick: wp.array(dtype=wp.int32)
    lane_seed: wp.array(dtype=wp.int32)       # per-world seed: the run's, or the world's own from a per-lane sequence
    qpos_init: wp.array(dtype=wp.float32)
    deform: wp.array2d(dtype=wp.float32)
    offsets: wp.array2d(dtype=wp.float32)
    ctrl_act: wp.array2d(dtype=wp.float32)
    ctrl_scale: wp.array(dtype=wp.float32)
    ctrl_mid: wp.array(dtype=wp.float32)


@wp.struct
class Const:
    seed: wp.int32
    nq: wp.int32
    nv: wp.int32
    nu: wp.int32
    n_seg: wp.int32
    reset_noise: wp.float32
    max_offset: wp.float32


@wp.func
def _stream_key(lane_seed: int, tick: int) -> int:
    """The random stream of one lane seed at one step: the two hashed together, so runs seeded s and s + 1 never replay
    each other a step apart, as a plain sum seed + tick would."""
    return wp.randi(wp.rand_init(lane_seed, tick))


@wp.kernel
def tick_kernel(env: EnvArrays):
    env.tick[0] = env.tick[0] + 1


@wp.kernel
def reset_kernel(env: EnvArrays, sim: SimArrays, dfm: DeformArrays,
                 seg: SegArrays, c: Const):
    w = wp.tid()
    if not env.mask[w]:
        return
    rs = wp.rand_init(_stream_key(env.lane_seed[w], env.tick[0]), w)
    for j in range(c.nq):
        sim.qpos[w, j] = env.qpos_init[j] + (2.0 * wp.randf(rs) - 1.0) * c.reset_noise
    for j in range(c.nv):
        # v4 Swimmer: qvel = init_qvel(0) + uniform(-noise, noise).
        sim.qvel[w, j] = (2.0 * wp.randf(rs) - 1.0) * c.reset_noise
        sim.qacc[w, j] = 0.0
        sim.qacc_warmstart[w, j] = 0.0
    apply_offsets(dfm, seg, env.deform, env.offsets, w, c.n_seg, c.max_offset)


@wp.kernel
def ctrl_kernel(env: EnvArrays, sim: SimArrays, c: Const):
    w = wp.tid()
    for a in range(c.nu):
        sim.ctrl[w, a] = (wp.clamp(env.ctrl_act[w, a], -1.0, 1.0)
                          * env.ctrl_scale[a] + env.ctrl_mid[a])
