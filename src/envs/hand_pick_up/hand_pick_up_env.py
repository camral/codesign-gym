"""Hand pick-up: a palm-down hand lifts an object off a table and turns it to
target orientations, with a different hand shape in every world.

The rotate-in-hand task with the hand hung palm-down over a table and the
object resting on it. Differences from rotate_in_hand:

- touching the table as well as the hand halves the reward multiplier, and a
goal counts only grasped and off the table;
- the shaping starts at the first lift, not at the respawn;
- the episode ends when the object is ``termination_dist`` from the palm in xy;
- ``max_ftip_return`` optionally rewards closing the distance from the
fingertips to the object (previous minus current, so opening it costs the
same);
- with ``train_rel_pos`` the goal is also a position ``rel_pos``, and the
orientation and position each take half the shaping. ``rel_pos`` is always
given in the palm frame: under ``goal_frame="palm"`` it moves with the palm,
under ``goal_frame="world"`` it is fixed where it lies at the palm's reset pose;
- the observation also carries the table-touch flag and the target position in
the palm frame (``rel_pos`` itself unless it is world-fixed), present whatever
the goal settings.

Code layout:

- hand_pick_up_env.py: the table, the spawn, and the overrides of
``RotateInHandParallelEnv``;
- the other modules are copies of rotate_in_hand's, so this package stands alone.
"""

from __future__ import annotations

import logging
import warnings

import mujoco
import numpy as np
import torch
import warp as wp

from .rotate_in_hand_env import (PALM_BODY, _SPAWN_CLEARANCE,
                                 RotateInHandParallelEnv, _quat_mat)

TABLE_NAME = "table"
_TABLE_THICK = 0.1                         # placed by its top face, the thickness extends down


def _geom_bottom(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> float:
    # World z of the bottom of geom g's box, the mirror of _geom_top
    aabb = np.asarray(m.geom_aabb[g], dtype=np.float64).reshape(2, 3)
    up = np.asarray(d.geom_xmat[g], dtype=np.float64).reshape(3, 3)[2]
    return float(d.geom_xpos[g][2] + up @ aabb[0] - np.abs(up) @ aabb[1])


def _depth_below_origin(m: mujoco.MjModel, body: int,
                        quat: np.ndarray) -> float:
    """How far ``body``'s geoms reach below its origin with the body turned
    ``quat``, over the corners of each geom's box."""
    turn = np.zeros(9)
    mujoco.mju_quat2Mat(turn, np.asarray(quat, dtype=np.float64))
    turn = turn.reshape(3, 3)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                      for z in (-1, 1)], dtype=np.float64)
    lowest = 0.0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != body:
            continue
        aabb = np.asarray(m.geom_aabb[g], dtype=np.float64).reshape(2, 3)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        corners = m.geom_pos[g] + (aabb[0] + signs * aabb[1]) @ rot.reshape(3, 3).T
        lowest = min(lowest, float((corners @ turn.T)[:, 2].min()))
    return -lowest


class HandPickUpParallelEnv(RotateInHandParallelEnv):
    """``RotateInHandParallelEnv`` with a table under a palm-down hand.

    Adds the table geometry and spawn point, the lift latch, and the memories
    of the position and fingertip shaping terms.
    """

    HAND_QUAT = (0.0, 1.0, 0.0, 0.0)      # half a turn about x: palm down
    EXTRA_OBS = 4                         # touching the table, the target position (palm frame)
    TABLE_TOUCH = 1

    def __init__(
        self,
        num_envs: int = 4,
        *,
        termination_dist: float = 0.15,   # object-palm xy distance that loses it
        table_gap: float = 0.0,           # table top below the palm origin, 0 derives it
        rel_pos=None,                     # target position, palm frame (hand-specific, no default)
        train_rel_pos: bool = False,
        goal_pos_tol: float = 0.01,
        max_ftip_return: float = 0.0,     # fingertip shaping per episode, 0 turns it off
        **kwargs,
    ):
        if termination_dist <= 0.0:
            raise ValueError("termination_dist is a distance and must be > 0, "
                             f"got {termination_dist}")
        if table_gap < 0.0:
            raise ValueError(f"table_gap must be >= 0 (0 derives it), got "
                             f"{table_gap}")
        if train_rel_pos and rel_pos is None:
            raise ValueError(
                "train_rel_pos needs rel_pos, the object's target position in "
                "the palm's frame. It depends on the hand, so there is no "
                "default.")
        pos = np.zeros(3) if rel_pos is None else np.asarray(rel_pos, float)
        if pos.shape != (3,):
            raise ValueError(f"rel_pos is an x,y,z position, got {rel_pos}")
        if goal_pos_tol <= 0.0:
            raise ValueError("goal_pos_tol is a distance and must be > 0, got "
                             f"{goal_pos_tol}")
        if max_ftip_return < 0.0:
            raise ValueError("max_ftip_return is a return and must be >= 0 (0 "
                             f"turns it off), got {max_ftip_return}")
        self.termination_dist = float(termination_dist)
        self.table_gap = float(table_gap)
        self.rel_pos = pos
        self.train_rel_pos = bool(train_rel_pos)
        self.goal_pos_tol = float(goal_pos_tol)
        self.max_ftip_return = float(max_ftip_return)
        # A world-fixed target, which moves relative to the palm.
        self._pos_world = (self.train_rel_pos
                           and kwargs.get("goal_frame", "palm") == "world")
        super().__init__(num_envs, **kwargs)

    # ------------------------------------------------------------------ #
    # Model assembly
    # ------------------------------------------------------------------ #
    def _build_world_spec(self) -> mujoco.MjSpec:
        """Add the table under the palm and set the spawn point on it.

        The table is a box, since MuJoCo collides no heightfield with a plane. A
        derived ``table_gap`` fits the object under the palm in any orientation.
        """
        spec = super()._build_world_spec()
        m = spec.compile()
        d = mujoco.MjData(m)
        d.qpos[:] = self._reset_pose(m)
        mujoco.mj_forward(m, d)
        palm = m.body(PALM_BODY).id
        obj = m.body(self._obj_body).id
        mine = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == palm]
        if not mine:
            raise RuntimeError(f"body {PALM_BODY!r} has no geoms, so there is "
                               "no palm to hang the table under")
        origin = float(d.xpos[palm][2])
        low = min(_geom_bottom(m, d, g) for g in mine)
        reach = max(float(m.geom_rbound[g]) for g in range(m.ngeom)
                    if int(m.geom_bodyid[g]) == obj)
        derived = origin - low + 2.0 * reach + _SPAWN_CLEARANCE
        gap = self.table_gap or derived
        if not self.table_gap:
            logging.getLogger(__name__).info(
                "table_gap 0: table top derived at %.0f mm below the palm origin", gap * 1e3)
        elif gap < derived and self.randomise_init_quat:
            warnings.warn(f"table_gap {gap * 1e3:.0f} mm is less than the "
                          f"{derived * 1e3:.0f} mm a randomly turned object needs, "
                          "so some spawns start inside the palm.", UserWarning, stacklevel=2)
        top = origin - gap
        xy = np.mean([d.geom_xpos[g][:2] for g in mine], axis=0)
        # How far the object reaches below its origin as it spawns
        if self.randomise_init_quat:
            below = reach
        else:
            turn = np.zeros(4)
            mujoco.mju_mulQuat(turn, d.xquat[palm], self.init_quat)
            below = _depth_below_origin(m, obj, turn)
        self._table_top = float(top)
        self._table_spawn = np.array([xy[0], xy[1],
                                      top + below + _SPAWN_CLEARANCE])

        table = spec.worldbody.add_geom()
        table.name = TABLE_NAME
        table.type = mujoco.mjtGeom.mjGEOM_BOX
        table.size = [self.termination_dist, self.termination_dist, 0.5 * _TABLE_THICK]
        table.pos = [xy[0], xy[1], top - 0.5 * _TABLE_THICK]
        table.rgba = [0.36, 0.40, 0.46, 0.95]

        self._check_start(spec, gap, reach)
        return spec

    def _check_start(self, spec: mujoco.MjSpec, gap: float,
                     reach: float) -> None:
        """Raise if the hand starts inside the table, or inside the object where
        it spawns, with every pad at full elevation. A randomly turned object has
        no one pose to test, so a hand reaching into its bounding sphere only warns."""
        m = spec.compile()
        for h in range(m.nhfield):
            a, n = int(m.hfield_adr[h]), int(m.hfield_nrow[h] * m.hfield_ncol[h])
            m.hfield_data[a:a + n] = self.hfield_offset_fraction
        d = mujoco.MjData(m)
        d.qpos[:] = self._reset_pose(m)
        palm, obj = m.body(PALM_BODY).id, m.body(self._obj_body).id
        mujoco.mj_kinematics(m, d)
        turn = np.zeros(4)
        mujoco.mju_mulQuat(turn, d.xquat[palm], self.init_quat)
        a = int(m.jnt_qposadr[m.body_jntadr[obj]])
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = self._table_spawn, turn
        mujoco.mj_forward(m, d)
        table = m.geom(TABLE_NAME).id
        hand = lambda g: int(m.geom_bodyid[g]) not in (0, obj)
        sunk = {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0.0:
                continue
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if not hand(g):
                    continue
                if other == table:
                    what = "the table"
                elif (int(m.geom_bodyid[other]) == obj
                      and not self.randomise_init_quat):
                    what = "the object"
                else:
                    continue
                sunk[what, g] = min(sunk.get((what, g), 0.0), float(c.dist))
        if sunk:
            names = ", ".join(
                f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))}"
                f" geom {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or g}"
                f" {-dist * 1e3:.1f} mm into {what}"
                for (what, g), dist in sorted(sunk.items(), key=lambda kv: kv[1]))
            raise RuntimeError(
                f"the hand starts inside something: {names}. Open the hand in "
                f"the keyframe init_key names, or move the table with table_gap "
                f"(now {gap * 1e3:.0f} mm below the palm origin).")

        if self.randomise_init_quat:
            mine = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == obj]
            near = 0
            for g in range(m.ngeom):
                if not hand(g) or not any(
                        m.geom_contype[g] & m.geom_conaffinity[o]
                        or m.geom_conaffinity[g] & m.geom_contype[o]
                        for o in mine):
                    continue
                box = m.geom_aabb[g].reshape(2, 3)
                p = (d.geom_xmat[g].reshape(3, 3).T
                     @ (self._table_spawn - d.geom_xpos[g]) - box[0])
                if np.linalg.norm(p - np.clip(p, -box[1], box[1])) < reach - 1e-4:
                    near += 1
            if near:
                warnings.warn(f"{near} hand geoms reach into the {reach * 1e3:.0f} mm "
                              "sphere a randomly turned object is dropped inside, so "
                              "some spawns start inside the hand. A fixed init_quat or "
                              "a more open keyframe avoids it.", UserWarning, stacklevel=2)

    def _spawn_point(self, m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        # On the table, under the middle of the palm
        return self._table_spawn.copy()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _alloc_device_state(self) -> None:
        super()._alloc_device_state()
        n, dev = self.num_envs, self.device
        # Whether the object was lifted yet (in-place on the persistent buffer
        # bound to env.armed), and the position term's memory
        self._armed.zero_()
        self._prev_pos = torch.zeros(n, device=dev)
        self._pos0 = torch.ones(n, device=dev)
        m = self.m
        # Fingertips: the leaf bodies of the hand's tree
        parents = set(int(p) for p in m.body_parentid[1:])
        root = m.body_rootid[self._palm_bid]
        self._tip_bids = torch.as_tensor(
            [b for b in range(1, m.nbody)
             if b not in parents and m.body_rootid[b] == root], device=dev)
        self._prev_tip = torch.zeros(n, device=dev)
        self._tip0 = torch.ones(n, device=dev)

    def _build_kernel_params(self) -> None:
        super()._build_kernel_params()
        c = self._consts
        c.pos_on = int(self.train_rel_pos)
        c.rel_pos = wp.vec3(*(float(v) for v in self.rel_pos))
        c.pos_tol = float(self.goal_pos_tol)
        # rel_pos carried into the world by the palm's reset pose.
        self.goal_pos_world = self._palm_pos + _quat_mat(self._palm_quat0) @ self.rel_pos
        self._goal_pos_world_t = torch.as_tensor(self.goal_pos_world, dtype=torch.float32,
                                                 device=self.device)
        c.goal_pos_w = wp.vec3(*(float(v) for v in self.goal_pos_world))
        self._rel_pos_obs = (torch.as_tensor(self.rel_pos, dtype=torch.float32, device=self.device)
                             / self.max_rel_dist).expand(self.num_envs, 3)

    # ------------------------------------------------------------------ #
    # Obs / reward
    # ------------------------------------------------------------------ #
    def _extra_obs(self) -> list[torch.Tensor]:
        return [self._touch_world.unsqueeze(1), self._goal_pos_obs()]

    def _goal_pos_obs(self) -> torch.Tensor:
        """The target position in the palm's frame, scaled like obj_pos."""
        if not self._pos_world:
            return self._rel_pos_obs
        # The world-fixed target as the palm sees it now
        xpos = wp.to_torch(self.gpu_data.xpos)[:, self._palm_bid]
        xmat = wp.to_torch(self.gpu_data.xmat)[:, self._palm_bid].view(-1, 3, 3)
        rel = torch.einsum("nji,nj->ni", xmat, self._goal_pos_world_t - xpos)
        return rel / self.max_rel_dist

    def _rebase_episode(self, mask: torch.Tensor) -> None:
        """Also HandPickUp's own state: the shaping waits for the first lift again, and the position and fingertip
        terms count from the respawn pose."""
        super()._rebase_episode(mask)
        self._armed[mask] = False
        self._prev_pos[mask] = self._pos_dist[mask]
        self._pos0[mask] = self._pos_dist[mask].clamp(min=self.goal_pos_tol)
        xpos = wp.to_torch(self.gpu_data.xpos)
        tips = (xpos[:, self._tip_bids] - xpos[:, self._obj_bid, None]).norm(dim=2).sum(dim=1)
        self._prev_tip[mask] = tips[mask]
        self._tip0[mask] = tips[mask].clamp(min=1e-6)

    def _shaping_live(self, respawned: torch.Tensor) -> torch.Tensor:
        # Latched: live from the step after the hand first holds the object clear of the table.
        # Mutated in place so force_kernel's CUDA-graphed view of env.armed stays current.
        live = self._armed & ~respawned
        self._armed.copy_(live | (self._gate > 0))
        return live

    def _pose_reward(self, progress: torch.Tensor,
                     live: torch.Tensor) -> torch.Tensor:
        if not self.train_rel_pos:
            return super()._pose_reward(progress, live)
        pos = self._pos_dist
        gain, self._prev_pos = self._step_gain(self._prev_pos, pos, live)
        # Scaled from the first lift and never re-based, since rel_pos never moves
        pos_progress = gain / self._pos0 * (0.5 * self.max_pose_return)
        self._pos0 = torch.where(live, self._pos0,
                                 pos.clamp(min=self.goal_pos_tol))
        return super()._pose_reward(0.5 * progress + pos_progress, live)

    def _extra_reward(self, respawned: torch.Tensor) -> torch.Tensor | float:
        # Fingertip term: previous minus current summed tip-object distance, scaled
        # at the respawn and never re-based, so an episode gives at most max_ftip_return
        if self.max_ftip_return <= 0.0:
            return 0.0
        xpos = wp.to_torch(self.gpu_data.xpos)
        gap = (xpos[:, self._tip_bids]
               - xpos[:, self._obj_bid, None]).norm(dim=2).sum(dim=1)
        gain, self._prev_tip = self._step_gain(self._prev_tip, gap, ~respawned)
        reward = gain / self._tip0 * self.max_ftip_return
        self._tip0 = torch.where(respawned, gap.clamp(min=1e-6), self._tip0)
        return reward

    def _object_lost(self) -> torch.Tensor:
        return self._obj_off_w[:, :2].norm(dim=1) > self.termination_dist

    def _goal_frame_origin(self, palm_pos: np.ndarray, palm_quat: np.ndarray,
                           obj_pos: np.ndarray) -> np.ndarray:
        # At rel_pos on the palm, for a goal that is a position too
        if not self.train_rel_pos:
            return obj_pos
        return palm_pos + _quat_mat(palm_quat) @ self.rel_pos

    @property
    def table_top(self) -> float:
        """World z of the table's top face."""
        return self._table_top
