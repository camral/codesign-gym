"""Panel cleaner: a legged robot with an arm and roller cleans an array of solar
panels, with a different body design in every world.

Every world runs the same scene: a legged base MJCF carrying an arm MJCF with a
cleaning roller on its tip, in front of an array of tilted solar panels. The
policy drives the robot's actuators. The panels' cells are goals visited in a
fixed order, and a cell is cleaned once the roller has rested on it for
``panel_dwell_steps`` steps. Reward is given for closing the distance to the
current goal and for every step spent cleaning. Each goal has its own step
budget, and running it out ends the episode. Optional extras are midpoints
between panels, new terrain every episode, a push on the base and leg noise.

Code layout:

- panel_cleaner_env.py: scene assembly, buffer and kernel-argument setup, spaces,
observation, the sweep and its reward, live tunables, and rendering;
- codesign_env.py: the co-design wrapper;
- panel_clean_kernels.py: the device work, per step and per reset;
- panel.py: the panel array, the visiting orders and the goal marker;
- tool.py: the roller;
- offsettable_logic.py: the link naming convention, the link deform, and the
spawn lift;
- parallel_env.py: the batched base class and its step program.

``PanelCleanParallelEnv`` is the native batched env, and
``PanelCleanerCoDesignEnv`` in codesign_env.py puts the co-design contract on top of it.
"""

from __future__ import annotations

import math
import os
import warnings

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from . import panel_clean_kernels as kern
from .offsettable_logic import (DeformArrays, SpawnLift, build_seg_arrays, find_segments,
                                length_safe_lo, release_geom_frames,
                                floor_reports, n_slots, slot_bounds)
from .panel import (RGBA_CELL, RGBA_CLEAN, MARKER_BODY, PanelGrid, add_goal_marker,
                    add_panels, add_midpoints)
from .parallel_env import ParallelMujocoEnv
from .tool import ROLLER_BODY, add_roller

_ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

# Margin of switched-off leg geoms: far below any distance in the scene, so they
# never make contact (see panel_clean_kernels._set_legs)
_LEG_OFF_MARGIN = -1.0e3

_CAMERA_MODES = ("fixed", "track")


class PanelCleanParallelEnv(ParallelMujocoEnv):
    """Batched panel-cleaning env with one body design per world.

    Besides the constructor settings, an instance holds:

    - model ids and addresses cached at setup, and the goal table of the sweep
    (sites, cell geoms, acceptance boxes, step budgets);
    - the embodiment tables: offsettable links and switchable leg geoms;
    - per-world device buffers the kernels hold warp views of, packed once into
    the kernel argument structs, including the roller and goal state that
    ``goal_kernel`` writes;
    - terrain buffers when terrain is on;
    - host-side episode state for the reward, and ``last_*`` outcome flags from
    the latest step.
    """

    _length_floor_warned = False      # see _warn_length_floor

    BASE_BODY = "base"

    def __init__(
        self,
        num_envs: int = 4,
        *,
        # arm_xml is attached to ant_xml's BASE_BODY, the result to scene_xml.
        # The XMLs' timestep is replaced by ``timestep``
        ant_xml: str = "panel_clean/dog.xml",
        arm_xml: str = "panel_clean/ant_arm.xml",
        scene_xml: str = "_common/scene_ant.xml",
        # Root bodies of the two leg chains that disable_legs removes (model specific)
        leg_chain_a: str | None = None,
        leg_chain_b: str | None = None,
        # Keyframes combined into the reset pose (missing names are skipped)
        init_key: str | tuple[str, ...] | list[str] = ("init_pos", "init_pos2"),
        max_offset: float = 0.1,                  # metres per unit of link embodiment action
        min_link_length: float = 0.02,            # deformed links are clamped to this
        min_length_pad: float = 0.02,             # safety margin above min_link_length for length_safe_lo
        rand_pos_range: float = 0.0,              # spawn xy jitter, half-side of a square [m]
        rand_heading_range: float = 0.0,          # spawn yaw jitter, +/- [deg]
        spawn_offset: float = 0.0,                # moves the spawn back along -x
        max_return: float = 10.0,
        max_goal_return: float = 7.5,             # share of max_return given for cleaning
        panel_dwell_steps: int = 1,               # total steps on a cell that clean it
        next_cell_step_limit: int = 100,          # step budget of a goal on the same panel
        next_panel_step_limit: int = 400,         # step budget of a goal on a new panel
        # How a run-out budget is reported. False terminates, since bootstrapping
        # through this clock rewards stalling. Set time_in_obs with either
        truncate_at_time_limit: bool = False,
        roller_radius: float = 0.05,
        roller_width: float = 0.25,
        roller_mass: float = 0.4,
        roller_solref=(0.015, 1.0),
        roller_friction=(0.05, 0.005, 0.0001),
        roller_align_tol: float = 0.5,            # max |dot(roller_y, normal)| for cell cleaning (~30 deg)
        goal_surface_tol: float = 0.02,           # roller height tolerance above a cell
        midpoints: bool = True,
        midpoint_tol: float = 0.3,                # half-extent of the box a midpoint is reached in
        midpoint_y_offset: float = 0.3,           # see panel.add_midpoints
        disable_legs: bool = False,               # adds the leg-switch embodiment entry
        terminate_out_of_field: bool = True,      # clear only for recording video
        field_pos: list[float] = [1.5, 0.0, 2.0],
        field_size: list[float] = [4.0, 3.0, 2.5],
        panel_rows: int = 1,
        panel_cols: int = 1,
        first_panel_pos: list[float] = [1.6, 0.0, 0.0],   # near panel's foot, centred in y
        panel_length: float = 1.0,
        panel_width: float = 2,
        panel_height: float = 1.5,
        panel_tilt_degs: float = 20.0,
        n_sensor_rows: int = 2,                   # cells per panel
        n_sensor_cols: int = 2,
        panel_thickness: float = 0.06,
        panel_gap_x: float = 1.0,
        panel_gap_y: float = 0.0,
        panel_traversal: str = "horizontal",      # see panel.TRAVERSALS
        collide_roller_only: bool = True,         # else the panels collide with the whole robot
        # End with body_contact_penalty when anything but the roller presses on a
        # panel deeper than body_contact_depth. Needs collide_roller_only=False
        terminate_on_body_contact: bool = False,
        body_contact_penalty: float = -1.0,
        body_contact_depth: float = 0.005,
        # New heightfield per world and episode over terrain_region ((x0, x1),
        # (y0, y1)), between the terrain_height (base, peak). Fractal value noise:
        # terrain_octaves octaves, each terrain_lacunarity times finer and
        # terrain_roughness times weaker, the first with terrain_base_freq features
        terrain: bool = False,
        terrain_region=((-0.6, 2.8), (-2.6, 2.6)),
        terrain_height=(0.0, 0.12),
        terrain_res=(28, 28),
        terrain_octaves: int = 4,
        terrain_base_freq: float = 3.0,
        terrain_lacunarity: float = 2.0,
        terrain_roughness: float = 0.5,
        terrain_contrast: float = 1.0,            # stretches heights about the mean
        terrain_seed: int = 0,
        terminate_off_green: bool = False,        # end when the base leaves the terrain patch
        # End when the base's height above the ground under it drops below this
        # fraction of the design's standing height at spawn. 0 turns it off
        min_base_height_frac: float = 0.0,
        camera_mode: str = "fixed",               # see _setup_camera
        phys_substeps: int = 8,
        max_rel_dist: float = 1.75,               # obs scale of the roller-goal vector
        vel_scale: float = 10.0,                  # obs scale of velocities
        time_in_obs: bool = True,                 # observe the share of the goal's budget left
        # Random horizontal push on the base, held for force_hold of every
        # force_every control steps
        force_mag: float = 0.0,
        force_every: int = 0,
        force_hold: int = 0,
        force_viz_scale: float = 0.02,
        leg_noise_mag: float = 0.0,               # random force on every leg body each step [N]
        timestep: float = 0.005,
        # parallel-runner params
        device: str = "cuda",
        viz_env_id: int = 0,
        render_mode: str | None = None,
        debug: bool = True,
        use_graph: bool = True,
        check_finite: bool = False,
        # None: sized from the scene, see below
        njmax: int | None = None,
        nconmax: int | None = None,
        naconmax: int | None = None,
        seed: int = 1234,
    ):
        self._xmls = (ant_xml, arm_xml, scene_xml)
        self.leg_chain_a = leg_chain_a
        self.leg_chain_b = leg_chain_b
        self.disable_legs = bool(disable_legs)
        if self.disable_legs and (leg_chain_a is None or leg_chain_b is None):
            raise ValueError("disable_legs needs leg_chain_a and leg_chain_b: "
                             "which chains to switch off is model specific")
        self._roller = dict(radius=roller_radius, width=roller_width,
                            mass=roller_mass, solref=tuple(roller_solref),
                            friction=tuple(roller_friction))
        self._panels = dict(
            rows=panel_rows, cols=panel_cols, first_pos=tuple(first_panel_pos),
            length=panel_length, width=panel_width, height=panel_height,
            tilt_degs=panel_tilt_degs, n_sensor_rows=n_sensor_rows,
            n_sensor_cols=n_sensor_cols, thickness=panel_thickness,
            gap_x=panel_gap_x, gap_y=panel_gap_y,
            collide_roller_only=collide_roller_only)
        self.panel_traversal = str(panel_traversal)
        if camera_mode not in _CAMERA_MODES:
            raise ValueError(f"unknown camera_mode {camera_mode!r}; "
                             f"have {sorted(_CAMERA_MODES)}")
        self.camera_mode = str(camera_mode)
        self.roller_radius = float(roller_radius)
        self.roller_align_tol = float(roller_align_tol)
        self.goal_surface_tol = float(goal_surface_tol)
        self.midpoints = bool(midpoints)
        self.midpoint_tol = float(midpoint_tol)
        self.midpoint_y_offset = float(midpoint_y_offset)
        if self.midpoints and self.midpoint_tol <= 0.0:
            raise ValueError("midpoint_tol is the half-extent of the box a "
                             "midpoint is reached in and must be positive; a "
                             "midpoint the roller cannot reach stops the sweep")
        self.terminate_out_of_field = bool(terminate_out_of_field)
        self.spawn_offset = float(spawn_offset)
        if self.spawn_offset < 0.0:
            raise ValueError("spawn_offset sets the robot back from the panels "
                             "along -x and cannot be negative")
        self.init_key = ((init_key,) if isinstance(init_key, str)
                         else tuple(init_key))
        self.max_offset = float(max_offset)
        self.min_link_length = float(min_link_length)
        self.min_length_pad = float(min_length_pad)
        if self.min_link_length <= 0.0:
            raise ValueError("min_link_length is a length floor and must be "
                             "positive; a link at zero length has no mass, no "
                             "inertia and no bounding volume")
        self.rand_pos_range = float(rand_pos_range)
        self.rand_heading_range = float(rand_heading_range)
        if min(self.rand_pos_range, self.rand_heading_range) < 0.0:
            raise ValueError("rand_pos_range and rand_heading_range are "
                             "half-extents and cannot be negative")
        self.max_return = float(max_return)
        self.max_goal_return = float(max_goal_return)
        self.panel_dwell_steps = int(panel_dwell_steps)
        self.next_cell_step_limit = int(next_cell_step_limit)
        self.next_panel_step_limit = int(next_panel_step_limit)
        if min(self.next_cell_step_limit, self.next_panel_step_limit) < 1:
            raise ValueError("next_cell_step_limit and next_panel_step_limit are "
                             "control-step budgets and must be at least 1")
        self.truncate_at_time_limit = bool(truncate_at_time_limit)
        self.field_pos = np.asarray(field_pos, dtype=np.float32)
        self.field_size = np.asarray(field_size, dtype=np.float32)
        # The spawn moved back, so widen the field's -x side to keep the roller inside
        if self.spawn_offset > 0.0:
            self.field_pos[0] -= 0.5 * self.spawn_offset
            self.field_size[0] += 0.5 * self.spawn_offset
        if terrain:
            # Panels sit on the terrain, so raise the field's top by the peak height
            peak = float(terrain_height[1]) - float(terrain_height[0])
            self.field_pos[2] += 0.5 * peak
            self.field_size[2] += 0.5 * peak
        self.max_rel_dist = float(max_rel_dist)
        self.vel_scale = float(vel_scale)
        self.time_in_obs = bool(time_in_obs)
        self.force_mag = float(force_mag)
        self.force_every = int(force_every)
        self.force_hold = int(force_hold)
        self.force_viz_scale = float(force_viz_scale)
        self.leg_noise_mag = float(leg_noise_mag)
        self.force_on = self.force_every > 0 and self.force_hold > 0
        self.timestep = float(timestep)
        self.check_finite = bool(check_finite)

        self.collide_roller_only = bool(collide_roller_only)
        self.terminate_on_body_contact = bool(terminate_on_body_contact)
        self.body_contact_penalty = float(body_contact_penalty)
        self.body_contact_depth = float(body_contact_depth)
        self.terrain = bool(terrain)
        self.terrain_region = terrain_region
        # The spawn moved back, so extend the terrain patch's -x edge too
        if self.terrain and self.spawn_offset > 0.0:
            (tx0, tx1), tyr = self.terrain_region
            self.terrain_region = ((tx0 - self.spawn_offset, tx1), tyr)
        self.terrain_height = terrain_height
        self.terminate_off_green = bool(terminate_off_green)
        self.min_base_height_frac = float(min_base_height_frac)
        self.terrain_res = terrain_res
        self.terrain_octaves = int(terrain_octaves)
        self.terrain_base_freq = float(terrain_base_freq)
        self.terrain_lacunarity = float(terrain_lacunarity)
        self.terrain_roughness = float(terrain_roughness)
        self.terrain_contrast = float(terrain_contrast)
        self.terrain_seed = int(terrain_seed)
        if self.terminate_on_body_contact and self.collide_roller_only:
            raise ValueError(
                "terminate_on_body_contact needs the panels to collide with the "
                "whole robot to have any contact to detect; set "
                "collide_roller_only=False")

        # Contacts per world: a single robot only contacts the ground/terrain and
        # 1-2 panels simultaneously regardless of total array size.
        per_world_con = (64
                         + (0 if collide_roller_only else 16)
                         + (32 if terrain else 0))
        nconmax = int(nconmax) if nconmax is not None else per_world_con
        njmax = int(njmax) if njmax is not None else max(128, 3 * nconmax)
        naconmax = (int(naconmax) if naconmax is not None
                    else nconmax * int(num_envs))

        super().__init__(
            num_envs, device=device, viz_env_id=viz_env_id,
            max_episode_steps=0, debug=debug,          # set from the goal budgets later
            seed=seed, phys_substeps=phys_substeps, render_mode=render_mode,
            use_graph=use_graph, njmax=njmax, nconmax=nconmax, naconmax=naconmax)

        self._cache_model_addresses()
        self._alloc_device_state()
        self._build_kernel_params()
        self._make_spaces()
        self._build_spawn_lift()

        # Step program (captured as one CUDA graph on GPU):
        # pre-substep: RNG tick, clear foul flag, disturbances, ctrl
        # per-substep: accumulate non-roller panel contacts across all substeps
        # post-substep: goal_kernel on fresh kinematics/velocities
        self.register_step_launch(self._launch_tick)
        if self.terminate_on_body_contact:
            self.register_step_launch(self._launch_zero_bad_contact)
        if self.force_on:
            self.register_step_launch(self._launch_force)
        if self.leg_noise_mag > 0.0 and self._n_leg_noise > 0:
            self.register_step_launch(self._launch_leg_noise)
        self.register_step_launch(self._launch_ctrl)
        if self.terminate_on_body_contact:
            self.register_substep_launch(self._launch_body_contact_substep)
        self.register_post_step_launch(self._launch_goal_step)

    # ------------------------------------------------------------------ #
    # Model assembly
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        ant_xml, arm_xml, scene_xml = self._xmls
        arm = mujoco.MjSpec.from_file(os.path.join(_ASSETS, arm_xml))
        add_roller(arm, **self._roller)
        arm.compile()   # an attached child's keyframes are namespaced correctly only if compiled first
        ant = mujoco.MjSpec.from_file(os.path.join(_ASSETS, ant_xml))
        ant.attach(arm, prefix="", frame=ant.body(self.BASE_BODY).add_frame())
        ant.compile()

        spec = mujoco.MjSpec.from_file(os.path.join(_ASSETS, scene_xml))
        spec.attach(ant, prefix="", frame=spec.worldbody.add_frame())
        self.grid: PanelGrid = add_panels(spec, **self._panels)
        if self.midpoints:                     # before the marker, which needs the full goal list
            add_midpoints(spec, self.grid, self.panel_traversal,
                          y_offset=self.midpoint_y_offset)
        add_goal_marker(spec, half=self.grid.marker_half, quat=self.grid.tilt_quat)
        # Visual-only box marking the bounds the roller must stay inside (group 4: semantic/task)
        spec.worldbody.add_geom(
            name="field", type=mujoco.mjtGeom.mjGEOM_BOX, group=4,
            pos=self.field_pos.tolist(), size=self.field_size.tolist(),
            contype=0, conaffinity=0, density=0.0, mass=0.0,
            rgba=[0.3, 0.6, 1.0, 0.1])
        self._add_terrain(spec)
        spec.option.timestep = self.timestep
        return spec

    def _add_terrain(self, spec: mujoco.MjSpec) -> None:
        """Add the terrain heightfield geom over ``terrain_region``. The CPU model
        gets one hfield, the device model one per world (see
        ``_augment_gpu_asset_spec``). Heights start flat."""
        if not self.terrain:
            return
        nrow, ncol = int(self.terrain_res[0]), int(self.terrain_res[1])
        (x0, x1), (y0, y1) = self.terrain_region
        base_h, peak_h = float(self.terrain_height[0]), float(self.terrain_height[1])

        for (px, py, _pz) in self.grid.panel_pos.values():
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                raise ValueError(
                    f"panel foot ({px:.2f}, {py:.2f}) is outside terrain_region "
                    f"x{(x0, x1)} y{(y0, y1)}; with terrain on the array is sat "
                    "on the surface, so the region must contain every panel")

        self._add_terrain_asset(spec, 0)
        g = spec.worldbody.add_geom()
        g.name = "terrain_geom"
        g.type = mujoco.mjtGeom.mjGEOM_HFIELD
        g.hfieldname = "terrain_0"             # per world after upload, see _setup_terrain
        g.pos = [0.5 * (x0 + x1), 0.5 * (y0 + y1), base_h]
        g.contype, g.conaffinity = 1, 1
        g.friction = [1.0, 0.01, 0.001]
        g.solref = [0.02, 1.0]
        g.material = "terrain_mat"

    def _add_terrain_asset(self, spec: mujoco.MjSpec, world: int) -> None:
        """Add one flat terrain hfield asset."""
        nrow, ncol = int(self.terrain_res[0]), int(self.terrain_res[1])
        (x0, x1), (y0, y1) = self.terrain_region
        base_h, peak_h = float(self.terrain_height[0]), float(self.terrain_height[1])
        hf = spec.add_hfield()
        hf.name = f"terrain_{world}"
        hf.nrow, hf.ncol = nrow, ncol
        hf.size = [0.5 * (x1 - x0), 0.5 * (y1 - y0),
                   max(peak_h - base_h, 1e-3), 5.0]    # (radius x, radius y, max elevation, base depth)
        hf.userdata = [0.0] * (nrow * ncol)

    def _augment_gpu_asset_spec(self, asset_spec: mujoco.MjSpec) -> bool:
        """Give the device model one terrain hfield per world."""
        if not self.terrain or self.num_envs <= 1:
            return False
        for world in range(1, self.num_envs):
            self._add_terrain_asset(asset_spec, world)
        return True

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _cache_model_addresses(self) -> None:
        """Cache model ids and addresses, build the goal table of the sweep, find
        the embodiment parts, and size the embodiment action and reward."""
        m = self.m
        self._roller_bid = m.body(ROLLER_BODY).id
        self._root_bid = int(m.body_rootid[self._roller_bid])
        self._base_bid = m.body(self.BASE_BODY).id
        self._marker_bid = m.body(MARKER_BODY).id
        def _bodies_named(*keys):
            out = []
            for bid in range(m.nbody):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if any(k in nm for k in keys):
                    out.append(bid)
            return out
        leg_noise = _bodies_named("knee") or _bodies_named("leg", "ankle")   # leg-noise targets
        self._leg_noise_bids_np = np.asarray(leg_noise, dtype=np.int32)
        self._roller_gid = int(m.geom(f"{ROLLER_BODY}_geom").id)
        # Plates are found by name, not by contact bit, since the roller has the bit too
        plate_mask = np.zeros(m.ngeom, dtype=np.int32)
        for gi in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if gname and gname.endswith("_plate"):
                plate_mask[gi] = 1
        self._plate_mask_np = plate_mask
        self._base_qadr = -1
        for j in range(m.njnt):
            if (int(m.jnt_bodyid[j]) == self._base_bid
                    and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                self._base_qadr = int(m.jnt_qposadr[j])
                break
        if self._base_qadr < 0 and (self.rand_pos_range > 0.0
                                    or self.rand_heading_range > 0.0):
            raise ValueError(
                f"rand_pos_range/rand_heading_range need a free joint on "
                f"{self.BASE_BODY!r} to move it with; this model has none")
        if self._base_qadr < 0 and self.min_base_height_frac > 0.0:
            raise ValueError(
                f"min_base_height_frac needs a free joint on {self.BASE_BODY!r} "
                "to read the base height from; this model has none")
        order = self.grid.goals(self.panel_traversal)
        self._goal_sid = torch.as_tensor(
            [m.site(g.site).id for g in order], dtype=torch.int32, device=self.device)
        is_cell = np.array([not g.is_midpoint for g in order])
        self._goal_is_cell = torch.as_tensor(is_cell, device=self.device)
        # A midpoint borrows geom 0, never painted (see _paint_cleaned)
        self._goal_gid = torch.as_tensor(
            [m.geom(g.geom).id if g.geom is not None else 0 for g in order],
            dtype=torch.long, device=self.device)
        self._cell_gids = np.array([m.geom(g).id for g in self.grid.geoms],
                                   dtype=np.int64)
        self.n_goals = len(order)
        self.n_cells = int(is_cell.sum())
        # A goal on another panel than the previous goal (and the first goal) gets
        # the long budget. A midpoint's key is unique, so both legs of a
        # panel-to-panel walk do
        keys = [g.key for g in order]
        budget = [self.next_panel_step_limit if i == 0 or k != keys[i - 1]
                  else self.next_cell_step_limit
                  for i, k in enumerate(keys)]
        self._goal_budget = torch.as_tensor(budget, dtype=torch.long,
                                            device=self.device)
        # In gaps, so the observed clock is 1 on a goal's first step and 0 on its last
        self._goal_budget_span = (self._goal_budget - 1).clamp(min=1).float()
        # A cell is reached with the roller resting on its footprint, a midpoint
        # by passing through a box around it
        cell_tol = np.where(is_cell, self.goal_surface_tol, self.midpoint_tol)
        self._goal_hx = torch.as_tensor(
            np.where(is_cell, self.grid.half[0], self.midpoint_tol),
            dtype=torch.float32, device=self.device)
        self._goal_hy = torch.as_tensor(
            np.where(is_cell, self.grid.half[1], self.midpoint_tol),
            dtype=torch.float32, device=self.device)
        self._goal_tol = torch.as_tensor(cell_tol, dtype=torch.float32,
                                         device=self.device)
        self._goal_standoff = torch.as_tensor(
            np.where(is_cell, self.roller_radius, 0.0),
            dtype=torch.float32, device=self.device)
        self._goal_dwell = torch.as_tensor(
            np.where(is_cell, max(self.panel_dwell_steps, 1), 1),
            dtype=torch.long, device=self.device)
        # Longest possible episode, for callers sizing rollouts. The per-goal
        # budgets always end an episode first
        self.max_episode_steps = int(sum(budget)) * self.phys_substeps
        self._cell_half = self.grid.half
        self._marker_lift = self.grid.marker_half[2]

        self._check_goals_in_field(m, order)

        self._segs = find_segments(m, self.min_link_length)
        self._raw_axis_lo = self._segs["axis_lo"].copy()
        self._segs["axis_lo"] = length_safe_lo(
            self._segs, self._raw_axis_lo, self.max_offset,
            self.min_link_length + self.min_length_pad)
        self._leg_geoms = self._find_legs(m)
        self.n_links = len(self._segs["body"])
        self.n_link_slots = n_slots(self._segs)
        self.n_leg_params = 1 if self.disable_legs else 0
        self.n_offsets = self.n_link_slots + self.n_leg_params
        self._init_qpos = self._reset_pose(m)
        if self.spawn_offset > 0.0 and self._base_qadr >= 0:
            self._init_qpos[self._base_qadr] -= self.spawn_offset
        # Written per world: sites on moved link ends, and midpoints following the terrain
        if len(self._segs["site_id"]) or (self.terrain and self.midpoints):
            self._make_field_per_world("site_pos")
        self._warn_length_floor()
        if self.disable_legs:
            self._make_field_per_world("geom_margin")

        # max_goal_return is split over the dwell steps of all cells, the rest of
        # max_return over closing the distance to every goal, midpoints included
        self.at_goal_rew = (
            self.max_goal_return / (self.n_cells * self.panel_dwell_steps)
            if self.panel_dwell_steps > 0 else 0.0)
        self.prog_gain = (self.max_return - self.at_goal_rew
                          * self.n_cells * self.panel_dwell_steps) / self.n_goals

        self._setup_terrain()

    def _setup_terrain(self) -> None:
        """Point each world's terrain geom at its own hfield, and build the
        buffers the terrain kernels read. Does nothing if terrain is off."""
        self._terrain_gid = -1
        if not self.terrain:
            return
        m = self.m
        self._terrain_gid = int(m.geom("terrain_geom").id)
        self._terrain_base_h = float(m.geom_pos[self._terrain_gid][2])
        self._make_field_per_world("geom_dataid")
        gd = wp.to_torch(self.gpu_model.geom_dataid)
        hfid = torch.arange(self.num_envs, dtype=gd.dtype, device=self.device)
        gd[:, self._terrain_gid] = hfid        # world w draws hfield w
        (x0, x1), (y0, y1) = self.terrain_region
        self._green_lo = torch.tensor([x0, y0], dtype=torch.float32, device=self.device)
        self._green_hi = torch.tensor([x1, y1], dtype=torch.float32, device=self.device)
        # For sampling world w's hfield (its own, see above) under the base
        self._hf_adr = wp.to_torch(self.gpu_model.hfield_adr)[:self.num_envs].long()
        self._hf_shape = (int(self.terrain_res[0]), int(self.terrain_res[1]))
        self._hf_scale = float(wp.to_torch(self.gpu_model.hfield_size)[0, 2])
        length = float(self._panels["length"])
        width = float(self._panels["width"])
        tilt = math.radians(float(self._panels["tilt_degs"]))
        stub = length / 5.0
        # Panel supports in the terrain patch: the column sits elbow_x ahead of
        # the foot, and the sampling window covers its footprint
        elbow_x = math.sin(tilt) * stub
        nr, nc = int(self.terrain_res[0]), int(self.terrain_res[1])
        cell_x = (x1 - x0) / max(nc - 1, 1)
        cell_y = (y1 - y0) / max(nr - 1, 1)
        self._panel_half_win = max(
            1, int(math.ceil(max(stub / max(cell_x, 1e-9),
                                 (width / 5.0) / max(cell_y, 1e-9)))))
        pbid, puv = [], []
        for (r, c) in self.grid.panel_pos:
            px, py, _pz = self.grid.panel_pos[(r, c)]
            pbid.append(int(m.body(f"panel_r{r}_c{c}").id))
            puv.append(((px + elbow_x - x0) / max(x1 - x0, 1e-9),
                        (py - y0) / max(y1 - y0, 1e-9)))
        self._panel_bid_wp = wp.array(np.asarray(pbid, dtype=np.int32),
                                      dtype=wp.int32, device=self.device)
        self._panel_uv_wp = wp.array(np.asarray(puv, dtype=np.float32),
                                     dtype=wp.vec2, device=self.device)
        self._n_terrain_panels = len(pbid)
        # A midpoint's z is its compiled z plus the mean height shift of its two panels
        order = self.grid.goals(self.panel_traversal)
        mid = [(j, g) for j, g in enumerate(order) if g.is_midpoint]
        pz = lambda key: float(m.body_pos[m.body(f"panel_r{key[0]}_c{key[1]}").id][2])
        pb = lambda key: int(m.body(f"panel_r{key[0]}_c{key[1]}").id)
        self._n_mid = len(mid)
        self._mid_sid_wp = wp.array(np.asarray([m.site(g.site).id for _, g in mid], dtype=np.int32), dtype=wp.int32, device=self.device)
        self._mid_bid_wp = wp.array(np.asarray([(pb(order[j - 1].key), pb(order[j + 1].key)) for j, _ in mid], dtype=np.int32).reshape(-1, 2),
                                    dtype=wp.int32, device=self.device)
        self._mid_z0_wp = wp.array(np.asarray([float(m.site_pos[m.site(g.site).id][2]) - 0.5 * (pz(order[j - 1].key) + pz(order[j + 1].key))
                                               for j, g in mid], dtype=np.float32), dtype=wp.float32, device=self.device)
        self._ep_count = torch.zeros(self.num_envs, dtype=torch.int32,   # episode count, the terrain seed
                                     device=self.device)
        self._ep_count_wp = wp.from_torch(self._ep_count)
        self._terrain_base_seed = int(self.terrain_seed)   # added to each world's seed

    def _launch_terrain(self) -> None:
        """Regenerate the terrain of the worlds resetting this step, and re-seat
        the robot, panels and midpoints on it."""
        nr, nc = int(self.terrain_res[0]), int(self.terrain_res[1])
        wp.launch(kern.ep_increment_kernel, dim=self.num_envs,
                  inputs=[self._env.mask, self._ep_count_wp])
        wp.launch(kern.terrain_kernel, dim=(self.num_envs, nr * nc),
                  inputs=[self._env.mask, self._ep_count_wp,
                          self.gpu_model.hfield_data, self.gpu_model.hfield_adr,
                          self.gpu_model.hfield_nrow, self.gpu_model.hfield_ncol,
                          self.terrain_octaves, self.terrain_base_freq,
                          self.terrain_lacunarity, self.terrain_roughness,
                          self.terrain_contrast, self._terrain_base_seed, self._env.lane_seed])
        if self._lift.enabled:
            (x0, x1), (y0, y1) = self.terrain_region
            # Seat the robot at its drawn pose, 2 cm clear (slack for sampling at geom centres)
            mjw.kinematics(self.gpu_model, self.gpu_data)
            wp.launch(kern.terrain_lift_kernel, dim=self.num_envs,
                      inputs=[self._lift.la, self._lift.zadr, self.gpu_model.hfield_data,
                              self.gpu_model.hfield_adr, self.gpu_model.hfield_nrow,
                              self.gpu_model.hfield_ncol, self.gpu_model.hfield_size,
                              float(x0), float(y0), float(x1 - x0), float(y1 - y0),
                              self._terrain_base_h, 0.02])
        wp.launch(kern.panel_height_kernel,
                  dim=(self.num_envs, self._n_terrain_panels),
                  inputs=[self._env.mask, self.gpu_model.hfield_data,
                          self.gpu_model.hfield_adr, self.gpu_model.hfield_nrow,
                          self.gpu_model.hfield_ncol, self.gpu_model.hfield_size,
                          self._panel_bid_wp, self._panel_uv_wp,
                          self._terrain_base_h, self._panel_half_win,
                          self.gpu_model.body_pos])
        if self._n_mid:
            wp.launch(kern.midpoint_height_kernel, dim=(self.num_envs, self._n_mid),
                      inputs=[self._env.mask, self._mid_sid_wp, self._mid_bid_wp, self._mid_z0_wp,
                              self.gpu_model.body_pos, self.gpu_model.site_pos])

    def _check_goals_in_field(self, m: mujoco.MjModel, order) -> None:
        """Raise if a goal's aim point is outside the field box (warn without
        ``terminate_out_of_field``). The roller would leave the field reaching for
        it, so the sweep could never pass it."""
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        sid = self._goal_sid.cpu().numpy()
        normal = d.site_xmat[sid].reshape(-1, 3, 3)[:, :, 2]
        aim = (d.site_xpos[sid]
               + normal * self._goal_standoff.cpu().numpy()[:, None])
        lo, hi = self.field_pos - self.field_size, self.field_pos + self.field_size
        outside = np.flatnonzero(~np.all((aim >= lo) & (aim <= hi), axis=1))
        if outside.size == 0:
            return
        need_lo = np.minimum(lo, aim.min(axis=0))
        need_hi = np.maximum(hi, aim.max(axis=0))
        detail = (
            f"{outside.size} of {len(order)} goals of the sweep sit outside the "
            f"field box: goals {outside[:8].tolist()}"
            f"{' ...' if outside.size > 8 else ''}, the first at "
            f"{np.round(aim[outside[0]], 3).tolist()}.\n"
            f"  field spans x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
            f"z[{lo[2]:.2f},{hi[2]:.2f}]; the cells need "
            f"x[{aim[:,0].min():.2f},{aim[:,0].max():.2f}] "
            f"y[{aim[:,1].min():.2f},{aim[:,1].max():.2f}] "
            f"z[{aim[:,2].min():.2f},{aim[:,2].max():.2f}].\n"
            f"  A box that contains both is field_pos="
            f"{np.round((need_lo + need_hi) / 2, 2).tolist()}, field_size="
            f"{np.round((need_hi - need_lo) / 2, 2).tolist()} (add margin: the "
            f"roller has to approach the cell, not just touch it).")
        if self.terminate_out_of_field:
            raise ValueError(
                "terminate_out_of_field is set and " + detail
                + "\n  As configured the episode ends the moment the roller "
                  "reaches for such a cell, so the sweep can never get past it.")
        warnings.warn("field box does not contain the whole sweep: " + detail,
                      RuntimeWarning, stacklevel=2)

    def _warn_length_floor(self) -> None:
        """Warn about links ``max_offset`` can drive into their length floor, once
        per process, since the floor can be intended."""
        hit = floor_reports(self.m, self._segs, self.max_offset)
        if hit and not PanelCleanParallelEnv._length_floor_warned:
            PanelCleanParallelEnv._length_floor_warned = True
            warnings.warn(
                f"max_offset={self.max_offset} can shorten "
                f"{len(hit)} link(s) past min_link_length="
                f"{self.min_link_length}, where the deform clamps them:\n  "
                + "\n  ".join(hit), RuntimeWarning, stacklevel=3)

    def _find_legs(self, m: mujoco.MjModel) -> np.ndarray:
        """Every geom in the two leg chains. Empty when ``disable_legs`` is off,
        so a model without those bodies still builds."""
        if not self.disable_legs:
            return np.zeros(0, dtype=np.int64)
        chain = {m.body(name).id for name in (self.leg_chain_a, self.leg_chain_b)}
        for b in range(m.nbody):                 # a parent always precedes its children
            if int(m.body_parentid[b]) in chain:
                chain.add(b)
        geoms = [g for b in sorted(chain)
                 for g in range(int(m.body_geomadr[b]),
                                int(m.body_geomadr[b]) + int(m.body_geomnum[b]))]
        if not geoms:
            raise RuntimeError(f"the {self.leg_chain_a!r}/{self.leg_chain_b!r} chains carry "
                               "no geoms; disable_legs has nothing to switch off")
        return np.array(geoms, dtype=np.int64)

    def _reset_pose(self, m: mujoco.MjModel) -> np.ndarray:
        """qpos0 with every ``init_key`` keyframe applied.

        Attaching fills a keyframe with qpos0 for the joints its model does not
        own, so each keyframe's difference from qpos0 is exactly its own part,
        and the keyframes combine by adding these differences. Two keyframes
        setting the same joint raise.
        """
        keys = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i)
                for i in range(m.nkey)]
        qpos, claimed = np.array(m.qpos0, dtype=np.float64), np.zeros(m.nq, bool)
        for name in self.init_key:
            if name not in keys:
                continue
            delta = m.key_qpos[keys.index(name)] - m.qpos0
            mine = np.abs(delta) > 1e-9
            if (mine & claimed).any():
                raise RuntimeError(
                    f"keyframe {name!r} rewrites qpos "
                    f"{np.flatnonzero(mine & claimed).tolist()}, which an earlier "
                    f"keyframe in {list(self.init_key)} already set")
            qpos, claimed = qpos + delta, claimed | mine
        return qpos.astype(np.float32)

    def _alloc_device_state(self) -> None:
        # The kernel argument structs hold warp views of most of these buffers,
        # so those must be written in place, never rebound
        n, dev = self.num_envs, self.device
        self._alloc_action_buffers(self.act_dim(), self.n_offsets)
        self._offsets_t = torch.zeros((n, self.n_offsets), device=dev)   # applied embodiment, normalised

        self._plate_mask_wp = wp.array(self._plate_mask_np, dtype=wp.int32, device=dev)
        self._bad_contact = torch.zeros(n, dtype=torch.int32, device=dev)   # allocated even when unused
        self._bad_contact_wp = wp.from_torch(self._bad_contact)

        self._leg_noise_bids_wp = wp.array(self._leg_noise_bids_np, dtype=wp.int32,
                                           device=dev)
        self._n_leg_noise = int(len(self._leg_noise_bids_np))

        self._force_phase = torch.zeros(n, dtype=torch.int32, device=dev)
        self._force_cycle = torch.zeros(n, dtype=torch.int32, device=dev)
        self._force_dir = torch.zeros((n, 2), dtype=torch.float32, device=dev)

        # Each respawn draws from its lane seed hashed with the tick
        self._tick = torch.zeros(1, dtype=torch.int32, device=dev)
        self._lane_seed = torch.full((n,), int(self.seed), dtype=torch.int32, device=dev)
        self._ep_seed = torch.zeros(n, dtype=torch.int32, device=dev)
        self._tunables = torch.zeros(kern.N_TUNABLES, device=dev)
        self._wp_tunables = wp.from_torch(self._tunables)
        self._qpos_init = torch.as_tensor(self._init_qpos, device=dev)

        # Read and written in place by goal_kernel
        self._goal_idx = torch.zeros(n, dtype=torch.int32, device=dev)
        self._goal_gid_i32 = self._goal_gid.to(torch.int32)
        self._roller_pos = torch.zeros((n, 3), device=dev)
        self._roller_vel = torch.zeros((n, 3), device=dev)
        self._roller_vel_local = torch.zeros((n, 3), device=dev)
        self._roller_quat = torch.zeros((n, 4), device=dev)
        self._goal_pt = torch.zeros((n, 3), device=dev)
        self._goal_normal = torch.zeros((n, 3), device=dev)
        self._roller_rel_local = torch.zeros((n, 3), device=dev)
        self._goal_dist = torch.zeros(n, device=dev)
        self._at_goal_t = torch.zeros(n, dtype=torch.bool, device=dev)
        self._left_field_t = torch.zeros(n, dtype=torch.bool, device=dev)

        # Host-side episode state - no kernel views these, so they may be rebound
        self._dwell = torch.zeros(n, dtype=torch.long, device=dev)
        self._cell_steps = torch.zeros(n, dtype=torch.long, device=dev)   # steps spent on the current goal
        self._rebase = torch.ones(n, dtype=torch.bool, device=dev)        # goal changed, re-base the progress reward
        self._prev_dist = torch.zeros(n, device=dev)                       # last step's roller-goal distance
        self._d0 = torch.ones(n, device=dev)                               # distance at the start of the goal
        self._diverged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._diverged_total = torch.zeros((), dtype=torch.long, device=dev)
        self._spawn_h = torch.zeros(n, device=dev)   # base height of the design standing at spawn

        # Obs and reward constants
        self._arange = torch.arange(n, device=dev)
        self._clean_rgba = torch.as_tensor(RGBA_CLEAN, device=dev)
        self._fpos = torch.as_tensor(self.field_pos, device=dev)
        self._fsize = torch.as_tensor(self.field_size, device=dev)
        off, scl = self._qpos_norm()
        self._qpos_off = torch.as_tensor(off, device=dev)
        self._qpos_scl = torch.as_tensor(scl, device=dev)

    def _qpos_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """(offset, scale) mapping qpos into about [-1, 1]: free-joint positions by
        the field box, quaternions unchanged, limited joints by range. All joints
        belong to the robot, since the panels are static."""
        m = self.m
        off = np.zeros(m.nq, dtype=np.float32)
        scl = np.ones(m.nq, dtype=np.float32)
        for j in range(m.njnt):
            adr = int(m.jnt_qposadr[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                off[adr:adr + 3], scl[adr:adr + 3] = self.field_pos, 1.0 / self.field_size
            elif m.jnt_limited[j]:
                scl[adr] = 1.0 / max(np.abs(m.jnt_range[j]).max(), 1e-6)
        return off, scl

    def _build_kernel_params(self) -> None:
        sim = kern.SimArrays()
        for attr in ("qpos", "qvel", "qacc", "qacc_warmstart", "time", "ctrl",
                     "xfrc_applied", "xpos", "xquat", "cvel", "subtree_com",
                     "site_xpos", "site_xmat"):
            setattr(sim, attr, getattr(self.gpu_data, attr))
        for attr in ("geom_margin", "geom_rgba", "body_pos"):
            setattr(sim, attr, getattr(self.gpu_model, attr))
        self._sim = sim

        dfm = DeformArrays()
        for attr in ("geom_pos", "geom_quat", "geom_size", "geom_rbound",
                     "geom_aabb", "body_pos", "body_ipos", "body_iquat",
                     "body_mass", "body_inertia", "site_pos"):
            setattr(dfm, attr, getattr(self.gpu_model, attr))
        self._dfm = dfm

        env = kern.EnvArrays()
        env.mask = wp.from_torch(self.reset_mask)
        env.qpos_init = wp.from_torch(self._qpos_init)
        env.deform = wp.from_torch(self._reset_act_buf)
        env.ctrl_act = wp.from_torch(self._act_buf)
        env.ctrl_scale = wp.from_torch(self.ctrl_scale)
        env.ctrl_mid = wp.from_torch(self.ctrl_mid)
        env.offsets = wp.from_torch(self._offsets_t)
        env.force_phase = wp.from_torch(self._force_phase)
        env.force_cycle = wp.from_torch(self._force_cycle)
        env.force_dir = wp.from_torch(self._force_dir)
        env.tick = wp.from_torch(self._tick)
        env.lane_seed = wp.from_torch(self._lane_seed)
        env.ep_seed = wp.from_torch(self._ep_seed)
        self._env = env

        self._seg = build_seg_arrays(self._segs, self.device)

        goal = kern.GoalArrays()
        goal.site = wp.from_torch(self._goal_sid)
        goal.idx = wp.from_torch(self._goal_idx)
        goal.hx = wp.from_torch(self._goal_hx)
        goal.hy = wp.from_torch(self._goal_hy)
        goal.tol = wp.from_torch(self._goal_tol)
        goal.standoff = wp.from_torch(self._goal_standoff)
        goal.roller_pos = wp.from_torch(self._roller_pos, dtype=wp.vec3)
        goal.roller_vel = wp.from_torch(self._roller_vel, dtype=wp.vec3)
        goal.roller_vel_local = wp.from_torch(self._roller_vel_local, dtype=wp.vec3)
        goal.roller_quat = wp.from_torch(self._roller_quat, dtype=wp.quat)
        goal.pos = wp.from_torch(self._goal_pt, dtype=wp.vec3)
        goal.normal = wp.from_torch(self._goal_normal, dtype=wp.vec3)
        goal.rel_local = wp.from_torch(self._roller_rel_local, dtype=wp.vec3)
        goal.dist = wp.from_torch(self._goal_dist)
        goal.at_goal = wp.from_torch(self._at_goal_t)
        goal.left_field = wp.from_torch(self._left_field_t)
        self._goal = goal

        pan = kern.PanelArrays()
        pan.cell = wp.array(self._cell_gids.astype(np.int32), dtype=wp.int32,
                            device=self.device)
        pan.rgba0 = wp.vec4(*RGBA_CELL)
        self._pan = pan

        # Built even without disable_legs, so the kernel never gets an unbound
        # array. geom_margin is per world only under disable_legs, so kernels may
        # index it by world only when c.n_legparam > 0
        legs = self._leg_geoms
        leg = kern.LegArrays()
        leg.geom = wp.array(legs.astype(np.int32), dtype=wp.int32, device=self.device)
        leg.margin0 = wp.array(self.m.geom_margin[legs].astype(np.float32),
                               dtype=wp.float32, device=self.device)
        leg.rgba0 = wp.array(self.m.geom_rgba[legs].astype(np.float32),
                             dtype=wp.vec4, device=self.device)
        leg.off_margin, leg.n_geom = _LEG_OFF_MARGIN, len(legs)
        self._leg_arr = leg

        c = kern.EnvConst()
        c.seed = int(self.seed)
        c.nq, c.nv, c.nu = int(self.m.nq), int(self.m.nv), int(self.m.nu)
        c.n_seg = int(self.n_links)
        c.n_slot = int(self.n_link_slots)
        c.n_legparam = int(self.n_leg_params)
        c.n_cell = int(len(self._cell_gids))
        c.force_every = max(int(self.force_every), 1)
        c.force_hold = int(self.force_hold)
        c.base_bid = self._base_bid
        c.base_qadr = int(self._base_qadr)
        c.roller_bid, c.root_bid = self._roller_bid, self._root_bid
        c.marker_bid, c.n_goals = self._marker_bid, int(self.n_goals)
        c.marker_lift = float(self._marker_lift)
        c.roller_align_tol = float(self.roller_align_tol)
        c.field_pos = wp.vec3(*self.field_pos.tolist())
        c.field_size = wp.vec3(*self.field_size.tolist())
        self._consts = c

        self._refresh_tunables()

    def _refresh_tunables(self) -> None:
        # Order must match the T_* indices in panel_clean_kernels
        self._tunables.copy_(torch.tensor(
            [self.max_offset, self.force_mag, self.rand_pos_range,
             np.deg2rad(self.rand_heading_range)], dtype=torch.float32))

    def act_dim(self) -> int:
        # Control action only - the embodiment action is a separate step() argument
        return int(self.m.nu)

    def obs_dim(self) -> int:
        # qpos + qvel, roller pos/quat/vel, goal pos/normal, roller - goal, the goal clock
        return int(self.m.nq) + int(self.m.nv) + 19 + int(self.time_in_obs)

    def _qpos_quat_dims(self, nq: int) -> list[int]:
        m, idx = self.m, []
        for j in range(m.njnt):
            a = int(m.jnt_qposadr[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                idx += range(a + 3, a + 7)
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                idx += range(a, a + 4)
        return [i for i in idx if i < nq]

    @staticmethod
    def _bounds_from_blocks(blocks) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate (size, (lo, hi) or None for unbounded) obs blocks into Box bounds."""
        lo = np.concatenate([np.full(n, -np.inf if b is None else b[0], np.float32) for n, b in blocks])
        hi = np.concatenate([np.full(n, np.inf if b is None else b[1], np.float32) for n, b in blocks])
        return lo, hi

    def _obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Blocks in _compute_obs order
        m, unit = self.m, (-1.0, 1.0)
        lo, hi = self._bounds_from_blocks([(m.nq, None), (m.nv, None), (3, None), (4, unit), (3, None), (3, None),
                                           (3, unit), (3, None), *([(1, (0.0, 1.0))] if self.time_in_obs else [])])
        q = self._qpos_quat_dims(m.nq)
        lo[q], hi[q] = -1.0, 1.0
        return lo, hi

    def _make_spaces(self) -> None:
        # Only quaternion, normal and clock entries have finite obs bounds. The
        # embodiment is not observed
        low, high = self._obs_bounds()
        assert low.shape == (self.obs_dim(),)
        self.single_observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32, seed=self.seed)
        self.single_action_space = spaces.Box(
            low=np.full(self.act_dim(), -1.0, dtype=np.float32),
            high=np.full(self.act_dim(), 1.0, dtype=np.float32),
            dtype=np.float32, seed=self.seed)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    # ------------------------------------------------------------------ #
    # Step-program launches (before the physics substeps)
    # ------------------------------------------------------------------ #
    def _launch_tick(self) -> None:
        wp.launch(kern.tick_kernel, dim=1, inputs=[self._env])   # advance the RNG tick

    def _launch_reset(self) -> None:
        """Explicit reset: bump the RNG tick, then respawn the masked worlds (their terrain is the caller's)."""
        self._launch_tick()
        self._launch_respawn()

    def _launch_autorespawn(self) -> None:
        self._launch_respawn()
        if self.terrain:
            self._launch_terrain()

    def _launch_respawn(self) -> None:
        wp.launch(kern.reset_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._dfm, self._seg, self._pan,
                          self._leg_arr, self._consts, self._wp_tunables])
        self._lift.launch()
        if self.min_base_height_frac > 0.0:
            # The lift stood the design on the flat floor, so this is its own standing
            # height, before any terrain lifts it
            z = self.qpos_torch()[:, self._base_qadr + 2] - self._lift.floor_z
            self._spawn_h.copy_(torch.where(self.reset_mask, z, self._spawn_h))

    def _build_spawn_lift(self) -> None:
        """Set up the spawn lift, which raises each respawned robot so it starts
        5 mm clear of the floor. Calibrated with one full reset on the all-zero
        embodiment - the RNG tick and reset mask are restored afterwards."""
        self._lift = SpawnLift(self.m, self.gpu_model, self.gpu_data, self._env.mask, self._env.qpos_init,
                               self.num_envs, self.device, margin=0.005, root_body=self._base_bid)
        tick = self._tick.clone()
        self.reset_mask.fill_(True)
        if self._reset_act_buf is not None:
            self._reset_act_buf.zero_()
        with self._wp_scope():
            self._launch_reset()
            self._lift.calibrate()
        self._tick.copy_(tick)
        self.reset_mask.fill_(False)

    def _launch_ctrl(self) -> None:
        wp.launch(kern.ctrl_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts])

    # ------------------------------------------------------------------ #
    # Post-physics & substep launches
    # ------------------------------------------------------------------ #
    def _launch_zero_bad_contact(self) -> None:
        wp.launch(kern.zero_int_kernel, dim=self.num_envs,
                  inputs=[self._bad_contact_wp])

    def _launch_body_contact_substep(self) -> None:
        """Accumulate non-roller panel contacts on every physics substep (inside CUDA graph)."""
        c = self.gpu_data.contact
        wp.launch(kern.body_panel_contact_kernel, dim=c.geom.shape[0],
                  inputs=[c.geom, c.worldid, c.dist, self.gpu_data.nacon,
                          self._plate_mask_wp, self._roller_gid,
                          self.body_contact_depth, self._bad_contact_wp,
                          self._env.mask, 0])

    def _launch_goal_step(self) -> None:
        """Run ``goal_kernel`` at the tail of the CUDA-graphed step program."""
        wp.launch(kern.goal_kernel, dim=self.num_envs,
                  inputs=[self._sim, self._goal, self._consts, self._env.mask, 0])

    def _launch_goal(self, only_reset: bool = False, mask: torch.Tensor | None = None) -> None:
        """Run ``goal_kernel`` on the post-substep pose, on torch's stream. ``only_reset`` limits it to the
        worlds in ``mask`` (default: ``self.reset_mask``)."""
        m_wp = self._env.mask if mask is None else wp.from_torch(mask)
        with self._wp_scope():
            wp.launch(kern.goal_kernel, dim=self.num_envs,
                      inputs=[self._sim, self._goal, self._consts, m_wp, int(only_reset)])

    def _launch_force(self) -> None:
        wp.launch(kern.force_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts, self._wp_tunables])

    def _launch_leg_noise(self) -> None:
        wp.launch(kern.leg_noise_kernel, dim=(self.num_envs, self._n_leg_noise),
                  inputs=[self._env.tick, self._env.lane_seed, self._leg_noise_bids_wp,
                          self.leg_noise_mag, self.gpu_data.xfrc_applied])

    def _launch_body_contact(self, only_reset: bool = False) -> None:
        """Flag worlds where anything but the roller presses on a panel, from this
        step's contacts. ``only_reset`` limits it to the worlds in ``reset_mask``."""
        if only_reset:
            self._bad_contact.masked_fill_(self.reset_mask, 0)
        else:
            self._bad_contact.zero_()
        c = self.gpu_data.contact
        with self._wp_scope():
            wp.launch(kern.body_panel_contact_kernel, dim=c.geom.shape[0],
                      inputs=[c.geom, c.worldid, c.dist, self.gpu_data.nacon,
                              self._plate_mask_wp, self._roller_gid,
                              self.body_contact_depth, self._bad_contact_wp,
                              self._env.mask, int(only_reset)])

    # ------------------------------------------------------------------ #
    # The sweep
    # ------------------------------------------------------------------ #
    def _sync_episode_state(self, only_reset: bool = False, from_step: bool = False) -> None:
        """Put respawned worlds back at the first goal, then read the goal geometry
        that the observation, reward and marker share. ``only_reset`` reads it for
        the worlds in ``reset_mask`` only."""
        if not from_step:
            fresh = self.reset_mask
            self._goal_idx.masked_fill_(fresh, 0)
            self._dwell = self._dwell.masked_fill(fresh, 0)
            self._cell_steps = self._cell_steps.masked_fill(fresh, 0)
            self._rebase = self._rebase | fresh
            self._launch_goal(only_reset)

    def _base_height(self) -> torch.Tensor:
        """Height of the base above the ground directly under it, per world: the
        floor plane at z = 0, or the world's own terrain, interpolated bilinearly
        between the hfield's grid points."""
        q = self.qpos_torch()[:, self._base_qadr:self._base_qadr + 3]
        if not self.terrain:
            return q[:, 2]
        nr, nc = self._hf_shape
        (x0, x1), (y0, y1) = self.terrain_region
        fc = ((q[:, 0] - x0) / (x1 - x0)).clamp(0.0, 1.0) * (nc - 1)
        fr = ((q[:, 1] - y0) / (y1 - y0)).clamp(0.0, 1.0) * (nr - 1)
        c0 = fc.long().clamp(max=nc - 2)
        r0 = fr.long().clamp(max=nr - 2)
        tc, tr = fc - c0, fr - r0
        data = wp.to_torch(self.gpu_model.hfield_data)
        at = lambda dr, dc: data[self._hf_adr + (r0 + dr) * nc + c0 + dc]
        v = ((1 - tr) * ((1 - tc) * at(0, 0) + tc * at(0, 1))
             + tr * ((1 - tc) * at(1, 0) + tc * at(1, 1)))
        return q[:, 2] - (self._terrain_base_h + v * self._hf_scale)

    def _paint_cleaned(self, cleaned: torch.Tensor) -> None:
        """Paint each just-cleaned cell in-place on device. Midpoints are masked out."""
        with self._wp_scope():
            wp.launch(
                kern.paint_cleaned_kernel,
                dim=self.num_envs,
                inputs=[
                    wp.from_torch(cleaned),
                    wp.from_torch(self._goal_idx),
                    wp.from_torch(self._goal_is_cell),
                    wp.from_torch(self._goal_gid_i32),
                    int(self.n_goals),
                    wp.vec4(*RGBA_CLEAN),
                    self.gpu_model.geom_rgba,
                ],
            )

    # ------------------------------------------------------------------ #
    # Obs / reward
    # ------------------------------------------------------------------ #
    def _goal_time_left(self) -> torch.Tensor:
        """Share of the current goal's budget still ahead, 1 on its first step and
        0 on its last.

        Read before the reward advances the step counter, so for a world whose
        goal just changed the counter is stale, and ``_rebase`` says so.
        """
        idx = self._goal_idx.clamp(max=self.n_goals - 1)
        spent = torch.where(self._rebase, 0, self._cell_steps)
        left = 1.0 - spent.float() / self._goal_budget_span[idx]
        return left.clamp_(0.0, 1.0).unsqueeze(1)

    def _compute_obs(self, only_reset: bool = False, from_step: bool = False) -> torch.Tensor:
        """The observation of every world. ``only_reset`` updates the goal, contact and divergence state of the
        worlds in ``reset_mask`` only. When ``from_step`` is True, ``goal_kernel`` and per-substep
        ``body_panel_contact_kernel`` have already executed inside the step program."""
        self._sync_episode_state(only_reset, from_step=from_step)
        if self.terminate_on_body_contact and not from_step:
            self._launch_body_contact(only_reset)
        obs = torch.cat([
            (self.qpos_torch() - self._qpos_off) * self._qpos_scl,
            self.qvel_torch() / self.vel_scale,
            (self._roller_pos - self._fpos) / self._fsize,
            self._roller_quat,
            self._roller_vel_local / self.vel_scale,
            (self._goal_pt - self._fpos) / self._fsize,
            self._goal_normal,
            self._roller_rel_local / self.max_rel_dist,
            *([self._goal_time_left()] if self.time_in_obs else []),
        ], dim=1)
        if self.check_finite:
            self._assert_finite(obs, "obs")
        # A diverged world never meets a terminal condition (comparisons with NaN
        # are False), so flag it for _reward_terminated_truncated to terminate,
        # and return a zero placeholder row
        diverged = ~torch.isfinite(obs).all(dim=1)
        if only_reset:                 # the other worlds keep their flags and are not counted again
            diverged = torch.where(self.reset_mask, diverged, self._diverged)
            self._diverged_total += (diverged & self.reset_mask).sum()
        else:
            self._diverged_total += diverged.sum()
        self._diverged = diverged
        return obs.masked_fill_(diverged[:, None], 0.0)

    def sample_observations(self, n: int = 1) -> torch.Tensor:
        """``n`` episode-start observations, from resets of the whole batch with
        the default embodiment. Use a separate env."""
        rows, collected = [], 0
        while collected < int(n):
            out = self.reset()                  # the co-design reset returns (obs, rejected, info)
            rows.append((out[0] if isinstance(out, tuple) else out).clone())
            collected += rows[-1].shape[0]
        return torch.cat(rows)[:int(n)]

    def _reward_terminated_truncated(self):
        # Every geometric quantity was written by goal_kernel during _compute_obs
        dist = self._goal_dist

        # Worlds starting a goal (respawned, or moved on last step): their
        # progress baseline and reward scale start from here
        fresh = self._rebase
        self._prev_dist = torch.where(fresh, dist, self._prev_dist)
        self._d0 = torch.where(fresh, dist.clamp(min=1e-3), self._d0)

        idx = self._goal_idx.clamp(max=self.n_goals - 1)
        is_cell = self._goal_is_cell[idx]

        at_goal = self._at_goal_t
        # Decaying dwell: holding steady on the cell accumulates dwell; slipping off decays it
        dwell = self._dwell.masked_fill(fresh, 0)
        self._dwell = torch.where(at_goal, dwell + 1, (dwell - 1).clamp(min=0))
        cleaned = self._dwell >= self._goal_dwell[idx]

        self._cell_steps = self._cell_steps.masked_fill(fresh, 0) + 1
        out_of_time = self._cell_steps >= self._goal_budget[idx]

        # Potential-based: summed over a goal it gives prog_gain * (d0 - d_end) / d0,
        # and moving away takes back what moving closer gave
        reward = (self.prog_gain / self._d0) * (self._prev_dist - dist)
        reward = reward + (at_goal & is_cell).float() * self.at_goal_rew
        self._prev_dist = dist.clone()

        self._paint_cleaned(cleaned)
        self._goal_idx += cleaned.to(torch.int32)
        self._rebase = cleaned
        all_clean = self._goal_idx >= self.n_goals

        left_field = self._left_field_t

        fouled = (self._bad_contact.bool() if self.terminate_on_body_contact
                  else torch.zeros_like(left_field))
        reward = reward + fouled.float() * self.body_contact_penalty
        self.last_body_contact = fouled

        if self.terminate_off_green and self.terrain and self._base_qadr >= 0:
            bxy = self.qpos_torch()[:, self._base_qadr:self._base_qadr + 2]
            off_green = ((bxy < self._green_lo) | (bxy > self._green_hi)).any(dim=1)
        else:
            off_green = torch.zeros_like(left_field)
        self.last_off_green = off_green

        if self.min_base_height_frac > 0.0:
            too_low = self._base_height() < self.min_base_height_frac * self._spawn_h
        else:
            too_low = torch.zeros_like(left_field)
        self.last_too_low = too_low

        # A finished sweep, a foul, leaving the field or the terrain, a sunk base and
        # divergence are true terminals. A run-out budget is reported per
        # truncate_at_time_limit, and never overrides a finished sweep
        terminated = all_clean | self._diverged | fouled | off_green | too_low
        if self.terminate_out_of_field:
            terminated = terminated | left_field
        if self.truncate_at_time_limit:
            truncated = ran_out = out_of_time & ~terminated
        else:
            ran_out = out_of_time & ~terminated
            terminated = terminated | out_of_time
            truncated = torch.zeros_like(terminated)
        if self.check_finite:
            self._assert_finite(reward, "reward")
        reward = reward.masked_fill(self._diverged, 0.0)

        # last_at_goal and last_left_field are views of goal_kernel's buffers,
        # valid until the next step
        self.last_at_goal = at_goal
        self.last_cleaned = cleaned
        self.last_all_clean = all_clean
        self.last_left_field = left_field
        self.last_time_limit = ran_out          # budget ran out, however it is reported
        self.last_diverged = self._diverged
        return reward, terminated, truncated

    @property
    def goals_cleaned(self) -> torch.Tensor:
        """Goals reached so far this episode, per world (cells and midpoints)."""
        return self._goal_idx

    @property
    def diverged_worlds(self) -> int:
        """Worlds ended so far by non-finite physics, should stay 0. Syncs with the device."""
        return int(self._diverged_total)

    # ------------------------------------------------------------------ #
    # Live tunables + host-side views
    # ------------------------------------------------------------------ #
    def change_max_offset(self, new_offset: float) -> None:
        """Set the metres per unit embodiment action for the links."""
        self.max_offset = float(new_offset)
        self._segs["axis_lo"] = length_safe_lo(
            self._segs, self._raw_axis_lo, self.max_offset,
            self.min_link_length + self.min_length_pad)
        wp.to_torch(self._seg.axis_lo).copy_(
            torch.as_tensor(self._segs["axis_lo"], dtype=torch.float32, device=self.device))
        self._warn_length_floor()
        self._refresh_tunables()

    def set_force_mag(self, newtons: float) -> None:
        """Set the perturbation force magnitude (0 to disable)."""
        self.force_mag = float(newtons)
        self._refresh_tunables()

    def set_spawn_randomisation(self, pos_range: float,
                                heading_range: float) -> None:
        """Set the spawn jitter (metres, degrees), applied from the next respawn."""
        self.rand_pos_range = float(pos_range)
        self.rand_heading_range = float(heading_range)
        self._refresh_tunables()

    @property
    def env_offsets(self) -> np.ndarray:
        """Host copy of each world's applied embodiment in metres, the leg switch stays 0 or 1."""
        scale = np.full(self.n_offsets, self.max_offset, dtype=np.float32)
        scale[self.n_link_slots:] = 1.0
        return self._offsets_t.detach().cpu().numpy() * scale

    @property
    def env_offsets_norm(self) -> np.ndarray:
        """Host copy of each world's applied embodiment, normalised like the embodiment action."""
        return self._offsets_t.detach().cpu().numpy()

    @property
    def offset_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of each embodiment entry, narrowed from [-1, 1] where the deform would clip."""
        return slot_bounds(self._segs, self.n_offsets)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _init_render_model(self, m: mujoco.MjModel) -> None:
        release_geom_frames(m, self._segs)

    def _hfield_upload_key(self, env_idx: int) -> tuple[int, int] | None:
        if not self.terrain:
            return None
        return (int(env_idx), int(self._ep_count[int(env_idx)]))

    def _mirror_render_model(self, m: mujoco.MjModel, env_idx: int) -> None:
        """Copy one world's deformed links, cell colours, marker and terrain into
        the CPU render ``MjModel`` ``m``."""
        w = int(env_idx)
        v, gm = m, self.gpu_model
        g, b, kid = self._segs["geom"], self._segs["body"], self._segs["child_id"]
        v.geom_pos[g] = wp.to_torch(gm.geom_pos)[w, g].cpu().numpy()
        v.geom_quat[g] = wp.to_torch(gm.geom_quat)[w, g].cpu().numpy()
        v.body_iquat[b] = wp.to_torch(gm.body_iquat)[w, b].cpu().numpy()
        v.body_ipos[b] = wp.to_torch(gm.body_ipos)[w, b].cpu().numpy()
        v.geom_size[g] = wp.to_torch(gm.geom_size)[w, g].cpu().numpy()
        v.geom_rbound[g] = wp.to_torch(gm.geom_rbound)[w, g].cpu().numpy()
        v.geom_aabb[g] = wp.to_torch(gm.geom_aabb)[w, g].reshape(len(g), 6).cpu().numpy()
        v.body_pos[kid] = wp.to_torch(gm.body_pos)[w, kid].cpu().numpy()
        sid = self._segs["site_id"]
        if len(sid):
            v.site_pos[sid] = wp.to_torch(gm.site_pos)[w, sid].cpu().numpy()
        cg = self._cell_gids
        v.geom_rgba[cg] = wp.to_torch(gm.geom_rgba)[w, cg].cpu().numpy()
        v.body_pos[self._marker_bid] = (
            wp.to_torch(gm.body_pos)[w, self._marker_bid].cpu().numpy())
        if self.disable_legs:
            lg = self._leg_geoms
            v.geom_rgba[lg] = wp.to_torch(gm.geom_rgba)[w, lg].cpu().numpy()
        # The visualizer culls against bvh_aabb, fixed at compile time. For a
        # one-geom body, the body's box is that geom's aabb
        for bid, gid in zip(self._segs["body"], self._segs["geom"]):
            adr = int(v.body_bvhadr[bid])
            if adr >= 0 and v.body_bvhnum[bid] == 1 and v.body_geomnum[bid] == 1:
                v.bvh_aabb[adr] = v.geom_aabb[gid]

        if self.terrain:
            nr = int(self.terrain_res[0]) * int(self.terrain_res[1])
            src = int(wp.to_torch(gm.hfield_adr)[w])
            dst = int(v.hfield_adr[0])
            v.hfield_data[dst:dst + nr] = (
                wp.to_torch(gm.hfield_data)[src:src + nr].cpu().numpy())
            pbid = wp.to_torch(self._panel_bid_wp).cpu().numpy()
            v.body_pos[pbid] = wp.to_torch(gm.body_pos)[w, pbid].cpu().numpy()
            if self._n_mid:
                mid = wp.to_torch(self._mid_sid_wp).cpu().numpy()
                v.site_pos[mid] = wp.to_torch(gm.site_pos)[w, mid].cpu().numpy()

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Draw the roller-to-goal vector and, when the push is on, the force on
        the base."""
        if scene.ngeom >= scene.maxgeom:
            return

        roller = self._roller_pos[env_idx].detach().cpu().numpy().astype(np.float64)
        goal = self._goal_pt[env_idx].detach().cpu().numpy().astype(np.float64)
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), np.array([0.2, 0.9, 0.3, 0.6], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.005, roller, goal)
        scene.ngeom += 1

        if not self.force_on or scene.ngeom >= scene.maxgeom:
            return
        f = wp.to_torch(self.gpu_data.xfrc_applied)[env_idx, self._base_bid, :2]
        f = f.detach().cpu().numpy().astype(np.float64)
        mag = float(np.linalg.norm(f))
        if mag < 1e-9:
            return

        base = self.xpos_torch()[env_idx, self._base_bid].detach().cpu().numpy()
        base = base.astype(np.float64)
        tip = base + np.array([f[0], f[1], 0.0]) / mag * (mag * self.force_viz_scale)

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), np.array([1.0, 0.25, 0.1, 0.9], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.1, base, tip)
        scene.ngeom += 1

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """``fixed`` frames the spawn and the whole array, ``track`` follows the base
        from behind, looking up the array."""
        p = self._panels
        robot_x = -self.spawn_offset
        far = p["first_pos"][0] + (p["length"] + p["gap_x"]) * (p["rows"] - 1)
        near_x = robot_x - 0.5
        far_x = far + p["length"] / 2 + 0.5
        span_x = far_x - near_x
        span_y = p["width"] * p["cols"] + p["gap_y"] * (p["cols"] - 1)
        if self.camera_mode == "track":
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = self._base_bid
            camera.azimuth = 20.0               # slightly off +x, over the robot's shoulder
            camera.elevation = -25.0
            camera.distance = 1.6 * max(p["first_pos"][0] + p["length"],
                                        span_y, 2.0)
            return
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.5 * (near_x + far_x), p["first_pos"][1],
                            p["height"] + 0.6]
        camera.elevation = -12.0
        camera.azimuth = 130.0
        camera.distance = 2.1 * max(span_x, span_y, 1.5)
