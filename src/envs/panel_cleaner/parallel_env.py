"""Batched MuJoCo env base class: N worlds stepped together on one GPU.

The subclass builds one ``MjSpec``. It is compiled once and uploaded as one
mujoco_warp model with batched data for all worlds (or with extra device-only
assets via ``_augment_gpu_asset_spec``, e.g. per-world terrain heightfields).
Per-world variation happens on the device: each field in
``PER_WORLD_MODEL_FIELDS`` gets one row per world, which the subclass's kernels
write.

A control step runs the step program: the launches registered with
``register_step_launch``, then ``phys_substeps`` physics substeps (each followed
by any ``register_substep_launch`` callbacks), then a kinematics and body
velocity pass, then any ``register_post_step_launch`` callbacks. On CUDA it is
captured once as a graph and replayed, and ``use_graph=False`` runs it directly.

``self.m`` keeps the compile-time values. ``render()`` and the viewer draw from
a single CPU copy of it, into which ``_mirror_render_model`` writes the drawn
world's device-side changes.
"""

from __future__ import annotations

import atexit
import copy
import warnings

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from core.mujoco_rendering import close_human, validate_render_mode

RENDER_H, RENDER_W = 480, 480


def copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy a batch into a persistent device buffer, coercing device and dtype.
    A single row is broadcast, extra rows are dropped."""
    src = src.to(device=dst.device, dtype=dst.dtype)
    if src.shape != dst.shape:
        src = src.expand(dst.shape) if src.shape[0] == 1 else src[:dst.shape[0]]
    dst.copy_(src)


class ParallelMujocoEnv:
    """Steps ``num_envs`` MuJoCo worlds in lockstep on one device.

    ``reset() -> obs`` and ``step(...) -> (obs, reward, terminated, truncated,
    info)`` work on batched tensors. A world that ends is respawned on the next
    step, after the step program and outside the graph, with that call's reset
    actions.
    """

    # Model fields the subclass's kernels write per world, one row per world.
    # All other fields keep one shared row.
    PER_WORLD_MODEL_FIELDS: tuple[str, ...] = (
        "geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound",
        "geom_rgba",
        "body_pos", "body_ipos", "body_iquat", "body_mass", "body_inertia",
    )
    # What set_const derives from the masses, per world so _refresh_constants
    # can give every design its own
    DERIVED_PER_WORLD_FIELDS: tuple[str, ...] = (
        "body_subtreemass", "body_invweight0", "dof_invweight0",
    )

    def __init__(
        self,
        num_envs: int,
        *,
        device: str = "cuda",
        viz_env_id: int = 0,
        max_episode_steps: int = 2000,
        debug: bool = False,
        seed: int = 0,
        phys_substeps: int = 1,
        render_mode: str | None = None,
        use_graph: bool = True,
        njmax: int = 128,
        nconmax: int | None = None,
        naconmax: int | None = None,
    ):
        wp.init()
        self.num_envs = num_envs
        self.device = device
        validate_render_mode(render_mode)
        self.render_mode = render_mode
        self.phys_substeps = phys_substeps
        self.viz_env_id = viz_env_id
        self.max_episode_steps = max_episode_steps
        self.debug = debug
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._is_cuda = device != "cpu" and wp.get_device(device).is_cuda
        self._wp_stream = (
            wp.stream_from_torch(
                torch.cuda.current_stream(torch.device(device)))
            if self._is_cuda else None)

        self._step_launches: list = []
        self._substep_launches: list = []
        self._post_step_launches: list = []
        self._graph = None
        self._warmed_up = False
        self._use_graph = bool(use_graph) and self._is_cuda

        # Kernels read these, never the caller's tensors, since launch arguments
        # freeze at capture. The subclass allocates them (_alloc_action_buffers)
        self._act_buf: torch.Tensor | None = None
        self._reset_act_buf: torch.Tensor | None = None

        self._render_m: mujoco.MjModel | None = None
        self._render_d: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.camera: mujoco.MjvCamera | None = None
        self._renderers: list[mujoco.Renderer] = []
        self._render_cleanup_registered = False
        self._viewer_m: mujoco.MjModel | None = None
        self._viewer_d: mujoco.MjData | None = None
        self._last_uploaded_hfield_key: tuple[int, int] | None = None

        self.m: mujoco.MjModel | None = None
        self.gpu_model = None
        self.gpu_data = None

        self.world_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_return = torch.zeros(num_envs, device=device)
        self.reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reset_pending = False

        self._build_model(self._build_base_spec(), njmax, nconmax, naconmax)
        self._init_control_ranges()

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        """Return the assembled, unmodified base spec."""
        raise NotImplementedError

    def _augment_gpu_asset_spec(self, asset_spec: mujoco.MjSpec) -> bool:
        """Override to add assets only the device model needs (e.g. per-world
        hfields). Return True if ``asset_spec`` was changed."""
        return False

    def _launch_reset(self) -> None:
        """Launch the per-world reset kernel, masked by ``self.reset_mask``."""
        raise NotImplementedError

    def _compute_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _reward_terminated_truncated(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reward, terminated, truncated), each [num_envs]."""
        raise NotImplementedError

    def _init_render_model(self, m: mujoco.MjModel) -> None:
        """Override to prepare a new render copy ``m``. Called once per copy."""

    def _mirror_render_model(self, m: mujoco.MjModel, env_idx: int) -> None:
        """Override to write device-side model changes of world ``env_idx``
        into the render copy ``m``."""

    def _hfield_upload_key(self, env_idx: int) -> tuple[int, int] | None:
        """Return a cache key for world ``env_idx``'s current hfield data, or
        None if the env has no dynamic heightfield."""
        return None

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """Override to aim the viewer and offscreen camera."""

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Override to add visual-only markers (used only when ``debug``)."""

    # ------------------------------------------------------------------ #
    # Model build: compile once -> batched upload -> per-world rows
    # ------------------------------------------------------------------ #
    def _build_model(self, base_spec: mujoco.MjSpec, njmax, nconmax, naconmax=None) -> None:
        m = base_spec.compile()
        if m.opt.noslip_iterations > 0:
            warnings.warn(f"disabling the noslip solver (noslip_iterations={int(m.opt.noslip_iterations)}): "
                          "mujoco_warp does not implement it, so batched physics will "
                          "differ slightly from a CPU run of the same model.", UserWarning, stacklevel=2)
        m.opt.noslip_iterations = 0
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), RENDER_W)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), RENDER_H)
        self.m = m

        asset_spec = base_spec.copy()
        upload_m = asset_spec.compile() if self._augment_gpu_asset_spec(asset_spec) else m
        upload_m.opt.noslip_iterations = 0
        with wp.ScopedDevice(self.device):
            self.gpu_model = mjw.put_model(upload_m)
            self.gpu_data = mjw.make_data(
                upload_m, nworld=self.num_envs, nconmax=nconmax, njmax=njmax,
                naconmax=naconmax)
            for attr in self.PER_WORLD_MODEL_FIELDS + self.DERIVED_PER_WORLD_FIELDS:
                self._make_field_per_world(attr)
            # Per world, so set_const gives each design its own solver tolerance scale
            mi = self.gpu_model.stat.meaninertia
            self.gpu_model.stat.meaninertia = wp.array(
                np.full(self.num_envs, float(mi.numpy()[0]), np.float32),
                dtype=mi.dtype, device=mi.device)
            mjw.set_const(self.gpu_model, self.gpu_data)
            mjw.forward(self.gpu_model, self.gpu_data)
            wp.synchronize()

    def _refresh_constants(self) -> None:
        """Recompute the mass-derived constants (``body_subtreemass``,
        ``*_invweight0``, ``stat.meaninertia``) for every world's current design."""
        with self._wp_scope():
            mjw.set_const(self.gpu_model, self.gpu_data)

    def _make_field_per_world(self, attr: str) -> None:
        """Replace the shared row of ``gpu_model.<attr>`` with one copy per world."""
        shared = getattr(self.gpu_model, attr, None)
        if shared is None:
            return
        if shared.ndim < 2 or shared.shape[0] != 1:
            return
        rows = np.repeat(shared.numpy(), self.num_envs, axis=0)
        setattr(self.gpu_model, attr, wp.array(rows, dtype=shared.dtype, device=shared.device))

    def _init_control_ranges(self) -> None:
        rng = self.m.actuator_ctrlrange.astype(np.float32)
        self.ctrl_scale = torch.as_tensor(
            (rng[:, 1] - rng[:, 0]) / 2, device=self.device)
        self.ctrl_mid = torch.as_tensor(
            (rng[:, 1] + rng[:, 0]) / 2, device=self.device)

    def _alloc_action_buffers(self, act_dim: int, reset_dim: int = 0) -> None:
        self._act_buf = torch.zeros(
            (self.num_envs, act_dim), dtype=torch.float32, device=self.device)
        self._reset_act_buf = None if reset_dim <= 0 else torch.zeros(
            (self.num_envs, reset_dim), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def _wp_scope(self):
        """Device scope for warp launches, on torch's stream when there is one."""
        dev = wp.ScopedDevice(self.device)
        return dev if self._wp_stream is None else _Nested(
            dev, wp.ScopedStream(self._wp_stream))

    def register_step_launch(self, fn) -> None:
        """Add a launch to the step program, before the physics substeps."""
        self._step_launches.append(fn)
        self.invalidate_graph()

    def register_substep_launch(self, fn) -> None:
        """Add a launch run after each physics substep, inside the step graph."""
        self._substep_launches.append(fn)
        self.invalidate_graph()

    def register_post_step_launch(self, fn) -> None:
        """Add a launch run after kinematics/com_vel, at the tail of the step graph."""
        self._post_step_launches.append(fn)
        self.invalidate_graph()

    def invalidate_graph(self) -> None:
        """Drop the captured graph; the next step recaptures."""
        self._graph = None

    def _run_step_program(self) -> None:
        for fn in self._step_launches:
            fn()
        for _ in range(self.phys_substeps):
            mjw.step(self.gpu_model, self.gpu_data)
            for fn in self._substep_launches:
                fn()
        # mjw.step integrates after forward(), so poses/velocities would be one substep stale
        mjw.kinematics(self.gpu_model, self.gpu_data)
        mjw.com_pos(self.gpu_model, self.gpu_data)
        mjw.com_vel(self.gpu_model, self.gpu_data)
        for fn in self._post_step_launches:
            fn()

    def _ensure_graph(self) -> None:
        """Capture the step program, running it once first to compile kernels."""
        if not self._use_graph or self._graph is not None:
            return
        with wp.ScopedDevice(self.device):
            if not self._warmed_up:
                self._run_step_program()
                self._warmed_up = True
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                self._run_step_program()
            self._graph = cap.graph

    def _step_logic(self) -> None:
        self._ensure_graph()
        with self._wp_scope():
            if self._graph is None:
                self._run_step_program()
            else:
                wp.capture_launch(self._graph)

    def _stage_actions(self, ctrl_actions, reset_actions) -> None:
        if self._act_buf is not None and ctrl_actions is not None:
            copy_into(self._act_buf, ctrl_actions)
        if self._reset_act_buf is not None and reset_actions is not None:
            copy_into(self._reset_act_buf, reset_actions)

    def reset(self) -> torch.Tensor:
        """Reset every world and return the first observation."""
        self._ensure_graph()
        self.reset_mask.fill_(True)
        if self._reset_act_buf is not None:
            self._reset_act_buf.zero_()
        with self._wp_scope():
            self._launch_reset()
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps.zero_()
        self.ep_return.zero_()
        self.reset_mask.zero_()
        self._reset_pending = False
        return self._compute_obs()

    def step(self, ctrl_actions: torch.Tensor, reset_actions: torch.Tensor = None):
        """Advance all worlds one control step: ``(obs, reward, terminated, truncated, info)``."""
        self._stage_actions(ctrl_actions, reset_actions)
        had_reset, self._reset_pending = self._reset_pending, False

        self._step_logic()
        self.world_steps += self.phys_substeps

        obs = self._compute_obs(from_step=True)
        reward, terminated, truncated = self._reward_terminated_truncated()
        if had_reset:
            obs, reward, terminated, truncated = self._respawn_pending(obs, reward, terminated, truncated)
        self.ep_return += reward

        done = terminated | truncated
        info = {
            "final_obs": obs.clone(),
            "final_return": self.ep_return.clone(),
            "final_steps": self.world_steps // self.phys_substeps,
        }
        self.reset_mask.copy_(done)
        # The step's one host sync: whether the next step has worlds to respawn
        self._reset_pending = bool(done.any())
        return obs, reward, terminated, truncated, info

    def _launch_autorespawn(self) -> None:
        """Respawn the worlds in ``reset_mask`` at an autoreset."""
        self._launch_respawn()

    def _rebase_episode(self, mask: torch.Tensor) -> None:
        """Start the reward's per-episode state for the worlds in ``mask`` from their respawn."""

    def _respawn_pending(self, obs, reward, terminated, truncated):
        """Respawn the worlds in ``reset_mask`` (they ended on the previous call)."""
        respawned = self.reset_mask.clone()
        with self._wp_scope():
            self._launch_autorespawn()
            wp.to_torch(self.gpu_data.ctrl)[respawned] = 0
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps.masked_fill_(respawned, 0)
        self.ep_return.masked_fill_(respawned, 0.0)
        diverged, total = self._diverged, getattr(self, "_diverged_total", None)
        total = None if total is None else total.clone()
        fresh = self._compute_obs(only_reset=True)
        self._diverged = torch.where(respawned, self._diverged, diverged)
        if total is not None:
            self._diverged_total = total
        self._rebase_episode(respawned)
        obs = torch.where(respawned[:, None], fresh, obs)
        return obs, reward.masked_fill(respawned, 0.0), terminated & ~respawned, truncated & ~respawned

    def qpos_torch(self) -> torch.Tensor:
        return wp.to_torch(self.gpu_data.qpos)

    def qvel_torch(self) -> torch.Tensor:
        return wp.to_torch(self.gpu_data.qvel)

    def xpos_torch(self) -> torch.Tensor:
        return wp.to_torch(self.gpu_data.xpos)

    def _assert_finite(self, t: torch.Tensor, name: str) -> None:
        if not bool(torch.isfinite(t).all()):
            raise RuntimeError(
                f"{name} at world step {self.world_steps} is not finite: {t}")

    # ------------------------------------------------------------------ #
    # Offscreen rendering of a single world
    # ------------------------------------------------------------------ #
    def _make_render_model(self) -> mujoco.MjModel:
        """Return a new CPU copy of ``self.m`` to draw worlds from."""
        m = copy.deepcopy(self.m)
        self._init_render_model(m)
        return m

    def render(self, env_idx: int | None = None) -> np.ndarray:
        """Draw world ``env_idx`` (default: the viewer world) offscreen. One
        renderer, model copy and MjData draw every world."""
        if env_idx is None:
            env_idx = self.viz_env_id
        if self.renderer is None:
            self._render_m = self._make_render_model()
            self._render_d = mujoco.MjData(self._render_m)
            self.renderer = mujoco.Renderer(self._render_m, RENDER_H, RENDER_W)
            self._renderers.append(self.renderer)
            self._register_render_cleanup()
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self._render_m, self.camera)
            self._setup_camera(self.camera)
            self._last_uploaded_hfield_key = None

        m, d = self._render_m, self._render_d
        self._mirror_render_model(m, env_idx)

        # Re-upload dynamic heightfield to OpenGL whenever the rendered world or episode changed
        hf_key = self._hfield_upload_key(env_idx)
        if hf_key is not None and hf_key != self._last_uploaded_hfield_key:
            ctx = getattr(self.renderer, "_mjr_context", None)
            if ctx is not None:
                mujoco.mjr_uploadHField(m, ctx, 0)
            self._last_uploaded_hfield_key = hf_key

        with self._wp_scope():
            mjw.get_data_into(d, m, self.gpu_data, env_idx)
        mujoco.mj_forward(m, d)
        self.renderer.update_scene(d, camera=self.camera)
        if self.debug:
            self._add_debug_visualizations(self.renderer.scene, env_idx)
        return self.renderer.render().copy()

    def _register_render_cleanup(self) -> None:
        if self._render_cleanup_registered:
            return
        atexit.register(_close_renderers, self._renderers)
        self._render_cleanup_registered = True

    def close(self) -> None:
        """Free the offscreen GL contexts. Idempotent."""
        close_human(self)
        _close_renderers(self._renderers)
        self.renderer = self.camera = None
        self._render_m = self._render_d = None
        self._last_uploaded_hfield_key = None


def _close_renderers(renderers: list) -> None:
    while renderers:
        renderer = renderers.pop()
        try:
            renderer.close()
        except Exception:
            pass


class _Nested:
    """Enter and exit several context managers as one."""

    def __init__(self, *ctxs):
        self._ctxs = ctxs

    def __enter__(self):
        for c in self._ctxs:
            c.__enter__()
        return self

    def __exit__(self, *exc):
        for c in reversed(self._ctxs):
            c.__exit__(*exc)
        return False
