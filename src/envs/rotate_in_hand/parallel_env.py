"""Batched MuJoCo env base class: N worlds stepped together on one GPU.

The subclass builds one ``MjSpec``. It is compiled once and uploaded as one
mujoco_warp model with batched data for all worlds. Per-world variation happens
on the device: each field in ``PER_WORLD_MODEL_FIELDS`` gets one row per world,
which the subclass's kernels write.

A control step runs the step program: the launches registered with
``register_step_launch``, the physics substeps, then a kinematics and body
velocity pass. On CUDA it is captured once as a graph and replayed, and
``use_graph=False`` runs it directly. Because launch arguments are frozen at capture:

- per-world decisions inside the graph are made by device masks, and per-step
inputs arrive through preallocated device buffers;
- capture-time constants are plain kernel arguments, and changing one requires
``invalidate_graph()``;
- live tunables sit in a device array the kernels read, so changing one needs
no recapture.

Observation, reward and the info dict run outside the graph, in torch.
``self.m`` keeps the compile-time values. ``render()`` and the viewer draw from
copies of it, into which ``_mirror_render_model`` writes the drawn world's
device-side changes, so one copy can draw any world.

Code layout:

- ParallelMujocoEnv: the base class, subclasses implement the "Subclass hooks";
- copy_into: stage a caller's batch into a persistent device buffer.
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

    Holds the compiled ``MjModel`` and the batched ``gpu_model``/``gpu_data``,
    the step program and its graph, the action staging buffers, per-world
    episode bookkeeping, and the render and viewer copies.
    """

    # Model fields the subclass's kernels write per world, one row per world.
    # All other fields keep one shared row
    PER_WORLD_MODEL_FIELDS: tuple[str, ...] = ()
    # What set_const derives from the masses, per world so _refresh_constants
    # can give every design its own
    DERIVED_PER_WORLD_FIELDS: tuple[str, ...] = ("body_subtreemass", "body_invweight0", "dof_invweight0")

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
        njmax: int | None = 256,
        nconmax: int | None = None,
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

        # Run warp on torch's stream so the two are ordered without host syncs.
        # The device must be passed explicitly: current_stream() otherwise
        # returns a stream on torch's default device, and warp would follow it
        self._is_cuda = device != "cpu" and wp.get_device(device).is_cuda
        self._wp_stream = (
            wp.stream_from_torch(
                torch.cuda.current_stream(torch.device(device)))
            if self._is_cuda else None)

        self._step_launches: list = []
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
        self._renderers: list[mujoco.Renderer] = []   # the atexit hook holds this list, not self
        self._render_cleanup_registered = False
        self._viewer_m: mujoco.MjModel | None = None    # set by mjview_utils
        self._viewer_d: mujoco.MjData | None = None

        self.m: mujoco.MjModel | None = None
        self.gpu_model = None
        self.gpu_data = None

        self.world_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_return = torch.zeros(num_envs, device=device)
        # Worlds the next step program respawns. Never rebound: the reset kernel
        # holds a warp view of it
        self.reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reset_pending = False         # some world ended last step and respawns on the next

        self._build_model(self._build_base_spec(), njmax, nconmax)
        self._init_control_ranges()

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        """Return the assembled, unmodified base spec."""
        raise NotImplementedError

    def _fill_world_rows(self) -> None:
        """Override to write per-world model rows that differ from the start. Runs
        after the upload, before the derived constants."""

    def _launch_reset(self) -> None:
        """Launch the per-world reset kernel, masked by ``self.reset_mask``.
        Launch-only, called by ``reset()`` outside the graph."""
        raise NotImplementedError

    def _compute_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _reward_terminated_truncated(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reward, terminated, truncated), each [num_envs]."""
        raise NotImplementedError

    def _init_render_model(self, m: mujoco.MjModel) -> None:
        """Override to prepare a new render copy ``m``. The base class calls it once
        per copy."""

    def _mirror_render_model(self, m: mujoco.MjModel, env_idx: int) -> None:
        """Override to write the device-side model changes of world ``env_idx``
        into the render copy ``m``. render() and the viewer call it before each
        draw."""

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """Override to aim the viewer and offscreen camera."""

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Override to add visual-only markers (used only when ``debug``)."""

    # ------------------------------------------------------------------ #
    # Model build: compile once -> batched upload -> per-world rows
    # ------------------------------------------------------------------ #
    def _build_model(self, spec: mujoco.MjSpec, njmax, nconmax) -> None:
        m = spec.compile()
        if m.opt.noslip_iterations > 0:
            warnings.warn(f"disabling the noslip solver (noslip_iterations={int(m.opt.noslip_iterations)}): "
                          "mujoco_warp does not implement it, so batched physics will "
                          "differ slightly from a CPU run of the same model.", UserWarning, stacklevel=2)
        m.opt.noslip_iterations = 0                       # mujoco_warp has no noslip solver
        # The offscreen framebuffer must fit render()'s frames
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), RENDER_W)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), RENDER_H)
        self.m = m

        with wp.ScopedDevice(self.device):
            self.gpu_model = mjw.put_model(m)
            self.gpu_data = mjw.make_data(
                m, nworld=self.num_envs, nconmax=nconmax, njmax=njmax)
            for name in self.PER_WORLD_MODEL_FIELDS + self.DERIVED_PER_WORLD_FIELDS:
                self._make_field_per_world(name)
            self._fill_world_rows()
            # Per world, so set_const gives each design its own solver
            # tolerance scale - a shared one is filled from world 0 only
            # Older MuJoCo Warp forks keep a shared model statistic; 4.0
            # computes its inertia scale per world from the current mass matrix.
            if hasattr(self.gpu_model, "stat"):
                mi = self.gpu_model.stat.meaninertia
                self.gpu_model.stat.meaninertia = wp.array(np.full(self.num_envs, float(mi.numpy()[0]), np.float32), dtype=mi.dtype, device=mi.device)
            mjw.set_const(self.gpu_model, self.gpu_data)
            mjw.forward(self.gpu_model, self.gpu_data)
            wp.synchronize()

    def _refresh_constants(self) -> None:
        """Recompute the mass-derived constants (``body_subtreemass``,
        ``*_invweight0``) for every world's current design.

        The deform writes masses and inertias but not these, so call it after a
        reset that changed designs. Keeps the state. Allocates, so never call it
        inside the step graph.
        """
        with self._wp_scope():
            mjw.set_const(self.gpu_model, self.gpu_data)

    def _make_field_per_world(self, name: str) -> None:
        """Replace the shared row of ``gpu_model.<name>`` with one copy per world."""
        shared = getattr(self.gpu_model, name)
        if shared.ndim < 2 or shared.shape[0] != 1:
            raise ValueError(f"gpu_model.{name} has shape {shared.shape}. Only a "
                             f"field with one shared row can be made per world.")
        rows = np.repeat(shared.numpy(), self.num_envs, axis=0)
        setattr(self.gpu_model, name, wp.array(rows, dtype=shared.dtype, device=shared.device))

    def _init_control_ranges(self) -> None:
        rng = self.m.actuator_ctrlrange.astype(np.float32)
        # Maps [-1, 1] onto each actuator's ctrl range
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
        """Add a launch to the step program, before the physics substeps.

        ``fn`` takes no arguments and may only launch kernels: no allocation, no
        host reads, no branching on device data.
        """
        self._step_launches.append(fn)
        self.invalidate_graph()

    def register_post_step_launch(self, fn) -> None:
        """Add a launch to the step program, after kinematics/com_vel."""
        self._post_step_launches.append(fn)
        self.invalidate_graph()

    def invalidate_graph(self) -> None:
        """Drop the captured graph, the next step recaptures. Call after changing
        anything baked in at capture (launch list, constant arguments,
        ``phys_substeps``)."""
        self._graph = None

    def _run_step_program(self) -> None:
        for fn in self._step_launches:
            fn()
        for _ in range(self.phys_substeps):
            mjw.step(self.gpu_model, self.gpu_data)
        # mjw.step integrates after forward(), so poses would be one substep stale.
        # Body velocities too: cvel from before the last integration against a pose
        # from after it would make an object held in the palm look like it moves
        mjw.kinematics(self.gpu_model, self.gpu_data)
        mjw.com_pos(self.gpu_model, self.gpu_data)
        mjw.com_vel(self.gpu_model, self.gpu_data)
        for fn in self._post_step_launches:
            fn()

    def _ensure_graph(self) -> None:
        """Capture the step program, running it once first to compile kernels.

        Capture only records, so only the warm-up changes the sim state. The first
        capture is therefore driven from ``reset()``, just before the state is
        rewritten. Capture runs on warp's stream (CUDA cannot capture torch's
        legacy default stream), and ``_step_logic`` replays it on torch's.
        """
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
        self._ensure_graph()               # warm-up scribbles, overwritten below
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
        """Advance all worlds one control step: ``(obs, reward, terminated, truncated, info)``.

        Autoreset is Gymnasium's NEXT_STEP: a world that ended on the previous call is respawned on this one and
        returns its first observation, with reward 0, ``terminated``/``truncated`` False and this call's action
        unused. The step program still advances every world; for the respawning worlds that move is discarded, as
        the respawn runs after it, outside the graph.
        """
        self._stage_actions(ctrl_actions, reset_actions)
        had_reset = self._reset_pending

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
            "final_steps": self.world_steps // self.phys_substeps,   # control steps
        }
        self.reset_mask.copy_(done)
        # The step's one host sync: whether the next step has worlds to respawn
        self._reset_pending = bool(done.any())
        return obs, reward, terminated, truncated, info

    def _launch_autorespawn(self) -> None:
        """Respawn the worlds in ``reset_mask`` at an autoreset. Override to add per-episode launches."""
        self._launch_respawn()

    def _rebase_episode(self, mask: torch.Tensor) -> None:
        """Start the reward's per-episode state for the worlds in ``mask`` from their respawn. Override if any."""

    def _respawn_pending(self, obs, reward, terminated, truncated):
        """Respawn the worlds in ``reset_mask`` (they ended on the previous call) and give them their first
        observation, reward 0 and no termination. Every other world's outputs are already computed, and the extra
        ``forward`` cannot change their physics: the next ``mjw.step`` recomputes it from the same state. The
        first observation runs the task kernels on the respawned worlds only, so the others' task state does not
        tick twice."""
        respawned = self.reset_mask.clone()
        with self._wp_scope():
            self._launch_autorespawn()
            wp.to_torch(self.gpu_data.ctrl)[respawned] = 0
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps.masked_fill_(respawned, 0)
        self.ep_return.masked_fill_(respawned, 0.0)
        diverged = self._diverged
        fresh = self._compute_obs(only_reset=True)
        self._diverged = torch.where(respawned, self._diverged, diverged)
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
    # Rendering
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
        m, d = self._render_m, self._render_d
        self._mirror_render_model(m, env_idx)
        with self._wp_scope():
            mjw.get_data_into(d, m, self.gpu_data, env_idx)
        mujoco.mj_forward(m, d)
        self.renderer.update_scene(d, camera=self.camera)
        if self.debug:
            self._add_debug_visualizations(self.renderer.scene, env_idx)
        return self.renderer.render().copy()

    def _register_render_cleanup(self) -> None:
        """Free this env's GL contexts at exit, before EGL shuts down.

        ``mujoco.egl`` registers ``eglTerminate`` with atexit when the first
        context is created, and a ``Renderer`` finalised after it raises from its
        destructor. atexit runs handlers in reverse order, and this one is always
        registered after that, so it runs first. It captures the renderer list,
        not ``self``, so it does not keep the env alive.
        """
        if self._render_cleanup_registered:
            return
        atexit.register(_close_renderers, self._renderers)
        self._render_cleanup_registered = True

    def close(self) -> None:
        """Free the offscreen GL contexts. Idempotent, and the atexit handler covers
        a skipped call."""
        close_human(self)
        _close_renderers(self._renderers)
        self.renderer = self.camera = None
        self._render_m = self._render_d = None


def _close_renderers(renderers: list) -> None:
    while renderers:
        renderer = renderers.pop()
        try:
            renderer.close()
        except Exception:
            pass                       # runs at exit, nothing useful left to do


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
