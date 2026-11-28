"""Batched MuJoCo environment: N worlds stepped together on one GPU.

Build one base ``MjSpec``, copy it per world, let a subclass mutate each copy,
compile them in parallel, and upload a single batched model+data through
``mujoco_warp`` with the differing fields stacked per world. Then step every
world at once.

The step program
----------------
A control step is a *fixed, ordered list of kernel launches* -- the step
program -- held in ``_step_launches`` plus the physics substeps. It is defined
in exactly one place (``_run_step_program``), and both execution paths use it:
the CUDA-graph path records it once and replays it with a single
``capture_launch``, and the un-captured fallback calls it directly. The two can
therefore never drift, and ``use_graph=False`` is a free debugging mode.

Adding a per-step feature is one call to ``register_step_launch``. There is no
capture method to override, no eager fallback to mirror, and no init-ordering
puzzle: capture is *lazy*, so a subclass may allocate buffers and register
launches at any point in its own ``__init__``.

What goes where
---------------
The rule that makes the above work, and the only one a subclass author needs:

* **Inside the graph** -- work that happens unconditionally every step
  (physics, per-step kernels, the masked per-world reset). Per-world "should I
  do this?" decisions are made by a mask *inside* the kernel; idle threads
  retire immediately, which is cheaper than the host round-trip needed to skip
  the launch. Anything that varies per step must arrive through a preallocated
  device buffer, because launch arguments are frozen when the graph is
  captured.
* **Outside the graph** -- host-gated, data-dependent work: obs, reward, the
  ``info`` dict. These stay in torch, where they are easy to iterate on.

Two tiers of parameter follow from the same constraint. *Capture-time
constants* (counts, body/joint addresses, substep count) are ordinary kernel
arguments; changing one calls ``invalidate_graph()`` and the next step
recaptures. *Live tunables* (anything a curriculum schedules) live in a small
device array the kernels read, so changing one is a single device write that
needs no recapture and no sync.

Mesh generation / decomposition / deformation is intentionally absent: a
"variant" differs from its siblings by spec *values* (geom sizes, poses,
inertia, ...), never by swapped mesh assets.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import os

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

# Offscreen render resolution (also used to size the models' framebuffers).
RENDER_H, RENDER_W = 480, 480
# Live offscreen GL contexts held at once, one per world being rendered.
RENDER_SLOT_LIMIT = 8

# Model fields that may legitimately differ between variants once only spec
# *values* change (no meshes). We stack these per world so every world sims with
# its own geometry and inertia. Dtype/shape are read back from the batched model
# itself, so we never hard-code a warp dtype.
PER_ENV_MODEL_ATTRS = (
    "qpos0",
    "geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound",
    "geom_friction", "geom_solref", "geom_solimp", "geom_rgba",
    "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_mass", "body_subtreemass", "body_inertia", "body_invweight0",
    "dof_invweight0",
)


def copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy an action batch into a persistent device buffer.

    Coerces device and dtype, and tolerates a caller that supplied one row per
    world-group rather than one per world (a single row is broadcast; a longer
    batch is truncated), which is what the demo scripts pass.
    """
    src = src.to(device=dst.device, dtype=dst.dtype)
    if src.shape != dst.shape:
        src = src.expand(dst.shape) if src.shape[0] == 1 else src[:dst.shape[0]]
    dst.copy_(src)


class ParallelMujocoEnv:
    """Steps ``num_envs`` MuJoCo worlds in lockstep on one device.

    Gymnasium-style ``reset() -> obs`` / ``step(...) -> (obs, reward,
    terminated, truncated, info)`` over batched tensors of leading dim
    ``num_envs``. Each world runs its own episode; a world that ends is reset at
    the *start* of the next call, by the reset kernel at the head of the step
    program, so it reports its true terminal obs first and its new episode is
    configured by the next call's ``embodiment``.
    """

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
    ):
        wp.init()
        self.num_envs = num_envs
        self.device = device
        from core.mujoco_rendering import validate_render_mode
        validate_render_mode(render_mode)
        self.render_mode = render_mode
        self.phys_substeps = phys_substeps
        self.viz_env_id = viz_env_id
        self.max_episode_steps = max_episode_steps
        self.debug = debug
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # warp and torch keep *separate* CUDA streams, so warp launches are not
        # ordered against the torch work touching the same buffers. Running
        # every warp launch on torch's stream makes that ordering implicit --
        # correct and cheaper than a host-side wp.synchronize() per step.
        # Pass the device explicitly: current_stream() with no argument returns
        # a stream on torch's *default* device, and warp binds the stream's
        # device, not self.device -- without this every world steps on cuda:0.
        self._is_cuda = device != "cpu" and wp.get_device(device).is_cuda
        self._wp_stream = (
            wp.stream_from_torch(
                torch.cuda.current_stream(torch.device(device)))
            if self._is_cuda else None)

        # Step program + its lazily captured graph.
        self._step_launches: list = []
        self._graph = None
        self._warmed_up = False
        self._use_graph = bool(use_graph) and self._is_cuda

        # Action staging buffers, allocated by the subclass once it knows its
        # own dimensions (see _alloc_action_buffers). Kernels read these, never
        # the caller's tensors, because launch args freeze at capture.
        self._act_buf: torch.Tensor | None = None
        self._embodiment_buf: torch.Tensor | None = None

        # Render/viewer scratch state (filled lazily; see render()/mjview_utils).
        self._render_m: mujoco.MjModel | None = None
        self._render_d: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self._renderer_m: mujoco.MjModel | None = None
        self.camera: mujoco.MjvCamera | None = None
        # One (MjData, Renderer, camera) per rendered world. Every world has its
        # own compiled MjModel, so a single slot keyed on "is this the model I
        # last drew?" is destroyed and rebuilt on every call as soon as more
        # than one world is rendered in rotation -- which is exactly what a
        # video batch does. Rebuilding a Renderer allocates a GL context, so
        # that thrash costs ~11x per frame single-process and far more when
        # several processes contend for the driver.
        self._render_slots: dict[int, tuple] = {}
        self._render_cleanup_registered = False

        # Simulation state, populated by _build_model_variants().
        self._base_spec: mujoco.MjSpec | None = None
        self.variants: list[mujoco.MjModel] = []
        self.m: mujoco.MjModel | None = None
        self.gpu_model = None
        self.gpu_data = None

        # Per-world episode bookkeeping.
        self.world_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_return = torch.zeros(num_envs, device=device)
        # Worlds the *next* step program will respawn. Device-resident and never
        # rebound: the reset kernel holds a warp view of it. Its host-side
        # summary rides along on the done.any() sync step() already pays for.
        self.reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reset_pending = False

        self._build_model_variants(self._build_base_spec(), njmax, nconmax)
        self._init_control_ranges()

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        """Return the assembled, *unmodified* base spec (shared by all envs)."""
        raise NotImplementedError

    def _modify_variant_spec(self, v_spec: mujoco.MjSpec, env_idx: int) -> None:
        """Mutate ``v_spec`` in place to make env ``env_idx`` differ. Spec values
        only (geom sizes/poses, masses, ...); no mesh swaps."""

    def _launch_reset(self) -> None:
        """Launch the per-world reset kernel, masked by ``self.reset_mask``.

        Launch-only, like everything in the step program: it is recorded into
        the graph *and* called directly by ``reset()``, so it must not read back
        to the host or branch on device data.
        """
        raise NotImplementedError

    def _compute_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _reward_terminated_truncated(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reward, terminated, truncated), each [num_envs].

        ``terminated`` = the episode genuinely ended (success or failure) -> the
        learner must NOT bootstrap. ``truncated`` = an artificial cutoff (e.g. a
        time limit) where the task would have continued -> the learner SHOULD
        bootstrap ``V(s')`` using the terminal obs in ``info['final_obs']``.
        """
        raise NotImplementedError

    # Filled by the env's reward computation with the per-step info keys.
    _step_info: dict = {}

    def _make_render_model(self) -> mujoco.MjModel:
        """Compile a fresh CPU MjModel for an offscreen render slot."""
        render_m = self._base_spec.copy().compile()
        render_m.opt.noslip_iterations = 0
        render_m.vis.global_.offwidth = max(int(render_m.vis.global_.offwidth), RENDER_W)
        render_m.vis.global_.offheight = max(int(render_m.vis.global_.offheight), RENDER_H)
        render_m.geom_sameframe[:] = self.m.geom_sameframe
        return render_m

    def _mirror_render_model(
        self, env_idx: int | None = None, render_m: mujoco.MjModel | None = None
    ) -> None:
        """Copy one world's device-side geometry back into its CPU ``MjModel``.

        ``env_idx`` selects the world to mirror; ``None`` means the viewer world
        (``viz_env_id``). The reset kernel deforms only the device-side model, so
        without this the CPU model -- what ``render()`` draws -- keeps the
        compile-time shape posed with the deformed body's qpos.
        """
        w = self.viz_env_id if env_idx is None else int(env_idx)
        if render_m is None:
            if len(self.variants) == self.num_envs:
                render_m = self.variants[w]
            elif w in self._render_slots:
                render_m = self._render_slots[w][0]
            else:
                render_m = self.m
        v, gm = render_m, self.gpu_model
        for name in ("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb", "body_pos", "body_ipos", "body_iquat",
                     "body_mass", "body_inertia", "site_pos"):
            src = getattr(gm, name, None)
            if src is None:
                continue
            rows = wp.to_torch(src)
            if rows.shape[0] < self.num_envs:           # shared by every world: nothing per-world to mirror
                continue
            getattr(v, name)[:] = rows[w].reshape(getattr(v, name).shape).cpu().numpy()
        # Subtree masses follow the mirrored masses (a tracking camera aims at the
        # subtree COM, sum(m * xipos) / subtreemass). Accumulated by hand: mj_setConst
        # would also reset stat.extent and geom_sameframe, which breaks the render.
        sub = v.body_mass.copy()
        for bid in range(v.nbody - 1, 0, -1):
            sub[v.body_parentid[bid]] += sub[bid]
        v.body_subtreemass[:] = sub
        # The visualizer culls against bvh_aabb, baked at compile time; a one-geom
        # body's box is just its geom's aabb.
        for bid in range(v.nbody):
            adr = int(v.body_bvhadr[bid])
            if adr >= 0 and v.body_bvhnum[bid] == 1 and v.body_geomnum[bid] == 1:
                v.bvh_aabb[adr] = v.geom_aabb[int(v.body_geomadr[bid])]

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """Override to aim the viewer/offscreen camera."""

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Override to inject visual-only markers (only used when debug)."""

    # ------------------------------------------------------------------ #
    # Model build: copy -> per-variant mutate -> compile -> batched upload
    # ------------------------------------------------------------------ #
    def _build_model_variants(self, base_spec, njmax, nconmax) -> None:
        self._base_spec = base_spec
        has_variant_hook = (
            type(self)._modify_variant_spec is not ParallelMujocoEnv._modify_variant_spec
        )
        if not has_variant_hook:
            # Fast single-model path: all worlds start from the same compiled MjModel
            # and deform per-world rows directly on device.
            m0 = base_spec.copy().compile()
            if int(m0.opt.noslip_iterations) > 0:
                print(f"[parallel_env] WARNING: disabling noslip solver "
                      f"(noslip_iterations={int(m0.opt.noslip_iterations)}) -- mujoco_warp does not "
                      f"implement it, so batched physics will differ slightly from "
                      f"a CPU run of the same model.")
            m0.opt.noslip_iterations = 0
            m0.vis.global_.offwidth = max(int(m0.vis.global_.offwidth), RENDER_W)
            m0.vis.global_.offheight = max(int(m0.vis.global_.offheight), RENDER_H)
            self.variants = [m0]
            self.m = m0
        else:
            # 1. One mutated spec copy per env, compiled in parallel.
            variant_specs = []
            for i in range(self.num_envs):
                v_spec = base_spec.copy()
                self._modify_variant_spec(v_spec, i)
                variant_specs.append(v_spec)

            workers = min(self.num_envs, os.cpu_count() or 1)
            if workers <= 1:
                self.variants = [s.compile() for s in variant_specs]
            else:
                self.variants = [None] * self.num_envs
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(s.compile): i for i, s in enumerate(variant_specs)}
                    for fut in concurrent.futures.as_completed(futs):
                        self.variants[futs[fut]] = fut.result()

            dropped = max(int(m.opt.noslip_iterations) for m in self.variants)
            if dropped > 0:
                print(f"[parallel_env] WARNING: disabling noslip solver "
                      f"(noslip_iterations={dropped}) -- mujoco_warp does not "
                      f"implement it, so batched physics will differ slightly from "
                      f"a CPU run of the same model.")
            for m_var in self.variants:
                m_var.opt.noslip_iterations = 0
                m_var.vis.global_.offwidth = max(int(m_var.vis.global_.offwidth), RENDER_W)
                m_var.vis.global_.offheight = max(int(m_var.vis.global_.offheight), RENDER_H)

            self.m = self.variants[0]

        # 2. Upload a single batched model+data, then overwrite the fields that
        #    differ between variants with per-world stacks.
        with wp.ScopedDevice(self.device):
            self.gpu_model = mjw.put_model(self.m)
            self.gpu_data = mjw.make_data(
                self.m, nworld=self.num_envs, nconmax=nconmax, njmax=njmax)
            for attr in PER_ENV_MODEL_ATTRS:
                self._make_field_per_world(attr)
            # Older MuJoCo Warp forks keep a shared model statistic; 4.0
            # computes its inertia scale per world from the current mass matrix.
            if hasattr(self.gpu_model, "stat"):
                mi = self.gpu_model.stat.meaninertia
                self.gpu_model.stat.meaninertia = wp.array(np.full(self.num_envs, float(mi.numpy()[0]), np.float32), dtype=mi.dtype, device=mi.device)
            mjw.set_const(self.gpu_model, self.gpu_data)
            mjw.forward(self.gpu_model, self.gpu_data)
            wp.synchronize()

    def _refresh_constants(self) -> None:
        """Recompute what MuJoCo derives from masses at compile time --
        ``body_subtreemass`` and ``body/dof_invweight0`` (constraint softness)
        for every world's current geometry. The deform writes masses and inertias but not these,
        so call it after a reset that changed designs. State is kept (set_const
        restores qpos); allocates, so never inside the step graph."""
        with self._wp_scope():
            mjw.set_const(self.gpu_model, self.gpu_data)

    def _make_field_per_world(self, attr: str) -> None:
        """Replace ``gpu_model.<attr>`` with a [num_envs, ...] stack of the same
        field across the compiled variants.

        Dtype and per-world layout are read back from the batched model's own
        array, so this works for scalar, vec3, quat, vec2 and vec5 fields and
        the packed ``geom_aabb`` without any hard-coded warp dtypes.
        """
        existing = getattr(self.gpu_model, attr, None)
        if existing is None:
            return
        try:
            if len(self.variants) == 1:
                dst = wp.empty(
                    (self.num_envs, *existing.shape[1:]),
                    dtype=existing.dtype,
                    device=existing.device,
                )
                dst_t = wp.to_torch(dst)
                dst_t.copy_(wp.to_torch(existing).expand_as(dst_t))
                setattr(self.gpu_model, attr, dst)
                return
            ref = existing.numpy()                 # shape (1, *per_world)
            per_world_shape = ref.shape[1:]
            stacked = np.stack([
                np.asarray(getattr(v, attr)).reshape(per_world_shape)
                for v in self.variants
            ]).astype(ref.dtype, copy=False)
            setattr(self.gpu_model, attr, wp.array(stacked, dtype=existing.dtype, device=existing.device))
        except Exception as e:  # pragma: no cover - defensive, per-field
            print(f"[parallel_env] skipping per-world field '{attr}': {e}")

    def _init_control_ranges(self) -> None:
        rng = self.m.actuator_ctrlrange.astype(np.float32)
        # Remap of [-1, 1] onto each actuator's ctrl range, as device arrays so
        # the in-graph ctrl kernel can read them.
        self.ctrl_scale = torch.as_tensor((rng[:, 1] - rng[:, 0]) / 2, device=self.device)
        self.ctrl_mid = torch.as_tensor((rng[:, 1] + rng[:, 0]) / 2, device=self.device)

    def _alloc_action_buffers(self, act_dim: int, embodiment_dim: int = 0) -> None:
        """Allocate the persistent buffers ``step()`` stages actions into."""
        self._act_buf = torch.zeros((self.num_envs, act_dim), dtype=torch.float32, device=self.device)
        self._embodiment_buf = None if embodiment_dim <= 0 else torch.zeros((self.num_envs, embodiment_dim), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def _wp_scope(self):
        """Device scope for warp launches, on torch's stream when there is one.

        Everything that reads or writes the batched state -- warp kernels here,
        torch ops in obs/reward -- then runs on a single stream, so the two
        libraries are ordered against each other without a host-side
        ``wp.synchronize()``.
        """
        dev = wp.ScopedDevice(self.device)
        return dev if self._wp_stream is None else _Nested(dev, wp.ScopedStream(self._wp_stream))

    # Set by an env whose obs/reward read cfrc_ext (see _run_step_program).
    rne_post_constraint = False

    def register_step_launch(self, fn) -> None:
        """Add a kernel launch to the step program, before the physics substeps.

        ``fn`` takes no arguments and must only launch kernels -- no allocation,
        no host reads, no branching on device data -- since it is recorded into
        a CUDA graph. Per-step state belongs in device buffers it closes over.
        """
        self._step_launches.append(fn)
        self.invalidate_graph()

    def invalidate_graph(self) -> None:
        """Drop the captured graph; the next step recaptures the program.

        Call after changing anything baked in at capture time (the launch list,
        a kernel's constant arguments, ``phys_substeps``). Live tunables do not
        need this -- that is the point of them.
        """
        self._graph = None

    def _run_step_program(self) -> None:
        """One control step, in order. The single definition of a step."""
        for fn in self._step_launches:
            fn()
        for _ in range(self.phys_substeps):
            mjw.step(self.gpu_model, self.gpu_data)
        # Contact/constraint forces per body (cfrc_ext, cfrc_int) are not part of
        # mjw.step; an env that reads them opts in, and they are computed once per
        # control step from the last substep's forward pass, as gymnasium's
        # _step_mujoco_simulation does with mj_rnePostConstraint.
        if self.rne_post_constraint:
            mjw.rne_postconstraint(self.gpu_model, self.gpu_data)
        # mjw.step is forward() then integrate, so xpos still describes the pose
        # from before the last integration; refresh it so obs/reward are not one
        # step stale.
        mjw.kinematics(self.gpu_model, self.gpu_data)

    def _ensure_graph(self) -> None:
        """Capture the step program, warming up the JIT first if need be.

        Stream capture *records* launches without executing them, so capture
        itself is side-effect free; only the warm-up scribbles on the sim state.
        That is why the first capture is driven from ``reset()``, immediately
        before the state is rewritten -- and why a later recapture (after
        ``invalidate_graph``) is safe mid-run: the kernels are already compiled,
        so no warm-up is needed.

        Capture stays on warp's own stream -- CUDA cannot capture the legacy
        default stream torch uses -- but the graph is replayed on torch's stream
        by ``_physics_step``, which is allowed.
        """
        if not self._use_graph or self._graph is not None:
            return
        with wp.ScopedDevice(self.device):
            if not self._warmed_up:
                self._run_step_program()   # JIT + allocation, outside the capture
                self._warmed_up = True
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                self._run_step_program()
            self._graph = cap.graph

    def _step_logic(self) -> None:
        self._ensure_graph()               # no-op once captured
        with self._wp_scope():
            if self._graph is None:
                self._run_step_program()
            else:
                wp.capture_launch(self._graph)

    def _stage_actions(self, ctrl_actions, embodiment) -> None:
        """Copy this step's actions into the buffers the step program reads."""
        if self._act_buf is not None and ctrl_actions is not None:
            copy_into(self._act_buf, ctrl_actions)
        if self._embodiment_buf is not None and embodiment is not None:
            copy_into(self._embodiment_buf, embodiment)

    def reset(self) -> torch.Tensor:
        """Reset every world (full batch) and return the first observation."""
        self._ensure_graph()               # warm-up scribbles; overwritten below
        self.reset_mask.fill_(True)
        if self._embodiment_buf is not None:
            self._embodiment_buf.zero_()   # no embodiment yet -> plain default
        with self._wp_scope():
            self._launch_reset()
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps.zero_()
        self.ep_return.zero_()
        self.reset_mask.zero_()
        self._reset_pending = False
        return self._compute_obs()

    def step(self, ctrl_actions: torch.Tensor, embodiment: torch.Tensor = None):
        """Advance all worlds one control step: ``(obs, reward, terminated, truncated, info)``.

        Autoreset is Gymnasium's NEXT_STEP: a world that ended on the previous call is respawned on this one and
        returns its first observation, with reward 0, ``terminated``/``truncated`` False and this call's action
        unused. The step program still advances every world (one graph for the batch); for the respawning worlds
        that move is discarded, as the respawn runs after it, outside the graph.
        """
        self._stage_actions(ctrl_actions, embodiment)
        had_reset, self._reset_pending = self._reset_pending, False

        self._step_logic()
        self.world_steps += self.phys_substeps

        obs = self._compute_obs()
        reward, terminated, truncated = self._reward_terminated_truncated()
        if had_reset:
            obs, reward, terminated, truncated = self._respawn_pending(obs, reward, terminated, truncated)
        self.ep_return += reward

        done = terminated | truncated
        # Per-step info the env published while computing the reward (the v4 keys).
        info = dict(self._step_info)
        # The one unavoidable host sync per step: the Gymnasium contract needs a
        # host-side decision about the info dict. Its result doubles as next
        # step's "did anything reset", so nothing else has to sync.
        if bool(done.any()):
            info["final_obs"] = obs.clone()
            info["final_return"] = self.ep_return.clone()
            info["final_steps"] = self.world_steps // self.phys_substeps   # control steps
            self._reset_pending = True
        self.reset_mask.copy_(done)
        return obs, reward, terminated, truncated, info

    # Convenience accessors onto the batched device state (zero-copy views).
    def _respawn_pending(self, obs, reward, terminated, truncated):
        """Respawn the worlds in ``reset_mask`` (they ended on the previous call) and give them their first
        observation, reward 0 and no termination. Every other world's outputs are already computed, and the extra
        ``forward`` cannot change their physics: the next ``mjw.step`` recomputes it from the same state."""
        respawned = self.reset_mask.clone()
        with self._wp_scope():
            self._launch_respawn()
            wp.to_torch(self.gpu_data.ctrl)[respawned] = 0
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps.masked_fill_(respawned, 0)
        self.ep_return.masked_fill_(respawned, 0.0)
        if self.render_mode is not None:
            self._mirror_render_model()
        diverged = self._diverged
        fresh = self._compute_obs()
        self._diverged = torch.where(respawned, self._diverged, diverged)
        obs = torch.where(respawned[:, None], fresh, obs)
        return obs, reward.masked_fill(respawned, 0.0), terminated & ~respawned, truncated & ~respawned

    def qpos_torch(self) -> torch.Tensor:
        return wp.to_torch(self.gpu_data.qpos)

    def qvel_torch(self) -> torch.Tensor:
        return wp.to_torch(self.gpu_data.qvel)

    def xpos_torch(self) -> torch.Tensor:
        return wp.to_torch(self.gpu_data.xpos)

    def _assert_finite(self, t: torch.Tensor, name: str) -> None:
        """ Checks for NaNs/infs/-infs and raises an error on one. """
        if not bool(torch.isfinite(t).all()):
            raise RuntimeError(
                f"{name} at world step {self.world_steps} is not finite: {t}")

    # ------------------------------------------------------------------ #
    # Offscreen rendering of a single world (headless video path)
    # ------------------------------------------------------------------ #
    def render(self, env_idx: int | None = None) -> np.ndarray:
        if env_idx is None:
            env_idx = self.viz_env_id

        slot = self._render_slots.get(env_idx)
        if slot is None:
            while len(self._render_slots) >= RENDER_SLOT_LIMIT:
                _, evicted = self._render_slots.popitem()
                evicted[2].close()
            render_m = (
                self.variants[env_idx]
                if len(self.variants) == self.num_envs
                else self._make_render_model()
            )
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(render_m, camera)
            self._setup_camera(camera)
            slot = (render_m, mujoco.MjData(render_m),
                    mujoco.Renderer(render_m, RENDER_H, RENDER_W), camera)
            self._render_slots[env_idx] = slot
            self._register_render_cleanup()
        render_m, data, renderer, camera = slot
        # Mirror *this* world's device-side model changes before drawing it.
        self._mirror_render_model(env_idx, render_m)
        # Kept in step with the active slot: mjview_utils drives the interactive
        # viewer through these attributes.
        self._render_m, self._render_d = render_m, data
        self.renderer, self._renderer_m, self.camera = renderer, render_m, camera

        with self._wp_scope():
            mjw.get_data_into(data, render_m, self.gpu_data, env_idx)
        mujoco.mj_forward(render_m, data)
        renderer.update_scene(data, camera=camera)
        if self.debug:
            self._add_debug_visualizations(renderer.scene, env_idx)
        return renderer.render().copy()

    def _register_render_cleanup(self) -> None:
        """Arrange for this env's GL contexts to be freed before EGL shuts down.

        ``mujoco.egl`` registers ``atexit(eglTerminate, EGL_DISPLAY)`` when the
        first GL context of the process is created. A ``Renderer`` still alive at
        interpreter exit is then finalized *after* that -- its ``__del__`` calls
        ``eglDestroyContext`` on a display that no longer exists, and PyOpenGL
        raises ``EGLError(EGL_NOT_INITIALIZED)`` out of a destructor, where it
        can only be printed and ignored.

        ``atexit`` runs handlers last-registered-first, and this one is always
        registered after the ``eglTerminate`` it has to precede: the display is
        created by the same ``Renderer(...)`` call that just returned, at the
        latest. Closing here is therefore ordered before the terminate.

        The handler captures the slot dict rather than ``self``, so registering
        it does not keep the environment alive to the end of the process.
        """
        if self._render_cleanup_registered:
            return
        atexit.register(_close_render_slots, self._render_slots)
        self._render_cleanup_registered = True

    def close(self) -> None:
        """Free the offscreen GL contexts. Idempotent, and safe to skip.

        Worth calling explicitly at the end of a long-lived process -- a tuning
        trial holds renderers for both contours -- but the atexit handler above
        covers the case where nobody does.
        """
        from core.mujoco_rendering import close_human
        close_human(self)
        _close_render_slots(self._render_slots)
        self.renderer = self._renderer_m = None
        self._render_m = self._render_d = None


def _close_render_slots(slots: dict) -> None:
    """Close every cached renderer, tolerating a context already torn down."""
    while slots:
        _, slot = slots.popitem()
        try:
            slot[2].close()
        except Exception:
            # Nothing useful is left to do about a context that cannot be freed,
            # and this runs at exit, where raising would only add noise.
            pass


class _Nested:
    """Enter/exit several context managers as one (device + stream scope)."""

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
