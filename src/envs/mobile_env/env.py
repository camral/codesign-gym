"""Single-instance mobile-env co-design environment.

A ``core.codesign_env.CoDesignEnv`` that wraps one mobile-env ``MComCore``
(central handler) and adds the Codesign Gym extensions: an ``embodiment_space``
over the base-station deployment, embodiment-aware ``_reset`` / ``_step`` hooks,
and action/observation **padding + masking** so a variable base-station count
presents fixed-shape spaces.

Design axis (embodiment): number/positions/tx-power of base stations. Control
policy: mobile-env's central cell-selection action (which BS each UE connects to).
Reward is mobile-env's mean UE utility, unchanged.

Base-station count is fixed at construction in mobile-env, so this env does *not*
support transition-level embodiment changes: an ``embodiment`` passed to ``step``
is ignored (with a warning on every such call), mirroring the API contract. A new design is
installed at ``reset`` by rebuilding the underlying ``MComCore``.

Padding: mobile-env's central handler sizes the action as
``MultiDiscrete([num_bs + 1] * num_ues)`` and the observation as a flat vector of
per-UE ``[connections(num_bs), snrs(num_bs), utility(1)]``. We advertise the
padded spaces at ``max_bs`` and, per step, (a) clip any action index that refers
to a non-existent (padded) station to NOOP, and (b) surface a boolean
``action_mask`` of shape ``(num_ues, max_bs + 1)`` in ``info`` so a policy can
avoid padded choices.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mobile_env.core.channels import EPSILON, OkumuraHata
from mobile_env.core.entities import UserEquipment

from core.codesign_env import CoDesignEnv

from .config import MobileEnvConfig
from . import embodiment as emb_mod


class BoundarySafeOkumuraHata(OkumuraHata):
    """Okumura-Hata channel with isolines that support stations on map boundaries."""

    def isoline(self, bs, ue_config, map_bounds, dthresh, num=32):
        width, height = map_bounds
        dummy = UserEquipment(None, **ue_config)
        isoline = []
        for theta in np.linspace(EPSILON, 2 * np.pi, num=num):
            x1, y1 = self.boundary_collison(theta, bs.x, bs.y, width, height)
            # Parametric interpolation remains finite for vertical and zero-length rays.
            xs = np.linspace(bs.x, x1, num=100)
            ys = np.linspace(bs.y, y1, num=100)

            datarates = []
            for x, y in zip(xs, ys):
                dummy.x, dummy.y = x, y
                snr = self.snr(bs, dummy)
                datarates.append(self.datarate(bs, dummy, snr))
            (indices,) = np.where(np.asarray(datarates) > dthresh)
            isoline.append((xs[np.max(indices)], ys[np.max(indices)]))
        xs, ys = zip(*isoline)
        return xs, ys


class MobileEnvCoDesign(CoDesignEnv):
    """Single mobile-network placement-and-connection co-design environment."""

    metadata = {"render_modes": ["rgb_array", "human"], "supports_transition_embodiment": False}

    def __init__(
        self,
        cfg: MobileEnvConfig,
        *,
        is_eval_env: bool = False,
        render_mode: Optional[str] = None,
    ):
        self.cfg = cfg
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        embodiment_space = emb_mod.make_embodiment_space(cfg)
        # Base sets self.embodiment_space and self.is_eval_env (and gym.Env init).
        super().__init__(embodiment_space=embodiment_space, is_eval_env=is_eval_env)
        self._base_embodiment = emb_mod.base_embodiment(cfg)

        self.num_ues = int(cfg.num_ues)
        self.max_bs = int(cfg.max_bs)

        # --- spaces (padded to max_bs) -----------------------------------
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        # --- runtime state ------------------------------------------------
        self._env = None                    # underlying MComCore (built on reset)
        self._env_embodiment: Optional[emb_mod.EmbType] = None
        self._build_seed: int = 0           # seed the underlying MComCore was built with
        self._embodiment: emb_mod.EmbType = self.base_embodiment
        self._num_bs = int(self._base_embodiment["num_bs"])
        self._valid = False
        self._rejected = False

    # ------------------------------------------------------------------ #
    # Space construction / padding
    # ------------------------------------------------------------------ #
    def _build_action_space(self) -> spaces.MultiDiscrete:
        # Central handler action is one choice per UE; its length (num_ues) is
        # fixed by the preset. Only the per-UE cardinality (max_bs + 1) grows with
        # the design, and the embodiment space bounds it at max_bs.
        return spaces.MultiDiscrete([self.max_bs + 1] * self.num_ues)

    def _build_observation_space(self) -> spaces.Box:
        # Per-UE layout padded to the design budget: [connections(max_bs), snrs(max_bs), utility(1)].
        # mobile-env's central observation is bounded to [-1, 1]; stations the design leaves out are 0.
        return spaces.Box(low=-1.0, high=1.0, shape=(self.num_ues * (2 * self.max_bs + 1),), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Codesign API helpers
    # ------------------------------------------------------------------ #
    @property
    def base_embodiment(self) -> emb_mod.EmbType:
        emb = self._base_embodiment
        return {"num_bs": int(emb["num_bs"]), "stations": np.array(emb["stations"], dtype=np.float32)}

    def validate_embodiment(self, embodiment: Mapping[str, Any]) -> bool:
        """Space check, then the network rules. Values become arrays first (num_bs keeps its number type, so 3.7 is
        still not an integer), so Gymnasium's ``contains`` does not warn about casting them."""
        if isinstance(embodiment, Mapping) and set(embodiment) == set(emb_mod.EMBODIMENT_KEYS):
            try:
                embodiment = {"num_bs": np.asarray(embodiment["num_bs"]), "stations": np.asarray(embodiment["stations"], dtype=np.float32)}
            except (TypeError, ValueError):
                return False
        return super().validate_embodiment(embodiment)

    def _validate_embodiment(self, embodiment: Mapping[str, Any]) -> bool:
        return emb_mod.validate_embodiment(embodiment, self.cfg)

    # ------------------------------------------------------------------ #
    # Observation padding / action masking
    # ------------------------------------------------------------------ #
    def _pad_obs(self, raw_obs: np.ndarray, num_bs: int) -> np.ndarray:
        """Repack mobile-env's flat obs (sized to ``num_bs``) into the padded
        ``max_bs`` layout, placing active values in the leading slots."""
        U, n, M = self.num_ues, num_bs, self.max_bs
        seg = np.asarray(raw_obs, dtype=np.float32).reshape(U, 2 * n + 1)
        conn, snr, util = seg[:, :n], seg[:, n:2 * n], seg[:, 2 * n:]
        out = np.zeros((U, 2 * M + 1), dtype=np.float32)
        out[:, :n] = conn                 # connections -> leading BS slots
        out[:, M:M + n] = snr             # snrs        -> leading BS slots
        out[:, 2 * M:] = util             # utility (1)
        flat = out.reshape(-1)
        return flat

    def _action_mask(self, num_bs: int) -> np.ndarray:
        """Boolean ``(num_ues, max_bs + 1)`` mask: NOOP (col 0) and the ``num_bs``
        real stations are valid; padded station columns are not."""
        mask = np.zeros((self.num_ues, self.max_bs + 1), dtype=bool)
        mask[:, 0] = True                 # NOOP always valid
        mask[:, 1:num_bs + 1] = True      # real stations
        return mask

    def _sanitize_action(self, action: np.ndarray, num_bs: int) -> np.ndarray:
        """Map padded/out-of-range action indices to NOOP so mobile-env never
        indexes a non-existent base station."""
        a = np.asarray(action).reshape(-1)[: self.num_ues].astype(int)
        a = np.clip(a, 0, self.max_bs)
        a[a > num_bs] = 0                 # padded station -> NOOP
        return a

    # ------------------------------------------------------------------ #
    # Underlying MComCore construction
    # ------------------------------------------------------------------ #
    def _build_mcom(self, embodiment: Mapping[str, Any], seed: int):
        from mobile_env.core.base import MComCore
        from mobile_env.core.entities import UserEquipment

        cfg = self.cfg
        config = MComCore.default_config()
        config["channel"] = BoundarySafeOkumuraHata
        config["width"], config["height"] = cfg.width, cfg.height
        config["EP_MAX_TIME"] = cfg.ep_max_time
        # Reproducible episodes: reseed the module RNGs from the (rotated) build
        # seed every reset. mobile-env's own reset(seed=...) is a no-op, so we bake
        # the seed in here and rebuild when it changes (see _reset).
        config["seed"] = int(seed)
        config["reset_rng_episode"] = True
        config["arrival_params"]["ep_time"] = cfg.ep_max_time
        config["arrival_params"]["reset_rng_episode"] = True
        config["movement_params"]["width"] = cfg.width
        config["movement_params"]["height"] = cfg.height
        config["movement_params"]["reset_rng_episode"] = True
        config["bs"] = {"bw": cfg.bw, "freq": cfg.freq, "tx": cfg.seed_tx, "height": cfg.bs_height}
        config["ue"] = {
            "velocity": cfg.velocity,
            "snr_tr": cfg.snr_tr,
            "noise": cfg.noise,
            "height": cfg.ue_height,
        }

        stations = emb_mod.stations_from_embodiment(embodiment, cfg)
        ues = [UserEquipment(i, **config["ue"]) for i in range(self.num_ues)]
        return MComCore(stations, ues, config, render_mode=self.render_mode)

    # ------------------------------------------------------------------ #
    # CoDesignEnv hook: reset
    # ------------------------------------------------------------------ #
    def _reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        embodiment: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, bool, dict]:
        gym.Env.reset(self, seed=seed)

        # The same full check as validate_embodiment (space, then the network rules), on the design as given: the
        # normalisation below would turn a num_bs of 3.7 into 3.
        if embodiment is not None and not self.validate_embodiment(embodiment):
            emb = None
        else:
            emb = self._normalize_emb(embodiment) if embodiment is not None else self.base_embodiment
        if emb is None or not self._validate_embodiment(emb):
            self._rejected = True
            self._valid = False
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, True, {}

        num_bs = int(emb["num_bs"])
        # Rebuild MComCore when the design OR the seed changes; otherwise reuse
        # (its reset reseeds the module RNGs from the build seed, so the episode is
        # reproducible). Without a seed the next build seed comes from this env's own
        # RNG -- entropy-seeded per lane, and reseeded by reset(seed=...) -- so
        # unseeded lanes and episodes differ while a seeded run replays exactly.
        build_seed = int(self.np_random.integers(2**31 - 1)) if seed is None else int(seed)
        reuse = (
            self._env is not None
            and build_seed == self._build_seed
            and emb_mod.embodiments_equal(self._env_embodiment, emb)
        )
        if not reuse:
            self._env = self._build_mcom(emb, build_seed)
            self._env_embodiment = emb
            self._build_seed = build_seed
        raw_obs, _ = self._env.reset()

        self._embodiment = emb
        self._num_bs = num_bs
        self._rejected = False
        self._valid = True

        obs = self._pad_obs(raw_obs, num_bs)
        info = {
            "embodiment_rejected": False,
            "num_bs": num_bs,
            "action_mask": self._action_mask(num_bs),
        }
        return obs, False, info

    # ------------------------------------------------------------------ #
    # CoDesignEnv hook: step
    # ------------------------------------------------------------------ #
    def _step(
        self,
        action: np.ndarray,
        embodiment: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self._valid:
            raise RuntimeError(
                "step() called on an invalid environment (embodiment was rejected "
                "at reset, or reset was never called)."
            )

        # Base stations are fixed at construction: transition-level embodiment changes
        # are unsupported. CoDesignEnv.step (metadata: supports_transition_embodiment=False)
        # already froze and warned about any supplied embodiment.
        num_bs = self._num_bs
        a = self._sanitize_action(action, num_bs)
        raw_obs, reward, terminated, truncated, _ = self._env.step(a)

        obs = self._pad_obs(raw_obs, num_bs)
        info = {
            "embodiment_rejected": False,
            "num_bs": num_bs,
            "action_mask": self._action_mask(num_bs),
            "mean_utility": float(reward),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_emb(embodiment: Mapping[str, Any]) -> emb_mod.EmbType:
        return {
            "num_bs": int(embodiment["num_bs"]),
            "stations": np.asarray(embodiment["stations"], dtype=np.float32),
        }

    # ------------------------------------------------------------------ #
    # Rendering (native mobile-env view + thin co-design overlay)
    # ------------------------------------------------------------------ #
    def _overlay_lines(self) -> list[str]:
        """Compact co-design status lines: budget, per-BS tx power, radio, mask."""
        cfg = self.cfg
        n = self._num_bs
        tx = np.asarray(self._embodiment["stations"], dtype=np.float32)[:n, 2]
        tx_str = ", ".join(f"{t:.0f}" for t in tx) if n else "-"
        return [
            f"{cfg.name}  ·  {n}/{cfg.max_bs} BS  (budget {cfg.min_bs}-{cfg.max_bs})",
            f"tx: [{tx_str}] dBm  (range {cfg.tx_min:.0f}-{cfg.tx_max:.0f})",
            f"{cfg.freq:.0f} MHz  ·  {cfg.num_ues} UEs @ {cfg.velocity:.1f} m/s",
            f"action slots: {n} live + {cfg.max_bs - n} padded (+ NOOP)",
        ]

    def _overlay_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Draw the co-design status panel onto an RGB frame (top-left). Falls back
        to the untouched frame if Pillow is unavailable."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:  # pragma: no cover - Pillow is a declared dependency
            return frame

        img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        font = ImageFont.load_default()
        lines = self._overlay_lines()

        pad = 6
        widths, heights = [], []
        for ln in lines:
            l, t, r, b = draw.textbbox((0, 0), ln, font=font)
            widths.append(r - l)
            heights.append(b - t)
        lh = max(heights) + 4
        box_w = max(widths) + 2 * pad
        box_h = lh * len(lines) + 2 * pad
        # Anchor bottom-left, in the figure's blank bottom margin, so the panel
        # does not sit over the simulation map.
        x0 = 6
        y0 = max(6, img.height - box_h - 6)
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 150))
        for i, ln in enumerate(lines):
            draw.text((x0 + pad, y0 + pad + i * lh), ln, fill=(255, 255, 255, 255), font=font)
        return np.asarray(img)

    def render(self):
        """Render mobile-env's native view with a thin co-design overlay.

        ``rgb_array`` returns an ``(H, W, 3)`` uint8 frame with the overlay drawn
        in; ``human`` shows the native pygame window and best-effort blits the same
        panel onto it. Returns ``None`` if no underlying env exists yet.
        """
        if self._env is None:
            return None
        frame = self._env.render()
        if self.render_mode == "rgb_array" and isinstance(frame, np.ndarray):
            return self._overlay_rgb(frame)
        if self.render_mode == "human":
            self._overlay_human()
        return frame

    def _overlay_human(self) -> None:  # pragma: no cover - needs a live display
        """Best-effort text panel on the native pygame window (human mode)."""
        try:
            import pygame

            win = getattr(self._env, "window", None)
            if win is None:
                return
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.SysFont(None, 18)
            lines = self._overlay_lines()
            surfs = [font.render(ln, True, (255, 255, 255)) for ln in lines]
            pad, lh = 6, 18
            box_w = max(s.get_width() for s in surfs) + 2 * pad
            box_h = lh * len(surfs) + 2 * pad
            panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 150))
            for i, s in enumerate(surfs):
                panel.blit(s, (pad, pad + i * lh))
            # bottom-left, in the blank bottom margin (clear of the map)
            win.blit(panel, (6, max(6, win.get_height() - box_h - 6)))
            pygame.display.flip()
        except Exception:
            return

    def close(self):
        if self._env is not None:
            self._env.close()
        return None
