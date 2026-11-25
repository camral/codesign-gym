"""Chain animation for the LQ structure environment.

Draws the wall-anchored mass-spring-damper chain: masses on a horizontal rail, springs
as zig-zags, the wall hatched on the left, with actuated masses and sensed masses marked.
Displacements are the (scaled) positions from the current state. This is render-only
internal machinery reachable through ``LQStructureEnv.render()``; it is not a user-facing
plotting API.
"""

from __future__ import annotations

import numpy as np

_REST_SPACING = 1.0
_DISP_SCALE = 0.6  # visual magnification of position displacements


def _spring_path(x0, x1, y=0.0, coils=6, amp=0.12):
    """Return (xs, ys) of a zig-zag spring between x0 and x1 at height y."""
    n = max(2, coils * 2)
    xs = np.linspace(x0, x1, n + 2)
    ys = np.full_like(xs, y)
    for k in range(1, n + 1):
        ys[k] = y + amp * (1 if k % 2 else -1)
    return xs, ys


class ChainRenderer:
    """Persistent matplotlib figure that renders successive chain states."""

    def __init__(self, n: int, mode: str, fps: int = 20):
        import matplotlib
        if mode == "rgb_array":
            matplotlib.use("Agg", force=False)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self.n = n
        self.mode = mode
        self.fps = fps
        self._pyplot = None

        if mode == "human":
            import matplotlib.pyplot as plt
            self._pyplot = plt
            self.fig, self.ax = plt.subplots(figsize=(1.6 * (n + 1), 2.6))
        else:
            self.fig = Figure(figsize=(1.6 * (n + 1), 2.6))
            self.canvas = FigureCanvasAgg(self.fig)
            self.ax = self.fig.add_subplot(111)

    def render(self, state: np.ndarray, embodiment: dict) -> np.ndarray | None:
        n = self.n
        q = np.asarray(state)[:n]
        masses = np.asarray(embodiment["masses"], dtype=float)
        actuators = np.asarray(embodiment["actuators"]).astype(bool)
        sensors = np.asarray(embodiment["sensors"]).astype(bool)

        ax = self.ax
        ax.clear()
        rest = _REST_SPACING * np.arange(1, n + 1)
        xpos = rest + _DISP_SCALE * q

        # Wall on the left.
        ax.axvline(0.0, color="0.3", lw=3)
        ax.add_patch(
            _rect(-0.18, -0.9, 0.18, 1.8, hatch="////", fill=False, ec="0.5")
        )

        # Springs: wall->m1, then m_{i-1}->m_i.
        prev = 0.0
        for i in range(n):
            xs, ys = _spring_path(prev, xpos[i])
            ax.plot(xs, ys, color="0.55", lw=1.2, zorder=1)
            prev = xpos[i]

        # Masses (size scaled by mass value), actuated ones filled.
        mmin, mmax = masses.min(), masses.max()
        for i in range(n):
            frac = 0.5 if mmax == mmin else (masses[i] - mmin) / (mmax - mmin)
            size = 0.18 + 0.14 * frac
            color = "#d1495b" if actuators[i] else "#3d5a80"
            ax.add_patch(_rect(xpos[i] - size / 2, -size / 2, size, size,
                               fill=True, fc=color, ec="k", zorder=3))
            if actuators[i]:
                ax.annotate("", xy=(xpos[i], -size / 2 - 0.28),
                            xytext=(xpos[i], -size / 2 - 0.05),
                            arrowprops=dict(arrowstyle="-|>", color="#d1495b", lw=1.6))
            if sensors[i]:
                ax.plot([xpos[i]], [size / 2 + 0.22], marker="v", ms=8,
                        color="#2a9d8f", zorder=4)

        ax.set_xlim(-0.4, _REST_SPACING * (n + 1))
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        ax.set_xlabel("position along chain")
        ax.set_title("LQ structure — red=actuated, ▼=sensor")
        self.fig.tight_layout()

        if self.mode == "human":
            self.fig.canvas.draw_idle()
            self._pyplot.pause(1.0 / max(self.fps, 1))
            return None

        self.canvas.draw()
        buf = np.asarray(self.canvas.buffer_rgba())
        return buf[..., :3].copy()

    def close(self):
        if self.mode == "human" and self._pyplot is not None:
            self._pyplot.close(self.fig)


def _rect(x, y, w, h, **kw):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, **kw)
