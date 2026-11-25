"""Coastal Solarpunk Eco-Island 2.5D Isometric Microgrid & Telemetry HUD.

Renders ``env.render()`` as a living 2.5D isometric Solarpunk floating island
with a smooth 24-hour sky/lighting cycle (starry night -> warm sunrise ->
crisp daylight -> golden sunset), paired with an intuitive right-hand telemetry
sidebar:

  - **Embodiment-reactive island layout**:
      * Solar farm scales from 0 (reclaimed meadow + trees) to 4x4 tilted
        crystalline panels with a tightly fitted concrete pad.
      * Megapack battery yard scales from 0 (reclaimed meadow) to 6 cabinets
        with live LED SoC gauges and an adaptive pad footprint.
      * Industrial genset scales in 3D footprint, height, and single/twin
        exhaust stacks with rising plume (or vanishes cleanly at 0 kW).
      * Utility pylon & feeder switch to a severed ISLANDED state when off-grid.
  - **Coastal Solarpunk world**: Layered grass/soil/bedrock floating island,
    shimmering turquoise eco-pond with sandy shoreline, paved campus boulevard,
    drifting weather clouds, stylized trees, and architectural campus buildings
    whose windows glow warm gold when served or alert crimson during shortfalls.
  - **Intuitive right-hand telemetry sidebar**:
      * 2x2 KPI summary card + PV/BESS/GEN Embodiment Sizing Gauges.
      * Fixed 24-hour clock timeline (00h..24h, advancing per day on 72h runs)
        so animations fill smoothly left-to-right without x-axis jitter.
      * Power Balance (Supply vs Demand), Battery SoC (with 20% reserve band),
        and Stacked Cumulative Cost (Operating + Capex amortization in $k),
        each with a live glowing NOW marker and inline readout.

Pure matplotlib vector geometry (no external assets, deterministic, fast on Agg).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.colors as _mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

# --- Energy & HUD Palette --------------------------------------------------
C = {
    "hud_bg":     "#090F1D",
    "hud_card":   "#101A2E",
    "hud_edge":   "#223354",
    "hud_grid":   "#1A2945",
    "hud_ink":    "#F1F5F9",
    "hud_ink2":   "#94A3B8",

    "solar":      "#F59E0B",
    "solar_hi":   "#FDE68A",
    "battery":    "#10B981",
    "battery_hi": "#6EE7B7",
    "genset":     "#F97316",
    "genset_hi":  "#FDBA74",
    "grid":       "#38BDF8",
    "grid_hi":    "#BAE6FD",
    "unmet":      "#EF4444",
    "unmet_hi":   "#FCA5A5",
    "muted":      "#64748B",
}

BUILDING_KIND = {
    "hospital":   ("Hospital", "cross"),
    "hotel":      ("Hotel", "sign"),
    "school":     ("School", "flag"),
    "office":     ("Office", "flat"),
    "restaurant": ("Diner", "awning"),
}


# ========================================================================= #
# Color, formatting & 24-hour sky theme
# ========================================================================= #
def _rgb(h: str) -> np.ndarray:
    return np.array(_mcolors.to_rgb(h), dtype=float)


def _mix(c0, c1, t: float) -> Tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    a = _rgb(c0) if isinstance(c0, str) else np.asarray(c0, float)
    b = _rgb(c1) if isinstance(c1, str) else np.asarray(c1, float)
    return tuple((1.0 - t) * a + t * b)


def _fmt_dollars(v: float) -> str:
    if abs(v) >= 1000.0:
        return f"${v / 1000.0:.1f}k"
    return f"${v:,.0f}"


def _sky_theme(hour_of_day: float) -> Dict:
    """Compute smooth sky, Solarpunk island, and lighting colors for [0, 24)."""
    h = float(hour_of_day) % 24.0
    if 5.0 <= h <= 19.0:
        day = float(np.sin(np.pi * (h - 5.0) / 14.0) ** 0.72)
    else:
        day = 0.0
    twilight = float(max(
        np.exp(-((h - 6.2) / 1.35) ** 2),
        np.exp(-((h - 18.5) / 1.35) ** 2),
    ))
    night = 1.0 - day

    sky_top = _mix(_mix("#060C1A", "#569AE3", day), "#27345E", twilight * 0.55)
    sky_bot = _mix(_mix("#0F1D38", "#E3F2FD", day), "#F49A63", twilight * 0.72)

    # Coastal Solarpunk terrain: lush emerald meadow + warm earth + slate bedrock
    plat_top = _mix("#123026", "#64C782", day)
    plat_l   = _mix("#2A2018", "#8B694E", day)
    plat_r   = _mix("#1C1510", "#6B4E37", day)
    rock_l   = _mix("#121824", "#4E5D70", day)
    rock_r   = _mix("#0B0F17", "#364252", day)
    grid_ln  = _mix("#1B4738", "#50B56E", day)
    pad_fc   = _mix("#1A283D", "#EEF3F7", day)
    pad_ec   = _mix("#334B6E", "#BDCCD9", day)

    # Warm cream architectural buildings by day, crisp slate by night
    b_top = _mix("#324563", "#FFFDF9", day)
    b_l   = _mix("#1E2C42", "#D9E1E8", day)
    b_r   = _mix("#273854", "#EBF0F5", day)
    b_ec  = _mix("#4A6288", "#94A3B8", day)

    pill_bg = _mix("#0C1527", "#FFFFFF", day * 0.93)
    pill_tx = _mix("#F8FAFC", "#0F172A", day)

    if twilight > 0.38 and h < 12.0:
        phase_lbl = "SUNRISE"
    elif twilight > 0.38 and h >= 12.0:
        phase_lbl = "SUNSET"
    elif day < 0.20:
        phase_lbl = "NIGHT"
    else:
        phase_lbl = "DAYLIGHT"

    return dict(
        h=h, day=day, night=night, twilight=twilight, phase_lbl=phase_lbl,
        sky_top=sky_top, sky_bot=sky_bot,
        plat_top=plat_top, plat_l=plat_l, plat_r=plat_r, rock_l=rock_l, rock_r=rock_r,
        grid_ln=grid_ln, pad_fc=pad_fc, pad_ec=pad_ec,
        b_top=b_top, b_l=b_l, b_r=b_r, b_ec=b_ec,
        pill_bg=pill_bg, pill_tx=pill_tx,
    )


def _design_frac(state: Dict, key: str, default_max: float) -> float:
    val = float(state["embodiment"].get(key, 0.0))
    bounds = state.get("embodiment_bounds", {})
    if key in bounds:
        lo, hi = bounds[key]
        if hi > lo + 1e-6:
            return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
    return float(np.clip(val / max(default_max, 1e-6), 0.0, 1.0))


# ========================================================================= #
# Isometric 2.5D projection primitives
# ========================================================================= #
_ISO = dict(ox=0.50, oy=0.195, sx=0.445, sy=0.228, sz=0.290)


def iso(u: float, v: float, z: float = 0.0) -> Tuple[float, float]:
    return (
        _ISO["ox"] + (u - v) * _ISO["sx"],
        _ISO["oy"] + (u + v) * _ISO["sy"] + z * _ISO["sz"],
    )


def iso_poly(ax, pts_uvz, fc, ec="none", lw=0.6, alpha=1.0, zorder=5.0):
    pts = [iso(u, v, z) for (u, v, z) in pts_uvz]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder))


def iso_box(ax, u0, v0, du, dv, z0, dz, c_top, c_left, c_right, ec="none", lw=0.65, zorder=5.0):
    p000 = iso(u0, v0, z0)
    p100 = iso(u0 + du, v0, z0)
    p110 = iso(u0 + du, v0 + dv, z0)
    p010 = iso(u0, v0 + dv, z0)
    p001 = iso(u0, v0, z0 + dz)
    p101 = iso(u0 + du, v0, z0 + dz)
    p111 = iso(u0 + du, v0 + dv, z0 + dz)
    p011 = iso(u0, v0 + dv, z0 + dz)
    ax.add_patch(Polygon([p000, p100, p101, p001], closed=True, fc=c_right, ec=ec, lw=lw, zorder=zorder))
    ax.add_patch(Polygon([p000, p010, p011, p001], closed=True, fc=c_left, ec=ec, lw=lw, zorder=zorder))
    ax.add_patch(Polygon([p001, p101, p111, p011], closed=True, fc=c_top, ec=ec, lw=lw, zorder=zorder + 0.05))


def iso_shadow(ax, u0, v0, du, dv, dz, sun_dx: float, sun_dy: float, day: float, zorder=2.4):
    if day < 0.08:
        return
    su, sv = sun_dx * dz * 0.55, sun_dy * dz * 0.55
    pts = [
        (u0, v0, 0.004), (u0 + du, v0, 0.004),
        (u0 + du + su, v0 + sv, 0.004), (u0 + du + su, v0 + dv + sv, 0.004),
        (u0 + su, v0 + dv + sv, 0.004), (u0, v0 + dv, 0.004),
    ]
    iso_poly(ax, pts, fc="#0F172A", alpha=0.14 * day, zorder=zorder)


def iso_conduit(ax, pts_uvz, capacity: float, power: float, wref: float, hue: str,
                *, red: bool = False, direction: int = 1, phase: float = 0.0,
                day: float = 0.5, zorder: float = 3.5):
    pts = [np.array(iso(u, v, z), float) for (u, v, z) in pts_uvz]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    wcap = 2.6 + 6.0 * (min(capacity / wref, 1.0) if wref > 0 else 0.0)
    util = min(abs(power) / capacity, 1.0) if capacity > 1e-6 else 0.0
    pipe_bg = _mix("#16233A", "#CBD5E1", day)

    ax.plot(xs, ys, color=pipe_bg, lw=wcap + 1.3, solid_capstyle="round", solid_joinstyle="round", zorder=zorder - 0.3)
    if util <= 0.01:
        return

    col = C["unmet"] if red else hue
    hi = C["unmet_hi"] if red else _mix(hue, "#FFFFFF", 0.58)
    w = max(wcap * (0.32 + 0.68 * util), 2.0)

    bloom_alpha = (0.28 - 0.14 * day) * (0.5 + 0.5 * util)
    ax.plot(xs, ys, color=col, lw=w + 4.8, alpha=bloom_alpha, solid_capstyle="round", solid_joinstyle="round", zorder=zorder - 0.1)
    ax.plot(xs, ys, color=col, lw=w, alpha=0.95, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    ax.plot(xs, ys, color=hi, lw=max(w * 0.34, 0.9), alpha=0.85, solid_capstyle="round", solid_joinstyle="round", zorder=zorder + 0.1)

    path_pts = pts if direction >= 0 else list(reversed(pts))
    seg_lens = [float(np.hypot(*(path_pts[i + 1] - path_pts[i]))) for i in range(len(path_pts) - 1)]
    tot_len = sum(seg_lens)
    if tot_len < 0.035:
        return
    n_pulses = max(1, min(5, int(tot_len / 0.068)))
    for k in range(n_pulses):
        d_target = ((k + phase) / n_pulses) * tot_len
        acc = 0.0
        for i, sl in enumerate(seg_lens):
            if acc + sl >= d_target and sl > 1e-6:
                t = (d_target - acc) / sl
                p = path_pts[i] + (path_pts[i + 1] - path_pts[i]) * t
                u_dir = (path_pts[i + 1] - path_pts[i]) / sl
                n_dir = np.array([-u_dir[1], u_dir[0]])
                sz = 0.0072
                tip = p + u_dir * (sz * 0.85)
                w1 = p - u_dir * (sz * 0.55) + n_dir * (sz * 0.62)
                w2 = p - u_dir * (sz * 0.55) - n_dir * (sz * 0.62)
                ax.plot([w1[0], tip[0], w2[0]], [w1[1], tip[1], w2[1]],
                        color="#FFFFFF", lw=1.25, solid_capstyle="round", solid_joinstyle="round",
                        alpha=0.92, zorder=zorder + 0.3)
                break
            acc += sl


# ========================================================================= #
# Sky, Clouds, Trees, Coastal Cove & Adaptive Embodiment Hardware
# ========================================================================= #
def _draw_sky_and_celestial(ax, th: Dict, pv_on: bool, step_idx: int):
    grad = np.linspace(0.0, 1.0, 32).reshape(-1, 1)
    sky = (1.0 - grad) * np.array(th["sky_bot"]) + grad * np.array(th["sky_top"])
    ax.imshow(sky[:, None, :], extent=[0, 1, 0, 1], origin="lower", aspect="auto", zorder=0)

    if th["night"] > 0.25:
        rng = np.random.default_rng(42)
        for x, y, r, a in zip(
            rng.uniform(0.03, 0.97, 42), rng.uniform(0.60, 0.97, 42),
            rng.uniform(0.0009, 0.0022, 42), rng.uniform(0.25, 0.85, 42) * (th["night"] ** 1.4),
        ):
            ax.add_patch(Circle((x, y), r, fc="#E2E8F0", ec="none", alpha=float(a), zorder=0.5))

    arc_t = np.linspace(0.16 * np.pi, 0.84 * np.pi, 64)
    ax.plot(0.50 + 0.41 * np.cos(arc_t), 0.59 + 0.32 * np.sin(arc_t),
            color="#F59E0B" if th["day"] > 0.15 else "#64748B", lw=0.9, ls=(0, (3, 4)), alpha=0.30, zorder=0.8)

    # Drifting weather clouds
    cloud_col = _mix("#1E293B", "#FFFFFF", th["day"])
    cloud_alpha = 0.20 + 0.52 * th["day"]
    drift = (step_idx * 0.036) % 1.25 - 0.12
    for (cx0, cy0, sc) in [(0.18, 0.90, 1.02), (0.58, 0.93, 0.84), (0.86, 0.85, 0.72)]:
        cx = (cx0 + drift) % 1.15 - 0.08
        for (dx, dy, rx, ry) in ((0.0, 0.0, 0.048 * sc, 0.018 * sc),
                                 (-0.028 * sc, -0.005 * sc, 0.032 * sc, 0.014 * sc),
                                 (0.030 * sc, -0.004 * sc, 0.034 * sc, 0.013 * sc)):
            ax.add_patch(Ellipse((cx + dx, cy0 + dy), rx * 2, ry * 2,
                                 fc=cloud_col, ec="none", alpha=cloud_alpha, zorder=1.4))

    h = th["h"]
    if 5.0 <= h <= 19.0:
        ang = 0.84 * np.pi - ((h - 5.0) / 14.0) * (0.68 * np.pi)
        cx, cy = 0.50 + 0.41 * np.cos(ang), 0.59 + 0.32 * np.sin(ang)
        sun_c = _mix("#F97316", "#F59E0B", th["day"])
        hi_c = _mix("#FDBA74", "#FEF08A", th["day"])
        boost = 1.15 if pv_on else 0.78
        ax.add_patch(Circle((cx, cy), 0.050 * boost, fc=sun_c, alpha=0.12, zorder=1.0))
        ax.add_patch(Circle((cx, cy), 0.028 * boost, fc=sun_c, alpha=0.25, zorder=1.1))
        ax.add_patch(Circle((cx, cy), 0.015, fc=hi_c, zorder=1.2))
        return float(np.cos(ang)) * -0.8, 0.45
    else:
        frac = min(((h - 19.0) % 24.0) / 10.0, 1.0)
        ang = 0.84 * np.pi - frac * (0.68 * np.pi)
        cx, cy = 0.50 + 0.41 * np.cos(ang), 0.59 + 0.32 * np.sin(ang)
        ax.add_patch(Circle((cx, cy), 0.032, fc="#94A3B8", alpha=0.14, zorder=1.0))
        ax.add_patch(Circle((cx, cy), 0.015, fc="#E2E8F0", alpha=0.92, zorder=1.1))
        ax.add_patch(Circle((cx + 0.006, cy + 0.004), 0.013, fc=th["sky_top"], zorder=1.2))
        return 0.35, 0.35


def _draw_iso_tree(ax, u, v, day: float, scale: float = 1.0, zorder=6.0):
    tx, ty = iso(u, v, 0.0)
    ax.plot([tx, tx], [ty, ty + 0.014 * scale], color=_mix("#475569", "#78716C", day), lw=1.6 * scale, zorder=zorder)
    leaf_c = _mix("#14532D", "#22C55E", day * 0.78)
    leaf_hi = _mix("#166534", "#4ADE80", day * 0.78)
    ax.add_patch(Circle((tx, ty + 0.020 * scale), 0.010 * scale, fc=leaf_c, ec="none", alpha=0.92, zorder=zorder + 0.1))
    ax.add_patch(Circle((tx - 0.002 * scale, ty + 0.023 * scale), 0.006 * scale, fc=leaf_hi, ec="none", alpha=0.75, zorder=zorder + 0.15))


def _draw_adaptive_pad(ax, u0, v0, du, dv, acc: str, th: Dict):
    iso_poly(ax, [(u0, v0, 0.003), (u0 + du, v0, 0.003), (u0 + du, v0 + dv, 0.003), (u0, v0 + dv, 0.003)],
             fc=th["pad_fc"], ec=_mix(acc, th["pad_ec"], th["day"] * 0.55), lw=0.95, alpha=0.94, zorder=2.1)


def _draw_solar_farm_adaptive(ax, pv_frac: float, pv_kw: float, pv_now: float, th: Dict):
    """Adaptive Solar Farm: pad & panel grid shrink/expand; unused space becomes lush meadow + trees."""
    if pv_kw < 1.0:
        # Zero PV: reclaim entire zone with lush Solarpunk grove!
        for (tu, tv, sc) in [(0.14, 0.68, 1.1), (0.25, 0.80, 1.25), (0.34, 0.66, 1.0), (0.18, 0.84, 0.95)]:
            _draw_iso_tree(ax, tu, tv, th["day"], scale=sc, zorder=6.2 + (2.0 - (tu + tv)))
        return None

    n_side = 1 + int(round(pv_frac * 3))  # 1..4
    rows = n_side
    cols = max(2, n_side) if pv_frac > 0.15 else 1
    pw, pd = 0.068, 0.056
    gap_u, gap_v = 0.016, 0.018
    arr_u = cols * pw + (cols - 1) * gap_u
    arr_v = rows * pd + (rows - 1) * gap_v

    # Anchor array neatly near the inner corner (u_right=0.41, v_bot=0.57) so conduit stays short!
    u_start = 0.41 - arr_u
    v_start = 0.57
    pad_m = 0.022
    _draw_adaptive_pad(ax, u_start - pad_m, v_start - pad_m * 0.7, arr_u + 2 * pad_m, arr_v + 1.7 * pad_m, C["solar"], th)

    # Reclaim unused outer corner of the solar meadow with extra trees when small!
    if pv_frac < 0.60:
        for (tu, tv) in [(0.10, 0.86), (0.16, 0.88), (0.10, 0.72)]:
            if tu < u_start - 0.03 or tv > v_start + arr_v + 0.03:
                _draw_iso_tree(ax, tu, tv, th["day"], scale=1.05, zorder=6.2 + (2.0 - (tu + tv)))

    on = pv_now > 0.5
    cell_fc = _mix("#172554", "#1D4ED8", 0.75 if on else 0.25)
    rim_c = C["solar_hi"] if on else _mix("#334155", "#64748B", th["day"])
    bus_c = "#93C5FD" if on else "#475569"

    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            pu = u_start + c * (pw + gap_u)
            pv = v_start + r * (pd + gap_v)
            zord = 6.2 + (2.0 - (pu + pv)) * 1.2
            iso_poly(ax, [(pu + 0.004, pv, 0.004), (pu + pw - 0.004, pv, 0.004),
                          (pu + pw - 0.004, pv + pd * 0.75, 0.004), (pu + 0.004, pv + pd * 0.75, 0.004)],
                     fc="#0F172A", alpha=0.20 * (0.4 + 0.6 * th["day"]), zorder=zord - 0.3)
            iso_poly(ax, [(pu, pv, 0.012), (pu + pw, pv, 0.012),
                          (pu + pw, pv + pd, 0.052), (pu, pv + pd, 0.052)],
                     fc=cell_fc, ec=rim_c, lw=0.72, zorder=zord)
            mx0, my0 = iso(pu + pw * 0.5, pv, 0.012)
            mx1, my1 = iso(pu + pw * 0.5, pv + pd, 0.052)
            ax.plot([mx0, mx1], [my0, my1], color=bus_c, lw=0.45, alpha=0.55, zorder=zord + 0.05)
            if on:
                iso_poly(ax, [(pu, pv + pd * 0.5, 0.032), (pu + pw * 0.55, pv + pd * 0.5, 0.032),
                              (pu + pw * 0.70, pv + pd, 0.052), (pu, pv + pd, 0.052)],
                         fc=C["solar_hi"], alpha=0.18, zorder=zord + 0.08)

    # Return anchor point for feeder & callout
    return (u_start + arr_u * 0.5, v_start, u_start + arr_u * 0.5, v_start + arr_v * 0.5)


def _draw_battery_yard_adaptive(ax, batt_frac: float, batt_kwh: float, soc: float, flow: float, th: Dict, sun_dx, sun_dy):
    """Adaptive Megapack Battery Yard (0..6 containers) with meadow reclamation."""
    if batt_kwh < 1.0:
        for (tu, tv, sc) in [(0.14, 0.24, 1.1), (0.28, 0.18, 1.0), (0.22, 0.32, 1.15)]:
            _draw_iso_tree(ax, tu, tv, th["day"], scale=sc, zorder=6.8 + (2.0 - (tu + tv)))
        return None

    n_units = 1 + int(round(batt_frac * 5))  # 1..6
    soc_c = float(np.clip(soc, 0.0, 1.0))
    led_c = C["battery"] if soc_c > 0.35 else (C["solar"] if soc_c > 0.15 else C["unmet"])

    du, dv, dz = 0.090, 0.112, 0.092
    all_coords = [
        (0.31, 0.25), (0.195, 0.25), (0.08, 0.25),
        (0.31, 0.10), (0.195, 0.10), (0.08, 0.10),
    ]
    coords = all_coords[:n_units]

    u_min = min(u for u, _ in coords)
    u_max = max(u + du for u, _ in coords)
    v_min = min(v for _, v in coords)
    v_max = max(v + dv for _, v in coords)
    pad_m = 0.022
    _draw_adaptive_pad(ax, u_min - pad_m, v_min - pad_m, (u_max - u_min) + 2 * pad_m, (v_max - v_min) + 2 * pad_m, C["battery"], th)

    # Reclaim unused meadow space with trees
    if n_units <= 3:
        for (tu, tv) in [(0.12, 0.12), (0.25, 0.12)]:
            if tv < v_min - 0.03:
                _draw_iso_tree(ax, tu, tv, th["day"], scale=1.0, zorder=7.5 + (2.0 - (tu + tv)))

    sh_top = _mix("#2C3E5B", "#F8FAFC", th["day"])
    sh_l   = _mix("#1A263C", "#CBD5E1", th["day"])
    sh_r   = _mix("#23324D", "#E2E8F0", th["day"])
    ec_col = led_c if abs(flow) > 0.5 else _mix("#3E547A", "#94A3B8", th["day"])

    # Painter's order (descending u+v)
    for (bu, bv) in sorted(coords, key=lambda p: -(p[0] + p[1])):
        zord = 6.8 + (2.0 - (bu + bv)) * 1.4
        iso_shadow(ax, bu, bv, du, dv, dz, sun_dx, sun_dy, th["day"], zorder=zord - 0.5)
        iso_box(ax, bu, bv, du, dv, 0.0, dz, sh_top, sh_l, sh_r, ec=ec_col, lw=0.75, zorder=zord)
        bar_w, bar_h = du * 0.72, dz * 0.55
        iso_poly(ax, [(bu + du * 0.14, bv, 0.018), (bu + du * 0.14 + bar_w, bv, 0.018),
                      (bu + du * 0.14 + bar_w, bv, 0.018 + bar_h), (bu + du * 0.14, bv, 0.018 + bar_h)],
                 fc="#0B1221", zorder=zord + 0.08)
        fill_h = bar_h * max(soc_c, 0.08)
        iso_poly(ax, [(bu + du * 0.16, bv, 0.020), (bu + du * 0.14 + bar_w - 0.003, bv, 0.020),
                      (bu + du * 0.14 + bar_w - 0.003, bv, 0.020 + fill_h), (bu + du * 0.16, bv, 0.020 + fill_h)],
                 fc=led_c, zorder=zord + 0.12)

    return ((u_min + u_max) * 0.5, v_max, (u_min + u_max) * 0.5, (v_min + v_max) * 0.5)


def _draw_genset_adaptive(ax, gen_frac: float, gen_kw: float, running: bool, step_idx: int, th: Dict, sun_dx, sun_dy):
    """Adaptive Genset Plant: pad & 3D housing scale cleanly; 0 kW reclaims meadow."""
    if gen_kw < 1.0:
        for (tu, tv) in [(0.56, 0.16), (0.64, 0.22)]:
            _draw_iso_tree(ax, tu, tv, th["day"], scale=1.05, zorder=8.0)
        return None

    du = 0.095 + 0.075 * gen_frac
    dv = 0.080 + 0.055 * gen_frac
    dz = 0.078 + 0.050 * gen_frac
    gu = 0.60 - du / 2.0
    gv = 0.19 - dv / 2.0
    pad_m = 0.024
    _draw_adaptive_pad(ax, gu - pad_m, gv - pad_m, du + 2 * pad_m, dv + 2 * pad_m, C["genset"], th)

    zord = 8.4
    iso_shadow(ax, gu, gv, du, dv, dz, sun_dx, sun_dy, th["day"], zorder=zord - 0.5)
    c_top = "#FB923C" if running else _mix("#334155", "#94A3B8", th["day"])
    c_l   = "#C2410C" if running else _mix("#1E293B", "#64748B", th["day"])
    c_r   = "#EA580C" if running else _mix("#273549", "#7E8FA6", th["day"])
    ec_c  = "#FDBA74" if running else _mix("#475569", "#64748B", th["day"])
    iso_box(ax, gu, gv, du, dv, 0.0, dz, c_top, c_l, c_r, ec=ec_c, lw=0.8, zorder=zord)

    for k in range(3):
        zf = 0.020 + k * (dz * 0.22)
        p0, p1 = iso(gu + du * 0.14, gv, zf), iso(gu + du * 0.58, gv, zf)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#1E1B18", lw=1.1, alpha=0.75, zorder=zord + 0.1)

    stacks = [(gu + du * 0.78, gv + dv * 0.65)]
    if gen_frac > 0.55:
        stacks.append((gu + du * 0.62, gv + dv * 0.65))
    for (su, sv) in stacks:
        ex, ey = iso(su, sv, dz)
        ax.plot([ex, ex], [ey, ey + 0.046], color="#64748B", lw=2.6, solid_capstyle="round", zorder=zord + 0.2)
        if running:
            ph = (step_idx % 3) * 0.005
            for dx, dy, rr, al in ((0.006, 0.058 + ph, 0.0085, 0.44),
                                   (0.016, 0.078 + ph, 0.0125, 0.30),
                                   (0.030, 0.098 + ph, 0.0165, 0.16)):
                ax.add_patch(Circle((ex + dx, ey + dy), rr,
                                    fc="#CBD5E1" if th["day"] < 0.5 else "#64748B",
                                    ec="none", alpha=al, zorder=zord + 0.3))
    return (0.60, gv + dv, 0.60, 0.19)


def _draw_grid_pylon(ax, off_grid: bool, grid_flow: float, th: Dict):
    px, py = iso(0.018, 0.46, 0.0)
    col = C["muted"] if off_grid else _mix("#38BDF8", "#0284C7", th["day"])
    if (not off_grid) and abs(grid_flow) > 0.5:
        ax.add_patch(Circle((px, py + 0.062), 0.050, fc=C["grid"], alpha=0.14 * (1.0 - 0.4 * th["day"]), zorder=4))

    h, w = 0.120, 0.042
    ax.plot([px - w * 0.55, px], [py, py + h], color=col, lw=2.0, zorder=9.0)
    ax.plot([px + w * 0.55, px], [py, py + h], color=col, lw=2.0, zorder=9.0)
    for f0, f1 in ((0.18, 0.46), (0.46, 0.72)):
        ax.plot([px - w * 0.55 * (1 - f0), px + w * 0.55 * (1 - f1)], [py + h * f0, py + h * f1], color=col, lw=0.85, alpha=0.75, zorder=9.0)
        ax.plot([px + w * 0.55 * (1 - f0), px - w * 0.55 * (1 - f1)], [py + h * f0, py + h * f1], color=col, lw=0.85, alpha=0.75, zorder=9.0)
    for fy, sp in ((0.76, 0.76), (0.50, 0.90)):
        yy = py + h * fy
        ax.plot([px - w * sp, px + w * sp], [yy, yy], color=col, lw=1.9, solid_capstyle="round", zorder=9.1)
    if off_grid:
        ax.plot([px - 0.030, px + 0.030], [py + 0.025, py + 0.092], color=C["unmet"], lw=2.4, zorder=9.5)
        ax.plot([px - 0.030, px + 0.030], [py + 0.092, py + 0.025], color=C["unmet"], lw=2.4, zorder=9.5)


def _draw_campus_buildings(ax, bldgs: List[Dict], th: Dict, sun_dx, sun_dy):
    n = len(bldgs)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    u0_z, u1_z = 0.56, 0.94
    v0_z, v1_z = 0.51, 0.93
    du_step = (u1_z - u0_z) / cols
    dv_step = (v1_z - v0_z) / rows
    bw = min(du_step * 0.68, 0.145)
    bd = min(dv_step * 0.66, 0.145)

    arch = {
        "flat":   (0.32, "#38BDF8"),
        "cross":  (0.25, "#EF4444"),
        "sign":   (0.27, "#10B981"),
        "flag":   (0.21, "#3B82F6"),
        "awning": (0.15, "#F59E0B"),
    }

    items = []
    for i, b in enumerate(bldgs):
        r, c = divmod(i, cols)
        bu = u0_z + (c + 0.5) * du_step - bw / 2.0
        bv = v0_z + (r + 0.5) * dv_step - bd / 2.0
        items.append((bu + bv, bu, bv, b))
    items.sort(key=lambda x: -x[0])

    for _, bu, bv, b in items:
        _, kind = BUILDING_KIND.get(b["name"], ("Load", "flat"))
        bh, acc = arch.get(kind, (0.22, "#38BDF8"))
        served_ok = b["served"] >= 0.99 * max(b["demand"], 1e-6)
        zord = 6.5 + (2.0 - (bu + bv)) * 2.2

        iso_shadow(ax, bu, bv, bw, bd, bh, sun_dx, sun_dy, th["day"], zorder=zord - 0.6)
        ec_c = C["unmet"] if not served_ok else th["b_ec"]
        iso_box(ax, bu, bv, bw, bd, 0.0, bh, th["b_top"], th["b_l"], th["b_r"], ec=ec_c, lw=0.85, zorder=zord)

        win_c = C["solar_hi"] if served_ok else C["unmet"]
        win_alpha = (0.95 if th["night"] > 0.35 else 0.78) if served_ok else 0.85
        n_fl = max(2, int(bh / 0.062))
        for fl in range(n_fl):
            zf = 0.028 + fl * (bh - 0.048) / max(n_fl, 1)
            p0, p1 = iso(bu + bw * 0.12, bv, zf), iso(bu + bw * 0.88, bv, zf)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=win_c, lw=2.0, alpha=win_alpha, solid_capstyle="butt", zorder=zord + 0.12)
            q0, q1 = iso(bu, bv + bd * 0.12, zf), iso(bu, bv + bd * 0.88, zf)
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=win_c, lw=2.0, alpha=win_alpha * 0.78, solid_capstyle="butt", zorder=zord + 0.12)

        if kind == "cross":
            cu, cv = bu + bw * 0.5, bv + bd * 0.5
            iso_poly(ax, [(cu - bw * 0.28, cv - bd * 0.10, bh + 0.005), (cu + bw * 0.28, cv - bd * 0.10, bh + 0.005),
                          (cu + bw * 0.28, cv + bd * 0.10, bh + 0.005), (cu - bw * 0.28, cv + bd * 0.10, bh + 0.005)],
                     fc=C["unmet"], zorder=zord + 0.22)
            iso_poly(ax, [(cu - bw * 0.10, cv - bd * 0.28, bh + 0.005), (cu + bw * 0.10, cv - bd * 0.28, bh + 0.005),
                          (cu + bw * 0.10, cv + bd * 0.28, bh + 0.005), (cu - bw * 0.10, cv + bd * 0.28, bh + 0.005)],
                     fc=C["unmet"], zorder=zord + 0.23)
        elif kind == "flat":
            iso_poly(ax, [(bu + bw * 0.18, bv + bd * 0.18, bh + 0.004), (bu + bw * 0.82, bv + bd * 0.18, bh + 0.004),
                          (bu + bw * 0.82, bv + bd * 0.82, bh + 0.004), (bu + bw * 0.18, bv + bd * 0.82, bh + 0.004)],
                     fc=acc, alpha=0.82, zorder=zord + 0.2)
            sx, sy = iso(bu + bw * 0.5, bv + bd * 0.5, bh + 0.005)
            ax.plot([sx, sx], [sy, sy + 0.034], color="#64748B", lw=1.4, zorder=zord + 0.25)
            ax.add_patch(Circle((sx, sy + 0.034), 0.0035, fc=C["grid"] if served_ok else C["unmet"], zorder=zord + 0.3))
        elif kind == "flag":
            iso_poly(ax, [(bu + bw * 0.15, bv + bd * 0.15, bh + 0.004), (bu + bw * 0.85, bv + bd * 0.15, bh + 0.004),
                          (bu + bw * 0.85, bv + bd * 0.85, bh + 0.004), (bu + bw * 0.15, bv + bd * 0.85, bh + 0.004)],
                     fc=acc, alpha=0.72, zorder=zord + 0.2)
            fx, fy = iso(bu + bw * 0.5, bv + bd * 0.5, bh + 0.005)
            ax.plot([fx, fx], [fy, fy + 0.034], color="#94A3B8", lw=1.4, zorder=zord + 0.25)
            ax.add_patch(Polygon([(fx, fy + 0.034), (fx + 0.021, fy + 0.026), (fx, fy + 0.018)], fc=C["grid"], zorder=zord + 0.3))
        elif kind == "awning":
            iso_poly(ax, [(bu + bw * 0.14, bv + bd * 0.14, bh + 0.004), (bu + bw * 0.86, bv + bd * 0.14, bh + 0.004),
                          (bu + bw * 0.86, bv + bd * 0.86, bh + 0.004), (bu + bw * 0.14, bv + bd * 0.86, bh + 0.004)],
                     fc=acc, alpha=0.88, zorder=zord + 0.2)
        else:
            iso_poly(ax, [(bu + bw * 0.18, bv + bd * 0.18, bh + 0.004), (bu + bw * 0.82, bv + bd * 0.18, bh + 0.004),
                          (bu + bw * 0.82, bv + bd * 0.82, bh + 0.004), (bu + bw * 0.18, bv + bd * 0.82, bh + 0.004)],
                     fc=acc, alpha=0.85, zorder=zord + 0.2)


# ========================================================================= #
# Main Coastal Solarpunk Eco-Island Diorama (left ~67%)
# ========================================================================= #
def draw_eco_island(ax, state: Dict, th: Dict):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    emb = state["embodiment"]
    d = state["dispatch"]
    soc = float(state["soc"])
    off_grid = bool(state["off_grid"])
    pmax = max(float(state["demand_peak"]), 1.0)
    step_idx = int(state.get("step_idx", 1))
    phase = (step_idx % 4) * 0.25

    pv_cap = float(emb["pv_capacity_kw"])
    batt_kwh = float(emb["battery_capacity_kwh"])
    batt_power = 0.25 * batt_kwh
    gen_cap = float(emb["genset_rating_kw"])
    grid_cap = float(state["grid_max"])
    wref = max(pv_cap, batt_power, gen_cap, pmax, 1.0)

    pv_frac = _design_frac(state, "pv_capacity_kw", pmax * 2.0)
    batt_frac = _design_frac(state, "battery_capacity_kwh", pmax * 4.0)
    gen_frac = _design_frac(state, "genset_rating_kw", pmax * 1.5)

    # 1. Sky, Stars, Drifting Clouds & Sun/Moon Arc
    sun_dx, sun_dy = _draw_sky_and_celestial(ax, th, pv_on=d["pv"] > 0.5, step_idx=step_idx)

    # 2. Layered Floating Island (Bedrock -> Soil -> Lush Emerald Meadow)
    iso_box(ax, 0.0, 0.0, 1.0, 1.0, -0.145, 0.070, th["plat_l"], th["rock_l"], th["rock_r"], ec="none", zorder=1.2)
    iso_box(ax, 0.0, 0.0, 1.0, 1.0, -0.075, 0.075, th["plat_top"], th["plat_l"], th["plat_r"], ec=th["grid_ln"], lw=0.75, zorder=1.5)
    for g in np.linspace(0.1, 0.9, 9):
        p0, p1 = iso(g, 0, 0.001), iso(g, 1, 0.001)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=th["grid_ln"], lw=0.40, alpha=0.36, zorder=1.7)
        q0, q1 = iso(0, g, 0.001), iso(1, g, 0.001)
        ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=th["grid_ln"], lw=0.40, alpha=0.36, zorder=1.7)

    # 3. Paved Isometric Campus Boulevard
    road_c = _mix("#162033", "#94A3B8", th["day"])
    iso_poly(ax, [(0.445, 0.02, 0.0022), (0.515, 0.02, 0.0022), (0.515, 0.98, 0.0022), (0.445, 0.98, 0.0022)],
             fc=road_c, alpha=0.72, zorder=1.85)
    iso_poly(ax, [(0.02, 0.43, 0.0022), (0.98, 0.43, 0.0022), (0.98, 0.49, 0.0022), (0.02, 0.49, 0.0022)],
             fc=road_c, alpha=0.72, zorder=1.85)

    # 4. Tranquil Turquoise Eco-Pond in front-right meadow corner
    water_c = _mix("#0C4A6E", "#38BDF8", th["day"])
    water_hi = _mix("#0284C7", "#BAE6FD", th["day"])
    shore_c = _mix("#292524", "#E7D5B4", th["day"])
    iso_poly(ax, [(0.74, 0.06, 0.002), (0.96, 0.06, 0.002), (0.96, 0.36, 0.002), (0.74, 0.36, 0.002)],
             fc=shore_c, zorder=1.95)
    iso_poly(ax, [(0.76, 0.08, 0.003), (0.94, 0.08, 0.003), (0.94, 0.34, 0.003), (0.76, 0.34, 0.003)],
             fc=water_c, ec=water_hi, lw=0.8, alpha=0.92, zorder=2.05)
    sh_off = (step_idx % 3) * 0.015
    for (wu, wv) in [(0.79 + sh_off, 0.16), (0.83 - sh_off * 0.5, 0.25), (0.85 + sh_off * 0.4, 0.13)]:
        p0, p1 = iso(wu, wv, 0.005), iso(wu + 0.052, wv, 0.005)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=water_hi, lw=1.0, alpha=0.55, zorder=2.15)

    # Campus plaza pad
    _draw_adaptive_pad(ax, 0.54, 0.49, 0.42, 0.46, C["grid"], th)

    # Perimeter Solarpunk trees
    for (tu, tv) in [(0.04, 0.92), (0.42, 0.95), (0.94, 0.96), (0.73, 0.39), (0.95, 0.39), (0.04, 0.05)]:
        _draw_iso_tree(ax, tu, tv, th["day"], zorder=6.5 + (2.0 - (tu + tv)))

    # 5. Central 60 Hz Transformer Hub
    hu, hv = 0.48, 0.46
    unmet_now = d["unmet"] > 0.5
    hub_c = C["unmet"] if unmet_now else C["grid"]
    iso_shadow(ax, hu - 0.046, hv - 0.046, 0.092, 0.092, 0.056, sun_dx, sun_dy, th["day"], zorder=5.2)
    iso_box(ax, hu - 0.046, hv - 0.046, 0.092, 0.092, 0.0, 0.056,
            _mix(hub_c, "#FFFFFF", 0.20), _mix(hub_c, "#0F172A", 0.35), hub_c,
            ec=C["unmet_hi"] if unmet_now else C["grid_hi"], lw=0.85, zorder=6.2)

    # 6. Adaptive Embodiment Hardware (pads + hardware + meadow reclamation)
    pv_anc = _draw_solar_farm_adaptive(ax, pv_frac, pv_cap, d["pv"], th)
    b_flow = float(d["battery"])
    bt_anc = _draw_battery_yard_adaptive(ax, batt_frac, batt_kwh, soc, b_flow, th, sun_dx, sun_dy)
    gn_anc = _draw_genset_adaptive(ax, gen_frac, gen_cap, d["genset"] > 0.5, step_idx, th, sun_dx, sun_dy)
    _draw_grid_pylon(ax, off_grid, d["grid"], th)
    _draw_campus_buildings(ax, d["by_building"], th, sun_dx, sun_dy)

    # 7. Luminous Directional Conduits (omitted cleanly when a module has 0 capacity!)
    if pv_anc is not None:
        iso_conduit(ax, [(pv_anc[0], pv_anc[1], 0.008), (pv_anc[0], hv, 0.008), (hu, hv, 0.008)],
                    pv_cap, d["pv"], wref, C["solar"], direction=1, phase=phase, day=th["day"])
    if bt_anc is not None:
        iso_conduit(ax, [(bt_anc[0], bt_anc[1], 0.008), (bt_anc[0], hv, 0.008), (hu, hv, 0.008)],
                    batt_power, b_flow, wref, C["battery"], direction=1 if b_flow >= 0 else -1, phase=phase, day=th["day"])
    if gn_anc is not None:
        iso_conduit(ax, [(gn_anc[0], gn_anc[1], 0.008), (gn_anc[0], hv, 0.008), (hu, hv, 0.008)],
                    gen_cap, d["genset"], wref, C["genset"], direction=1, phase=phase, day=th["day"])
    if not off_grid:
        g_flow = float(d["grid"])
        iso_conduit(ax, [(0.02, hv, 0.008), (hu, hv, 0.008)],
                    grid_cap, g_flow, wref, C["grid"], direction=1 if g_flow >= 0 else -1, phase=phase, day=th["day"])
    else:
        p0, p1 = iso(0.02, hv, 0.008), iso(hu - 0.05, hv, 0.008)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C["muted"], lw=1.4, ls=(0, (4, 4)), alpha=0.65, zorder=3.0)

    load_hue = "#F8FAFC" if th["day"] < 0.45 else "#1E293B"
    iso_conduit(ax, [(hu, hv, 0.008), (0.74, hv, 0.008), (0.74, 0.70, 0.008)],
                pmax, d["demand"], wref, load_hue, red=unmet_now, direction=1, phase=phase, day=th["day"])

    # 8. Clean Perimeter Callouts with Leader Lines
    b_txt = (f"dischg {b_flow:.0f} kW" if b_flow > 0.5 else (f"charge {-b_flow:.0f} kW" if b_flow < -0.5 else "standby"))
    g_txt = "ISLANDED · off-grid" if off_grid else (f'import {d["grid"]:.0f} kW' if d["grid"] >= 0 else f'export {-d["grid"]:.0f} kW')
    n_b = len(d["by_building"])
    c_sub = f'{n_b} bldg{"s" if n_b > 1 else ""} · UNMET {d["unmet"]:.0f} kW' if unmet_now else f'{n_b} bldg{"s" if n_b > 1 else ""} · 100% served'

    callouts = [
        (0.16, 0.84, iso(pv_anc[2], pv_anc[3], 0.05) if pv_anc else iso(0.24, 0.76, 0.02),
         f'SOLAR · {pv_cap:.0f} kW' if pv_cap >= 1 else "SOLAR · NONE",
         f'{d["pv"]:.0f} kW live' if pv_cap >= 1 else "meadow reclaimed", C["solar"] if pv_cap >= 1 else C["muted"]),
        (0.11, 0.49, iso(0.018, 0.46, 0.06), "UTILITY GRID" if not off_grid else "GRID (OFF)",
         g_txt, C["muted"] if off_grid else C["grid"]),
        (0.19, 0.09, iso(bt_anc[2], bt_anc[3], 0.04) if bt_anc else iso(0.22, 0.22, 0.02),
         f'BESS · {batt_kwh:.0f} kWh' if batt_kwh >= 1 else "BESS · NONE",
         f'SoC {100*soc:.0f}% · {b_txt}' if batt_kwh >= 1 else "meadow reclaimed", C["battery"] if batt_kwh >= 1 else C["muted"]),
        (0.71, 0.09, iso(gn_anc[2], gn_anc[3], 0.05) if gn_anc else iso(0.60, 0.19, 0.02),
         f'GENSET · {gen_cap:.0f} kW' if gen_cap >= 1 else "GENSET · NONE",
         f'{d["genset"]:.0f} kW live' if gen_cap >= 1 else "meadow reclaimed", C["genset"] if gen_cap >= 1 else C["muted"]),
        (0.84, 0.84, iso(0.76, 0.74, 0.24), f'CAMPUS · {d["demand"]:.0f} kW',
         c_sub, C["unmet"] if unmet_now else C["grid"]),
    ]
    for (cx, cy, anchor, t1, t2, acc) in callouts:
        ax.plot([cx, anchor[0]], [cy, anchor[1]], color=acc, lw=0.85, ls=(0, (2, 2)), alpha=0.55, zorder=12)
        ax.add_patch(Circle(anchor, 0.0042, fc=acc, ec="none", alpha=0.85, zorder=13))
        ax.text(
            cx, cy, f"{t1}\n{t2}",
            ha="center", va="center", fontsize=7.6, fontweight="bold",
            color=th["pill_tx"], linespacing=1.32, zorder=15,
            bbox=dict(boxstyle="round,pad=0.36,rounding_size=0.30",
                      facecolor=th["pill_bg"], edgecolor=acc, linewidth=1.15, alpha=0.93),
        )


# ========================================================================= #
# Intuitive Right-Hand Telemetry Sidebar (Fixed 24h Window + Live Callouts)
# ========================================================================= #
def _draw_intuitive_sidebar(fig, state: Dict, rec: Dict):
    hr = rec["hour"]
    cursor = float(hr[-1])
    # Pin x-axis to the current 24-hour day window [day0, day0 + 24] so animations
    # fill smoothly left-to-right with zero horizontal rescaling jitter!
    day0 = int(cursor // 24) * 24
    x_lo, x_hi = day0 - 0.5, day0 + 24.5
    pmax = max(float(state["demand_peak"]), 1.0)
    emb = state["embodiment"]

    gs = fig.add_gridspec(
        5, 1,
        left=0.702, right=0.984, bottom=0.048, top=0.900,
        height_ratios=[0.78, 0.68, 1.38, 0.90, 0.98],
        hspace=0.42,
    )

    # --- 1. 2x2 KPI Summary Card ---
    ax_k = fig.add_subplot(gs[0, 0])
    ax_k.set_xlim(0, 1); ax_k.set_ylim(0, 1); ax_k.axis("off")
    ax_k.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.06",
                                  fc=C["hud_card"], ec=C["hud_edge"], lw=1.0))
    tot_cost = float(np.cumsum(-rec["reward"])[-1])
    renew = 100.0 * float(rec["pv"].sum()) / max(float(rec["demand"].sum()), 1e-6)
    unmet = float(rec["unmet"].sum())
    soc_pct = 100.0 * float(state["soc"])
    for i, (lbl, val, vcol, acc) in enumerate([
        ("TOTAL COST", _fmt_dollars(tot_cost), C["hud_ink"], C["grid"]),
        ("RENEWABLE", f"{renew:.0f}%", C["solar_hi"], C["solar"]),
        ("BATTERY SoC", f"{soc_pct:.0f}%", C["battery_hi"] if soc_pct > 20 else C["unmet_hi"], C["battery"]),
        ("UNMET LOAD", f"{unmet:,.0f} kWh" if unmet > 0.5 else "0 kWh",
         C["unmet_hi"] if unmet > 0.5 else C["battery_hi"], C["unmet"] if unmet > 0.5 else C["battery"]),
    ]):
        r, c = divmod(i, 2)
        x0, y0 = 0.055 + c * 0.49, 0.87 - r * 0.48
        ax_k.add_patch(Rectangle((x0, y0 + 0.03), 0.36, 0.028, fc=acc))
        ax_k.text(x0, y0 - 0.02, lbl, fontsize=6.5, fontweight="bold", color=C["hud_ink2"], va="top")
        ax_k.text(x0, y0 - 0.20, val, fontsize=11.8, fontweight="bold", color=vcol, va="top")

    # --- 2. Embodiment Sizing Gauges Card ---
    ax_g = fig.add_subplot(gs[1, 0])
    ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1); ax_g.axis("off")
    ax_g.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.06",
                                  fc=C["hud_card"], ec=C["hud_edge"], lw=1.0))
    gauges = [
        ("SOLAR", emb["pv_capacity_kw"], "kW", _design_frac(state, "pv_capacity_kw", pmax * 2), C["solar"]),
        ("BESS", emb["battery_capacity_kwh"], "kWh", _design_frac(state, "battery_capacity_kwh", pmax * 4), C["battery"]),
        ("GENSET", emb["genset_rating_kw"], "kW", _design_frac(state, "genset_rating_kw", pmax * 1.5), C["genset"]),
    ]
    for idx, (lbl, val, unit, fr, col) in enumerate(gauges):
        yy = 0.77 - idx * 0.31
        ax_g.text(0.05, yy, lbl, fontsize=6.8, fontweight="bold", color=col, va="center")
        ax_g.text(0.95, yy, f"{val:.0f} {unit} ({100*fr:.0f}%)", fontsize=6.6, fontweight="bold",
                  color=C["hud_ink"], ha="right", va="center")
        ax_g.add_patch(FancyBboxPatch((0.24, yy - 0.07), 0.38, 0.14, boxstyle="round,pad=0.002,rounding_size=0.04", fc="#17233B", ec="none"))
        if fr > 0.005:
            ax_g.add_patch(FancyBboxPatch((0.24, yy - 0.07), max(0.38 * fr, 0.014), 0.14, boxstyle="round,pad=0.002,rounding_size=0.04", fc=col, ec="none"))

    # --- 3. Power Balance (Supply vs Demand) on Fixed 24h Clock ---
    ax_d = fig.add_subplot(gs[2, 0])
    ax_d.axvspan(day0 - 0.5, day0 + 6, color="#040812", alpha=0.48, zorder=0)
    ax_d.axvspan(day0 + 19, day0 + 24.5, color="#040812", alpha=0.48, zorder=0)
    bot = np.zeros_like(hr, dtype=float)
    for vals, col in [
        (rec["pv"], C["solar"]),
        (np.clip(rec["battery"], 0, None), C["battery"]),
        (rec["genset"], C["genset"]),
        (np.clip(rec["grid"], 0, None), C["grid"]),
        (rec["unmet"], C["unmet"]),
    ]:
        ax_d.bar(hr, vals, bottom=bot, width=0.80, color=col, lw=0, alpha=0.92, zorder=2)
        bot += vals
    bneg = np.zeros_like(hr, dtype=float)
    for vals, col in [(np.clip(rec["battery"], None, 0), C["battery"]),
                      (np.clip(rec["grid"], None, 0), C["grid"]),
                      (-rec["curtail"], C["muted"])]:
        ax_d.bar(hr, vals, bottom=bneg, width=0.80, color=col, alpha=0.35, hatch="////", lw=0, zorder=2)
        bneg += vals
    ax_d.plot(hr, rec["demand"], color="#FFFFFF", lw=1.8, zorder=4)
    ax_d.scatter([cursor], [rec["demand"][-1]], s=24, color="#FFFFFF", edgecolor=C["grid"], linewidth=1.1, zorder=6)
    ax_d.axhline(0, color=C["hud_edge"], lw=0.8, zorder=3)
    ax_d.axvline(cursor, color=C["grid"], lw=1.0, ls=(0, (3, 2)), alpha=0.85, zorder=5)
    ax_d.set_title("POWER BALANCE (kW)", fontsize=7.4, fontweight="bold", color=C["hud_ink"], loc="left", pad=14)
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=4.5, mfc=col, mec="none")
               for col in (C["solar"], C["battery"], C["genset"], C["grid"], C["unmet"])]
    handles.append(plt.Line2D([0], [0], color="#FFFFFF", lw=1.5))
    leg = ax_d.legend(handles, ["PV", "BESS", "Gen", "Grid", "Unmet", "Load"], fontsize=6.0, ncol=6,
                      loc="lower right", bbox_to_anchor=(1.0, 1.01), frameon=False, handletextpad=0.2, columnspacing=0.55)
    for t in leg.get_texts():
        t.set_color(C["hud_ink2"])

    # --- 4. Battery State of Charge (%) with Low-Reserve Zone & Live Dot ---
    ax_s = fig.add_subplot(gs[3, 0])
    ax_s.axvspan(day0 - 0.5, day0 + 6, color="#040812", alpha=0.48, zorder=0)
    ax_s.axvspan(day0 + 19, day0 + 24.5, color="#040812", alpha=0.48, zorder=0)
    ax_s.axhspan(0, 20, color=C["unmet"], alpha=0.09, zorder=1)
    ax_s.axhline(20, color=C["unmet"], lw=0.75, ls="--", alpha=0.55, zorder=2)
    ax_s.text(day0 + 24.0, 23, "20% reserve", ha="right", va="bottom", fontsize=5.8, color=C["unmet_hi"], alpha=0.85)
    soc_arr = 100.0 * rec["soc"]
    ax_s.plot(hr, soc_arr, color=C["battery_hi"], lw=1.9, zorder=4)
    ax_s.fill_between(hr, 0, soc_arr, color=C["battery"], alpha=0.18, zorder=2)
    ax_s.scatter([cursor], [soc_arr[-1]], s=24, color=C["battery_hi"], edgecolor="#FFFFFF", linewidth=0.9, zorder=6)
    ax_s.axvline(cursor, color=C["grid"], lw=1.0, ls=(0, (3, 2)), alpha=0.85, zorder=5)
    ax_s.set_ylim(0, 105)
    ax_s.set_yticks([0, 50, 100])
    ax_s.set_yticklabels(["0%", "50%", "100%"])
    ax_s.set_title(f"BATTERY CHARGE  ·  {soc_arr[-1]:.0f}% now", fontsize=7.4, fontweight="bold", color=C["hud_ink"], loc="left", pad=4)

    # --- 5. Stacked Cumulative Cost (Operating + Capex Amortization in $k) ---
    ax_c = fig.add_subplot(gs[4, 0])
    ax_c.axvspan(day0 - 0.5, day0 + 6, color="#040812", alpha=0.48, zorder=0)
    ax_c.axvspan(day0 + 19, day0 + 24.5, color="#040812", alpha=0.48, zorder=0)
    op_c = np.cumsum(rec["op_cost"])
    tot_c = np.cumsum(-rec["reward"])
    ax_c.fill_between(hr, 0, op_c, color=C["grid"], alpha=0.16, zorder=2, label="Operating")
    ax_c.fill_between(hr, op_c, tot_c, color=C["solar"], alpha=0.24, zorder=2.5, label="Capex")
    ax_c.plot(hr, op_c, color=C["grid_hi"], lw=1.5, zorder=4)
    ax_c.plot(hr, tot_c, color=C["solar_hi"], lw=1.8, zorder=5)
    ax_c.scatter([cursor], [tot_c[-1]], s=24, color=C["solar_hi"], edgecolor="#FFFFFF", linewidth=0.9, zorder=6)
    ax_c.axvline(cursor, color=C["grid"], lw=1.0, ls=(0, (3, 2)), alpha=0.85, zorder=5)
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: _fmt_dollars(y)))
    ax_c.set_title(f"CUMULATIVE COST  ·  {_fmt_dollars(tot_c[-1])} total", fontsize=7.4, fontweight="bold", color=C["hud_ink"], loc="left", pad=4)
    leg_c = ax_c.legend(fontsize=6.2, frameon=False, loc="upper left", ncol=2, handlelength=1.2)
    for t in leg_c.get_texts():
        t.set_color(C["hud_ink2"])

    ticks = [day0, day0 + 6, day0 + 12, day0 + 18, day0 + 24]
    tick_lbls = ["00h", "06h", "12h", "18h", "24h"]
    for a in (ax_d, ax_s, ax_c):
        a.set_facecolor(C["hud_card"])
        a.set_xlim(x_lo, x_hi)
        a.set_xticks(ticks)
        a.set_xticklabels(tick_lbls if a is ax_c else [])
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(C["hud_edge"])
        a.tick_params(colors=C["hud_ink2"], labelsize=6.6)
        a.grid(axis="y", color=C["hud_grid"], lw=0.6)
        a.set_axisbelow(True)


# ========================================================================= #
# Public entry point
# ========================================================================= #
def render_env_figure(state: Dict, rec: Dict, fig=None):
    """Compose the Coastal Solarpunk Eco-Island (left) + Intuitive Telemetry Sidebar (right)."""
    if fig is None:
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=C["hud_bg"])
    else:
        fig.clf()
        fig.set_facecolor(C["hud_bg"])

    step_idx = int(state.get("step_idx", len(rec["hour"])))
    hour_of_day = (step_idx - 1) % 24
    th = _sky_theme(hour_of_day)

    # Left 67%: full-height Coastal Solarpunk Eco-Island
    ax_iso = fig.add_axes([0.015, 0.032, 0.665, 0.870])
    draw_eco_island(ax_iso, state, th)

    # Right 28%: Intuitive Telemetry Sidebar
    _draw_intuitive_sidebar(fig, state, rec)

    # Header bar
    fig.suptitle(state["title"], fontsize=14.0, fontweight="bold", color=C["hud_ink"], x=0.018, ha="left", y=0.972)
    fig.text(0.018, 0.930, state["subtitle"], fontsize=8.8, color=C["hud_ink2"], ha="left")

    clock_ec = C["solar"] if th["day"] > 0.15 or th["twilight"] > 0.38 else C["grid"]
    fig.text(
        0.680, 0.949, f"{hour_of_day:02d}:00  ·  {th['phase_lbl']}",
        ha="right", va="center", fontsize=8.4, fontweight="bold",
        color=C["solar_hi"] if clock_ec == C["solar"] else C["grid_hi"],
        bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.32",
                  facecolor=C["hud_card"], edgecolor=clock_ec, linewidth=1.0),
    )
    return fig
