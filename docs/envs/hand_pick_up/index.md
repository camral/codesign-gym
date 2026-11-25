# Shape-shifting Hand: Pick Up

```{include} ../../_generated/readmes/hand_pick_up.md
```

## At a glance

| | |
|---|---|
| Ids | `HandPickUp-Cube`, `HandPickUp-Hammer`, `HandPickUp-SmallHammer` |
| Design | `Box(248,)` float32: 245 pad cells in `[0, 1]`, then 3 link offsets in `[-1, 1]` (the same pads and links as [Rotate](../rotate_in_hand/index.md)) |
| Observation | `Box(61,)` float32 (Cube); `Box(67,)` (Hammer, SmallHammer) |
| Action | `Box(19,)` float32 in `[-1, 1]` (velocity targets: 3 wrist, 16 finger); `Box(22,)` for the hammers, whose wrist also slides in x, y, z |
| Episode | Steps of 0.04 s. Cube: up to 375 steps (15 s), at most 6 s per goal. Hammers: up to 175 steps (7 s) |
| Mid-episode design changes | Yes, through `step(action, embodiment)` |
| Autoreset | Next step, keeping each copy's latest design; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | **Recommended:** `pip install -e '.[morphhand]'` (a MuJoCo Warp fork with the heightfield collision fixes). Without it the env runs on upstream mujoco-warp, whose heightfield contacts are unreliable. |
| Device | CUDA when available, else CPU; `device=` to choose |
| Source | [`src/envs/hand_pick_up`](../../../src/envs/hand_pick_up) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
