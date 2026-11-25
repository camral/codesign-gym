# Shape-shifting Hand: Rotate

```{include} ../../_generated/readmes/rotate_in_hand.md
```

## At a glance

| | |
|---|---|
| Ids | `RotateInHand-Cube` |
| Design | `Box(248,)` float32: 245 pad cells in `[0, 1]`, then 3 link offsets in `[-1, 1]` |
| Observation | `Box(57,)` float32 |
| Action | `Box(19,)` float32 in `[-1, 1]` (joint velocity targets: 3 wrist, 16 finger) |
| Episode | Up to 375 steps of 0.04 s (15 s); at most 7.5 s per goal |
| Mid-episode design changes | Yes, through `step(action, embodiment)` |
| Autoreset | Next step, keeping each copy's latest design; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | **Recommended:** `pip install -e '.[morphhand]'` (a MuJoCo Warp fork with the heightfield collision fixes). Without it the env runs on upstream mujoco-warp, whose heightfield contacts are unreliable. |
| Device | CUDA when available, else CPU; `device=` to choose |
| Source | [`src/envs/rotate_in_hand`](../../../src/envs/rotate_in_hand) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
