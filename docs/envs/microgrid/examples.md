# Examples

## A rule-based dispatcher

A simple merit-order rule: let solar cover what it can, charge the battery with any surplus, then fill a shortfall from the battery, and finally from the grid (or the generator off-grid).
The design is not observed, so the policy takes it as an argument to convert kW into action fractions.

```python
import numpy as np
from codesign_gym import make


def dispatch(obs, design, grid_kw):
    """Merit-order rule. grid_kw is the grid limit: 600 for one building, 0 off-grid."""
    soc, demand, solar = obs[0], obs[1], obs[2]
    battery_kw = max(0.25 * design["battery_capacity_kwh"], 1e-6)
    net = demand - solar
    if net < 0:                                                  # surplus: charge
        return np.array([-min(1.0, -net / battery_kw), 0.0, 0.0])
    stored = 0.9 * soc * design["battery_capacity_kwh"]          # deliverable energy
    from_battery = min(net, 0.9 * battery_kw, stored)
    rest = net - from_battery
    battery = from_battery / (0.9 * battery_kw)
    if grid_kw > 0:
        return np.array([battery, 0.0, min(1.0, rest / grid_kw)])
    return np.array([battery, min(1.0, rest / max(design["genset_rating_kw"], 1e-6)), 0.0])


env, base = make("Microgrid-Random")
obs, rejected, info = env.reset(seed=0)
print(info["city"], info["building"])
total, done = 0.0, False
while not done:
    obs, reward, terminated, truncated, info = env.step(dispatch(obs, base, grid_kw=600.0))
    total += reward
    done = terminated or truncated
print(f"return {total:.1f}")
env.close()
```

```text
minneapolis office
return -933.9
```

Compare with an all-zero action, which leaves most of the load unmet: its first step alone costs about −1335.

## Compare designs in parallel

Off-grid, the design decides whether the load can be met at all. Evaluate four designs on the same three days by giving every copy the same seed (an integer seed would give each copy its own stream, and so a different city, building and start hour):

```python
import numpy as np
from codesign_gym import make_vec

designs = {
    "base": {"pv_capacity_kw": 225.0, "battery_capacity_kwh": 450.0, "genset_rating_kw": 180.0},
    "generator only": {"pv_capacity_kw": 0.0, "battery_capacity_kwh": 0.0, "genset_rating_kw": 360.0},
    "solar + battery": {"pv_capacity_kw": 450.0, "battery_capacity_kwh": 900.0, "genset_rating_kw": 0.0},
    "small generator": {"pv_capacity_kw": 450.0, "battery_capacity_kwh": 900.0, "genset_rating_kw": 120.0},
}
envs, base = make_vec("Microgrid-OffGridSingle", len(designs))
design_list = list(designs.values())
obs, rejected, info = envs.reset(seed=[3] * len(designs), embodiment=design_list)

returns = np.zeros(len(designs))
unmet = np.zeros(len(designs))
capex = info["capex"]
for _ in range(72):
    actions = np.stack([dispatch(o, d, grid_kw=0.0) for o, d in zip(obs, design_list)])
    obs, reward, terminated, truncated, info = envs.step(actions)
    returns += reward
    unmet += [d["unmet"] for d in info["dispatch"]]
for name, r, u, c in zip(designs, returns, unmet, capex):
    print(f"{name:16s} return {r:9.1f}  unmet {u:6.0f} kWh  capex {c:5.0f}")
envs.close()
```

```text
base             return  -11829.2  unmet    795 kWh  capex   370
generator only   return   -4630.6  unmet      0 kWh  capex   148
solar + battery  return  -80638.8  unmet   8005 kWh  capex   592
small generator  return  -25229.2  unmet   2228 kWh  capex   641
```

Under this dispatcher, unmet load at $10/kWh dominates, and the per-episode share of capital cost is small. This greedy rule never plans ahead, so the ranking says as much about the policy as about the designs. A better policy could rank them differently, which is why the design and the control are optimized together.

## Grow the battery mid-episode

During training a design can change between steps. Stored energy carries over, so the state of charge drops when the battery grows, and the added capacity is reported in `info["capex_delta"]` rather than charged to the reward:

```python
import numpy as np
from codesign_gym import make

env, base = make("Microgrid-Random")
obs, rejected, info = env.reset(seed=0)
for _ in range(3):
    obs, reward, terminated, truncated, info = env.step(np.array([-1.0, 0.0, 1.0]))  # charge, import
print(f"soc {info['soc']:.3f} of {info['embodiment']['battery_capacity_kwh']:.0f} kWh")

bigger = {**base, "battery_capacity_kwh": 900.0}
obs, reward, terminated, truncated, info = env.step(np.zeros(3), embodiment=bigger)
print(f"soc {info['soc']:.3f} of {info['embodiment']['battery_capacity_kwh']:.0f} kWh, capex_delta {info['capex_delta']:.2f}")
env.close()
```

```text
soc 1.000 of 450 kWh
soc 0.500 of 900 kWh, capex_delta 32.36
```
