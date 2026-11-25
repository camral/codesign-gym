# Design space

The design sizes the installation. It is a `dict` of three scalars, each a `Box((), float32)`:

| Key | Unit | Bounds (single building) | Bounds (campus) | Meaning |
|---|---|---|---|---|
| `pv_capacity_kw` | kW | 0 – 450 | 0 – 1800 | Installed solar capacity |
| `battery_capacity_kwh` | kWh | 0 – 900 | 0 – 3600 | Usable battery energy |
| `genset_rating_kw` | kW | 0 – 360 | 0 – 1440 | Rated power of the diesel generator |

Every bound is a multiple of the nominal demand peak (`demand_peak_kw`): 1.5× for solar, 3× for the battery and 1.2× for the generator.
The single-building ids (`Random`, `Forecast`, `OffGridSingle`) use a 300 kW peak; the campus ids (`Campus`, `OffGridCampus`) sum four buildings to a 1200 kW peak.
Zero is allowed for every key and means "none installed".

Some battery properties are fixed and not part of the design:

| Property | Value |
|---|---|
| Charge and discharge power limit | 0.25 × capacity (a 4-hour battery) |
| Efficiency | 0.9, applied both when charging and when discharging |
| State of charge at the start of each episode | 0.5 |

## Base design

The base design is the midpoint of each bound:

```python
from codesign_gym import make

env, base = make("Microgrid-Random")
print(base)
env.close()

env, base = make("Microgrid-Campus")
print(base)
env.close()
```

```text
{'pv_capacity_kw': 225.0, 'battery_capacity_kwh': 450.0, 'genset_rating_kw': 180.0}
{'pv_capacity_kw': 900.0, 'battery_capacity_kwh': 1800.0, 'genset_rating_kw': 720.0}
```

## Build a design

A design is a plain dict with all three keys. Python floats and NumPy scalars both work:

```python
from codesign_gym import make

env, base = make("Microgrid-OffGridSingle")
design = {"pv_capacity_kw": 400.0, "battery_capacity_kwh": 800.0, "genset_rating_kw": 120.0}
print(env.validate_embodiment(design))                         # True
print(env.validate_embodiment({**design, "pv_capacity_kw": 500.0}))  # False: above 450
env.close()
```

## Capital cost

Each design has a capital cost, paid once per episode as a lump on the first step. It is the total cost scaled by the share of a 10-year lifetime that one episode covers:

```text
capex_total = 1000 $/kW · pv + 300 $/kWh · battery + 500 $/kW · genset
first-step charge = capex_total · horizon / (10 · 8760)
```

For the base single-building design that is $450,000 in total and $123.29 on the first step of a 24-hour episode ($369.86 for the 72-hour off-grid episode).

## What is rejected

`validate_embodiment` returns `False`, and `reset` returns `rejected=True`, for a design with:

- missing or extra keys,
- a value outside its bounds,
- a non-finite value (NaN or infinity).

There are no other constraints. A design with no solar, no battery and no generator is valid; it just pays heavily for unmet load.

A rejected reset does not raise. Its observation is all zeros and its `info` is empty, and `step` raises `RuntimeError` until a reset succeeds.
An invalid design passed to `step` raises `ValueError`.

With `make_vec`, `validate_embodiment` returns a `(num_envs,)` bool array and never raises. It accepts a single dict (applied to every copy), a list of dicts (with `None` for "no design"), or a stacked dict whose values have a leading `num_envs` axis (a leading axis of 1 is applied to every copy).
Wrong keys or a batch of the wrong size give `False`.

## Changing the design mid-episode

During training, a design passed to `step` takes effect from that step on. The microgrid is rebuilt for the rest of the episode, and the stored energy (in kWh) carries over, clipped to the new battery capacity.
The change is not charged to the reward: only the first step pays capital cost. `info["capex_delta"]` reports the amortized cost of any capacity added, so you can shape your own reward with it.

With `is_eval_env=True`, a design passed to `step` is dropped with a warning. In training (`is_eval_env=False`), a design passed to `step` on a copy's autoreset step in `make_vec` becomes the design of its new episode.

## Build options

`make` and `make_vec` forward keyword arguments to the environment. Useful ones:

| Kwarg | Default | Effect |
|---|---|---|
| `horizon` | 24 (72 off-grid) | Episode length in hours |
| `demand_peak_kw` | 300 (1200 campus) | Load scale; the design bounds and grid limit scale with it |
| `n_buildings` | 1 (4 campus) | Number of buildings summed into the load |
| `off_grid` | `False` (`True` off-grid) | Remove the utility grid |
| `forecast_horizon`, `forecast_noise` | 0, 0.0 (12, 0.25 in Forecast) | Add noisy forecasts to the observation |
| `import_price`, `export_price` | 0.30, 0.05 $/kWh | Grid prices |
| `genset_fuel_cost` | 0.40 $/kWh | Generator running cost |
| `unmet_penalty`, `overgen_penalty` | 10.0, 2.0 $/kWh | Penalties for unmet load and excess generation |
| `capex_coeffs`, `lifetime_years` | see above, 10 | Capital cost model |
| `grid_max_import`, `grid_max_export` | 2 × peak (0 off-grid) | Grid limits in kW |
| `allow_transition_embodiment` | `True` | Set `False` to ignore designs passed to `step` |

```python
from codesign_gym import make

env, base = make("Microgrid-Campus", demand_peak_kw=2000.0, horizon=48)
print(base)
env.close()
```

```text
{'pv_capacity_kw': 1500.0, 'battery_capacity_kwh': 3000.0, 'genset_rating_kw': 1200.0}
```
