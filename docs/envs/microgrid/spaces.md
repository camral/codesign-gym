# Observations, actions and rewards

Each step is one hour. Demand and sunlight come from the real hourly profiles bundled with python-microgrid: a city (Houston, Raleigh, San Francisco, New York or Minneapolis) for solar, and one or more reference buildings (hotel, hospital, restaurant, school, office) for load.
Every reset samples a city, the buildings and a random start hour in the year. Each building's load is scaled so its yearly peak is 300 kW.

## Observation

A `Box(0, inf)` float32 vector:

| Index | Content |
|---|---|
| `[0]` | Battery state of charge, 0 – 1 |
| `[1]` | Demand this hour, kW |
| `[2]` | Solar power available this hour, kW (designed capacity × capacity factor) |
| `[3]` | Hour of day / 24 |
| `[4]` | Grid import price, $/kWh (constant 0.30) |
| `[5:5+H]` | Forecast only: demand for the next `H` hours |
| `[5+H:5+2H]` | Forecast only: available solar for the next `H` hours |

`Microgrid-Forecast` uses `H = 12`, giving 29 values. Each forecast value has independent Gaussian noise with a standard deviation of 25% of the value (redrawn every step) and is clipped at 0.
The design itself is not in the observation, but solar availability scales with `pv_capacity_kw`.

## Action

A `Box(3,)` vector in `[-1, 1]`, scaled by the design:

| Index | Meaning | Scale |
|---|---|---|
| `[0]` | Battery: + discharge, − charge | × 0.25 × `battery_capacity_kwh` kW |
| `[1]` | Generator output; ≤ 0 turns it off | × `genset_rating_kw` kW |
| `[2]` | Grid: + import, − export | × grid limit (600 kW single building, 2400 kW campus, 0 off-grid) |

Some details matter for a policy:

- **Solar is not an action.** After your battery, generator and grid commands, solar fills whatever demand is left, up to what is available. Any solar not needed is curtailed automatically (`info["dispatch"]["pv_curtailment"]`).
- **Nothing is imported unless you ask.** An all-zero action leaves the load to solar alone, and the shortfall is unmet load at $10/kWh.
- **Oversupply is penalized.** Energy from the battery, generator and grid beyond the demand counts as over-generation at $2/kWh.
- **Battery losses.** Discharging at full power delivers 0.9 × the power limit to the load; charging at full power stores 0.9 × what it draws. The battery cannot discharge below empty or charge above full.
- **Exporting is a load.** Grid export must itself be supplied, so exporting more than the surplus creates unmet load.
- Off-grid, the grid action does nothing.

## Reward

The reward is the negative operating cost of the hour, minus the capital-cost lump on the first step:

```text
reward = -( 0.30 · grid_import − 0.05 · grid_export + 0.40 · genset_output
            + 10 · unmet_load + 2 · overgeneration )  −  capex charge (first step only)
```

All energies are kWh over the hour. The prices and penalties are build options (see [Design space](embodiment.md)).

## Episode

24 steps (one day) in `Random`, `Forecast` and `Campus`; 72 steps (three days) in the two off-grid ids. Episodes end by truncation only.

## Info

| Key | Content |
|---|---|
| `embodiment` | The design in use |
| `capex` | The first-step capital charge for this design |
| `capex_total` | The design's total capital cost, $ |
| `city`, `building` | Reset only: the sampled solar city and buildings (joined with `+`) |
| `capex_charged` | Step only: capital cost taken from this step's reward (non-zero on the first step) |
| `capex_delta` | Step only: amortized cost of capacity added by a mid-episode design change (not in the reward) |
| `op_cost` | Step only: operating cost of the hour, $ |
| `soc` | Step only: state of charge after the step |
| `dispatch` | Step only: a dict of kW per source: `demand`, `pv`, `pv_curtailment`, `battery`, `genset`, `grid`, `unmet`, `overgen` (battery and grid signed as in the action) |
| `embodiment_rejected` | Step only: rejection flag |

With `make_vec`, numeric keys become arrays of shape `(num_envs,)` and the others (`embodiment`, `dispatch`, `city`, `building`) become lists with one entry per copy.
