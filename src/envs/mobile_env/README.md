# Wireless Network

Ids: `Network-Urban`, `Network-Suburban`, `Network-Motorway`. Uses: [mobile-env](https://github.com/stefanbschneider/mobile-env).

A cellular network serving users who move around a map.

- **Design:** how many base stations to deploy, plus each station's position `(x, y)` and transmit power, given as `num_bs` and a padded `(max_bs, 3)` array.
- **Control:** which station each user connects to or disconnects from (picking a connected station drops it).
- **Reward:** the users' mean quality of experience, from −1 to +1 per step, with no cost for stations.
- **Design changes in `step`:** not supported; `step` ignores them with a warning. A new design rebuilds the network at reset.

Two budgets limit the design: the station count (`min_bs` to `max_bs`), and the total transmit power, capped at 80% of `max_bs` stations running at `tx_max`.

Observation and action spaces are padded to `max_bs`. `info["action_mask"]` marks the deployed stations, and an action that picks a padded station does nothing.

| Id | Setup | Stations | Return [max / min] |
|---|---|---|---|
| `Network-Urban` | Pedestrians, capacity-limited, small low-power cells. | 3–8 | +100 / −100 |
| `Network-Suburban` | Slow vehicles, micro and macro cells. | 3–7 | +100 / −100 |
| `Network-Motorway` | Sparse, fast-moving users and a few high-power macro cells. | 1–4 | +100 / −100 |

Episodes last 100 steps.
