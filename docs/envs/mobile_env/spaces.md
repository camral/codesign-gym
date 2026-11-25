# Observations, actions and rewards

Users move around the map with random waypoints at the preset's speed. All users are present for the whole episode. The policy is a central controller that manages every user's connections.

Below, `U` is the number of users and `M` is `max_bs`:

| Id | `U` | `M` |
|---|---|---|
| `Network-Urban` | 15 | 8 |
| `Network-Suburban` | 20 | 7 |
| `Network-Motorway` | 10 | 4 |

## Observation

A `Box(U · (2M + 1),)` float32 vector in `[-1, 1]`: one block of `2M + 1` values per user, in user order. Reshape it with `obs.reshape(U, 2 * M + 1)`. Within a user's block:

| Slice | Content |
|---|---|
| `[0:M]` | Connected to station `k` (1) or not (0) |
| `[M:2M]` | SNR to station `k`, divided by the user's best SNR (so the best station reads 1) |
| `[2M]` | The user's utility, from −1 to 1 (−1 when it has no connection) |

Slots for stations beyond `num_bs` are always 0. The design itself is not in the observation.

## Action

A `MultiDiscrete([M + 1] * U)` vector with one entry per user:

| Value | Effect |
|---|---|
| `0` | No-op: keep the user's connections |
| `k` (1 ≤ `k` ≤ `num_bs`) | **Toggle** the connection to station `k − 1`: connect if not connected and the SNR is above the user's threshold, disconnect if already connected |
| `k > num_bs` | A padded station; treated as a no-op |

The action is a toggle, not a choice of station. Sending the same station index every step connects on one step and disconnects on the next. A user can be connected to several stations at once, and its data rate is the sum over them.
Connections that fall out of range are dropped automatically before the action is applied.

`info["action_mask"]` is a `(U, M + 1)` bool array that is True for the no-op and for built stations. It marks which values are meaningful, not which toggles will succeed.

## Reward

The reward is the mean utility over users, in `[-1, 1]` per step:

```text
utility = clip(10 · log10(data rate), -20, 20) / 20        (−1 with no connection)
reward  = mean over users of utility
```

A station splits its resources equally: with `n` users connected, each gets `1/n` of the rate it would get alone. So connecting everyone to the strongest station is not always best. Station count and power are limited only by the design budget; there is no cost term in the reward.

## Episode

100 steps. Episodes end by truncation only. Each reset with a seed replays the same user movement for the same design; without a seed, every copy and every episode draws a new one.

## Info

| Key | Content |
|---|---|
| `num_bs` | Number of built stations |
| `action_mask` | `(U, M + 1)` bool mask of meaningful actions (`(num_envs, U, M + 1)` with `make_vec`) |
| `mean_utility` | Step only: same as the reward |
| `embodiment_rejected` | Rejection flag |
