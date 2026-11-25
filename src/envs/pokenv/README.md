# Pokémon

Ids: `Pokenv-Random`, `Pokenv-MaxPower`, `Pokenv-Heuristic`, `Pokenv-LeagueG1` to `Pokenv-LeagueG9`, `Pokenv-LeagueG1H` to `Pokenv-LeagueG9H`, `Pokenv-LeagueRand`, `Pokenv-LeagueRandH` (23 ids).
Uses: [pokemon-codesign-env](https://github.com/camral/pokemon-codesign-env), which contains [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) and [poke-env](https://github.com/hsahovic/poke-env).

You design a Pokémon team and train a policy to battle with it.

- **Design:** the team. By default nothing is designed; pass `embodiment_config` to choose which team parameters to design.
- **Control:** one battle action per turn, with legal actions in the observation's `action_mask`.
- **Reward:** dense: the change in a battle value built from HP fractions, −2 per fainted Pokémon and ±10 for the result (see the reward section of the docs).
- **Design changes in `step`:** not supported; the team is fixed for the battle.

## Install

You need Node.js 22 or later, with `npm`.

```bash
pip install -e '.[showdown]'
codesign-gym-install-showdown
```

The env starts its own Showdown servers and stops them on `close()`.

## Use

```python
import numpy as np
from codesign_gym import make

if __name__ == "__main__":          # required: the env starts worker processes
    env, base = make("Pokenv-Random")
    obs, rejected, info = env.reset(seed=0, embodiment=base)
    action = np.flatnonzero(obs["action_mask"])[0]
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
```

Keep the `if __name__ == "__main__":` guard. Without it, `make` fails with `Could not start all sharded environment workers`.

## Ids

| Id | Opponent | Return [max / min] |
|---|---|---|
| `Pokenv-Random` | Switches Pokémon with probability 0.2, otherwise picks a random legal move. | +28 / −28 |
| `Pokenv-MaxPower` | Always picks its highest base-power move. | +28 / −28 |
| `Pokenv-Heuristic` | poke-env's type- and stat-aware heuristic player. | +28 / −28 |
| `Pokenv-LeagueG<N>` | The Elite Four and Champion of generation N in a row; a loss ends the episode. | +132 / −27 (gen 1–4, 7, 9), +140 / −28 (gen 5), +124 / −26 (gen 6), +130 / −26 (gen 8) |
| `Pokenv-LeagueRand` | A random generation's League each episode. | up to +140 / −28, depending on the generation drawn |

Add `H` to a League id for hard mode, where HP, status, PP and fainted Pokémon carry over between trainers.

## Differences from other envs

- **Same-step autoreset.** The step that ends a battle already returns the next battle's first observation; the finished battle's are in `info["final_obs"]` and `info["final_info"]`.
- **No partial reset.** All copies reset together, so giving one copy a new team means resetting them all. A partial `reset_mask` raises `NotImplementedError`.
- **`None` design** at reset means that copy's base team.
- **Rendering** uses `render_mode="html"`. Once a battle ends, `render()` returns its replay as HTML (one per copy, or `None`, for `make_vec`), which you can save and open in a browser.

## Options

Extra options to `make` and `make_vec` go straight to `pokemon_codesign_env.make_vec`, e.g. `shards`, `envs_per_worker`, `step_timeout`, `format_config`, `embodiment_config`, `observation_config`.

For the design space, team format and rejection, see the [pokemon-codesign-env docs](https://github.com/camral/pokemon-codesign-env/blob/main/docs/codesign_gym_api.md).
