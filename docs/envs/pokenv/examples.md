# Examples

Every script that builds a Pokémon env needs the `if __name__ == "__main__":` guard, because the env starts worker
processes with `spawn`. Building an env also starts its own Showdown server, which `close()` stops.

## A masked random policy

Play one full battle against `Pokenv-Random`, picking uniformly among the legal actions. Nothing is designed, so both
sides field a random team of six (see [Design space](embodiment.md#the-default-nothing-designed)):

```python
import numpy as np
from codesign_gym import make

if __name__ == "__main__":
    env, base = make("Pokenv-Random")
    rng = np.random.default_rng(0)
    obs, rejected, info = env.reset(seed=0, embodiment=base)
    total, turns, done = 0.0, 0, False
    while not done:
        action = rng.choice(np.flatnonzero(obs["action_mask"]))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        turns += 1
        done = terminated or truncated
    print(f"{turns} actions, return {total:+.2f}")   # the return lies in [-28, 28]
    env.close()
```

The seed fixes both teams, not the result: damage rolls and critical hits come from the Showdown server.

## Design species and natures for a population

Design the species and nature of all six Pokémon, and score four candidate teams in parallel, one per copy. Each copy
plays one battle with a random policy; a real method would train the policy too. Copies autoreset on their own, so only
the reward of each copy's first battle is counted:

```python
import numpy as np
from codesign_gym import make_vec
from pokemon_codesign_env.configs import AttributeMode, FormatConfig, SelectionMode, TeamSearchConfig

if __name__ == "__main__":
    config = TeamSearchConfig(format=FormatConfig.from_showdown("gen4customgame"), team_size=6,
                              species=SelectionMode.OPTIMIZED, natures=AttributeMode.OPTIMIZED)
    envs, base = make_vec("Pokenv-Random", 4, embodiment_config=config)
    space = envs.embodiment_space
    space.seed(0)
    population = [space.sample() for _ in range(4)]
    print(envs.validate_embodiment(population))       # [ True  True  True  True]

    rng = np.random.default_rng(0)
    obs, rejected, info = envs.reset(seed=0, embodiment=population)
    scores, finished = np.zeros(4), np.zeros(4, dtype=bool)
    while not finished.all():
        actions = np.array([rng.choice(np.flatnonzero(mask)) for mask in obs["action_mask"]])
        obs, reward, terminated, truncated, info = envs.step(actions)
        scores += np.where(finished, 0.0, reward)
        finished |= terminated | truncated
    for design, score in zip(population, scores):
        print(f"{score:+6.2f}", [mon["species"] for mon in design.values()])
    envs.close()
```

An optimizer that works on vectors can pass a `(4, 12)` array of `space.to_index(design)` rows to `reset` instead of
the dicts. Rejected designs come back flagged in `rejected` rather than raising; drop or penalize them and reset again
before stepping.

## A League

`Pokenv-LeagueG4` is the Platinum Elite Four (Aaron, Bertha, Flint, Lucian) and then Cynthia. An `embodiment_config` for a
League id uses the League's format. The episode's results are in the final step's `info["final_info"]`:

```python
import numpy as np
from codesign_gym import make

if __name__ == "__main__":
    env, base = make("Pokenv-LeagueG4")
    rng = np.random.default_rng(0)
    obs, rejected, info = env.reset(seed=0)
    print(info["trainer"], obs["league"])          # Aaron [0 5 0]
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(rng.choice(np.flatnonzero(obs["action_mask"])))
        total += reward
        done = terminated or truncated
    final = info["final_info"]
    print(f"return {total:+.2f}, trainers defeated: {final['trainers_defeated']}, last trainer: {final['trainer']}")
    env.close()
```

`Pokenv-LeagueRand` draws a League of a random generation for each episode (always in the Gen 9 League format), and the
`H` ids carry damage over between trainers.

## Save a battle replay

With `render_mode="html"`, `render()` returns the last finished battle as a self-contained HTML replay (`None` before
the first battle ends). Open the saved file in a browser:

```python
import tempfile
from pathlib import Path

import numpy as np
from codesign_gym import make

if __name__ == "__main__":
    env, base = make("Pokenv-Random", render_mode="html")
    rng = np.random.default_rng(0)
    obs, rejected, info = env.reset(seed=0)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(rng.choice(np.flatnonzero(obs["action_mask"])))
        done = terminated or truncated
    path = Path(tempfile.gettempdir()) / "pokenv_replay.html"
    path.write_text(env.render(), encoding="utf-8")
    print(f"saved {path}")
    env.close()
```

With `make_vec`, `render()` returns one replay (or `None`) per copy.
