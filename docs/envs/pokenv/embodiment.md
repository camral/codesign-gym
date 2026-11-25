# Design space

The design is your side's team. Which parts of it are designed is up to you: by default nothing is, and the design is the
empty dict `{}`.

## The default: nothing designed

Without an `embodiment_config`, the design space is empty and every battle uses a fresh random team, redrawn at each reset:

```python
from codesign_gym import make

if __name__ == "__main__":          # required: the env starts worker processes
    env, base = make("Pokenv-Random")
    print(base)                     # {}
    print(env.embodiment_space)     # TeamDesignSpace(0 Pokemon, 0 indices)
    env.close()
```

That random team has six Pokémon, like the observation's six team slots and the baseline opponents' teams. The default
`team_size` is six, or the format's maximum if that is smaller; set `team_size` in an `embodiment_config` (below) to
change it.

## Choosing what to design

Pass a `TeamSearchConfig` as `embodiment_config` to `make` or `make_vec`. Each parameter set to `OPTIMIZED` becomes part
of the design; the rest are filled in by the env (randomly or with Showdown's default). The config's format must be the
one the id battles in:

| Id | Format |
|---|---|
| `Pokenv-Random` | `gen4customgame` |
| `Pokenv-MaxPower` | `gen6customgame` |
| `Pokenv-Heuristic` | `gen9customgame` |
| `Pokenv-LeagueG<N>`, `Pokenv-LeagueG<N>H` | `gen<N>pokemonleague` |
| `Pokenv-LeagueRand`, `Pokenv-LeagueRandH` | `gen9pokemonleague` |

The designable parameters are `species`, `moves`, `evs`, `ivs`, `items`, `natures`, `abilities`, `happiness`, `gender`
and `tera_types` (Gen 9 only). `moves`, `abilities`, `gender` and `tera_types` can only be designed together with
`species`. The config also takes limits such as `team_size`, `ev_budget` and `generations_allowed`; the
[package's design-space page](package/design-space-api.md) covers them.

Design the species and nature of six Pokémon:

```python
from codesign_gym import make
from pokemon_codesign_env.configs import AttributeMode, FormatConfig, SelectionMode, TeamSearchConfig

config = TeamSearchConfig(format=FormatConfig.from_showdown("gen4customgame"), team_size=6,
                          species=SelectionMode.OPTIMIZED, natures=AttributeMode.OPTIMIZED)

if __name__ == "__main__":
    env, base = make("Pokenv-Random", embodiment_config=config)
    print(env.embodiment_space)                 # TeamDesignSpace(6 Pokemon, 12 indices)
    print(env.embodiment_space.spaces["pokemon_0"])
    # Dict('nature': Categorical(25 values), 'species': Categorical(525 values))
    print(base["pokemon_0"])                    # a random member, e.g. {'species': 'Heatran', 'nature': 'Bold'}
    env.close()
```

## The team dict

A design is a dict with one entry per Pokémon, `pokemon_0` to `pokemon_<team_size - 1>`, each holding the designed
parameters only:

```python
design = {
    "pokemon_0": {"species": "Garchomp", "nature": "Jolly"},
    "pokemon_1": {"species": "Gengar", "nature": "Timid"},
    # ... up to pokemon_5
}
```

With everything designed (here in Gen 9), one Pokémon's entry has these keys:

| Key | Value |
|---|---|
| `species` | Species name |
| `moves` | List of move names (legal for the species) |
| `item` | Held item |
| `ability` | Ability (legal for the species) |
| `nature` | Nature name |
| `evs`, `ivs` | `{"hp", "atk", "def", "spa", "spd", "spe"}` → int, capped per stat and by a total budget |
| `happiness` | 0–255 |
| `gender` | `"M"`, `"F"` or `"N"` |
| `tera_type` | Tera type |

The bounds (how many moves, the EV caps, which species) come from the format. Custom Game formats are permissive: Gen 9
Custom Game allows 1–24 moves and 255 EVs in every stat.

## Flat vectors

For optimizers that want a fixed-width integer vector, `embodiment_space.flat_space` is a `MultiDiscrete` view, and
`to_index` / `from_index` convert exactly between it and the dict:

```python
from codesign_gym import make

if __name__ == "__main__":
    env, base = make("Pokenv-Random", embodiment_config=config)
    space = env.embodiment_space
    print(space.flat_space)                     # MultiDiscrete([ 25 525  25 525 ... ]), 12 entries
    row = space.to_index(base)
    print(space.from_index(row) == base)        # True
    env.close()
```

Most points of the box are not legal designs, and `from_index` raises on them rather than repairing them. Sample with
`embodiment_space.sample()` and mutate from there. The [package page](package/design-space-api.md#flat-vectors) explains
why, and how indices relate to each species' legal values.

## Base design

`base` (and each copy's base team in `make_vec`, kept in `envs.base_embodiments`) is one `sample()` of the design space,
drawn when the env is built. It is not seeded, so it differs between builds and between copies. Passing `None` to
`reset` (or a `None` entry in a per-copy list) uses that copy's base team.

## Rejection

`validate_embodiment` rejects only designs Showdown cannot build, never ones that battle badly. It runs two checks:
membership in the space (`embodiment_space.contains`, microseconds) and a full team build by Showdown (a fraction of a
second). `contains` alone can pass a design Showdown refuses, for example two moves that are each legal for a species but
not together.

A rejected design at `reset` does not raise. The reset returns `rejected=True`, the reason is in
`info["embodiment_rejection_reason"]`, and `step` raises until a reset succeeds:

```python
import copy
from codesign_gym import make

if __name__ == "__main__":
    env, base = make("Pokenv-Random", embodiment_config=config)
    bad = copy.deepcopy(base)
    bad["pokemon_0"]["species"] = "Not-A-Pokemon"
    print(env.embodiment_space.contains(bad))   # False
    print(env.validate_embodiment(bad))         # False
    obs, rejected, info = env.reset(seed=0, embodiment=bad)
    print(rejected, info["embodiment_rejection_reason"])
    # True embodiment is not a member of embodiment_space
    try:
        env.step(0)
    except RuntimeError as error:
        print("step refused")                   # step refused
    env.close()
```

With `make_vec`, `validate_embodiment` takes one design or a per-copy list and returns a `(num_envs,)` bool array, and
the vector `step` refuses to run while any copy holds a rejected design. A rejected copy's observation is all zeros with
only action 0 legal.

The team is fixed for a battle, so designs change only at `reset`. Because every copy resets together, giving one copy a
new team means resetting all of them. Pass the other copies' current designs again to keep them; a `None` entry puts a
copy back on its base team.
