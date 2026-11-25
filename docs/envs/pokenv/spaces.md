# Observations, actions and rewards

## Observation

A `Dict` of what a human player can see on screen, split by data type. The sizes are the same for every id:

| Key | Space | Content |
|---|---|---|
| `categorical` | `MultiDiscrete(345)` int64 | Ids of species, moves, items, abilities, natures, types, status, weather, fields, side conditions, effects (0 = empty or unknown) |
| `continuous` | `Box(281,)` float32 | Turn, HP fractions, move power / accuracy / PP / priority, stat boosts, stats, side-condition counters, mostly in `[0, 1]` (boosts and priority in `[-1, 1]`) |
| `binary` | `MultiBinary(104)` | Flags: forced switch, trapped, which mechanics are available, and per Pokémon present / revealed / active / fainted and more |
| `action_mask` | `MultiBinary(n_actions)` | 1 for each legal action |

The first block of features describes the battle (turn, weather, fields, side conditions) and the active Pokémon's four
moves; then come six slots for your team and six for the opponent's, holding only what has been revealed. Hidden
opponent information (unrevealed Pokémon, moves, items) stays 0. The design itself is not a separate entry; your own
team's species, moves and items appear in your team slots as they would on screen.

League ids add two entries:

| Key | Space | Content |
|---|---|---|
| `league` | `MultiDiscrete(3)` | Current trainer index (0–4), trainers left, difficulty (0 or 1) |
| `league_generation` | `Discrete` | The League's generation (1–9) |

Every feature has a name. The encoder that builds the observation lists them, without starting a server:

```python
from pokemon_codesign_env.configs import FormatConfig
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder

encoder = HumanVisibleBattleEncoder.from_showdown(FormatConfig.from_showdown("gen4customgame"))
names = encoder.get_schema()["feature_names"]
print(len(names["categorical"]), len(names["continuous"]), len(names["binary"]))   # 345 281 104
print(names["binary"][:3])          # ['battle.force_switch', 'battle.trapped', 'battle.wait']
print(encoder.feature_indices["own_team.0.hp_fraction"])                          # ('continuous', 42)
```

`encoder.get_feature(obs, name)` reads one named value from an observation. To change what is observed, pass an
`ObservationConfig` as `observation_config` to `make`.

## Action

`Discrete(n)`, where `n` depends only on the generation:

| Generation | Ids | Actions |
|---|---|---|
| 1–5 | `Pokenv-Random`, `Pokenv-LeagueG1` … `G5` | 10 |
| 6 | `Pokenv-MaxPower`, `Pokenv-LeagueG6` | 14 |
| 7 | `Pokenv-LeagueG7` | 18 |
| 8 | `Pokenv-LeagueG8` | 22 |
| 9 | `Pokenv-Heuristic`, `Pokenv-LeagueG9`, `Pokenv-LeagueRand` | 26 |

| Index | Action |
|---|---|
| 0–5 | Switch to team slot 0–5 |
| 6–9 | Use move 1–4 |
| 10–13 | Move and Mega Evolve |
| 14–17 | Move and Z-move |
| 18–21 | Move and Dynamax |
| 22–25 | Move and Terastallize |

Always sample from `action_mask`. An illegal action does not raise: poke-env sends Showdown's default choice instead.

## Reward

The reward is the change, each step, in a value of the battle state:

```text
value = Σ_own (hp_fraction − 2 · fainted) − Σ_opponent (hp_fraction − 2 · fainted) ± 10 (win / loss)
```

Opponent Pokémon not yet revealed count at full HP, so the value starts at 0 (in hard-mode Leagues, later battles can start below 0 because damage carries over). A six-against-six battle therefore returns
at most +28 (a win without losing any HP: 6 + 2 · 6 + 10) and at least −28. A League episode adds up its battles, so it
returns at most +140 (five such wins) and, since a loss ends it, at least −28. Most League trainers have fewer than six Pokémon,
so the bounds depend on the generation: +132 / −27 (gen 1–4, 7, 9), +140 / −28 (gen 5), +124 / −26 (gen 6) and +130 / −26 (gen 8).
With a team size other than six, the bounds shift accordingly.

## Episode

An episode is one battle against the id's opponent. The baseline opponents field a new random team each battle
(`gen4randombattle`, `gen6randombattle` and `gen9randombattle` sets). In a League episode, beating a trainer moves straight
on to the next one inside the same episode, and the episode ends after a loss or a win against the Champion. Hard-mode
ids (`H`) carry HP, fainted Pokémon, status, PP and lost held items over between trainers; easy mode heals the team.

A won or lost battle ends with `terminated=True`. Each side is counted against its own team size, so a win over a
League trainer with fewer Pokémon than your team is a termination too, and the League goes on to the next trainer.
`truncated=True` marks any other finish, such as a tie.

## Autoreset

Autoreset is Gymnasium's `SAME_STEP` mode, unlike the other codesign-gym envs. The step that ends a battle already returns
the first observation of the next battle, with the same design; the last observation and info of the finished battle are
in `info["final_obs"]` and `info["final_info"]`. Partial resets are not supported: `reset(options={"reset_mask": m})`
raises `NotImplementedError` unless every entry of `m` is true.

## Info

| Key | When | Content |
|---|---|---|
| `embodiment_rejected` | reset | Whether the design was rejected |
| `embodiment_rejection_reason` | reset, if rejected | Why |
| `final_obs`, `final_info` | battle end | The finished battle's last observation and info |
| `episode` | battle end | `{"r": return, "l": length, "t": seconds}` of the finished episode, as Gymnasium's `RecordEpisodeStatistics` writes them but counted for same-step autoreset, so they are exact |
| `replay_html` | battle end, `render_mode="html"` | The battle replay as HTML |
| `generation`, `random_league`, `difficulty`, `trainer_index`, `trainer`, `trainers_defeated` | League ids, every step | League progress |
| `battle_won`, `league_complete`, `defeated_trainer` | League ids, after a battle | The battle's result (`defeated_trainer` only after a won battle that is not the last; `battle_won` is absent when a battle is truncated) |
| `sharded_worker_restarted` | after a timeout | A worker exceeded `step_timeout` (default 60 s) and was restarted; the step returns reward `step_timeout_reward` (default 0) |

From `make_vec`, each key holds one value per copy, with a boolean `_key` array marking which copies set it, as in
Gymnasium.
