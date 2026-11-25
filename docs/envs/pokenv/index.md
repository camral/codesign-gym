# Pokémon

```{include} ../../_generated/readmes/pokenv.md
```

## At a glance

| | |
|---|---|
| Ids | `Pokenv-Random`, `Pokenv-MaxPower`, `Pokenv-Heuristic`, `Pokenv-LeagueG1` … `Pokenv-LeagueG9`, `Pokenv-LeagueRand`, each League id also with an `H` (hard) suffix — 23 ids |
| Design | A team `dict`, `{}` by default; `embodiment_config` chooses what is designed |
| Observation | `Dict` of `categorical` (345), `continuous` (281), `binary` (104) and `action_mask`; League ids add `league` and `league_generation` |
| Action | `Discrete(10)` to `Discrete(26)`, set by the generation |
| Episode | One battle; a League episode is up to five battles |
| Mid-episode design changes | No (a design passed to `step` is ignored with a warning) |
| Autoreset | Same step; no partial reset |
| Render modes | `html` (battle replay) |
| Extra install | `pip install -e '.[showdown]'`, then `codesign-gym-install-showdown` (Node.js 22+) |
| Source | [`src/envs/pokenv`](../../../src/envs/pokenv) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```

```{toctree}
:maxdepth: 1
:caption: pokemon-codesign-env package

package/guide
package/design-space-api
package/env-classes
```
