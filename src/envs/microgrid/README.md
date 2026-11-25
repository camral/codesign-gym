# Microgrid

Ids: `Microgrid-Random`, `Microgrid-Forecast`, `Microgrid-Campus`, `Microgrid-OffGridSingle`, `Microgrid-OffGridCampus`. Uses: python-microgrid (pymgrid).

An energy system of solar panels, a battery and a generator, run against random demand and sunlight.

- **Design:** solar capacity, battery size and generator rating.
- **Control:** battery, generator and grid commands; solar covers the remaining load automatically and the excess is curtailed.
- **Reward:** operating returns, with a large penalty for unmet load.
- **Design changes in `step`:** supported during training.

Industry usually sizes the system first and assumes a fixed dispatch rule. Here the two can't be judged apart: a battery is worthless to a policy that doesn't plan ahead and valuable to one that does.

| Id | Setup | Return [max / min] |
|---|---|---|
| `Microgrid-Random` | One grid-connected building, random demand and sunlight (24 h). | +720 / ≈−270,000 |
| `Microgrid-Forecast` | Random, plus noisy forecasts in the observation. | +720 / ≈−270,000 |
| `Microgrid-Campus` | Several buildings combined into one larger grid-connected microgrid. | +2,880 / ≈−1,080,000 |
| `Microgrid-OffGridSingle` | One building with no grid, only solar, battery and generator (72 h). | 0 / ≈−378,000 |
| `Microgrid-OffGridCampus` | Several buildings with no grid (72 h). | 0 / ≈−1,512,000 |

The minimums are worst-case bounds; realistic returns are far higher.
