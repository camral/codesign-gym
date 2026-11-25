# Codesign Gym

Codesign Gym is a set of co-design benchmarks with a Gymnasium-like API.
In co-design you optimize two things together: the **design** of an agent (its body, layout or team) and the **control policy** that runs it.
The API calls the design the *embodiment*.

```python
from codesign_gym import make

env, base = make("LQStructure-LQR")
design = env.embodiment_space.sample()
obs, rejected, info = env.reset(seed=0, embodiment=design)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc
Install the package and run a first episode.
:::

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc
Designs, rejection, vector envs, seeds and autoreset.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
A full co-design loop, mid-episode design changes, batches of designs.
:::

:::{grid-item-card} Environments
:link: envs/index
:link-type: doc
Every environment, each with its own design space, spaces and examples.
:::
::::

```{toctree}
:hidden:
:caption: Guide

getting-started
concepts
tutorials/index
```

```{toctree}
:hidden:
:caption: Environments

envs/index
```

```{toctree}
:hidden:
:caption: Reference

api
adding-an-environment
```
