# Agent-Centric Analysis of Role Specialization and Cooperative Behavior in MARL

Code for the PRIMA 2026 paper:

> Hiroki Kuwano, Masayuki Tanaka, Takako Yoshida.
> *Agent-Centric Analysis of Role Specialization and Cooperative Behavior in
> Multi-Agent Reinforcement Learning.*
> PRIMA 2026, Kumamoto, Japan. Springer, Lecture Notes in Artificial Intelligence.

This repository is a snapshot of the code used to produce the results in the
paper. It is not under active development; ongoing work happens elsewhere.

## What the paper does

Agents are trained with Independent PPO on their own local observations in the
Overcooked-V2 `cramped_room_3x4` layout, under three training regimes, and all
pairings of the resulting agents are evaluated.

| Agent type | Training regime | Code |
| --- | --- | --- |
| **A** | single-agent training | `experiments/solo/` |
| **B** | two-agent, shared policy parameters | `experiments/shared/` |
| **C**, **D** | two-agent, independent policy parameters (player 0, player 1) | `experiments/independent/` |

Evaluation of all pairings is in `experiments/evaluation/`.

## Setup

Requires Python 3.10 and a GPU-enabled JAX install.

```bash
pip install -e . --no-deps
```

### Patching jaxmarl

The PyPI build of `jaxmarl` fails on `import jaxmarl` because of the
mabrax to brax to glfw dependency chain. Two files in the installed package need
a local edit.

**`<jaxmarl>/environments/__init__.py`**

```python
# before
from .mabrax import Ant, Humanoid, Hopper, Walker2d, HalfCheetah

# after
try:
    from .mabrax import Ant, Humanoid, Hopper, Walker2d, HalfCheetah
except ImportError:
    pass
```

**`<jaxmarl>/registration.py`**

Remove `Ant, Humanoid, Hopper, Walker2d, HalfCheetah` from the import block at
the top and add the following just after it:

```python
try:
    from .environments import Ant, Humanoid, Hopper, Walker2d, HalfCheetah
    _MABRAX_AVAILABLE = True
except ImportError:
    _MABRAX_AVAILABLE = False
```

Then guard the `# 4. MABrax` branch inside `def make(...)` with
`if _MABRAX_AVAILABLE:`.

This is a local change to the installed package and is not included here. It has
to be reapplied if `jaxmarl` is reinstalled or upgraded.

Check the install:

```bash
python -c "from overcooked_research.env import OvercookedV2WithEvents; print('OK')"
```

## Running

Training, 16 seeds per regime:

```bash
python experiments/solo/train.py        ENV_KWARGS.layout=cramped_room_3x4 NUM_SEEDS=16 SEED=0
python experiments/shared/train.py      ENV_KWARGS.layout=cramped_room_3x4 NUM_SEEDS=16 SEED=0
python experiments/independent/train.py ENV_KWARGS.layout=cramped_room_3x4 NUM_SEEDS=16 SEED=0
```

Weights and the config that produced them are written to
`experiments/<exp>/results/<timestamp>/seed_<i>/`.

Evaluation:

```bash
python experiments/evaluation/evaluate.py
```

Pairings are listed in a YAML file giving a checkpoint per player; see
`experiments/evaluation/config.yaml` and `experiments/evaluation/sameseed/` for
examples. Each evaluation runs 256 rollouts and writes `episodes.csv` and
`steps.csv.gz`.

`experiments/<exp>/tsubame/` holds the job scripts used on the TSUBAME4.0
supercomputer. They are specific to that system and are not needed to run the
code locally.

## Repository layout

```
src/overcooked_research/
  env/      Overcooked-V2 subclass with an event log
  train/    network, Transition, batchify and other shared pieces
  eval/     evaluation loop
experiments/
  solo/ shared/ independent/   training, one folder per regime
  evaluation/                  all-pairings evaluation
```

Each `experiments/<exp>/` holds `train.py`, `config.yaml`, `tsubame/run.sh` and
its own `README.md`.

## Citation

```bibtex
@inproceedings{kuwano2026agentcentric,
  title     = {Agent-Centric Analysis of Role Specialization and Cooperative
               Behavior in Multi-Agent Reinforcement Learning},
  author    = {Kuwano, Hiroki and Tanaka, Masayuki and Yoshida, Takako},
  booktitle = {PRIMA 2026: Principles and Practice of Multi-Agent Systems},
  series    = {Lecture Notes in Artificial Intelligence},
  publisher = {Springer},
  year      = {2026}
}
```

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

Built on [JaxMARL](https://github.com/flairox/jaxmarl).
