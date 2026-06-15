# TRANSFER — self-interest reward mixing

Standard IPPO trained on **individual** rewards mixed by a self-interest weight `s`:

```
r_i' = s · r_i + (1 − s) / (n − 1) · Σ_{j≠i} r_j
```

`s = 1` is fully selfish; `s = 1/n` weights everyone equally. Each env sets its own
`s_interest` (see `transfer_cnn_<env>.yaml`).

## `s` vs `ratio` (depends on agent count `n`)

`ratio` = how much each *other* agent's reward is weighted relative to your own:

```
ratio = (1 − s) / ((n − 1) · s)        ⇔        s = 1 / (1 + ratio · (n − 1))
```

So the same `ratio` maps to a different `s` as `n` grows. All shipped envs use
`ratio = 1/2` (each other agent counts half as much as yourself):

| Env | `n` | `s_interest` (`ratio=1/2`) |
|---|---|---|
| coins | 2 | 0.6667 |
| pd_arena | 4 | 0.4 |
| gift, mushrooms | 5 | 0.3333 |
| coop_mining | 6 | 0.2857 |
| cleanup, harvest_open/closed/partnership | 7 | 0.25 |

The exact `s` value for each env can be found in its `transfer_cnn_<env>.yaml` config.

The training loop passes `current_timestep` into `env.step`, so `s` can be **scheduled**
over training via `s_interest_schedule` + `s_interest_change_every` (phased self-interest
transfer). Without a schedule, the fixed `s_interest` is used.

| `ENV_KWARGS` | Meaning |
|---|---|
| `interest` | Switch — `True` to enable |
| `s_interest` | Fixed self-interest weight `s` |
| `s_interest_schedule` | Optional list of `s` values cycled over phases |
| `s_interest_change_every` | Steps per schedule phase |

```bash
python algorithms/train.py --algo TRANSFER --env coins
python algorithms/train.py --algo TRANSFER --env pd_arena ENV_KWARGS.s_interest=0.4
```
