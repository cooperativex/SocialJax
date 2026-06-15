# SVO — Social Value Orientation

Standard IPPO trained on **individual** rewards that the environment reshapes toward a
target social value orientation. All SVO logic is in the env (`get_svo_rewards`), toggled
via `ENV_KWARGS`.

Per step, each agent's reward becomes:

```
θ_i = arctan2(mean_others, r_i)        # reward angle (own vs others)
U_i = r_i − n · w · |θ_i − θ_ideal|    # SVO-shaped reward
```

Only `svo_target_agents` are shaped; `SVOLogWrapper` keeps the raw reward as
`original_rewards` and logs `svo_theta`.

| `ENV_KWARGS` | Meaning |
|---|---|
| `svo` | Switch — `True` to enable |
| `svo_w` | Weight (0 = selfish, larger = stronger pull); default `0.5` |
| `svo_ideal_angle_degrees` | `0°` selfish · `45°` equality · `90°` altruistic |
| `svo_target_agents` | Agents the shaping applies to |

```bash
python algorithms/train.py --algo SVO --env coins
python algorithms/train.py --algo SVO --env coins ENV_KWARGS.svo_w=0.5 ENV_KWARGS.svo_ideal_angle_degrees=45
```
