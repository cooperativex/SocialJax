# IRAT Implementation Guide

## Algorithm Overview

IRAT (Individual Reward Assisted Team Policy Learning) uses:
- **2 Policies**: Individual Policy π^i and Team Policy π̂^i
- **2 Critics**: Individual Critic V^i and Team Critic V̂^i
- **2 Rewards**: Individual reward r^i and Team reward r̂ (sum of all individual rewards)

## Key Modifications from MAPPO

### 1. Network Architecture Changes

```python
# MAPPO (1 actor + 1 critic):
actor_network = Actor(...)
critic_network = Critic(...)

# IRAT (2 actors + 2 critics):
ind_actor_network = Actor(...)      # Individual policy π^i
ind_critic_network = Critic(...)    # Individual critic V^i (uses local obs)
team_actor_network = Actor(...)     # Team policy π̂^i
team_critic_network = Critic(...)   # Team critic V̂^i (uses global state)
```

### 2. Transition Data Structure

```python
class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray

    # Individual policy
    ind_action: jnp.ndarray
    ind_value: jnp.ndarray
    ind_log_prob: jnp.ndarray

    # Team policy
    team_action: jnp.ndarray
    team_value: jnp.ndarray
    team_log_prob: jnp.ndarray

    # Rewards
    ind_reward: jnp.ndarray   # r^i
    team_reward: jnp.ndarray  # r̂ = sum(r^i)

    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
```

### 3. Environment Step (_env_step function)

```python
def _env_step(runner_state, unused):
    train_states, env_state, last_obs, last_done, rng = runner_state

    # SELECT ACTIONS FROM BOTH POLICIES
    obs_batch = prepare_obs(last_obs)

    # Individual policy samples action
    ind_pi = ind_actor_network.apply(train_states[0].params, obs_batch)
    ind_action = ind_pi.sample(seed=rng)
    ind_log_prob = ind_pi.log_prob(ind_action)

    # Team policy samples action (executed in environment)
    team_pi = team_actor_network.apply(train_states[2].params, obs_batch)
    team_action = team_pi.sample(seed=rng)
    team_log_prob = team_pi.log_prob(team_action)

    # EXECUTE TEAM POLICY ACTION (as per IRAT paper)
    env_act = unbatchify(team_action, env.agents, ...)

    # COMPUTE VALUES
    # Individual critic uses local observation
    ind_value = ind_critic_network.apply(train_states[1].params, obs_batch)

    # Team critic uses global state
    world_state = prepare_world_state(last_obs)
    team_value = team_critic_network.apply(train_states[3].params, world_state)

    # STEP ENVIRONMENT
    obsv, env_state, reward, done, info = env.step(env_state, env_act)

    # COMPUTE REWARDS
    ind_reward = batchify_numpy(reward, ...)  # Individual rewards
    team_reward = ind_reward.sum(axis=0)  # Team reward = sum of individual

    transition = Transition(
        global_done=done["__all__"],
        done=last_done,
        ind_action=ind_action,
        ind_value=ind_value,
        ind_log_prob=ind_log_prob,
        team_action=team_action,
        team_value=team_value,
        team_log_prob=team_log_prob,
        ind_reward=ind_reward,
        team_reward=team_reward,
        obs=obs_batch,
        world_state=world_state,
        info=info
    )

    return runner_state, transition
```

### 4. Advantage Calculation (Dual GAE)

```python
def _calculate_gae(traj_batch, last_ind_val, last_team_val):
    # Individual advantages A^i
    def _get_ind_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.ind_value, transition.ind_reward
        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        return (gae, value), gae

    _, ind_advantages = jax.lax.scan(
        _get_ind_advantages,
        (jnp.zeros_like(last_ind_val), last_ind_val),
        traj_batch,
        reverse=True,
    )
    ind_targets = ind_advantages + traj_batch.ind_value

    # Team advantages Â^i
    def _get_team_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.team_value, transition.team_reward
        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        return (gae, value), gae

    _, team_advantages = jax.lax.scan(
        _get_team_advantages,
        (jnp.zeros_like(last_team_val), last_team_val),
        traj_batch,
        reverse=True,
    )
    team_targets = team_advantages + traj_batch.team_value

    return ind_advantages, ind_targets, team_advantages, team_targets
```

### 5. Loss Functions (4 separate losses)

```python
def _update_minbatch(train_states, batch_info):
    ind_actor_ts, ind_critic_ts, team_actor_ts, team_critic_ts = train_states
    traj_batch, ind_advantages, ind_targets, team_advantages, team_targets = batch_info

    # --- INDIVIDUAL POLICY LOSS ---
    def _ind_actor_loss_fn(params, traj_batch, gae):
        pi = ind_actor_network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.ind_action)

        ratio = jnp.exp(log_prob - traj_batch.ind_log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        entropy = pi.entropy().mean()
        return loss_actor - config["ENT_COEF"] * entropy, (loss_actor, entropy)

    # --- INDIVIDUAL CRITIC LOSS ---
    def _ind_critic_loss_fn(params, traj_batch, targets):
        value = ind_critic_network.apply(params, traj_batch.obs)
        value_pred_clipped = traj_batch.ind_value + (value - traj_batch.ind_value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        return config["VF_COEF"] * value_loss, value_loss

    # --- TEAM POLICY LOSS ---
    def _team_actor_loss_fn(params, traj_batch, gae):
        pi = team_actor_network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.team_action)

        ratio = jnp.exp(log_prob - traj_batch.team_log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        entropy = pi.entropy().mean()
        return loss_actor - config["ENT_COEF"] * entropy, (loss_actor, entropy)

    # --- TEAM CRITIC LOSS ---
    def _team_critic_loss_fn(params, traj_batch, targets):
        value = team_critic_network.apply(params, traj_batch.world_state)
        value_pred_clipped = traj_batch.team_value + (value - traj_batch.team_value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        return config["VF_COEF"] * value_loss, value_loss

    # Compute gradients and update all 4 networks
    ind_actor_loss, ind_actor_grads = jax.value_and_grad(_ind_actor_loss_fn, has_aux=True)(
        ind_actor_ts.params, traj_batch, ind_advantages
    )
    ind_critic_loss, ind_critic_grads = jax.value_and_grad(_ind_critic_loss_fn, has_aux=True)(
        ind_critic_ts.params, traj_batch, ind_targets
    )
    team_actor_loss, team_actor_grads = jax.value_and_grad(_team_actor_loss_fn, has_aux=True)(
        team_actor_ts.params, traj_batch, team_advantages
    )
    team_critic_loss, team_critic_grads = jax.value_and_grad(_team_critic_loss_fn, has_aux=True)(
        team_critic_ts.params, traj_batch, team_targets
    )

    # Apply gradients
    ind_actor_ts = ind_actor_ts.apply_gradients(grads=ind_actor_grads)
    ind_critic_ts = ind_critic_ts.apply_gradients(grads=ind_critic_grads)
    team_actor_ts = team_actor_ts.apply_gradients(grads=team_actor_grads)
    team_critic_ts = team_critic_ts.apply_gradients(grads=team_critic_grads)

    train_states = (ind_actor_ts, ind_critic_ts, team_actor_ts, team_critic_ts)

    return train_states, loss_info
```

### 6. Runner State Structure

```python
# MAPPO:
runner_state = ((actor_train_state, critic_train_state), env_state, obs, done, rng)

# IRAT:
runner_state = (
    (ind_actor_train_state, ind_critic_train_state, team_actor_train_state, team_critic_train_state),
    env_state,
    obs,
    done,
    rng
)
```

## Key Implementation Notes

1. **Action Execution**: Execute TEAM policy action in environment (not individual)
2. **Reward Collection**: Compute team reward as sum of individual rewards
3. **Critic Inputs**:
   - Individual critic: Local observation
   - Team critic: Global state (concatenated observations)
4. **Training**: All 4 networks updated simultaneously each epoch
5. **Evaluation**: Use TEAM policy for final evaluation

## Testing Checklist

- [ ] Individual policy samples actions correctly
- [ ] Team policy samples actions correctly
- [ ] Team actions executed in environment
- [ ] Individual rewards collected
- [ ] Team reward = sum(individual rewards)
- [ ] Individual advantages computed with ind_reward
- [ ] Team advantages computed with team_reward
- [ ] All 4 networks update successfully
- [ ] Loss values are reasonable
- [ ] Training converges

## References

- Paper: https://arxiv.org/abs/2202.03612
- GitHub: https://github.com/MDrW/ICML2022-IRAT
