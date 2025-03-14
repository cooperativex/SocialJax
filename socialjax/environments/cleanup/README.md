# Clean Up

Clean Up is a multiplayer game where players earn rewards by collecting apples, each worth +1 reward. Apples grow in an orchard and their regrowth depends on the cleanliness of a nearby river. Pollution accumulates in the river at a constant rate and once pollution surpasses a certain threshold, the apple growth rate drops to zero. Players have the option to perform a cleaning action that removes small amounts of pollution from the river. For continuous apple growth, the group must keep river pollution levels consistently low over time. 

```python
import jax
import jax.numpy as jnp
from PIL import Image
from socialjax import make
from pathlib import Path
import math


# load environment
num_agents=7
grid_size = (16,22)
num_inner_steps=100
num_outer_steps=1
rng = jax.random.PRNGKey(123)
env = make('clean_up',
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
    )
rng, _rng = jax.random.split(rng)

root_dir = f"random_actions_gif/a{num_agents}_g{grid_size}_i{num_inner_steps}_o{num_outer_steps}"
path = Path(root_dir + "/state_pics")
path.mkdir(parents=True, exist_ok=True)

for o_t in range(num_outer_steps):
    obs, old_state = env.reset(_rng)

    # render each timestep
    pics = []
    pics1 = []
    pics2 = []

    img = env.render(old_state)
    Image.fromarray(img).save(f"{root_dir}/state_pics/init_state.png")
    pics.append(img)

    for t in range(num_inner_steps):

        rng, *rngs = jax.random.split(rng, num_agents+1)
        actions = [jax.random.choice(
            rngs[a],
            a=env.action_space(0).n,
            p=jnp.array([0.1, 0.1, 0.09, 0.09, 0.09, 0.19, 0.05, 0.1, 0.5])
        ) for a in range(num_agents)]

        obs, state, reward, done, info = env.step_env(
            rng, old_state, [a for a in actions]
        )

        print('###################')
        print(f'timestep: {t} to {t+1}')
        print(f'actions: {[action.item() for action in actions]}')
        print("###################")

        img = env.render(state)
        Image.fromarray(img).save(
            f"{root_dir}/state_pics/state_{t+1}.png"
        )
        pics.append(img)

        old_state = state

    # create and save gif
    print("Saving GIF")
    pics = [Image.fromarray(img) for img in pics]
    pics[0].save(
    f"{root_dir}/state_outer_step_{o_t+1}.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=200,
    loop=0,
    )
```