<h1 align="center">SocialJax</h1>


*A suite of sequential social dilemma environments for multi-agent reinforcement learning in JAX*



<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="/docs/images/step_150_reward_common_coins.gif" alt="coins_common" width="10%">
            <img src="/docs/images/step_150_reward_common_harvestopen.gif" alt="harvest_open_common" width="18.5%">
            <img src="/docs/images/step_150_reward_common_closed.gif" alt="harvest_closed_common" width="18.5%">
            <img src="/docs/images/step_150_reward_common_cleanup.gif" alt="clean_up_common" width="19.8%">
            <img src="/docs/images/step_250_reward_common_coop_mining.gif" alt="coop_mining_common" width="14%">
        </div>
        <div class="row" align="centre">
            <img src="/docs/images/step_150_reward_individual_coins.gif" alt="coins_individual" width="10%">
            <img src="/docs/images/step_150_reward_individual_harvestopen.gif" alt="harvest_open_individual" width="18.5%">
            <img src="/docs/images/step_150_reward_individual_closed.gif" alt="harvest_closed_individual" width="18.5%">
            <img src="/docs/images/step_150_reward_individual_cleanup.gif" alt="clean_up_individual" width="19.8%">
            <img src="/docs/images/step_250_reward_individual_coop_mining.gif" alt="coop_mining_individual" width="14%">
        </div>
    </div>
</div>


SocialJax leverages JAX's high-performance GPU capabilities to accelerate multi-agent reinforcement learning in sequential social dilemmas. We are committed to providing a more efficient and diverse suite of environments for studying social dilemmas. We provide JAX implementations of the following environments: Coins, Commons Harvest: Open, Commons Harvest: Closed, Clean Up, Territory, and Coop Mining, which are derived from [Melting Pot 2.0](https://github.com/google-deepmind/meltingpot/) and feature commonly studied mixed incentives.


Our [blog](https://sites.google.com/view/socialjax/home) presents more details and analysis on agents' policy and performance.

## Installation

First: Clone the repository
```bash
git clone https://github.com/cooperativex/SocialJax.git
cd SocialJax
```


Second: Environment Setup.

Option one: Using peotry
make sure you have python 3.10
  1. Install Peotry
       ```bash
       curl -sSL https://install.python-poetry.org | python3 -
       export PATH="$HOME/.local/bin:$PATH"
       ```

  2. Install requirements     
       ```bash
       poetry install --no-root
       poetry run pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```
  3. Run code
       ```bash
       poetry run python algothrims/IPPO/ippo_cnn_coins.py 
       ```

Option two: requirements.txt
  1. Conda
       ```bash
       conda create -n SocialJax python=3.10
       conda activate SocialJax
       ```

  2. Install requirements
       ```bash
       pip install -r requirements.txt
       pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```bash
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```

  3. Run code
       ```bash
       python algothrims/IPPO/ippo_cnn_coins.py 
       ```

## Environments

We introduce the environments and use Schelling diagrams to demonstrate whether the environments are social dilemmas. 

| Environment                | Description                                                                                      | Schelling Diagrams Proof |
|----------------------------|--------------------------------------------------------------------------------------------------|:------------------------:|
| Coins                      | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coins)         |&check;                   |
| Commons Harvest: Open      | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest)|&check;                   |
| Commons Harvest: Closed    | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/common_harvest)|&check;                   |
| Clean Up                   | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/cleanup)       |&check;                   |
| Territory                  | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/territory)     |&cross;                   |
| Coop Mining                | [Link](https://github.com/cooperativex/SocialJax/tree/main/socialjax/environments/coop_mining)   |&check;                   |

#### Important Notes:
- *Due to algorithmic limitations, agents may not always learn the optimal actions. As a result, Schelling diagrams can prove that the environment is social dilemmas, but they cannot definitively prove that the environment is not social dilemmas.*

## Quick Start

SocialJax interfaces follow [JaxMARL](https://github.com/FLAIROx/JaxMARL/) which takes inspiration from the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [Gymnax](https://github.com/RobertTLange/gymnax).

### Make an Environment
You can create an environment using the ```make``` function:
```python
import jax
import socialjax

env = make('clean_up')
```

### Example

Find more fixed policy [examples](https://github.com/cooperativex/SocialJax/tree/main/fixed_policy).

```python
import jax
import socialjax
from socialjax import make

num_agents = 7
env = make('clean_up', num_agents=num_agents)
rng = jax.random.PRNGKey(259)
rng, _rng = jax.random.split(rng)

for t in range(100):
     rng, *rngs = jax.random.split(rng, num_agents+1)
     actions = [jax.random.choice(
          rngs[a],
          a=env.action_space(0).n,
          p=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
     ) for a in range(num_agents)]

     obs, state, reward, done, info = env.step_env(
          rng, old_state, [a for a in actions]
            )
```


## Citation
```
@misc{guo2025socialjaxevaluationsuitemultiagent,
      title={SocialJax: An Evaluation Suite for Multi-agent Reinforcement Learning in Sequential Social Dilemmas}, 
      author={Zihao Guo and Richard Willis and Shuqing Shi and Tristan Tomilin and Joel Z. Leibo and Yali Du},
      year={2025},
      eprint={2503.14576},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.14576}, 
}
```
## See Also

