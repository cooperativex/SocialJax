<h1 align="center">SocialJax</h1>


*A suite of sequential social dilemma environments for multi-agent reinforcement learning in JAX*

<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="/docs/images/step_150_reward_common_coins.gif" alt="coins_common" width="10%">
            <img src="/docs/images/step_150_reward_common_harvestopen.gif" alt="harvest_open_common" width="18.5%">
            <img src="/docs/images/step_150_reward_common_closed.gif" alt="harvest_closed_common" width="18.5%">
            <img src="/docs/images/step_150_reward_common_cleanup.gif" alt="clean_up_common" width="19.8%">
            <img src="/docs/images/step_250_reward_common_coop_mining.gif" alt="coop_mining_common" width="20%">
        </div>
        <div class="row" align="centre">
            <img src="/docs/images/step_150_reward_individual_coins.gif" alt="coins_individual" width="10%">
            <img src="/docs/images/step_150_reward_individual_harvestopen.gif" alt="harvest_open_individual" width="20%">
            <img src="/docs/images/step_150_reward_individual_closed.gif" alt="harvest_closed_individual" width="20%">
            <img src="/docs/images/step_150_reward_individual_cleanup.gif" alt="clean_up_individual" width="20%">
            <img src="/docs/images/step_250_reward_individual_coop_mining.gif" alt="coop_mining_individual" width="20%">
        </div>
    </div>
</div>


Our [blog](https://sites.google.com/view/socialjax/home) presents more details and analysis on agents' policy and performance.

# Installation

First: Clone the repository
```
git clone https://github.com/cooperativex/SocialJax.git
```


Second: Environment Setup.

Option one: Using peotry
make sure you have python 3.10
  1. Install Peotry
       ```
       curl -sSL https://install.python-poetry.org | python3 -
       ```
       ```
       export PATH="$HOME/.local/bin:$PATH"
       ```
       ```
       poetry --version
       ```
    
  2. Install requirements     
       ```
       poetry install --no-root
       ```
       ```
       poetry run pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```
  3. Run code
       ```
       poetry run python algothrims/IPPO/ippo_cnn_coins.py 
       ```

Option two: requirements.txt
  1. Create conda environment
       ```
       conda create -n SocialJax python=3.10
       ```
       ```
       conda activate SocialJax
       ```

  2. Install requirements
       ```
       pip install -r requirements.txt
       ```
       ```
       pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
       ```
       ```
       export PYTHONPATH=./socialjax:$PYTHONPATH
       ```

  3. Run code
       ```
       python algothrims/IPPO/ippo_cnn_coins.py 
       ```
