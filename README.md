<h1 align="center">SocialJax</h1>

*A suite of sequential social dilemma environments for multi-agent reinforcement learning in JAX*

Our [blog](https://sites.google.com/view/socialjax/home) presents more details and analysis on agents' policy and performance.

# Installation

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
