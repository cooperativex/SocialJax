<h1 align="center">SocialJax</h1>

*A suite of sequential social dilemma environments for multi-agent reinforcement learning in JAX*



# Installation

make sure you have python 3.10.5
---

Peotry:

```
curl -sSL https://install.python-poetry.org | python3 -
```
```
export PATH="$HOME/.local/bin:$PATH"
```
```
poetry --version
```
```
poetry install --no-root
```
```
poetry run pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
```
export PYTHONPATH=./socialjax:$PYTHONPATH
```
```
poetry run python algothrims/IPPO/ippo_cnn_coins.py 
```

---

Requirements.txt
```
conda create -n SocialJax python=3.10
```
```
conda activate SocialJax
```
```
pip install -r requirements.txt
```
```
pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
```
export PYTHONPATH=./socialjax:$PYTHONPATH
```
```
python algothrims/IPPO/ippo_cnn_coins.py 
```
