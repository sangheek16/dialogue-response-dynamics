# About
This repo contains datasets and code for COLING 2022 paper _“No, they did not”: Dialogue response dynamics in pre-trained language
models_, by Sanghee J. Kim, Lang Yu and Allyson Ettinger.

# Repo Structures

- `src/` contains source code to generate dataset and duplicate experiment results in the paper
- `datasets/` contains raw data and datasets used in the paper

# Dataset
- `names.csv`, `np.csv` and `vp.csv` are raw datasets that can be used as inputs for `input_generator.py` to generate an input .csv file -- this will have the same format as `used_items.csv`
- `used_items.csv` contains the items that were used in this paper

# Code
- `evaluate.py` is built on Huggingface's [transformer](https://github.com/huggingface/transformers) and requires [minicons](https://github.com/kanishkamisra/minicons)
- `input_generator.py` randomly samples from `names.csv`, `np.csv` and `vp.csv` and generates input items that can be used for model evaluation
- for replication of the results in this paper, use `used_items.csv` under `src/` instead of generating new items
- `evaluate.py` produces model output probabilities / conditional log-probabilities / pseudo-log-likelihoods of causal language model and masked language models. It produces output for *header selection* (section 5.1), *rejection* (section 5.2), *conjunction* (section 5.3) and *ellipsis* (section 6) tasks.
- `probe.py` contains code for generating probing results (section 5.4)

# Usage
To run `evaluate.py`, update config at the beginning of the code (e.g., input path, output path, etc.). Also update the name of model (`model_name` variable) and type of task (`task` variable) you wish to explore, at the end of the code. Then run `python3 evaluate.py`.