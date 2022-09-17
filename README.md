# About
This repo contains datasets and code for COLING 2022 paper _“No, they did not”: Dialogue response dynamics in pre-trained language
models_, by Sanghee J. Kim, Lang Yu and Allyson Ettinger.

# Repo Structures

- `src/` contains source code to generate dataset and duplicate experiment results in the paper
- `datasets/` contains raw data and datasets used in the paper

# Dataset
- `names.csv`, `np.csv` and `vp.csv` are raw datasets we used to generate analysis datasets. They can be used as the input to `input_generator.py` to generate files in the same format as `used_items.csv`.
- `used_items.csv` are items we used in the paper.

# Code
- code is built on Huggingface's [transformer](https://github.com/huggingface/transformers)
- evaluation requires [minicons](https://github.com/kanishkamisra/minicons)
- `input_generator.py` randomly samples from names.csv, np.csv and vp.csv to generate analysis dataset we used in the paper. To replicate the result with exact input, use "used_items.csv" included in the folder instead.
- `evaluate.py` contains analysis for header selection (section 5.1), rejection (section 5.2), conjunction (section 5.3) and ellipsis (section 6).
- `probe.py` contains code for generating probing results (section 5.4)

# Usage
To run, update config part at the begining of the file (model, model type, input output path etc.), then run `python3 file_to_run.py`.
