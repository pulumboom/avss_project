## About

This repository contains an implementation of several architectures and training configs to solve BSS (blind speech separation) problem.

Our best weights for ConvTasNet available: [here](https://disk.yandex.ru/d/oZw-Vy3YpZRQ-Q)

Report on the completed work: [here](https://123.ru)

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME convtasnet.yaml
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
