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
python3 inference.py -cn=inference.yaml
```
In the `inference.yaml` you can specify:
- `defaults.model` - name of model config and model itself
- `defaults.dataset.audio_path` - path to the directory **containing** `test/mix` folder with audio files
- `defaults.dataloader.batch_size` - batch size
- `inferencer.save_path` - path to directory where to save predictions (in subfolders `s1` and `s2` with `[name].wav` files).
If not absolute path is provided, they will be stored in `./data/saved/[save_path]` folder
- `inferencer.from_pretrained` - path to the file with model weights

To calculate metrics:

```bash
python3 metrics_eval.py -cn=metrics_eval.yaml
```

In the `metrics_eval.yaml` you can specify:
- `defaults.metrics` - metrics config name (e.g. `audio_metrics` - "SI-SNRi", "SDRi") In `defaults.metrics.inference._target` can be `PESQ, SDRi, SI-SNRi, STOI`.
- `pred_path` - path to the directory with predictions (in subfolders `s1` and `s2` with `[name].wav` files).
- `true_path` - path to the directory with true sources (in subfolders `s1` and `s2` with `[name].wav` files).
- `show_all` - if `True`, will show metrics for each file, otherwise will show mean value.

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
