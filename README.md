# MLOps with Weights & Biases


<p align="center">
  <img width="800" src="/README/wandb_logo.png">
</p>

This toy-keras-project (MNIST classification with CNNs) showcases [Weights & Biases](https://wandb.ai/site) capabilities to easily achieve some level of MLOps in your ML projects. Wandb UI is a powerful tool and you're going to go back to it while iterating over the typical ML lifecycle of your project, as described below.

The project consists of 3 python scripts living inside the `app/pipelines` folder: `data_prep.py`, `train.py` and `evaluate.py` which allows you to:
- prepare your training, validation and test datasets
- train a baseline model
- perform hyperparameters tuning (directly in wandb using sweeps)
- retrain your candidate model
- evaluate your final model before moving it to production
These steps should be performed in this order, at least the first time.

Thanks to wandb you **and your team** have:
- version control and lineage of your datasets and models
- experiment tracking by logging almost everything your heart desires
- a very powerful, framework agnostic and easy-to-use tool for hyperparameters tuning (goodbye keras-tuner my old friend)
- tables on data and run results which allow you to:
  - study input distributions and avoid data leakage
  - perform error analysis
- the capability to write and share markdown reports directly linked to your tables or other artifacts you logged to wandb

## Requirements
- A [wandb account](https://wandb.ai/site/pricing). It's free for personal use and you have unlimited tracking and 100GB storage for artifacts.
- You have to create a `.env` file in the `app/` folder (which is git-ignored) containing your wandb API key and other information depending on the script you're using (more information later). It would look like:
```
WANDB_API_KEY="********************************"
WANDB_ENTITY="fratambot"
WANDB_PROJECT="MNIST"
WANDB_MODEL_RETRAIN="CNN_MNIST:v0"
WANDB_MODEL_EVAL="CNN_MNIST:v1"
```

## Install
- Clone the repository, create a virtual environment with **python 3.9** and activate it e.g. :
```
conda create --name MLOps-with-wandb python==3.9
conda activate MLOps-with-wandb
```
- Install the required packages:
```
conda ...
```
