# MLOps with Weights & Biases


<p align="center">
  <img width="800" src="/README/wandb_logo.png">
</p>

This toy-keras-project (MNIST classification with CNNs) showcases [Weights & Biases](https://wandb.ai/site) capabilities to easily achieve some level of MLOps in your ML projects. This README both explains how to use the code in the repo and as a "tutorial" on how to interact with wandb UI while iterating over the typical ML lifecycle of your project.

The project consists of 3 python scripts living inside the `app/pipelines` folder: `data_prep.py`, `train.py` and `evaluate.py` which allows you to:
- prepare your training, validation and test datasets
- train a baseline model
- perform hyperparameters tuning (directly in wandb)
- retrain your candidate model
- evaluate your final model before moving it to production

Thanks to wandb you **and your team** have:
- version control and lineage of your datasets and models
- experiment tracking by logging almost everything your heart desires
- a very powerful, framework agnostic and easy-to-use tool for hyperparameters tuning (goodbye keras-tuner my old friend)
- tables which allow you to quickly perform EDA and more importantly error analysis
- the capability to write and share markdown reports directly linked to your tables or other artifacts you logged

## Requirements


## Install
