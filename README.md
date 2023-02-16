# MLOps with Weights & Biases


![wandb_logo](https://user-images.githubusercontent.com/20300069/219328542-b5ff6cbe-5082-4e0e-bf3d-96db3b30b90e.png)


This toy-keras-project (MNIST classification with CNNs) showcases [Weights & Biases](https://wandb.ai/site) capabilities to easily achieve some level of MLOps in your ML projects. Wandb UI is a powerful tool and you're going to go back to it while iterating over the typical ML lifecycle of your project, as described below.

The project consists of 3 python scripts living inside the `app/pipelines` folder: `data_prep.py`, `train.py` and `evaluate.py` which allows you to:
- prepare your training, validation and test datasets
- train a baseline model
- perform hyperparameters tuning (directly in wandb using sweeps)
- retrain your candidate model
- evaluate your final model before moving it to production

**These steps should be executed in this order, at least the first time.**

Thanks to wandb you **and your team** have:
- version control and lineage of your datasets and models
- experiment tracking by logging almost everything your heart desires
- a very powerful, framework agnostic and easy-to-use tool for hyperparameters tuning (goodbye keras-tuner my old friend)
- tables on data and run results which allow you to:
  - study input distributions and avoid data leakage
  - perform error analysis
- the capability to write and share markdown reports directly linked to your tables or other artifacts you logged to wandb

## Requirements & Installation
- You need to have a [wandb account](https://wandb.ai/site/pricing). It's free for personal use and you have unlimited tracking and 100GB storage for artifacts.
- clone the repository and [create a virtual environment from the given yaml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) with the required packages :
```
conda env create -f environment.yaml
```
- activate the environment:
```
conda activate MLOps-with-wandb
```
- Create a `.env` file in the `app/` folder (which is git-ignored) containing your [wandb API key](https://wandb.ai/authorize) and other information depending on the script you're using (more information later). The file would look like:
```
WANDB_API_KEY="********************************"
WANDB_ENTITY="fratambot"
WANDB_PROJECT="MNIST"
WANDB_MODEL_RETRAIN="CNN_MNIST:v0"
WANDB_MODEL_EVAL="CNN_MNIST:v1"
```

## Data Preparation & Baseline model

To prepare your datasets you can run the `app/pipelines/data-prep.py` script: it will sample 10% of the keras MNIST dataset and split it into 70 % training / 20% validation / 10% test with stratification over the classes.

You can change these default values by passing them. For more info consult:
```
python app/pipelines/data-prep.py --help
```
- **inputs: None**
- **outputs:**
  - media: an histogram showing the labels distribnution for the 3 datasets (rescaled wrt the relative split proportion)
  - table: a wandb table
  - artifact: npz

This is waht you'll find on wandb and how to interact with it through the UI:

https://user-images.githubusercontent.com/20300069/219327479-93d57c9c-4759-4b2a-b0b3-02e963b58619.mov



