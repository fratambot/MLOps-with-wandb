# MLOps with Weights & Biases


![wandb_logo](https://user-images.githubusercontent.com/20300069/219328542-b5ff6cbe-5082-4e0e-bf3d-96db3b30b90e.png)


This toy-keras-project (MNIST classification with CNNs, duh) showcases [Weights & Biases](https://wandb.ai/site) capabilities to easily achieve complete MLOps maturity for experimentation and model development in your ML projects. Wandb UI is a powerful tool and you're going to use it while iterate over your model development lifecycle and important passages will be shown below in videos.

In this project you have 3 python scripts living inside the `app/pipelines` folder: `data_prep.py`, `train.py` and `evaluate.py` which allows you to:
- prepare your training, validation and test datasets
- train a baseline model
- perform hyperparameters tuning (directly in wandb using sweeps)
- retrain your candidate model
- evaluate your final model before moving it to production

**These steps should be executed in this order, at least the first time.**

Thanks to wandb, you **and your team** have:
- version control and lineage of your [datasets and models](https://docs.wandb.ai/guides/artifacts)
- [experiment tracking](https://docs.wandb.ai/guides/track) by logging almost everything your ❤ desires and thanks to the [model registry](https://docs.wandb.ai/guides/models/walkthrough) you can effortlessly transition models over lifecycle and easily hands off models to your teammates
- a very powerful, framework-agnostic and easy-to-use tool for [hyperparameters tuning](https://docs.wandb.ai/guides/sweeps) (goodbye keras-tuner my old friend)
- [tables](https://docs.wandb.ai/guides/data-vis/tables-quickstart) to log almost any type of data which allow you to:
  - study input distributions and avoid data leakage
  - perform error analysis
- the capability to write and share markdown [reports](https://docs.wandb.ai/guides/reports) directly linked to your tables or other artifacts you logged to wandb

## Requirements & Installation
- You need to have a [wandb account](https://wandb.ai/site/pricing). It's free for personal use and you have unlimited tracking and 100GB storage for artifacts.
- create a private or open project and give it a name, e.g. "MNIST"
- clone the repository and [create a virtual environment from the given yaml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). All the required packages will be installed as well :
```
conda env create -f environment.yaml
```
- activate the environment:
```
conda activate MLOps-with-wandb
```
- Create a `.env` file (which is git-ignored) in the `app/` folder containing your [wandb API key](https://wandb.ai/authorize) and other information depending on the script you're using (more information later). The file would look like:
```
WANDB_API_KEY="********************************"
WANDB_ENTITY="fratambot"
WANDB_PROJECT="MNIST"
WANDB_MODEL_RETRAIN="MNIST_CNN:v0"
WANDB_MODEL_EVAL="MNIST_CNN:v1"
```

## Data Preparation & Baseline model

To prepare your datasets you can run the `app/pipelines/data-prep.py` script: it will sample 10% of the keras MNIST dataset and split it into 70 % training / 20% validation / 10% test with stratification over the classes.

You can change these default values by passing them. For more info, consult:
```
python app/pipelines/data-prep.py --help
```
- **requirements**:
  - `WANDB_API_KEY`
  - `WANDB_PROJECT`
- **inputs:** None
- **outputs:**
  - artifact: a training / validation / test dataset collection in a file called `split-data.npz` 
  - media: an histogram showing the labels distribution for the 3 datasets (rescaled with respect to the relative split proportion)
  - table: a wandb table
  
This is what you'll find on wandb and how to interact with it through the UI:

https://user-images.githubusercontent.com/20300069/219327479-93d57c9c-4759-4b2a-b0b3-02e963b58619.mov

<br/>

To train a baseline model you can run the `app/pipelines/train.py` script: it will train for 5 epochs a CNN (defined in `app/models/models.py`) with default hyperparameters which you can change by passing them. For more info, consult:
```
python app/pipelines/train.py --help
```
- **requirements**:
  - `WANDB_API_KEY`
  - `WANDB_PROJECT`
  - `WANDB_ENTITY`
- **inputs:**
  - artifact: the latest version of `split-data.npz`
- **outputs:**
  - artifact: a trained keras model in a file called `CNN_model.h5`
  - metrics: automagically logged using wandb [keras callback](https://docs.wandb.ai/guides/integrations/keras)

This is what you'll find on wandb and how to interact with it through the UI:

https://user-images.githubusercontent.com/20300069/219338386-05e9d1de-ac1b-4c6f-9bf6-9a8f8fc3a56c.mov

<br/>

At the end you can create and [share a nice report](https://wandb.ai/fratambot/MNIST/reports/Data-preparation-Baseline-model--VmlldzozNTcwODg3) with your findings and insights for your team.


## Hyperparameters tuning

You can perform hyperparameters tuning using [wandb sweeps](https://docs.wandb.ai/guides/sweeps).
For that you will run the training script with the boolean `--tune` flag:
```
python app/pipelines/train.py --tune
```
The script will create a wandb agent performing a Bayesian search with a default value (`max_sweep=30`) of 30 runs at most over a set of hyperparameters choices defined in the `sweep.yaml` file living in the `app/pipelines` folder. You can change the [sweep configuration](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) according to your preferences and adjust `--max_sweep`, `--epochs` and other performance parameters according to your infrastructure, time and resources.

- **requirements**:
  - `WANDB_API_KEY`
  - `WANDB_PROJECT`
  - `WANDB_ENTITY`
- **inputs:**
  - artifact: the latest version of `split-data.npz`
- **outputs:**
  - artifact: a trained keras model in a file called `CNN_model.h5` for each sweep
  - metrics: automagically logged using wandb [keras callback](https://docs.wandb.ai/guides/integrations/keras)
  
The sweep visualization in wandb is probably the most impressive one allowing you and your team to easily compare models with different hyperparameters, pick up the best candidate and put it in the **model registry** to moving it further in the model development lifecycle.
Moreover you can also evaluate which parameters have the most impact looking at the parameters importance autogenerated plot and, everybody's favourite, the parallel coordinates's plot.

This is how it looks like in the UI:

https://user-images.githubusercontent.com/20300069/219373521-97141c69-196c-43af-9fc0-305682fcf137.mov

<br/>

Your observations and insights on the hyperparameters tuning step can be shared with your team in a [report](https://wandb.ai/fratambot/MNIST/reports/Hyperparams-tuning--VmlldzozNTcyMzMw) linked to your sweep runs for further investigation.

## Retraining

Once you have found a candidate model with the best combination of hyperparameters values you should probably retrain it on more epochs before evaluating it. For this task you'll use the `train.py` script with the boolean `--retrain` flag. The model and its hyperparameters values will be loaded from wandb and **you should NOT change them !**
```
python app/pipelines/train.py --retrain --epochs=15
```
- **requirements**:
  - `WANDB_API_KEY`
  - `WANDB_PROJECT`
  - `WANDB_ENTITY`
  - **`WANDB_MODEL_RETRAIN`** (you can find the "candidate" model id in the model registry. See at the end of the previous video)
- **inputs:**
  - artifact: the latest version of `split-data.npz`
  - artifact: the model you tagged as "candidate" in your model registry (e.g. `WANDB_MODEL_RETRAIN="MNIST_CNN:v0"`)
- **outputs:**
  - artifact: a retrained version of your candidate model in a file called `CNN_model.h5`
  - metrics: automagically logged using wandb [keras callback](https://docs.wandb.ai/guides/integrations/keras)

If your model doesn't overfit you can tag it as "to_evaluate" in your model registry for the next step :

https://user-images.githubusercontent.com/20300069/219394058-40926977-0d20-4c63-a957-3b4cac5d2560.mov

<br/>

## Evaluation

Now that you have retrained your candidate model for more epochs it's time to evaluate it. For this task you'll use the `app/pipelines/evaluate.py` script.
The evaluation will be performed on the validation set (again) and on the test set that your model has never seen before. The comparison  allows to estimate the degree of overfit to the validation set, if present.

For this task a table with images generated from the numpy arrays examples (`X_val` and `X_test`) wil be built. This operation might take a while depending on the size of your validation and test size (if you kept the default parameters in `data_prep.py`, there will be 1401 validation examples and 700 test examples to convert into images and it will take ~17MB of hard disk and wandb storage space). You can decide to not generate the examples images using the boolean `--no-generate_images` flag:
```
python app/pipelines/evaluate.py --no-generate-images
```
But it is **strongly suggested to run the script without flags**, grab a cup of ☕ and generate the images because they can be really useful for error analysis.

- **requirements**:
  - `WANDB_API_KEY`
  - `WANDB_PROJECT`
  - `WANDB_ENTITY`
  - **`WANDB_MODEL_EVAL`** (you can find the model id "to_evaluate" in the model registry. See at the end of the previous video)
- **inputs:**
  - artifact: the latest version of `split-data.npz`
  - artifact: the model you tagged as "to_evaluate" in your model registry (e.g. `WANDB_MODEL_RETRAIN="MNIST_CNN:v1"`)
- **outputs:**
  - metrics: loss and categorical accuracy for both the validation and test set
  - media: confusion matrices for both the validation and test sets
  - table: a 3 columns table with the MNIST image of the example, the true label and the predicted label for both the validation and test set
  
This is how it looks like in the UI:
  
https://user-images.githubusercontent.com/20300069/219407381-aa4979c2-f08b-4d95-9dcf-595a420958b5.mov

<br/>

And, as usual, you can write and share a [detailed report](https://api.wandb.ai/links/fratambot/we8lsjrj) for the production team
  
## Extra resources

If you want to learn more on Weights & Biases, here are some extra resources:

- the wandb [documentation](https://docs.wandb.ai/) which is very rich and points you to examples on github and colab
- the **free** "Effective MLOps: model development" [course](https://www.wandb.courses/courses/effective-mlops-model-development) by wandb
- the wandb [blog](https://wandb.ai/fully-connected) "Fully connected"
- the wandb [white paper](https://wandb.ai/site/holistic-mlops-whitepaper-download) "MLOps: a holistic approach"
