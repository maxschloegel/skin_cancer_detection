# DL for Skin Cancer Detection

This repository contains the code to train a Neural Network to diagnose different pigment skin lesions into benign and malignentdifferentiate between benign and malign lesions.


## Overview

1. Dataset
2. Training
3. MLflow Tracking Server

## Dataset

The dataset used for this project is the HAM10000 dataset.
It contains 10000 labeled images of skin lesions with 7 different classes. 
For more information and downloading the dataset see the [HAM10000-Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

> Tschandl, P. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions [dataset]. Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T 


## Training

For training you first need to set up the virtual environment and then start the training process.
At the current stage training will automatically create an sqlite-DB in this project's root directory which will store all models and metrics.
If at a later stage you want to have an online storeg (e.g. Amazon S3 instance) more instructions will follow soon.

### 1. Set up the Virtual Environment

To set up the virtual environment use your favourite tool. In my case, I use python `virtualenv` to set it up.
First create it and then activate it, so we can intall the dependencies afterwards.

```shell
virtualenv venv --python=3.10

. venv/bin/activate
```

Then install the dependencies:

```shell
pip install -r requirements.txt
```


### 2. Start Training

To start the training process run

```shell
python train.py
```


## MLFlow Tracking Server

The Storage for MLFlow model tracking is defined in the training script:
```python
tracking_uri = 'sqlite:///mydb.sqlite'
```

The Tracking Server can now be started via:
```shell
mlflow ui --backend-store-uri sqlite:///mydb.sqlite
```

In the UI you can look at the different training runs and register models to the model registry.

### Setting up MLflow Tracking Server online

Here we will describe in detail how to set up an online MLflow Tracking Server and connect it to an online available model registry.


## Coming Soon

The following features and additionas will come soon. For each:
- Keep README up to date for each of the To-Dos

- [x] Create Github repository with first commit
- [ ] Enable training parameter settings with config-file or hydra.
- [ ] Use [DVC](https://dvc.org/) for data version control of training data
  - [ ] First set it up locally
  - [ ] Make it flexible enough to be switch to AWS S3 later on
- [ ] Set up CT (continuous training)
  - [ ] Create github runner with GPU (firts locally then later with AWS using [`CML`](https://cml.dev/))
  - [ ] Create github-actions
- [ ] Implement Tests + CI/CD
  - [ ] Write tests
  - [ ] Add CI to github-actions
- [ ] Refactor training script:
  - [ ] Create own model.py-file for lightning model
  - [ ] Outsource all configs to external config-file
- [ ] Deploy model with RestAPI endpoint via AWS
  - [ ] Either FastAPI or MLflow's own deployment method
- [ ] Use [`Evidently`](https://www.evidentlyai.com/) for performance tracking
- [ ] Deploy MLflow tracking server via AWS
- [ ] Set up cost monitoring of AWS instances
- [ ] Understand `checkpoint`-directory
