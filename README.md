# DL for Skin Cancer Detection

This repository contains the code to train a Neural Network to differentiate between benign and malign lesions.


## Overview

1. Dataset
2. Training
3. MLflow Tracking Server
4. FastAPI
5. Self-hosted Github Runner
6. Coming Soon

## 1. Dataset

The dataset used for this project is the HAM10000 dataset.
It contains 10000 labeled images of skin lesions with 7 different classes. 
For more information and downloading the dataset see the [HAM10000-Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

> Tschandl, P. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions [dataset]. Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T 


## 2. Training

For training you first need to set up the virtual environment and then start the training process.
At the current stage training will automatically create an sqlite-DB in this project's root directory which will store all models and metrics.
If at a later stage you want to have an online storeg (e.g. Amazon S3 instance) more instructions will follow soon.

### 2.1. Set up the Virtual Environment

To set up the virtual environment use your favourite tool. In my case, I use python `virtualenv` to set it up.
First create it and then activate it, so we can intall the dependencies afterwards.

```shell
virtualenv venv --python=3.10

. venv/bin/activate
```

Then install the dependencies:

```shell
pip install .
```
> Note: [`poetry`](https://python-poetry.org/) is used in this repository to create the pyproject.toml file

### 2.2. Start Training

To start the training process run

```shell
python src/skin_cancer_detection/train.py
```


## 3. MLFlow Tracking Server

The Storage for MLFlow model tracking is defined in the training script:
```python
tracking_uri = 'sqlite:///mydb.sqlite'
```

The Tracking Server can now be started via:
```shell
mlflow ui --backend-store-uri sqlite:///mydb.sqlite
```

In the UI you can look at the different training runs and register models to the model registry.

### 3.1. Setting up MLflow Tracking Server online

Here we will describe in detail how to set up an online MLflow Tracking Server and connect it to an online available model registry.

## 4. FastAPI

In the end the model will be deployed via FastAPI (see [`src/app/main.py`](https://github.com/maxschloegel/skin_cancer_detection/blob/main/src/app/main.py))
To run the app you need to have `uvicorn` installed and run:

```shell
uvicorn src.app.main:app --reload
```
You can set the specific port by adding the argument `--port 8001`, see [uvicorn's website](https://www.uvicorn.org/settings/) for more information.


## 5. Self-Hosted Github-Runner

CML allows you to run your own github-runner, i.e. if you need GPUs for computations but don't have access to AWS et al.
On their [website](https://cml.dev/doc/self-hosted-runners) they explain in detail how to set that up. In short, you can use their docker image with every necessary package pre-installed `docker://iterativeai/cml:0-dvc2-base1-gpu`.
This container has all the Python, CUDA, cml dependenies installed.
Then you can start an interactive session via
```shell
docker run -it iterativeai/cml:0-dvc2-base1-gpu bash
```
To my experience it is important here to add `bash` at the end. I don't understand 100% why, but it seems that when restarting the container (after it has been stopped) `bash` keep the container from exiting immediatly. I assume that is because `bash` is waiting for input from the user and this keeps the container alive.
When you want to use your GPU in the runner, you need to add this to the `run`-command. You need to make sure that docker is able to use the GPU first.
See [here](https://stackoverflow.com/a/58432877) to install `nvidia-container-toolkit`. After restarting docker you then can enable GPUs. 
In addition to GPU usage we also need access to a larger shared memory (shm). To deal with this issue, you can pass `--shm_size=8gb` (or less, 8gb seem to be plenty).
```shell
docker run -it --gpus all --shm-size=8gb iterativeai/cml:0-dvc2-base1-gpu bash
```
To see if the container has GPU access, run `nvidia-smi` inside the container.

This container can be stopped and started again. If you want to access the containers bash, you need to start it in interactive mode:
```shell
docker start -i <container_name>
```
Here you could for example test with `nvidia-smi` if you actually have access to the GPU(s).
If you cannot remember the name of your container you can see a list of all containers and their names with (it also lists their status):
```shell
docker ps -a
```

Once the session is running and you tested the GPU, you can then start the CML-runner with the following command from inside the container.
```shell
cml runner launch \
  --repo="$REPOSITORY_URL" \
  --token="$PERSONAL_ACCESS_TOKEN" \
  --labels="cml-gpu" \
  --idle-timeout="never"  # or "3min", "1h", etc..
```
Here:
- `REPOSITORY_URL` is the url of the repository you need the GPU-runner for
- `PERSONAL_ACCESS_TOKEN` is the PAT for that repository which allows you to create the runner

## 6. Coming Soon

The following features and additionas will come soon. For each:
- Keep README up to date for each of the To-Dos

- [x] Create Github repository with first commit
- [x] Enable training parameter settings with config-file or hydra.
- [ ] Implement Tests + CI/CD
  - [x] Create CI with github-actions (pytest and flake8)
  - [x] Initialize Code as Python Package
  - [ ] Write actual tests
- [ ] Use [DVC](https://dvc.org/) for data version control of training data
  - [ ] First set it up locally
  - [ ] Make it flexible enough to be switch to AWS S3 later on
- [ ] Set up CT (continuous training)
  - [ ] Create github runner with GPU (firts locally then later with AWS using [`CML`](https://cml.dev/))
  - [ ] Add training to github-actions
- [ ] Refactor training script:
  - [ ] Create own model.py-file for lightning model
  - [ ] Outsource all configs to external config-file
- [ ] Deploy MLflow tracking server via AWS
- [x] Deploy model with RestAPI endpoint via AWS
  - [x] Deploy local FastAPI using local mlflow model registry
  - [ ] Deploy FastApi-app via AWS
- [ ] Use [`Evidently`](https://www.evidentlyai.com/) for performance tracking
- [ ] Set up cost monitoring of AWS instances
- [ ] Understand `checkpoint`-directory
