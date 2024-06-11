# Bachelors Thesis

Pairing a heartrate to a video sequence of a person with a neural network

## Disclaimer
The following repo, may have missing files, but should be fully functional. The dataset used for the thesis is not publically available and has to be protected by GDPR rules, therefore all notebooks and potential revealing figures has been cencored out.

## Abstract 
It is evident from studies that there are some kind of correlation between the visual perceptive information of a person and their respective interoceptive state and that it is possible
for humans to infer these states [[1]](#1). The question beckons of what factors plays a significant role in revealing such a state for a person, but this is not easily answered, even
if humans are able to infer these states with bigger precision than chance they remain
unclear in their approach.
Doing the same task, but with an AI model instead of a person may give significant insight
in exactly what features of the human appearance may play a factor into revealing their
interoceptive state. In this project an AI is tasked with associating a heart rate with one of
two given video footage of people. The model will make use of various proven techniques
such as (2+1)D convolutional layers and projection convolutions, in order to interpret video
data and keep the model complexity down. Facing a lot of system capacity challenges and
limitations the simple designed model in contrary to expectations end up not achieving
expected results, despite multiple previous studies proving AI’s to be effective at such
tasks.

## Online Platform Integrations
### DVC
Note that the project utilizes dvc for data versioning, and the data is not publically available. Therefore the data has to be downloaded from the google drive link provided in the dvc.yaml file. The data is not publically available and requires authentication and access from DTU.

### WANDB
The provided code utilizes WANDB for keeping track of the experiments, and the code is setup to log the experiments to the WANDB platform.

### Docker
The project is setup with a dockerfile for easy deployment of the model, and the dockerfile is setup to be able to run the training script.

### Hydra
The project is setup with hydra for easy configuration of the hyperparameters, and the configuration file is located in the config folder.


## Getting started Locally
Note that the project is setup with a makefile for easy setup and deployment of the model, and the following steps will guide you through the process of setting up the project locally.

1. First setup the virtual environment with
```make create_environment```

2. Activate and enter the virtual environment

3. Then you can install the requirements with the following make command:
    ```make requirements```

    For development purposes, you can install the requirements_dev.txt file with the following command:
    ```make dev_requirements```

    WARNING: If you want to use torch with an NVIDIA GPU, you need to use one of the following command depending on your CUDA version:

    for CUDA 11.8:
        ```make requirements_gpu_cu118```

    For CUDA 12.1:
        ```make requirements_gpu_cu121```

4. Download data from google drive through DVC, note that you will need authentication since the data is not publically available. This can be done with the following command:
    ```make data```

    Beaware, that you might have to authenticate your identity with a google account.

5. Configure your hyperparameters with the train_model.yaml file in the config folder. 
    Can be found at mnist_signlanguage/config/train_model.yaml

6. Train the model on the data with:
    ```make train```




### Prerequisites

<!-- bullet list -->
- Python 3.10.11

### Docker setup

You can either make use of the make commands for the simplicty or run them manually for more control

#### Training
1. Build
    ```make docker_build_train```
2. Run
    ```make docker_run_train```

1. Build
    ```docker build -f dockerfiles/train_model.dockerfile . -t {imagename}:{imageversion}```

    ```{imagename}``` is the name of your docker build image

    ```{imageversion}``` is the version of that particular image

    example:
    ```docker build -f dockerfiles/train_model.dockerfile . -t trainer:latest```

2. Run
    ```docker run --name {containername} {imagename}:{imageversion}```

    ```{containername}``` is the name of your docker container

    ```{imagename}``` is the name of your docker build image

    ```{imageversion}``` is the version of that particular image

    example:
    ```docker run --name experiment trainer:latest```



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── bachelors_thesis  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```


## Bibliography
<a id="1">[1]</a>  Alejandro Galvez-Pol et al. “People can identify the likely owner of heartbeats by
looking at individuals’ faces”. In: Cortex 151 (2022), pp. 176–187. ISSN: 0010-9452.
DOI: https://doi.org/10.1016/j.cortex.2022.03.003. URL: https://www.sciencedirect.
com/science/article/pii/S0010945222000685.


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

