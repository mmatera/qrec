# Introduction

In the folder Generic_cal there is a python file Experiment.py. This file has the instructions on how to run an experiment with no model. The model aware agent is used in the folder Model_Semi_aware.

All the data from the experiment is going to be saved in a dictionary "details". The dictionary has the atributes
```
"index" -> Experiment_index to save the data from the run.

"alpha" -> Intensity used in the experiment.

"ep" -> epsilon,

"betas" -> discrete values of beta accesible by the experiment,

"experience" -> Each element contains the beta used, the observation, the maximum likelihood election and the reward obtained.

"Ps_greedy", -> No estoy seguro, tiene que ver con la mejor respuesta posible.

"seed" -> Seed used in the experiment.

"mean_rewards" -> list containing all the mean rewards during the experiment
```
To start we have to define this dictionary or load it from a .pickle file. After this we need to define the amount of experiments and the hiperparameters related to the enhanced Q-learning algorithm. Finally we have to run one of the functions Experiment_run or Model_experiment. Both functions return the details dictionary completed during the experiment with which we can plot all the information.

# How the code works

The full skeleton of the script is inside the folder "qrec" in the files Stage_run.py and utils.py.

## Stage_run.py

The objective of this script is to run the experiment and following the hiperparameters save the values of interest and make the decisions.

## Utils.py
This is the backbone of the code, it is responsable for making the decisions given the observations and the parameters, returns what it considers random and makes the decisions for the Q-learning.
