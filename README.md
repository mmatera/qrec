# Introduction

In the folder Generic_cal there is a python file Experiment.py. This file has the instructions on how to run an experiment.

All the data from the experiment is going to be saved in a dictionary "details". The dictionary has the atributes

"index" -> Experiment_index to save the data from the run.
"alpha" -> Intensity used in the experiment.
"ep" -> epsilon,
"betas" -> discrete values of beta accesible by the experiment,
"experience":[],
"Ps_greedy":[],
"seed" -> Seed used in the experiment.
"epsilon" -> Initial value of epsilon
"mean_rewards" -> list containing all the mean rewards during the experiment
