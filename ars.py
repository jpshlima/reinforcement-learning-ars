# -*- coding: utf-8 -*-

# ARS - Augmented Random Search, a Reinforcement Learning algorithm
# from paper 2018 "paper name"
# by Hoika et al.


"""
@author: jpshlima
"""

# imports
import os
import numpy as np

# hyperparameters
class Hp():
    # constructor
    def __init__(self):
        self.num_steps = 1000
        self.episode_lenght = 1000
        self.learning_rate = 0.02
        self.num_directions = 16 # number of noise matrices
        self.num_best_directions = 16 # number of used matrices, must be <= num_directions
        assert self.num_best_directions <= self.num_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = ''

# normalization        