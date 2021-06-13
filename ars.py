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

# normalization (paper calls it normalization but actually it is standardization)
class Normalizer():
    # constructor
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs) # counts the number of states
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs) # for variance calc
        self.var = np.zeros(num_inputs) # variance

    def observe(self, x):
        self.n += 1. # counter
        self.last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - self.last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 0.001)
    
    def normalize(self, inputs):
        obs_mean = self.mean 
        obs_std = np.sqrt(self.var)
        return(inputs - obs_mean) / obs_std


# AI class
class Policy():
    # constructor
    def __init__(self, input_size, output_size):
        # weights matrix
        self.theta = np.zeros((output_size, input_size))
    
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == 'positive':
            return (self.theta + hp.noise * delta).dot(input)
        else:
            return (self.theta - hp.noise * delta).dot(input)
        
    def sample_deltas(self):
        # function to return noise matrices
        return [np.random.randn(*self.theta.shape) for _ in range(hp.num_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        # rollout is a list of positive/negative rewards and noise
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg)*d
        # sigma_r is rewards std.
        self.theta += hp.learning_rate / (hp.best_directions * sigma_r) * step
        
        
        


        
        
        