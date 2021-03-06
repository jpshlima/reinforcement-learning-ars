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
        # weight update method
        step = np.zeros(self.theta.shape)
        # rollout is a list of positive/negative rewards and noise
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg)*d
        # sigma_r is rewards std.
        self.theta += hp.learning_rate / (hp.best_directions * sigma_r) * step
    
    
# explores the environment 
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    # loop while for exploring the environment
    while not done and num_plays < hp.episode_lenght:
        # observes the initialized state
        normalizer.observe(state)
        # normalizes the inputs
        state = normalizer.normalize(state)
        # evaluates the inputs and provides outputs for actions
        action = policy.evaluate(state, delta, direction)
        # applies the action in the environment i.e., takes a step
        state, reward, done, _ = env.step(action)
        # balances the rewards
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    # returns sum of rewards
    return sum_rewards

# AI training
def train(env, policy, normalizer, hp):
    for step in range(hp.num_steps):
        # starts o noise (deltas) and positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.num_directions
        negative_rewards = [0] * hp.num_directions
        
        # gets positive rewards
        for k in range(hp.num_directions):
            positive_rewards[k] = explore(env, normalizer, direction = 'positive', deltas[k])
        
        # gets negative rewards
        for k in range(hp.num_directions):
            negative_rewards[k] = explore(env, normalizer, direction = 'negative', deltas[k])
        
        # gets all rewards and evaluate
        all_rewards = np.array(positive_rewards + negative_rewards) # numpy array to obtain .std method
        sigma_r = all_rewards.std()
        
        # sorting rollouts and selecting best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards))} # dict to obtain .sort method
        order = sorted(scores.keys(), key = lambda x: scores[x], reverse = True)[:hp.num_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # updating policy/weights
        policy.update(rollouts, sigma_r)
        
        # printing updated final reward
        reward_evaluation = explore(env, normalizer, policy)
        print('Step: ', step, ' Reward: ', reward_evaluation)
        

        


        
        
        