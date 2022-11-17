import random
import numpy as np
import gym
import time
from gym import spaces
import os
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:

    def __init__(self, env, n_components=100, sigma=1, constant_sigma=True):
        self.env = env
        self.n_components = n_components
        if constant_sigma:
            self.sigma = np.full(n_components, sigma)
        else: self.set_non_constant_sigma()
        center_shape = (n_components, env.observation_space.shape[0])
        self.min = env.observation_space.low
        self.max = env.observation_space.high
        self.center = np.random.uniform(self.min, self.max, center_shape)

    def encode(self, state,):
        feats = np.zeros(self.n_components)
        # normalize data [else overflow]
        state_std = (state - self.min) / (self.max - self.min)
        scaled_state = state_std * (self.max - self.min) + self.min

        for i in range(self.n_components):
            arg = (scaled_state - self.center[i])
            arg = -np.linalg.norm(arg) / (2*self.sigma[i])**2
            feats[i] = arg
        return feats

    @property
    def size(self):
        return self.n_components

    def set_non_constant_sigma(self):
        pass




class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder,
        alpha=0.1, alpha_decay=0.99,
        gamma=0.999, epsilon=0.3, epsilon_decay=0.99, lambda_=0.9):
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1, 1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)

       # FORWARD_VERS
        action_prime = self.epsilon_greedy(s_prime)
        td_error = reward + (1-done)*self.gamma*self.Q(s_prime_feats)[action_prime] - self.Q(s_feats)[action]
        delta_w = td_error*s_feats
        self.weights[action] += self.alpha*delta_w

        ''' BACK-VERS
        error = reward + (1-done)*self.gamma*self.Q(s_prime_feats).max() - self.Q(s_feats)[action]

        self.traces += s_feats
        self.weights += self.alpha*error*self.traces
        self.traces = self.traces*self.gamma*self.lambda_
        '''
    def update_alpha_epsilon(self): # modify
        self.epsilon = max(0.1, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay

        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None):  # modify
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return self.env.action_space.sample()
        return self.policy(state)

        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
