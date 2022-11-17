import random

import numpy as np
import gym
import time
from gym import spaces
import os


def value_iteration(env):
    gamma = 0.99
    iters = 100

    # initialize values
    values = np.zeros((env.num_states))
    best_actions = np.zeros((env.num_states), dtype=int)
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    for i in range(iters):
        v_old = values.copy()
        for s in range(env.num_states):
            state = STATES[s]

            if (state == env.end_state).all() or i >= env.max_steps:
                # if we reach the termination condition, we cannot perform any action
                continue

            max_va = -np.inf
            best_a = 0
            for a in range(env.num_actions):
                next_state_prob = env.transition_probabilities(state, a).flatten()

                va = (next_state_prob * (REWARDS + gamma * v_old)).sum()

                if va > max_va:
                    max_va = va
                    best_a = a
            values[s] = max_va
            best_actions[s] = best_a

    return best_actions.reshape((env.height, env.width))


'''
reference : https://gist.github.com/tuxdna/7e29dd37300e308a80fc1559c343c545
et. al = Andrea Giuseppe Di Francesco
'''


def policy_iteration(env, gamma=0.99, iters=100):

    values = np.zeros((env.num_states))
    policy = np.zeros((env.num_states), dtype=np.uint8)
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()

    # POPULATE STATES
    i = 0
    for row in range(env.height):
        for col in range(env.width):
            state = np.array([row, col], dtype=np.uint8)
            STATES[i] = state
            i += 1

    # POLICY ITERATION
    i = 0
    changed = True

    while(changed and i<iters):
        i += 1
        changed = False

        # POLICY EVALUATION

        v_old = values.copy()
        for s in range(env.num_states):
            state = STATES[s]
            next_state_prob = env.transition_probabilities(state, policy[s]).flatten()
            values[s] = (next_state_prob * (REWARDS[s] + gamma * v_old)).sum()

        # POLICY IMPROVMENT
        for s in range(env.num_states):

            state = STATES[s]
            v_old = values.copy()
            max_v = -np.inf

            for a in range(env.num_actions):
                next_state_prob = env.transition_probabilities(state, a).flatten()
                v_greedy = (next_state_prob * (REWARDS[s] + gamma * v_old)).sum()

                if v_greedy > max_v:
                    policy[s] = a
                    max_v = v_greedy
                    changed = True

    return policy.reshape((env.height, env.width))
