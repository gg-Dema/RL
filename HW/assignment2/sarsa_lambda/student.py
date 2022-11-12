import numpy as np
import random


def epsilon_greedy_action(env, Q, state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore action space
    action = Q[state, :].argmax()
    return action



def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_=0.9, initial_epsilon=1.0, n_episodes=50000):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    ############# keep this shape for the Q!
    Q = np.random.rand(env.observation_space.n, env.action_space.n)

    # init epsilon
    epsilon = initial_epsilon

    received_first_reward = False

    print("TRAINING STARTED")
    print("...")
    total_reward = 0
    for ep in range(n_episodes):

        eligibility = np.zeros((env.observation_space.n, env.action_space.n))

        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False

        while not done:
            ############## simulate the action
            next_state, reward, done, info, _ = env.step(action)
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)

            error = reward + (1-done)*gamma*Q[next_state, next_action] - Q[state, action]

            # update eligibility [frequency]
            eligibility[state, action] += 1


            ############## update q table and eligibility
            Q = Q + alpha*error*eligibility
            eligibility = eligibility*gamma*lambda_

            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)

            # update current state and action
            state = next_state
            action = next_action

        # update current epsilon
        total_reward += reward

        if received_first_reward and ep > 50:

            epsilon = 0.99 * epsilon

    print(f"total_reward {total_reward}")
    print(f" % of success : {total_reward / n_episodes}")
    print("TRAINING FINISHED")
    return Q
