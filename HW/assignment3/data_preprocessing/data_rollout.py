import csv
import numpy as np
import gym
from imageio import imwrite
from random import choice, uniform


def get_action(state):
    # eps = perturbation
    eps1 = uniform(-0.2, 0.2)   # left/right (+-1)
    eps2 = uniform(0, 0.1)      # speed up
    eps3 = uniform(0, 0.05)     # break
    eps = [eps1, eps2, eps3]
    act = choice([
        [1,  0, 0],  # all to left
        [-1, 0, 0],  # all to right
        [0,  1, 0]   # speed up
    ])
    return np.add(act, eps)



env = gym.make('CarRacing-v2', continuous=True, render_mode='human', domain_randomize=True)
episodes = 100
steps = 200

file = open('../rollouts/data_memory_cell.csv', 'w')
writer = csv.writer(file)
writer.writerow(['render_path', 'action'])

for eps in range(episodes):

    state = env.reset()
    env.render()
    for t in range(steps):
        action = get_action(state)
        state, _, _, _, _ = env.step(action)
        if t % 5 == 0:
            i = ('000' + str(t//5))[-3:]
            file_name = f'car_{eps}_{i}.jpg'
            #save img
            imwrite('../rollouts/CarRacing_random/'+file_name, state)
            #save img_path + action
            writer.writerow([file_name, action])
    print(f"Episode[{eps+1}/{episodes}]")

file.close()

