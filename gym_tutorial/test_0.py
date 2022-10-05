import gym
import matplot.pyplot as plt
'''
 evn class
 contains :
       - observation space (STRUCT)
               | basic form: screen
       - action space
               |
       FUNCT(): 
            reset --> init state + [
                    return observation 
                    reward
                    done
                    ]
            step  --> action in input, applies to env
'''

env = gym.make("mountainCar-v0")
obs = env.reset()
random_action = env.action_space.sample()
new_obs, reward, done, info = env.step(random_action)

