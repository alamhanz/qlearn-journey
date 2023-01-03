import random
from IPython.display import clear_output
import gym
import numpy as np
from pathlib import Path
import time
import sys
sys.path.append('../../src/tools/')
from rltools import get_discrete_state

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'
VERSION = 'v5'

env = gym.make("MountainCar-v0",render_mode = 'rgb_array').env
# env = gym.make("MountainCar-v0",render_mode = 'human').env

# Hyperparameters
alpha = 0.15 ## Learning Rate
gamma = 0.98
epsilon = 0.95

Observation = [35, 25]
np_array_win_size = np.array([0.07, 0.008])
discrete_adjust = np.array([23,12])

# Initiation
# q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table = np.load(PATH_MODEL+'qlearn_{}.npy'.format('v4')) 
print(q_table.shape)

# For plotting metrics
total_reward = 0
all_mean_reward = []
all_pos = []

for i in range(1, 70000):
    q_table_old = q_table.copy()
    state = env.reset()[0]
    discrete_state = get_discrete_state(state,np_array_win_size,discrete_adjust)

    episode_reward, epoch = 0, 0
    done = False
    while not done:
        
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[discrete_state]) # Exploit learned values

        new_state, reward, done, _ ,_  = env.step(action)
        new_discrete_state = get_discrete_state(new_state,np_array_win_size,discrete_adjust)
        
        episode_reward += reward

        ## update q
        old_value = q_table[discrete_state+(action,)]
        next_max = np.max(q_table[new_discrete_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[discrete_state+(action,)] = new_value
        
        discrete_state = new_discrete_state
        epoch += 1

        all_pos.append(new_state[0])

        if epoch>=180:
            done = True

    err = ((q_table-q_table_old).mean())*1000
    total_reward += episode_reward 

    checkp = 100
    if i % checkp == 0: 
        print('-'*10)
        print('EPISODE : ', i)

        mean_reward = total_reward / checkp
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0
        all_mean_reward.append(mean_reward)

        np.save(PATH_MODEL+'qlearn_{}.npy'.format(VERSION), q_table)
        epsilon = epsilon*(1-0.0002)
        print('epsilon decay : ',round(epsilon,3))

        # for mountain car
        print('Max position : {}'.format(str(round(max(all_pos),3))))
        all_pos = []

print("Training finished.\n")
np.save(PATH_MODEL+'qlearn_{}.npy'.format(VERSION), q_table)
np.save(PATH_MODEL+'mean_reward_{}.npy'.format(VERSION), np.array(all_mean_reward)) 