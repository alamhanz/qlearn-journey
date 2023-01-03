import random
from IPython.display import clear_output
import gym
import numpy as np
from pathlib import Path
import time
import argparse
import sys
sys.path.append('../../src/tools/')
from rltools import get_discrete_state

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()

VERSION = args.version

env = gym.make("MountainCar-v0",render_mode = 'human').env

# Hyperparameters
if VERSION in ('v0','v1','v6'):
    Observation = [75, 55]
    np_array_win_size = np.array([0.03, 0.003])
    discrete_adjust = np.array([45,26])
elif VERSION in ('v2','v3','v4','v5'):
    Observation = [35, 25]
    np_array_win_size = np.array([0.07, 0.008])
    discrete_adjust = np.array([23,12])
elif VERSION in ('v7','v8','v9','v10','v11'):
    Observation = [110, 90]
    np_array_win_size = np.array([0.02, 0.0018])
    discrete_adjust = np.array([70,43])
else:
    print('no version exist')

# Initiation
q_table = np.load(PATH_MODEL+'qlearn_'+VERSION+'.npy') 
print(q_table.shape)

# For plotting metrics
total_reward = 0 
total_episode_time = 0
all_mean_time = []
all_mean_reward = []

for i in range(1, 5):
    t0 = time.time() 
    q_table_old = q_table.copy()
    state = env.reset()[0]
    discrete_state = get_discrete_state(state,np_array_win_size,discrete_adjust)

    episode_reward, epoch = 0, 0
    done = False
    while not done:
        
        action = np.argmax(q_table[discrete_state]) # Exploit learned values

        new_state, reward, done, _ ,_  = env.step(action)
        new_discrete_state = get_discrete_state(new_state,np_array_win_size,discrete_adjust)
        
        episode_reward += reward
        discrete_state = new_discrete_state
        epoch += 1

        # print(new_state)

        if episode_reward>=250:
            done = True

    t1 = time.time()
    episode_time = t1 - t0 

    print('-'*10)
    print('EPISODE : ', i)
    print("Episode Time: " + str(episode_time))
    all_mean_time.append(episode_time)

    print("Episode Reward: " + str(episode_reward))
    total_reward = 0
    all_mean_reward.append(episode_reward)

print("Test finished.\n")
print('Average Time :',np.mean(all_mean_time))
print('Average Rewards :',np.mean(all_mean_reward))
