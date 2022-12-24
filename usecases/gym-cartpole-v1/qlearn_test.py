import random
from IPython.display import clear_output
import gym
import numpy as np
from pathlib import Path
import time
import argparse

def get_discrete_state(state,version):
    if version in ('v0','v3'):
        discrete_state = state/np_array_win_size + np.array([15,10,1,10])
    else:
        addon = np.array([6,14,35,13])
        discrete_state = state/np_array_win_size + addon
        
        for i in [0,1,2,3]:
            if discrete_state[i]>=Observation[i]:
                discrete_state[i] = Observation[i] - 1
            elif discrete_state[i]<0:
                discrete_state[i] = 0
    return tuple(discrete_state.astype(np.int64))

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()

VERSION = args.version

env = gym.make("CartPole-v1",render_mode = 'human').env

# Hyperparameters
if VERSION in ('v0','v3'):
    Observation = [30, 30, 50, 50] #2.25 mil
    np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
else:
    Observation =  [11, 29, 70, 25] #558k
    np_array_win_size = np.array([0.3, 0.2, 0.008, 0.3])

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
    discrete_state = get_discrete_state(state,VERSION)

    episode_reward, epoch = 0, 0
    done = False
    while not done:
        
        action = np.argmax(q_table[discrete_state]) # Exploit learned values

        new_state, reward, done, _ ,_  = env.step(action)
        new_discrete_state = get_discrete_state(new_state,VERSION)
        
        episode_reward += reward
        discrete_state = new_discrete_state
        epoch += 1

        if episode_reward>=3000:
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
