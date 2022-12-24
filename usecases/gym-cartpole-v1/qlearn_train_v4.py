import random
from IPython.display import clear_output
import gym
import numpy as np
from pathlib import Path
import time

def get_discrete_state(state):
    addon = np.array([6,14,35,13])
    discrete_state = state/np_array_win_size + addon
    
    for i in [0,1,2,3]:
        if discrete_state[i]>=Observation[i]:
            discrete_state[i] = Observation[i] - 1
        elif discrete_state[i]<0:
            discrete_state[i] = 0

    # print(state)
    # print(discrete_state)
    # print('-')

    return tuple(discrete_state.astype(np.int64))

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'
VERSION = 'v4'

env = gym.make("CartPole-v1",render_mode = 'human').env

# Hyperparameters
alpha = 0.12 ## Learning Rate
gamma = 0.95
epsilon = 0.9

Observation =  [11, 29, 70, 25] #558k
np_array_win_size = np.array([0.3, 0.2, 0.008, 0.3])


# Initiation
q_table = np.load(PATH_MODEL+'qlearn_v2.npy') 
print(q_table.shape)

# For plotting metrics
total_reward = 0 
total_episode_time = 0
all_mean_time = []
all_mean_reward = []

for i in range(1, 45000):
    t0 = time.time() 
    q_table_old = q_table.copy()
    state = env.reset()[0]
    discrete_state = get_discrete_state(state)

    episode_reward, epoch = 0, 0
    done = False
    while not done:
        
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[discrete_state]) # Exploit learned values

        new_state, reward, done, _ ,_  = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        episode_reward += reward

        ## update q
        old_value = q_table[discrete_state+(action,)]
        next_max = np.max(q_table[new_discrete_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[discrete_state+(action,)] = new_value
        
        discrete_state = new_discrete_state
        epoch += 1

    err = ((q_table-q_table_old).mean())*1000

    t1 = time.time()
    episode_time = t1 - t0
    total_episode_time += episode_time
    total_reward += episode_reward 

    checkp= 80
    if i % checkp == 0: 
        print('-'*10)
        print('EPISODE : ', i)
        mean = total_episode_time / checkp
        print("Time Average: " + str(mean))
        total_episode_time = 0
        all_mean_time.append(mean)

        mean_reward = total_reward / checkp
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0
        all_mean_reward.append(mean_reward)

        np.save(PATH_MODEL+'qlearn_'+VERSION+'.npy', q_table)
        epsilon = epsilon*(1-0.0005)
        print('epsilon decay : ',round(epsilon,3))

print("Training finished.\n")
np.save(PATH_MODEL+'qlearn_'+VERSION+'.npy', q_table) 
np.save(PATH_MODEL+'mean_time_'+VERSION+'.npy', np.array(all_mean_time)) 
np.save(PATH_MODEL+'mean_reward_'+VERSION+'.npy', np.array(all_mean_reward)) 