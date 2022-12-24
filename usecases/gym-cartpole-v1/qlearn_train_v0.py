import random
from IPython.display import clear_output
import gym
import numpy as np
from pathlib import Path
import time

def get_discrete_state(state):
    discrete_state = state/np_array_win_size + np.array([15,10,1,10])

    return tuple(discrete_state.astype(np.int64))

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'

env = gym.make("CartPole-v1",render_mode = 'human').env

# Hyperparameters
alpha = 0.12 ## Learning Rate
gamma = 0.95
epsilon = 0.9

Observation = [30, 30, 50, 50] #2.25 mil
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])


# Initiation
q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
# q_table = np.load(PATH_MODEL+'qlearn_v0.npy') 
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

    checkp = 80
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

        np.save(PATH_MODEL+'qlearn_v0.npy', q_table)
        epsilon = epsilon*(1-0.0005)
        print('epsilon decay : ',round(epsilon,3))

print("Training finished.\n")
np.save(PATH_MODEL+'qlearn_v0.npy', q_table) 
np.save(PATH_MODEL+'mean_time_v0.npy', np.array(all_mean_time)) 
np.save(PATH_MODEL+'mean_reward_v0.npy', np.array(all_mean_reward)) 