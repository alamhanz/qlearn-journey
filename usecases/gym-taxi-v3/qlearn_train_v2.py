import random
from IPython.display import clear_output
import gym
import numpy as np
from pathlib import Path

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'

env = gym.make("Taxi-v3",render_mode = 'ansi').env
env.s = 328

# Initiation
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.25
gamma = 0.5
epsilon = 0.6

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 50000):
    q_table_old = q_table.copy()
    state = env.reset()[0]

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, outbound, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    err = ((q_table-q_table_old).mean())*1000

    if i % 2500 == 0:
        clear_output(wait=True)
        print("Episode: ",i,", conv: ", err)
    
print("Training finished.\n")
np.save(PATH_MODEL+'qlearn_v2.npy', q_table) 