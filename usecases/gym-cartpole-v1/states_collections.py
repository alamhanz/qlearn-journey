import sys
sys.path.insert(1,'../../src/')
from tools.rltools import print_frames
import gym
from pathlib import Path
import os
import numpy as np

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'

env = gym.make("CartPole-v1",render_mode = 'human').env
EPISODE = 2500

epochs = 0
frames = [] # for animation
state = env.reset() # state start anywhere
all_state = []

for i in range(EPISODE):
    # print(i)
    done = False
    env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, outbound, info = env.step(action)
        print(state)
        all_state.append(state)

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1
    
# print_frames(frames)

all_state = np.array(all_state)
print("\nTimesteps taken (or reward for cartpole case): {}".format(epochs))
np.save(PATH_MODEL+'states_collections.npy', all_state)
print(all_state.shape)