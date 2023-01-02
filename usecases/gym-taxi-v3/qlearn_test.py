import sys
sys.path.insert(1,'../../src/')
from tools.rltools import print_frames
from IPython.display import clear_output
import gym
import numpy as np
from pathlib import Path

direct = Path().absolute()
curr_path = direct.__str__().split('\\')[-1]
PATH_MODEL = '../../artifacts/'+curr_path+'/'

env = gym.make("Taxi-v3",render_mode = 'ansi').env
env.s = 328

# get qlearn
q_table = np.load(PATH_MODEL+'qlearn_v3.npy') 

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.4

# For plotting metrics
all_epochs = []
all_penalties = []
frames = []
state = env.reset()[0] 
epochs, penalties, reward, = 0, 0, 0
done = False

while not done:
    action = np.argmax(q_table[state])
    next_state, reward, done, outbound, info = env.step(action)
    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    state = next_state
    epochs += 1



print_frames(frames)

print("Test finished.\n")

print("\nTimesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))