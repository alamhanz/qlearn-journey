import sys
sys.path.insert(1,'../../src/')
from tools.rltools import print_frames
import gym
import os

env = gym.make("MountainCar-v0",render_mode = 'human').env
# env.render()
env.s = 328  # set environment to illustration's state

epochs = 0
frames = [] # for animation
state = env.reset() # state start anywhere
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, outbound, info = env.step(action)
    print(state, reward, done)

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

print("\nTimesteps taken (or reward for cartpole case): {}".format(epochs))
