from IPython.display import clear_output
from time import sleep
import numpy as np

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.3)

def get_discrete_state(state, win_size, adjust):
    discrete_state = state/win_size + adjust
    return tuple(discrete_state.astype(np.int64))