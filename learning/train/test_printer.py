import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'train'))

import environments
import gym
import numpy as np

env = gym.make('FlexPrinterEnv-v0')
env.set_meshid(14)

env.reset()

idx = 0
while True:
    action = np.array([0.8, 0.0])
    obs, reward, done, info = env.step(action)
    
    env.render_preview()
    if done:
        break
    idx += 1
