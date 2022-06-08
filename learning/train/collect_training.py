import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'train'))

import time
from collections import deque

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
gym.logger.set_level(40)
import argparse
import math

import environments
from utils import *

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import cv2

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--model-path', type = str, required = True)

args = parser.parse_args()

model_dir = Path(args.model_path).parent.parent
model_args = parse_model_args(os.path.join(model_dir, 'args.txt'))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
device = torch.device('cpu')

env = gym.make(model_args.env_name)

checkpoint = 0

for i in range(135):
    file_path = args.model_path+str(checkpoint)+'.pt'
    if not os.path.isfile(file_path):
        print_error('Model file does not exist')
        continue

    actor_critic, ob_rms = torch.load(file_path)
    actor_critic.to(device)
    actor_critic.eval()

    with torch.no_grad():
        env.set_meshid(i)
        ob = env.reset()
        cv2.imwrite('tmp/target'+str(i)+'.png', env.target*255)
        env.plt = 0
        done = False
        total_reward = 0.
        sts = 0
        while not done:
            ob = np.transpose(ob, (2, 0, 1))
            if ob_rms is not None:
                ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10., 10.)
            ob = torch.Tensor(ob).unsqueeze(0).to(device)
            _, action, _, _ = actor_critic.act(ob, None, None, deterministic = True)

            action_np = action.squeeze(0).detach().cpu().numpy()
            action_np = np.clip(action_np, -1., 1.)
            v_xy = (action_np[0]+1.0)/2.0
            velocity = (1-v_xy)*env.speed_limit[0] + v_xy*env.speed_limit[1]

            ob, reward, done, info = env.step(action.squeeze(0).detach().cpu().numpy())
            a = action.squeeze(0).detach().cpu().numpy()
            sts += 1
            env.render_preview()
            step_params = info['step_params']
            total_reward += reward
    env.render_preview()

    step_param = step_params[-1]
