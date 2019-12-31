import autograd.numpy as np
import gym

import rl
from rl.envs.control.quanser.qube.ctrl import SwingUpCtrl

# quanser cartpole env
env = gym.make('QQube-RL-v0')
env._max_episode_steps = 10000

ctrl = SwingUpCtrl(ref_energy=0.04, energy_gain=30.0, acc_max=5.0)

obs = env.reset()
done = False
while not done:
    env.render()
    act = ctrl(obs)
    obs, _, _, _ = env.step(act)
