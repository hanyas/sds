import autograd.numpy as np
import gym

import sds
from sds.envs.quanser.cartpole.ctrl import SwingUpCtrl


# quanser cartpole env
env = gym.make('Quanser-Cartpole-v0')
env._max_episode_steps = 1000000

ctrl = SwingUpCtrl()

obs = env.reset()
for n in range(500):
    act = ctrl(obs)
    obs, _, done, _ = env.step(act)
    if done:
        break

    if np.mod(n, 50) == 0:
        env.render()
