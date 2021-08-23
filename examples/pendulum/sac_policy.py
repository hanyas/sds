import numpy as np

import gym
import sds

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv


env = gym.make('Pendulum-ID-v1')
env._max_episode_steps = 200
env.unwrapped.dt = 0.02
env.unwrapped.sigma = 1e-4
env.unwrapped.uniform = True

ulim = env.action_space.high

dm_obs = env.observation_space.shape[0]
dm_act = env.action_space.shape[0]

env = DummyVecEnv([lambda: env])

model = SAC(MlpPolicy, env,
            gamma=0.99, verbose=1,
            learning_rate=1e-3,
            policy_kwargs={'layers': [64, 64],
                           'reg_weight': 1e-32})

model.learn(total_timesteps=100000, log_interval=10)


obs, act = [], []
nb_rollouts, nb_steps = 25, 200
for n in range(nb_rollouts):
    _obs = np.empty((nb_steps, dm_obs))
    _act = np.empty((nb_steps, dm_act))

    x = env.reset()
    for t in range(nb_steps):
        u, _ = model.predict(x)
        _obs[t, :], _act[t, :] = x, u
        u = np.clip(u, -ulim, ulim)
        x, r, _, _ = env.step(u)

    obs.append(_obs)
    act.append(_act)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=dm_obs + dm_act, figsize=(12, 4))
for _obs, _act in zip(obs, act):
    for k, col in enumerate(ax[:-1]):
        col.plot(_obs[:, k])
    ax[-1].plot(_act)
plt.show()

# # save ctl
# model.save("sac_pendulum")
