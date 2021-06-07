import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class HybridPoleWithWall(gym.Env):

    def __init__(self, rarhmm):
        self.state_dim = 2
        self.act_dim = 1
        self.obs_dim = 2

        self.dt = 0.01
        self.sigma = 1e-8

        self.g = np.array([0., 0.])
        self.gw = - np.array([1e0, 1e0])

        self.xmax = np.array([0.25, 1.5])
        self.xmin = np.array([-0.25, -1.5])
        self.observation_space = spaces.Box(low=self.xmin,
                                            high=self.xmax,
                                            dtype=np.float64)

        self.uw = - np.array([1e0])
        self.umax, self.umin = np.array([4.]), np.array([-4.])
        self.action_space = spaces.Box(low=self.umin,
                                       high=self.umax,
                                       dtype=np.float64)

        self.rarhmm = rarhmm

        self.obs = None
        self.hist_obs = np.empty((0, self.obs_dim))
        self.hist_act = np.empty((0, self.act_dim))

        self.np_random = None

        self.seed()

    def dynamics(self, xhist, uhist):
        xhist, uhist = np.atleast_2d(xhist, uhist)
        xn = self.rarhmm.step(xhist, uhist, stoch=False, average=True)
        return xn

    def rewrad(self, x, u):
        return (x - self.g).T @ np.diag(self.gw) @ (x - self.g)\
               + u.T @ np.diag(self.uw) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        self.hist_act = np.vstack((self.hist_act, act))
        self.obs = self.dynamics(self.hist_obs, self.hist_act)
        self.hist_obs = np.vstack((self.hist_obs, self.obs))
        return self.obs, None, False, {}

    def reset(self):
        pass

    # for plotting
    def fake_step(self, obs, act):
        nxt_obs = self.dynamics(obs, act)
        return nxt_obs
